import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate)
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    cohen_kappa_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.cluster import KMeans

# ------------------- UTILITÁRIOS DE ACTIVE LEARNING -------------------
def compute_entropy(probas: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(probas * np.log(probas + eps), axis=1)


def select_by_strategy(model, X_pool: np.ndarray, strategy: str, k: int, committee_models=None) -> np.ndarray:
    if X_pool.shape[0] == 0:
        return np.array([], dtype=int)
    probas = model.predict_proba(X_pool)
    if strategy == 'entropy':
        return np.argsort(-compute_entropy(probas))[:k]
    if strategy == 'margin':
        sorted2 = np.sort(probas, axis=1)[:, -2:]
        margins = sorted2[:, 1] - sorted2[:, 0]
        return np.argsort(margins)[:k]
    if strategy == 'random':
        return np.random.choice(len(X_pool), size=min(k, len(X_pool)), replace=False)
    if strategy == 'uncertainty_diversity':
        scores = compute_entropy(probas)
        top = np.argsort(-scores)[: min(len(X_pool), k * 3)]
        labels = KMeans(n_clusters=min(k, len(top)), random_state=42).fit_predict(X_pool[top])
        selected = []
        for c in np.unique(labels):
            idxs = np.where(labels == c)[0]
            chosen = top[idxs[np.argmax(scores[top][idxs])]]
            selected.append(chosen)
        return np.array(selected[:k])
    if strategy == 'query_by_committee' and committee_models:
        votes = np.stack([m.predict(X_pool) for m in committee_models], axis=1).astype(int)
        def vote_entropy(v: np.ndarray) -> float:
            n_classes = np.max(v) + 1
            onehots = np.eye(n_classes)[v]
            ps = onehots.mean(axis=0)
            eps = 1e-12
            return -np.sum(ps * np.log(ps + eps))
        entropies = np.apply_along_axis(vote_entropy, 1, votes)
        return np.argsort(-entropies)[:k]
    return np.array([], dtype=int)

# ------------------- RUN ACTIVE LEARNING -------------------
def run_active_learning(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    initial_label_ratio: float = 0.1,
    selection_size: int = 20,
    n_iterations: int = 10,
    strategy: str = 'entropy',
    sampler_name: str = 'SMOTE',
    cv_folds: int = 5,
    n_trials: int = 10,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    np.random.seed(random_state)
    # hold-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    # initial labeling
    n_initial = max(1, int(initial_label_ratio * len(X_train)))
    init_idx = np.random.choice(len(X_train), size=n_initial, replace=False)
    mask_L = np.zeros(len(X_train), dtype=bool)
    mask_L[init_idx] = True
    X_L, y_L = X_train.iloc[mask_L].reset_index(drop=True), y_train.iloc[mask_L].reset_index(drop=True)
    X_U, y_U = X_train.iloc[~mask_L].reset_index(drop=True), y_train.iloc[~mask_L].reset_index(drop=True)

    sampler = {
        'Oversampling': RandomOverSampler(random_state=random_state),
        'SMOTE': SMOTE(random_state=random_state),
        'TomekLinks': TomekLinks(),
        'SMOTETomek': SMOTETomek(random_state=random_state)
    }[sampler_name]
    imp = SimpleImputer(strategy='mean')
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    perf_hist = []

    for it in range(1, n_iterations + 1):
        # hyperparameter tuning with Optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            pipe = ImbPipeline([
                ('imputer', imp),
                ('sampler', sampler),
                ('clf', RandomForestClassifier(**params, random_state=random_state))
            ])
            scores = cross_validate(
                pipe, X_L, y_L, cv=cv,
                scoring='f1_weighted', n_jobs=-1
            )
            return np.mean(scores['test_score'])
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params

        # train and cross-validate with best params
        pipeline = ImbPipeline([
            ('imputer', imp),
            ('sampler', sampler),
            ('clf', RandomForestClassifier(**best_params, random_state=random_state))
        ])
        scores = cross_validate(
            pipeline, X_L, y_L, cv=cv,
            scoring=['accuracy','precision_weighted','recall_weighted','f1_weighted','roc_auc'],
            n_jobs=-1
        )
        perf_hist.append({
            'iteration': it,
            'accuracy': np.mean(scores['test_accuracy']),
            'precision': np.mean(scores['test_precision_weighted']),
            'recall': np.mean(scores['test_recall_weighted']),
            'f1': np.mean(scores['test_f1_weighted']),
            'roc_auc': np.mean(scores['test_roc_auc'])
        })

        # selection step
        X_L_imp = imp.fit_transform(X_L)
        base_clf = RandomForestClassifier(**best_params, random_state=random_state).fit(X_L_imp, y_L)
        X_U_imp = imp.transform(X_U)
        if strategy == 'query_by_committee':
            committee = [clone(base_clf).fit(X_L_imp, y_L) for _ in range(3)]
            idx = select_by_strategy(base_clf, X_U_imp, strategy, selection_size, committee)
        else:
            idx = select_by_strategy(base_clf, X_U_imp, strategy, selection_size)
        idx = idx[idx < len(X_U)]
        X_L = pd.concat([X_L, X_U.iloc[idx]], ignore_index=True)
        y_L = pd.concat([y_L, y_U.iloc[idx]], ignore_index=True)
        X_U = X_U.drop(idx).reset_index(drop=True)
        y_U = y_U.drop(idx).reset_index(drop=True)

    # final evaluation
    final_pipe = ImbPipeline([
        ('imputer', imp),
        ('sampler', sampler),
        ('clf', RandomForestClassifier(**best_params, random_state=random_state))
    ])
    final_pipe.fit(X_L, y_L)
    y_pred = final_pipe.predict(X_test)
    # confusion matrices
    cm_abs = confusion_matrix(y_test, y_pred)
    cm_rel = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, final_pipe.predict_proba(X_test)[:,1])
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:,1]),
        'precision_class1': precision_score(y_test, y_pred, pos_label=1, zero_division=1),
        'recall_class1': recall_score(y_test, y_pred, pos_label=1, zero_division=1),
        'f1_class1': f1_score(y_test, y_pred, pos_label=1, zero_division=1),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'confusion_abs': cm_abs.tolist(),
        'confusion_rel': cm_rel.tolist(),
        'roc_curve_fpr': fpr.tolist(),
        'roc_curve_tpr': tpr.tolist()
    }
    return pd.DataFrame(perf_hist), final_metrics

# ------------------- APP STREAMLIT -------------------
def main():
    st.title("Active Learning Pipeline")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if not uploaded:
        st.stop()
    data = pd.read_csv(uploaded)
    st.write("## Preview dos dados")
    st.dataframe(data.head())

    target = st.selectbox("Selecione a coluna alvo", data.columns)
    test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
    initial_ratio = st.sidebar.slider("Proporção inicial rotulada", 0.05, 1.0, 0.1, step=0.05)
    selection_size = st.sidebar.number_input("Batch size (AL)", min_value=1, value=20)
    n_iterations = st.sidebar.number_input("Número de iterações (AL)", min_value=1, value=10)
    strategy = st.sidebar.selectbox(
        "Estratégia de AL",
        ['entropy','margin','random','uncertainty_diversity','query_by_committee']
    )
    sampler_name = st.sidebar.selectbox(
        "Método de balanceamento",
        ['Oversampling','SMOTE','TomekLinks','SMOTETomek']
    )
    n_trials = st.sidebar.number_input("Trials Optuna por iteração", min_value=1, value=10)

    if st.button("Rodar Active Learning"):
        X = data.drop(columns=[target])
        y = data[target]
        hist, final = run_active_learning(
            X, y,
            test_size=test_size,
            initial_label_ratio=initial_ratio,
            selection_size=selection_size,
            n_iterations=n_iterations,
            strategy=strategy,
            sampler_name=sampler_name,
            cv_folds=5,
            n_trials=n_trials,
            random_state=42
        )
        st.write("## Histórico de Performance (CV por iteração)")
        st.dataframe(hist)
        st.line_chart(hist.set_index('iteration')[['accuracy','precision','recall','f1','roc_auc']])
        st.write("## Métricas Finais no Hold-out")
        labels = sorted(y.unique())
        st.write("### Matriz de Confusão Absoluta")
        cm_abs_arr = np.array(final['confusion_abs'])
        fig_cm1, ax_cm1 = plt.subplots()
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_abs_arr, display_labels=labels)
        # adiciona o cmap 'Blues' e exibe colorbar
        disp1.plot(ax=ax_cm1, cmap='Blues', colorbar=True)
        ax_cm1.set_title('Matriz de Confusão Absoluta')
        st.pyplot(fig_cm1)

        st.write("### Matriz de Confusão Relativa")
        cm_rel_arr = np.array(final['confusion_rel'])
        fig_cm2, ax_cm2 = plt.subplots()
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_rel_arr, display_labels=labels)
        # mesmo cmap para a relativa
        disp2.plot(ax=ax_cm2, cmap='Blues', colorbar=True)
        ax_cm2.set_title('Matriz de Confusão Relativa')
        st.pyplot(fig_cm2)

        st.write("### Curva ROC Final")
        # plot ROC curve
        fig, ax = plt.subplots()
        ax.plot(final['roc_curve_fpr'], final['roc_curve_tpr'], label=f"AUC = {final['roc_auc']:.2f}")
        ax.plot([0,1],[0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        st.write("### Demais métricas:")
        st.json({k: v for k, v in final.items() if k not in ['confusion_abs','confusion_rel','roc_curve_fpr','roc_curve_tpr']})

if __name__ == '__main__':
    main()
