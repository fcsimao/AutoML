import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Importações do scikit-learn, incluindo RepeatedStratifiedKFold para CV repetida
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, 
                                     train_test_split, RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, average_precision_score

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

# Modelos de boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Otimização de hiperparâmetros
import optuna

#########################################
# Função para retornar CV repetida
#########################################
def get_cv(cv, n_repeats=3):
    # Agora os folds não são fixos; a semente foi removida.
    return RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats)

#########################################
# Função de balanceamento (para treinamento)
#########################################
def apply_balancing(strategy, X, y):
    if strategy == 'Tomek':
         balancer = TomekLinks()
    elif strategy == 'Oversampling':
         balancer = RandomOverSampler()
    elif strategy == 'Undersampling':
         balancer = RandomUnderSampler()
    elif strategy == 'SMOTE':
         balancer = SMOTE()
    elif strategy == 'Híbrido (Tomek + SMOTE)':
         balancer = SMOTETomek()
    else:
         raise ValueError("Estratégia de balanceamento desconhecida!")
    X_res, y_res = balancer.fit_resample(X, y)
    return X_res, y_res

#########################################
# Função para otimização de hiperparâmetros usando somente Optuna
#########################################
def optimize_hyperparameters(model, param_space, X, y, cv=3, n_repeats=3):
    """
    Otimiza os hiperparâmetros usando Optuna com a métrica f1.
    """
    cv_strategy = get_cv(cv, n_repeats)
    def objective(trial):
         params = {}
         for key, values in param_space.items():
              params[key] = trial.suggest_categorical(key, values)
         model.set_params(**params)
         score = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1').mean()
         return score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params

#########################################
# Função para encontrar o melhor pipeline (usando somente Optuna)
#########################################
def find_best_pipeline(X, y, X_test, y_test, balancing_strategies, models, param_spaces, cv=3, n_repeats=3):
    best_score = -float("inf")
    best_config = {}
    for bal in balancing_strategies:
         X_res, y_res = apply_balancing(bal, X, y)
         for model_name, model in models.items():
              param_space_model = param_spaces.get(model_name, {})
              try:
                  best_params = optimize_hyperparameters(model, param_space_model, X_res, y_res, cv=cv, n_repeats=n_repeats)
                  model.set_params(**best_params)
                  cv_strategy = get_cv(cv, n_repeats)
                  cv_score = cross_val_score(model, X_res, y_res, cv=cv_strategy, scoring='f1').mean()
                  model.fit(X_res, y_res)
                  y_test_pred = model.predict(X_test)
                  test_score = f1_score(y_test, y_test_pred)
                  diff = cv_score - test_score
                  st.write(f"Config: {model_name} | Bal: {bal} => CV F1: {cv_score:.4f}, Test F1: {test_score:.4f}, Diff: {diff:.4f}")
                  if cv_score > best_score:
                        best_score = cv_score
                        best_config = {
                             "balancing": bal,
                             "model": model_name,
                             "params": best_params,
                             "score": cv_score,
                             "test_score": test_score,
                             "diff": diff
                        }
              except Exception as e:
                  st.write(f"Erro com {model_name}, {bal}: {e}")
    return best_config

#########################################
# Função para seleção ativa com estratégias de query
#########################################
def select_query_indices(model, X_pool, X_train, y_train, query_size, query_strategy, y_full):
    qs = query_strategy.lower()
    if qs == "entropy":
         if hasattr(model, "predict_proba"):
             probs = model.predict_proba(X_pool)
             entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1)
             query_idx = np.argsort(entropies)[-query_size:]
         else:
             query_idx = np.random.choice(len(X_pool), size=query_size, replace=False)
    elif qs == "margin":
         if hasattr(model, "predict_proba"):
             probs = model.predict_proba(X_pool)
             sorted_probs = np.sort(probs, axis=1)
             margins = sorted_probs[:, -1] - sorted_probs[:, -2]
             query_idx = np.argsort(margins)[:query_size]
         else:
             query_idx = np.random.choice(len(X_pool), size=query_size, replace=False)
    elif qs in ["random", "aleatória"]:
         query_idx = np.random.choice(len(X_pool), size=query_size, replace=False)
    elif qs in ["query_by_committee", "comitê"]:
         from sklearn.linear_model import LogisticRegression
         from sklearn.ensemble import RandomForestClassifier
         from sklearn.svm import SVC
         committee_models = [
             LogisticRegression(max_iter=1000, random_state=42),
             RandomForestClassifier(random_state=42),
             SVC(probability=True, random_state=42)
         ]
         votes = []
         for cm in committee_models:
             cm.fit(X_train, y_train)
             preds = cm.predict(X_pool)
             votes.append(preds)
         votes = np.array(votes)
         vote_entropies = []
         n_classes = len(np.unique(y_full))
         for i in range(votes.shape[1]):
             counts = np.bincount(votes[:, i], minlength=n_classes)
             vote_dist = counts / len(committee_models)
             entropy_val = -np.sum(vote_dist * np.log(vote_dist + 1e-12))
             vote_entropies.append(entropy_val)
         vote_entropies = np.array(vote_entropies)
         query_idx = np.argsort(vote_entropies)[-query_size:]
    else:
         query_idx = np.random.choice(len(X_pool), size=query_size, replace=False)
    return query_idx

#########################################
# Função para active learning com re-otimização incremental (usando somente Optuna)
#########################################
def active_learning_loop(model, param_space, X_pool, y_pool, X_initial, y_initial,
                         n_iterations=10, query_size=10, cv=3, n_repeats=3, progress_callback=None,
                         balancing_strategy='SMOTE', query_strategy="entropy", y_full=None):
    """
    Loop de Active Learning com re-otimização incremental dos hiperparâmetros.
    Em cada iteração:
      - Aplica balanceamento aos dados de treinamento.
      - Re-otimiza os hiperparâmetros com os dados rotulados acumulados (usando Optuna).
      - Treina o modelo com os novos hiperparâmetros.
      - Seleciona novas amostras do pool com base na estratégia de query ativa.
      - Incorpora as amostras selecionadas ao conjunto rotulado.
    
    Retorna:
      performance_history (lista de métricas por iteração), o modelo final, e os conjuntos finais (X_train e y_train).
    """
    performance_history = []
    X_train, y_train = X_initial.copy(), y_initial.copy()
    
    for iteration in range(n_iterations):
         msg = f"Iteração {iteration+1}/{n_iterations}"
         if progress_callback:
             progress_callback(msg)
         else:
             st.write(msg)
         
         # Aplica balanceamento aos dados de treinamento
         X_train_bal, y_train_bal = apply_balancing(balancing_strategy, X_train, y_train)
         
         # Re-otimização incremental com Optuna
         def objective_iter(trial):
             n_estimators = trial.suggest_int("n_estimators", 50, 500)
             max_depth = trial.suggest_int("max_depth", 5, 50)
             min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
             min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
             clf = RandomForestClassifier(
                 n_estimators=n_estimators,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf
             )
             return cross_val_score(clf, X_train_bal, y_train_bal, cv=cv, scoring="f1").mean()
         
         study_iter = optuna.create_study(direction="maximize")
         study_iter.optimize(objective_iter, n_trials=5)
         best_params_iter = study_iter.best_trial.params
         
         model.set_params(**best_params_iter)
         model.fit(X_train_bal, y_train_bal)
         
         cv_strategy = get_cv(cv, n_repeats)
         score = cross_val_score(model, X_train_bal, y_train_bal, cv=cv_strategy, scoring='f1').mean()
         performance_history.append({"iteration": iteration+1, "cv_f1": score, "n_labeled": len(y_train)})
         if progress_callback:
             progress_callback(f"{msg} - CV F1 Score: {score:.4f}")
         
         # Seleciona novas amostras do pool com a estratégia escolhida
         query_idx = select_query_indices(model, X_pool, X_train, y_train, query_size, query_strategy, y_full)
         X_query = X_pool[query_idx]
         y_query = y_pool[query_idx]
         
         # Atualiza os conjuntos de treinamento: "revela" os rótulos selecionados
         X_train = np.concatenate([X_train, X_query])
         y_train = np.concatenate([y_train, y_query])
         X_pool = np.delete(X_pool, query_idx, axis=0)
         y_pool = np.delete(y_pool, query_idx, axis=0)
         
         time.sleep(0.5)
         
    return performance_history, model, X_train, y_train

#########################################
# Função para plotar as métricas de avaliação
#########################################
def plot_evaluation_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    ax_cm.set_xlabel('Predito')
    ax_cm.set_ylabel('Verdadeiro')
    ax_cm.set_title('Matriz de Confusão')
    
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('Taxa de Falsos Positivos')
        ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
        ax_roc.set_title('Curva ROC')
        ax_roc.legend(loc="lower right")
    else:
        fig_roc = None

    if y_score is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision, label=f'AP = {ap:.2f}')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Curva Precision-Recall')
        ax_pr.legend(loc="lower left")
    else:
        fig_pr = None

    return fig_cm, fig_roc, fig_pr

#########################################
# Função principal (Streamlit App)
#########################################
def main():
    st.title("Active Learning com Otimização de Pipeline (F1-Score) - Usando somente Optuna")
    st.write("Faça upload do seu arquivo CSV para iniciar o processo.")
    
    uploaded_file = st.sidebar.file_uploader("Selecione um arquivo CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview dos dados:", data.head())
        target_col = st.sidebar.selectbox("Selecione a coluna alvo", data.columns)
        if target_col:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Divisão dos dados sem fixar a semente
            X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            # Cria DataFrame completo de treinamento
            df_train_full = X_train_full.copy()
            df_train_full[target_col] = y_train_full.copy()
            # Cria a versão mascarada dos dados de treinamento (mantendo, por exemplo, 20% dos rótulos)
            df_train_masked = df_train_full.copy()
            mask = np.random.rand(len(df_train_masked)) >= 0.2
            df_train_masked.loc[mask, target_col] = np.nan
            
            st.write("DataFrame de treino completo:", df_train_full.shape)
            st.write("DataFrame de treino mascarado:", df_train_masked.shape)
            st.write(f"Conjunto de teste (não balanceado): {X_test.shape}")
            
            # Define os conjuntos para active learning:
            # Conjunto inicial (rotulado)
            df_initial = df_train_full[df_train_masked[target_col].notnull()].copy()
            X_initial_pipeline = df_initial.drop(columns=[target_col])
            y_initial_pipeline = df_initial[target_col]
            X_initial = X_initial_pipeline.values
            y_initial = y_initial_pipeline.values
            
            # Pool não rotulado (onde os rótulos foram mascarados)
            df_pool = df_train_full[df_train_masked[target_col].isnull()].copy()
            X_pool = df_pool.drop(columns=[target_col]).values
            y_pool = df_pool[target_col].values  # Rótulos verdadeiros, mas "ocultos" até serem adquiridos pelo AL
            
            # Estratégia de query ativa
            query_strategy = st.sidebar.selectbox("Estratégia de Query Ativa", ["entropy", "margin", "random", "query_by_committee"])
            
            # Parâmetros do Active Learning
            n_iterations = st.sidebar.number_input("Número de iterações", min_value=1, max_value=50, value=10, step=1)
            selection_size = st.sidebar.number_input("Número de amostras por iteração", min_value=1, max_value=100, value=20, step=1)
            
            # Configurações para a busca de pipeline
            balancing_strategies = ['Tomek', 'Oversampling', 'Undersampling', 'SMOTE', 'Híbrido (Tomek + SMOTE)']
            models = {
                "RandomForest": RandomForestClassifier(random_state=42),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "LightGBM": LGBMClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "SVC": SVC(probability=True, random_state=42),
                "KNN": KNeighborsClassifier(),
                "DecisionTree": DecisionTreeClassifier(random_state=42)
            }
            
            param_spaces = {
                "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
                "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]},
                "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5, 7]},
                "LightGBM": {"n_estimators": [50, 100], "num_leaves": [31, 50]},
                "LogisticRegression": {"C": [0.1, 1, 10]},
                "SVC": {"C": [0.1, 1, 10], "kernel": ['rbf']},
                "KNN": {"n_neighbors": [3, 5, 7]},
                "DecisionTree": {"max_depth": [None, 10, 20]}
            }
            
            # Busca pelo melhor pipeline usando somente Optuna
            if "best_pipeline" not in st.session_state:
                st.header("Busca pelo Melhor Pipeline (usando somente Optuna)")
                with st.spinner("Testando configurações..."):
                     best_pipeline = find_best_pipeline(X_initial_pipeline.values, y_initial_pipeline.values, 
                                                        X_test, y_test.values,
                                                        balancing_strategies, models, param_spaces,
                                                        cv=3, n_repeats=3)
                if best_pipeline:
                    st.success("Melhor pipeline encontrado!")
                    st.write(best_pipeline)
                    st.session_state.best_pipeline = best_pipeline
                else:
                    st.error("Nenhum pipeline foi encontrado.")
                    return
            else:
                st.header("Melhor Pipeline Encontrado")
                st.write(st.session_state.best_pipeline)
                best_pipeline = st.session_state.best_pipeline
            
            st.header("Active Learning")
            if st.button("Iniciar Active Learning"):
                 progress_area = st.empty()
                 log_area = st.empty()
                 
                 best_model_name = st.session_state.best_pipeline["model"]
                 best_params = st.session_state.best_pipeline["params"]
                 best_balancing = st.session_state.best_pipeline["balancing"]
                 
                 model = models[best_model_name]
                 model.set_params(**best_params)
                 
                 def progress_callback(msg):
                     progress_area.write(msg)
                     log_area.write(msg)
                 
                 # Active Learning com re-otimização incremental usando somente Optuna
                 performance_history, final_model, X_train_final, y_train_final = active_learning_loop(
                     model=model,
                     param_space=param_spaces.get(best_model_name, {}),
                     X_pool=X_pool.copy(), y_pool=y_pool.copy(),
                     X_initial=X_initial, y_initial=y_initial,
                     n_iterations=int(n_iterations),
                     query_size=int(selection_size),
                     cv=3,
                     n_repeats=3,
                     progress_callback=progress_callback,
                     balancing_strategy=best_balancing,
                     query_strategy=query_strategy,
                     y_full=y.values
                 )
                 
                 st.success("Active Learning concluído!")
                 
                 test_score = f1_score(y_test, final_model.predict(X_test))
                 st.write(f"Desempenho final no conjunto de teste (dados reais): {test_score:.4f}")
                 
                 # Exibe o número de amostras finais do conjunto de treinamento
                 st.write("Número de amostras no conjunto final de treinamento:", len(y_train_final))
                 
                 fig_cm, fig_roc, fig_pr = plot_evaluation_metrics(final_model, X_test, y_test)
                 
                 st.subheader("Matriz de Confusão")
                 st.pyplot(fig_cm)
                 
                 if fig_roc is not None:
                     st.subheader("Curva ROC")
                     st.pyplot(fig_roc)
                 
                 if fig_pr is not None:
                     st.subheader("Curva Precision-Recall")
                     st.pyplot(fig_pr)
    else:
        st.write("Por favor, faça o upload de um arquivo CSV na barra lateral.")

if __name__ == "__main__":
    main()
