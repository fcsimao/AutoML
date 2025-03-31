import os
import time
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import optuna
import streamlit as st
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, cohen_kappa_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Otimização Bayesiana via scikit-optimize
from skopt import BayesSearchCV

# Integração com LIME
from lime.lime_tabular import LimeTabularExplainer

# Fairlearn para mitigação de vies
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

# --- Integração com LangChain para geração de relatório ---
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#############################################
# Funções de pré-processamento e balanceamento
#############################################

def load_data():
    uploaded_file = st.file_uploader("📂 Carregue seu dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def balance_data(X, y, method):
    if method == "SMOTE":
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
    elif method == "Oversampling":
        df = pd.concat([X, y], axis=1)
        df_minority = df[df[y.name] != df[y.name].mode()[0]]
        df_minority_upsampled = resample(
            df_minority, replace=True, n_samples=len(df[df[y.name] == df[y.name].mode()[0]]), random_state=42
        )
        df_balanced = pd.concat([df[df[y.name] == df[y.name].mode()[0]], df_minority_upsampled])
        X_bal, y_bal = df_balanced.drop(columns=[y.name]), df_balanced[y.name]
    elif method == "Undersampling":
        df = pd.concat([X, y], axis=1)
        df_majority = df[df[y.name] == df[y.name].mode()[0]]
        df_minority = df[df[y.name] != df[y.name].mode()[0]]
        df_majority_downsampled = resample(
            df_majority, replace=False, n_samples=len(df_minority), random_state=42
        )
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        X_bal, y_bal = df_balanced.drop(columns=[y.name]), df_balanced[y.name]
    elif method == "Tomek":
        tomek = TomekLinks(sampling_strategy='auto')
        X_bal, y_bal = tomek.fit_resample(X, y)
    elif method == "Híbrido":
        smote_tomek = SMOTETomek(random_state=42)
        X_bal, y_bal = smote_tomek.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y
    return X_bal, y_bal

#############################################
# Funções de otimização e criação de modelos
#############################################

def optimize_hyperparameters_with_optuna(model_class, X_train, y_train):
    def objective(trial):
        params = {}
        if model_class in [GradientBoostingClassifier, XGBClassifier, LGBMClassifier]:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
        if model_class in [RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
        if model_class in [RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier]:
            params["max_depth"] = trial.suggest_int("max_depth", 5, 50)
        model = model_class(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
        return np.mean(scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    return study.best_params

def create_model(model_class, **params):
    if model_class == SVC:
        return model_class(probability=True, **params)
    return model_class(**params)

# Função de otimização armazenada em cache
@st.cache_data(show_spinner=True)
def run_optimization(X_train_scaled, y_train, X_test_scaled, y_test):
    balance_methods = ["Tomek", "Oversampling", "Undersampling", "SMOTE", "Híbrido"]
    all_results = []
    for balance_method in balance_methods:
        X_train_bal, y_train_bal = balance_data(X_train_scaled, y_train, balance_method)
        results_df = train_and_evaluate_models(X_train_bal, X_test_scaled, y_train_bal, y_test, balance_method)
        all_results.append(results_df)
    final_results = pd.concat(all_results).sort_values(by="F1 Final Test", ascending=False)
    final_results.reset_index(drop=True, inplace=True)
    return final_results

def train_and_evaluate_models(X_train, X_final_test, y_train, y_final_test, balance_method):
    models = {
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "LogisticRegression": LogisticRegression,
        "SVC": SVC,
        "KNN": KNeighborsClassifier,
        "DecisionTree": DecisionTreeClassifier
    }
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100, 200, 300, 500],
                         "max_depth": [None, 10, 20, 30, 40],
                         "min_samples_split": [2, 5, 10]},
        "GradientBoosting": {"n_estimators": [50, 100, 200],
                             "learning_rate": [0.01, 0.1, 0.2],
                             "max_depth": [3, 5, 10]},
        "XGBoost": {"n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 10]},
        "LightGBM": {"n_estimators": [50, 100, 200],
                     "learning_rate": [0.01, 0.1, 0.2],
                     "max_depth": [3, 5, 10]},
        "LogisticRegression": {"C": [0.1, 1, 10, 100],
                               "penalty": ["l2"],
                               "solver": ["lbfgs", "saga"]},
        "SVC": {"C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]},
        "KNN": {"n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]},
        "DecisionTree": {"max_depth": [None, 10, 20, 30, 40],
                         "min_samples_split": [2, 5, 10]}
    }
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model_class in models.items():
        best_params_optuna = optimize_hyperparameters_with_optuna(model_class, X_train, y_train)
        best_model_optuna = create_model(model_class, **best_params_optuna)
        best_model_optuna.fit(X_train, y_train)
        
        param_grid = param_grids.get(name, {})
        search_random = RandomizedSearchCV(create_model(model_class), param_grid, n_iter=10, cv=skf,
                                            scoring='f1_weighted', n_jobs=-1, random_state=42)
        search_random.fit(X_train, y_train)
        best_params_random = search_random.best_params_
        best_model_random = create_model(model_class, **best_params_random)
        best_model_random.fit(X_train, y_train)
        
        grid = GridSearchCV(create_model(model_class), param_grid, cv=skf, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_params_grid = grid.best_params_
        best_model_grid = create_model(model_class, **best_params_grid)
        best_model_grid.fit(X_train, y_train)
        
        bayes = BayesSearchCV(create_model(model_class), param_grid, n_iter=10, cv=skf,
                              scoring='f1_weighted', n_jobs=-1, random_state=42)
        bayes.fit(X_train, y_train)
        best_params_bayes = bayes.best_params_
        best_model_bayes = create_model(model_class, **best_params_bayes)
        best_model_bayes.fit(X_train, y_train)
        
        for method, best_model, best_params in zip(
            ["Optuna", "RandomizedSearchCV", "GridSearchCV", "BayesSearchCV"],
            [best_model_optuna, best_model_random, best_model_grid, best_model_bayes],
            [best_params_optuna, best_params_random, best_params_grid, best_params_bayes]
        ):
            y_pred = best_model.predict(X_final_test)
            final_f1 = f1_score(y_final_test, y_pred, average='weighted', zero_division=1)
            accuracy = accuracy_score(y_final_test, y_pred)
            recall_val = recall_score(y_final_test, y_pred, pos_label=1, zero_division=1)
            precision_val = precision_score(y_final_test, y_pred, average='weighted', zero_division=1)
            kappa = cohen_kappa_score(y_final_test, y_pred)
            cv_f1_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='f1_weighted')
            cv_f1_mean = np.mean(cv_f1_scores)
            cv_f1_std = np.std(cv_f1_scores)
            overfitting = abs(cv_f1_mean - final_f1) > 0.05
            stable = cv_f1_std < 0.02
            results.append({
                "Modelo": name,
                "F1 Final Test": final_f1,
                "Accuracy": accuracy,
                "Precision": precision_val,
                "Recall": recall_val,
                "Kappa": kappa,
                "Balanceamento": balance_method,
                "Otimização": method,
                "Melhores Hiperparâmetros": best_params,
                "Overfitting": "Sim" if overfitting else "Não",
                "Modelo Estável": "Sim" if stable else "Não",
                "trained_model": best_model
            })
    return pd.DataFrame(results)

#############################################
# Funções de plotagem e interpretabilidade
#############################################

def plot_roc_curve(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise AttributeError("O modelo não suporta predict_proba ou decision_function.")
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC')
    ax.legend(loc="lower right")
    return fig

def plot_precision_recall_curve(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise AttributeError("O modelo não suporta predict_proba ou decision_function.")
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker='.', label=f'AP = {avg_precision:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curva Precision-Recall')
    ax.legend()
    return fig, avg_precision

def plot_confusion_matrices(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Classe 0", "Classe 1"])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Matriz de Confusão")
    fig.tight_layout()
    return fig

# Nova função para o modelo fair-aware
def plot_confusion_matrices_fair(model, X_test, y_test, sensitive_features):
    y_pred = model.predict(X_test, sensitive_features=sensitive_features)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Classe 0", "Classe 1"])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Matriz de Confusão (Fairness)")
    fig.tight_layout()
    return fig

def plot_feature_importance(model, X_train):
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_names = np.array(X_train.columns)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        ax.barh(range(len(importances)), importances[indices], align='center')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(feature_names[indices])
        ax.set_title("Importância das Features")
        ax.set_xlabel("Importância")
        return fig
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        indices = np.argsort(coef)
        ax.barh(range(len(coef)), coef[indices], align='center')
        ax.set_yticks(range(len(coef)))
        ax.set_yticklabels(feature_names[indices])
        ax.set_title("Coeficientes das Features")
        ax.set_xlabel("Valor Absoluto")
        return fig
    else:
        st.write("Modelo não possui atributos de importância para as features.")
        return None

def plot_shap_summary(model, X_train):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False, feature_names=X_train.columns)
    fig = plt.gcf()
    return fig, shap_values

def extract_shap_summary_text(shap_values, X_train, top_n=3):
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_train.columns
    df_shap = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
    df_shap = df_shap.sort_values('mean_abs_shap', ascending=False)
    top_features = df_shap.head(top_n)
    summary_text = "Principais features segundo SHAP: " + ", ".join(
        f"{row['feature']} ({row['mean_abs_shap']:.4f})" for _, row in top_features.iterrows()
    ) + "."
    return summary_text

def plot_lime_explanation(model, X_train, X_test, feature_names, class_names, instance_idx=0):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=X_test.iloc[instance_idx],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    fig = explanation.as_pyplot_figure()
    return fig, explanation

def extract_lime_summary_text(model, X_train, X_test, feature_names, class_names, instance_idx=0, top_n=3):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=X_test.iloc[instance_idx],
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    top_features = explanation.as_list()[:top_n]
    summary_text = "Principais features segundo LIME: " + ", ".join([f"{feat} ({weight:.4f})" for feat, weight in top_features]) + "."
    return summary_text

#############################################
# Função para calcular métricas de viés
#############################################
def compute_bias_metrics(y_true, y_pred, protected_attribute):
    groups = np.unique(protected_attribute)
    if len(groups) != 2:
        st.write("A análise de viés requer exatamente 2 grupos.")
        return None
    idx1 = protected_attribute == groups[0]
    idx2 = protected_attribute == groups[1]
    positive_rate1 = np.mean(y_pred[idx1] == 1)
    positive_rate2 = np.mean(y_pred[idx2] == 1)
    demographic_parity_diff = positive_rate1 - positive_rate2
    tpr1 = np.sum((y_pred[idx1] == 1) & (y_true[idx1] == 1)) / np.sum(y_true[idx1] == 1) if np.sum(y_true[idx1]==1) > 0 else np.nan
    tpr2 = np.sum((y_pred[idx2] == 1) & (y_true[idx2] == 1)) / np.sum(y_true[idx2] == 1) if np.sum(y_true[idx2]==1) > 0 else np.nan
    tpr_diff = tpr1 - tpr2
    fpr1 = np.sum((y_pred[idx1] == 1) & (y_true[idx1] == 0)) / np.sum(y_true[idx1] == 0) if np.sum(y_true[idx1]==0) > 0 else np.nan
    fpr2 = np.sum((y_pred[idx2] == 1) & (y_true[idx2] == 0)) / np.sum(y_true[idx2] == 0) if np.sum(y_true[idx2]==0) > 0 else np.nan
    fpr_diff = fpr1 - fpr2
    return {
        "Demographic Parity Difference": demographic_parity_diff,
        "True Positive Rate Difference": tpr_diff,
        "False Positive Rate Difference": fpr_diff
    }

#############################################
# Integração com LangChain para gerar relatório
#############################################
prompt_template_text = """
Você é um especialista em machine learning e análise de modelos preditivos aplicados à evasão escolar. Com base nos dados abaixo, gere um relatório dividido em duas seções.

---------------------------
**1. Análise Quantitativa do Modelo**

- **Tabela de Ranking dos Modelos:**  
{ranking}

- **Descrição da Curva Precision-Recall:**  
{prec_recall_description}

- **Descrição da Matriz de Confusão:**  
{conf_matrix_description}

- **Métricas adicionais:**  
Recall, Acurácia, Precisão e F1 Final Test.

Informe:
- Qual foi o melhor modelo escolhido, incluindo a técnica de balanceamento e o método de otimização.
- A robustez e a estabilidade desse modelo.
- A capacidade discriminatória do modelo, com base na curva Precision-Recall e outras métricas.
- Observações gerais e possíveis limitações.

---------------------------
**2. Orientações para Gestores com Insights de Explicabilidade**

- **Resumo Textual dos Valores SHAP:**  
{shap_description}

- **Resumo Textual da Explicação LIME:**  
{lime_description}

Forneça recomendações práticas e orientações para gestores, destacando:
- Quais features são mais relevantes na predição da evasão escolar e como elas impactam as decisões do modelo.
- Como os insights dos resumos SHAP e LIME podem orientar intervenções estratégicas.
- Sugestões de ações (por exemplo, intervenções, programas de reforço ou apoio socioeconômico).
- Uma conclusão integrada com implicações práticas e passos futuros recomendados.
---------------------------
"""
prompt_template = PromptTemplate(
    input_variables=["ranking", "prec_recall_description", "conf_matrix_description", "feature_importance", "shap_description", "lime_description"],
    template=prompt_template_text
)
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.5)
chain = LLMChain(llm=llm, prompt=prompt_template)
def generate_ai_insights(ranking, prec_recall_description, conf_matrix_description, feature_importance, shap_description, lime_description):
    response = chain.run({
        "ranking": ranking,
        "prec_recall_description": prec_recall_description,
        "conf_matrix_description": conf_matrix_description,
        "feature_importance": feature_importance,
        "shap_description": shap_description,
        "lime_description": lime_description
    })
    return response

#############################################
# Interface Streamlit
#############################################

st.title("📊 AutoML para Predição de Evasão Escolar")

# Carrega os dados
data = load_data()
if data is not None:
    st.write("### Preview do Dataset", data.head(10))
    target = st.selectbox("🎯 Selecione a variável alvo:", data.columns)
    X = data.drop(columns=[target])
    y = data[target]
    
    # Divisão dos dados
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Armazena os dados no session_state para persistência entre reexecuções
    st.session_state.X_train_raw = X_train_raw.copy()
    st.session_state.X_test_raw = X_test_raw.copy()
    st.session_state.y_train = y_train.copy()
    st.session_state.y_test = y_test.copy()
    
    # Seleção de atributos protegidos para análise de viés (opcional)
    protected_attrs = st.multiselect("Selecione os atributos protegidos para análise de viés (opcional):", list(data.columns))
    
    # Pré-processamento: escalonamento
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    # Usa o índice de X_test_raw para evitar erro de shape
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_train_raw.columns, index=X_test_raw.index)
    
    if st.button("🚀 Executar AutoML"):
        start_time = time.time()  # Inicia a medição do tempo de execução
        st.write("🔄 **Executando AutoML... Aguarde ⏳**")
        final_results = run_optimization(X_train_scaled, y_train, X_test_scaled, y_test)
        end_time = time.time()
        execution_time = end_time - start_time
        
        st.write("### 📊 Ranking dos Modelos")
        final_results_display = final_results.drop(columns=["trained_model"])
        st.dataframe(final_results_display, width=2000)
        
        # Armazena o ranking no session_state para uso posterior na seção de fairness
        st.session_state.final_results_display = final_results_display
        
        # Seleciona o melhor modelo com base no F1 score de validação
        best_row = final_results.loc[final_results['F1 Final Test'].idxmax()]
        best_model = best_row["trained_model"]
        st.session_state.best_model = best_model  # Armazena o modelo original para uso posterior
        
        st.write("### Detalhes do Melhor Modelo Selecionado")
        st.write(f"**Modelo:** {best_row['Modelo']}")
        st.write(f"**Balanceamento:** {best_row['Balanceamento']}")
        st.write(f"**Otimização:** {best_row['Otimização']}")
        st.write(f"**F1 Final Test:** {best_row['F1 Final Test']:.4f}")
        st.metric("Tempo de Execução (s)", f"{execution_time:.2f}")
        
        # ---- RESULTADOS ORIGINAIS (sem fairness) ----
        st.write("## Resultados Originais")
        st.write("### Curva ROC")
        fig_roc = plot_roc_curve(best_model, X_test_scaled, y_test)
        st.pyplot(fig_roc)
        
        st.write("### Curva Precision-Recall")
        fig_prec, avg_prec = plot_precision_recall_curve(best_model, X_test_scaled, y_test)
        st.pyplot(fig_prec)
        prec_recall_description = f"A Curva Precision-Recall apresenta uma média de precisão (AP) de {avg_prec:.2f}."
        
        st.write("### Matriz de Confusão")
        fig_cm = plot_confusion_matrices(best_model, X_test_scaled, y_test)
        st.pyplot(fig_cm)
        cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
        conf_matrix_description = f"A matriz de confusão apresenta os seguintes valores:\n{cm}"
        
        st.write("### Importância das Features")
        fig_importance = plot_feature_importance(best_model, X_train_scaled)
        if fig_importance:
            st.pyplot(fig_importance)
        else:
            st.write("O modelo selecionado não possui atributos de importância para as features.")

        st.write("### Explicação SHAP")
        fig_shap, shap_values = plot_shap_summary(best_model, X_train_scaled)
        st.pyplot(fig_shap)
        shap_description = extract_shap_summary_text(shap_values, X_train_scaled)
        
        st.write("### Explicação LIME")
        lime_fig, lime_explanation = plot_lime_explanation(
            model=best_model,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            feature_names=X_train_scaled.columns.tolist(),
            class_names=["Classe 0", "Classe 1"],
            instance_idx=0
        )
        st.pyplot(lime_fig)
        lime_description = extract_lime_summary_text(
            model=best_model,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            feature_names=X_train_scaled.columns.tolist(),
            class_names=["Classe 0", "Classe 1"],
            instance_idx=0
        )
        
        if protected_attrs:
            st.write("### Análise de Viés (Modelo Original)")
            for attr in protected_attrs:
                prot_attr_values = data.loc[y_test.index, attr].values
                bias_metrics = compute_bias_metrics(y_test.values, best_model.predict(X_test_scaled), prot_attr_values)
                st.write(f"**Atributo protegido: {attr}**")
                st.write(bias_metrics)
        
        insights = generate_ai_insights(
            ranking=final_results_display.to_string(index=False),
            prec_recall_description=prec_recall_description,
            conf_matrix_description=conf_matrix_description,
            feature_importance="(Texto resumido da importância das features)",  # Ajuste conforme necessário
            shap_description=shap_description,
            lime_description=lime_description
        )
        st.write("### Relatório Gerado pela IA (Resultados Originais)")
        st.write(insights)
    
    # ---- BLOCO DE FAIRNESS ----
    if protected_attrs:
        apply_fairness = st.checkbox("Aplicar estratégia de fairness e exibir novos resultados (somente métricas com predict)")
        if apply_fairness:
            # Verifica se o modelo e os dados necessários estão no session_state
            if "best_model" not in st.session_state or "final_results_display" not in st.session_state:
                st.write("Primeiro, execute o AutoML para gerar o modelo e os resultados.")
            else:
                st.write("### Aplicando estratégia de Fairness")
                best_model_orig = st.session_state.best_model
                # Mapeamento para ThresholdOptimizer
                constraint_map = {
                    "DemographicParity": "demographic_parity",
                    "EqualizedOdds": "equalized_odds"
                }
                fairness_constraint = st.selectbox("Selecione a restrição de fairness:", ["DemographicParity", "EqualizedOdds"])
                selected_constraint = constraint_map[fairness_constraint]
                chosen_attr = protected_attrs[0]  # Usa o primeiro atributo protegido
                # Recupera os sensitive features do conjunto de treinamento e teste a partir do session_state
                sensitive_feature_train = st.session_state.X_train_raw[chosen_attr].values
                sensitive_feature_test = st.session_state.X_test_raw[chosen_attr].values
                
                st.write(f"Aplicando ThresholdOptimizer com restrição {fairness_constraint} usando o atributo {chosen_attr}...")
                
                # Cria o objeto de restrição para o ExponentiatedGradient
                if fairness_constraint == "DemographicParity":
                    constraint_obj = DemographicParity()
                else:
                    constraint_obj = EqualizedOdds()
                fair_model = ExponentiatedGradient(best_model_orig, constraints=constraint_obj)
                fair_model.fit(X_train_scaled, y_train, sensitive_features=sensitive_feature_train)
                threshold_optimizer = ThresholdOptimizer(estimator=fair_model, constraints=selected_constraint, prefit=True)
                threshold_optimizer.fit(X_train_scaled, y_train, sensitive_features=sensitive_feature_train)
                best_model_fair = threshold_optimizer
                st.write("Modelo fair-aware ajustado com ThresholdOptimizer.")
                
                # Realiza a predição passando os sensitive features
                y_pred_new = best_model_fair.predict(X_test_scaled, sensitive_features=sensitive_feature_test)
                new_accuracy = accuracy_score(y_test, y_pred_new)
                new_precision = precision_score(y_test, y_pred_new, average='weighted', zero_division=1)
                new_recall = recall_score(y_test, y_pred_new, pos_label=1, zero_division=1)
                new_f1 = f1_score(y_test, y_pred_new, average='weighted', zero_division=1)
                st.write("## Resultados Após Fairness (usando predict)")
                st.write(f"Acurácia: {new_accuracy:.4f}")
                st.write(f"Precisão: {new_precision:.4f}")
                st.write(f"Recall: {new_recall:.4f}")
                st.write(f"F1 Score: {new_f1:.4f}")
                
                st.write("### Nova Matriz de Confusão")
                # Utiliza a nova função que passa sensitive_features
                fig_cm_new = plot_confusion_matrices_fair(best_model_fair, X_test_scaled, y_test, sensitive_features=sensitive_feature_test)
                st.pyplot(fig_cm_new)
                
                st.write("### Nova Análise de Viés")
                for attr in protected_attrs:
                    prot_attr_values = data.loc[y_test.index, attr].values
                    bias_metrics_new = compute_bias_metrics(y_test.values, best_model_fair.predict(X_test_scaled, sensitive_features=sensitive_feature_test), prot_attr_values)
                    st.write(f"**Atributo protegido: {attr}**")
                    st.write(bias_metrics_new)
                
                final_results_display = st.session_state.final_results_display
                insights_fair = generate_ai_insights(
                    ranking=final_results_display.to_string(index=False),
                    prec_recall_description=f"Acurácia, Precisão, Recall e F1 após fairness: {new_accuracy:.4f}, {new_precision:.4f}, {new_recall:.4f}, {new_f1:.4f}",
                    conf_matrix_description=f"A nova matriz de confusão:\n{confusion_matrix(y_test, best_model_fair.predict(X_test_scaled, sensitive_features=sensitive_feature_test))}",
                    feature_importance="(Texto resumido da importância das features após fairness)",  # Ajuste conforme necessário
                    shap_description="(Os resultados de SHAP não foram recalculados após fairness, pois predict_proba não está disponível)",
                    lime_description="(Os resultados de LIME não foram recalculados após fairness, pois predict_proba não está disponível)"
                )
                st.write("### Relatório Gerado pela IA (Após Fairness)")
                st.write(insights_fair)
