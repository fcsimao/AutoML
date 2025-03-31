"""
Aplicação Streamlit que integra os módulos de pré-processamento, 
otimização de hiperparâmetros e Active Learning.
Utiliza 15% dos dados reservados para avaliação final e validação cruzada interna no loop de Active Learning.
Adiciona gráficos de Precision-Recall e LIME para explicabilidade.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

from preprocessing_visualization import (
    load_dataset, display_dataset_info, display_distribution,
    split_dataset, reduce_labels, plot_distribution, plot_roc_curve, plot_confusion_matrices
)
from hyperparameter_optimization import (
    cached_optuna_optimization, cached_random_search
)
from active_learning import active_learning_loop_train

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, N_ITERATIONS_DEFAULT, SELECTION_SIZE_DEFAULT, N_TRIALS_OPTUNA

def main():
    st.title("Active Learning com Oversampling, Otimização e Redução de Rótulos")
    
    # Sidebar: Configurações de Hiperparâmetros
    st.sidebar.subheader("Configurações de Hiperparâmetros")
    rf_n_estimators_min = st.sidebar.number_input("n_estimators mínimo", value=50, step=1)
    rf_n_estimators_max = st.sidebar.number_input("n_estimators máximo", value=500, step=1)
    rf_max_depth_min = st.sidebar.number_input("max_depth mínimo", value=5, step=1)
    rf_max_depth_max = st.sidebar.number_input("max_depth máximo", value=50, step=1)
    rf_min_samples_split_min = st.sidebar.number_input("min_samples_split mínimo", value=2, step=1)
    rf_min_samples_split_max = st.sidebar.number_input("min_samples_split máximo", value=10, step=1)
    rf_min_samples_leaf_min = st.sidebar.number_input("min_samples_leaf mínimo", value=1, step=1)
    rf_min_samples_leaf_max = st.sidebar.number_input("min_samples_leaf máximo", value=10, step=1)
    
    st.sidebar.subheader("Configurações de Validação Cruzada")
    cv_folds = st.sidebar.number_input("Número de folds para CV", value=3, min_value=2, step=1)
    
    st.sidebar.subheader("Configurações do Optuna")
    n_trials = st.sidebar.number_input("Número de trials para otimização em cada iteração", value=5, min_value=1, step=1)
    
    st.sidebar.subheader("Estratégia de Active Learning")
    strategy = st.sidebar.selectbox("Selecione a estratégia", options=[
        "entropy", "margin", "random", "query_by_committee"
    ])
    
    # Constrói os intervalos com base nos inputs da sidebar
    n_estimators_range = (rf_n_estimators_min, rf_n_estimators_max)
    max_depth_range = (rf_max_depth_min, rf_max_depth_max)
    min_samples_split_range = (rf_min_samples_split_min, rf_min_samples_split_max)
    min_samples_leaf_range = (rf_min_samples_leaf_min, rf_min_samples_leaf_max)
    
    st.markdown("Carregue seu dataset (CSV) com 100% dos rótulos. **A coluna alvo deve ser 'Target'**.")
    uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        data_full = load_dataset(uploaded_file)
        target = "Target"
        display_dataset_info(data_full, target)
        display_distribution(data_full, target)
        
        # Divisão em treinamento e validação (15% dos dados para avaliação final)
        X_full = data_full.drop(columns=[target])
        y_full = data_full[target]
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X_full, y_full, test_size=0.15, random_state=RANDOM_STATE, stratify=y_full
        )
        data_train_full = X_train_full.copy()
        data_train_full[target] = y_train_full.copy()
        st.write("### Conjunto de Validação (15% dos dados completos)")
        st.dataframe(X_val.head())
        
        # Redução de rótulos no conjunto de treinamento
        st.markdown("### Redução de Rótulos no Conjunto de Treinamento")
        percent_labeled = st.slider("Porcentagem de rótulos a manter no treinamento", min_value=5, max_value=100, value=20)
        data_train_masked = X_train_full.copy()
        data_train_masked[target] = y_train_full.copy()
        data_train_masked = reduce_labels(data_train_masked, target, percent_labeled, RANDOM_STATE)
        st.write(f"### Exemplo após Redução ({percent_labeled}% dos rótulos mantidos)")
        st.dataframe(data_train_masked.head())
        st.write("**Tipo após redução:**", data_train_masked["Target"].dtype)
        st.write("**Valores únicos após redução:**", data_train_masked["Target"].unique())
        plot_distribution(data_train_masked, target, "Distribuição (Treinamento após Redução)")
        
        # Oversampling nos dados rotulados
        st.markdown("### Aplicação do Oversampling (apenas nos dados rotulados do treinamento)")
        labeled_data = data_train_masked[data_train_masked[target].notnull()]
        X_labeled = labeled_data.drop(columns=[target])
        y_labeled = labeled_data[target].astype(int)
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_over, y_over = ros.fit_resample(X_labeled, y_labeled)
        st.write("### Distribuição Após Oversampling")
        dist_over = pd.DataFrame(y_over.value_counts()).reset_index()
        dist_over.columns = ["Target", "Count"]
        dist_over["Percent"] = (dist_over["Count"] / len(y_over)) * 100
        st.dataframe(dist_over)
        st.bar_chart(dist_over.set_index("Target")["Count"])
        
        total_dataset = len(data_full)
        st.write("Total de instâncias:", total_dataset)
        
        # Otimização de Hiperparâmetros
        st.markdown("### Otimização de Hiperparâmetros (usando dados rotulados do treinamento)")
        opt_method = st.selectbox("Método de otimização", ["Optuna", "Random Search"])
        if opt_method == "Optuna":
            best_params = cached_optuna_optimization(
                X_labeled, y_labeled, N_TRIALS_OPTUNA,
                n_estimators_range, max_depth_range,
                min_samples_split_range, min_samples_leaf_range,
                cv_folds=cv_folds
            )
            st.write("Melhores hiperparâmetros (Optuna):", best_params)
        else:
            best_params = cached_random_search(
                X_labeled, y_labeled,
                n_estimators_range, max_depth_range,
                min_samples_split_range, min_samples_leaf_range,
                cv_folds=cv_folds
            )
            st.write("Melhores hiperparâmetros (Random Search):", best_params)
        
        # Configurações do Active Learning
        st.markdown("### Configurações do Active Learning")
        n_iterations = st.number_input("Número de iterações", min_value=1, value=N_ITERATIONS_DEFAULT)
        selection_size = st.number_input("Número de amostras adquiridas por iteração", min_value=1, value=SELECTION_SIZE_DEFAULT)
        
        if st.button("Iniciar Active Learning"):
            performance_history, final_model = active_learning_loop_train(
                data_train_masked, data_train_full, target, best_params,
                n_iterations, selection_size, cv_folds=cv_folds, strategy=strategy, n_trials=n_trials
            )
            st.write("### Histórico de Performance (Active Learning)")
            perf_df = pd.DataFrame(performance_history)
            st.dataframe(perf_df)
            fig1, ax1 = plt.subplots()
            ax1.plot(perf_df["iteration"], perf_df["f1"], marker="o", linestyle="--")
            ax1.set_xlabel("Iteração")
            ax1.set_ylabel("F1 Score (Weighted)")
            ax1.set_title("Evolução do F1 Score no Active Learning")
            st.pyplot(fig1)
            
            # Avaliação final utilizando os 15% de dados reservados para validação
            st.write("### Avaliação Final no Conjunto de Validação (15% dos dados completos)")
            y_val_pred = final_model.predict(X_val)
            final_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=1)
            final_acc = accuracy_score(y_val, y_val_pred)
            final_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=1)
            final_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=1)
            st.write("**Métricas do Modelo Final:**")
            st.write(f"Acurácia: {final_acc:.4f}")
            st.write(f"Precisão: {final_precision:.4f}")
            st.write(f"Recall: {final_recall:.4f}")
            st.write(f"F1 Score: {final_f1:.4f}")
            
            # Cálculo de Especificidade e Taxa de Falsos Negativos (para problemas binários)
            cm = confusion_matrix(y_val, y_val_pred)
            if len(np.unique(y_val)) == 2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                st.write(f"Especificidade (True Negative Rate): {specificity:.4f}")
                st.write(f"Taxa de Falsos Negativos (FNR): {fnr:.4f}")
            else:
                st.write("Cálculo de Especificidade e FNR não está implementado para classificação multiclasse.")
            
            # Exibição de interpretações das métricas
            st.write("### Interpretação das Métricas")
            st.write("""
**Definições:**

- **Rótulo 1:** O aluno evadiu.
- **Rótulo 0:** O aluno não evadiu.

**Recall (Sensibilidade) para evasão:**  
Calculado como TP / (TP + FN). Representa a proporção de alunos que realmente evadiram que foram corretamente identificados.  
Valor calculado: {:.4f}

**Precisão para evasão:**  
Calculado como TP / (TP + FP). Indica, entre os alunos previstos como evadindo, a proporção que de fato evadiu.  
Valor calculado: {:.4f}

**Especificidade (True Negative Rate):**  
Calculado como TN / (TN + FP). Representa a proporção de alunos que não evadiram que foram corretamente identificados.  
Valor calculado: {:.4f}

**Taxa de Falsos Negativos (FNR):**  
Calculada como FN / (TP + FN).  
Valor calculado: {:.4f}
            """.format(final_recall, final_precision, specificity, fnr))
            
            # Quadro comparativo de métricas
            last_iter = perf_df.iloc[-1]
            training_metrics = {
                "F1 Score (CV Active Learning)": last_iter["f1"],
                "Número de Registros Rotulados": last_iter["labeled_count"]
            }
            validation_metrics = {
                "F1 Score (15% Validação)": final_f1,
                "Número de Registros Rotulados": len(y_val)
            }
            metrics_df = pd.DataFrame([training_metrics, validation_metrics], index=["Treino", "Validação"])
            st.write("### Quadro Comparativo de Métricas")
            st.dataframe(metrics_df)
            
            # Matriz de Confusão
            from sklearn.metrics import ConfusionMatrixDisplay
            fig2, ax2 = plt.subplots()
            disp = ConfusionMatrixDisplay.from_estimator(final_model, X_val, y_val, cmap=plt.cm.Blues, ax=ax2)
            ax2.set_title("Matriz de Confusão (Validação)")
            st.pyplot(fig2)
            
            # Curva ROC (para classificação binária e modelo com predict_proba)
            if hasattr(final_model, "predict_proba") and len(np.unique(y_val)) == 2:
                pos_label = np.unique(y_val)[1]
                fpr, tpr, thresholds = roc_curve(y_val, final_model.predict_proba(X_val)[:, 1], pos_label=pos_label)
                roc_auc = auc(fpr, tpr)
                fig3, ax3 = plt.subplots()
                ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax3.set_xlim([0.0, 1.0])
                ax3.set_ylim([0.0, 1.05])
                ax3.set_xlabel('False Positive Rate')
                ax3.set_ylabel('True Positive Rate')
                ax3.set_title('Curva ROC')
                ax3.legend(loc="lower right")
                st.pyplot(fig3)
            else:
                st.write("Curva ROC não disponível para classificação multiclasse ou modelo sem predict_proba.")
            
            # Gráfico de Precision-Recall
            if hasattr(final_model, "predict_proba") and len(np.unique(y_val)) == 2:
                probas = final_model.predict_proba(X_val)[:, 1]
                precision, recall, _ = precision_recall_curve(y_val, probas)
                avg_precision = average_precision_score(y_val, probas)
                fig4, ax4 = plt.subplots()
                ax4.plot(recall, precision, color='purple', lw=2, label=f'AP = {avg_precision:.2f}')
                ax4.set_xlabel('Recall')
                ax4.set_ylabel('Precision')
                ax4.set_title('Precision-Recall Curve')
                ax4.legend(loc="lower left")
                st.pyplot(fig4)
            else:
                st.write("Curva Precision-Recall não disponível.")
            
            # Gráfico de Importância das Features (horizontal)
            if hasattr(final_model, "feature_importances_"):
                importances = final_model.feature_importances_
                features = X_train_full.columns  # Certifique-se de que X_train_full contém todas as features usadas
                indices = np.argsort(importances)[::-1]
                sorted_features = features[indices]
                sorted_importances = importances[indices]
                
                fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
                ax_feat.barh(range(len(sorted_features)), sorted_importances, align='center')
                ax_feat.set_yticks(range(len(sorted_features)))
                ax_feat.set_yticklabels(sorted_features)
                ax_feat.invert_yaxis()  # Para que a feature mais importante apareça no topo
                ax_feat.set_title("Importância das Features")
                ax_feat.set_xlabel("Importância")
                st.pyplot(fig_feat)
            else:
                st.write("O modelo não fornece importâncias das features.")
            
            # Informações adicionais para debug
            st.write("Colunas de X_val:", X_val.columns)
            st.write("Número de colunas:", len(X_val.columns))
            st.write("Valores únicos no campo Target:", y_val.unique())
            st.write("Distribuição das classes no campo Target:")
            st.write(y_val.value_counts())

if __name__ == "__main__":
    main()
