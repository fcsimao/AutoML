# preprocessing_visualization.py
"""
Módulo para pré-processamento e visualização dos dados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from config import RANDOM_STATE

@st.cache_data(show_spinner="Carregando dataset...")
def load_dataset(file) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.

    Parameters:
        file: Arquivo CSV carregado (por exemplo, via st.file_uploader)

    Returns:
        pd.DataFrame: Dataset lido.
    """
    return pd.read_csv(file)

def display_dataset_info(data: pd.DataFrame, target: str):
    """
    Exibe informações iniciais do dataset no Streamlit.

    Parameters:
        data (pd.DataFrame): Dataset completo.
        target (str): Nome da coluna alvo.
    """
    st.write("### Dataset Original (100% rotulado)")
    st.dataframe(data.head())
    st.write(f"**Tipo da coluna '{target}':**", data[target].dtype)
    st.write("**Primeiros valores:**", data[target].head())
    st.write("**Valores únicos:**", data[target].unique())

def display_distribution(data: pd.DataFrame, target: str):
    """
    Exibe a distribuição da coluna alvo e gera um gráfico de barras.

    Parameters:
        data (pd.DataFrame): Dataset.
        target (str): Nome da coluna alvo.
    """
    st.write("### Distribuição (Antes da Redução)")
    dist = data[target].value_counts(dropna=False).reset_index()
    dist.columns = [target, "Count"]
    dist["Percent"] = (dist["Count"] / len(data)) * 100
    st.dataframe(dist)
    st.bar_chart(dist.set_index(target)["Count"])

def split_dataset(data: pd.DataFrame, target: str, test_size: float = 0.15):
    """
    Realiza a divisão do dataset em treinamento e validação.

    Parameters:
        data (pd.DataFrame): Dataset completo.
        target (str): Nome da coluna alvo.
        test_size (float): Proporção dos dados para validação.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_val, y_train, y_val

def reduce_labels(data: pd.DataFrame, target: str, percentage: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Reduz a quantidade de rótulos no dataset, mascarando uma porcentagem dos mesmos.

    Parameters:
        data (pd.DataFrame): Dataset de treinamento.
        target (str): Nome da coluna alvo.
        percentage (int): Porcentagem de rótulos a serem mantidos.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        pd.DataFrame: Dataset com rótulos reduzidos.
    """
    data_masked = data.copy()
    np.random.seed(random_state)
    mask = np.random.rand(len(data_masked)) < (percentage / 100.0)
    data_masked.loc[~mask, target] = np.nan
    data_masked[target] = data_masked[target].apply(lambda x: int(x) if pd.notnull(x) else x)
    return data_masked

def plot_distribution(data: pd.DataFrame, target: str, title: str = "Distribuição") -> None:
    """
    Plota um gráfico de barras com a distribuição da coluna alvo.

    Parameters:
        data (pd.DataFrame): Dataset.
        target (str): Nome da coluna alvo.
        title (str): Título a ser exibido.
    """
    dist = data[target].value_counts(dropna=False).reset_index()
    dist.columns = [target, "Count"]
    dist["Percent"] = (dist["Count"] / len(data)) * 100
    st.write(title)
    st.dataframe(dist)
    st.bar_chart(dist.set_index(target)["Count"])

def plot_roc_curve(model, X_test, y_test):
    """
    Plota a curva ROC para modelos de classificação binária.

    Parameters:
        model: Modelo treinado que implementa predict_proba.
        X_test (pd.DataFrame): Conjunto de teste.
        y_test (pd.Series): Rótulos reais do conjunto de teste.

    Returns:
        fig: Figura matplotlib com a ROC curve.
    """
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots()
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

def plot_confusion_matrices(model, X_test, y_test):
    """
    Plota as matrizes de confusão (valores absolutos e normalizada).

    Parameters:
        model: Modelo treinado.
        X_test (pd.DataFrame): Conjunto de teste.
        y_test (pd.Series): Rótulos reais do conjunto de teste.

    Returns:
        fig: Figura matplotlib com as matrizes de confusão.
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                          display_labels=["Classe 0", "Classe 1"],
                                          cmap=plt.cm.Blues,
                                          normalize=None,
                                          ax=axes[0])
    axes[0].set_title("Matriz de Confusão - Valores Absolutos")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                          display_labels=["Classe 0", "Classe 1"],
                                          cmap=plt.cm.Blues,
                                          normalize='true',
                                          ax=axes[1])
    axes[1].set_title("Matriz de Confusão - Normalizada")
    fig.tight_layout()
    return fig
