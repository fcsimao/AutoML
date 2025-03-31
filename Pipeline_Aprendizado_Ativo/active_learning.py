# active_learning.py
"""
Módulo para o loop de Active Learning utilizando estratégias tradicionais.
Nesta versão, em cada iteração, os hiperparâmetros são re-otimizados com Optuna
utilizando os dados rotulados acumulados.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
import streamlit as st
from config import RANDOM_STATE

def compute_entropy(probas):
    """
    Calcula a entropia para cada vetor de probabilidades.
    
    Parameters:
        probas (np.array): Array de probabilidades (n_samples x n_classes).
        
    Returns:
        np.array: Array com a entropia para cada amostra.
    """
    epsilon = 1e-12
    return -np.sum(probas * np.log(probas + epsilon), axis=1)

def active_learning_loop_train(data_train_masked, data_train_full, target, best_params,
                               n_iterations=10, selection_size=20, cv_folds=3, strategy="entropy",
                               n_trials=5):
    """
    Realiza o loop de Active Learning utilizando os dados de treinamento mascarados.
    
    Em cada iteração:
      1. Avalia a performance atual via validação cruzada (cv_folds) sobre os dados rotulados acumulados.
      2. Re-otimiza os hiperparâmetros utilizando Optuna com os dados atuais (n_trials).
      3. Treina um modelo com os novos hiperparâmetros.
      4. Seleciona novas amostras do pool não rotulado com base na estratégia escolhida.
    
    Ao final, treina o modelo final com todos os dados rotulados adquiridos.
    
    Parameters:
        data_train_masked (pd.DataFrame): Dados de treinamento com rótulos mascarados (NaN).
        data_train_full (pd.DataFrame): Dados de treinamento com rótulos completos.
        target (str): Nome da coluna alvo.
        best_params (dict): Hiperparâmetros iniciais (serão atualizados a cada iteração).
        n_iterations (int): Número de iterações do loop.
        selection_size (int): Número de amostras a serem adquiridas por iteração.
        cv_folds (int): Número de folds para validação cruzada.
        strategy (str): Estratégia de seleção das amostras. Opções: "entropy", "margin", "random", "query_by_committee".
        n_trials (int): Número de trials para a re-otimização dos hiperparâmetros em cada iteração.
        
    Returns:
        tuple: (performance_history, final_model)
    """
    # Separa os dados: X (features) e y (rótulos completos)
    X_train = data_train_masked.drop(columns=[target])
    y_masked_train = data_train_masked[target]
    y_full_train = data_train_full[target]

    # Define conjuntos inicialmente rotulados e não rotulados
    labeled_mask = y_masked_train.notnull()
    X_L = X_train[labeled_mask].reset_index(drop=True)
    y_L = y_masked_train[labeled_mask].astype(int).reset_index(drop=True)
    X_U = X_train[~labeled_mask].reset_index(drop=True)
    y_U = y_full_train[~labeled_mask].astype(int).reset_index(drop=True)

    performance_history = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Configura validação cruzada
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Função auxiliar para seleção de amostras pelas estratégias tradicionais
    def select_indices(model, X_U, strategy, selection_size):
        if strategy == "entropy":
            probas = model.predict_proba(X_U)
            uncertainty = compute_entropy(probas)
            return np.argsort(-uncertainty)[:selection_size]
        elif strategy == "margin":
            probas = model.predict_proba(X_U)
            sorted_probas = np.sort(probas, axis=1)[:, ::-1]
            margin = sorted_probas[:, 0] - sorted_probas[:, 1]
            return np.argsort(margin)[:selection_size]
        elif strategy == "random":
            return np.random.choice(len(X_U), size=selection_size, replace=False)
        elif strategy == "query_by_committee":
            committee_preds = []
            for seed in [42, 43, 44]:
                m = RandomForestClassifier(random_state=seed, **best_params)
                m.fit(X_L, y_L)
                committee_preds.append(m.predict(X_U))
            committee_preds = np.array(committee_preds)
            disagreement = []
            for i in range(committee_preds.shape[1]):
                preds = committee_preds[:, i]
                counts = np.bincount(preds)
                most_common = counts.max() / len(committee_preds)
                disagreement.append(1 - most_common)
            disagreement = np.array(disagreement)
            return np.argsort(-disagreement)[:selection_size]
        else:
            probas = model.predict_proba(X_U)
            uncertainty = compute_entropy(probas)
            return np.argsort(-uncertainty)[:selection_size]

    for iteration in range(n_iterations):
        if X_U.empty:
            st.info("Todos os dados não rotulados foram adquiridos.")
            break

        # Avaliação via CV com os hiperparâmetros atuais
        cv_scores = []
        for train_idx, test_idx in skf.split(X_L, y_L):
            model_cv = RandomForestClassifier(random_state=42, **best_params)
            model_cv.fit(X_L.iloc[train_idx], y_L.iloc[train_idx])
            preds = model_cv.predict(X_L.iloc[test_idx])
            cv_scores.append(f1_score(y_L.iloc[test_idx], preds, average='weighted', zero_division=1))
        avg_f1 = np.mean(cv_scores)
        performance_history.append({
            "iteration": iteration,
            "f1": avg_f1,
            "labeled_count": len(X_L)
        })
        status_text.text(f"Iteração {iteration}: F1 CV score = {avg_f1:.4f}")

        # Re-otimização dos hiperparâmetros com os dados rotulados atuais usando Optuna
        def objective_iter(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 5, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            clf = RandomForestClassifier(
                random_state=42,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
            return cross_val_score(clf, X_L, y_L, cv=cv_folds, scoring="f1_weighted").mean()

        study_iter = optuna.create_study(direction="maximize")
        study_iter.optimize(objective_iter, n_trials=n_trials)
        best_params = study_iter.best_trial.params

        # Treina o modelo com os novos hiperparâmetros otimizados
        model = RandomForestClassifier(random_state=42, **best_params)
        model.fit(X_L, y_L)

        # Seleciona novas amostras do conjunto não rotulado
        indices = select_indices(model, X_U, strategy, selection_size)
        X_selected = X_U.iloc[indices]
        y_selected = y_U.iloc[indices]

        # Atualiza os conjuntos: adiciona as amostras selecionadas ao conjunto rotulado e remove do pool não rotulado
        X_L = pd.concat([X_L, X_selected], ignore_index=True)
        y_L = pd.concat([y_L, y_selected], ignore_index=True)
        X_U = X_U.drop(X_selected.index).reset_index(drop=True)
        y_U = y_U.drop(y_selected.index).reset_index(drop=True)

        progress_bar.progress((iteration + 1) / n_iterations)

    # Treina o modelo final com todos os dados rotulados adquiridos
    final_model = RandomForestClassifier(random_state=42, **best_params)
    final_model.fit(X_L, y_L)
    return performance_history, final_model
