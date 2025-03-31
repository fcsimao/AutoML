# hyperparameter_optimization.py
"""
Módulo para otimização de hiperparâmetros utilizando Optuna e RandomizedSearchCV.
"""

import numpy as np
import optuna
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from config import RF_N_ESTIMATORS_RANGE, RF_MAX_DEPTH_RANGE, RF_MIN_SAMPLES_SPLIT_RANGE, RF_MIN_SAMPLES_LEAF_RANGE

def objective(trial, X_labeled, y_labeled,
              n_estimators_range=None, max_depth_range=None,
              min_samples_split_range=None, min_samples_leaf_range=None,
              cv_folds=3):
    """
    Função objetivo para otimização via Optuna.
    
    Parameters:
        trial: Objeto trial do Optuna.
        X_labeled (pd.DataFrame): Dados de treinamento rotulados.
        y_labeled (pd.Series): Rótulos dos dados de treinamento.
        n_estimators_range (tuple): Intervalo para n_estimators.
        max_depth_range (tuple): Intervalo para max_depth.
        min_samples_split_range (tuple): Intervalo para min_samples_split.
        min_samples_leaf_range (tuple): Intervalo para min_samples_leaf.
        cv_folds (int): Número de folds para validação cruzada.
        
    Returns:
        float: Média do f1_weighted score da validação cruzada.
    """
    if n_estimators_range is None:
        n_estimators_range = RF_N_ESTIMATORS_RANGE
    if max_depth_range is None:
        max_depth_range = RF_MAX_DEPTH_RANGE
    if min_samples_split_range is None:
        min_samples_split_range = RF_MIN_SAMPLES_SPLIT_RANGE
    if min_samples_leaf_range is None:
        min_samples_leaf_range = RF_MIN_SAMPLES_LEAF_RANGE

    n_estimators = trial.suggest_int("n_estimators", n_estimators_range[0], n_estimators_range[1])
    max_depth = trial.suggest_int("max_depth", max_depth_range[0], max_depth_range[1])
    min_samples_split = trial.suggest_int("min_samples_split", min_samples_split_range[0], min_samples_split_range[1])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", min_samples_leaf_range[0], min_samples_leaf_range[1])
    
    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    score = cross_val_score(clf, X_labeled, y_labeled, cv=cv_folds, scoring='f1_weighted').mean()
    return score

def random_search(X_labeled, y_labeled,
                  n_estimators_range=None, max_depth_range=None,
                  min_samples_split_range=None, min_samples_leaf_range=None,
                  cv_folds=3):
    """
    Otimiza hiperparâmetros utilizando Randomized Search.
    
    Parameters:
        X_labeled (pd.DataFrame): Dados de treinamento rotulados.
        y_labeled (pd.Series): Rótulos dos dados de treinamento.
        n_estimators_range (tuple): Intervalo para n_estimators.
        max_depth_range (tuple): Intervalo para max_depth.
        min_samples_split_range (tuple): Intervalo para min_samples_split.
        min_samples_leaf_range (tuple): Intervalo para min_samples_leaf.
        cv_folds (int): Número de folds para validação cruzada.
        
    Returns:
        dict: Melhor conjunto de hiperparâmetros encontrados.
    """
    if n_estimators_range is None:
        n_estimators_range = RF_N_ESTIMATORS_RANGE
    if max_depth_range is None:
        max_depth_range = RF_MAX_DEPTH_RANGE
    if min_samples_split_range is None:
        min_samples_split_range = RF_MIN_SAMPLES_SPLIT_RANGE
    if min_samples_leaf_range is None:
        min_samples_leaf_range = RF_MIN_SAMPLES_LEAF_RANGE

    param_distributions = {
        "n_estimators": list(range(n_estimators_range[0], n_estimators_range[1] + 1, 50)),
        "max_depth": [None] + list(range(max_depth_range[0], max_depth_range[1] + 1, 5)),
        "min_samples_split": list(range(min_samples_split_range[0], min_samples_split_range[1] + 1)),
        "min_samples_leaf": list(range(min_samples_leaf_range[0], min_samples_leaf_range[1] + 1))
    }
    clf = RandomForestClassifier(random_state=42)
    rs = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=20,
        cv=cv_folds,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1
    )
    rs.fit(X_labeled, y_labeled)
    return rs.best_params_

@st.cache_data(show_spinner="Otimizando hiperparâmetros com Optuna...")
def cached_optuna_optimization(X_labeled, y_labeled, n_trials,
                               n_estimators_range, max_depth_range,
                               min_samples_split_range, min_samples_leaf_range,
                               cv_folds=3):
    """
    Executa a otimização via Optuna com caching.
    
    Parameters:
        X_labeled (pd.DataFrame): Dados de treinamento rotulados.
        y_labeled (pd.Series): Rótulos dos dados de treinamento.
        n_trials (int): Número de trials para o Optuna.
        n_estimators_range (tuple): Intervalo para n_estimators.
        max_depth_range (tuple): Intervalo para max_depth.
        min_samples_split_range (tuple): Intervalo para min_samples_split.
        min_samples_leaf_range (tuple): Intervalo para min_samples_leaf.
        cv_folds (int): Número de folds para validação cruzada.
        
    Returns:
        dict: Melhor conjunto de hiperparâmetros encontrados.
    """
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective_func = lambda trial: objective(
        trial, X_labeled, y_labeled,
        n_estimators_range=n_estimators_range,
        max_depth_range=max_depth_range,
        min_samples_split_range=min_samples_split_range,
        min_samples_leaf_range=min_samples_leaf_range,
        cv_folds=cv_folds
    )
    study.optimize(objective_func, n_trials=n_trials)
    return study.best_trial.params

@st.cache_data(show_spinner="Otimizando hiperparâmetros com Random Search...")
def cached_random_search(X_labeled, y_labeled, n_estimators_range, max_depth_range,
                         min_samples_split_range, min_samples_leaf_range,
                         cv_folds=3):
    """
    Executa a otimização via Random Search com caching.
    
    Parameters:
        X_labeled (pd.DataFrame): Dados de treinamento rotulados.
        y_labeled (pd.Series): Rótulos dos dados de treinamento.
        n_estimators_range (tuple): Intervalo para n_estimators.
        max_depth_range (tuple): Intervalo para max_depth.
        min_samples_split_range (tuple): Intervalo para min_samples_split.
        min_samples_leaf_range (tuple): Intervalo para min_samples_leaf.
        cv_folds (int): Número de folds para validação cruzada.
        
    Returns:
        dict: Melhor conjunto de hiperparâmetros encontrados.
    """
    return random_search(
        X_labeled, y_labeled,
        n_estimators_range=n_estimators_range,
        max_depth_range=max_depth_range,
        min_samples_split_range=min_samples_split_range,
        min_samples_leaf_range=min_samples_leaf_range,
        cv_folds=cv_folds
    )

def optimize_hyperparameters_with_optuna(model_class, X_train, y_train, n_trials=10):
    """
    Otimiza hiperparâmetros usando Optuna para um modelo dado.
    
    Parameters:
        model_class: Classe do modelo (ex: RandomForestClassifier).
        X_train (pd.DataFrame): Dados de treinamento.
        y_train (pd.Series): Rótulos de treinamento.
        n_trials (int): Número de iterações (trials) do Optuna.
        
    Returns:
        dict: Melhor conjunto de hiperparâmetros encontrados.
    """
    def objective_optuna(trial):
        params = {}
        if model_class == RandomForestClassifier:
            params["n_estimators"] = trial.suggest_int("n_estimators", RF_N_ESTIMATORS_RANGE[0], RF_N_ESTIMATORS_RANGE[1])
            params["max_depth"] = trial.suggest_int("max_depth", RF_MAX_DEPTH_RANGE[0], RF_MAX_DEPTH_RANGE[1])
        model = model_class(random_state=42, **params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
        return np.mean(scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna, n_trials=n_trials)
    return study.best_params

def create_model(model_class, **params):
    """
    Cria uma instância do modelo com os parâmetros fornecidos.
    
    Parameters:
        model_class: Classe do modelo.
        params: Parâmetros para inicialização do modelo.
        
    Returns:
        model: Instância do modelo criado.
    """
    return model_class(random_state=42, **params)
