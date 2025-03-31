# config.py
"""
Arquivo de configuração para parâmetros globais.
"""

# Parâmetros globais
RANDOM_STATE = 42
N_ITERATIONS_DEFAULT = 10
SELECTION_SIZE_DEFAULT = 20
N_TRIALS_OPTUNA = 20

# Intervalos padrão de hiperparâmetros para RandomForest
RF_N_ESTIMATORS_RANGE = (50, 500)
RF_MAX_DEPTH_RANGE = (5, 50)
RF_MIN_SAMPLES_SPLIT_RANGE = (2, 10)
RF_MIN_SAMPLES_LEAF_RANGE = (1, 10)
