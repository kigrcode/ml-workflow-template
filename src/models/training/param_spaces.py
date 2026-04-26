"""
Hyperparameter search spaces for all models in the registry.
Used by both Optuna and RandomizedSearchCV.

Optuna spaces: callables that take a trial and return a param dict
Random search spaces: dicts of parameter names to distributions.

"""

import numpy as np

# =========================================================
# OPTUNA PARAMETER SPACES
# =========================================================

def logistic_regression_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "max_iter": 1000,
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "n_jobs": -1,
    }


def linear_regression_space(trial):
    # Linear regression has no meaningful hyperparameters to tune
    return {"n_jobs": -1}


def ridge_regression_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
    }


def lasso_regression_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "max_iter": 1000,
    }


def random_forest_classifier_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "n_jobs": -1,
    }


def random_forest_regressor_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "n_jobs": -1,
    }


def extra_trees_classifier_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "n_jobs": -1,
    }


def extra_trees_regressor_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "n_jobs": -1,
    }


def xgboost_classifier_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }


def xgboost_regressor_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "tree_method": "hist",
        "n_jobs": -1,
    }


def lightgbm_classifier_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }


def lightgbm_regressor_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }


def catboost_classifier_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "verbose": 0,
    }


def catboost_regressor_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "verbose": 0,
    }


# =========================================================
# REGISTRY — maps model name to param space function
# =========================================================

PARAM_SPACES = {
    "logistic_regression": logistic_regression_space,
    "linear_regression": linear_regression_space,
    "ridge_regression": ridge_regression_space,
    "lasso_regression": lasso_regression_space,
    "random_forest_classifier": random_forest_classifier_space,
    "random_forest_regressor": random_forest_regressor_space,
    "extra_trees_classifier": extra_trees_classifier_space,
    "extra_trees_regressor": extra_trees_regressor_space,
    "xgboost_classifier": xgboost_classifier_space,
    "xgboost_regressor": xgboost_regressor_space,
    "lightgbm_classifier": lightgbm_classifier_space,
    "lightgbm_regressor": lightgbm_regressor_space,
    "catboost_classifier": catboost_classifier_space,
    "catboost_regressor": catboost_regressor_space,
}


# =========================================================
# RANDOM SEARCH SPACES
# =========================================================
# Used when tuning.method = "random_search" in config.yaml
# scipy.stats distributions for continuous parameters

from scipy.stats import uniform, loguniform, randint

RANDOM_SEARCH_SPACES = {
    "logistic_regression": {
        "C": loguniform(1e-4, 10.0),
        "solver": ["lbfgs", "saga"],
        "max_iter": [1000],
        "n_jobs": [-1],
    },
    "ridge_regression": {
        "alpha": loguniform(1e-4, 10.0),
    },
    "lasso_regression": {
        "alpha": loguniform(1e-4, 10.0),
        "max_iter": [1000],
    },
    "random_forest_classifier": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2"],
        "n_jobs": [-1],
    },
    "random_forest_regressor": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2"],
        "n_jobs": [-1],
    },
    "extra_trees_classifier": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2"],
        "n_jobs": [-1],
    },
    "extra_trees_regressor": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2"],
        "n_jobs": [-1],
    },
    "xgboost_classifier": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0.0, 1.0),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 1.5),
        "tree_method": ["hist"],
        "n_jobs": [-1],
    },
    "xgboost_regressor": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0.0, 1.0),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 1.5),
        "tree_method": ["hist"],
        "n_jobs": [-1],
    },
    "lightgbm_classifier": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.01, 0.3),
        "num_leaves": randint(20, 150),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 50),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
    },
    "lightgbm_regressor": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.01, 0.3),
        "num_leaves": randint(20, 150),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(5, 50),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
    },
    "catboost_classifier": {
        "iterations": randint(100, 500),
        "depth": randint(3, 10),
        "learning_rate": loguniform(0.01, 0.3),
        "l2_leaf_reg": uniform(1.0, 9.0),
        "bagging_temperature": uniform(0.0, 1.0),
        "border_count": randint(32, 255),
        "verbose": [0],
    },
    "catboost_regressor": {
        "iterations": randint(100, 500),
        "depth": randint(3, 10),
        "learning_rate": loguniform(0.01, 0.3),
        "l2_leaf_reg": uniform(1.0, 9.0),
        "bagging_temperature": uniform(0.0, 1.0),
        "border_count": randint(32, 255),
        "verbose": [0],
    },
}