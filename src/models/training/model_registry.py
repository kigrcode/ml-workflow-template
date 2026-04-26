from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

# Optional imports — gracefully handled if not installed
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


MODEL_REGISTRY = {
    # Baseline linear models
    "logistic_regression": {
        "model": LogisticRegression,
        "task": "classification",
        "default_params": {
            "max_iter": 1000,
            "solver": "lbfgs",
            "n_jobs": -1
        }
    },
    "linear_regression": {
        "model": LinearRegression,
        "task": "regression",
        "default_params": {
            "n_jobs": -1
        }
    },
    "ridge_regression": {
        "model": Ridge,
        "task": "regression",
        "default_params": {
            "alpha": 1.0
        }
    },
    "lasso_regression": {
        "model": Lasso,
        "task": "regression",
        "default_params": {
            "alpha": 1.0,
            "max_iter": 1000
        }
    },

    # Tree ensembles
    "random_forest_classifier": {
        "model": RandomForestClassifier,
        "task": "classification",
        "default_params": {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1
        }
    },
    "random_forest_regressor": {
        "model": RandomForestRegressor,
        "task": "regression",
        "default_params": {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1
        }
    },
    "extra_trees_classifier": {
        "model": ExtraTreesClassifier,
        "task": "classification",
        "default_params": {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1
        }
    },
    "extra_trees_regressor": {
        "model": ExtraTreesRegressor,
        "task": "regression",
        "default_params": {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1
        }
    },
}

# Conditionally add boosted models
if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["xgboost_classifier"] = {
        "model": XGBClassifier,
        "task": "classification",
        "default_params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": -1
        }
    }
    MODEL_REGISTRY["xgboost_regressor"] = {
        "model": XGBRegressor,
        "task": "regression",
        "default_params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1
        }
    }

if LIGHTGBM_AVAILABLE:
    MODEL_REGISTRY["lightgbm_classifier"] = {
        "model": LGBMClassifier,
        "task": "classification",
        "default_params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31
        }
    }
    MODEL_REGISTRY["lightgbm_regressor"] = {
        "model": LGBMRegressor,
        "task": "regression",
        "default_params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31
        }
    }

if CATBOOST_AVAILABLE:
    MODEL_REGISTRY["catboost_classifier"] = {
        "model": CatBoostClassifier,
        "task": "classification",
        "default_params": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0
        }
    }
    MODEL_REGISTRY["catboost_regressor"] = {
        "model": CatBoostRegressor,
        "task": "regression",
        "default_params": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0
        }
    }