"""
Hyperparameter tuning utilities.
Supports random_search and optuna methods.
Method is configured via config.yaml under tuning.method.
"""

from rich import print
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator

# Optuna is optional — gracefully handled if not installed
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =========================================================
# SCORING HELPER
# =========================================================
def get_scoring_metric(task: str, config: dict) -> str:
    """
    Auto-detect scoring metric from task type.
    Can be overridden in config.yaml under tuning.scoring.
    """
    if "scoring" in config.get("tuning", {}):
        return config["tuning"]["scoring"]

    if task == "classification":
        return "roc_auc"
    elif task == "regression":
        return "neg_root_mean_squared_error"
    else:
        raise ValueError(f"Unknown task type: {task}")


# =========================================================
# RANDOM SEARCH
# =========================================================
def train_random_search(
    base_model: BaseEstimator,
    param_distributions: dict,
    X_train,
    y_train,
    cv: int,
    n_iter: int,
    scoring: str,
    random_state: int,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """
    Tune hyperparameters using RandomizedSearchCV.
    More efficient than GridSearchCV for large parameter spaces.
    param_distributions: dict of parameter names to distributions or lists
    """
    print("[bold green]Tuning with RandomizedSearchCV...[/bold green]")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"[bold green]Best score:[/bold green] {search.best_score_:.4f}")
    print(f"[bold green]Best params:[/bold green] {search.best_params_}")

    return search


# =========================================================
# OPTUNA
# =========================================================
def train_optuna(
    model_class,
    param_space_fn,
    X_train,
    y_train,
    cv: int,
    n_trials: int,
    scoring: str,
    random_state: int,
    n_jobs: int = -1,
):
    """
    Tune hyperparameters using Optuna Bayesian optimisation.
    More efficient than random search — learns from previous trials.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Run: pip install optuna "
            "or set tuning.method to random_search in config.yaml"
        )

    print("[bold green]Tuning with Optuna...[/bold green]")

    def objective(trial):
        params = param_space_fn(trial)
        model = model_class(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return scores.mean()

    direction = "minimize" if "neg" in scoring else "maximize"

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"[bold green]Best score:[/bold green] {study.best_value:.4f}")
    print(f"[bold green]Best params:[/bold green] {study.best_params}")

    return study


# =========================================================
# ORCHESTRATOR
# =========================================================
def run_tuning(
    model_class,
    X_train,
    y_train,
    task: str,
    config: dict,
    param_space,
    base_model: BaseEstimator = None,
):
    """
    Run hyperparameter tuning based on method in config.yaml.
    Scoring metric is auto-detected from task type unless
    overridden in config.yaml under tuning.scoring.

    For random_search: param_space should be a dict of distributions
    For optuna: param_space should be a callable that takes a trial
    Returns either a RandomizedSearchCV object or an Optuna study.
    """
    tuning_config = config.get("tuning", {})
    method = tuning_config.get("method", "optuna")
    n_trials = tuning_config.get("n_trials", 50)
    cv = tuning_config.get("cv_folds", 5)
    random_state = tuning_config.get("random_state", 42)
    scoring = get_scoring_metric(task, config)

    print(f"Tuning method:  {method}")
    print(f"Scoring metric: {scoring}")
    print(f"Trials/iters:   {n_trials}")
    print(f"CV folds:       {cv}")

    if method == "random_search":
        if base_model is None:
            raise ValueError(
                "base_model must be provided for random_search. "
                "Pass an instantiated model with default params."
            )
        return train_random_search(
            base_model=base_model,
            param_distributions=param_space,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            n_iter=n_trials,
            scoring=scoring,
            random_state=random_state,
        )

    elif method == "optuna":
        return train_optuna(
            model_class=model_class,
            param_space_fn=param_space,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            n_trials=n_trials,
            scoring=scoring,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown tuning method: {method}. "
            "Options are: random_search, optuna"
        )