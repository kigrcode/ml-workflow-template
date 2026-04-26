from typing import Dict, Any, Optional, List, Tuple

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

from src.models.training.model_registry import MODEL_REGISTRY
from src.models.training.validation import validate_param_overrides
from src.models.training.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    extract_primary_metric,
)


class ModelTrainer:
    """
    Stateful multi-model trainer with:
    - flat model registry
    - default hyperparameters + per-model overrides
    - strict validation of overrides
    - task-aware metrics and CV
    - configurable model selection via run_all / model list
    - CatBoost receives raw data, all other models receive preprocessed data

    IMPORTANT: The preprocessor passed to fit() must be UNFITTED.
    It will be fitted inside each CV fold on training data only,
    preventing data leakage.
    """

    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]] = MODEL_REGISTRY,
        random_state: int = 42,
    ):
        self.model_registry = model_registry
        self.random_state = random_state

        self.task_: Optional[str] = None
        self.leaderboard_: Optional[pd.DataFrame] = None
        self.models_: Dict[str, Any] = {}
        self.cv_results_: Dict[str, Dict[str, Any]] = {}
        self.feature_names_: Optional[List[str]] = None

    # =========================================================
    # PUBLIC API
    # =========================================================
    def fit(
        self,
        X,
        y,
        preprocessor,
        param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        cv: int = 5,
        run_all: bool = True,
        model_subset: Optional[List[str]] = None,
    ) -> "ModelTrainer":
        """
        X:              raw pandas DataFrame (unfitted preprocessor handles transformation)
        y:              array-like target
        preprocessor:   UNFITTED FeatureEngineeringPipeline instance
        param_overrides: dict[model_name] -> dict of hyperparameters
        cv:             number of CV folds
        run_all:        if True, run all models for the detected task
                        if False, only run models in model_subset
        model_subset:   list of model names to run when run_all=False
        """
        if param_overrides is None:
            param_overrides = {}

        X = self._ensure_dataframe(X)
        y = np.asarray(y)

        self.task_ = self._infer_task(y)
        self.feature_names_ = list(X.columns)

        print(f"Task detected: {self.task_}")

        splitter = self._get_cv_splitter(y, cv)

        leaderboard_rows: List[Dict[str, Any]] = []
        self.models_.clear()
        self.cv_results_.clear()

        for model_name, entry in self._iter_models_for_task(self.task_):

            # Apply model subset filter if run_all is False
            if not run_all and model_subset:
                if model_name not in model_subset:
                    continue

            model_class = entry["model"]
            default_params = entry.get("default_params", {})
            overrides = param_overrides.get(model_name, {})

            validate_param_overrides(model_class, overrides)
            params = {**default_params, **overrides}

            print(f"\nTraining: {model_name}")

            cv_metrics, train_time = self._cross_validate_model(
                model_name=model_name,
                model_class=model_class,
                params=params,
                X=X,
                y=y,
                preprocessor=preprocessor,
                splitter=splitter,
            )

            primary_name, primary_value = extract_primary_metric(self.task_, cv_metrics)

            leaderboard_rows.append({
                "model_name": model_name,
                "primary_metric_name": primary_name,
                "primary_metric_value": primary_value,
                "roc_auc": cv_metrics.get("roc_auc"),
                "f1": cv_metrics.get("f1"),
                "precision": cv_metrics.get("precision"),
                "recall": cv_metrics.get("recall"),
                "accuracy": cv_metrics.get("accuracy"),
                "log_loss": cv_metrics.get("log_loss"),
                "rmse": cv_metrics.get("rmse"),
                "mae": cv_metrics.get("mae"),
                "r2": cv_metrics.get("r2"),
                "training_time_sec": train_time,
            })

            # Fit final model on full data
            final_model = self._fit_full_model(
                model_name=model_name,
                model_class=model_class,
                params=params,
                X=X,
                y=y,
                preprocessor=preprocessor,
            )

            self.models_[model_name] = final_model
            self.cv_results_[model_name] = {
                "metrics": cv_metrics,
                "training_time_sec": train_time,
                "params": params,
            }

        # Build leaderboard
        leaderboard = pd.DataFrame(leaderboard_rows)
        leaderboard.sort_values(
            by="primary_metric_value",
            ascending=(self.task_ == "regression"),
            inplace=True,
        )
        leaderboard.reset_index(drop=True, inplace=True)
        self.leaderboard_ = leaderboard

        print("\nTraining complete.")
        return self

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================
    @staticmethod
    def _ensure_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _infer_task(y) -> str:
        unique = np.unique(y)
        if y.dtype.kind in ("i", "b") and len(unique) <= 20:
            return "classification"
        return "regression"

    def _get_cv_splitter(self, y, cv: int):
        if self.task_ == "classification":
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            return KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

    def _iter_models_for_task(self, task: str):
        for name, entry in self.model_registry.items():
            if entry.get("task") == task:
                yield name, entry

    def _is_catboost_model(self, model_name: str) -> bool:
        return "catboost" in model_name

    def _get_cat_features_indices(self, X: pd.DataFrame) -> List[int]:
        cat_cols = X.select_dtypes(exclude=["number"]).columns
        return [X.columns.get_loc(c) for c in cat_cols]

    def _cross_validate_model(
        self,
        model_name: str,
        model_class,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
        preprocessor,
        splitter,
    ) -> Tuple[Dict[str, float], float]:

        fold_metrics: List[Dict[str, float]] = []
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            if self._is_catboost_model(model_name):
                X_train_cb = self._prepare_catboost_data(X_train_raw)
                X_val_cb = self._prepare_catboost_data(X_val_raw)
                cat_features = self._get_cat_features_indices(X_train_cb)
                model = model_class(**params)
                model.fit(
                    X_train_cb,
                    y_train,
                    cat_features=cat_features if cat_features else None,
                )
                if self.task_ == "classification":
                    y_proba = model.predict_proba(X_val_cb)
                    y_proba_for_metric = (
                        y_proba if len(np.unique(y)) > 2 else y_proba[:, 1]
                    )
                    y_pred = model.predict(X_val_cb)
                    metrics = compute_classification_metrics(y_val, y_proba_for_metric, y_pred)
                else:
                    y_pred = model.predict(X_val_cb)
                    metrics = compute_regression_metrics(y_val, y_pred)

            else:
                # Clone preprocessor for each fold to prevent leakage
                import copy
                fold_preprocessor = copy.deepcopy(preprocessor)
                X_train_trans = fold_preprocessor.fit_transform(X_train_raw, y_train)
                X_val_trans = fold_preprocessor.transform(X_val_raw)

                model = model_class(**params)
                model.fit(X_train_trans, y_train)

                if self.task_ == "classification":
                    y_proba = model.predict_proba(X_val_trans)
                    y_proba_for_metric = (
                        y_proba if len(np.unique(y)) > 2 else y_proba[:, 1]
                    )
                    y_pred = model.predict(X_val_trans)
                    metrics = compute_classification_metrics(y_val, y_proba_for_metric, y_pred)
                else:
                    y_pred = model.predict(X_val_trans)
                    metrics = compute_regression_metrics(y_val, y_pred)

            fold_metrics.append(metrics)

        train_time = time.time() - start_time

        all_keys = fold_metrics[0].keys()
        avg_metrics = {
            k: float(np.nanmean([m[k] for m in fold_metrics])) for k in all_keys
        }

        return avg_metrics, train_time

    def _fit_full_model(
        self,
        model_name: str,
        model_class,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
        preprocessor,
    ):
        if self._is_catboost_model(model_name):
            X_cb = self._prepare_catboost_data(X)
            cat_features = self._get_cat_features_indices(X_cb)
            model = model_class(**params)
            model.fit(
                X_cb,
                y,
                    cat_features=cat_features if cat_features else None,
            )
            return model
        else:
            import copy
            final_preprocessor = copy.deepcopy(preprocessor)
            X_trans = final_preprocessor.fit_transform(X, y)
            model = model_class(**params)
            model.fit(X_trans, y)
            return model
        
    def _prepare_catboost_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        CatBoost requires categorical NaN values to be strings.
        Convert all non-numeric column NaNs to 'missing'.
        """
        X = X.copy()
        cat_cols = X.select_dtypes(exclude=["number"]).columns
        for col in cat_cols:
            X[col] = X[col].fillna("missing").astype(str)
        return X