import pandas as pd
import numpy as np
import hashlib
from typing import Optional, Dict
from .type_detection import detect_feature_types


class FeatureEngineeringPipeline:
    """
    A unified, dataset-agnostic feature engineering pipeline.
    Handles:
        - type detection
        - imputers
        - encoders
        - scalers
        - datetime feature extraction
        - feature selection (optional)
    """

    # =========================================================
    # INITIALISATION
    # =========================================================
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_types = None

        self.imputers = {}
        self.encoders = {}
        self.scalers = {}
        self.datetime_features = {}
        self.selected_features = None
        self.datetime_reference = {}

        self.fitted = False

    # =========================================================
    # PRIVATE FIT HELPERS
    # =========================================================
    def _fit_imputers(self, df: pd.DataFrame):
        imputers = {}

        for col in self.feature_types["numeric"]:
            imputers[col] = ("median", df[col].median())

        for col in self.feature_types["categorical"]:
            if df[col].dropna().empty:
                imputers[col] = ("mode", "missing")
            else:
                imputers[col] = ("mode", df[col].mode(dropna=True).iloc[0])

        for col in self.feature_types["high_cardinality"]:
            imputers[col] = ("missing_token", "missing")

        for col in self.feature_types["binary"]:
            imputers[col] = ("median", df[col].median())

        self.imputers = imputers

    def _fit_encoders(self, df: pd.DataFrame):
        encoders = {}

        for col in self.feature_types["categorical"]:
            categories = df[col].dropna().unique().tolist()
            encoders[col] = ("onehot", categories)

        for col in self.feature_types["high_cardinality"]:
            n_components = self.config.get("hashing_dim", 32)
            encoders[col] = ("hashing", n_components)

        self.encoders = encoders

    def _fit_scalers(self, df: pd.DataFrame):
        scalers = {}
        method = self.config.get("scaler", "standard")

        for col in self.feature_types["numeric"]:
            series = df[col]

            if method == "standard":
                mean = series.mean()
                std = series.std() if series.std() != 0 else 1.0
                scalers[col] = ("standard", mean, std)

            elif method == "minmax":
                min_val = series.min()
                max_val = series.max()
                range_val = max_val - min_val if max_val != min_val else 1.0
                scalers[col] = ("minmax", min_val, range_val)

            elif method == "robust":
                median = series.median()
                iqr = series.quantile(0.75) - series.quantile(0.25)
                iqr = iqr if iqr != 0 else 1.0
                scalers[col] = ("robust", median, iqr)

        self.scalers = scalers

    def _fit_datetime_features(self, df: pd.DataFrame):
        dt_features = {}

        for col in self.feature_types["datetime"]:
            dt_features[col] = [
                "year", "month", "day", "dayofweek", "is_weekend"
            ]

            if hasattr(df[col].dt, "hour"):
                dt_features[col].append("hour")
            if hasattr(df[col].dt, "minute"):
                dt_features[col].append("minute")
            if hasattr(df[col].dt, "second"):
                dt_features[col].append("second")

        self.datetime_features = dt_features

    def _fit_feature_selection(self, df: pd.DataFrame, target=None):
        fs_config = self.config.get("feature_selection", None)
        if fs_config is None:
            self.selected_features = list(df.columns)
            return

        variance_threshold = fs_config.get("variance_threshold", 0.0)
        corr_threshold = fs_config.get("correlation_threshold", None)
        protected = set(fs_config.get("protected_features", []))

        selected = pd.Index(df.columns)

        if variance_threshold and variance_threshold > 0.0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            variances = df[numeric_cols].var()
            keep_numeric = variances[variances > variance_threshold].index
            selected = selected.intersection(
                keep_numeric.union(df.columns.difference(numeric_cols))
            )

        if corr_threshold is not None:
            numeric_cols = df[selected].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

                to_drop = set()
                for col in upper.columns:
                    for row in upper.index:
                        if pd.isna(upper.loc[row, col]):
                            continue
                        if upper.loc[row, col] > corr_threshold:
                            cand_keep, cand_drop = row, col
                            if cand_keep in protected:
                                pass
                            elif cand_drop in protected:
                                cand_keep, cand_drop = cand_drop, cand_keep
                            to_drop.add(cand_drop)

                selected = selected.difference(to_drop)

        self.selected_features = list(selected)

    # =========================================================
    # PRIVATE APPLY HELPERS
    # =========================================================
    def _apply_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (_, value) in self.imputers.items():
            df[col] = df[col].fillna(value)
        return df

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, (enc_type, params) in self.encoders.items():

            if enc_type == "onehot":
                categories = params
                for cat in categories:
                    df[f"{col}__{cat}"] = (df[col] == cat).astype(int)
                df = df.drop(columns=[col])

            elif enc_type == "hashing":
                n_components = params
                hashed = np.zeros((len(df), n_components))

                for i, val in enumerate(df[col].astype(str)):
                    h = int(hashlib.md5(val.encode()).hexdigest(), 16)
                    bucket = h % n_components
                    hashed[i, bucket] += 1

                for j in range(n_components):
                    df[f"{col}__hash_{j}"] = hashed[:, j]

                df = df.drop(columns=[col])

        return df

    def _apply_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, params in self.scalers.items():
            method = params[0]

            if method == "standard":
                _, mean, std = params
                df[col] = (df[col] - mean) / std

            elif method == "minmax":
                _, min_val, range_val = params
                df[col] = (df[col] - min_val) / range_val

            elif method == "robust":
                _, median, iqr = params
                df[col] = (df[col] - median) / iqr

        return df

    def _apply_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, feats in self.datetime_features.items():
            df[f"{col}__missing"] = df[col].isna().astype(int)
            df[col] = pd.to_datetime(df[col], errors="coerce")

            if "year" in feats:
                df[f"{col}__year"] = df[col].dt.year
            if "month" in feats:
                df[f"{col}__month"] = df[col].dt.month
            if "day" in feats:
                df[f"{col}__day"] = df[col].dt.day
            if "dayofweek" in feats:
                df[f"{col}__dayofweek"] = df[col].dt.dayofweek
            if "is_weekend" in feats:
                df[f"{col}__is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            if "hour" in feats:
                df[f"{col}__hour"] = df[col].dt.hour
            if "minute" in feats:
                df[f"{col}__minute"] = df[col].dt.minute
            if "second" in feats:
                df[f"{col}__second"] = df[col].dt.second

            ref = self.datetime_reference.get(col, None)
            if ref is not None:
                df[f"{col}__days_since"] = (df[col] - ref).dt.days

            df = df.drop(columns=[col])

        return df

    def _apply_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            return df

        df = df.copy()

        for col in self.selected_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.selected_features]

    # =========================================================
    # INTERNAL TRANSFORM (NO FEATURE SELECTION)
    # =========================================================
    def _transform_no_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._apply_imputers(df)
        df = self._parse_datetime(df)
        df = self._apply_datetime_features(df)
        df = self._apply_encoders(df)
        df = self._apply_scalers(df)
        return df

    # =========================================================
    # PRIVATE DATETIME PARSER
    # =========================================================
    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_types["datetime"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    # =========================================================
    # PUBLIC FIT
    # =========================================================
    def fit(self, df: pd.DataFrame, target=None):
        df = df.copy()

        self.feature_types = detect_feature_types(df, config=self.config)
        df = self._parse_datetime(df)

        self.datetime_reference = {}
        for col in self.feature_types["datetime"]:
            parsed = pd.to_datetime(df[col], errors="coerce")
            self.datetime_reference[col] = parsed.min()

        self._fit_imputers(df)
        self._fit_datetime_features(df)
        self._fit_encoders(df)
        self._fit_scalers(df)

        full = self._transform_no_feature_selection(df)
        self._fit_feature_selection(full, target)

        self.fitted = True
        return self

    # =========================================================
    # PUBLIC TRANSFORM
    # =========================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        df = self._transform_no_feature_selection(df)
        df = self._apply_feature_selection(df)
        return df

    # =========================================================
    # PUBLIC FIT + TRANSFORM
    # =========================================================
    def fit_transform(self, df: pd.DataFrame, target=None) -> pd.DataFrame:
        self.fit(df, target=target)
        return self.transform(df)