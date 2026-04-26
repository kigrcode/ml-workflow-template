# ================================================================
# SECTION 1 — Imports & Global Utilities
# ================================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ------------------------------------------------
# Safe sampling for large datasets
# ------------------------------------------------
def _safe_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Return a sampled dataframe if too large.
    Ensures plotting functions remain performant.
    """
    if len(df) > sample_size:
        return df.sample(sample_size, random_state=42)
    return df


# ------------------------------------------------
# Robust feature-type detection
# ------------------------------------------------
def _detect_feature_types(df: pd.DataFrame, max_categories: int = 20):
    """
    Detect numeric and categorical features in a generic, dataset-agnostic way.
    - Numeric = int/float
    - Categorical = object/category/bool OR numeric with low cardinality
    """
    numeric_features = []
    categorical_features = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            # Numeric but low cardinality → treat as categorical
            if series.nunique(dropna=True) <= max_categories:
                categorical_features.append(col)
            else:
                numeric_features.append(col)

        elif (
            series.dtype == "object"
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical_features.append(col)

    return numeric_features, categorical_features


# ------------------------------------------------
# Universal categorical cleaner
# ------------------------------------------------
def _clean_categorical_features(
    df: pd.DataFrame,
    categorical_features: list,
    max_unique: int = 50,
    max_missing_ratio: float = 0.5,
    min_freq_ratio: float = 0.01,
):
    """
    Clean categorical features to ensure stable association metrics.
    Removes:
    - high-cardinality features
    - features with too many missing values
    - features dominated by a single category
    """
    cleaned = []

    for c in categorical_features:
        nunique = df[c].nunique(dropna=True)
        missing_ratio = df[c].isna().mean()
        top_freq = df[c].value_counts(normalize=True, dropna=True).iloc[0]

        if nunique > max_unique:
            continue
        if missing_ratio > max_missing_ratio:
            continue
        if top_freq > (1 - min_freq_ratio):
            continue

        cleaned.append(c)

    return cleaned

# ================================================================
# SECTION 2 — Univariate EDA
# ================================================================

def plot_numeric_distribution(df: pd.DataFrame, column: str):
    """
    Plot histogram + KDE for a numeric column.
    Works for any dataset with numeric features.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 20,
    top_n: int = 20
):
    """
    Plot distribution for a categorical column.
    If cardinality is too high, show a summary instead of a bar plot.
    This prevents unreadable plots and keeps the function generic.
    """
    n_unique = df[column].nunique(dropna=True)

    if n_unique > max_categories:
        print(f"[SKIPPED] '{column}' has {n_unique} unique values (>{max_categories}).")
        print("Top categories:")
        display(df[column].value_counts().head(top_n))

        rare_pct = (
            df[column].value_counts(normalize=True)
            .tail(-top_n)
            .sum() * 100
        )
        print(f"Rare categories account for {rare_pct:.2f}% of rows.")
        return

    plt.figure(figsize=(8, 5))
    df[column].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_missingness(df: pd.DataFrame):
    """
    Plot missing values per column.
    Works for any dataset with missing values.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values detected.")
        return

    plt.figure(figsize=(10, 5))
    missing.sort_values().plot(kind="bar")
    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_zeroness(df: pd.DataFrame, threshold: float = 0.5):
    """
    Plot the percentage of zeros in each numeric column.
    Useful for sparse datasets (e.g., financial, sensor, or log data).
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    zero_stats = (
        (df[numeric_cols] == 0)
        .sum()
        .sort_values(ascending=False)
        .to_frame(name="zero_count")
    )
    zero_stats["zero_pct"] = zero_stats["zero_count"] / len(df)
    zero_stats["zero_heavy"] = zero_stats["zero_pct"] > threshold

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=zero_stats.reset_index(),
        x="index",
        y="zero_pct",
        hue="zero_heavy",
        dodge=False
    )
    plt.xticks(rotation=45)
    plt.ylabel("Percentage of Zeros")
    plt.title("Zeroness Overview")
    plt.tight_layout()
    plt.show()

    return zero_stats


def plot_numeric_target_distribution(df: pd.DataFrame, target: str):
    """
    Plot distribution for a numerical target variable.
    Includes histogram, KDE, and boxplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df[target], kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {target}")

    sns.boxplot(x=df[target], ax=axes[1])
    axes[1].set_title(f"Boxplot of {target}")

    plt.tight_layout()
    plt.show()

    display(df[target].describe())


def detect_problem_type(df: pd.DataFrame, target: str, max_classes: int = 20):
    """
    Determine whether the target represents a classification or regression task.
    Generic logic:
    - object/category/bool → classification
    - numeric with few unique values → classification
    - otherwise → regression
    """
    dtype = df[target].dtype
    n_unique = df[target].nunique()

    if dtype == "object" or dtype.name == "category" or dtype == "bool":
        return "classification"

    if np.issubdtype(dtype, np.number) and n_unique <= max_classes:
        return "classification"

    return "regression"

# ================================================================
# SECTION 3 — Pairwise Feature–Target EDA
# ================================================================

def _is_binary(series):
    unique_vals = series.dropna().unique()
    return len(unique_vals) == 2


def _plot_numeric_vs_numeric(df, feature, target, save_path):
    """Scatter + regression line for numeric → numeric."""
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=feature, y=target, scatter_kws={"alpha": 0.5})
    plt.title(f"{feature} vs {target} (Numeric → Numeric)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def _plot_categorical_association_heatmap(matrix, save_path):
    """
    Plots a heatmap for categorical–categorical association matrices
    (e.g., Cramér's V or Theil's U).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Categorical Association Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def _plot_categorical_vs_numeric(df, feature, target, save_path):
    """
    Adaptive plot for categorical → numeric:
    - If target is binary → barplot of mean target
    - If target is continuous → boxplot
    """
    plt.figure(figsize=(8, 5))

    # Detect binary target
    unique_vals = df[target].dropna().unique()
    is_binary_target = len(unique_vals) == 2

    if is_binary_target:
        # Barplot of mean target (proportions)
        sns.barplot(data=df, x=feature, y=target, estimator=np.mean)
        plt.ylabel(f"Mean {target}")
        plt.title(f"{feature} vs {target} (Categorical → Binary)")
    else:
        # Standard boxplot for continuous target
        sns.boxplot(data=df, x=feature, y=target)
        plt.title(f"{feature} vs {target} (Categorical → Numeric)")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def _plot_numeric_vs_categorical(df, feature, target, save_path):
    """Boxplot for numeric → categorical."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=target, y=feature)
    plt.title(f"{feature} vs {target} (Numeric → Categorical)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def _plot_categorical_vs_categorical(df, feature, target, save_path):
    """Stacked bar chart for categorical → categorical."""
    plt.figure(figsize=(8, 5))
    ctab = pd.crosstab(df[feature], df[target], normalize="index")
    ctab.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="viridis")
    plt.title(f"{feature} vs {target} (Categorical → Categorical)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def analyze_feature_target_relationships(
    df: pd.DataFrame,
    target: str,
    problem_type: str = None,
    max_categories: int = 20,
    sample_size: int = 5000,
    save_dir: str = "reports/figures/eda",
):
    """
    Analyze each feature's relationship with the target using the appropriate plot.
    Automatically detects feature types and handles high-cardinality categories.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Detect target type
    # ---------------------------------------------------------
    if problem_type is None:
        problem_type = detect_problem_type(df, target)

    print(f"\n🔍 Pairwise Feature–Target Analysis ({problem_type})")
    print("Saving plots to:", save_dir)

    # Coerce binary object targets to numeric if possible
    if df[target].dtype == "object":
        try:
            df[target] = df[target].astype(float)
        except:
            pass

    is_target_numeric = pd.api.types.is_numeric_dtype(df[target])
    is_target_categorical = not is_target_numeric or df[target].nunique() <= max_categories

    # ---------------------------------------------------------
    # Detect feature types
    # ---------------------------------------------------------
    numeric_features, categorical_features = _detect_feature_types(df, max_categories)

    # ---------------------------------------------------------
    # Loop through features
    # ---------------------------------------------------------
    for feature in df.columns:
        if feature == target:
            continue

        series = df[feature]

        # Skip unusable features
        if series.nunique(dropna=True) <= 1:
            continue
        if series.isna().all():
            continue

        # Determine feature type
        is_feature_numeric = feature in numeric_features
        is_feature_categorical = feature in categorical_features

        # ---------------------------------------------------------
        # Sample only for numeric–numeric plots
        # ---------------------------------------------------------
        if is_feature_numeric and is_target_numeric:
            df_plot = _safe_sample(df[[feature, target]].dropna(), sample_size)
        else:
            df_plot = df[[feature, target]].dropna()

        # Save path
        save_path = os.path.join(save_dir, f"{feature}_vs_{target}.png")

        # ---------------------------------------------------------
        # Routing logic
        # ---------------------------------------------------------

        # Numeric → Numeric
        if is_feature_numeric and is_target_numeric:
            _plot_numeric_vs_numeric(df_plot, feature, target, save_path)

        # Categorical → Numeric
        elif is_feature_categorical and is_target_numeric:
            if df_plot[feature].nunique() > max_categories:
                print(f"Skipping {feature}: too many categories.")
                continue
            _plot_categorical_vs_numeric(df_plot, feature, target, save_path)

        # Numeric → Categorical
        elif is_feature_numeric and is_target_categorical:
            _plot_numeric_vs_categorical(df_plot, feature, target, save_path)

        # Categorical → Categorical
        elif is_feature_categorical and is_target_categorical:
            if df_plot[feature].nunique() > max_categories:
                print(f"Skipping {feature}: too many categories.")
                continue
            _plot_categorical_vs_categorical(df_plot, feature, target, save_path)

        else:
            print(f"Skipping {feature}: unsupported type combination.")



# ================================================================
# SECTION 4 — Feature–Feature Association Metrics
# ================================================================

# ------------------------------------------------
# Numeric ↔ Numeric
# ------------------------------------------------
def _compute_numeric_correlations(df, method="pearson"):
    """Compute numeric correlation matrix."""
    return df.corr(method=method)


def _plot_correlation_heatmap(corr, save_path):
    """Standard correlation heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def _plot_clustered_correlation_heatmap(corr, save_path):
    """Clustered correlation heatmap."""
    sns.clustermap(corr, cmap="coolwarm", figsize=(12, 12))
    plt.title("Clustered Correlation Heatmap")
    plt.savefig(save_path)
    plt.show()


def _plot_scatter_matrix(df, sample_size, save_path):
    """Scatter matrix for numeric features."""
    df_sample = df.sample(min(len(df), sample_size), random_state=42)
    sns.pairplot(df_sample)
    plt.savefig(save_path)
    plt.show()


# ------------------------------------------------
# Categorical ↔ Categorical
# ------------------------------------------------
def _compute_cramers_v(x, y):
    """
    Compute Cramér's V for two categorical variables.
    Safe against:
    - sparse tables
    - zero rows/columns
    - division-by-zero
    """
    confusion_matrix = pd.crosstab(x, y)

    if confusion_matrix.size == 0:
        return np.nan

    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()

    if n == 0:
        return np.nan

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Bias correction
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    rcorr = r - ((r - 1)**2)/(n - 1)
    kcorr = k - ((k - 1)**2)/(n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan

    return np.sqrt(phi2corr / denom)


def _compute_cramers_v_matrix(df, categorical_features):
    """Compute full Cramér's V matrix."""
    matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)

    for f1 in categorical_features:
        for f2 in categorical_features:
            matrix.loc[f1, f2] = _compute_cramers_v(df[f1], df[f2])

    return matrix.astype(float)


def _conditional_entropy(x, y):
    """Helper for Theil's U."""
    y_counts = y.value_counts(normalize=True)
    entropy = 0.0

    for y_val, p_y in y_counts.items():
        x_subset = x[y == y_val]
        x_counts = x_subset.value_counts(normalize=True)
        entropy += p_y * (-np.sum(x_counts * np.log2(x_counts + 1e-9)))

    return entropy


def _theils_u(x, y):
    """U(X|Y): how much knowing Y reduces uncertainty in X."""
    s_xy = _conditional_entropy(x, y)
    x_counts = x.value_counts(normalize=True)
    s_x = -np.sum(x_counts * np.log2(x_counts + 1e-9))

    if s_x == 0:
        return np.nan

    return (s_x - s_xy) / s_x


def _compute_theils_u_matrix(df, categorical_features):
    """Compute full Theil's U matrix."""
    matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)

    for f1 in categorical_features:
        for f2 in categorical_features:
            matrix.loc[f1, f2] = _theils_u(df[f1], df[f2])

    return matrix.astype(float)


# ------------------------------------------------
# Numeric ↔ Categorical
# ------------------------------------------------
def _compute_anova_f_scores(df, numeric_features, categorical_feature):
    """
    Compute ANOVA F-scores for numeric features grouped by a categorical feature.
    """
    scores = {}

    for f in numeric_features:
        groups = [df[f][df[categorical_feature] == cat] for cat in df[categorical_feature].unique()]

        try:
            f_stat, _ = f_oneway(*groups)
            scores[f] = f_stat
        except:
            scores[f] = np.nan

    return pd.Series(scores).sort_values(ascending=False)


def _compute_mutual_information(df, numeric_features, categorical_features):
    """
    Compute MI for numeric → categorical.
    Returns a dict: {categorical_feature: Series of MI scores}
    """
    mi_results = {}

    for cat in categorical_features:
        try:
            encoded = df[cat].astype("category").cat.codes
            mi = mutual_info_classif(df[numeric_features], encoded)
            mi_results[cat] = pd.Series(mi, index=numeric_features)
        except:
            continue

    return mi_results


# ------------------------------------------------
# Redundancy Detection
# ------------------------------------------------
def _detect_redundant_features(corr, cramers_v, mi_results, anova_results, thresholds):
    """
    Identify redundant features across:
    - numeric ↔ numeric
    - categorical ↔ categorical
    - numeric ↔ categorical (MI + ANOVA)
    """
    redundant = []

    # Numeric redundancy
    for f1 in corr.columns:
        for f2 in corr.columns:
            if f1 < f2 and abs(corr.loc[f1, f2]) > thresholds["corr"]:
                redundant.append((f1, f2, "corr", corr.loc[f1, f2]))

    # Categorical redundancy
    if not cramers_v.empty:
        for f1 in cramers_v.columns:
            for f2 in cramers_v.columns:
                if f1 < f2 and cramers_v.loc[f1, f2] > thresholds["cramers_v"]:
                    redundant.append((f1, f2, "cramers_v", cramers_v.loc[f1, f2]))

    # Numeric–categorical redundancy (MI)
    for cat, mi_series in mi_results.items():
        for num, val in mi_series.items():
            if val > thresholds["mi"]:
                redundant.append((num, cat, "mutual_info", val))

    # Numeric–categorical redundancy (ANOVA)
    for cat, series in anova_results.items():
        for num, val in series.items():
            if val > thresholds["anova"]:
                redundant.append((num, cat, "anova_f", val))

    return pd.DataFrame(redundant, columns=["feature_1", "feature_2", "metric", "value"])   

# ================================================================
# SECTION 5 — Feature–Feature Orchestrator
# ================================================================

def analyze_feature_feature_relationships(
    df: pd.DataFrame,
    sample_size: int = 3000,
    corr_method: str = "pearson",
    save_dir: str = "reports/figures/eda",
    redundancy_thresholds: dict = None,
):
    """
    Analyze feature–feature relationships across:
    - numeric ↔ numeric
    - categorical ↔ categorical
    - numeric ↔ categorical (MI + ANOVA)
    Returns a dictionary of results.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Default thresholds
    if redundancy_thresholds is None:
        redundancy_thresholds = {
            "corr": 0.85,
            "cramers_v": 0.8,
            "mi": 0.5,
            "anova": 50.0,
        }

    # ---------------------------------------------------------
    # Detect feature types
    # ---------------------------------------------------------
    numeric_features, categorical_features = _detect_feature_types(df)

    # Clean categorical features for stable association metrics
    categorical_features = _clean_categorical_features(df, categorical_features)

    # Initialize results
    results = {}

    # ---------------------------------------------------------
    # Numeric ↔ Numeric
    # ---------------------------------------------------------
    if len(numeric_features) > 1:
        corr = _compute_numeric_correlations(df[numeric_features], method=corr_method)
        results["correlation_matrix"] = corr

        _plot_correlation_heatmap(corr, os.path.join(save_dir, "correlation_heatmap.png"))
        _plot_clustered_correlation_heatmap(corr, os.path.join(save_dir, "clustered_correlation_heatmap.png"))
        _plot_scatter_matrix(df[numeric_features], sample_size, os.path.join(save_dir, "scatter_matrix.png"))
    else:
        corr = pd.DataFrame()
        results["correlation_matrix"] = corr

    # ---------------------------------------------------------
    # Categorical ↔ Categorical
    # ---------------------------------------------------------
    if len(categorical_features) > 1:
        cramers_v = _compute_cramers_v_matrix(df, categorical_features)
        results["cramers_v"] = cramers_v

        _plot_categorical_association_heatmap(
            cramers_v, os.path.join(save_dir, "categorical_association_heatmap.png")
        )

        theils_u = _compute_theils_u_matrix(df, categorical_features)
        results["theils_u"] = theils_u
    else:
        cramers_v = pd.DataFrame()
        theils_u = pd.DataFrame()
        results["cramers_v"] = cramers_v
        results["theils_u"] = theils_u

    # ---------------------------------------------------------
    # Numeric ↔ Categorical (MI + ANOVA)
    # ---------------------------------------------------------
    if len(numeric_features) > 0 and len(categorical_features) > 0:
        mi_results = _compute_mutual_information(df, numeric_features, categorical_features)
        results["mutual_information"] = mi_results

        # ANOVA for each categorical feature
        anova_results = {}
        for cat in categorical_features:
            anova_results[cat] = _compute_anova_f_scores(df, numeric_features, cat)

        results["anova_f_scores"] = anova_results

    else:
        mi_results = {}
        anova_results = {}
        results["mutual_information"] = mi_results
        results["anova_f_scores"] = anova_results

    # ---------------------------------------------------------
    # Redundancy Detection
    # ---------------------------------------------------------
    redundant = _detect_redundant_features(
        corr,
        cramers_v,
        mi_results,
        anova_results,
        redundancy_thresholds
    )
    results["redundant_features"] = redundant

    return results






