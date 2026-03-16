"""
Data Scaling Module
====================
Loads the fully-processed DataFrame, selects the numeric features
relevant for clustering / PCA / anomaly-detection, applies
StandardScaler, generates before-vs-after visualisations,
persists the scaled data and the fitted scaler to disk.

Why StandardScaler?
    Clustering algorithms (K-Means, DBSCAN, …) and PCA rely on
    distance metrics.  Features on vastly different scales would
    dominate those distances, so we normalise every feature to
    zero mean and unit variance.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed",
                                   "online_retail_processed.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")

# Numeric columns that are meaningful for clustering.
# We deliberately exclude identifier / date-string columns.
FEATURES_FOR_SCALING = [
    "Quantity",
    "UnitPrice",
    "TotalPrice",
    "Month",
    "Day",
    "Hour",
    "is_weekend",
    "is_night",
    "avg_purchase_value",
    "purchase_frequency",
    "unique_products_bought",
    "avg_time_between_purchases",
    "weekend_purchase_ratio",
    "night_purchase_ratio",
    "country_diversity",
]


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Read the feature-engineered CSV."""
    df = pd.read_csv(path)
    print(f"[load] Processed data shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Select & prepare features
# ---------------------------------------------------------------------------
def select_features(df: pd.DataFrame,
                    features: list[str] | None = None) -> pd.DataFrame:
    """
    Extract only the numeric feature columns needed for scaling.
    Any remaining NaNs are filled with the column median so that
    StandardScaler doesn't break.
    """
    if features is None:
        features = FEATURES_FOR_SCALING

    df_features = df[features].copy()

    nan_counts = df_features.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        print(f"[select] Filling NaN with median in: "
              f"{cols_with_nan.index.tolist()}")
        df_features.fillna(df_features.median(), inplace=True)

    print(f"[select] Feature matrix shape: {df_features.shape}")
    return df_features


# ---------------------------------------------------------------------------
# 3. Scale
# ---------------------------------------------------------------------------
def scale_features(df_features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to *df_features*.

    Returns
    -------
    df_scaled : DataFrame  – same columns, scaled values
    scaler    : fitted StandardScaler (for inverse-transform later)
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_array,
                             columns=df_features.columns,
                             index=df_features.index)
    print("[scale] StandardScaler applied ✓")
    print(f"        Means  (should be ≈ 0): "
          f"{np.round(df_scaled.mean().values, 6).tolist()}")
    print(f"        StdDev (should be ≈ 1): "
          f"{np.round(df_scaled.std().values, 6).tolist()}")
    return df_scaled, scaler


# ---------------------------------------------------------------------------
# 4. Visualisations
# ---------------------------------------------------------------------------
def plot_feature_distributions_before_after(
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        save_dir: str = ARTIFACTS_DIR) -> None:
    """Side-by-side box-plots: raw vs scaled feature distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Before
    axes[0].set_title("Feature Distributions — Before Scaling", fontsize=13)
    sns.boxplot(data=df_before, orient="h", ax=axes[0])
    axes[0].set_xlabel("Value")

    # After
    axes[1].set_title("Feature Distributions — After Scaling", fontsize=13)
    sns.boxplot(data=df_after, orient="h", ax=axes[1])
    axes[1].set_xlabel("Value (standardised)")

    plt.tight_layout()
    out = os.path.join(save_dir, "scaling_before_after_boxplot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_scaled_histograms(df_scaled: pd.DataFrame,
                           save_dir: str = ARTIFACTS_DIR) -> None:
    """Histogram grid of every scaled feature."""
    n_cols = 3
    n_features = len(df_scaled.columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(df_scaled.columns):
        sns.histplot(df_scaled[col], bins=40, kde=True, ax=axes[idx])
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel("")

    # hide unused sub-plots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Scaled Feature Distributions", fontsize=15, y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, "scaled_feature_histograms.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_correlation_heatmap(df_scaled: pd.DataFrame,
                             save_dir: str = ARTIFACTS_DIR) -> None:
    """Correlation heat-map of the scaled features."""
    plt.figure(figsize=(12, 10))
    corr = df_scaled.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap (Scaled Data)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, "scaled_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
def save_scaled_data(df_scaled: pd.DataFrame,
                     out_dir: str = PROCESSED_DIR,
                     filename: str = "online_retail_scaled.csv") -> str:
    """Save the scaled feature DataFrame to CSV."""
    out = os.path.join(out_dir, filename)
    df_scaled.to_csv(out, index=False)
    print(f"[save] Scaled data → {out}")
    return out


def save_scaler(scaler: StandardScaler,
                out_dir: str = MODELS_DIR,
                filename: str = "standard_scaler.pkl") -> str:
    """Persist the fitted StandardScaler for later inverse-transforms."""
    out = os.path.join(out_dir, filename)
    joblib.dump(scaler, out)
    print(f"[save] Scaler → {out}")
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_scaling_pipeline(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Execute the full data-scaling pipeline.

    Parameters
    ----------
    df : DataFrame, optional
        The feature-engineered DataFrame.  If not supplied, the
        processed CSV is loaded from disk.

    Returns
    -------
    df_scaled : DataFrame with StandardScaler-transformed features.
    """
    _ensure_dirs()

    # 1. Load
    if df is None:
        df = load_processed_data()

    # 2. Select numeric features
    df_features = select_features(df)

    # 3. Scale
    df_scaled, scaler = scale_features(df_features)

    # 4. Visualisations
    plot_feature_distributions_before_after(df_features, df_scaled)
    plot_scaled_histograms(df_scaled)
    plot_correlation_heatmap(df_scaled)

    # 5. Save
    save_scaled_data(df_scaled)
    save_scaler(scaler)

    return df_scaled


if __name__ == "__main__":
    run_scaling_pipeline()
