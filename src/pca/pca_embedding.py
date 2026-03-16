"""
PCA — Behavior Embedding Module
================================
Loads the StandardScaler-transformed data, applies PCA to reduce
the high-dimensional feature space into a compact behavioral
embedding, generates rich visualisations, and persists everything
to disk.

Why PCA?
    • Removes correlated / redundant features.
    • Projects data into a lower-dimensional space while retaining
      the maximum variance — perfect for downstream clustering and
      anomaly detection.
    • Makes 2-D / 3-D scatter-plot visualisations possible.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCALED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed",
                                "online_retail_scaled.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")


def _ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load scaled data
# ---------------------------------------------------------------------------
def load_scaled_data(path: str = SCALED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] Scaled data shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Fit full PCA (all components) – for analysis
# ---------------------------------------------------------------------------
def fit_full_pca(df: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA with ALL components so we can inspect explained variance
    and decide how many to keep.
    """
    pca_full = PCA()
    transformed = pca_full.fit_transform(df)
    print(f"[pca] Full PCA fitted – {pca_full.n_components_} components")
    return transformed, pca_full


# ---------------------------------------------------------------------------
# 3. Choose n_components & re-fit
# ---------------------------------------------------------------------------
def fit_reduced_pca(df: pd.DataFrame,
                    n_components: int | None = None,
                    variance_threshold: float = 0.95) -> tuple[pd.DataFrame, PCA]:
    """
    If *n_components* is given, use it directly.
    Otherwise pick the smallest n that captures ≥ *variance_threshold*
    of total variance.

    Returns
    -------
    df_pca : DataFrame with columns PC1 … PCn
    pca    : fitted PCA object
    """
    # first, full fit to decide n
    if n_components is None:
        pca_tmp = PCA()
        pca_tmp.fit(df)
        cumvar = np.cumsum(pca_tmp.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, df.shape[1])
        print(f"[pca] Auto-selected n_components = {n_components} "
              f"(captures {cumvar[n_components-1]*100:.2f}% variance, "
              f"threshold={variance_threshold*100:.0f}%)")

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(df)

    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(transformed, columns=cols, index=df.index)

    print(f"[pca] Reduced PCA shape: {df_pca.shape}")
    print(f"      Total variance retained: "
          f"{sum(pca.explained_variance_ratio_)*100:.2f}%")

    return df_pca, pca


# ---------------------------------------------------------------------------
# 4. Visualisations
# ---------------------------------------------------------------------------
def plot_explained_variance(pca_full: PCA,
                            save_dir: str = ARTIFACTS_DIR) -> None:
    """Scree plot + cumulative explained-variance curve."""
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n = len(evr)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # bar – individual
    ax1.bar(range(1, n + 1), evr, alpha=0.55, label="Individual", color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(range(1, n + 1))

    # line – cumulative
    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), cumvar, "ro-", label="Cumulative")
    ax2.axhline(y=0.95, color="gray", linestyle="--", linewidth=0.8, label="95 % threshold")
    ax2.set_ylabel("Cumulative Variance")

    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.title("PCA — Explained Variance", fontsize=14, pad=30)
    plt.tight_layout()
    out = os.path.join(save_dir, "pca_explained_variance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_2d_scatter(df_pca: pd.DataFrame,
                    pca: PCA,
                    save_dir: str = ARTIFACTS_DIR) -> None:
    """Scatter plot on the first two principal components."""
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca["PC1"], df_pca["PC2"],
                alpha=0.15, s=5, c="steelblue", edgecolors="none")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    plt.title("PCA — 2-D Behavior Embedding", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, "pca_2d_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_3d_scatter(df_pca: pd.DataFrame,
                    pca: PCA,
                    save_dir: str = ARTIFACTS_DIR) -> None:
    """3-D scatter on the first three principal components."""
    if df_pca.shape[1] < 3:
        print("[plot] Skipping 3-D scatter — fewer than 3 components.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"],
               alpha=0.15, s=5, c="steelblue", edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.set_title("PCA — 3-D Behavior Embedding", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, "pca_3d_scatter.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_loadings_heatmap(pca: PCA,
                          feature_names: list[str],
                          save_dir: str = ARTIFACTS_DIR) -> None:
    """Heat-map showing how each original feature loads onto each PC."""
    n = pca.n_components_
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n)],
        index=feature_names,
    )
    plt.figure(figsize=(max(8, n * 1.2), max(6, len(feature_names) * 0.5)))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5)
    plt.title("PCA Component Loadings", fontsize=14)
    plt.ylabel("Original Feature")
    plt.xlabel("Principal Component")
    plt.tight_layout()
    out = os.path.join(save_dir, "pca_loadings_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
def save_pca_data(df_pca: pd.DataFrame,
                  out_dir: str = PROCESSED_DIR,
                  filename: str = "online_retail_pca.csv") -> str:
    out = os.path.join(out_dir, filename)
    df_pca.to_csv(out, index=False)
    print(f"[save] PCA data → {out}")
    return out


def save_pca_model(pca: PCA,
                   out_dir: str = MODELS_DIR,
                   filename: str = "pca_model.pkl") -> str:
    out = os.path.join(out_dir, filename)
    joblib.dump(pca, out)
    print(f"[save] PCA model → {out}")
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pca_pipeline(df_scaled: pd.DataFrame | None = None,
                     n_components: int | None = None,
                     variance_threshold: float = 0.95) -> pd.DataFrame:
    """
    Execute the full PCA pipeline.

    Parameters
    ----------
    df_scaled : DataFrame, optional
        Scaled feature matrix.  Loaded from disk if not provided.
    n_components : int, optional
        Force a specific number of components.
        If None, auto-select to capture *variance_threshold* of variance.
    variance_threshold : float
        Minimum cumulative variance to keep (default 95 %).

    Returns
    -------
    df_pca : DataFrame with PC1 … PCn columns.
    """
    _ensure_dirs()

    # 1. Load
    if df_scaled is None:
        df_scaled = load_scaled_data()

    feature_names = df_scaled.columns.tolist()

    # 2. Full PCA (for analysis)
    _, pca_full = fit_full_pca(df_scaled)

    # 3. Reduced PCA
    df_pca, pca = fit_reduced_pca(df_scaled,
                                  n_components=n_components,
                                  variance_threshold=variance_threshold)

    # 4. Visualisations
    plot_explained_variance(pca_full)
    plot_2d_scatter(df_pca, pca)
    plot_3d_scatter(df_pca, pca)
    plot_loadings_heatmap(pca, feature_names)

    # 5. Persist
    save_pca_data(df_pca)
    save_pca_model(pca)

    return df_pca


if __name__ == "__main__":
    run_pca_pipeline()
