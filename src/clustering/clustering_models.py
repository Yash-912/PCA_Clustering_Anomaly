"""
Clustering Module
=================
Loads the PCA behavioral embeddings and applies unsupervised clustering
algorithms to discover distinct transaction/customer segments.

Implemented algorithms:
    1. K-Means
    2. DBSCAN
    3. Hierarchical (Agglomerative) Clustering

Note on Scalability:
    The full dataset has ~400k rows. K-Means scales well to this size,
    but DBSCAN and Hierarchical compute pairwise distance matrices which
    require O(N^2) memory. For DBSCAN and Hierarchical clustering, we 
    process a representative random sample to avoid MemoryError.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCA_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "online_retail_pca.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")


def _ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_pca_data(path: str = PCA_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] PCA data shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Train Models
# ---------------------------------------------------------------------------
def train_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> tuple[KMeans, np.ndarray]:
    """Train K-Means on the full dataset."""
    print(f"[kmeans] Fitting K-Means with k={n_clusters} on {len(df)} rows...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(df)
    print("[kmeans] Training complete.")
    return kmeans, labels


def train_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 10) -> tuple[DBSCAN, np.ndarray, pd.DataFrame]:
    """
    Train DBSCAN. Since DBSCAN does not scale well to 400k rows,
    we randomly sample the data to avoid OOM crashes if the dataset is too large.
    """
    if len(df) > 20000:
        print(f"[dbscan] Dataset too large ({len(df)} rows) for standard DBSCAN.")
        print("[dbscan] Downsampling to 20,000 rows for DBSCAN training...")
        df_sample = df.sample(n=20000, random_state=42).copy()
    else:
        df_sample = df.copy()

    print(f"[dbscan] Fitting DBSCAN (eps={eps}, min_samples={min_samples}) on {len(df_sample)} rows...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_sample)
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f"[dbscan] Estimated clusters: {n_clusters_} | Noise points: {n_noise_}")
    
    return dbscan, labels, df_sample


def train_hierarchical(df: pd.DataFrame, n_clusters: int = 5) -> tuple[AgglomerativeClustering, np.ndarray, pd.DataFrame]:
    """
    Train Hierarchical Clustering. Requires O(N^2) memory.
    We aggressively downsample to 10,000 rows to prevent MemoryError.
    """
    if len(df) > 10000:
        print(f"[hierarchical] Dataset too large ({len(df)} rows) for O(N^2) memory.")
        print("[hierarchical] Downsampling to 10,000 rows for Hierarchical clustering...")
        df_sample = df.sample(n=10000, random_state=42).copy()
    else:
        df_sample = df.copy()

    print(f"[hierarchical] Fitting Agglomerative (k={n_clusters}) on {len(df_sample)} rows...")
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(df_sample)
    print("[hierarchical] Training complete.")
    
    return agg, labels, df_sample


# ---------------------------------------------------------------------------
# 3. Visualisations
# ---------------------------------------------------------------------------
def plot_cluster_scatter_2d(df: pd.DataFrame, labels: np.ndarray, 
                            title: str, filename: str, save_dir: str = ARTIFACTS_DIR) -> None:
    """Plot the first two principal components colored by cluster assignment."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=df.iloc[:, 0], 
        y=df.iloc[:, 1], 
        hue=labels, 
        palette="tab10", 
        alpha=0.6, 
        edgecolor=None,
        s=15
    )
    plt.title(title, fontsize=14)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(save_dir, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------------------------
def run_clustering_pipeline(df_pca: pd.DataFrame | None = None) -> None:
    _ensure_dirs()

    if df_pca is None:
        df_pca = load_pca_data()

    # --- 1. K-Means ---
    kmeans_k = 5
    kmeans_model, kmeans_labels = train_kmeans(df_pca, n_clusters=kmeans_k)
    
    # Save Model & Plot
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    plot_cluster_scatter_2d(df_pca, kmeans_labels, 
                            f"K-Means Clustering (k={kmeans_k})", "kmeans_clusters_2d.png")
    
    # Optional: calculate silhouette on a small sample to avoid hour-long computation
    silhouette_samp = df_pca.sample(n=10000, random_state=42)
    silhouette_lbls = kmeans_model.predict(silhouette_samp)
    score = silhouette_score(silhouette_samp, silhouette_lbls)
    print(f"[eval] K-Means Silhouette Score (10k sample): {score:.4f}\n")


    # --- 2. DBSCAN ---
    # PCA components are unit-scaled roughly, so eps=0.5 to 2.0 is common. We'll use 1.0.
    dbscan_model, dbscan_labels, df_dbscan = train_dbscan(df_pca, eps=1.0, min_samples=15)
    joblib.dump(dbscan_model, os.path.join(MODELS_DIR, "dbscan_model.pkl"))
    plot_cluster_scatter_2d(df_dbscan, dbscan_labels, 
                            "DBSCAN Clustering", "dbscan_clusters_2d.png")
    print("")

    # --- 3. Hierarchical ---
    hier_k = 5
    hier_model, hier_labels, df_hier = train_hierarchical(df_pca, n_clusters=hier_k)
    # Note: AgglomerativeClustering doesn't implement predict() to save naturally for new points like K-Means does
    joblib.dump(hier_model, os.path.join(MODELS_DIR, "hierarchical_model.pkl"))
    plot_cluster_scatter_2d(df_hier, hier_labels, 
                            f"Hierarchical Clustering (k={hier_k})", "hierarchical_clusters_2d.png")
    print("")
    
    # Merge KMeans labels onto full dataset to save to CSV
    df_clustered = df_pca.copy()
    df_clustered["Cluster_KMeans"] = kmeans_labels
    out_csv = os.path.join(PROCESSED_DIR, "online_retail_clustered.csv")
    df_clustered.to_csv(out_csv, index=False)
    print(f"[save] Clustered dataset saved → {out_csv}")


if __name__ == "__main__":
    run_clustering_pipeline()
