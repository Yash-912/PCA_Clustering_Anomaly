"""
Cluster Evaluation Module
=========================
Calculates internal validation metrics for the clustering algorithms:
    - Silhouette Score: Measures how tightly grouped the clusters are (higher is better, -1 to 1).
    - Davies-Bouldin Index: Measures the ratio of within-cluster scatter to between-cluster separation (lower is better).
    - Calinski-Harabasz Index: Variance ratio criterion (higher is better).

Note:
    In unsupervised learning, there is no "ground truth" to calculate traditional 
    accuracy (like 95%). Instead, we use mathematical metrics that evaluate cluster 
    density and separation.
    
    Since computing these metrics requires pairwise distances (O(N^2) memory and time), 
    evaluating on the full 400k dataset would take hours or crash. We evaluate 
    the algorithms on a representative random sample of 10,000 rows.
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCA_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "online_retail_pca.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

def _ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data & Sample
# ---------------------------------------------------------------------------
def load_and_sample_data(path: str = PCA_DATA_PATH, sample_size: int = 10000) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) > sample_size:
        print(f"[eval] Downsampling from {len(df)} to {sample_size} rows for O(N^2) metric evaluation...")
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()
    return df_sample

# ---------------------------------------------------------------------------
# 2. Compute Metrics
# ---------------------------------------------------------------------------
def evaluate_models(df_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Fits the algorithms on the sample and calculates metrics.
    We re-fit here on the exact same sample because DBSCAN and Hierarchical
    Clustering do not have a .predict() method for unseen data in scikit-learn.
    """
    print("[eval] Fitting K-Means (k=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    kmeans_labels = kmeans.fit_predict(df_sample)
    
    print("[eval] Fitting DBSCAN (eps=1.0, min_samples=15)...")
    dbscan = DBSCAN(eps=1.0, min_samples=15)
    dbscan_labels = dbscan.fit_predict(df_sample)
    
    print("[eval] Fitting Hierarchical (k=5)...")
    agg = AgglomerativeClustering(n_clusters=5)
    hier_labels = agg.fit_predict(df_sample)
    
    results = []
    
    for name, labels in [("K-Means", kmeans_labels), ("DBSCAN", dbscan_labels), ("Hierarchical", hier_labels)]:
        # If DBSCAN groups everything into noise (-1), metrics will fail
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            print(f"[eval] {name} found < 2 clusters. Skipping metrics.")
            results.append({
                "Model": name,
                "Silhouette Score": np.nan,
                "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan,
                "Valid Clusters Found": n_clusters
            })
            continue
            
        print(f"[eval] Calculating metrics for {name}...")
        sil = silhouette_score(df_sample, labels)
        db = davies_bouldin_score(df_sample, labels)
        ch = calinski_harabasz_score(df_sample, labels)
        
        results.append({
            "Model": name,
            "Silhouette Score": sil,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch,
            "Valid Clusters": n_clusters
        })
        
    df_results = pd.DataFrame(results)
    return df_results

# ---------------------------------------------------------------------------
# 3. Create Visual Metric Table
# ---------------------------------------------------------------------------
def plot_metrics_table(df_results: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    """Save the dataframe as a nice PNG table artifact."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Format floats for the table
    df_fmt = df_results.copy()
    for col in ["Silhouette Score", "Davies-Bouldin", "Calinski-Harabasz"]:
        df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
    table = ax.table(cellText=df_fmt.values,
                     colLabels=df_fmt.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color headers
    for i in range(len(df_fmt.columns)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(color="white", weight="bold")
        
    plt.title("Clustering Metrics Comparison", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    out = os.path.join(save_dir, "clustering_metrics_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_evaluation_pipeline():
    _ensure_dirs()
    
    df_sample = load_and_sample_data()
    df_results = evaluate_models(df_sample)
    
    print("\n=== CLUSTERING METRICS ===")
    print(df_results.to_string(index=False))
    print("==========================\n")
    
    # Save the dataframe as a CSV and a beautiful PNG artifact
    df_results.to_csv(os.path.join(ARTIFACTS_DIR, "clustering_metrics.csv"), index=False)
    plot_metrics_table(df_results)
    print(f"[save] Metrics saved to {ARTIFACTS_DIR}/clustering_metrics.csv")

if __name__ == "__main__":
    run_evaluation_pipeline()
