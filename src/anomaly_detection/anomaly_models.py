"""
Anomaly Detection Module
========================
Loads the PCA-embedded behavioral data and applies Unsupervised Anomaly 
Detection algorithms to identify outright outliers (e.g., massive bulk 
buyers, highly unusual spending patterns, potential fraud).

Implemented algorithms:
    1. Isolation Forest (Scales well, runs on full dataset)
    2. Local Outlier Factor / LOF (O(N^2) memory, runs on sample)

Generates anomaly scores, flags anomalous transactions (-1), and saves
the visualisations and data to disk.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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
def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.01) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    """
    Train Isolation Forest on the full dataset.
    contamination: The proportion of outliers in the data set (1% by default).
    Returns the model, predictions (1 for normal, -1 for anomaly), and anomaly scores.
    """
    print(f"[iforest] Fitting Isolation Forest on {len(df)} rows with contamination={contamination}...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    
    # Fit and predict
    labels = iso_forest.fit_predict(df)
    
    # Anomaly scores (lower means more abnormal)
    scores = iso_forest.decision_function(df)
    
    n_anomalies = list(labels).count(-1)
    print(f"[iforest] Found {n_anomalies} anomalies automatically.")
    
    return iso_forest, labels, scores


def train_lof(df: pd.DataFrame, contamination: float = 0.01) -> tuple[LocalOutlierFactor, np.ndarray, pd.DataFrame]:
    """
    Train Local Outlier Factor (LOF).
    Requires pairwise distances (O(N^2) memory), so we downsample to 20,000 rows.
    """
    if len(df) > 20000:
        print(f"[lof] Dataset too large ({len(df)} rows) for Local Outlier Factor memory constraints.")
        print("[lof] Downsampling to 20,000 rows for LOF training...")
        df_sample = df.sample(n=20000, random_state=42).copy()
    else:
        df_sample = df.copy()

    print(f"[lof] Fitting LOF on {len(df_sample)} rows with contamination={contamination}...")
    # novelty=False is default, meaning fit_predict is the standard way to get labels on training data
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
    
    labels = lof.fit_predict(df_sample)
    
    n_anomalies = list(labels).count(-1)
    print(f"[lof] Found {n_anomalies} anomalies in the sample.")
    
    return lof, labels, df_sample

# ---------------------------------------------------------------------------
# 3. Visualisations
# ---------------------------------------------------------------------------
def plot_anomaly_scatter(df: pd.DataFrame, labels: np.ndarray, title: str, filename: str, save_dir: str = ARTIFACTS_DIR) -> None:
    """Plot PC1 vs PC2, coloring anomalies red (-1) and normal points blue (1)."""
    # Map labels back to string categories for plotting legend
    status = np.where(labels == 1, "Normal", "Anomaly")
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=df.iloc[:, 0], 
        y=df.iloc[:, 1], 
        hue=status, 
        palette={"Normal": "steelblue", "Anomaly": "crimson"}, 
        alpha=0.6, 
        s=15,
        edgecolor=None
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(save_dir, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_anomaly_scores(scores: np.ndarray, save_dir: str = ARTIFACTS_DIR) -> None:
    """Histogram of Isolation Forest anomaly scores."""
    plt.figure(figsize=(10, 5))
    sns.histplot(scores, bins=50, color="purple", kde=True)
    
    plt.axvline(x=0, color='red', linestyle='--', label='Anomaly Decision Boundary (< 0)')
    plt.title("Isolation Forest Anomaly Scores Distribution", fontsize=14)
    plt.xlabel("Anomaly Score (Negative = Outlier, Positive = Normal)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "anomaly_scores_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")

# ---------------------------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------------------------
def run_anomaly_pipeline(df_pca: pd.DataFrame | None = None) -> None:
    _ensure_dirs()

    if df_pca is None:
        df_pca = load_pca_data()

    # We assume roughly 1% of transactions/customers are anomalies (heavy outliers)
    outlier_fraction = 0.01

    # --- 1. Isolation Forest ---
    iso_model, iso_labels, iso_scores = train_isolation_forest(df_pca, contamination=outlier_fraction)
    joblib.dump(iso_model, os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))
    
    plot_anomaly_scatter(df_pca, iso_labels, 
                         "Isolation Forest — Anomaly Detection (1% Contamination)", 
                         "isolation_forest_anomalies_2d.png")
                         
    plot_anomaly_scores(iso_scores)


    # --- 2. Local Outlier Factor ---
    lof_model, lof_labels, df_lof = train_lof(df_pca, contamination=outlier_fraction)
    plot_anomaly_scatter(df_lof, lof_labels, 
                         "Local Outlier Factor — Anomaly Detection (Sampled)", 
                         "lof_anomalies_2d.png")
    
    # Save the labels onto the final fully processed dataframe
    # So we can track both Cluster IDs and Anomaly Status together
    final_output_path = os.path.join(PROCESSED_DIR, "online_retail_clustered.csv")
    
    if os.path.exists(final_output_path):
        df_final = pd.read_csv(final_output_path)
        df_final["Is_Anomaly_IForest"] = iso_labels
        df_final["Anomaly_Score_IForest"] = iso_scores
        df_final.to_csv(final_output_path, index=False)
        print(f"\n[save] Overwrote {final_output_path} with Anomaly status columns added.")
    else:
        print("\n[warn] online_retail_clustered.csv not found to append anomaly labels.")


if __name__ == "__main__":
    run_anomaly_pipeline()
