"""
Cluster Interpretation Module
=============================
Loads the original (unscaled) engineer features and merges them with
the K-Means cluster assignments to calculate human-readable business
statistics. 

This helps us interpret what each cluster represents (e.g. "High Value", 
"Occasional Buyers", etc.) and generates summary tables and visualisations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "online_retail_processed.csv")
CLUSTERED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "online_retail_clustered.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

def _ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data & Merge
# ---------------------------------------------------------------------------
def load_and_merge_data() -> pd.DataFrame:
    """
    Load the feature-engineered data (with real-world values like $)
    and attach the K-Means cluster labels.
    The row order is identical because we didn't shuffle during scaling/PCA.
    """
    df_raw = pd.read_csv(PROCESSED_DATA_PATH)
    df_clusters = pd.read_csv(CLUSTERED_DATA_PATH)
    
    # We use K-Means as it was our best performing model
    df_raw["Cluster"] = df_clusters["Cluster_KMeans"]
    return df_raw

# ---------------------------------------------------------------------------
# 2. Compute Cluster Profiles
# ---------------------------------------------------------------------------
def compute_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the median/mean of key business metrics for each cluster.
    Using median for monetary/quantity features as they can be skewed by outliers.
    """
    # Define features we care about for business interpretation
    features_to_profile = [
        "TotalPrice",
        "Quantity",
        "avg_purchase_value",
        "purchase_frequency",
        "unique_products_bought",
        "avg_time_between_purchases",
        "weekend_purchase_ratio",
        "night_purchase_ratio",
        "country_diversity"
    ]
    
    # Group by cluster and calculate median (robust to outliers)
    # and also get the count of customers/transactions in each cluster
    cluster_stats = df.groupby("Cluster")[features_to_profile].median().round(2)
    cluster_stats["Transaction_Count"] = df.groupby("Cluster").size()
    cluster_stats["Customer_Count"] = df.groupby("Cluster")["CustomerID"].nunique()
    
    return cluster_stats

# ---------------------------------------------------------------------------
# 3. Label Clusters
# ---------------------------------------------------------------------------
def generate_cluster_labels(cluster_stats: pd.DataFrame) -> dict:
    """
    Heuristically assign a human-readable name to each cluster based on its stats.
    (This is a simple rules-based approach for the 5-cluster K-Means model).
    """
    labels = {}
    
    for cluster_id, row in cluster_stats.iterrows():
        name = []
        
        # Spend behavior
        if row["TotalPrice"] > cluster_stats["TotalPrice"].quantile(0.8):
            name.append("High Spend")
        elif row["TotalPrice"] < cluster_stats["TotalPrice"].quantile(0.2):
            name.append("Low Spend")
            
        # Frequency behavior
        if row["purchase_frequency"] > cluster_stats["purchase_frequency"].quantile(0.8):
            name.append("Frequent")
        elif row["purchase_frequency"] < cluster_stats["purchase_frequency"].quantile(0.2):
            name.append("Occasional")
            
        # Time behavior
        if row["weekend_purchase_ratio"] > 0.5:
            name.append("Weekend")
            
        # If no specific extreme, call it Average/Standard
        if not name:
            label = "Standard / Typical Buyers"
        else:
            label = " ".join(name) + " Buyers"
            
        labels[cluster_id] = label
        
    return labels

# ---------------------------------------------------------------------------
# 4. Visualisations
# ---------------------------------------------------------------------------
def plot_cluster_profiles(cluster_stats: pd.DataFrame, cluster_names: dict, save_dir: str = ARTIFACTS_DIR) -> None:
    """Plot key metrics comparing clusters."""
    # Reset index to make plotting easier
    plot_df = cluster_stats.reset_index()
    plot_df["Cluster_Name"] = plot_df["Cluster"].map(lambda x: f"C{x}: {cluster_names[x]}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total Price (Spend per transaction)
    sns.barplot(x="TotalPrice", y="Cluster_Name", data=plot_df, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Median Transaction Value ($)")
    axes[0, 0].set_ylabel("")
    
    # 2. Purchase Frequency
    sns.barplot(x="purchase_frequency", y="Cluster_Name", data=plot_df, ax=axes[0, 1], palette="magma")
    axes[0, 1].set_title("Median Purchase Frequency (Invoices)")
    axes[0, 1].set_ylabel("")
    
    # 3. Transaction Count (Size of cluster)
    sns.barplot(x="Transaction_Count", y="Cluster_Name", data=plot_df, ax=axes[1, 0], palette="rocket")
    axes[1, 0].set_title("Cluster Size (Total Transactions)")
    axes[1, 0].set_ylabel("")
    
    # 4. Unique Products Bought
    sns.barplot(x="unique_products_bought", y="Cluster_Name", data=plot_df, ax=axes[1, 1], palette="mako")
    axes[1, 1].set_title("Median Unique Products Bought")
    axes[1, 1].set_ylabel("")
    
    plt.suptitle("Cluster Profiles: What does each group represent?", fontsize=18, y=1.02)
    plt.tight_layout()
    
    out = os.path.join(save_dir, "cluster_interpretation_profiles.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_interpretation_pipeline():
    _ensure_dirs()
    
    # 1. Load & merge
    df = load_and_merge_data()
    
    # 2. Compute profiles
    cluster_stats = compute_cluster_profiles(df)
    
    # 3. Generate human labels
    cluster_names = generate_cluster_labels(cluster_stats)
    cluster_stats["Cluster_Profile"] = pd.Series(cluster_names)
    
    # Rearrange columns so profile name is first
    cols = ["Cluster_Profile", "Customer_Count", "Transaction_Count"] + [c for c in cluster_stats.columns if c not in ["Cluster_Profile", "Customer_Count", "Transaction_Count"]]
    cluster_stats = cluster_stats[cols]
    
    print("\n" + "="*80)
    print("CLUSTER INTERPRETATION (MEDIAN VALUES)")
    print("="*80)
    
    # Print out nicely formatted interpretation
    for cluster_id in sorted(cluster_stats.index):
        row = cluster_stats.loc[cluster_id]
        print(f"\n🏷️  Cluster {cluster_id}: {row['Cluster_Profile']}")
        print("-" * 40)
        print(f"👥 Size: {row['Customer_Count']:,.0f} unique customers ({row['Transaction_Count']:,.0f} transactions)")
        print(f"💰 Median Spend per Tx: ${row['TotalPrice']:.2f}")
        print(f"🔄 Purchase Frequency:  {row['purchase_frequency']} invoices")
        print(f"🛒 Unique Products:     {row['unique_products_bought']} items")
        print(f"⏱️  Avg Days Btw Tx:     {row['avg_time_between_purchases']:.1f} days")
        print(f"📅 Weekend Ratio:       {row['weekend_purchase_ratio']*100:.1f}%")
        
    print("\n" + "="*80 + "\n")
    
    # 4. Save results and plots
    out_csv = os.path.join(ARTIFACTS_DIR, "cluster_profiles.csv")
    cluster_stats.to_csv(out_csv)
    print(f"[save] Cluster stats saved to {out_csv}")
    
    plot_cluster_profiles(cluster_stats, cluster_names)


if __name__ == "__main__":
    run_interpretation_pipeline()
