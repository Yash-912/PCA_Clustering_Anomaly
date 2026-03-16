"""
Data Preprocessing Module
=========================
Loads raw online retail data, cleans it (handles missing values, filters invalid
rows, converts dtypes), generates EDA visualisation artifacts, and saves the
preprocessed DataFrame to data/processed/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Paths  (everything is relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "online_retail.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Load
# ---------------------------------------------------------------------------
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[load] Raw shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2.  Inspect & visualise missing values
# ---------------------------------------------------------------------------
def inspect_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of missing-value counts / percentages."""
    nulls = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Values": nulls, "Percentage": null_pct})
    print("[inspect] Missing-value summary:")
    print(missing_df)
    return missing_df


def plot_missing_values(missing_df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    """Bar-plot of missing-value percentages; saved to *save_dir*."""
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_df.index, y=missing_df["Percentage"])
    plt.xticks(rotation=45)
    plt.title("Missing Values Percentage")
    plt.tight_layout()
    out = os.path.join(save_dir, "missing_values_pct.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 3.  Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    * Drop rows where CustomerID is NaN
    * Drop rows where Description is NaN
    * Keep only Quantity > 0 and UnitPrice > 0
    * Convert InvoiceDate → datetime, CustomerID → int
    """
    df = df.dropna(subset=["CustomerID"])
    df = df.dropna(subset=["Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    print(f"[clean] Cleaned shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4.  EDA Visualisations (preprocessing stage)
# ---------------------------------------------------------------------------
def plot_quantity_distribution(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Quantity"], bins=2)
    plt.title("Quantity Distribution")
    plt.tight_layout()
    out = os.path.join(save_dir, "quantity_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_unit_price_distribution(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["UnitPrice"], bins=5)
    plt.title("Unit Price Distribution")
    plt.tight_layout()
    out = os.path.join(save_dir, "unit_price_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_top_countries(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    top_countries = df["Country"].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title("Top Countries by Orders")
    plt.xlabel("Orders")
    plt.tight_layout()
    out = os.path.join(save_dir, "top_countries_by_orders.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 5.  Save preprocessed data
# ---------------------------------------------------------------------------
def save_preprocessed(df: pd.DataFrame, out_dir: str = PROCESSED_DIR,
                      filename: str = "online_retail_cleaned.csv") -> str:
    """Persist the cleaned DataFrame to CSV and return the path."""
    out = os.path.join(out_dir, filename)
    df.to_csv(out, index=False)
    print(f"[save] Preprocessed data → {out}")
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline() -> pd.DataFrame:
    """Execute the full preprocessing pipeline and return the cleaned df."""
    _ensure_dirs()

    # 1. Load
    df = load_raw_data()

    # 2. Inspect & visualise missing values
    missing_df = inspect_missing(df)
    plot_missing_values(missing_df)

    # 3. Clean
    df = clean_data(df)

    # 4. EDA plots that belong to the preprocessing stage
    plot_quantity_distribution(df)
    plot_unit_price_distribution(df)
    plot_top_countries(df)

    # 5. Save preprocessed (cleaned) data – *before* feature engineering
    save_preprocessed(df)

    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()
