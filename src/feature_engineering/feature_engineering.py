"""
Feature Engineering Module
===========================
Takes the *cleaned* DataFrame produced by the preprocessing pipeline,
engineers transaction-level and customer-level features, generates
associated visualisation artifacts, and saves the fully processed
DataFrame to data/processed/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths  (everything relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "online_retail_cleaned.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def _ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Load cleaned data
# ---------------------------------------------------------------------------
def load_cleaned_data(path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed / cleaned CSV."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"[load] Cleaned data shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2.  Transaction-level features
# ---------------------------------------------------------------------------
def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create:
      * TotalPrice  = Quantity × UnitPrice
      * Year / Month / Day / Hour  (from InvoiceDate)
      * is_weekend  (Saturday=5 / Sunday=6)
      * is_night    (Hour < 6 or Hour >= 22)
    """
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"]  = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"]   = df["InvoiceDate"].dt.day
    df["Hour"]  = df["InvoiceDate"].dt.hour
    df["is_weekend"] = df["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"]   = ((df["Hour"] < 6) | (df["Hour"] >= 22)).astype(int)
    print("[features] Transaction-level features added.")
    return df


# ---------------------------------------------------------------------------
# 3.  Customer-level features
# ---------------------------------------------------------------------------
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per CustomerID:
      * avg_purchase_value
      * purchase_frequency       (# unique invoices)
      * unique_products_bought   (# unique stock codes)
      * avg_time_between_purchases (days)
      * weekend_purchase_ratio
      * night_purchase_ratio
      * country_diversity        (# unique countries)
    Then merge back onto *df* and return.
    """
    # -- avg_purchase_value
    avg_purchase_value = (
        df.groupby("CustomerID")
          .agg(avg_purchase_value=("TotalPrice", "mean"))
    )

    # -- purchase_frequency
    purchase_frequency = (
        df.groupby("CustomerID")
          .agg(purchase_frequency=("InvoiceNo", "nunique"))
    )

    # -- unique_products_bought
    unique_products = (
        df.groupby("CustomerID")
          .agg(unique_products_bought=("StockCode", "nunique"))
    )

    # -- avg_time_between_purchases
    def _avg_days_between(group: pd.DataFrame) -> float:
        dates = group["InvoiceDate"].sort_values().drop_duplicates()
        if len(dates) < 2:
            return np.nan
        diffs = dates.diff().dropna().dt.days
        return diffs.mean()

    avg_time_between = (
        df.groupby("CustomerID")
          .apply(_avg_days_between)
          .to_frame("avg_time_between_purchases")
    )

    # -- weekend_purchase_ratio
    weekend_ratio = (
        df.groupby("CustomerID")
          .agg(weekend_purchase_ratio=("is_weekend", "mean"))
    )

    # -- night_purchase_ratio
    night_ratio = (
        df.groupby("CustomerID")
          .agg(night_purchase_ratio=("is_night", "mean"))
    )

    # -- country_diversity
    country_diversity = (
        df.groupby("CustomerID")
          .agg(country_diversity=("Country", "nunique"))
    )

    # join all customer-level features
    customer_features = (
        avg_purchase_value
        .join(purchase_frequency)
        .join(unique_products)
        .join(avg_time_between)
        .join(weekend_ratio)
        .join(night_ratio)
        .join(country_diversity)
    )

    print(f"[features] Customer-level features built – shape {customer_features.shape}")
    print(customer_features.head())

    # merge onto main df
    df = df.merge(customer_features, on="CustomerID", how="left")
    print(f"[features] Merged – final shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4.  Feature-engineering visualisations
# ---------------------------------------------------------------------------
def plot_top_products(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    top_products = (
        df.groupby("Description")["Quantity"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title("Top 10 Best Selling Products")
    plt.xlabel("Quantity Sold")
    plt.tight_layout()
    out = os.path.join(save_dir, "top_10_best_selling_products.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_revenue_by_country(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    revenue_country = (
        df.groupby("Country")["TotalPrice"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=revenue_country.values, y=revenue_country.index)
    plt.title("Top Countries by Revenue")
    plt.xlabel("Revenue")
    plt.tight_layout()
    out = os.path.join(save_dir, "top_countries_by_revenue.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_monthly_sales_trend(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    monthly_sales = df.groupby("Month")["TotalPrice"].sum()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    out = os.path.join(save_dir, "monthly_sales_trend.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


def plot_hourly_sales_trend(df: pd.DataFrame, save_dir: str = ARTIFACTS_DIR) -> None:
    hourly_sales = df.groupby("Hour")["TotalPrice"].sum()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=hourly_sales.index, y=hourly_sales.values, marker="o")
    plt.title("Hourly Sales Trend")
    plt.xlabel("Hour")
    plt.ylabel("Revenue")
    plt.tight_layout()
    out = os.path.join(save_dir, "hourly_sales_trend.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 5.  Save fully processed data
# ---------------------------------------------------------------------------
def save_processed(df: pd.DataFrame,
                   out_dir: str = PROCESSED_DIR,
                   filename: str = "online_retail_processed.csv") -> str:
    out = os.path.join(out_dir, filename)
    df.to_csv(out, index=False)
    print(f"[save] Processed data → {out}")
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_feature_engineering_pipeline(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Execute the full feature-engineering pipeline.

    Parameters
    ----------
    df : DataFrame, optional
        If not supplied, the cleaned CSV is loaded from disk.

    Returns
    -------
    DataFrame with all engineered features.
    """
    _ensure_dirs()

    # 1.  Load (if not passed in)
    if df is None:
        df = load_cleaned_data()

    # 2.  Transaction-level features
    df = add_transaction_features(df)

    # 3.  Customer-level features
    df = build_customer_features(df)

    # 4.  Visualisations
    plot_top_products(df)
    plot_revenue_by_country(df)
    plot_monthly_sales_trend(df)
    plot_hourly_sales_trend(df)

    # 5.  Persist
    save_processed(df)

    return df


if __name__ == "__main__":
    run_feature_engineering_pipeline()
