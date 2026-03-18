# 🛒 E-Commerce Behavior Embedding & Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An end-to-end **Unsupervised Machine Learning pipeline** built to ingest raw e-commerce transaction logs, engineer behavioral features, reduce dimensionality, segment customers into interpretable groups, and proactively flag anomalous/fraudulent behaviors. 

This project bridges the gap between raw data processing and interpretable business value through a fully modular architecture and an interactive web dashboard.

---

## 🌟 Key Features & Pipeline Overview

The repository is modularized into dedicated pipeline stages:

1. **`src/data_preprocessing/`**: Cleans raw data, handles missing values, and enforces strict logical boundary rules.
2. **`src/feature_engineering/`**: Derives heavy analytical features grouped both by transaction and historically by `CustomerID` (e.g., `avg_purchase_value`, `purchase_frequency`, `weekend_purchase_ratio`).
3. **`src/data_preprocessing/data_scaling.py`**: Normalizes highly skewed features using `StandardScaler` in preparation for distance-based ML algorithms.
4. **`src/pca/pca_embedding.py`**: Condenses 15 behavioral features into an optimized 10-Dimensional space using **Principal Component Analysis (PCA)**; retaining >96% of the dataset variance.
5. **`src/clustering/`**: Fits and evaluates **K-Means**, **DBSCAN**, and **Agglomerative Hierarchical** clustering. Evaluated strictly via Silhouette Scores and the Davies-Bouldin Index. (K-Means achieved the strongest mathematical boundaries).
6. **`src/clustering/cluster_interpretation.py`**: Maps unsupervised statistical assignments back to human-readable personas (e.g., *"Weekend Buyers"*, *"High Value Occasional Buyers"*). 
7. **`src/anomaly_detection/`**: Employs **Isolation Forests** and **Local Outlier Factor (LOF)** on high-variance clusters to automatically trap extreme outliers, bulk purchasers, and potential anomalies.
8. **`src/inference.py`**: A deployment-ready inference simulator that pushes a brand-new untracked transaction through all historical `.pkl` artifacts and instantly predicts its behavioral segment and anomaly threshold.

---

## 📊 Streamlit Dashboard

Explore the data dynamically! The project includes a full-fledged **Streamlit Web Application** featuring:
- **Interactive 3D / 2D PCA Embeddings** (Exploratory Data Analysis capability)
- **Cluster Profiles & Demographics** (Median analytical summaries for Business Intelligence)
- **Anomaly Detection Distributions** (Identifying isolation density boundaries)

### Running the Dashboard Locally:
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Boot the web application
streamlit run dashboard/app.py
```
*The app will automatically launch at `http://localhost:8501`*

---

## ⚙️ Model Evaluation Results

Since unsupervised learning lacks labeled "ground truth" to measure basic accuracy, we rigorously tested the algorithms mathematically based on cluster density and boundary separation:

| Model | Silhouette Score <br>*(Higher is better)* | Davies-Bouldin Index <br>*(Lower is better)* | Calinski-Harabasz <br>*(Higher is better)* |
| :--- | :--- | :--- | :--- |
| **K-Means** | **0.2227** 🏆 | **1.1895** 🏆 | **2692.83** 🏆 |
| **Hierarchical** | 0.1817 | 1.4091 | 2379.13 |
| **DBSCAN** | 0.1208 | 2.0901 | 377.58 |

*(Evaluations were computationally scaled effectively to handle the ~400,000+ row volume without crashing $O(N^2)$ memory constraints).*

---

## 📁 Repository Structure

```text
├── artifacts/                  # Generated scatter plots, boxplots, histograms, and metric PNGs
├── dashboard/
│   └── app.py                  # Streamlit Interactive UI
├── data/
│   ├── raw/                    # Raw inputs (online_retail.csv)
│   └── processed/              # Modularized outputs (.csv step tracks)
├── models/                     # Saved pickled model artifacts for production inference
│   ├── standard_scaler.pkl
│   ├── pca_model.pkl
│   ├── kmeans_model.pkl
│   ├── isolation_forest_model.pkl
│   └── ...
├── notebooks/                  # Original conceptual scratchpad (eda.ipynb)
├── src/                        # Modular pipeline logic
│   ├── anomaly_detection/
│   ├── clustering/
│   ├── data_preprocessing/
│   ├── feature_engineering/
│   ├── pca/
│   └── inference.py            # Real-time transaction inference simulator
└── README.md
```

---

## 🚀 How to Run the Pipeline

Execute the modules sequentially to reproduce the data tracked output:

```bash
# 1. Cleanse Data
python -m src.data_preprocessing.data_preprocessing

# 2. Engineer Behavioral Rules
python -m src.feature_engineering.feature_engineering

# 3. Standardize Distance Constraints
python -m src.data_preprocessing.data_scaling

# 4. Dimensionality Reduction (PCA)
python -m src.pca.pca_embedding

# 5. Fit Cluster Algorithms
python -m src.clustering.clustering_models

# 6. Profile resulting Business Interpretations
python -m src.clustering.cluster_interpretation

# 7. Flag System Anomalies
python -m src.anomaly_detection.anomaly_models

# 8. Simulate Production Input
python src/inference.py
```
