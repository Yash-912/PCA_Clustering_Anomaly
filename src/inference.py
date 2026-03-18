"""
Inference Pipeline
==================
Simulates deploying our unsupervised learning models into a production backend.

Given a new customer's behavioral feature vector, this script loads the trained 
artifacts (Scaler -> PCA -> K-Means / Isolation Forest) and instantly outputs:
    1. The customer's assigned Cluster (Segment).
    2. Whether the customer is an Anomaly.
"""

import os
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "standard_scaler.pkl")
PCA_PATH = os.path.join(MODELS_DIR, "pca_model.pkl")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
IFOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest_model.pkl")

# The exact 15 features our pipeline was trained on
FEATURES = [
    "Quantity", "UnitPrice", "TotalPrice", "Month", "Day", "Hour", 
    "is_weekend", "is_night", "avg_purchase_value", "purchase_frequency", 
    "unique_products_bought", "avg_time_between_purchases", 
    "weekend_purchase_ratio", "night_purchase_ratio", "country_diversity"
]

def load_models():
    """Load all necessary serialized models from disk."""
    print("[inference] Loading trained models...")
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    iforest = joblib.load(IFOREST_PATH)
    return scaler, pca, kmeans, iforest

def predict_single_customer(customer_data: dict) -> dict:
    """
    Passes a single customer record through the entire data science pipeline.
    """
    # 1. Convert to DataFrame to match expected input formatting
    df = pd.DataFrame([customer_data])
    
    # Ensure columns match training order exactly
    df = df[FEATURES]
    
    # Load Models
    scaler, pca, kmeans, iforest = load_models()
    
    # 2. Scale the Data (using the historical mean/std from training)
    scaled_data = scaler.transform(df)
    
    # 3. PCA Embedding (compress 15 features to 10 components)
    pca_embedded_data = pca.transform(scaled_data)
    
    # 4. Predict Cluster
    cluster_label = kmeans.predict(pca_embedded_data)[0]
    
    # 5. Predict Anomaly (-1 means Anomaly, 1 means Normal)
    anomaly_status = iforest.predict(pca_embedded_data)[0]
    anomaly_score = iforest.decision_function(pca_embedded_data)[0]
    
    result = {
        "Assigned_Cluster": int(cluster_label),
        "Is_Anomaly": bool(anomaly_status == -1),
        "Anomaly_Score": float(anomaly_score),
        "Profile_Meaning": get_cluster_meaning(cluster_label)
    }
    
    return result

def get_cluster_meaning(cluster_id: int) -> str:
    """Helper function mocking a database lookup for cluster names."""
    profiles = {
        0: "Weekend Buyers",
        1: "Typical Daily Buyers (Low-Mid Spend)",
        2: "Typical Daily Buyers (Upper-Mid Spend)",
        3: "Low Spend, Highly Frequent Buyers (Wholesalers/Distributors)",
        4: "High Spend, Occasional Buyers (Massive Outliers)"
    }
    return profiles.get(cluster_id, "Unknown Profile")

if __name__ == "__main__":
    # Simulate a new incoming customer from the backend/database
    new_customer = {
        "Quantity": 5000,                      
        "UnitPrice": 15.0,                  
        "TotalPrice": 75000.0,                 
        "Month": 11,                           
        "Day": 15,                             
        "Hour": 14,                            
        "is_weekend": 0,                       
        "is_night": 0,                         
        "avg_purchase_value": 75000.0,         
        "purchase_frequency": 1,               
        "unique_products_bought": 1,           
        "avg_time_between_purchases": 0.0,     
        "weekend_purchase_ratio": 0.0,         
        "night_purchase_ratio": 0.0,           
        "country_diversity": 1                 
    }
    
    print("\n--- INCOMING CUSTOMER DATA ---")
    for k, v in new_customer.items():
        print(f"{k}: {v}")
        
    print("\n--- RUNNING INFERENCE PIPELINE ---")
    prediction = predict_single_customer(new_customer)
    
    print("\n--- FINAL PREDICTION RESULT ---")
    print(f"Cluster Assigned: {prediction['Assigned_Cluster']}")
    print(f"Cluster Meaning:  {prediction['Profile_Meaning']}")
    print(f"Is Anomaly?:      {'🚨 YES' if prediction['Is_Anomaly'] else '✅ NO'}")
    print(f"Anomaly Score:    {prediction['Anomaly_Score']:.4f}")
    print("-" * 30 + "\n")
