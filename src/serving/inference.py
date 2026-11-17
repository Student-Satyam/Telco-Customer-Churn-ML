"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
from pathlib import Path

import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Docker uses /app/model; Windows uses the saved local MLflow model
# Docker deployment model directory
DOCKER_MODEL_DIR = Path("/app/model")

# GitHub-tracked deployment model directory
DEPLOYMENT_MODEL_DIR = (
    BASE_DIR
    / "src"
    / "serving"
    / "model"
    / "deployment_model"
)

# Original MLflow directory (local fallback)
LOCAL_MODEL_DIR = (
    BASE_DIR
    / "mlruns"
    / "448585764050151295"
    / "07b1e21bee924419b25775c9a9941281"
    / "artifacts"
    / "model"
)

# Select the available model directory
if DOCKER_MODEL_DIR.exists():
    MODEL_DIR = str(DOCKER_MODEL_DIR)
elif DEPLOYMENT_MODEL_DIR.exists():
    MODEL_DIR = str(DEPLOYMENT_MODEL_DIR)
else:
    MODEL_DIR = str(LOCAL_MODEL_DIR)

print(f"Loading model from: {MODEL_DIR}")

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.sklearn.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from local MLflow tracking
        import glob
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
# === FEATURE SCHEMA LOADING ===


model_path = Path(MODEL_DIR)

# Check both possible locations
possible_paths = [
    model_path / "feature_columns.txt",
    model_path.parent / "feature_columns.txt"
]

feature_file = next(
    (path for path in possible_paths if path.exists()),
    None
)

if feature_file is None:
    raise FileNotFoundError(
        f"feature_columns.txt not found. Checked: {possible_paths}"
    )

with open(feature_file, encoding="utf-8") as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
# Same threshold selected during model validation
CHURN_THRESHOLD = 0.50


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform one raw customer row using the exact training feature schema.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Start with all 30 training features set to zero.
    encoded = pd.DataFrame(
        0,
        index=df.index,
        columns=FEATURE_COLS,
        dtype=int
    )

    # STEP 1: Numeric features
    numeric_features = NUMERIC_COLS + ["SeniorCitizen"]

    for col in numeric_features:
        if col in df.columns and col in encoded.columns:
            encoded[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int if col == "SeniorCitizen" else float)
            )

    # STEP 2: Binary features
    for col, mapping in BINARY_MAP.items():
        if col in df.columns and col in encoded.columns:
            values = df[col].astype(str).str.strip()
            encoded[col] = (
                values.map(mapping)
                .fillna(0)
                .astype(int)
            )

    # STEP 3: Multi-category features
    # The first category in training is the reference category.
    category_mapping = {
        "MultipleLines": {
            "No": None,
            "No phone service": "MultipleLines_No phone service",
            "Yes": "MultipleLines_Yes"
        },
        "InternetService": {
            "DSL": None,
            "Fiber optic": "InternetService_Fiber optic",
            "No": "InternetService_No"
        },
        "OnlineSecurity": {
            "No": None,
            "No internet service": "OnlineSecurity_No internet service",
            "Yes": "OnlineSecurity_Yes"
        },
        "OnlineBackup": {
            "No": None,
            "No internet service": "OnlineBackup_No internet service",
            "Yes": "OnlineBackup_Yes"
        },
        "DeviceProtection": {
            "No": None,
            "No internet service": "DeviceProtection_No internet service",
            "Yes": "DeviceProtection_Yes"
        },
        "TechSupport": {
            "No": None,
            "No internet service": "TechSupport_No internet service",
            "Yes": "TechSupport_Yes"
        },
        "StreamingTV": {
            "No": None,
            "No internet service": "StreamingTV_No internet service",
            "Yes": "StreamingTV_Yes"
        },
        "StreamingMovies": {
            "No": None,
            "No internet service": "StreamingMovies_No internet service",
            "Yes": "StreamingMovies_Yes"
        },
        "Contract": {
            "Month-to-month": None,
            "One year": "Contract_One year",
            "Two year": "Contract_Two year"
        },
        "PaymentMethod": {
            "Bank transfer (automatic)": None,
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check"
        }
    }

    for col, categories in category_mapping.items():
        if col not in df.columns:
            continue

        value = df[col].iloc[0]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value not in categories:
            raise ValueError(
                f"Unknown category '{value}' for column '{col}'"
            )

        feature_name = categories[value]

        if feature_name is not None:
            if feature_name not in encoded.columns:
                raise ValueError(
                    f"Feature '{feature_name}' is missing from training schema"
                )

            encoded.loc[df.index[0], feature_name] = 1

    # STEP 4: Return exact training columns and order.
    return encoded[FEATURE_COLS]

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    try:
        # Get churn probabilities for the customer
        probabilities = model.predict_proba(df_enc)

        # Extract probability of churn (class 1)
        churn_probability = float(probabilities[0][1])

        # Apply the threshold selected during validation
        result = int(churn_probability >= CHURN_THRESHOLD)

        print(f"Churn probability: {churn_probability:.2%}")
        print(f"Selected threshold: {CHURN_THRESHOLD}")

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # === STEP 4: Convert to Business-Friendly Output ===
    if result == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"