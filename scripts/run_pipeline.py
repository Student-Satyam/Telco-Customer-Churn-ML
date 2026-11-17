#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from posthog import project_root
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data                    # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_telco_data    # Data quality validation

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Create a valid file URI for Windows
    mlruns_path = args.mlflow_uri or (Path(project_root) / "mlruns").as_uri()  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")           # Model type for comparison
      
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df)  # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Apply feature engineering transformations
        df_enc = build_features(df, target_col=target)  # Binary encoding + one-hot encoding
        
        # IMPORTANT: Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} features")

        # === CRITICAL: Save Feature Metadata for Serving Consistency ===
        # This ensures serving pipeline uses exact same features in exact same order
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        # Save locally for development serving
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # ESSENTIAL: Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use identical transformations
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: Train/Test Split ===
        print("📊 Splitting data...")
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        
        
     
        # First split: separate training data from test data
        X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42
)      

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
    y_train_full,
    test_size=0.25,
    stratify=y_train_full,
    random_state=42
)

        print(
    f"Training: {len(X_train)} | "
    f"Validation: {len(X_val)} | "
    f"Testing: {len(X_test)}"
)
        
        
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === CRITICAL: Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 5: Model Training with Optimized Hyperparameters ===
        print("🤖 Training XGBoost model...")
        
        # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
        # In production, consider using hyperparameter optimization tools like Optuna
        model = XGBClassifier(
            # Tree structure parameters
            n_estimators=301,        # Number of trees (OPTIMIZED)
            learning_rate=0.034,     # Step size shrinkage (OPTIMIZED)  
            max_depth=7,            # Maximum tree depth (OPTIMIZED)
            
            # Regularization parameters
            subsample=0.95,         # Sample ratio of training instances
            colsample_bytree=0.98,  # Sample ratio of features for each tree
            
            # Performance parameters
            n_jobs=-1,              # Use all CPU cores
            random_state=42,        # Reproducible results
            eval_metric="logloss",  # Evaluation metric
            
            # ESSENTIAL: Handle class imbalance
            scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)
        )

        # === Train Model and Track Training Time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Validation-Based Threshold Selection ===
        print("\n📊 Selecting best threshold using validation data...")

        # Step 1: Get churn probabilities from validation data
        val_proba = model.predict_proba(X_val)[:, 1]

        # Step 2: Try different classification thresholds
        thresholds = [0.20, 0.30, 0.35, 0.40, 0.50]
        threshold_results = []

        print("\n🎯 Validation Threshold Comparison:")
        print("Threshold | Precision | Recall | F1 Score")
        print("-" * 45)

        for threshold in thresholds:
            val_pred = (val_proba >= threshold).astype(int)

            val_precision = precision_score(
                y_val, val_pred, zero_division=0
            )
            val_recall = recall_score(
                y_val, val_pred, zero_division=0
            )
            val_f1 = f1_score(
                y_val, val_pred, zero_division=0
            )

            threshold_results.append({
                "threshold": threshold,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1
            })

            print(
                f"   {threshold:.2f}    |"
                f"   {val_precision:.3f}   |"
                f"  {val_recall:.3f} |"
                f"  {val_f1:.3f}"
            )

        # Step 3: Select threshold with the highest validation F1 score
        best_result = max(
            threshold_results,
            key=lambda result: result["f1"]
        )

        best_threshold = best_result["threshold"]

        print(f"\n✅ Best threshold: {best_threshold:.2f}")
        print(f"   Validation Precision: {best_result['precision']:.3f}")
        print(f"   Validation Recall: {best_result['recall']:.3f}")
        print(f"   Validation F1: {best_result['f1']:.3f}")

        # Save the selected threshold in MLflow
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_metric("validation_precision", best_result["precision"])
        mlflow.log_metric("validation_recall", best_result["recall"])
        mlflow.log_metric("validation_f1", best_result["f1"])

        # === FINAL TEST EVALUATION ===
        print("\n📊 Evaluating final model on unseen test data...")

        # Generate predictions and track inference time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]

        # Apply the threshold selected using validation data
        y_pred = (proba >= best_threshold).astype(int)

        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        # Calculate final test metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, proba)

        # Log final test metrics to MLflow
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print("\n🎯 Final Test Performance:")
        print(f"   Selected Threshold: {best_threshold:.2f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        print(f"   ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""
