"""
Citibike demand prediction model training.

This module handles:
1. Loading and preprocessing feature data
2. Model selection and hyperparameter tuning
3. Model evaluation and validation
4. Model persistence and metrics logging
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prep_data(features_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load feature data and prepare for training.
    
    Args:
        features_path: Path to parquet file with features
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    # Validate required columns
    required_cols = ['trips', 'ds', 'hour', 'start_station_id']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Split features/target
    y = df['trips']
    X = df.drop(['trips', 'ds', 'start_station_id'], axis=1)
    
    logger.info(f"Features shape: {X.shape}, unique stations: {df['start_station_id'].nunique()}")
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, Dict]:
    """
    Train model and compute performance metrics.
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        Tuple of (trained model, metrics dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    
    # Compute cross-val score first
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring='neg_root_mean_squared_error'
    )
    
    # Fit on full training set
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "cv_rmse_mean": float(-cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "feature_importance": dict(zip(X.columns, model.feature_importances_)),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Model RMSE: {metrics['rmse']:.2f}")
    return model, scaler, metrics

def save_artifacts(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    metrics: Dict,
    out_path: str
) -> None:
    """
    Save model artifacts and metrics.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        metrics: Performance metrics
        out_path: Path to save model
    """
    # Save model and scaler together
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    artifacts = {
        "model": model,
        "scaler": scaler
    }
    # Only include feature names if available (scikit-learn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        artifacts["feature_names"] = model.feature_names_in_
    joblib.dump(artifacts, out_path)
    
    # Save metrics
    metrics_path = os.path.join(os.path.dirname(out_path), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved model to {out_path} and metrics to {metrics_path}")

def main(features_path: str, out_path: str):
    """Main training pipeline."""
    try:
        # Load and prep data
        X, y = load_and_prep_data(features_path)
        
        # Train and evaluate
        model, scaler, metrics = train_model(X, y)
        
        # Save everything
        save_artifacts(model, scaler, metrics, out_path)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True,
                  help="Path to input features parquet file")
    p.add_argument("--out", required=True,
                  help="Path to save trained model artifacts")
    args = p.parse_args()
    main(args.features, args.out)
