"""
Prediction Script
Load trained models and make predictions on new data with the same structure
as the original dataset (for_train_x_2025_10_24_clean.csv)

Usage:
    python predict.py --input new_data.csv --output predictions.csv --model RNN
    
    python predict.py --input new_data.csv --output predictions.csv --model LSTM
    
    python predict.py --input new_data.csv --output predictions.csv --model best  # Auto-select best
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


def load_saved_artifacts(models_path: str = './saved_models'):
    """Load all saved preprocessing artifacts."""
    artifacts = {}
    
    # Load temporal weights
    tw_path = os.path.join(models_path, 'temporal_weights.json')
    if os.path.exists(tw_path):
        with open(tw_path, 'r') as f:
            artifacts['temporal_weights'] = json.load(f)
    
    # Load feature names
    fn_path = os.path.join(models_path, 'feature_names.json')
    if os.path.exists(fn_path):
        with open(fn_path, 'r') as f:
            artifacts['feature_names'] = json.load(f)
    
    # Load all dates (for temporal feature extraction)
    dates_path = os.path.join(models_path, 'all_dates.json')
    if os.path.exists(dates_path):
        with open(dates_path, 'r') as f:
            artifacts['all_dates'] = json.load(f)
    
    return artifacts


def extract_temporal_features(dates: np.ndarray, all_dates_dict: dict = None) -> np.ndarray:
    """
    Extract temporal features from dates.
    Same logic as chunk_17_phase_3_temporal.py
    
    Args:
        dates: Array of date values
        all_dates_dict: Dictionary mapping date -> index (for consistent encoding)
        
    Returns:
        Array of temporal features (9 features per sample)
    """
    # Convert to pandas datetime if needed
    if dates.dtype != np.datetime64:
        dates = pd.to_datetime(dates, format='%Y%m%d')
    
    n = len(dates)
    temporal_features = np.zeros((n, 9))
    
    # Get min date for relative calculations
    if all_dates_dict:
        min_date = min(all_dates_dict.keys())
        min_date = pd.to_datetime(str(min_date), format='%Y%m%d')
    else:
        min_date = dates.min()
    
    for i, date in enumerate(dates):
        # Year (normalized)
        temporal_features[i, 0] = (date.year - 2020) / 5
        
        # Month (normalized)
        temporal_features[i, 1] = date.month / 12
        
        # Day of month (normalized)
        temporal_features[i, 2] = date.day / 31
        
        # Day of week (normalized)
        temporal_features[i, 3] = date.dayofweek / 7
        
        # Quarter
        temporal_features[i, 4] = date.quarter / 4
        
        # Day of year (normalized)
        temporal_features[i, 5] = date.dayofyear / 365
        
        # Days since min date (normalized)
        days_since = (date - min_date).days
        temporal_features[i, 6] = days_since / 3650  # ~10 years
        
        # Week of year
        temporal_features[i, 7] = date.isocalendar()[1] / 52
        
        # Month start indicator (binary)
        temporal_features[i, 8] = 1.0 if date.day <= 7 else 0.0
    
    return temporal_features


def calculate_temporal_weights(dates: np.ndarray, all_dates_dict: dict = None) -> np.ndarray:
    """
    Calculate temporal weights for dates.
    Weight formula: newer dates get higher weights (1-10 range)
    
    Args:
        dates: Array of date values
        all_dates_dict: Dictionary mapping date -> index
        
    Returns:
        Array of temporal weights
    """
    if dates.dtype != np.datetime64:
        dates = pd.to_datetime(dates, format='%Y%m%d')
    
    n = len(dates)
    weights = np.ones(n)
    
    # Get date range
    if all_dates_dict:
        sorted_dates = sorted([int(d) for d in all_dates_dict.keys()])
        min_date_val = sorted_dates[0]
        max_date_val = sorted_dates[-1]
    else:
        min_date_val = int(dates.min().strftime('%Y%m%d'))
        max_date_val = int(dates.max().strftime('%Y%m%d'))
    
    date_range = max_date_val - min_date_val
    
    if date_range > 0:
        for i, date in enumerate(dates):
            if hasattr(date, 'strftime'):
                date_val = int(date.strftime('%Y%m%d'))
            else:
                date_val = int(date)
            
            # Normalize to 0-1
            normalized = (date_val - min_date_val) / date_range
            
            # Map to 1-10 range
            weights[i] = 1.0 + normalized * 9.0
    
    return weights


def preprocess_new_data(df: pd.DataFrame, 
                       date_column: str,
                       feature_columns: list,
                       artifacts: dict,
                       models_path: str = './saved_models') -> np.ndarray:
    """
    Preprocess new data using saved artifacts.
    
    Args:
        df: Input dataframe
        date_column: Name of date column
        feature_columns: List of feature column names
        artifacts: Saved preprocessing artifacts
        models_path: Path to saved models
        
    Returns:
        Preprocessed feature array
    """
    # Extract dates
    dates = df[date_column].values
    
    # Extract temporal features
    all_dates_dict = artifacts.get('all_dates')
    temporal_features = extract_temporal_features(dates, all_dates_dict)
    
    # Calculate temporal weights
    temporal_weights = calculate_temporal_weights(dates, all_dates_dict)
    
    # Get original features
    X = df[feature_columns].values.astype(np.float32)
    
    # Combine features
    X_with_temporal = np.hstack([X, temporal_features])
    
    # Apply temporal weights: X_weighted = X * sqrt(weights)
    X_weighted = X_with_temporal * np.sqrt(temporal_weights[:, np.newaxis])
    
    return X_weighted


def load_model(arch_name: str, models_path: str = './saved_models') -> tf.keras.Model:
    """Load a trained model."""
    model_path = os.path.join(models_path, f'{arch_name}_model.keras')
    return tf.keras.models.load_model(model_path)


def predict(model: tf.keras.Model, 
            X: np.ndarray, 
            threshold: float = 0.5) -> tuple:
    """
    Make predictions on new data.
    
    Args:
        model: Trained Keras model
        X: Preprocessed feature array
        threshold: Binary classification threshold (default: 0.5)
        
    Returns:
        Tuple of (probabilities, binary_predictions)
    """
    probabilities = model.predict(X, verbose=0).flatten()
    binary_predictions = (probabilities >= threshold).astype(int)
    return probabilities, binary_predictions


def main():
    parser = argparse.ArgumentParser(description='Make predictions on new data')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file')
    parser.add_argument('--model', default='RNN', choices=['RNN', 'LSTM', 'best'], 
                        help='Model to use for prediction')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Binary classification threshold')
    parser.add_argument('--models-path', default='./saved_models',
                        help='Path to saved models')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.models_path):
        print(f"Error: Models path '{args.models_path}' does not exist.")
        print("Run the training pipeline first to save models.")
        sys.exit(1)
    
    # Load saved artifacts
    print("Loading saved artifacts...")
    artifacts = load_saved_artifacts(args.models_path)
    
    if not artifacts:
        print("Error: No artifacts found in models path.")
        sys.exit(1)
    
    # Determine which model to use
    if args.model == 'best':
        # For now, default to RNN (usually has best precision)
        # Could add logic to read precision metrics
        model_name = 'RNN'
    else:
        model_name = args.model
    
    model_path = os.path.join(args.models_path, f'{model_name}_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_name}' not found at {model_path}")
        sys.exit(1)
    
    # Load model
    print(f"Loading {model_name} model...")
    model = load_model(model_name, args.models_path)
    
    # Load and preprocess new data
    print(f"Loading new data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Get feature columns (same as training)
    feature_columns = artifacts.get('feature_names', [])
    if not feature_columns:
        print("Error: Feature names not found in artifacts")
        sys.exit(1)
    
    # Check if all feature columns exist in new data
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in new data: {missing_cols}")
        print("Using available columns...")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Get date column (assuming it's named 'Date' or similar)
    # Try to find date column
    date_col_candidates = ['Date', 'DATE', 'date', 'Tradedate', 'TRADEDATE', 'date_int']
    date_column = None
    for col in date_col_candidates:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        # Try to find integer date column
        for col in df.columns:
            if df[col].dtype == np.int64 or df[col].dtype == np.int32:
                if df[col].min() > 20000000:  # Likely YYYYMMDD format
                    date_column = col
                    break
    
    if date_column is None:
        print("Error: Could not find date column in new data")
        sys.exit(1)
    
    print(f"Using date column: {date_column}")
    
    # Preprocess new data
    print("Preprocessing new data...")
    X_new = preprocess_new_data(df, date_column, feature_columns, artifacts, args.models_path)
    
    # Make predictions
    print("Making predictions...")
    probabilities, predictions = predict(model, X_new, args.threshold)
    
    # Save results
    results = pd.DataFrame({
        'probability': probabilities,
        'prediction': predictions
    })
    results.to_csv(args.output, index=False)
    
    print(f"\nPredictions saved to {args.output}")
    print(f"Total samples: {len(predictions)}")
    print(f"Positive predictions: {predictions.sum()}")
    print(f"Positive rate: {predictions.mean():.2%}")
    
    # Print distribution by probability ranges
    print("\nProbability distribution:")
    for low, high in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        count = ((probabilities >= low) & (probabilities < high)).sum()
        print(f"  {low:.1f}-{high:.1f}: {count}")


if __name__ == '__main__':
    main()
