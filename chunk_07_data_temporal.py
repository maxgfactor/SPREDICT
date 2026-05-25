"""
Chunk 07: Temporal Features
Temporal feature extraction and weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def extract_temporal_features(dates: np.ndarray, 
                             date_format: str = 'YYYYMMDD') -> Dict[str, np.ndarray]:
    """
    Extract temporal features from dates
    
    Args:
        dates: Array of dates (YYYYMMDD format as integers)
        date_format: Format of dates (default 'YYYYMMDD')
        
    Returns:
        Dictionary of temporal features
    """
    assert isinstance(dates, np.ndarray), "dates must be np.ndarray"
    assert dates.ndim == 1, "dates must be 1D"
    
    features = {}
    
    # Convert to pandas datetime
    dates_str = dates.astype(str)
    dates_dt = pd.to_datetime(dates_str, format='%Y%m%d', errors='coerce')
    
    # Extract components
    features['year'] = dates_dt.year.values
    features['month'] = dates_dt.month.values
    features['day'] = dates_dt.day.values
    features['dayofweek'] = dates_dt.dayofweek.values
    features['dayofyear'] = dates_dt.dayofyear.values
    features['quarter'] = dates_dt.quarter.values
    features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
    
    # Time-based features
    features['days_from_start'] = (dates_dt - dates_dt.min()).days
    features['days_to_end'] = (dates_dt.max() - dates_dt).days
    
    return features


def apply_temporal_weighting_strategy(dates: np.ndarray,
                                     strategy_config: Optional[Dict] = None) -> np.ndarray:
    """
    Apply temporal weighting strategy
    
    Args:
        dates: Array of dates (YYYYMMDD format)
        strategy_config: Configuration for weighting strategy
        
    Returns:
        Array of temporal weights (n_samples,)
    """
    if strategy_config is None:
        strategy_config = {'type': 'linear', 'multiplier': 9.0}
    
    assert isinstance(dates, np.ndarray), "dates must be np.ndarray"
    
    strategy_type = strategy_config.get('type', 'linear')
    multiplier = strategy_config.get('multiplier', 9.0)
    
    # Convert dates to numeric for calculation
    dates_numeric = pd.to_numeric(dates, errors='coerce')
    valid_mask = ~np.isnan(dates_numeric)
    
    if strategy_type == 'linear':
        # Linear decay: newer samples get higher weight
        min_date = np.min(dates_numeric[valid_mask])
        max_date = np.max(dates_numeric[valid_mask])
        date_range = max_date - min_date if max_date > min_date else 1
        
        weights = np.ones(len(dates))
        weights[valid_mask] = 1.0 + multiplier * (dates_numeric[valid_mask] - min_date) / date_range
        
    elif strategy_type == 'exponential':
        # Exponential decay
        min_date = np.min(dates_numeric[valid_mask])
        max_date = np.max(dates_numeric[valid_mask])
        date_range = max_date - min_date if max_date > min_date else 1
        
        weights = np.ones(len(dates))
        normalized_dates = (dates_numeric[valid_mask] - min_date) / date_range
        weights[valid_mask] = np.exp(multiplier * normalized_dates)
        
    else:
        # Uniform weights
        weights = np.ones(len(dates))
    
    # Ensure all weights are positive and finite
    weights = np.maximum(weights, 0.1)  # Minimum weight of 0.1
    weights = np.where(np.isfinite(weights), weights, 1.0)
    
    return weights


def apply_advanced_temporal_weighting(dates: np.ndarray,
                                     strategy_config: Optional[Dict] = None,
                                     unified_features: Optional[np.ndarray] = None,
                                     feedback_data: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Apply advanced temporal weighting with optional feature-based adjustments
    
    Args:
        dates: Array of dates
        strategy_config: Weighting strategy configuration
        unified_features: Optional additional features for weighting
        feedback_data: Optional feedback from previous phases
        
    Returns:
        Tuple of (weights, temporal_features_dict)
    """
    # Get basic temporal weights
    weights = apply_temporal_weighting_strategy(dates, strategy_config)
    
    # Extract temporal features
    temporal_features = extract_temporal_features(dates)
    
    # Add weights to temporal features
    temporal_features['weights'] = weights
    
    return weights, temporal_features


def validate_temporal_features(dates: np.ndarray, temporal_features: Dict) -> bool:
    """
    Validate temporal feature extraction
    
    Args:
        dates: Original dates array
        temporal_features: Dictionary of extracted features
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(temporal_features, dict), "temporal_features must be dict"
    
    # Check required keys
    required_keys = ['year', 'month', 'day', 'dayofweek', 'weights']
    for key in required_keys:
        assert key in temporal_features, f"Missing temporal feature: {key}"
        assert isinstance(temporal_features[key], np.ndarray), f"{key} must be np.ndarray"
        assert len(temporal_features[key]) == len(dates), f"{key} length mismatch"
    
    # Validate weights
    weights = temporal_features['weights']
    assert np.all(weights > 0), "All weights must be positive"
    assert np.all(np.isfinite(weights)), "Weights must be finite"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing temporal features...")
    
    # Create test dates
    np.random.seed(42)
    n_samples = 1000
    start_date = 20220101
    end_date = 20230101
    dates = np.random.randint(start_date, end_date, n_samples)
    
    print(f"Created {len(dates)} dates from {dates.min()} to {dates.max()}")
    
    # Test feature extraction
    features = extract_temporal_features(dates)
    print(f"[pass] Extracted {len(features)} temporal features")
    for key, value in list(features.items())[:3]:
        print(f"   {key}: shape={value.shape}, range=[{value.min()}, {value.max()}]")
    
    # Test weighting
    weights = apply_temporal_weighting_strategy(dates, {'type': 'linear', 'multiplier': 9.0})
    print(f"[pass] Generated weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    # Test advanced weighting
    weights_adv, features_dict = apply_advanced_temporal_weighting(dates)
    print(f"[pass] Advanced weighting: {len(features_dict)} features returned")
    
    # Validate
    validate_temporal_features(dates, features_dict)
    
    print("\n[pass] All temporal feature tests passed")