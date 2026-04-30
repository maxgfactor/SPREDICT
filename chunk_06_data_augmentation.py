"""
Chunk 06: Data Augmentation
Fraud case augmentation and validation
"""

import numpy as np
from typing import Tuple


def augment_sparse_fraud_cases(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                              target_fraud_rate: float = 0.005,
                              max_samples: int = 50000,
                              noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment fraud cases to achieve target fraud rate
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        dates: Temporal data (n_samples,)
        target_fraud_rate: Target fraud rate (default 0.5%)
        max_samples: Maximum total samples after augmentation
        noise_std: Standard deviation of Gaussian noise for augmentation
        
    Returns:
        Tuple of (X_augmented, y_augmented, dates_augmented)
    """
    # Validate inputs
    assert isinstance(X, np.ndarray), "X must be np.ndarray"
    assert isinstance(y, np.ndarray), "y must be np.ndarray"
    assert isinstance(dates, np.ndarray), "dates must be np.ndarray"
    assert len(X) == len(y) == len(dates), "Length mismatch"
    
    fraud_rate = y.mean()
    
    # If fraud rate is already acceptable, return as-is
    if fraud_rate >= target_fraud_rate:
        return X, y, dates
    
    # Get fraud indices
    fraud_indices = np.where(y == 1)[0]
    if len(fraud_indices) == 0:
        return X, y, dates
    
    current_fraud_count = len(fraud_indices)
    target_total = int(current_fraud_count / target_fraud_rate)
    target_total = min(target_total, max_samples)
    
    if target_total <= len(y):
        return X, y, dates
    
    # Calculate how many new samples to create
    n_augment = target_total - len(y)
    
    # Create augmented fraud samples
    augmented_X = []
    augmented_y = []
    augmented_dates = []
    
    # Continue creating samples until we reach target
    created = 0
    np.random.seed(42)  # For reproducibility
    
    while created < n_augment:
        # Sample fraud cases with replacement
        sample_size = min(len(fraud_indices), n_augment - created)
        sampled_indices = np.random.choice(fraud_indices, sample_size, replace=True)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, X[sampled_indices].shape)
        
        augmented_X.append(X[sampled_indices] + noise)
        augmented_y.append(y[sampled_indices])
        augmented_dates.append(dates[sampled_indices])
        
        created += sample_size
    
    # Stack augmented data
    augmented_X = np.vstack(augmented_X)
    augmented_y = np.hstack(augmented_y)
    augmented_dates = np.hstack(augmented_dates)
    
    # Combine with original data
    X_result = np.vstack([X, augmented_X])
    y_result = np.hstack([y, augmented_y])
    dates_result = np.hstack([dates, augmented_dates])
    
    return X_result, y_result, dates_result


def validate_augmentation_output(original_X: np.ndarray, original_y: np.ndarray,
                                augmented_X: np.ndarray, augmented_y: np.ndarray) -> bool:
    """
    Validate augmentation preserved structure and improved fraud rate
    
    Args:
        original_X: Original feature matrix
        original_y: Original labels
        augmented_X: Augmented feature matrix
        augmented_y: Augmented labels
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    # Validate feature count preserved
    assert augmented_X.shape[1] == original_X.shape[1], \
        f"Feature count changed: {original_X.shape[1]} -> {augmented_X.shape[1]}"
    
    # Validate dtypes preserved
    assert augmented_X.dtype == original_X.dtype, \
        f"X dtype changed: {original_X.dtype} -> {augmented_X.dtype}"
    assert augmented_y.dtype == original_y.dtype, \
        f"y dtype changed: {original_y.dtype} -> {augmented_y.dtype}"
    
    # Validate fraud rate improved or maintained
    original_rate = original_y.mean()
    augmented_rate = augmented_y.mean()
    assert augmented_rate >= original_rate, \
        f"Fraud rate decreased: {original_rate:.4f} -> {augmented_rate:.4f}"
    
    # Validate we have equal or more samples
    assert len(augmented_X) >= len(original_X), \
        f"Sample count decreased: {len(original_X)} -> {len(augmented_X)}"
    
    # Validate all values are finite
    assert np.all(np.isfinite(augmented_X)), "Augmented X contains non-finite values"
    assert np.all(np.isfinite(augmented_y)), "Augmented y contains non-finite values"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing data augmentation...")
    
    # Create synthetic data with very low fraud rate
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)
    # Only 0.1% fraud rate
    fraud_indices = np.random.choice(n_samples, size=1, replace=False)
    y[fraud_indices] = 1
    dates = np.random.randint(20220101, 20230101, n_samples)
    
    print(f"Original: {len(X)} samples, fraud rate: {y.mean():.4f}")
    
    # Apply augmentation
    X_aug, y_aug, dates_aug = augment_sparse_fraud_cases(
        X, y, dates, target_fraud_rate=0.005, max_samples=5000
    )
    
    print(f"Augmented: {len(X_aug)} samples, fraud rate: {y_aug.mean():.4f}")
    
    # Validate output
    validate_augmentation_output(X, y, X_aug, y_aug)
    print("[PASS] Augmentation validation passed")
    
    print("\n[PASS] All augmentation tests passed")