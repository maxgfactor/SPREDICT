"""
Chunk 05: Data Manager
Data loading, validation, and preprocessing
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataManager:
    """Manages data loading, validation, and preprocessing"""
    
    def __init__(self, config: Dict):
        """
        Initialize DataManager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._raw_target_values = None
        self._raw_target_column = None
        self._sampled_indices = None  # Store sampled indices for Phase 1
    
    def load_data(self, winsorize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and comprehensively validate stock data CSV file
        
        Args:
            winsorize: If True, apply global winsorization (current behavior).
                      If False, skip winsorization (per-arch mode in Phase 4).
        
        Returns:
            Tuple of (X, y, dates) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Binary labels (n_samples,)
            - dates: Temporal data (n_samples,) in YYYYMMDD format
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data validation fails
        """
        data_path = self.config['DATA_PATH']
        
        # Strict requirement: file must exist
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CRITICAL: Stock data file not found at '{data_path}'\n"
                f"This file is required for the stock analysis system to operate.\n"
                f"Please ensure the CSV file exists at the specified path.\n"
                f"Expected format: CSV with columns including date and target (signal indicator)"
            )
        
        # Load and validate CSV structure
        return self._load_and_validate_csv(data_path, winsorize=winsorize)
    
    def _load_and_validate_csv(self, data_path: str, winsorize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Comprehensive CSV loading and validation
        
        Args:
            data_path: Path to CSV file
            winsorize: If True, apply global winsorization in _apply_feature_engineering
            
        Returns:
            Tuple of (X, y, dates)
        """
        # Load CSV
        df = pd.read_csv(data_path)
        
        # Flexible sample count
        min_samples = self.config['MIN_SAMPLES']
        if len(df) < min_samples:
            raise ValueError(f"Insufficient data: {len(df)} samples found, minimum {min_samples} required")
        
        # Check for required columns
        if df.shape[1] < 3:  # Need at least features + date + target
            raise ValueError(f"Insufficient columns: {df.shape[1]} found, need at least 3")
        
        # Auto-detect date and target columns
        date_col_idx = self._detect_date_column(df)
        target_col_idx = self.config['TARGET_COLUMN_INDEX']
        
        # Convert negative index to actual index (e.g., -1 -> last column index)
        if target_col_idx < 0:
            target_col_idx = df.shape[1] + target_col_idx
        
        # Extract data components
        dates = pd.to_numeric(df.iloc[:, date_col_idx], errors='coerce').values
        raw_target = df.iloc[:, target_col_idx].values
        
        # Create feature matrix (exclude date and target columns)
        feature_cols = [i for i in range(df.shape[1]) if i not in [date_col_idx, target_col_idx]]
        X = df.iloc[:, feature_cols].apply(pd.to_numeric, errors='coerce').values
        
        # Handle continuous targets - keep as raw/continuous for threshold optimization
        if self.config['TARGET_TYPE'] == 'continuous':
            # Keep target as raw/continuous values (don't convert to binary)
            y = raw_target  # Keep continuous
            self._raw_target_values = raw_target
            self._raw_target_column = df.columns[target_col_idx]
        else:
            y = raw_target.astype(int)
            self._raw_target_column = df.columns[target_col_idx]
        
        # Apply log transform if configured (Step 1 - Option C: sign + magnitude)
        if self.config['LOG_TRANSFORM_TARGET']:
            sign = np.sign(y)
            magnitude = np.abs(y)
            y = sign * np.log1p(magnitude)
            self._log_transform_applied = True
        else:
            self._log_transform_applied = False
        if hasattr(self, 'logger') and self.logger:
            self.logger.log(f"   [feature] Log-transformed target (LOG_TRANSFORM_TARGET={self.config['LOG_TRANSFORM_TARGET']})", 'info')
        
        # Store feature column names (excluding date and target)
        self._feature_columns = [df.columns[i] for i in feature_cols]
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(dates) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        dates = dates[valid_mask].astype(int)
        
        # Apply stratified sampling if configured
        if self.config['USE_SAMPLING']:
            X, y, dates = self._apply_stratified_sampling(X, y, dates)
        
        # Apply feature engineering (Step 4)
        X = self._apply_feature_engineering(X, winsorize=winsorize)
        
        # Validate data contract
        self._validate_data_output(X, y, dates, min_samples, self.config)
        
        return X, y, dates
    
    def is_log_transform_applied(self) -> bool:
        """Check if log transform was applied to target variable."""
        return getattr(self, '_log_transform_applied', False)
    
    def _apply_feature_engineering(self, X: np.ndarray, winsorize: bool = True) -> np.ndarray:
        """
        Apply feature engineering: winsorization, ratio features, log transform.
        
        Args:
            X: Feature matrix
            winsorize: If True and WINSORIZE_FEATURES is enabled, apply global winsorization.
                      If False, skip winsorization (for per-arch mode).
            
        Returns:
            Transformed feature matrix
        """
        if X is None or len(X) == 0:
            return X
        
        original_features = X.shape[1]
        
        # Step 4a: Winsorize features (skipped when winsorize=False for per-arch mode)
        if winsorize and self.config['WINSORIZE_FEATURES']:
            low_pct = self.config['WINSORIZE_PERCENTILE_LOW']
            high_pct = self.config['WINSORIZE_PERCENTILE_HIGH']
            for col in range(X.shape[1]):
                p_low, p_high = np.percentile(X[:, col], [low_pct, high_pct])
                X[:, col] = np.clip(X[:, col], p_low, p_high)
            print(f"   [feature] Winsorized {X.shape[1]} features at {low_pct}/{high_pct} percentiles")
        
        # Step 4c: Log-transform highly skewed features
        if self.config['LOG_TRANSFORM_FEATURES']:
            skewed_indices = self.config['HIGHLY_SKEWED_FEATURES']
            transformed_count = 0
            for idx in skewed_indices:
                if idx < X.shape[1]:
                    X[:, idx] = np.sign(X[:, idx]) * np.log1p(np.abs(X[:, idx]))
                    transformed_count += 1
            if transformed_count > 0:
                print(f"   [feature] Log-transformed {transformed_count} skewed features: columns {skewed_indices}")
        
        # Step 4b: Add ratio features (must be after winsorization and log-transform)
        if self.config['ADD_RATIO_FEATURES'] and self._feature_columns:
            new_features = []
            added_names = []
            feature_names = self._feature_columns
            
            try:
                if 'Price' in feature_names and '52W_High' in feature_names:
                    price_idx = feature_names.index('Price')
                    high_52w_idx = feature_names.index('52W_High')
                    ratio_1 = X[:, price_idx] / (X[:, high_52w_idx] + 1e-8)
                    new_features.append(ratio_1)
                    added_names.append('Price_to_52W_High')
                
                if 'Volume' in feature_names and 'Avg_Volume' in feature_names:
                    vol_idx = feature_names.index('Volume')
                    avg_vol_idx = feature_names.index('Avg_Volume')
                    ratio_2 = X[:, vol_idx] / (X[:, avg_vol_idx] + 1e-8)
                    new_features.append(ratio_2)
                    added_names.append('Volume_to_Avg_Volume')
                
                if 'Price' in feature_names and '52W_Low' in feature_names:
                    price_idx = feature_names.index('Price')
                    low_52w_idx = feature_names.index('52W_Low')
                    ratio_3 = X[:, price_idx] / (X[:, low_52w_idx] + 1e-8)
                    new_features.append(ratio_3)
                    added_names.append('Price_to_52W_Low')
                
                if new_features:
                    X = np.column_stack([X] + new_features)
                    self._feature_columns.extend(added_names)
                    print(f"   [feature] Added {len(new_features)} ratio features")
            except Exception as e:
                print(f"   [feature] Warning: Could not add ratio features: {e}")
        
        if X.shape[1] > original_features:
            print(f"   [feature] Total features: {original_features} -> {X.shape[1]}")
        
        return X
    
    def _detect_date_column(self, df: pd.DataFrame) -> int:
        """
        Detect date column index
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Index of date column
        """
        # Try to find a column with dates in YYYYMMDD format
        for col_idx in range(df.shape[1]):
            try:
                values = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
                if values.min() > 19000000 and values.max() < 21000000:
                    return col_idx
            except Exception:
                continue
        
        # Default to second-to-last column
        return -2
    
    def _apply_stratified_sampling(self, X: np.ndarray, y: np.ndarray, 
                                  dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply stratified sampling to maintain signal rate distribution
        
        Args:
            X: Feature matrix
            y: Labels
            dates: Temporal data
            
        Returns:
            Sampled (X, y, dates)
        """
        sample_size = self.config['SAMPLE_SIZE']
        total_samples = len(y)
        
        if total_samples <= sample_size:
            return X, y, dates
        
        # Sort by date descending (most recent dates first)
        sorted_indices = np.argsort(dates)[::-1]  # Descending order
        sampled_indices = sorted_indices[:sample_size]
        
        # Store sampled indices for Phase 1
        self._sampled_indices = sampled_indices
        
        return X[sampled_indices], y[sampled_indices], dates[sampled_indices]
    
    def augment_signal_cases(self, X: np.ndarray, y: np.ndarray, 
                            dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augment signal cases to balance dataset
        
        Args:
            X: Feature matrix
            y: Labels
            dates: Temporal data
            
        Returns:
            Augmented (X, y, dates)
        """
        signal_rate = y.mean()
        
        # If signal rate is acceptable, return as-is
        if signal_rate >= 0.001:  # At least 0.1% signal
            return X, y, dates
        
        # Augment signal cases
        signal_indices = np.where(y == 1)[0]
        if len(signal_indices) == 0:
            return X, y, dates
        
        max_samples = self.config['AUGMENTATION_MAX_SAMPLES']
        target_signal_rate = 0.005  # 0.5% signal rate
        
        current_signal_count = len(signal_indices)
        target_total = int(current_signal_count / target_signal_rate)
        target_total = min(target_total, max_samples)
        
        if target_total <= len(y):
            return X, y, dates
        
        # Create augmented signal samples with small noise
        n_augment = target_total - len(y)
        n_repeats = (n_augment // len(signal_indices)) + 1
        
        augmented_X = []
        augmented_y = []
        augmented_dates = []
        
        for _ in range(n_repeats):
            if len(augmented_X) >= n_augment:
                break
            
            noise = np.random.normal(0, 0.01, X[signal_indices].shape)
            augmented_X.append(X[signal_indices] + noise)
            augmented_y.append(y[signal_indices])
            augmented_dates.append(dates[signal_indices])
        
        augmented_X = np.vstack(augmented_X)[:n_augment]
        augmented_y = np.hstack(augmented_y)[:n_augment]
        augmented_dates = np.hstack(augmented_dates)[:n_augment]
        
        # Combine with original
        X = np.vstack([X, augmented_X])
        y = np.hstack([y, augmented_y])
        dates = np.hstack([dates, augmented_dates])
        
        return X, y, dates
    
    def concentrate_signal_cases(self, X: np.ndarray, y: np.ndarray, 
                                dates: np.ndarray, min_signal_per_date: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Concentrate on periods with signal activity
        
        Args:
            X: Feature matrix
            y: Labels
            dates: Temporal data
            min_signal_per_date: Minimum signal cases per date (0 = no filtering)
            
        Returns:
            Filtered (X, y, dates)
        """
        if min_signal_per_date == 0:
            return X, y, dates
        
        # Count signal per date
        unique_dates = np.unique(dates)
        dates_with_signal = []
        
        for date in unique_dates:
            mask = dates == date
            if np.sum(y[mask]) >= min_signal_per_date:
                dates_with_signal.append(date)
        
        if len(dates_with_signal) == 0:
            return X, y, dates
        
        # Filter to dates with sufficient signal
        mask = np.isin(dates, dates_with_signal)
        return X[mask], y[mask], dates[mask]
    
    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for modeling (scaling, type conversion)
        
        Args:
            X: Feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        # Convert to float32 for TensorFlow
        X = X.astype(np.float32)
        
        # Handle any remaining NaN values
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        return X
    
    def _validate_data_output(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray, 
                             min_samples: int, config: Dict = None) -> None:
        """
        Validate data loading output
        
        Args:
            X: Feature matrix
            y: Labels
            dates: Temporal data
            min_samples: Minimum required samples
            config: Optional config to check TARGET_TYPE
            
        Raises:
            ValueError: If validation fails
        """
        assert isinstance(X, np.ndarray), "X must be np.ndarray"
        assert isinstance(y, np.ndarray), "y must be np.ndarray"
        assert isinstance(dates, np.ndarray), "dates must be np.ndarray"
        
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
        assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
        assert dates.ndim == 1, f"dates must be 1D, got {dates.ndim}D"
        
        assert len(X) == len(y) == len(dates), "Length mismatch between X, y, dates"
        assert len(X) >= min_samples, f"Insufficient samples: {len(X)} < {min_samples}"
        
        # Validate y is binary (only when TARGET_TYPE is not 'continuous')
        target_type = (config or {}).get('TARGET_TYPE', 'binary')
        if target_type != 'continuous':
            unique_y = np.unique(y)
            assert set(unique_y).issubset({0, 1}), f"y must be binary, got {unique_y}"
        
        # Validate dates
        assert dates.min() > 19000000, f"Dates too old: {dates.min()}"
        assert dates.max() < 21000000, f"Dates too futuristic: {dates.max()}"


def validate_data_output(X: np.ndarray, y: np.ndarray, dates: np.ndarray, 
                        min_samples: int = 30, config: Dict = None) -> bool:
    """
    Validate the standard data contract (X, y, dates)
    
    Args:
        X: Feature matrix
        y: Labels
        dates: Temporal data
        min_samples: Minimum required samples
        config: Optional config to check TARGET_TYPE
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Validate types and dimensions
    assert isinstance(X, np.ndarray), "X must be np.ndarray"
    assert isinstance(y, np.ndarray), "y must be np.ndarray"
    assert isinstance(dates, np.ndarray), "dates must be np.ndarray"
    
    assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
    assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
    assert dates.ndim == 1, f"dates must be 1D, got {dates.ndim}D"
    
    # Validate consistent lengths
    if not (len(X) == len(y) == len(dates)):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, dates={len(dates)}")
    
    # Validate minimum samples
    if len(X) < min_samples:
        raise ValueError(f"Insufficient samples: {len(X)} < {min_samples}")
    
    # Validate y is binary (only when TARGET_TYPE is not 'continuous')
    target_type = (config or {}).get('TARGET_TYPE', 'binary')
    if target_type != 'continuous':
        unique_y = np.unique(y)
        if not set(unique_y).issubset({0, 1}):
            raise ValueError(f"y must be binary (0, 1), got {unique_y}")
    
    # Validate dates are reasonable (YYYYMMDD format)
    if dates.min() < 19000000:
        raise ValueError(f"Dates too old: {dates.min()}")
    if dates.max() > 21000000:
        raise ValueError(f"Dates too futuristic: {dates.max()}")
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing DataManager...")
    
    # Create test config
    config = {
        'DATA_PATH': 'test_data.csv',
        'USE_SAMPLING': False,
        'SAMPLE_SIZE': 1000,
        'MIN_SAMPLES': 10,
        'TARGET_TYPE': 'binary',
        'TARGET_THRESHOLD': 0.5,
        'AUGMENTATION_MAX_SAMPLES': 50000
    }
    
    # Test with synthetic data if file doesn't exist
    if not os.path.exists('test_data.csv'):
        print("Creating synthetic test data...")
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        dates = np.random.randint(20220101, 20230101, n_samples)
        
        # Save as CSV
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['date'] = dates
        df['target'] = y
        df.to_csv('test_data.csv', index=False)
        print("[pass] Test data created")
    
    # Test data loading
    try:
        data_manager = DataManager(config)
        X, y, dates = data_manager.load_data()
        
        print(f"[pass] Data loaded: X={X.shape}, y={y.shape}, dates={dates.shape}")
        print(f"   Signal rate: {y.mean():.3f}")
        print(f"   Date range: {dates.min()} to {dates.max()}")
        
        # Validate output
        validate_data_output(X, y, dates, config['MIN_SAMPLES'], config)
        
    except FileNotFoundError as e:
        print(f"[warning] File not found (expected if test_data.csv missing): {e}")
    
    print("\n[pass] DataManager tests completed")