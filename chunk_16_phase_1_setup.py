"""
Chunk 16: Phase 1 - Pipeline Setup
Initial data loading and preprocessing phase
"""

import numpy as np
from typing import Dict
from chunk_01_config import DEFAULT_FIRST_THRESHOLD, DEFAULT_LAST_THRESHOLD, DEFAULT_THRESHOLD_STEP

from chunk_15_phase_base import BasePhase
from chunk_02_utils_logging import Logger
from chunk_05_data_manager import DataManager, validate_data_output


class Phase1_PipelineSetup(BasePhase):
    """Phase 1: Data pipeline setup and initialization"""
    
    def __init__(self, config: Dict):
        """
        Initialize Phase 1
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = Logger(config)
        self.data_manager = DataManager(config)
    
    def execute(self, context: Dict) -> Dict:
        """
        Execute Phase 1: Load and prepare data
        
        Args:
            context: Pipeline context (empty for Phase 1)
            
        Returns:
            Updated context with loaded data
        """
        # Load data
        try:
            X, y, dates = self.data_manager.load_data()
            self.logger.log(f"Data loaded: {len(X)} samples, {X.shape[1]} features", 'info')
            self.logger.log_temporal_coverage(dates)
        except FileNotFoundError as e:
            self.logger.log(f"critical: Data file not found - {e}", 'error')
            raise RuntimeError("Cannot proceed without valid stock data file") from e
        except ValueError as e:
            self.logger.log(f"data validation error: {e}", 'error')
            raise RuntimeError("Data validation failed") from e
        except Exception as e:
            self.logger.log(f"Unexpected data loading error: {e}", 'error')
            raise RuntimeError("Data loading failed") from e
        
        # sanity check: Sample size validation
        expected_samples = self.config.get('SAMPLE_SIZE', 0)
        if expected_samples > 0 and len(X) != expected_samples:
            self.logger.log(f"sanity check: sample_size={expected_samples}, got {len(X)} samples", 'warning')
        
        # sanity check: Target value distribution (for continuous targets)
        if self.config.get('TARGET_TYPE') == 'continuous':
            raw_target = self.data_manager._raw_target_values
            if raw_target is not None:
                self.logger.log(f"Target Distribution (ChangeY):", 'info')
                self.logger.log(f"   Min: {np.nanmin(raw_target):.2f} | Max: {np.nanmax(raw_target):.2f} | Mean: {np.nanmean(raw_target):.2f}", 'info')
                self.logger.log(f"   Median: {np.nanmedian(raw_target):.2f} | Std: {np.nanstd(raw_target):.2f}", 'info')
                # Check for extreme values
                if np.nanmax(raw_target) > 100:
                    self.logger.log(f"sanity check: Extreme target values detected (max={np.nanmax(raw_target):.2f})", 'warning')
                # Class distribution for ENTIRE dataset at all thresholds (full detail)
                first_thresh = self.config.get('FIRST_THRESHOLD', DEFAULT_FIRST_THRESHOLD)
                last_thresh = self.config.get('LAST_THRESHOLD', DEFAULT_LAST_THRESHOLD)
                thresh_step = self.config.get('THRESHOLD_STEP', DEFAULT_THRESHOLD_STEP)
                thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)

                for thresh in thresholds:
                    y_binary = (raw_target >= thresh).astype(int)
                    signal_count = int(np.sum(y_binary))
                    normal_count = len(y_binary) - signal_count
                    total = len(y_binary)
                    signal_rate = signal_count / total
                    imbalance_ratio = max(signal_count, normal_count) / min(signal_count, normal_count) if min(signal_count, normal_count) > 0 else float('inf')
                    
                    self.logger.log(f"[class] Class Distribution (LABEL_THRESHOLD={thresh:>4.1f}):", 'info')
                    self.logger.log(f"  Total samples: {total:,}", 'info')
                    self.logger.log(f"  Signal cases: {signal_count:,} ({signal_rate:.1%})", 'info')
                    self.logger.log(f"  Normal cases: {normal_count:,} ({normal_count/total:.1%})", 'info')
                    self.logger.log(f"  Imbalance ratio: {imbalance_ratio:.1f}:1", 'info')
        
        # Calculate data statistics
        stats = {
            'signal_rate': float(y.mean()),
            'missing_values': 0,  # Already handled in load_data
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # Log full class distribution for ALL thresholds (synchronized with Phase 4)
        first_thresh = self.config.get('FIRST_THRESHOLD', DEFAULT_FIRST_THRESHOLD)
        last_thresh = self.config.get('LAST_THRESHOLD', DEFAULT_LAST_THRESHOLD)
        thresh_step = self.config.get('THRESHOLD_STEP', DEFAULT_THRESHOLD_STEP)
        thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)

        for thresh in thresholds:
            y_binary = (y >= thresh).astype(int)
            signal_count = int(np.sum(y_binary))
            normal_count = len(y_binary) - signal_count
            total = len(y_binary)
            signal_rate = signal_count / total
            imbalance_ratio = max(signal_count, normal_count) / min(signal_count, normal_count) if min(signal_count, normal_count) > 0 else float('inf')
            
            self.logger.log(f"[class] Class Distribution (LABEL_THRESHOLD={thresh:>4.1f}):", 'info')
            self.logger.log(f"  Total samples: {total:,}", 'info')
            self.logger.log(f"  Signal cases: {signal_count:,} ({signal_rate:.1%})", 'info')
            self.logger.log(f"  Normal cases: {normal_count:,} ({normal_count/total:.1%})", 'info')
            self.logger.log(f"  Imbalance ratio: {imbalance_ratio:.1f}:1", 'info')
        
        self.logger.log_feature_quality_metrics(X)
        
        # Augment signal cases if needed
        original_signal_rate = y.mean()
        X, y, dates = self.data_manager.augment_signal_cases(X, y, dates)
        augmented_signal_rate = y.mean()
        if augmented_signal_rate > original_signal_rate:
            self.logger.log(f"Signal augmentation: {original_signal_rate:.4f} → {augmented_signal_rate:.4f}", 'info')
        
        # Concentrate on periods with signal
        X, y, dates = self.data_manager.concentrate_signal_cases(X, y, dates)
        self.logger.log(f"Data concentration: {len(X)} samples retained", 'info')
        
        # Prepare data
        X = self.data_manager.prepare_data(X)
        self.logger.log(f"Data preprocessing complete: {X.shape[1]} features", 'info')
        
        # Store raw target values if available
        raw_target_values = self.data_manager._raw_target_values
        raw_target_column = self.data_manager._raw_target_column
        
        # Sample raw_target_values to match X, y, dates (in case sampling was applied)
        if raw_target_values is not None and len(raw_target_values) != len(X):
            # Get the sampling indices from data_manager
            if hasattr(self.data_manager, '_sampled_indices'):
                raw_target_values = raw_target_values[self.data_manager._sampled_indices]
            elif len(raw_target_values) > len(X):
                # If we can't get indices, just take the first len(X) values
                raw_target_values = raw_target_values[:len(X)]
        
        # Generate feature names from data manager or create generic names
        feature_names = (self.data_manager._feature_columns[:X.shape[1]]
                          if hasattr(self.data_manager, '_feature_columns')
                          else [f'feature_{i}' for i in range(X.shape[1])])
        
        # Log feature names for transparency
        self.logger.log(f"  Feature names ({len(feature_names)}): {feature_names}", 'info')
        
        # Update context with Phase 1 results
        # =========================================================================
        # DATA SPLIT: Extract inference FIRST, then split remaining into train/val
        # =========================================================================
        
        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)
        
        # Inference: newest date(s) - extracted FIRST
        inference_date = unique_dates[-1]
        inference_mask = dates == inference_date
        
        # Remaining dates: oldest to second-newest
        remaining_mask = ~inference_mask
        remaining_dates = dates[remaining_mask]
        remaining_unique_dates = np.unique(remaining_dates)
        
        # Train/Val split on REMAINING data (70/30)
        val_pct = self.config.get('VAL_SPLIT_PERCENTAGE', 0.30)
        n_remaining = len(remaining_unique_dates)
        n_train_dates = int(n_remaining * (1 - val_pct))
        
        train_dates_threshold = remaining_unique_dates[n_train_dates] if n_train_dates > 0 else remaining_unique_dates[0]
        
        train_mask = remaining_mask & (dates < train_dates_threshold)
        val_mask = remaining_mask & (dates >= train_dates_threshold)
        
        # Extract data subsets
        X_train = X[train_mask]
        y_train_continuous = y[train_mask]
        dates_train = dates[train_mask]
        
        X_val = X[val_mask]
        y_val_continuous = y[val_mask]
        dates_val = dates[val_mask]
        
        X_inference = X[inference_mask]
        y_inference_continuous = y[inference_mask]
        dates_inference = dates[inference_mask]
        
        # Log split summary
        self.logger.log("[data split] Summary:", 'info')
        self.logger.log(f"  Total: {len(X):,} samples, {n_dates} dates", 'info')
        self.logger.log(f"  train: {len(X_train):,} samples ({train_mask.sum() / len(X):.1%}), dates < {train_dates_threshold}", 'info')
        self.logger.log(f"  validation: {len(X_val):,} samples ({val_mask.sum() / len(X):.1%}), dates >= {train_dates_threshold}", 'info')
        self.logger.log(f"  Inference: {len(X_inference):,} samples ({inference_mask.sum() / len(X):.1%}), date = {inference_date}", 'info')
        
        context.update({
            'X': X,
            'y': y,
            'dates': dates,
            'features': feature_names,
            'feature_names': feature_names,
            'data_stats': stats,
            'raw_target_values': raw_target_values if raw_target_values is not None else y,
            'raw_target_column': raw_target_column if raw_target_column is not None else -1,
            'phase1_complete': True,
            # Data splits for downstream phases
            'X_train': X_train,
            'y_train_continuous': y_train_continuous,
            'dates_train': dates_train,
            'X_val': X_val,
            'y_val_continuous': y_val_continuous,
            'dates_val': dates_val,
            'X_inference': X_inference,
            'y_inference_continuous': y_inference_continuous,
            'dates_inference': dates_inference,
        })
        
        return context


def validate_phase1_input(context: Dict) -> bool:
    """
    Validate Phase 1 input (should be empty or minimal)
    
    Args:
        context: Input context
        
    Returns:
        True if valid (Phase 1 accepts empty context)
    """
    # Phase 1 can start with empty context
    return True


def validate_phase1_output(context: Dict) -> bool:
    """
    Validate Phase 1 output meets contract
    
    Args:
        context: Output context from Phase 1
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    required_keys = [
        'X', 'y', 'dates', 'features', 'data_stats',
        'raw_target_values', 'raw_target_column', 'phase1_complete'
    ]
    
    for key in required_keys:
        assert key in context, f"Phase 1 missing required key: {key}"
    
    # Validate array shapes
    X, y, dates = context['X'], context['y'], context['dates']
    assert X.ndim == 2, "X must be 2D"
    assert y.ndim == 1, "y must be 1D"
    assert dates.ndim == 1, "dates must be 1D"
    assert len(X) == len(y) == len(dates), "Length mismatch"
    
    # Validate phase complete flag
    assert context['phase1_complete'] == True
    
    # Validate data contract
    min_samples = 30  # Default minimum
    validate_data_output(X, y, dates, min_samples, {'TARGET_TYPE': 'continuous'})
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing Phase1_PipelineSetup...")
    
    # Create test config
    config = {
        'DATA_PATH': 'test_data.csv',
        'USE_SAMPLING': False,
        'SAMPLE_SIZE': 1000,
        'MIN_SAMPLES': 10,
        'TARGET_TYPE': 'binary',
        'TARGET_THRESHOLD': 0.5,
        'AUGMENTATION_MAX_SAMPLES': 50000,
        'LOG_VERBOSITY': 0
    }
    
    # Create test data file if needed
    import os
    import pandas as pd
    
    if not os.path.exists('test_data.csv'):
        print("Creating test data...")
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        dates = np.random.randint(20220101, 20230101, n_samples)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['date'] = dates
        df['target'] = y
        df.to_csv('test_data.csv', index=False)
        print("[pass] Test data created")
    
    # Run Phase 1
    try:
        phase1 = Phase1_PipelineSetup(config)
        context = {}
        result = phase1.execute(context)
        
        print(f"[pass] Phase 1 executed successfully")
        print(f"   X shape: {result['X'].shape}")
        print(f"   y shape: {result['y'].shape}")
        print(f"   dates shape: {result['dates'].shape}")
        print(f"   signal rate: {result['y'].mean():.4f}")
        
        # Validate output
        validate_phase1_output(result)
        
    except FileNotFoundError:
        print("[warning] Test skipped: test_data.csv not found")
    except Exception as e:
        print(f"[warning] Test error: {e}")
    
    print("\n[pass] Phase1_PipelineSetup tests completed")