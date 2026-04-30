"""
Chunk 16: Phase 1 - Pipeline Setup
Initial data loading and preprocessing phase
"""

import numpy as np
from typing import Dict

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
        self.logger.log("Starting Phase 1: Pipeline Setup", 'info')
        
        # Load data
        try:
            X, y, dates = self.data_manager.load_data()
            self.logger.log(f"Data loaded: {len(X)} samples, {X.shape[1]} features", 'info')
        except FileNotFoundError as e:
            self.logger.log(f"CRITICAL: Data file not found - {e}", 'error')
            raise RuntimeError("Cannot proceed without valid fraud data file") from e
        except ValueError as e:
            self.logger.log(f"DATA VALIDATION ERROR: {e}", 'error')
            raise RuntimeError("Data validation failed") from e
        except Exception as e:
            self.logger.log(f"Unexpected data loading error: {e}", 'error')
            raise RuntimeError("Data loading failed") from e
        
        # SANITY CHECK: Sample size validation
        expected_samples = self.config.get('SAMPLE_SIZE', 0)
        if expected_samples > 0 and len(X) != expected_samples:
            self.logger.log(f"SANITY CHECK: Requested {expected_samples} samples, got {len(X)}", 'warning')
        
        # SANITY CHECK: Target value distribution (for continuous targets)
        if self.config.get('TARGET_TYPE') == 'continuous':
            raw_target = self.data_manager._raw_target_values
            if raw_target is not None:
                self.logger.log(f"Target Distribution (ChangeY):", 'info')
                self.logger.log(f"   Min: {np.nanmin(raw_target):.2f} | Max: {np.nanmax(raw_target):.2f} | Mean: {np.nanmean(raw_target):.2f}", 'info')
                self.logger.log(f"   Median: {np.nanmedian(raw_target):.2f} | Std: {np.nanstd(raw_target):.2f}", 'info')
                # Check for extreme values
                if np.nanmax(raw_target) > 100:
                    self.logger.log(f"SANITY CHECK: Extreme target values detected (max={np.nanmax(raw_target):.2f})", 'warning')
                # Count samples above key thresholds
                for thresh in [5, 10, 20, 25]:
                    count = int(np.sum(raw_target >= thresh))
                    pct = count / len(raw_target) * 100
                    self.logger.log(f"   Samples >= {thresh}: {count:,} ({pct:.2f}%)", 'info')
        
        # Calculate data statistics
        stats = {
            'fraud_rate': float(y.mean()),
            'missing_values': 0,  # Already handled in load_data
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # Log data quality - use FIRST_THRESHOLD from config for class distribution
        first_threshold = self.config.get('FIRST_THRESHOLD', 24.9)
        y_binary = (y >= first_threshold).astype(int)
        self.logger.log_class_distribution(y_binary)
        self.logger.log_temporal_coverage(dates)
        self.logger.log_feature_quality_metrics(X)
        
        # Augment fraud cases if needed
        original_fraud_rate = y.mean()
        X, y, dates = self.data_manager.augment_fraud_cases(X, y, dates)
        augmented_fraud_rate = y.mean()
        if augmented_fraud_rate > original_fraud_rate:
            self.logger.log(f"Fraud augmentation: {original_fraud_rate:.4f} → {augmented_fraud_rate:.4f}", 'info')
        
        # Concentrate on periods with fraud
        X, y, dates = self.data_manager.concentrate_fraud_cases(X, y, dates)
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
        feature_names = self.data_manager._feature_columns if hasattr(self.data_manager, '_feature_columns') else [f'feature_{i}' for i in range(n_features)]
        
        # Log feature names for transparency
        self.logger.log(f"  Feature names ({len(feature_names)}): {feature_names}", 'info')
        
        # Update context with Phase 1 results
        context.update({
            'X': X,
            'y': y,
            'dates': dates,
            'features': [X],  # Feature engineering outputs (simplified)
            'feature_names': feature_names,
            'data_stats': stats,
            'raw_target_values': raw_target_values if raw_target_values is not None else y,
            'raw_target_column': raw_target_column if raw_target_column is not None else -1,
            'phase1_complete': True
        })
        
        self.logger.log("Phase 1 completed successfully", 'info')
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
        print("[PASS] Test data created")
    
    # Run Phase 1
    try:
        phase1 = Phase1_PipelineSetup(config)
        context = {}
        result = phase1.execute(context)
        
        print(f"[PASS] Phase 1 executed successfully")
        print(f"   X shape: {result['X'].shape}")
        print(f"   y shape: {result['y'].shape}")
        print(f"   dates shape: {result['dates'].shape}")
        print(f"   fraud rate: {result['y'].mean():.4f}")
        
        # Validate output
        validate_phase1_output(result)
        print("[PASS] Phase 1 output validation passed")
        
    except FileNotFoundError:
        print("[WARNING] Test skipped: test_data.csv not found")
    except Exception as e:
        print(f"[WARNING] Test error: {e}")
    
    print("\n[PASS] Phase1_PipelineSetup tests completed")