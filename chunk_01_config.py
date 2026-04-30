"""
Chunk 01: Configuration
Defines CONFIG dictionary and validation
"""

import os
from typing import Dict, Any


# Configuration - Production Environment Only
CONFIG = {
    'DATA_PATH': 'for_train_x_2025_10_24_clean.csv',
    'USE_SAMPLING': False,  # Use entire dataset
    'SAMPLE_SIZE': 99999999,  # Use all samples (large number exceeds dataset size)
    'FORCE_SAMPLING': False,
    'MIN_SAMPLES': 30,  # Reduced from 100 for clean dataset
    'TARGET_TYPE': 'continuous',  # Continuous targets (price changes) for fraud detection
    'DATE_COLUMN_INDEX': -1,  # Auto-detect date column
    'TARGET_COLUMN_INDEX': -1,  # Auto-detect target column
    'TEMPORAL_MULTIPLIER': 9.0,
    'LOG_VERBOSITY': 2,
    'AUGMENTATION_MAX_SAMPLES': 50000,
    
    # ============================================================================
    # THRESHOLD CONFIGURATION - CRITICAL FOR UNDERSTANDING PIPELINE
    # ============================================================================
    # LABEL THRESHOLDS (used to create binary labels from continuous y):
    # - Target column "ChangeY" contains continuous values (0 to 32,500+)
    # - Phase 4 searches for optimal threshold that maximizes precision
    # - Labels created: y_binary = (y >= threshold).astype(int)
    'FIRST_THRESHOLD': 20.0,  # Start of threshold search range (for binary labels)
    'LAST_THRESHOLD': 0.0,    # End of threshold search range
    'THRESHOLD_STEP': -2.0,  # Step size (11 steps: 20.0, 18.0, ..., 0.0)
    
    # PREDICTION THRESHOLD (used to convert model outputs to binary):
    # - Model outputs probabilities (0-1) from model.predict()
    # - Binary predictions: (predictions >= 0.5).astype(int)
    # - This is the STANDARD threshold for probability outputs
    'PREDICTION_THRESHOLD': 0.5,  # Threshold for converting predictions to binary (standard sigmoid threshold)
    
    # Model Architecture
    'latent_dim': 32,
    'filters': [32, 64, 128],
    'kernel_sizes': [3, 5, 7],
    'units': 64,
    'layers': 2,
    'heads': 4,
    'dim': 64,
    'cnn_filters': 64,
    'lstm_units': 32,
    'dropout': 0.1,
    'MIN_ENSEMBLE_SIZE': 5,
    'MAX_TRAINING_ATTEMPTS': 5,
    'VERBOSE_TENSORFLOW_LOGGING': False,
    'VERBOSE_PROCESSING_LOGGING': False,
    'INPUT_DIM': 37,
    
    # Hyperparameter Optimization Configuration
    'ENABLE_HYPERPARAM_OPTIMIZATION': True,  # Enable by default
    'HYPERPARAM_OPTIMIZATION_EPOCHS': 20,
    'HYPERPARAM_OPTIMIZATION_TRIALS': 20,  # Increased for dual loss function testing
    
    # Threshold Safeguard: minimum positive predictions required to accept a threshold
    # Prevents precision gaming (predicting almost nothing → artificially high P)
    # Dynamic calculation: max(MIN_POSITIVE_ABSOLUTE, n_samples * MIN_POSITIVE_PERCENTAGE)
    'MIN_POSITIVE_PREDICTIONS': 1000,  # Legacy (fixed value, deprecated)
    'MIN_POSITIVE_PERCENTAGE': 0.005,  # 0.5% of samples (lowered for selective models)
    'MIN_POSITIVE_ABSOLUTE': 50,       # Absolute floor (lowered from 100)
    'MIN_PRECISION_OVER_BASELINE': 0.05,  # Precision must beat baseline by at least 5%
    'MIN_POS_PRED_RATIO': 0.0001,          # Min 0.01% of predictions must be positive
    'MAX_POS_PRED_RATIO': 0.70,            # Max 70% of predictions can be positive
    
    # HPO-specific thresholds (Apr 4, 2026)
    # Lower thresholds during HPO to allow more exploration for struggling architectures
    # Architecture-specific based on MaxPred capability:
    # - Dense/RNN: Working well, keep current thresholds
    # - VAE/LSTM/Transformer: Struggling, lower thresholds for more exploration
    # - CNN: Very low MaxPred, lowest thresholds
    'HPO_MIN_POSITIVE_PERCENTAGE': {
        'Dense': 0.0005,        # ~36 (10% of 355)
        'RNN': 0.0005,          # ~36 (10% of 355)
        'VAE': 0.0001,          # ~7 (10% of 71)
        'CNN': 0.00005,         # ~4 (10% of 36)
        'LSTM': 0.0001,         # ~7 (10% of 71)
        'Transformer': 0.0001,  # ~7 (10% of 71)
    },
    'HPO_MIN_POSITIVE_ABSOLUTE': {
        'Dense': 5,             # 10% of 50
        'RNN': 5,               # 10% of 50
        'VAE': 2,               # 10% of 20
        'CNN': 1,               # 10% of 10
        'LSTM': 2,              # 10% of 20
        'Transformer': 2,      # 10% of 20
    },
    
    # Post-HPO Threshold Search (Apr 5, 2026)
    # Run a second threshold search AFTER HPO to find optimal threshold for HPO model
    'ENABLE_POST_HPO_THRESHOLD_SEARCH': True,
    
    # Focal Loss Configuration (for precision-focused training)
    'USE_FOCAL_LOSS': False,  # Global default (disabled for safety - use per-arch config below)
    
    # Per-architecture Focal Loss configuration (Option B)
    # Each architecture can have custom alpha/gamma or be disabled
    'FOCAL_LOSS_CONFIG': {
        'VAE': {'enabled': True, 'alpha': 0.75, 'gamma': 1.5},
        'Dense': {'enabled': True, 'alpha': 0.5, 'gamma': 1.0},
        'CNN': {'enabled': True, 'alpha': 0.75, 'gamma': 1.5},
        'RNN': {'enabled': True, 'alpha': 0.5, 'gamma': 1.0},
        'LSTM': {'enabled': True, 'alpha': 0.75, 'gamma': 1.5},
        'Transformer': {'enabled': True, 'alpha': 0.75, 'gamma': 1.5},
    },
    
    # Legacy focal loss parameters (deprecated in favor of per-arch config)
    'FOCAL_LOSS_ALPHA': 0.5,
    'FOCAL_LOSS_GAMMA': 1.0,
    
    # Model Saving Configuration
    'SAVE_TRAINED_MODELS': True,  # Save best models after training
    'MODELS_PATH': './saved_models',  # Path to save trained models
    
    # Date Split Configuration
    # - Top 2 newest dates are always held out (Inference + Held Out)
    # - Remaining dates split by percentage for Training vs Validation
    'VAL_SPLIT_PERCENTAGE': 0.30,  # 30% of remaining dates for validation
    'TOP_DATES_HELD_OUT': 2,  # Number of newest dates to hold out
    
    # Feature Importance Analysis (Phase X)
    'FEATURE_ANALYSIS_ENABLED': True,  # Enable feature importance analysis before Phase 4
    'FEATURE_ANALYSIS_SAMPLE_SIZE': 100000,  # Subsample for faster analysis
    'FEATURE_PRUNE_PERCENTILE': 20,  # Drop bottom N% features by importance
    'FEATURE_ANALYSIS_REPORT_PATH': './feature_importance_report.txt',  # Output path
    
    # Per-architecture hyperparameter search spaces (REVISED - March 2026)
    'HYPERPARAM_SEARCH_SPACE': {
        'Dense': {
            'units': [32, 64, 128, 256, 512],  # EXPANDED: wider range (Apr 4, 2026)
            'layers': [1, 2, 3],  # EXPANDED: deeper networks (Apr 4, 2026)
            'dropout': [0.05, 0.1, 0.2, 0.3],  # EXPANDED: add lower dropout (Apr 4, 2026)
            'learning_rate': [0.0001, 0.0005, 0.001],  # EXPANDED: finer granularity (Apr 4, 2026)
            'epochs': [8, 10, 12, 15, 20],  # EXPANDED: more flexibility (Apr 4, 2026)
            'alpha': [0.5, 0.75, 1.0, 1.25],  # EXPANDED: remove 0.25 (causes TP=0), add 1.25 (Apr 4, 2026)
            'gamma': [1.0, 2.0, 2.5, 3.0, 3.5],  # EXPANDED: higher range for TP focus (Apr 4, 2026)
        },
        'VAE': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],  # NEW: dual loss
            'latent_dim': [64, 96, 128],  # NARROWED: 64-128 optimal based on HPO analysis (Apr 4, 2026)
            'learning_rate': [0.0005, 0.001, 0.0015],  # NARROWED: lower range based on HPO trials (Apr 4, 2026)
            'dropout': [0.03, 0.05, 0.07],  # NARROWED: 0.05 dominant in TP>0 trials (Apr 4, 2026)
            'alpha': [0.75, 1.0, 1.25],  # KEPT: 0.75 and 1.0 both in top performers
            'gamma': [2.0, 2.5, 3.0],  # KEPT: all work
        },
        'CNN': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],  # ADDED: focal_loss (Apr 8, 2026)
            'filters': [32, 64, 128, 256],  # EXPANDED: wider range (Apr 8, 2026)
            'kernel_size': [3, 5, 7],  # Keep same
            'dropout': [0.05, 0.1, 0.2],  # EXPANDED: higher dropout (Apr 8, 2026)
            'learning_rate': [0.001, 0.002, 0.005],  # EXPANDED: higher LR (Apr 8, 2026)
            'epochs': [20, 30, 40, 50],  # EXPANDED: more training (Apr 8, 2026)
            'alpha': [0.75, 1.0],  # Keep same
            'gamma': [2.0, 2.5, 3.0],  # Keep same
        },
        'RNN': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],  # ADDED: focal_loss (Apr 8, 2026)
            'units': [32, 64, 128],  # EXPANDED: wider range (Apr 8, 2026)
            'dropout': [0.05, 0.1, 0.15],  # EXPANDED: slightly higher (Apr 8, 2026)
            'learning_rate': [0.001, 0.002, 0.005],  # EXPANDED: higher LR (Apr 8, 2026)
            'epochs': [10, 15, 20, 30],  # EXPANDED: more training (Apr 8, 2026)
            'alpha': [0.75, 1.0, 1.25],  # Keep same
            'gamma': [2.0, 2.5, 3.0, 3.5],  # Keep same
        },
        'LSTM': {
            'loss_function': ['binary_crossentropy'],  # REMOVED: focal_loss - too aggressive
            'lstm_units': [8, 16, 32],  # REDUCED: smaller units for sharper output
            'dropout': [0.02, 0.03, 0.05, 0.1],  # EXPANDED: lower dropout for higher predictions (Apr 4, 2026)
            'learning_rate': [0.0005, 0.001, 0.002],  # EXPANDED: higher LR for prediction range (Apr 4, 2026)
            'epochs': [12, 15, 20, 25],  # EXPANDED: more training time (Apr 4, 2026)
            'alpha': [0.75, 1.0],  # TUNED: for focal_loss
            'gamma': [2.0, 2.5, 3.0],  # TUNED: higher for selectivity
        },
        'Transformer': {
            'loss_function': ['binary_crossentropy'],  # REMOVED: focal_loss - too aggressive
            'dim': [32, 64],  # NARROWED: removed 128 due to NaN failures (Apr 4, 2026)
            'heads': [1, 2],  # KEEP: single head works
            'dropout': [0.02, 0.03, 0.05, 0.1, 0.2],  # EXPANDED: lower dropout for higher predictions (Apr 4, 2026)
            'learning_rate': [0.0001, 0.0002, 0.0005],  # EXPANDED: lower LR to prevent NaN (Apr 4, 2026)
            'alpha': [0.75, 1.0, 1.25],  # EXPANDED: higher alpha for precision (Apr 4, 2026)
            'gamma': [1.5, 2.0, 2.5, 3.0],  # EXPANDED: add lower gamma values (Apr 4, 2026)
        },
        'XGBoost': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'scale_pos_weight': [100, 200, 259],
            'min_child_weight': [1, 5, 10],
            'subsample': [0.7, 0.8, 0.9],
        },
        'CatBoost': {
            'iterations': [100, 200, 500],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'auto_class_weights': ['Balanced', 'SqrtBalanced'],
            'l2_leaf_reg': [1, 3, 5],
        },
        'LightGBM': {
            'n_estimators': [100, 200, 500],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'scale_pos_weight': [100, 200, 259],
            'min_child_samples': [50, 100, 200],
            'subsample': [0.7, 0.8, 0.9],
        },
        'Boosting_Adaptive': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
        },
    },

    # Ensemble Configuration (REVISED - March 2026)
    'ENSEMBLE_MIN_PRECISION': 0.40,  # Architecture must have val_precision > 0.40
    'ENSEMBLE_WEIGHTING': 'precision_weighted',  # weight = precision_i / sum(precision)
    'ENSEMBLE_VOTE_THRESHOLD': 0.5,  # At least 2/4 agree to predict fraud
    'FALLBACK_ARCHITECTURE': 'RNN',  # Highest Phase 5 precision
}

# Required configuration keys for validation
REQUIRED_CONFIG_KEYS = [
    'DATA_PATH', 'USE_SAMPLING', 'SAMPLE_SIZE', 'MIN_SAMPLES',
    'TARGET_TYPE', 'DATE_COLUMN_INDEX',
    'TARGET_COLUMN_INDEX', 'INPUT_DIM', 'AUGMENTATION_MAX_SAMPLES',
    'latent_dim', 'filters', 'units', 'dropout',
    'FIRST_THRESHOLD', 'LAST_THRESHOLD', 'THRESHOLD_STEP', 'PREDICTION_THRESHOLD',
    'ENABLE_HYPERPARAM_OPTIMIZATION', 'HYPERPARAM_OPTIMIZATION_EPOCHS', 'HYPERPARAM_OPTIMIZATION_TRIALS',
    'MIN_POSITIVE_PREDICTIONS',
    # Ensemble configuration
    'ENSEMBLE_MIN_PRECISION', 'ENSEMBLE_WEIGHTING', 'FALLBACK_ARCHITECTURE',
    'MIN_POSITIVE_PERCENTAGE', 'MIN_POSITIVE_ABSOLUTE', 'FOCAL_LOSS_CONFIG',
]

# Configuration key types for validation
CONFIG_TYPES = {
    'DATA_PATH': str,
    'USE_SAMPLING': bool,
    'SAMPLE_SIZE': int,
    'FORCE_SAMPLING': bool,
    'MIN_SAMPLES': int,
    'TARGET_TYPE': str,
    'DATE_COLUMN_INDEX': int,
    'TARGET_COLUMN_INDEX': int,
    'TEMPORAL_MULTIPLIER': (int, float),
    'LOG_VERBOSITY': int,
    'AUGMENTATION_MAX_SAMPLES': int,
    'latent_dim': int,
    'filters': list,
    'kernel_sizes': list,
    'units': int,
    'layers': int,
    'heads': int,
    'dim': int,
    'cnn_filters': int,
    'lstm_units': int,
    'dropout': (int, float),
    'MIN_ENSEMBLE_SIZE': int,
    'MAX_TRAINING_ATTEMPTS': int,
    'VERBOSE_TENSORFLOW_LOGGING': bool,
    'VERBOSE_PROCESSING_LOGGING': bool,
    'INPUT_DIM': int,
    'FIRST_THRESHOLD': (int, float),
    'LAST_THRESHOLD': (int, float),
    'THRESHOLD_STEP': (int, float),
    'PREDICTION_THRESHOLD': (int, float),
    'ENABLE_HYPERPARAM_OPTIMIZATION': bool,
    'HYPERPARAM_OPTIMIZATION_EPOCHS': int,
    'HYPERPARAM_OPTIMIZATION_TRIALS': int,
    'MIN_POSITIVE_PREDICTIONS': int,
    'USE_FOCAL_LOSS': bool,
    'FOCAL_LOSS_ALPHA': (int, float),
    'FOCAL_LOSS_GAMMA': (int, float),
    # Ensemble configuration types
    'ENSEMBLE_MIN_PRECISION': (int, float),
    'ENSEMBLE_WEIGHTING': str,
    'FALLBACK_ARCHITECTURE': str,
    # Threshold dynamic configuration
    'MIN_POSITIVE_PERCENTAGE': (int, float),
    'MIN_POSITIVE_ABSOLUTE': int,
    # Focal loss configuration
    'FOCAL_LOSS_CONFIG': dict,
}


def validate_config_structure(config: Dict[str, Any]) -> bool:
    """
    Ensure CONFIG has all required keys with correct types
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check all required keys present
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check types
    for key, expected_type in CONFIG_TYPES.items():
        if key in config:
            value = config[key]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Config key '{key}' has wrong type. Expected {expected_type}, got {type(value)}"
                )
    
    # Validate numeric constraints
    if config['MIN_SAMPLES'] <= 0:
        raise ValueError(f"MIN_SAMPLES must be positive, got {config['MIN_SAMPLES']}")
    
    if config['SAMPLE_SIZE'] < config['MIN_SAMPLES']:
        raise ValueError(
            f"SAMPLE_SIZE ({config['SAMPLE_SIZE']}) must be >= MIN_SAMPLES ({config['MIN_SAMPLES']})"
        )
    
    if not 0 <= config['dropout'] <= 1:
        raise ValueError(f"dropout must be in [0, 1], got {config['dropout']}")
    
    if not 0 <= config['PREDICTION_THRESHOLD'] <= 1:
        raise ValueError(f"PREDICTION_THRESHOLD must be in [0, 1], got {config['PREDICTION_THRESHOLD']}")
    
    # Validate file path is string and not empty
    if not config['DATA_PATH'] or not isinstance(config['DATA_PATH'], str):
        raise ValueError(f"DATA_PATH must be non-empty string")
    
    return True


def get_config() -> Dict[str, Any]:
    """
    Get validated configuration
    
    Returns:
        Validated CONFIG dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    validate_config_structure(CONFIG)
    return CONFIG.copy()


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values and validate
    
    Args:
        updates: Dictionary of updates to apply
        
    Returns:
        Updated and validated configuration
    """
    global CONFIG
    CONFIG.update(updates)
    validate_config_structure(CONFIG)
    return CONFIG.copy()


if __name__ == "__main__":
    # Self-test
    print("Validating configuration...")
    try:
        validate_config_structure(CONFIG)
        print("[PASS] Configuration validation passed")
        print(f"   Total config keys: {len(CONFIG)}")
        print(f"   Required keys present: {len(REQUIRED_CONFIG_KEYS)}")
        print(f"   Data path: {CONFIG['DATA_PATH']}")
        print(f"   Sample size: {CONFIG['SAMPLE_SIZE']}")
        print(f"   Min samples: {CONFIG['MIN_SAMPLES']}")
    except ValueError as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        raise