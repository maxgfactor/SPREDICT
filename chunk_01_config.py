"""
Chunk 01: Configuration
Defines CONFIG dictionary and validation
"""

import os
from typing import Dict, Any

# Suppress TensorFlow/CUDA warnings for CPU-only execution
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# Configuration - Production Environment Only
CONFIG = {
    'DATA_PATH': 'for_train_x_2025_10_24_clean.csv',
    'USE_SAMPLING': True,          # Enable sampling for faster testing (May 5, 2026)
    'SAMPLE_SIZE': 184408,        # ~25 dates worth (~2.7% of dataset) - most recent dates (halved May 18, 2026)
    'FORCE_SAMPLING': True,       # Force this sample size
    'MIN_SAMPLES': 30,  # Reduced from 100 for clean dataset
    'TARGET_TYPE': 'continuous',  # Continuous targets (price changes) for fraud detection
    'LOG_TRANSFORM_TARGET': False,  # Disabled (May 5, 2026) - use raw ChangeY values, restore April 8 behavior
    'DATE_COLUMN_INDEX': -1,  # Auto-detect date column
    'TARGET_COLUMN_INDEX': -1,  # Auto-detect target column
    'TEMPORAL_MULTIPLIER': 9.0,
    'LOG_VERBOSITY': 2,
    'AUGMENTATION_MAX_SAMPLES': 50000,
    
    # ============================================================================
    # IMBALANCE HANDLING CONFIGURATION - Step 3
    # ============================================================================
    'DYNAMIC_CLASS_WEIGHTS': True,  # Calculate scale_pos_weight from actual class ratio
    'PREDICTION_THRESHOLD_SEARCH': False,  # Disabled - use fixed 0.5 for consistency
    'PREDICTION_THRESHOLD_MIN': 0.1,  # Start of prediction threshold search
    'PREDICTION_THRESHOLD_MAX': 0.5,  # End of prediction threshold search
    'PREDICTION_THRESHOLD_STEP': 0.05,  # Step size for prediction threshold search
    'CALIBRATE_PREDICTIONS': False,  # Apply isotonic calibration
    
    # ============================================================================
    # FEATURE ENGINEERING CONFIGURATION - Step 4
    # ============================================================================
    'WINSORIZE_FEATURES': True,  # Clip features at percentiles
    'WINSORIZE_PERCENTILE_LOW': 1,  # Lower percentile for winsorization
    'WINSORIZE_PERCENTILE_HIGH': 99,  # Upper percentile for winsorization
    'ADD_RATIO_FEATURES': True,  # Create ratio features
    'LOG_TRANSFORM_FEATURES': True,  # Apply log1p to skewed features
    'HIGHLY_SKEWED_FEATURES': [0, 1, 4, 5],  # Feature indices with high skew (from pipeline log)
    
    # ============================================================================
    # DIAGNOSTICS AND VALIDATION - New Features (Items 1, 2, 4)
    # ============================================================================
    'FEATURE_STABILITY_ANALYSIS': True,  # Track feature ranking consistency across temporal folds
    'TRACK_INFERENCE_LATENCY': True,  # Track prediction time per sample
    'SLIDING_WINDOW_VALIDATION': True,  # Enable sliding window temporal validation
    'PERMUTATION_IMPORTANCE': True,  # Run permutation importance on all models
    'MIN_DATES_THRESHOLD': 30,  # Minimum unique dates required for diagnostics (raises ValueError if below)
    'INFERENCE_LATENCY_SAMPLE_SIZE': 10000,  # Number of samples to measure for latency
    
    # ============================================================================
    # THRESHOLD CONFIGURATION - CRITICAL FOR UNDERSTANDING PIPELINE
    # ============================================================================
    # LABEL THRESHOLDS (used to create binary labels from continuous y):
    # - Target column "ChangeY" contains continuous values (0 to 32,500+)
    # - Phase 4 searches for optimal threshold that maximizes precision
    # - Labels created: y_binary = (y >= threshold).astype(int)
    # LABEL thresholds for binarizing y (NOT prediction thresholds). y_binary = (y >= thresh).
    'FIRST_THRESHOLD': 20.0,
    'LAST_THRESHOLD': 0.0,
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
    'HYPERPARAM_OPTIMIZATION_TRIALS': 30,  # Hard cap — each arch gets exactly 30 trials (May 18, 2026)
    
    # HPO Target and Continuation (May 11, 2026 / Updated May 13, 2026)
    'HPO_TARGET_PRECISION': 0.60,  # Stop HPO when precision >= this
    'HPO_CONTINUE_UNTIL_TARGET': False,  # Hard cap at HYPERPARAM_OPTIMIZATION_TRIALS trials (May 18, 2026)
    'HPO_STAGNATION_THRESHOLD': 50,  # Stop if no improvement for N trials (was 30, raised May 13, 2026)
    
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
        'Transformer': 2,        # 10% of 20
    },
    
    # Sklearn-specific safeguard overrides (May 6, 2026)
    # Lower thresholds for gradient boosting models that struggle with rare positive class
    'SKLEARN_SAFEGUARDS': {
        'MIN_PRECISION_OVER_BASELINE': 0.01,  # 1% instead of 5%
        'MIN_POSITIVE_PERCENTAGE': 0.001,  # 0.1% instead of 0.5%
        'MIN_POSITIVE_ABSOLUTE': 10,  # 10 instead of 50
    },
    
    # Neural architecture-specific safeguard overrides (May 6, 2026)
    # Lower thresholds for neural models that produce low prediction ranges
    'NEURAL_SAFEGUARDS': {
        'MIN_POSITIVE_PERCENTAGE': 0,  # Disable percentage-based, use only absolute
        'MIN_POSITIVE_ABSOLUTE': 5,  # 5 instead of 50
        'PATIENCE': 10,  # Higher patience for neural models
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
    
    # Per-architecture hyperparameter search spaces (GIS RECONFIGURED - May 13, 2026)
    # Root cause: all 6 NNs had MaxPred << 0.5 (CNN:0.004, LSTM:0.032, RNN:0.066, VAE:0.092, Transformer:0.044)
    # Trees stagnated due to small spaces + missing key params (colsample, gamma)
    'HYPERPARAM_SEARCH_SPACE': {
        # Iteration 1: Maximize phase (CatBoost already at 0.5381)
        'CatBoost': {
            'iterations': [300, 400, 500],       # was [100, 200]
            'depth': [4, 5, 6],                  # was [6, 8] — shallower to reduce overfit
            'learning_rate': [0.03, 0.05, 0.08, 0.1],  # finer granularity, lower start
            'auto_class_weights': ['Balanced', 'SqrtBalanced'],
            'l2_leaf_reg': [1, 3, 5, 7],         # was [3, 5, 10] — lower reg option
        },
        # Iteration 1: LightGBM
        'LightGBM': {
            'n_estimators': [300, 500, 800],     # was [200, 500]
            'num_leaves': [31, 63, 127],          # was [31, 63]
            'learning_rate': [0.03, 0.05, 0.08],  # added lower LR
            'scale_pos_weight': [300, 400, 500, 700],  # was [400, 500]
            'min_child_samples': [50, 100, 200],  # was [100, 200]
            'reg_alpha': [0.01, 0.1, 0.5, 1.0],  # was [0.1, 0.5]
            'reg_lambda': [0.5, 1.0, 5.0, 10.0],  # was [1.0, 5.0]
            'subsample': [0.6, 0.7, 0.8, 0.9],   # was [0.7, 0.8]
            'colsample_bytree': [0.6, 0.8, 1.0],  # NEW — feature subsampling
            'min_split_gain': [0.0, 0.01, 0.1],  # NEW — min gain to make split
        },
        # Iteration 1: XGBoost (severe overfit — Train AUC 0.88, Val AUC 0.00)
        'XGBoost': {
            'n_estimators': [100, 200, 300, 500],  # was [200, 500] — add smaller to reduce overfit
            'max_depth': [3, 5, 7],             # was [5, 7] — added shallower depth
            'learning_rate': [0.01, 0.03, 0.05, 0.1],  # lower start, finer grid
            'scale_pos_weight': [200, 400, 500],  # lower min to reduce over-prediction
            'min_child_weight': [10, 50, 100, 200],  # was [50, 100, 200]
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],  # wider L1
            'reg_lambda': [1.0, 5.0, 10.0],     # higher L2 for overfit control
            'subsample': [0.6, 0.7, 0.8],       # keep
            'colsample_bytree': [0.5, 0.7, 1.0],  # NEW — feature subsampling
            'gamma': [0, 0.1, 0.5],             # NEW — min split loss reduction
        },
        # Iteration 2: Dense (MaxPred max 1.0 but all thresholds rejected)
        'Dense': {
            'units': [64, 128, 256, 512, 1024],  # was [32, 64, 128, 256, 512]
            'layers': [2, 3, 4],                  # was [1, 2, 3] — deeper networks
            'dropout': [0.1, 0.2, 0.3, 0.4],     # was [0.05, 0.1, 0.2, 0.3]
            'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],  # finer granularity
            'epochs': [15, 20, 30, 40],          # was [8, 10, 12, 15, 20]
            'alpha': [1.0, 1.25, 1.5],           # was [0.5, 0.75, 1.0, 1.25]
            'gamma': [2.0, 2.5, 3.0, 4.0],      # was [1.0, 2.0, 2.5, 3.0, 3.5]
            'batch_size': [32, 64, 128, 256],    # NEW
            'activation': ['relu', 'leaky_relu', 'selu'],  # NEW
        },
        # Iteration 2: CNN (MaxPred max 0.0042 — completely broken)
        'CNN': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],
            'filters': [64, 128, 256, 512],     # was [32, 64, 128, 256]
            'kernel_size': [3, 5, 7, 11],        # was [3, 5, 7] — larger receptive field
            'dropout': [0.0, 0.05, 0.1, 0.2],   # allow no dropout
            'learning_rate': [0.0005, 0.001, 0.002, 0.005, 0.01],  # wider range
            'epochs': [30, 50, 80, 100],         # much more training
            'alpha': [0.75, 1.0],
            'gamma': [2.0, 2.5, 3.0],
            'layers': [1, 2, 3],                # NEW — number of conv layers
            'pooling': ['max', 'avg', 'none'],   # NEW — pooling type
        },
        # Iteration 3: RNN (MaxPred max 0.0661)
        'RNN': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],
            'units': [64, 128, 256],             # was [32, 64, 128]
            'dropout': [0.0, 0.05, 0.1],        # allow no dropout
            'learning_rate': [0.0005, 0.001, 0.002, 0.005],
            'epochs': [20, 30, 50],             # was [10, 15, 20, 30]
            'alpha': [0.75, 1.0, 1.25],
            'gamma': [2.0, 2.5, 3.0, 3.5],
            'layers': [1, 2],                    # NEW — number of RNN layers
        },
        # Iteration 3: LSTM (MaxPred max 0.0316 — completely broken)
        'LSTM': {
            'lstm_units': [32, 64, 128, 256],    # MAJOR expansion from [8, 16, 32]
            'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
            'learning_rate': [0.0005, 0.001, 0.002, 0.005],
            'epochs': [20, 30, 50],             # was [12, 15, 20, 25]
            'alpha': [0.75, 1.0],
            'gamma': [2.0, 2.5, 3.0],
            'layers': [1, 2],                    # NEW — number of LSTM layers
            'bidirectional': [True, False],      # NEW
        },
        # Iteration 4: VAE (MaxPred max 0.0919 — completely broken)
        'VAE': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],
            'latent_dim': [32, 64, 128, 256],   # added smaller dimension
            'learning_rate': [0.0005, 0.001, 0.002, 0.005],  # higher LR options
            'dropout': [0.0, 0.02, 0.05, 0.1],  # allow no dropout
            'epochs': [30, 50, 80],             # NEW — training epochs
            'alpha': [0.75, 1.0, 1.25],
            'gamma': [2.0, 2.5, 3.0],
            'encoder_layers': [1, 2, 3],         # NEW — encoder depth
            'decoder_layers': [1, 2, 3],         # NEW — decoder depth
        },
        # Iteration 4: Transformer (MaxPred max 0.0443 — very narrow)
        'Transformer': {
            'loss_function': ['binary_crossentropy'],  # keep (focal removed)
            'dim': [64, 128, 256],             # was [32, 64] — 128 restored with lower LR
            'heads': [2, 4, 8],                 # was [1, 2]
            'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
            'learning_rate': [0.00005, 0.0001, 0.0002],  # even lower for stability
            'epochs': [20, 30, 50],             # NEW — training epochs
            'alpha': [0.75, 1.0, 1.25],
            'gamma': [1.5, 2.0, 2.5, 3.0],
            'ff_dim': [64, 128, 256],           # NEW — feed-forward dimension
            'layers': [1, 2, 4],                 # NEW — transformer layers
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
    'TARGET_TYPE', 'LOG_TRANSFORM_TARGET', 'DATE_COLUMN_INDEX',
    'TARGET_COLUMN_INDEX', 'INPUT_DIM', 'AUGMENTATION_MAX_SAMPLES',
    'latent_dim', 'filters', 'units', 'dropout',
    'FIRST_THRESHOLD', 'LAST_THRESHOLD', 'THRESHOLD_STEP', 'PREDICTION_THRESHOLD',
    'ENABLE_HYPERPARAM_OPTIMIZATION', 'HYPERPARAM_OPTIMIZATION_EPOCHS', 'HYPERPARAM_OPTIMIZATION_TRIALS',
    'HPO_TARGET_PRECISION', 'HPO_CONTINUE_UNTIL_TARGET', 'HPO_STAGNATION_THRESHOLD',
    'MIN_POSITIVE_PREDICTIONS',
    # Ensemble configuration
    'ENSEMBLE_MIN_PRECISION', 'ENSEMBLE_WEIGHTING', 'FALLBACK_ARCHITECTURE',
    'MIN_POSITIVE_PERCENTAGE', 'MIN_POSITIVE_ABSOLUTE', 'FOCAL_LOSS_CONFIG',
    # Imbalance handling (Step 3)
    'DYNAMIC_CLASS_WEIGHTS', 'PREDICTION_THRESHOLD_SEARCH',
    'PREDICTION_THRESHOLD_MIN', 'PREDICTION_THRESHOLD_MAX', 'PREDICTION_THRESHOLD_STEP',
    'CALIBRATE_PREDICTIONS',
    # Feature engineering (Step 4)
    'WINSORIZE_FEATURES', 'WINSORIZE_PERCENTILE_LOW', 'WINSORIZE_PERCENTILE_HIGH',
    'ADD_RATIO_FEATURES', 'LOG_TRANSFORM_FEATURES', 'HIGHLY_SKEWED_FEATURES',
    # Diagnostics and validation (Items 1, 2, 4)
    'FEATURE_STABILITY_ANALYSIS', 'TRACK_INFERENCE_LATENCY', 'SLIDING_WINDOW_VALIDATION',
    'PERMUTATION_IMPORTANCE', 'MIN_DATES_THRESHOLD', 'INFERENCE_LATENCY_SAMPLE_SIZE',
]

# Configuration key types for validation
CONFIG_TYPES = {
    'DATA_PATH': str,
    'USE_SAMPLING': bool,
    'SAMPLE_SIZE': int,
    'FORCE_SAMPLING': bool,
    'MIN_SAMPLES': int,
    'TARGET_TYPE': str,
    'LOG_TRANSFORM_TARGET': bool,
    'DATE_COLUMN_INDEX': int,
    'TARGET_COLUMN_INDEX': int,
    'TEMPORAL_MULTIPLIER': (int, float),
    'LOG_VERBOSITY': int,
    'AUGMENTATION_MAX_SAMPLES': int,
    # Imbalance handling (Step 3)
    'DYNAMIC_CLASS_WEIGHTS': bool,
    'PREDICTION_THRESHOLD_SEARCH': bool,
    'PREDICTION_THRESHOLD_MIN': (int, float),
    'PREDICTION_THRESHOLD_MAX': (int, float),
    'PREDICTION_THRESHOLD_STEP': (int, float),
    'CALIBRATE_PREDICTIONS': bool,
    # Feature engineering (Step 4)
    'WINSORIZE_FEATURES': bool,
    'WINSORIZE_PERCENTILE_LOW': (int, float),
    'WINSORIZE_PERCENTILE_HIGH': (int, float),
    'ADD_RATIO_FEATURES': bool,
    'LOG_TRANSFORM_FEATURES': bool,
    'HIGHLY_SKEWED_FEATURES': list,
    # Diagnostics and validation (Items 1, 2, 4)
    'FEATURE_STABILITY_ANALYSIS': bool,
    'TRACK_INFERENCE_LATENCY': bool,
    'SLIDING_WINDOW_VALIDATION': bool,
    'PERMUTATION_IMPORTANCE': bool,
    'MIN_DATES_THRESHOLD': int,
    'INFERENCE_LATENCY_SAMPLE_SIZE': int,
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
    'HPO_TARGET_PRECISION': (int, float),
    'HPO_CONTINUE_UNTIL_TARGET': bool,
    'HPO_STAGNATION_THRESHOLD': int,
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
        print(f"   Total config keys: {len(CONFIG)}")
        print(f"   Required keys present: {len(REQUIRED_CONFIG_KEYS)}")
        print(f"   Data path: {CONFIG['DATA_PATH']}")
        print(f"   Sample size: {CONFIG['SAMPLE_SIZE']}")
        print(f"   Min samples: {CONFIG['MIN_SAMPLES']}")
    except ValueError as e:
        print(f"[error] Configuration validation failed: {e}")
        raise