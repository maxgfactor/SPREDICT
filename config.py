"""
config.py — Master Configuration
Refactored from chunk_01_config.py (2026-08-07).
Defines CONFIG dictionary and validation.
"""

import os
from typing import Dict, Any

# Threshold search defaults (used as fallback across all phases)
PREDICTION_THRESHOLD_DEFAULT = 0.5  # fallback if config key is missing

# Suppress TensorFlow/CUDA warnings for CPU-only execution
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# Configuration - Production Environment Only
CONFIG = {
    'DATA_PATH': 'sampled_184408.csv',
    'USE_SAMPLING': True,          # Enable sampling for faster testing (May 5, 2026)
    'SAMPLE_SIZE': 184408,        # Max rows to keep — pipeline caps to min(SAMPLE_SIZE, actual)
    'USE_TEMPORAL_WEIGHTING': True,   # ADDED — gates TemporalWeighting + Inference temporal recomputation
    'RANDOM_SEED': 42,                # ADDED — anchors all RNG for reproducibility
    'MIN_SAMPLES': 30,  # Reduced from 100 for clean dataset
    'TARGET_TYPE': 'continuous',  # Continuous targets (price changes) for stock analysis
    'LOG_TRANSFORM_TARGET': False,  # Disabled (May 5, 2026) - use raw ChangeY values, restore April 8 behavior
    'TARGET_COLUMN_INDEX': -1,  # Auto-detect target column
    'TEMPORAL_MULTIPLIER': 3.0,
    'LOG_VERBOSITY': 2,
    'AUGMENTATION_MAX_SAMPLES': 50000,

    # ============================================================================
    # IMBALANCE HANDLING CONFIGURATION - Step 3
    # ============================================================================
    'DYNAMIC_CLASS_WEIGHTS': True,  # Calculate scale_pos_weight from actual class ratio
    'PREDICTION_THRESHOLD_SEARCH': True,  # Enabled (full-feature test) - search optimal prediction threshold
    'PREDICTION_THRESHOLD_MIN': 0.1,  # Start of prediction threshold search
    'PREDICTION_THRESHOLD_MAX': 0.5,  # End of prediction threshold search
    'PREDICTION_THRESHOLD_STEP': 0.05,  # Step size for prediction threshold search
    # XGBoost precision targeting (independent flag — does not activate PREDICTION_THRESHOLD_SEARCH)
    'PREDICTION_XGBOOST_PRECISION_TARGETING': True,
    'PREDICTION_COVERAGE_RATES': [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50],
    'PREDICTION_TARGET_PRECISION': 0.60,
    'PREDICTION_MAX_COVERAGE': 0.50,

    # ============================================================================
    # FEATURE ENGINEERING CONFIGURATION - Step 4
    # ============================================================================
    'WINSORIZE_FEATURES': True,  # Clip features at percentiles
    'WINSORIZE_PERCENTILE_LOW': 3,  # Lower percentile for winsorization (A1 — tighter left-tail clipping)
    'WINSORIZE_PERCENTILE_HIGH': 97,  # A2 — relaxed right-tail clip (was 95); per-arch winsorization may be needed later
    'PER_ARCH_WINSORIZE': {
        'RNN':  {'low': 3, 'high': 97},
        'Dense':  {'low': 3, 'high': 92},
        'LSTM':  {'low': 3, 'high': 92},
        'CNN':   {'low': 3, 'high': 95},
        'VAE':  {'low': 3, 'high': 97},
        'Transformer':  {'low': 3, 'high': 95},
        'CatBoost':  {'low': 1, 'high': 99},
        'LightGBM':  {'low': 1, 'high': 99},
        'XGBoost':  {'low': 3, 'high': 97},
    },
    'ADD_RATIO_FEATURES': True,  # Create ratio features
    'LOG_TRANSFORM_FEATURES': True,  # Apply log1p to skewed features
    'HIGHLY_SKEWED_FEATURES': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20],  # Log1p all skewed features (excludes Ticker_id=6, RSI_14=17)
    'LOG_TRANSFORM_RATIO_FEATURES': True,  # Log1p ratio features after computation (extreme skew 200+)
    'POLY_INTERACTIONS': [
        ('52W_Low', 'SMA20'),              # rank#1 × rank#6
        ('Change', 'Volume'),               # rank#3 × rank#8
        ('Price_to_52W_High', 'Perf_Week'), # rank#1 × rank#5
        ('52W_Low', 'Price_to_Prev_Close'), # rank#2 × rank#7
    ],
    'BIN_RSI': True,  # Discretize RSI_14 into 4 zones (oversold/neutral/overbought)

    # ============================================================================
    # DIAGNOSTICS AND VALIDATION - New Features (Items 1, 2, 4)
    # ============================================================================
    'FEATURE_STABILITY_ANALYSIS': True,  # Track feature ranking consistency across temporal folds
    'TRACK_INFERENCE_LATENCY': True,  # Track prediction time per sample
    'SLIDING_WINDOW_VALIDATION': True,  # Enable sliding window temporal validation
    'PERMUTATION_IMPORTANCE': True,  # Run permutation importance on all models
    'MIN_DATES_THRESHOLD': 30,  # Minimum unique dates required for diagnostics (raises ValueError if below)
    'INFERENCE_LATENCY_SAMPLE_SIZE': 10000,  # Number of samples to measure for latency
    'STABILITY_RF_ESTIMATORS': 10,   # ADDED — random forest estimators for feature stability analysis

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
    'THRESHOLD_STEP': -5.0,  # 5 thresholds: 20, 15, 10, 5, 0

    # PREDICTION THRESHOLD (used to convert model outputs to binary):
    # - Model outputs probabilities (0-1) from model.predict()
    # - Binary predictions: (predictions >= 0.5).astype(int)
    # - This is the STANDARD threshold for probability outputs
    'PREDICTION_THRESHOLD': 0.5,  # Threshold for converting predictions to binary
    'ZERO_DIVISION_MODE': 0,  # sklearn zero_division: 0=honest (TP=0→0.0), 1=hide (TP=0→1.0)
    'BASELINE_EPOCHS': 3,  # Epochs for section 1 baseline (was 1 — ensures meaningful predictions at high label thresholds)

    # Model Architecture
    'latent_dim': 32,
    'units': 64,
    'layers': 2,
    'heads': 4,
    'dim': 64,
    'kernel_size': 5,
    'filters': 64,
    'lstm_units': 32,
    'dropout': 0.1,
    'activation': 'relu',
    'bidirectional': False,
    'pooling': 'global_avg',
    'encoder_layers': 2,
    'decoder_layers': 3,
    'ff_dim': 128,

    # Per-architecture default learning rates (used for baseline, threshold search, and fallback):
    # These should align with the FLOOR of each architecture's HPO search space.
    'DEFAULT_LEARNING_RATES': {
        'VAE': 0.0005,
        'CNN': 0.001,       # was 0.0001 — matches HPO floor
        'RNN': 0.001,
        'LSTM': 0.001,      # was 0.0001 — matches HPO floor and RNN
        'Dense': 0.001,
        'Transformer': 0.0001,  # Transformer HPO range is [0.00005, 0.0002]
    },

    # Architecture subset — empty list = run all architectures
    'ACTIVE_ARCHITECTURES': [],
    # Architecture classification groups (used for scaler/safeguard gating)
    'NEURAL_ARCHITECTURES': ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer'],
    'TREE_ARCHITECTURES': ['CatBoost', 'LightGBM', 'XGBoost'],
    'ARCH_CSV_ORDER': ['CatBoost', 'LightGBM', 'XGBoost', 'VAE', 'Dense', 'CNN', 'RNN', 'LSTM', 'Transformer'],
    # Per-architecture epoch overrides (two contexts: fast HPO retrain vs final training)
    'HPO_RETRAIN_EPOCHS': {'Dense': 15, 'VAE': 20, 'CNN': 15, 'RNN': 15, 'LSTM': 15, 'Transformer': 15},
    'FINAL_TRAIN_EPOCHS': {'Dense': 15, 'VAE': 30, 'CNN': 15, 'RNN': 15, 'LSTM': 15, 'Transformer': 15},
    'MIN_ENSEMBLE_SIZE': 5,
    # XGBoost Freeze (GIS §10) — skip HPO, use iter10 winning params directly
    'XGBOOST_FROZEN_PARAMS': {
        'skip_hpo': False,
        'feature_kept_indices': None,
        'hyperparams': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'min_child_weight': 200,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'subsample': 0.6,
            'colsample_bytree': 0.7,
            'gamma': 0.5,
        },
    },
    'INPUT_DIM': 37,

    # ============================================================================
    # AUGMENTATION CONFIGURATION
    # ============================================================================
    'AUGMENTATION_MIN_SIGNAL_RATE': 0.001,       # ADDED — minimum signal rate for augmentation
    'AUGMENTATION_TARGET_SIGNAL_RATE': 0.005,    # ADDED — target signal rate for augmentation
    'AUGMENTATION_NOISE_STD': 0.01,              # ADDED — noise std for augmentation

    # ============================================================================
    # KL ANNEALING CONFIGURATION
    # ============================================================================
    'KL_WARMUP_EPOCHS': 10,                      # ADDED — KL annealing warmup epochs
    'KL_MAX_WEIGHT': 1.0,                        # ADDED — max KL weight for annealing
    'KL_SAMPLING_MAX_WEIGHT': 0.1,               # ADDED — KL max weight override for sampling layer

    # Hyperparameter Optimization Configuration
    'ENABLE_HYPERPARAM_OPTIMIZATION': True,  # Enable by default
    'HYPERPARAM_OPTIMIZATION_EPOCHS': 5,
    'HYPERPARAM_OPTIMIZATION_TRIALS': 3,  # Hard cap — each arch gets exactly 3 trials (reduced from 5, Jul 11, 2026)

    # HPO Target and Continuation (May 11, 2026 / Updated May 13, 2026)
    'HPO_TARGET_PRECISION': 0.60,  # Stop HPO when precision >= this
    'HPO_CONTINUE_UNTIL_TARGET': False,  # Hard cap at HYPERPARAM_OPTIMIZATION_TRIALS trials (May 18, 2026)
    'HPO_STAGNATION_THRESHOLD': 50,  # Stop if no improvement for N trials (was 30, raised May 13, 2026)

    # HPO trial rejection gates (ADDED — literals formerly hardcoded in chunk_21)
    'HPO_DEGENERACY_STD_THRESHOLD': 0.005,       # ADDED — degeneracy gate for HPO trial rejection
    'HPO_RECALL_GATE_MARGIN': 0.01,              # ADDED — recall-based rejection margin in HPO
    'HPO_RNN_TP_FLOOR': 100,                     # ADDED — RNN-specific TP floor in HPO
    'HPO_LOSS_REDUCTION_THRESHOLD': 0.05,        # ADDED — INERT (literal lives only in dead assess_learning/assess_model_learning); kept for config fidelity
    'HPO_PRECISION_IMPROVEMENT_MIN_SHORT': 0.001,  # ADDED — INERT (same dead-code source); kept for config fidelity
    'HPO_PRECISION_IMPROVEMENT_MIN_LONG': 0.005,   # ADDED — INERT (same dead-code source); kept for config fidelity
    'NN_LOG_TP_ARCHS': ['Dense', 'CNN', 'RNN', 'LSTM', 'Transformer'],  # ADDED — archs using log(TP) in HPO scoring

    # Threshold Safeguard: minimum positive predictions required to accept a threshold
    # Prevents precision gaming (predicting almost nothing → artificially high P)
    # Dynamic calculation: max(MIN_POSITIVE_ABSOLUTE, n_samples * MIN_POSITIVE_PERCENTAGE)
    'MIN_PRECISION_OVER_BASELINE': 0.02,  # Precision must beat baseline by at least 2% (GIS Tier 2 — raised from 1%)
    'MIN_POS_PRED_RATIO': 0.0005,          # A4 — relaxed from 0.1% to 0.05% (pre-Tier-2 was 0.01%)
    'MAX_POS_PRED_RATIO': 0.65,            # A4 — relaxed from 60% to 65% (pre-Tier-2 was 70%)

    # HPO-specific thresholds (Apr 4, 2026)
    # Lower thresholds during HPO to allow more exploration for struggling architectures
    # Architecture-specific based on MaxPred capability:
    # - Dense/RNN: Working well, keep current thresholds
    # - VAE/LSTM/Transformer: Struggling, lower thresholds for more exploration
    # - CNN: Very low MaxPred, lowest thresholds
    # Sklearn-specific safeguard overrides (May 6, 2026)
    # Lower thresholds for gradient boosting models that struggle with rare positive class
    'SKLEARN_SAFEGUARDS': {
        'MIN_PRECISION_OVER_BASELINE': 0.01,  # 1% (relaxed for sklearn)
        'MIN_POSITIVE_PERCENTAGE': 0.001,  # 0.1% (relaxed for sklearn)
        'MIN_POSITIVE_ABSOLUTE': 10,  # 10 (relaxed for sklearn)
        'MAX_POS_PRED_RATIO': 1.0,  # Disable pos_pred_ratio cap for sklearn (allows L_THRESHOLD=0.0)
    },

    # Neural architecture-specific safeguard overrides (May 6, 2026)
    # Lower thresholds for neural models that produce low prediction ranges
    'NEURAL_SAFEGUARDS': {
        'MIN_POSITIVE_PERCENTAGE': 0,  # Disable percentage-based, use only absolute
        'MIN_POSITIVE_ABSOLUTE': 5,  # 5 (lower floor for neural models)
        'PATIENCE': 10,  # Higher patience for neural models
        'MIN_PRECISION_OVER_BASELINE': 0.01,  # 1% (relaxed for neural — matches SKLEARN_SAFEGUARDS)
    },
    'PATIENCE': 10,  # Top-level key for direct access (matching NEURAL_SAFEGUARDS value)

    # Post-HPO Threshold Search (Apr 5, 2026)
    # Run a second threshold search AFTER HPO to find optimal threshold for HPO model
    'ENABLE_POST_HPO_THRESHOLD_SEARCH': True,
    'POST_HPO_THRESHOLD_PATIENCE': 999,  # No early stopping during post-HPO threshold sweep (999 ≈ unlimited)

    # Focal Loss Configuration (for precision-focused training)
    'USE_FOCAL_LOSS': True,  # Global default (enabled for full-feature test - use per-arch config below)

    # Per-architecture Focal Loss configuration (Option B)
    # Each architecture can have custom alpha/gamma or be disabled
    'FOCAL_LOSS_CONFIG': {
        'VAE': {'enabled': True, 'alpha': 0.75, 'gamma': 1.5},
        'Dense': {'enabled': True, 'alpha': 0.25, 'gamma': 2.0},
        'CNN': {'enabled': True, 'alpha': 0.25, 'gamma': 2.0},
        'RNN': {'enabled': True, 'alpha': 0.25, 'gamma': 2.0},
        'LSTM': {'enabled': True, 'alpha': 0.25, 'gamma': 2.0},
        'Transformer': {'enabled': True, 'alpha': 0.25, 'gamma': 2.0},
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
    'FEATURE_IMPORTANCE_METHODS': {    # Individual method toggles — disabled methods skipped entirely
        'correlation': True,
        'tree': True,
        'permutation': True,
        'neural': True,
        'shap': True,
        'ablation': True,
    },
    'FEATURE_ANALYSIS_SAMPLE_SIZE': 100000,  # Subsample for faster analysis
    'FEATURE_PRUNE_PERCENTILE': 0,  # Keep all features — let XGBoost split-select
    'FEATURE_ANALYSIS_REPORT_PATH': './feature_importance_report.txt',  # Output path
    'FI_TRAIN_EPOCHS': 10,   # ADDED — feature importance training epochs
    'FI_BATCH_SIZE': 256,    # ADDED — feature importance training batch size

    # Feature importance analysis internals (moved from CONFIG_FEATURE_ANALYSIS)
    'CORRELATION_THRESHOLDS': [0.0, 0.5, 1.0, 2.0],
    'TREE_ESTIMATORS': 200,
    'PERMUTATION_REPEATS': 5,
    'SHAP_SAMPLE_SIZE': 5000,

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
            'num_leaves': [31, 63],               # was [31, 63, 127] — 127 needs >5 HPO epochs
            'learning_rate': [0.03, 0.05, 0.08],  # added lower LR
            'min_child_samples': [50, 100, 200],  # was [100, 200]
            'reg_alpha': [0.01, 0.1, 0.5],        # was [0.01, 0.1, 0.5, 1.0] — 1.0 triggers degeneracy with other reg
            'reg_lambda': [0.5, 1.0, 5.0],        # was [0.5, 1.0, 5.0, 10.0] — 10.0 same
            'subsample': [0.6, 0.7, 0.8, 0.9],   # was [0.7, 0.8]
            'colsample_bytree': [0.6, 0.8, 1.0],  # NEW — feature subsampling
            'min_split_gain': [0.0, 0.01],        # was [0.0, 0.01, 0.1] — 0.1 too high for 5-epoch HPO
        },
        # Iteration 1: XGBoost (severe overfit — Train AUC 0.88, Val AUC 0.00)
        'XGBoost': {
            'n_estimators': [100, 200, 300, 500],  # was [200, 500] — add smaller to reduce overfit
            'max_depth': [3, 5, 7],             # was [5, 7] — added shallower depth
            'learning_rate': [0.01, 0.03, 0.05, 0.1],  # lower start, finer grid
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
            'loss_function': ['binary_crossentropy', 'focal_loss'],
            'alpha': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],  # expanded: Tier 3 FocalLoss
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
            'alpha': [0.25, 0.5, 0.75, 1.0],
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
            'alpha': [0.25, 0.5, 0.75, 1.0, 1.25],
            'gamma': [2.0, 2.5, 3.0, 3.5],
            'layers': [1, 2],                    # NEW — number of RNN layers
        },
        # Iteration 3: LSTM (MaxPred max 0.0316 — completely broken)
        'LSTM': {
            'lstm_units': [32, 64, 128, 256],    # MAJOR expansion from [8, 16, 32]
            'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
            'learning_rate': [0.0005, 0.001, 0.002, 0.005],
            'epochs': [20, 30, 50],             # was [12, 15, 20, 25]
            'loss_function': ['binary_crossentropy', 'focal_loss'],
            'alpha': [0.25, 0.5, 0.75, 1.0],
            'gamma': [2.0, 2.5, 3.0],
            'layers': [1, 2],                    # NEW — number of LSTM layers
            'bidirectional': [True, False],      # NEW
        },
        # Iteration 4: VAE (MaxPred max 0.0919 — completely broken)
        'VAE': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],
            'latent_dim': [32, 64],   # capped at 64 — higher dims cause latent collapse (std_pred < 0.0006)
            'learning_rate': [0.0005, 0.001, 0.002, 0.005],  # higher LR options
            'dropout': [0.0, 0.02, 0.05, 0.1],  # allow no dropout
            'epochs': [30, 50, 80],             # NEW — training epochs
            'alpha': [0.25, 0.5, 0.75, 1.0, 1.25],
            'gamma': [2.0, 2.5, 3.0],
            'encoder_layers': [1, 2, 3],         # NEW — encoder depth
            'decoder_layers': [1, 2, 3],         # NEW — decoder depth
        },
        # Iteration 4: Transformer (MaxPred max 0.0443 — very narrow)
        'Transformer': {
            'loss_function': ['binary_crossentropy', 'focal_loss'],  # Tier 3: focal restored
            'dim': [64, 128, 256],             # was [32, 64] — 128 restored with lower LR
            'heads': [2, 4, 8],                 # was [1, 2]
            'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
            'learning_rate': [0.00005, 0.0001, 0.0002],  # even lower for stability
            'epochs': [20, 30, 50],             # NEW — training epochs
            'alpha': [0.25, 0.5, 0.75, 1.0, 1.25],
            'gamma': [1.5, 2.0, 2.5, 3.0],
            'ff_dim': [64, 128, 256],           # NEW — feed-forward dimension
            'layers': [1, 2, 4],                 # NEW — transformer layers
        },
    },

    # Ensemble Configuration (REVISED - March 2026)
    'ENSEMBLE_MIN_PRECISION': 0.50,  # Architecture must have val_precision > 0.50 (was 0.53 — relaxed to include RNN 0.5290, LightGBM 0.5246)
    'ENSEMBLE_WEIGHTING': 'uniform',  # Uniform averaging (GIS Tier 1 — prevents CatBoost dominance)
    'FALLBACK_ARCHITECTURE': 'VAE',  # Highest val precision (P=0.5416, GIS Tier 1)
    # Temporal Precision Gap Analysis (Phase Xb)
    'TEMPORAL_GAP_N_DAYS': 3,        # Number of unique dates in each tail (overrides FRACTION if > 0)
    'TEMPORAL_GAP_TAIL_FRACTION': 0.33,  # Fraction fallback if N_DAYS <= 0
    'TEMPORAL_GAP_SIGNIFICANCE': 0.05,   # ADDED — temporal precision gap significance threshold
    'TREE_EARLY_STOPPING_ROUNDS': 10,  # Early stopping patience for XGBoost/LightGBM/CatBoost

    # Metric percentage thresholds (ADDED — pct-above metric)
    'METRIC_PCT_THRESHOLDS': [0.01, 0.02, 0.05, 0.10, 0.20, 0.50],

    # Backward Elimination Configuration (Phase BE)
    'BACKWARD_ELIMINATION_ENABLED': True,  # Master toggle — enabled for iter23
    'BE_PROXY_TRAIN_EPOCHS': 10,            # Training epochs for proxy models
    'BE_PROXY_ENSEMBLE_SIZE': 1,            # Proxy ensemble size (1=single, >1=ensemble vote)
    'BE_STRATIFY_SPLIT_RATIO': 0.20,        # Validation fraction for elimination loop
    'BE_ELIMINATION_STEPS': 0.50,           # Fraction of bottom features to drop each iteration
    'BE_MIN_FEATURES': 10,                  # Safety floor — never prune below N features
    'BE_TOLERANCE': 0.01,                   # Max fractional val precision drop per step
}

# Required configuration keys for validation
REQUIRED_CONFIG_KEYS = [
    'DATA_PATH', 'USE_SAMPLING', 'SAMPLE_SIZE', 'MIN_SAMPLES',
    'TARGET_TYPE', 'LOG_TRANSFORM_TARGET',
    'TARGET_COLUMN_INDEX', 'INPUT_DIM', 'AUGMENTATION_MAX_SAMPLES',
    'latent_dim', 'units', 'dropout',
    'FIRST_THRESHOLD', 'LAST_THRESHOLD', 'THRESHOLD_STEP', 'PREDICTION_THRESHOLD', 'ZERO_DIVISION_MODE', 'BASELINE_EPOCHS',
    'ENABLE_HYPERPARAM_OPTIMIZATION', 'HYPERPARAM_OPTIMIZATION_EPOCHS', 'HYPERPARAM_OPTIMIZATION_TRIALS',
    'HPO_TARGET_PRECISION', 'HPO_CONTINUE_UNTIL_TARGET', 'HPO_STAGNATION_THRESHOLD',
    # Ensemble configuration
    'ENSEMBLE_MIN_PRECISION', 'ENSEMBLE_WEIGHTING', 'FALLBACK_ARCHITECTURE',
    'FOCAL_LOSS_CONFIG', 'PATIENCE',
    # Imbalance handling (Step 3)
    'DYNAMIC_CLASS_WEIGHTS', 'PREDICTION_THRESHOLD_SEARCH',
    'PREDICTION_THRESHOLD_MIN', 'PREDICTION_THRESHOLD_MAX', 'PREDICTION_THRESHOLD_STEP',
    'PREDICTION_XGBOOST_PRECISION_TARGETING',
    'PREDICTION_COVERAGE_RATES', 'PREDICTION_TARGET_PRECISION', 'PREDICTION_MAX_COVERAGE',
    # Feature engineering (Step 4)
    'WINSORIZE_FEATURES', 'WINSORIZE_PERCENTILE_LOW', 'WINSORIZE_PERCENTILE_HIGH',
    'PER_ARCH_WINSORIZE',
    'ADD_RATIO_FEATURES', 'LOG_TRANSFORM_FEATURES', 'HIGHLY_SKEWED_FEATURES',
    'LOG_TRANSFORM_RATIO_FEATURES', 'POLY_INTERACTIONS', 'BIN_RSI',
    # Model, path, temporal, split keys (defensive coverage)
    'MODELS_PATH', 'SAVE_TRAINED_MODELS', 'TEMPORAL_MULTIPLIER',
    'VAL_SPLIT_PERCENTAGE',
    # Feature importance analysis
    'FEATURE_ANALYSIS_ENABLED', 'FEATURE_IMPORTANCE_METHODS',
    'FEATURE_ANALYSIS_SAMPLE_SIZE', 'FEATURE_PRUNE_PERCENTILE',
    'FEATURE_ANALYSIS_REPORT_PATH',
    # Diagnostics and validation (Items 1, 2, 4)
    'FEATURE_STABILITY_ANALYSIS', 'TRACK_INFERENCE_LATENCY', 'SLIDING_WINDOW_VALIDATION',
    'PERMUTATION_IMPORTANCE', 'MIN_DATES_THRESHOLD', 'INFERENCE_LATENCY_SAMPLE_SIZE',
    # Feature importance analysis internals
    'CORRELATION_THRESHOLDS', 'TREE_ESTIMATORS',
    'PERMUTATION_REPEATS', 'SHAP_SAMPLE_SIZE',
    # Additional global configs (defensive registration)
    'LOG_VERBOSITY',
    'ACTIVE_ARCHITECTURES', 'NEURAL_ARCHITECTURES', 'TREE_ARCHITECTURES', 'ARCH_CSV_ORDER', 'HPO_RETRAIN_EPOCHS', 'FINAL_TRAIN_EPOCHS', 'XGBOOST_FROZEN_PARAMS',
    'layers', 'heads', 'dim', 'kernel_size', 'filters', 'lstm_units',
    'MIN_ENSEMBLE_SIZE',
    'USE_FOCAL_LOSS', 'FOCAL_LOSS_ALPHA', 'FOCAL_LOSS_GAMMA',
    'MIN_PRECISION_OVER_BASELINE', 'MIN_POS_PRED_RATIO', 'MAX_POS_PRED_RATIO',
    'SKLEARN_SAFEGUARDS', 'NEURAL_SAFEGUARDS',
    'ENABLE_POST_HPO_THRESHOLD_SEARCH', 'POST_HPO_THRESHOLD_PATIENCE', 'TOP_DATES_HELD_OUT',
    'HYPERPARAM_SEARCH_SPACE',
    'TEMPORAL_GAP_N_DAYS', 'TEMPORAL_GAP_TAIL_FRACTION',
    'TREE_EARLY_STOPPING_ROUNDS',
    # Backward Elimination (Phase BE)
    'BACKWARD_ELIMINATION_ENABLED',
    'BE_PROXY_TRAIN_EPOCHS',
    'BE_PROXY_ENSEMBLE_SIZE',
    'BE_STRATIFY_SPLIT_RATIO',
    'BE_ELIMINATION_STEPS',
    'BE_MIN_FEATURES',
    'BE_TOLERANCE',
    # Refactor additions (2026-08-07)
    'USE_TEMPORAL_WEIGHTING', 'RANDOM_SEED',
    'HPO_DEGENERACY_STD_THRESHOLD', 'HPO_RECALL_GATE_MARGIN', 'HPO_RNN_TP_FLOOR',
    'HPO_LOSS_REDUCTION_THRESHOLD', 'HPO_PRECISION_IMPROVEMENT_MIN_SHORT',
    'HPO_PRECISION_IMPROVEMENT_MIN_LONG',
    'AUGMENTATION_MIN_SIGNAL_RATE', 'AUGMENTATION_TARGET_SIGNAL_RATE', 'AUGMENTATION_NOISE_STD',
    'KL_WARMUP_EPOCHS', 'KL_MAX_WEIGHT', 'KL_SAMPLING_MAX_WEIGHT',
    'TEMPORAL_GAP_SIGNIFICANCE', 'STABILITY_RF_ESTIMATORS',
    'FI_TRAIN_EPOCHS', 'FI_BATCH_SIZE', 'METRIC_PCT_THRESHOLDS', 'NN_LOG_TP_ARCHS',
]

# Configuration key types for validation
CONFIG_TYPES = {
    'DATA_PATH': str,
    'USE_SAMPLING': bool,
    'SAMPLE_SIZE': int,
    'MIN_SAMPLES': int,
    'TARGET_TYPE': str,
    'LOG_TRANSFORM_TARGET': bool,
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
    'PREDICTION_XGBOOST_PRECISION_TARGETING': bool,
    'PREDICTION_COVERAGE_RATES': list,
    'PREDICTION_TARGET_PRECISION': (int, float),
    'PREDICTION_MAX_COVERAGE': (int, float),
    # Feature engineering (Step 4)
    'WINSORIZE_FEATURES': bool,
    'WINSORIZE_PERCENTILE_LOW': (int, float),
    'WINSORIZE_PERCENTILE_HIGH': (int, float),
    'PER_ARCH_WINSORIZE': dict,
    'ADD_RATIO_FEATURES': bool,
    'LOG_TRANSFORM_FEATURES': bool,
    'HIGHLY_SKEWED_FEATURES': list,
    'LOG_TRANSFORM_RATIO_FEATURES': bool,
    'POLY_INTERACTIONS': list,
    'BIN_RSI': bool,
    # Diagnostics and validation (Items 1, 2, 4)
    'FEATURE_STABILITY_ANALYSIS': bool,
    'TRACK_INFERENCE_LATENCY': bool,
    'SLIDING_WINDOW_VALIDATION': bool,
    'PERMUTATION_IMPORTANCE': bool,
    'MIN_DATES_THRESHOLD': int,
    'INFERENCE_LATENCY_SAMPLE_SIZE': int,
    'latent_dim': int,
    'units': int,
    'layers': int,
    'heads': int,
    'dim': int,
    'kernel_size': int,
    'filters': int,
    'lstm_units': int,
    'ACTIVE_ARCHITECTURES': list,
    'XGBOOST_FROZEN_PARAMS': dict,
    'NEURAL_ARCHITECTURES': list,
    'TREE_ARCHITECTURES': list,
    'ARCH_CSV_ORDER': list,
    'HPO_RETRAIN_EPOCHS': dict,
    'FINAL_TRAIN_EPOCHS': dict,
    'dropout': (int, float),
    'MIN_ENSEMBLE_SIZE': int,
    'INPUT_DIM': int,
    'FIRST_THRESHOLD': (int, float),
    'LAST_THRESHOLD': (int, float),
    'THRESHOLD_STEP': (int, float),
    'PREDICTION_THRESHOLD': (int, float),
    'ZERO_DIVISION_MODE': int,
    'BASELINE_EPOCHS': int,
    'ENABLE_HYPERPARAM_OPTIMIZATION': bool,
    'HYPERPARAM_OPTIMIZATION_EPOCHS': int,
    'HYPERPARAM_OPTIMIZATION_TRIALS': int,
    'HPO_TARGET_PRECISION': (int, float),
    'HPO_CONTINUE_UNTIL_TARGET': bool,
    'HPO_STAGNATION_THRESHOLD': int,
    'USE_FOCAL_LOSS': bool,
    'FOCAL_LOSS_ALPHA': (int, float),
    'FOCAL_LOSS_GAMMA': (int, float),
    # Ensemble configuration types
    'ENSEMBLE_MIN_PRECISION': (int, float),
    'ENSEMBLE_WEIGHTING': str,
    'FALLBACK_ARCHITECTURE': str,
    # Focal loss configuration
    'FOCAL_LOSS_CONFIG': dict,
    # Additional validated keys
    'PATIENCE': int,
    'SAVE_TRAINED_MODELS': bool,
    'MODELS_PATH': str,
    'VAL_SPLIT_PERCENTAGE': (int, float),
    'FEATURE_ANALYSIS_ENABLED': bool,
    'FEATURE_IMPORTANCE_METHODS': dict,
    'FEATURE_ANALYSIS_SAMPLE_SIZE': int,
    'FEATURE_PRUNE_PERCENTILE': (int, float),
    'FEATURE_ANALYSIS_REPORT_PATH': str,
    # Feature importance analysis internals
    'CORRELATION_THRESHOLDS': list,
    'TREE_ESTIMATORS': int,
    'PERMUTATION_REPEATS': int,
    'SHAP_SAMPLE_SIZE': int,
    # Additional safeguard and container types
    'MIN_PRECISION_OVER_BASELINE': (int, float),
    'MIN_POS_PRED_RATIO': (int, float),
    'MAX_POS_PRED_RATIO': (int, float),
    'SKLEARN_SAFEGUARDS': dict,
    'NEURAL_SAFEGUARDS': dict,
    'ENABLE_POST_HPO_THRESHOLD_SEARCH': bool,
    'POST_HPO_THRESHOLD_PATIENCE': int,
    'TOP_DATES_HELD_OUT': int,
    'HYPERPARAM_SEARCH_SPACE': dict,
    'TEMPORAL_GAP_N_DAYS': int,
    'TEMPORAL_GAP_TAIL_FRACTION': float,
    'TREE_EARLY_STOPPING_ROUNDS': int,
    # Backward Elimination (Phase BE)
    'BACKWARD_ELIMINATION_ENABLED': bool,
    'BE_PROXY_TRAIN_EPOCHS': int,
    'BE_PROXY_ENSEMBLE_SIZE': int,
    'BE_STRATIFY_SPLIT_RATIO': (int, float),
    'BE_ELIMINATION_STEPS': (int, float),
    'BE_MIN_FEATURES': int,
    'BE_TOLERANCE': (int, float),
    # Refactor additions (2026-08-07)
    'USE_TEMPORAL_WEIGHTING': bool,
    'RANDOM_SEED': int,
    'HPO_DEGENERACY_STD_THRESHOLD': (int, float),
    'HPO_RECALL_GATE_MARGIN': (int, float),
    'HPO_RNN_TP_FLOOR': int,
    'HPO_LOSS_REDUCTION_THRESHOLD': (int, float),
    'HPO_PRECISION_IMPROVEMENT_MIN_SHORT': (int, float),
    'HPO_PRECISION_IMPROVEMENT_MIN_LONG': (int, float),
    'AUGMENTATION_MIN_SIGNAL_RATE': (int, float),
    'AUGMENTATION_TARGET_SIGNAL_RATE': (int, float),
    'AUGMENTATION_NOISE_STD': (int, float),
    'KL_WARMUP_EPOCHS': int,
    'KL_MAX_WEIGHT': (int, float),
    'KL_SAMPLING_MAX_WEIGHT': (int, float),
    'TEMPORAL_GAP_SIGNIFICANCE': (int, float),
    'STABILITY_RF_ESTIMATORS': int,
    'FI_TRAIN_EPOCHS': int,
    'FI_BATCH_SIZE': int,
    'METRIC_PCT_THRESHOLDS': list,
    'NN_LOG_TP_ARCHS': list,
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
    
    # Validate PER_ARCH_WINSORIZE percentile ranges
    per_arch = config.get('PER_ARCH_WINSORIZE', {})
    for arch_name, bounds in per_arch.items():
        if not isinstance(bounds, dict):
            continue
        low = bounds.get('low', 0)
        high = bounds.get('high', 100)
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            continue
        if not (0 <= low < 100):
            raise ValueError(f"PER_ARCH_WINSORIZE['{arch_name}']['low']={low} must be in [0, 100)")
        if not (0 < high <= 100):
            raise ValueError(f"PER_ARCH_WINSORIZE['{arch_name}']['high']={high} must be in (0, 100]")
        if low >= high:
            raise ValueError(f"PER_ARCH_WINSORIZE['{arch_name}']: low={low} must be < high={high}")
    
    return True
