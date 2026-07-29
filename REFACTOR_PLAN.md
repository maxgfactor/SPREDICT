# Pipeline Refactoring Plan

**Goal**: Reverse-engineer the existing 24-file / ~12,000-line stock prediction pipeline into 9 focused files (~5,800 lines) that produce **identical** log output, preserve every feature toggle, and make the architecture immediately readable.

**Method**: Every log line in `pipeline_cpu_iter38.log` maps to specific code. Each new file is designed so that a developer reading the log output can find the exact code that produced it.

---

## Table of Contents

1. [Current State Inventory](#1-current-state-inventory)
2. [Target File Structure](#2-target-file-structure)
3. [File-by-File Content Specification](#3-file-by-file-content-specification)
   - [config.py](#31-configpy)
   - [logging.py](#32-loggingpy)
   - [data_loader.py](#33-data_loaderpy)
   - [models.py](#34-modelspy)
   - [training.py](#35-trainingpy)
   - [evaluate.py](#36-evaluatepy)
   - [phases.py](#37-phasespy)
   - [phase_4.py](#38-phase_4py)
   - [pipeline.py](#39-pipelinepy)
4. [Log-to-Code Map](#4-log-to-code-map)
5. [Dependency Graph](#5-dependency-graph)
6. [Modularization Preserved](#6-modularization-preserved)
7. [Risk Assessment](#7-risk-assessment)
8. [Execution Order](#8-execution-order)

---

## 1. Current State Inventory

### 24 Active Pipeline Files

| # | File | Lines | Role | Key Contents |
|---|---|---|---|---|
| 1 | `chunk_01_config.py` | 665 | Config definition + validation | CONFIG dict, REQUIRED_CONFIG_KEYS, CONFIG_TYPES, validate_config_structure |
| 2 | `chunk_02_utils_logging.py` | 370 | Logger class | Logger with 12 public methods, source-file tagging |
| 3 | `chunk_04_utils_metrics.py` | 757 | Metric formulas (stateless) | 25 functions: precision, recall, AUC, F1, MCC, Brier, Kappa, Gini, etc. |
| 4 | `chunk_05_data_manager.py` | 584 | Data loading + preprocessing | DataManager class: CSV loading, feature engineering, sampling |
| 5 | `chunk_07_data_temporal.py` | 190 | Temporal features + weighting | 4 standalone functions: extract, apply (linear/exponential/advanced), validate |
| 6 | `chunk_08_models_base.py` | 619 | Base NN builders | VAE (SamplingLayer, VAEClassifier), Dense, CNN, RNN, LSTM builders |
| 7 | `chunk_09_models_advanced.py` | 484 | Advanced NN builders | Transformer (positional encoding), TabNet, GNN (SAGE, GAT), hybrid models |
| 8 | `chunk_10_models_ensemble.py` | 318 | Ensemble builders | Stacking, bagging, extra trees, boosting, precision-weighted ensemble |
| 9 | `chunk_11_models_sklearn.py` | 508 | Sklearn models + FocalLoss | XGBoost, LightGBM, CatBoost wrappers, FocalLoss Keras class |
| 10 | `chunk_12_evaluation_evaluator.py` | 817 | Evaluation orchestration | Evaluator class: 23 methods wrapping metrics + threshold search |
| 11 | `chunk_13_state_manager.py` | 213 | Pipeline state management | StateManager: feedback loops, backtracking, context storage |
| 12 | `chunk_14_models_trainer.py` | 534 | Training orchestration | ModelTrainer, KLAnnealingCallback, build_architecture |
| 13 | `chunk_15_phase_base.py` | 72 | BasePhase ABC | Abstract execute() interface for all phases |
| 14 | `chunk_16_phase_1_setup.py` | 335 | Phase 1 | Data load, class distribution, date-based train/val/inference split |
| 15 | `chunk_17_phase_3_temporal.py` | 182 | Phase 3 | Temporal weight generation |
| 16 | `chunk_18_phase_4_ensemble.py` | **2218** | Phase 4 — monolithic | 5 evaluation sections, HPO, ensemble, model saving, diagnostics |
| 17 | `chunk_19_phase_5_optimization.py` | 586 | Phase 5 | Model loading, inference, consensus voting, final metrics |
| 18 | `chunk_20_pipeline_main.py` | 553 | Orchestrator | PipelineOrchestrator run(), metrics_summary.csv, main() |
| 19 | `chunk_21_hyperparam_optimizer.py` | 401 | Optuna HPO | HyperparameterOptimizer: per-architecture search spaces |
| 20 | `chunk_22_model_loader.py` | 239 | Model loading | load_saved_models, load_model_metadata |
| 21 | `chunk_XX_feature_importance.py` | 654 | Feature importance analysis | FeatureImportanceAnalyzer: 6 methods, auto-pruning |
| 22 | `chunk_XX_phase_backward_elimination.py` | 281 | Backward elimination | PhaseBE: per-architecture proxy model feature ranking |
| 23 | `chunk_XX_phase_feature_analysis_a.py` | 139 | Phase Xa orchestrator | Subsampling, 6-method importance, per-threshold pruning |
| 24 | `chunk_XX_phase_feature_analysis_b.py` | 269 | Phase Xb temporal gap | Recent vs older precision comparison |
| | **TOTAL** | **11,988** | | |

### Waste/Redundancy Categories

| Category | Lines | Percentage |
|---|---|---|
| `__main__` test blocks (24 files × ~25 lines avg) | ~600 | 5% |
| Phase validation functions (same pattern, 7×) | ~500 | 4% |
| Metric wrapper layering (chunk_04 calls from chunk_12) | ~300 | 2.5% |
| Phase 4 section duplication (5× ~200-line blocks) | ~600 | 5% |
| Model builder interface inconsistency | ~200 | 1.5% |
| **Total removable** | **~2,200** | **18%** |

---

## 2. Target File Structure

### 9 Files, ~5,800 Lines

```
cicd/
├── config.py        #  200 lines — CONFIG dict + type schema + validation
├── logging.py       #  200 lines — Logger with source-file tagging
├── data_loader.py   #  500 lines — CSV loading, preprocessing, feature engineering, temporal weighting
├── models.py        #  900 lines — All 9 architecture builders + FocalLoss + SklearnModelWrapper + ensemble
├── training.py      #  450 lines — ModelTrainer + KLAnnealing + HyperparameterOptimizer (Optuna)
├── evaluate.py      #  800 lines — All metric functions + Evaluator class + threshold search + diagnostics
├── phases.py        #  700 lines — BasePhase + DataSetup + TemporalWeighting + FeatureImportance + FeaturePruning + TemporalPrecisionGap
├── phase_4.py       # 1400 lines — Phase 4: parameterized 5-section evaluation + ensemble + model saving
└── pipeline.py      #  550 lines — PipelineOrchestrator + Inference + model_loader + main()
```

### Savings Breakdown

| Refactored File | Source Files | Source Lines | Target Lines | Saved |
|---|---|---|---|---|
| `config.py` | `chunk_01_config.py` | 665 | 200 | 465 |
| `logging.py` | `chunk_02_utils_logging.py` | 370 | 200 | 170 |
| `data_loader.py` | `chunk_05_data_manager.py`, `chunk_07_data_temporal.py` | 774 | 500 | 274 |
| `models.py` | `chunk_08` + `chunk_09` + `chunk_10` + `chunk_11` | 1929 | 900 | 1029 |
| `training.py` | `chunk_14_models_trainer.py`, `chunk_21_hyperparam_optimizer.py` | 935 | 450 | 485 |
| `evaluate.py` | `chunk_04_utils_metrics.py`, `chunk_12_evaluation_evaluator.py` | 1574 | 800 | 774 |
| `phases.py` | `chunk_15` + `chunk_16` + `chunk_17` + `chunk_XX_xa` + `chunk_XX_fi` + `chunk_XX_be` + `chunk_XX_xb` | 1932 | 700 | 1232 |
| `phase_4.py` | `chunk_18_phase_4_ensemble.py` | 2218 | 1400 | 818 |
| `pipeline.py` | `chunk_20` + `chunk_22` + `chunk_19` + `chunk_13` | 1591 | 550 | 1041 |
| **TOTAL** | **24 files** | **11,988** | **~5,900** | **~6,088 (51%)** |

### What Goes Where (cross-reference)

Current `chunk_N` → New file mapping:

| Current File | New File(s) | What |
|---|---|---|
| `chunk_01_config.py` | `config.py` | CONFIG + validation |
| `chunk_02_utils_logging.py` | `logging.py` | Logger class |
| `chunk_04_utils_metrics.py` | `evaluate.py` | Stateless metric functions (Section 1) |
| `chunk_05_data_manager.py` | `data_loader.py` | DataManager class |
| `chunk_07_data_temporal.py` | `data_loader.py` | Temporal feature functions → DataManager methods |
| `chunk_08_models_base.py` | `models.py` | VAE, Dense, CNN, RNN, LSTM builders (Section 2) |
| `chunk_09_models_advanced.py` | `models.py` | Transformer, TabNet, GNN, hybrids (Section 2 + Section 6 stubs) |
| `chunk_10_models_ensemble.py` | `models.py` | Ensemble builders (Section 4) + SklearnModelWrapper (Section 5) |
| `chunk_11_models_sklearn.py` | `models.py` | Tree builders (Section 3) + FocalLoss (Section 1) + SklearnModelWrapper (Section 5) |
| `chunk_12_evaluation_evaluator.py` | `evaluate.py` | Evaluator class (Section 2) + threshold search (Section 3) |
| `chunk_13_state_manager.py` | `pipeline.py` | StateManager class (Section 4 of pipeline.py) |
| `chunk_14_models_trainer.py` | `training.py` | ModelTrainer + KLAnnealingCallback (Section 1-2) |
| `chunk_15_phase_base.py` | `phases.py` | BasePhase ABC (Section 1) |
| `chunk_16_phase_1_setup.py` | `phases.py` | DataSetup class (Section 2) |
| `chunk_17_phase_3_temporal.py` | `phases.py` | TemporalWeighting class (Section 3) |
| `chunk_18_phase_4_ensemble.py` | `phase_4.py` | ModelTraining class |
| `chunk_19_phase_5_optimization.py` | `pipeline.py` | Inference class (Section 1) |
| `chunk_20_pipeline_main.py` | `pipeline.py` | PipelineOrchestrator + main (Section 3) |
| `chunk_21_hyperparam_optimizer.py` | `training.py` | HyperparameterOptimizer (Section 3) |
| `chunk_22_model_loader.py` | `pipeline.py` | model_loader functions (Section 2) |
| `chunk_XX_feature_importance.py` | `phases.py` | Inlined into FeatureImportance (Section 4) |
| `chunk_XX_phase_backward_elimination.py` | `phases.py` | FeaturePruning class (Section 5) |
| `chunk_XX_phase_feature_analysis_a.py` | `phases.py` | FeatureImportance class (Section 4) |
| `chunk_XX_phase_feature_analysis_b.py` | `phases.py` | TemporalPrecisionGap class (Section 6) |

---

## 3. File-by-File Content Specification

### 3.1 `config.py` (200 lines)

**Source**: `chunk_01_config.py` (665 lines)

**Keep (200 lines)**:
- The master `CONFIG` dictionary — every key and value identical
- Comments for each config key (critical for understanding)
- `REQUIRED_CONFIG_KEYS` list (runtime validation)
- `CONFIG_TYPES` dict (runtime type checking)
- `validate_config_structure()` function (66 lines of validation logic)
- Default/fallback values (e.g., `PREDICTION_THRESHOLD_DEFAULT = 0.5`)
- Environment variable setup (`TF_CPP_MIN_LOG_LEVEL = '3'`)

**Remove (465 lines)**:
- `get_config()` function — never called (only the CONFIG import is used)
- `update_config()` function — never called
- `__main__` test block (lines 580-664, 85 lines of test code with hardcoded asserts)
- Redundant inline comments (every config key already has a comment in the dict)
- Unused imports and dead code paths

**Signature**:
```python
# config.py — Master Configuration

import os
import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any

PREDICTION_THRESHOLD_DEFAULT = 0.5

# Suppress TensorFlow/CUDA warnings for CPU-only execution
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

CONFIG = {
    'DATA_PATH': 'sampled_184408.csv',
    'USE_SAMPLING': True,
    'SAMPLE_SIZE': 184408,
    'USE_TEMPORAL_WEIGHTING': True,   # ADDED — gates TemporalWeighting + Inference temporal recomputation
    'RANDOM_SEED': 42,                # ADDED — anchors all RNG for reproducibility
    # ... all 80+ config keys ...
}

REQUIRED_CONFIG_KEYS = ['DATA_PATH', 'USE_SAMPLING', 'SAMPLE_SIZE', 'USE_TEMPORAL_WEIGHTING', 'RANDOM_SEED', ...]

CONFIG_TYPES = {'DATA_PATH': str, 'USE_SAMPLING': bool, 'SAMPLE_SIZE': int, 'USE_TEMPORAL_WEIGHTING': bool, 'RANDOM_SEED': int, ...}

def validate_config_structure(config: Dict) -> bool:
    """Validate config has all required keys and correct types."""
```

---

### 3.2 `logging.py` (200 lines)

**Source**: `chunk_02_utils_logging.py` (370 lines)

**Keep (200 lines)**:
- `Logger` class with 9 public methods (7 active, 2 dead code preserved for spec compliance)

| Method | Purpose | Used By | Log Example |
|---|---|---|---|---|
| `__init__(config)` | Initialize with verbosity level | Every phase | — |
| `log(msg, level)` | Format and print tagged log line | Everything | `[chunk_16_phase_1_setup.py] [info] Data loaded: 184408 samples, 34 features` |
| `format_metric(value)` | Format metric with sign/trend | Phase 4 sections | `VALIDATION_PRECISION=0.5039` |
| `get_trend_indicator(v1, v2)` | ↑/↓/→ comparison | HPO stagnation check | `↑ (0.0015 improvement)` |
| `log_feature_quality_metrics(X)` | Per-feature stats | Phase 1 | per-feature mean, std, skew, kurtosis |
| `log_class_distribution(y, dates, thresholds)` | Signal rate per threshold | Phase 1 | `Signal cases: 94,965 (51.5%)` |
| `log_temporal_coverage(dates)` | Date range summary | Phase 1 | `Temporal coverage: 32 dates, 20250910 to 20251023` |

**Also kept (dead code, compiled but never called in pipeline — preserved for spec compliance)**:
| `format_phase_1_5_standardized(...)` | Phase 1.5 report formatter | Never called in pipeline | — |
| `format_standard_metric_report(...)` | Standardized metric report | Never called in pipeline | — |

**Remove (170 lines)**:
- `validate_logger_instance()` — defensive check, only called in `__main__`
- `log_system_performance()`, `log_data_flow_metrics()`, `log_final_evaluation()` — defined but never called in pipeline
- `__main__` test block (25 lines)
- Duplicated format strings that can be inlined in `log()`
- Unused helper methods

**Critical**: The `log()` method's source-file tagging format `[filename.py]` must be preserved exactly, as it's documented in SPEC.md §1.1.

---

### 3.3 `data_loader.py` (500 lines)

**Source**: `chunk_05_data_manager.py` (584) + `chunk_07_data_temporal.py` (190)

**Keep from `chunk_05`**:

| DataManager Method | Used By | Log Evidence |
|---|---|---|
| `__init__(config)` | Phase 1 constructor | — |
| `load_data(winsize=True)` | Phase 1.execute | `Data loaded: 184408 samples, 34 features` |
| `_load_and_validate_csv(path)` | load_data | `Dataset: 6,727,216 rows x 23 cols, 937.4 MB` |
| `_apply_feature_engineering(X, winsorize)` | load_data | Produces log lines 24-29 (ratio, poly, RSI) |
| `_apply_stratified_sampling(X, y, dates)` | load_data | `Sampling: size=184408` |
| `augment_signal_cases(X, y, dates)` | Phase 1 | `Signal augmentation: 0.0324 → 0.0324` |
| `concentrate_signal_cases(X, y, dates)` | Phase 1 | `Data concentration: 184408 samples retained` |
| `prepare_data(X)` | Phase 1 | Final preprocessing pass |
| `_validate_data_output(X, y, dates, ...)` | load_data | Internal validation |
| `is_log_transform_applied()` | Phase 4, Phase 5 | `LOG_TRANSFORM_TARGET=False` |

**Keep from `chunk_07`** (merged as DataManager methods):
- `extract_temporal_features(dates)` — produces 9 temporal features
- `apply_temporal_weighting_strategy(dates, strategy_config: Dict)` — produces weights
  Called with `{'type': 'linear', 'multiplier': config['TEMPORAL_MULTIPLIER']}`
  (Actual behavior preserved from chunk_07:48-99 — supports linear/exponential/uniform strategies.)
- `validate_temporal_features(features)` — shape/dtype check

**Remove (274 lines)**:
- `_detect_date_column()` — replaced by column name lookup
- `validate_data_output()` standalone — merged into `_validate_data_output()`
- `apply_advanced_temporal_weighting()` — not used in production
- Both test `__main__` blocks (~50 lines)
- Redundant getter methods inlined

**Feature Engineering Pipeline** (order preserved from log):
```python
1. Winsorize features (global: low=3, high=97)
2. Log1p transform on 19 skewed features (indices [0-5, 7-16, 18-20])
3. Add 5 ratio features (Price_to_52W_High, Volume_to_Avg_Volume, ...)
4. Log1p transform ratio features
5. Add 4 polynomial interactions (52W_Low_x_SMA20, ...)
6. Bin RSI_14 into 4 zones (oversold, neutral_low, neutral_high, overbought)
7. Result: 21 raw → 34 features
```

---

### 3.4 `models.py` (900 lines)

**Source**: `chunk_08` (619) + `chunk_09` (484) + `chunk_10` (318) + `chunk_11` (508)

**Section structure**:

```python
# ============================================================================
# Section 1: Loss Functions (~50 lines)
# ============================================================================
class FocalLoss(tf.keras.losses.Loss):
    """From chunk_11_models_sklearn.py. alpha/gamma from FOCAL_LOSS_CONFIG."""

# ============================================================================
# Section 2: Neural Architecture Builders (~400 lines)
# ============================================================================
def build_vae_model(config, input_dim) -> tf.keras.Model:
    """Includes SamplingLayer + VAEClassifier logic from chunk_08."""
def build_dense_model(config, input_dim) -> tf.keras.Model:
    """Was build_dense_model in chunk_08."""
def build_cnn_model(config, input_dim) -> tf.keras.Model:
    """Was build_cnn_model in chunk_08. CNN feature extractor variant inlined."""
def build_rnn_model(config, input_dim) -> tf.keras.Model:
    """Was build_rnn_model in chunk_08."""
def build_lstm_model(config, input_dim) -> tf.keras.Model:
    """Was build_lstm_model in chunk_08."""
def build_transformer_model(config, input_dim) -> tf.keras.Model:
    """Was build_transformer_model in chunk_09. ExpandDimsLayer + PosEncoding inlined."""

# ============================================================================
# Section 3: Tree Architecture Builders (~200 lines)
# ============================================================================
def build_xgboost_model(config, input_dim, y) -> SklearnModelWrapper:
    """Was build_xgboost_model in chunk_11."""
def build_lightgbm_model(config, input_dim, y) -> SklearnModelWrapper:
    """Was build_lightgbm_model in chunk_11."""
def build_catboost_model(config, input_dim, y) -> SklearnModelWrapper:
    """Was build_catboost_model in chunk_11.
    IMPORTANT: Must set verbose=0, logging_level='Silent' to prevent
    CatBoostParamException during HPO (chunk_21 generated 'verbose_eval'
    parameter crashes CatBoost; fix via verbose=0 in constructor).
    """

# ============================================================================
# Section 4: Ensemble Builders (~100 lines)
# ============================================================================
def create_precision_ensemble(models, weights) -> object:
    """Was create_precision_ensemble in chunk_10."""
def build_stacking_meta_model(config, input_dim) -> object:
    """Was build_stacking_meta_model in chunk_10."""

# ============================================================================
# Section 5: SklearnModelWrapper (~100 lines)
# ============================================================================
class SklearnModelWrapper:
    """Universal sklearn wrapper with fit/predict/save/load. From chunk_10+chunk_11."""
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def save(self, path): ...      # .joblib serialization
    def load(self, path): ...
    def decision_function(self, X): ...

# ============================================================================
# Section 6: Dormant Builders (~50 lines — stubs with NotImplementedError)
# ============================================================================
# These 14 builders are registered in ModelTrainer's dispatch table
# (training.py:build_architecture) but not in ACTIVE_ARCHITECTURES by default.
# If activated via config, they raise NotImplementedError with a clear message
# pointing to the pre-refactoring source files.
#
# Stubs: build_tabnet_model, build_gnn_sage_model, build_gnn_gat_model,
# build_hybrid_cnn_lstm_model, build_hybrid_transformer_gnn_model,
# build_simple_attention_model, build_isolation_forest_model,
# build_oneclass_svm_model, build_svm_model, build_cnn_feature_extractor,
# build_bagging_random_forest_model,
# build_extra_trees_ensemble_model, build_boosting_adaptive_model, build_dense_fallback
#
# Stub pattern:
# def build_<arch>(config, input_dim, y=None, loss_fn=None):
#     raise NotImplementedError(
#         f"Architecture '<arch>' is dormant in the refactored pipeline. "
#         f"See chunk_XX_models_advanced.py (or _ensemble.py / _sklearn.py) "
#         f"for the original implementation."
#     )
```

**Key unification — consistent builder interface**:
```python
def build_<arch>(config: Dict, input_dim: int, y: Optional[np.ndarray] = None,
                 loss_fn: Optional[str] = None) -> Union[tf.keras.Model, SklearnModelWrapper]:
```
All builders accept `config` (for hyperparams), `input_dim` (feature count), and return either a Keras model or SklearnModelWrapper. This lets `training.py` dispatch by architecture name without a 12-branch if/elif.

**Remove (1029 lines)**:
- `validate_model_output()` — dead code
- `validate_sklearn_model()` — dead code
- `validate_ensemble_output()` — dead code
- All 4 test `__main__` blocks (~120 lines total)
- Dormant builder implementations moved to stubs (code preserved, signature retained, but not imported in default path)
- Redundant docstrings (the builder names and signatures are self-documenting)
- `ExpandDimsLayer` class — implementation inlined into `build_transformer_model()`

---

### 3.5 `training.py` (450 lines)

**Source**: `chunk_14_models_trainer.py` (534) + `chunk_21_hyperparam_optimizer.py` (401)

**Section structure**:

```python
# ============================================================================
# Section 1: Callbacks (~50 lines)
# ============================================================================
class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """VAE KL divergence annealing. warmup=10, max_kl=0.1. From chunk_14."""

# ============================================================================
# Section 2: ModelTrainer (~250 lines)
# ============================================================================
class ModelTrainer:
    def __init__(self, config: Dict, logger: Logger):
        ...

    def build_architecture(self, arch_name: str, input_dim: int,
                           y_train: np.ndarray) -> Union[tf.keras.Model, SklearnModelWrapper]:
        """Dispatch to models.py builder by arch_name. Handles focal_loss toggles."""
        ...

    def build_architecture_with_params(self, arch_name: str, input_dim: int,
                                       hyperparams: Dict) -> Union[tf.keras.Model, SklearnModelWrapper]:
        """Merge hyperparams into config, then build. Used by HPO."""
        ...

    def train_model(self, model, X, y, validation_data=None, epochs=15,
                    batch_size=256, sample_weight=None) -> TrainResult:
        """
        Unified training for both Keras and sklearn models.
        Keras: class_weight handling, focal_loss→class_weight=None,
               3D reshape for sequential models, callbacks (TerminateOnNaN,
               EarlyStopping(patience=20), KLAnnealing for VAE).
        Sklearn: _train_sklearn_model() with TREE_EARLY_STOPPING_ROUNDS,
                 eval_set, scale_pos_weight for XGBoost.
        Returns TrainResult namedtuple.
        """

# TrainResult namedtuple (replaces loose returns from chunk_14)
TrainResult = namedtuple('TrainResult',
    ['model', 'history', 'train_loss', 'val_loss', 'training_time'])

# ============================================================================
# Section 3: HyperparameterOptimizer (~150 lines)
# ============================================================================
class HyperparameterOptimizer:
    """Optuna Bayesian optimization. From chunk_21."""
    def __init__(self, config: Dict, arch_name: str, logger: Logger):
        ...

    def optimize(self, X_train, y_train, X_val, y_val,
                 input_dim, model_trainer) -> Dict:
        """
        1. Study = optuna.create_study(direction='maximize')
        2. HYPERPARAM_SEARCH_SPACE[arch_name] defines params
        3. HYPERPARAM_OPTIMIZATION_TRIALS trials (default 3)
        4. Each trial: build_architecture_with_params → train_model → evaluate
           Wrap each trial in try/except Exception:
           - On success: log trial metrics, update best
           - On exception: log "TRIAL N failed: {error}", continue to next trial
           - On degenerate predictions (std_pred < 0.005): log "REJECTED — degenerate", continue
           - On near-constant positive predictions (recall≈1.0, precision≈base_rate):
             log "REJECTED - near-constant positive predictions", continue
           - On insufficient true_positives (RNN only, TP<100):
             log "REJECTED - true_positives=N < 100", continue
         5. Log baseline before HPO: "[section 2] [{ARCH}] [hyperparameter_optimization search
            baseline] LABEL_THRESHOLD=X" (one line), then on the next line:
            "[section 2] [{ARCH}] [hyperparameter_optimization search baseline] VALIDATION_PRECISION=X ..."
         6. Log each trial: "TRIAL N/N: {params}" + val metrics (each on own line)
         7. After completion: "[section 2] [{ARCH}] [best trial] LABEL_THRESHOLD=X,
            VALIDATION_PRECISION=X..." + best hyperparams
         8. Objective = precision (trees, VAE) or precision*log(TP+1) (NNs)
         9. HPO_TARGET_PRECISION=0.60 early stop
        10. HPO_STAGNATION_THRESHOLD=50 stop
        11. Returns best_hyperparams dict
        """

    def get_search_space_summary(self) -> str:
        """Log-friendly string of search space."""
```

**Remove (485 lines)**:
- `validate_training_output()` — replaced by TrainResult typing
- `validate_trainer_instance()` — inlined into constructor
- Both test `__main__` blocks (~60 lines)
- `train_multiple_architectures()` — loop belongs in phase_4.py, not trainer
- Duplicated docstrings

---

### 3.6 `evaluate.py` (700 lines)

**Source**: `chunk_04_utils_metrics.py` (757) + `chunk_12_evaluation_evaluator.py` (817)

**Structure**:

```python
# ============================================================================
# Section 1: Stateless Metric Functions (~200 lines)
# All from chunk_04_utils_metrics.py.
# Each function: (y_true, y_pred/y_proba) → float
# No class wrappers — direct function calls.
# ============================================================================
def calculate_precision(y_true, y_pred):
    return safe_divide(TP, TP + FP)
def calculate_recall(y_true, y_pred):
    return safe_divide(TP, TP + FN)
def calculate_auc(y_true, y_proba):
    try: return roc_auc_score(y_true, y_proba)
    except: return 0.5
def calculate_f1(y_true, y_pred):
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0
def calculate_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)
def calculate_average_precision(y_true, y_proba):
    return average_precision_score(y_true, y_proba)
def calculate_specificity(y_true, y_pred):
    return safe_divide(TN, TN + FP)
def calculate_fpr(y_true, y_pred):
    return safe_divide(FP, FP + TN)
def calculate_f2_score(y_true, y_pred):
    return 5 * P * R / (4 * P + R) if (P + R) > 0 else 0.0
def calculate_brier_score(y_true, y_proba):
    return mean((y_true - y_proba) ** 2)
def calculate_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)
def calculate_informedness(y_true, y_pred):
    return metrics['recall'] + metrics['specificity'] - 1
def calculate_markedness(y_true, y_pred):
    return metrics['precision'] + NPV - 1
def calculate_gini(y_true, y_proba):
    return max(0, 2 * calculate_auc(y_true, y_proba) - 1)
def calculate_balanced_accuracy(y_true, y_pred):
    return (calculate_recall(y_true, y_pred) + calculate_specificity(y_true, y_pred)) / 2
def calculate_optimal_threshold(y_true, y_proba):
    """Youden's J: max(TPR - FPR). Returns threshold value."""
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0

# ============================================================================
# Section 2: Evaluator Class (~300 lines)
# From chunk_12_evaluation_evaluator.py — metric orchestration.
# ============================================================================
class Evaluator:
    def __init__(self, config: Dict, logger: Logger):
        ...

    def calculate_confusion_matrix(self, y_true, y_pred) -> Tuple[int, int, int, int]:
        """Returns (TP, FP, TN, FN)."""

    def evaluate_at_threshold(self, y_true, y_proba, label='VALIDATION') -> Dict:
        """
        Compute ALL 24 metrics at once.
        Returns dict with keys: PRECISION, TRUE_POSITIVES, FALSE_POSITIVES,
        TRUE_NEGATIVES, FALSE_NEGATIVES, MAX_PREDICTION, MEAN_PREDICTION,
        RECALL, F1_SCORE, AUC, SPECIFICITY, FALSE_POSITIVE_RATE, F2_SCORE,
        MCC, PRAUC, BALANCED_ACCURACY, Brier, Kappa, Informedness, Markedness,
        Gini, OPTIMAL_THRESHOLD, STD_PREDICTION, PCT_ABOVE_THRESHOLD
        """

    def find_optimal_threshold(self, model, X_val, y_val_raw,
                               thresholds_list, arch_type='neural', ...) -> Dict:
        """
        For each threshold:
        1. y_binary = (y_val_raw >= threshold).astype(int)
        2. model.predict(X_val) → probs
        3. binary = (probs >= 0.5).astype(int)
        4. evaluate_at_threshold(y_binary, binary, probs)
        5. Apply safeguard gates:
           - neural: MIN_PRECISION_OVER_BASELINE=0.01, MIN_POS_ABS=5
           - tree:   MIN_PRECISION_OVER_BASELINE=0.01, MIN_POS_PCT=0.001
           - both:   MAX_POS_PRED_RATIO=0.65
        6. Return threshold with highest valid val precision
        """

# ============================================================================
# Section 3: XGBoost Coverage Sweep (~100 lines)
# ============================================================================
def search_coverage_thresholds(y_true, y_proba, coverage_rates,
                               target_precision, max_coverage):
    """
    XGBoost coverage sweep. Always called for XGBoost in Phase 4 Section 5.
    9 coverage rates from config: [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
                                   0.10, 0.25, 0.50]
    For each rate: find prediction threshold that achieves that coverage.

    Gated by PREDICTION_XGBOOST_PRECISION_TARGETING toggle:
    - True:  try precision-targeted thresholds first; if none reach 0.60 → F1-optimal fallback
    - False: skip precision targeting, return F1-optimal threshold directly

    In both cases returns a threshold (either precision-targeted or F1-optimal).
    (Per iter38 log: XGBoost precision_target=0.60 not met → fallback to F1-optimal)
    """

# ============================================================================
# Section 4: Diagnostic / Analysis Functions (~150 lines)
# All from chunk_04_utils_metrics.py. Used in Phase 4 SECTION 3 (Diagnostics).
# ============================================================================
def get_prediction_percentiles(predictions: np.ndarray) -> Dict[str, float]:
    """Compute percentiles (p1, p5, p25, p50, p75, p95, p99) of predictions.
    Internal helper for format_diagnostic_string()."""

def format_diagnostic_string(predictions, prefix="") -> str:
    """Format prediction distribution for diagnostic logging. Called 6+ times in Phase 4.
    Uses get_prediction_percentiles() and get_round_threshold_density() internally."""
def inverse_log_transform(y_log) -> np.ndarray:
    """Inverse of log1p transform. Called in Phase 4 Section 5 and Phase 5."""
def calculate_temporal_drift(segment_metrics) -> Dict:
    """Compare precision/recall across train/val/inference segments."""
def calculate_permutation_importance(model, X, y_true, ...) -> Dict:
    """Shuffle each feature, measure precision drop."""
def calculate_prediction_entropy(predictions) -> float:
    """Entropy of prediction distribution."""
def calculate_logit_compression(predictions) -> float:
    """Measure of prediction concentration."""
def calculate_mutual_information(predictions, y_true, threshold) -> float:
    """MI between predictions and labels."""
def analyze_loss_distribution(loss_history: List[float]) -> Dict[str, float]:
    """Analyze loss convergence: final_loss, convergence_rate, plateau_detected, trend.
    Called in Phase 4 diagnostics section. (Was chunk_04:136)"""
def calculate_ks_test(positive_preds, negative_preds) -> Dict:
    """Kolmogorov-Smirnov test: max separation between class predictions."""
def calculate_bhattacharyya_distance(positive_preds, negative_preds) -> float:
    """Bhattacharyya distance between class prediction distributions."""

**Key change**: Eliminates the layering where `chunk_12` wraps every `chunk_04` function. Now there is ONE call chain: `evaluate_at_threshold()` directly calls `calculate_precision()` etc. — no intermediate wrapper.

**Remove (874 lines)**:
- All standalone wrapper functions from chunk_04 that duplicate Evaluator methods (e.g., `safe_average_precision_score()` — directly use `average_precision_score` with try/except)
- `assess_model_learning()`, `assess_learning()` — dead code (only in `__main__` tests)
- `format_metric_value()` from chunk_04 — removed as duplicate. Logger.format_metric() in
  logging.py has an identical implementation. All callers use the Logger version.
- `cross_validate()` from chunk_12 — not used in pipeline (was a utility)
- `validate_evaluator_instance()` — dead code
- `validate_evaluator_output()` — dead code
- Both test `__main__` blocks (~80 lines)
- Duplicated try/except patterns (centralized into `safe_divide` and single try/except per metric)

---

### 3.7 `phases.py` (700 lines)

**Source**: `chunk_15` (72) + `chunk_16` (335) + `chunk_17` (182) + `chunk_XX_xa` (139) + `chunk_XX_fi` (654) + `chunk_XX_be` (281) + `chunk_XX_xb` (269) = **1,932 lines**

**Structure**:

```python
# ============================================================================
# Section 1: BasePhase ABC (~30 lines)
# ============================================================================
class BasePhase(ABC):
    """Abstract base class for all pipeline phases."""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = None   # Set by subclass' own __init__

    # DESIGN CHANGE (vs current code):
    # Phases currently create their own Logger in __init__.
    # In the refactored version, PipelineOrchestrator creates one Logger
    # and passes it via phase.logger = Logger(config) after construction,
    # or each phase still creates its own. The plan uses the latter approach
    # to minimize diff — each phase creates self.logger = Logger(config).

    @abstractmethod
    def execute(self, context: Dict) -> Dict:
        """Execute phase logic. Returns updated context."""
        pass

    # DESIGN CHANGE (new): Class-level documentation of phase contracts.
    # Not present in current code. Provides self-documenting data flow.
    # Used by validate_context() to verify phase inputs/outputs.
    #
    # DESIGN NOTE on validation: The current pipeline calls
    # validate_phaseN_output() after each phase (chunk_20:166-172,190).
    # These assert array shapes match, weights are positive, etc.
    # The refactored version keeps validate_context() for basic key checks
    # and adds typed return annotations for value-level validation.
    # The CONTEXT_CONSUMED/PRODUCED lists enable static validation;
    # runtime value checks (shape, positivity, finiteness) are inlined
    # into each phase's execute() as debug assertions.
    CONTEXT_CONSUMED = []   # Documented context keys this phase reads
    CONTEXT_PRODUCED = []   # Documented context keys this phase writes

# === Per-Phase __init__ patterns (design preservation) ===
# DataSetup:           Logger(config), DataManager(config)
# TemporalWeighting:       Logger(config), Evaluator(config) [dead], StateManager() [dead]
# FeatureImportance:        Logger(config)
# FeaturePruning:    Logger(config)
# ModelTraining:          Logger(config), StateManager() [dead], Evaluator(config)
# TemporalPrecisionGap:    Logger(config)
# Inference:  Logger(config), Evaluator(config), DataManager(config)
#
# Design decision: Keep per-phase object creation in each phase's own __init__
# rather than injecting from PipelineOrchestrator. This minimizes diff
# and keeps each phase self-contained for testing.

# ============================================================================
# Section 2: DataSetup (~150 lines)
# ============================================================================
class DataSetup(BasePhase):
    """
    Phase 1: Load CSV → filter NaNs/zeros → sample → feature engineer →
    class distribution → train/val/inference split.
    """
    CONTEXT_CONSUMED = []  # Phase 1 starts with empty context
    CONTEXT_PRODUCED = ['X', 'y', 'dates', 'X_train', 'y_train_continuous',
                        'dates_train', 'X_val', 'y_val_continuous', 'dates_val',
                        'X_inference', 'y_inference_continuous', 'dates_inference',
                        'raw_target_values', 'raw_target_column', 'feature_names',
                        'data_stats', 'phase1_complete']

    def execute(self, context):
        # 1. DataManager.load_data(winsorize=not PER_ARCH_WINSORIZE)
        #    → X, y, dates (with _raw_target_values stored post-filter)
        # 2. Log target distribution (min, max, mean, median, std)
        #    → iter38 log lines 33-35
        # 3. Log class distribution at all 5 thresholds (20, 15, 10, 5, 0)
        #    → iter38 log lines 38-85
        # 4. Log feature quality metrics (per-feature stats)
        #    → iter38 log lines 87-121
        # 5. DataManager.augment_signal_cases()
        # 6. DataManager.concentrate_signal_cases()
        # 7. DataManager.prepare_data()
        # 8. Date-based split:
        #    - TOP_DATES_HELD_OUT=2 newest → inference
        #    - Remaining: VAL_SPLIT_PERCENTAGE=0.30 → validation
        #    → iter38 log lines 127-130
        # 9. Store all splits in context
        pass

# ============================================================================
# Section 3: TemporalWeighting (~60 lines)
# ============================================================================
class TemporalWeighting(BasePhase):
    """
    Phase 3: Generate temporal weights and extract temporal features.
    Checks USE_TEMPORAL_WEIGHTING toggle.
    """
    CONTEXT_CONSUMED = ['X', 'y', 'dates', 'phase1_complete']
    CONTEXT_PRODUCED = ['temporal_weights', 'temporal_features', 'phase3_complete']

    def execute(self, context):
        # 0. Skip if USE_TEMPORAL_WEIGHTING=False
        #    Must still set phase3_complete so Phase 4 validation passes.
        # 1. extract_temporal_features(dates) → 9 temporal features
        # 2. apply_temporal_weighting_strategy(dates, {'type': 'linear', 'multiplier': config['TEMPORAL_MULTIPLIER']})
        #    → weights (min=1.0, max=4.0, mean=2.601)
        #    → iter38 log lines 131
        # 3. Store in context
        pass

# ============================================================================
# Section 4: FeatureImportance (~200 lines)
# ============================================================================
class FeatureImportance(BasePhase):
    """
    Phase Xa: 6-method feature importance analysis with per-threshold pruning.
    Checks FEATURE_ANALYSIS_ENABLED toggle.
    """
    CONTEXT_CONSUMED = ['X', 'raw_target_values', 'y', 'feature_names']
    CONTEXT_PRODUCED = ['threshold_kept_indices', 'threshold_dropped_indices',
                        'pruned_feature_indices', 'feature_importance_results',
                        'phaseXa_complete']

    def execute(self, context):
        # 1. Skip if FEATURE_ANALYSIS_ENABLED=False
        # 2. Get active methods from FEATURE_IMPORTANCE_METHODS config
        # 3. Subsample to FEATURE_ANALYSIS_SAMPLE_SIZE (100k from 184k)
        #    → iter38 log line 142
        # 4. For each threshold in [20, 15, 10, 5, 0]:
        #    - y_binary = (y_raw >= threshold).astype(int)
        #    - Run 6 analysis methods (from FeatureImportanceAnalyzer methods,
        #      refactored into standalone helpers):
        #      a. _analyze_correlation: Pearson + Spearman + point-biserial
        #      b. _analyze_tree: RF + GBM feature_importances_
        #      c. _analyze_permutation: Dense model, shuffle each feature
        #      d. _analyze_neural: extract input layer weights
        #      e. _analyze_shap: SHAP value analysis (chunk_XX_fi:398 — 6th method)
        #      f. _analyze_ablation: per-feature Dense model AUC
        #      g. _compute_consolidated_ranking: mean rank across methods
        #    - Prune bottom FEATURE_PRUNE_PERCENTILE (0 = keep all)
        # 5. Compute cross-threshold stats (always/never/borderline)
        #    → iter38 log lines 174-176
        # 6. Store per-threshold pruning results

        # DESIGN NOTE: FeatureImportanceAnalyzer.generate_report() and save_report()
        # (chunk_XX_fi:537,592) are NOT preserved — report file output is dropped.
        # The 6 analysis methods + _build_quick_dense() helper + _compute_consolidated_ranking()
        # are converted to 8 standalone helper functions inlined here.
        pass

# ============================================================================
# Section 5: FeaturePruning (~200 lines)
# ============================================================================
class FeaturePruning(BasePhase):
    """
    Phase BE: Per-architecture backward feature elimination.
    Checks BACKWARD_ELIMINATION_ENABLED toggle.
    """
    CONTEXT_CONSUMED = ['X', 'raw_target_values', 'y',
                        'threshold_kept_indices', 'feature_names']
    CONTEXT_PRODUCED = ['threshold_kept_indices']  # Overwritten with per-arch dict

    def execute(self, context):
        # 1. Skip if BACKWARD_ELIMINATION_ENABLED=False
        # 2. For each architecture in ACTIVE_ARCHITECTURES:
        #    - Register proxy model (XGBoost, LightGBM, CatBoost, or RF)
        #    - For each label threshold in [20, 15, 10, 5, 0]:
        #      a. Train proxy model on all features
        #      b. Rank features by importance
        #      c. Drop BE_ELIMINATION_STEPS fraction (0.50)
        #      d. Retrain proxy, compute val precision
        #      e. If precision drop > BE_TOLERANCE (0.01), stop
        #      f. Repeat until stops or BE_MIN_FEATURES (10) reached
        #    → iter38 log lines 195-239
        # 3. Store per-architecture, per-threshold kept_indices
        pass

# ============================================================================
# Section 6: TemporalPrecisionGap (~60 lines)
# ============================================================================
class TemporalPrecisionGap(BasePhase):
    """
    Phase Xb: Compare precision on recent vs older validation dates.
    """
    # NOTE: temporal_weights was removed from CONTEXT_CONSUMED — the variable
    # was fetched at the start of execute() but never referenced after line 29
    # (chunk_XX_phase_feature_analysis_b.py). The analysis is entirely date-based.
    CONTEXT_CONSUMED = ['val_predictions', 'val_dates', 'val_y_raw',
                        'arch_names', 'optimal_thresholds',
                        'phase4_complete']
    CONTEXT_PRODUCED = ['phaseXb_complete']

    def execute(self, context):
        # 0. Gate: check if required context keys exist (val_predictions, etc.)
        #    If missing: log "[skip] Phase Xb: Temporal Precision Gap validation
        #    skipped (no validator)" and return context unchanged
        # 1. Extract recent (TEMPORAL_GAP_N_DAYS newest) and older dates
        # 2. Compute precision on both subsets for each architecture
        # 3. Log gap analysis
        #    → iter38 log lines 2271-2290
        pass
```

**Key compression**: Feature importance code goes from 2 files (793 lines) to ~200 lines by:
- Inlining the 6 analysis methods into `FeatureImportance` instead of a separate `FeatureImportanceAnalyzer` class
- Parameterizing the 5 correlation thresholds and 4 ranking methods rather than hardcoding each as a separate method
- Removing the report generation and file save logic from the class (moved to `execute()` call site)

**Remove (1,232 lines)**:
- `FeatureImportanceAnalyzer` class and its 14 methods (split into 6 simple helper functions)
- `validate_phaseN_input()` functions — replaced by `CONTEXT_CONSUMED`/`CONTEXT_PRODUCED`
- `validate_phaseN_output()` functions — validation logic inlined into each phase's execute()
  as debug assertions (shape checks, positivity, finiteness). The `validate_phaseN_output()`
  standalone functions are removed; their assertions move into each phase's execute() body.
- All 7 test `__main__` blocks (~150 lines)
- `run_phase_xa()` / `run_phase_xb()` / `run_phase_be()` standalone functions — callers directly instantiate the classes
- Redundant imports across 7 files (consolidated into `phases.py` imports)
- `TemporalWeighting` dead objects: StateManager import + self.evaluator and self.state_manager instantiation
  (These are created but never used in `TemporalWeighting`'s execute path)

---

### 3.8 `phase_4.py` (1,400 lines)

**Source**: `chunk_18_phase_4_ensemble.py` (2,218 lines)

**Structure**:
```python
class ModelTraining(BasePhase):
    """
    Phase 4: The core training pipeline.
    5 evaluation sections per architecture + ensemble + model saving.
    Architecture loop: Section1(threshold search) → Section2(HPO) →
    Section3(election) → Section4(post-HPO) → Section5(final).
    """
    CONTEXT_CONSUMED = ['X', 'y', 'dates', 'temporal_weights', 'raw_target_values',
                        'feature_names', 'threshold_kept_indices',
                        'phase1_complete', 'phase3_complete']
    CONTEXT_PRODUCED = ['arch_names', 'arch_final_metrics', 'arch_winsor_bounds',
                        'optimal_thresholds', 'best_hyperparams_list',
                        'best_val_precision_list', 'final_ensemble',
                        'ensemble_precision', 'ensemble_participants',
                        'val_predictions', 'val_dates', 'val_y_raw',
                        'phase4_complete']
    # DESIGN NOTE: Trained model objects are NOT stored in context (memory).
    # Phase 5 always loads them from disk. SAVE_TRAINED_MODELS=True is required
    # for Phase 5 inference to produce results. When False, Phase 5 logs a
    # warning and returns early (no inference metrics).

    def execute(self, context):
        # =====================================================================
        # SECTION 1: Data Preparation
        # =====================================================================
        # 1. Validate context keys
        # 2. Extract X, y_binary, y_raw, dates, temporal_weights
        # 3. Temporal weighting: X_weighted = X * sqrt(temporal_weights)
        # 4. Time-based train/val split (70/30 by date)
        # 5. Preserve non-winsorized feature copies for per-arch winsorization
        # 6. Build architecture list from config
        #    → iter38 log lines 240-531

        # =====================================================================
        # SECTION 2: Architecture Loop
        # =====================================================================
        for arch_name in architecture_list:
            # --- Phase B: Per-architecture winsorization ---
            # PER_ARCH_WINSORIZE[arch_name] defines low/high percentiles
            # → iter38 log: "Winsor bounds: low=3, high=92" for each arch

            # --- StandardScaler for NN architectures ---
            # → iter38 log: scaler params logged

            # --- Section 1: Threshold Optimization (parameterized) ---
            result1 = self._run_evaluation_section(
                section='section1',
                model=self._build_and_train(config, arch_name, ...),
                X_val=X_val_arch, y_val_raw=y_val_raw,
                thresholds=self.config['THRESHOLDS'],
                arch_name=arch_name
            )
            # → iter38 log lines 540-570: baseline diagnostics
            # → iter38 log lines 571-990: threshold search per arch

            # --- Section 2: Hyperparameter Optimization ---
            if self.config.get('ENABLE_HYPERPARAM_OPTIMIZATION'):
                hpo_result = self._run_hpo(arch_name, X_train, y_train, X_val, y_val)
            # → iter38 log lines 991-1020

            # --- Section 3: HPO Election Gate (3 branches) ---
            # Branch 1: HPO improved → use HPO model (tag: HYPERPARAMETER_OPTIMIZATION)
            # Branch 2: HPO did NOT improve → carry over S1 (tag: PRE-HYPERPARAMETER_OPTIMIZATION)
            # Branch 3: All HPO trials rejected → use S1 baseline (tag: PRE-HYPERPARAMETER_OPTIMIZATION)
            # Safety net: recalculate precision from TP/FP
            #
            # Section 5 model selection (5 branches, not 3):
            #   Branch A: use_post_hpo_model — post-HPO threshold found better operating point
            #     than S3's elected model (post_hpo_prec > section3_precision). Retrain HPO
            #     model with FINAL_TRAIN_EPOCHS even if hpo_improved is False.
            #     Logs: "Retraining post-HPO model at LABEL_THRESHOLD=..."
            #   Branch B: hpo_improved — retrain HPO model with FINAL_TRAIN_EPOCHS
            #     Logs: "Retraining HPO model for N epochs (FINAL_TRAIN_EPOCHS)"
            #   Branch C: has best_hyperparams but no HPO improvement + threshold_opt_model exists
            #     → NO retraining, use threshold_opt_model directly. epochs=3 (metadata only).
            #     Logs: "Using threshold-optimized model (HPO did not improve)"
            #   Branch D: has best_hyperparams but no HPO improvement + no threshold_opt_model
            #     → Build fresh model, retrain with best_hyperparams epochs + sqrt(weights_train).
            #   Branch E: no best_hyperparams at all (default arch params)
            #     → Build default, retrain with FINAL_TRAIN_EPOCHS + sqrt(weights_train).

            # --- Section 4: Post-HPO Threshold Search ---
            if self.config.get('ENABLE_POST_HPO_THRESHOLD_SEARCH'):
                # Uses elected model (S1 or HPO best)
                # NO retraining — only re-evaluate at different thresholds
                # If post-hpo precision > S3 precision → adopt new threshold
                # FALLBACK: if ALL Section 4 thresholds are rejected (always happens
                # in current production), carry forward Section 3 results directly.
                # CRITICAL — Section 3 fallback overrides val_pred and train_pred:
                #   If HPO didn't improve: replace val_pred, train_pred with
                #     threshold_opt_model.predict(...) — Section 1 model's predictions
                #   If HPO improved: replace val_pred with hpo_val_pred
                # This means Section 5 retrained model predictions are DISCARDED
                # when Section 4 is rejected — must NOT forward retrained preds.
                # Must log "Section 4 all rejected - using Section 3 results for Section 5" for
                # log parity. Never let Section 5 receive None from Section 4.
            # → iter38 log lines 1021-1060

            # --- Section 5: Final Training ---
            # 1. Select best model+threshold from S1/S2/S3/S4
            #    If Section 4 all rejected: carry S3 results (PRE-HYPERPARAMETER_OPTIMIZATION or HYPERPARAMETER_OPTIMIZATION)
            # 2. Retrain with FINAL_TRAIN_EPOCHS[arch_name]
            #    Log: "Retraining HPO model for N epochs (FINAL_TRAIN_EPOCHS)"
            #    OR: "Using threshold-optimized model (HPO did not improve)"
            # 3. sample_weight = sqrt(temporal_weights[train_mask])
            # 4. Predict train + val
            #    Log: "Train predictions: mean=X, std=X, min=X, max=X"
            #         "validation predictions: mean=X, std=X, min=X, max=X"
            #         "Train predictions: X% positive predictions (N / N)"
             #         "validation predictions:   X% positive predictions (N / N)"  (3 spaces after colon for alignment)
            # 5. Log diagnostic prediction_buckets via format_diagnostic_string()
            # 6. XGBoost-specific coverage sweep:
            #    search_coverage_thresholds with 9 coverage rates
            #    → iter38 log: XGBoost precision_target=0.60 not met
            # 7. Compute all metrics → arch_final_metrics
            #    Log: "[section 5] [{ARCH}] [final] LABEL_THRESHOLD=X,"
            #         "[section 5] [{ARCH}] [final] train_precision=X ..."
            #         "[section 5] [{ARCH}] [final] VALIDATION_PRECISION=X ..."
            #         "{ARCH} loss=X pred_threshold=X epochs=N"
            # 8. Log: "{ARCH} training time: Xs"
            #    Log: "[pass] {ARCH} trained successfully"
            # → iter38 log lines 1061-1800

        # =====================================================================
        # SECTION 3: Diagnostics
        # =====================================================================
        # Feature stability analysis (FEATURE_STABILITY_ANALYSIS)
        # Inference latency (TRACK_INFERENCE_LATENCY)
        # Sliding window validation (SLIDING_WINDOW_VALIDATION)
        # Permutation importance (PERMUTATION_IMPORTANCE)
        # → iter38 log lines 1800-1900

        # =====================================================================
        # SECTION 4: Ensemble Assembly
        # =====================================================================
        # 1. Filter architectures by minimum validation_precision > ENSEMBLE_MIN_PRECISION (0.50)
        #    Log: "Filtering architectures by minimum validation_precision > 0.50:"
        #    Per arch: "{ARCH}: validation_precision=X PASS/FAIL (Min Req=0.50)"
        #    Log: "ensemble: N architectures ([list])"
        #    Log: "  Ensemble weighting: uniform"
        #    Log diagnostic-ensemble cumulative binary_split_predictions
        # 2. If no archs pass: fallback to FALLBACK_ARCHITECTURE (VAE)
        #    If no archs trained at all: create dummy_ensemble returning 0.5
        #    Log: "No models trained successfully, using fallback"
        #    If archs trained but none pass min precision: fallback to best model
        #    Log: "Skipping {ARCH} (precision X < 0.5)" for each failing arch
        # 3. Union/set of label thresholds from passing archs
        # 4. create_precision_ensemble(models, weights='uniform')
        # 5. Evaluate ensemble on validation set
        #    Log: "Ensemble precision (validation): X (LABEL_THRESHOLD=X, prediction_binary_split=X)"
        # 6. Log ARCHITECTURE RANKING (by validation_precision): rank, arch, metrics
        # 7. Log HPO IMPACT SUMMARY: per-arch pre/post HPO precision + hpo_improved=Yes/No
        # 8. Log TRAINING TIME SUMMARY: per-arch time + total
        # → iter38 log lines 1900-2000: Dense+Transformer pass, ensemble P=0.5003

        # =====================================================================
        # SECTION 5: Model Persistence
        # =====================================================================
        # if SAVE_TRAINED_MODELS:
        # 1. Clean ./saved_models/
        # 2. Save each model: .keras for NN, .joblib for tree
        # 3. Save StandardScaler: .joblib for NN
        # 4. Save metadata: .json (thresholds, hyperparams, winsor bounds)
        # 5. Save temporal_weights.json, feature_names.json
        #    split_date.txt, all_dates.json
        # → iter38 log lines 2000-2100

        return context

    def _run_evaluation_section(self, section, model, X_val, y_val_raw,
                                thresholds, arch_name) -> Dict:
        """
        PARAMETERIZED evaluation section — replaces 5 near-identical inline blocks.
        Used for Section 1 (threshold search) and Section 4 (post-HPO only).
        NOT used for Section 5 — Section 5 evaluation is inline in execute() (see Section 5 step 4-7).
        Section 5 has ~520 lines of unique logic (retraining branches, XGBoost sweep,
        Section 3 fallback override, arch_final_metrics storage) that cannot be
        cleanly parameterized into a generic evaluation function.
        Each call:
        1. model.predict(X_val) → val_pred
        2. Predict validation: check for degenerate cases:
           a. If val_pred.std() < 1e-6: log "[warning] Near-constant predictions"
           b. If no predictions >= 0.5: log "[warning] No predictions >= 0.5!"
           c. If all predictions are negative: log "[warning] Model predicts ALL NEGATIVES"
           d. NaN/Inf in predictions: log warning, clip to [1e-7, 1-1e-7]
           e. Predictions outside [0,1]: log warning, clip to [1e-7, 1-1e-7]
        3. val_binary = (val_pred >= 0.5).astype(int)
        4. For each label threshold: binarize y, compute CM + all metrics
        5. Apply safeguard gates (neural or tree)
        6. If ALL thresholds rejected: log "[warning] All label thresholds rejected
           — using baseline metrics at LT=0.0 (fallback)", return baseline (first LT)
        7. Return best threshold + metrics dict
        """

    def _select_and_retrain_model(self, arch_name, X_train_opt, y_train_optimal,
                                   X_val_opt, y_val_binarized, weights_train, ...) -> Tuple:
        """
        Section 5 model selection + retraining. 5 branches:
          A. use_post_hpo_model: post_hpo_prec > section3_precision → retrain HPO model
          B. hpo_improved: retrain HPO model with FINAL_TRAIN_EPOCHS
          C. best_hyperparams + no HPO improve + threshold_opt_model exists → NO retrain
          D. best_hyperparams + no HPO improve + no threshold_opt_model → fresh retrain
          E. no best_hyperparams → default arch retrain
        All retraining uses sample_weight=np.sqrt(weights_train).
        Returns: (trained_model, training_history, train_epochs, pred_threshold)
        """

    def _apply_section3_fallback(self, threshold_source, hpo_improved, ...) -> Dict:
        """
        When Section 4 is rejected (threshold_source == 'section3'), override
        Section 5's computed val_pred/train_pred/metrics with Section 2/3 values.
        CRITICAL: retrained model predictions are DISCARDED in this path.
          If not hpo_improved: use threshold_opt_model.predict(...) for val_pred + train_pred
          If hpo_improved: use hpo_val_pred for val_pred
        Returns: overridden val_precision, val_recall, val_cm, val_pred, train_pred, etc.
        """

    def _run_hpo(self, arch_name, X_train, y_train, X_val, y_val,
                 input_dim, model_trainer) -> Dict:
        """
        HyperparameterOptimizer.optimize() → election gate logic.
        3-branch result: improved / not-improved / no-HPO.

        Exception handling: if all HPO trials crash or are rejected,
        fall back to 'no-HPO' branch (use baseline from Section 1).
        Each trial is wrapped in try/except to isolate crashes.
        """

    def _select_model(self, section_config) -> str:
        """Return 'hpo' or 'baseline' or 'none' based on election result."""
```

**The parameterized `_run_evaluation_section`** is the key savings — ~600 lines removed (5 × 200 → 5 × 80):

| Before | After |
|---|---|
| 5 inline blocks with duplicated predict→binary→CM→metrics→store | 1 function called 5 times with section config |
| Each section 200+ lines with its own variable names (section1_TP, section2_TP...) | Shared local variables in one function |
| Bug-prone: cross-section variable contamination (Pattern 3 in SPEC §2.4) | Impossible: variables scoped to function call |
| Changes to evaluation logic must be made 5× | Change once in `_run_evaluation_section` |

**Remove (818 lines)**:
- All 5 inline evaluation blocks → `_run_evaluation_section()` parameterized
- `_validate_diagnostics_requirements()` — inlined into execute()
- `_validate_input()` — inlined into constructor
- `_get_arch_feature_dict()` — inlined into architecture loop
- `validate_phase4_input()` / `validate_phase4_output()` — replaced by CONTEXT_CONSUMED/PRODUCED
- Test `__main__` block (~80 lines)
- Duplicated logging format strings (centralized in `_log_section_metrics()`)

---

### 3.9 `pipeline.py` (550 lines)

**Source**: `chunk_20` (553) + `chunk_22` (239) + `chunk_19` (586) + `chunk_13` (213) = **1,591 lines**

**Imports (all merged)**:
```python
import os, sys, time, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
from config import CONFIG, validate_config_structure
from logging import Logger
from data_loader import DataManager
from models import (build_dense_model, build_vae_model, build_cnn_model, ...)
from training import ModelTrainer, HyperparameterOptimizer
from evaluate import Evaluator, format_diagnostic_string, inverse_log_transform
from phases import (DataSetup, TemporalWeighting,
                    FeatureImportance, FeaturePruning,
                    TemporalPrecisionGap)
from phase_4 import ModelTraining
```

**Structure**:

```python
# ============================================================================
# Section 1: Inference (~250 lines)
# ============================================================================
class Inference(BasePhase):
    """
    Inference phase. Models loaded from disk; inference data from
    PipelineOrchestrator context (matching current pipeline behavior).

    CONTEXT_CONSUMED: ['X_inference', 'y_inference', 'dates_inference',
                       'feature_names', 'USE_TEMPORAL_WEIGHTING']
    CONTEXT_PRODUCED: ['final_predictions', 'architecture_results',
                       'inference_complete', 'phase5_complete']

    DESIGN NOTE: The original chunk_19 docstring claimed this phase
    loads fresh data from 'for_train_x_2025_10_24_clean.csv', and the
    code had `data_path = config['DATA_PATH']` (line 87), but this
    variable was NEVER referenced — the real data source has always
    been context['X_inference'] (line 105). The refactored version
    matches actual current behavior (context-based), not the wrong
    docstring. CSV-loading can be reintroduced as a future optimization.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.evaluator = Evaluator(config)
        self.data_manager = DataManager(config)

    def execute(self, context) -> Dict:
        # 0. Load models + scalers + metadata from MODELS_PATH via load_saved_models()
        #    Log: "Loading saved models from ./saved_models..."
        #    Per arch NN: "Loaded {ARCH} model"
        #    Per arch tree: "Loaded {ARCH} model (sklearn)"
        #    Also: "Loaded {ARCH} metadata: label_threshold=X"
        #    If no models on disk: log warning, return context unchanged
        #    DESIGN CONSTRAINT: SAVE_TRAINED_MODELS=True required
        # 1. Receive inference data from context (X_inference, y_inference_continuous,
        #    dates_inference, feature_names) — NO CSV loading
        #    Log: "Receiving inference data from context (NO CSV loading)..."
        #    "  Inference samples: N  Inference date(s): [list]  Feature count: N"
        # 2. Apply temporal weighting (recomputed from dates_inference + TEMPORAL_MULTIPLIER)
        #    Gate: USE_TEMPORAL_WEIGHTING=False → skip, use X_raw directly
        #    Log: "  Inference data: RAW (no temporal weighting applied)"
        #        "  X shape (RAW, no weighting): (N, N)  y shape: (N,)"
        #        "  Temporal weights applied: min=X, max=X"
        # 3. Per-architecture inference loop:
        #    a. Load scaler + winsor bounds from metadata
        #       If scaler missing: log "[warning] No scaler found for {ARCH}"
        #       If winsor bounds missing: log "[warning] Per-arch winsor active but
        #          winsor_bounds missing from {ARCH} metadata"
        #    b. Apply per-arch winsorization, StandardScaler.transform()
        #       Log: "  {ARCH}: using N features (optimal threshold's pruning)"
        #    c. model.predict() → inf_pred
        #       Guard: shape mismatch → log "[skip] Shape mismatch - model expects
        #              different input dimensions: {e}", skip arch
        #       Guard: NaN/Inf → log warning, np.nan_to_num
        #       Guard: outside [0,1] → log warning, clip to [1e-7, 1-1e-7]
        #       Log: "   predictions: mean=X, std=X, min=X, max=X"
        #             "   INFERENCE predictions: X% positive predictions (N / N)"
        #             "   {ARCH} t=X: inference_precision=X ... (full metrics)"
        #    d. Log per-arch results: "{ARCH} inference_precision=X ..."
        #    e. If arch predicted signal rows: "{ARCH} MODEL PREDICTED SIGNAL (N rows)"
        # 4. Consensus voting: min_votes = max(3, len(majority_archs))
        #    Log: "Majority label threshold: X (N archs)"
        #    "Consolidated Predicted Signal (N architectures, min N votes):"
        #    "  Total consensus rows: N"
        # 5. Fallback: best single arch if no consensus or empty
        # 6. Log FINAL PREDICTION RESULTS (sorted by inference_precision)
        # 7. "Best architecture: {ARCH} (inference_precision: X)"
        # 8. Log "phase 5 total time: Xs  data points evaluated: N"
        # 9. Store final_predictions in context
        pass

# ============================================================================
# Section 2: Model Loader Functions (~100 lines)
# ============================================================================
def load_saved_models(models_path: str, config: Dict) -> Dict[str, Any]:
    """
    Unified model loader. Replaces 3 separate calls from chunk_22:
    load_models_with_metadata() + load_scaler().

    NOTE: temporal_weights.json and feature_names.json are NOT loaded here.
    Phase 5 receives these from pipeline context (set by Phase 1/3/4), not from disk.
    The load_preprocessing_params() function from chunk_22 is dead — never called in production.

    Returns {arch_name: {'model': <keras.Model|SklearnModelWrapper>,
                         'scaler': <StandardScaler|None>,
                         'metadata': {...}}}
    If models_path is empty or missing, returns {}.
    """

def load_model_metadata(models_path: str, arch_name: str) -> Dict:
    """Load {arch_name}_metadata.json and return as dict."""

def load_scaler(models_path: str, arch_name: str) -> Any:
    """Load {arch_name}_scaler.joblib or return None."""

# ============================================================================
# Section 3: PipelineOrchestrator (~150 lines)
# ============================================================================
class PipelineOrchestrator:
    """Main pipeline orchestrator. Runs all phases sequentially."""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = Logger(config)
        self.timings = {}

    def run(self) -> Dict:
        # 0. Anchor random seeds for reproducibility
        #    (Must happen BEFORE any phase logic, model init, or sampling.)
        seed = self.config.get('RANDOM_SEED', 42)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # 1. Instantiate phases
        # Design: phases create their own loggers internally (current pattern preserved)
        phases = [
            DataSetup(self.config),
            TemporalWeighting(self.config),
            FeatureImportance(self.config),
            FeaturePruning(self.config),
            ModelTraining(self.config),
            TemporalPrecisionGap(self.config),
        ]

        # 2. Run phases, capture timings
        context = {}
        for phase in phases:
            t0 = time.time()
            context = phase.execute(context)
            elapsed = time.time() - t0
            self.timings[phase.name] = elapsed
            # → iter38 log lines 2435-2442: phase timings

        # 3. Run Phase 5 (depends on saved models)
        phase5 = Inference(self.config)
        context = phase5.execute(context)

        # 4. Metrics Review
        #    → iter38 log lines 2408-2434: architecture ranking
        # Sort arch_final_metrics by precision
        # Log "[stat] Final Results:" + inference_precision, inference_recall,
        #   inference_f1, inference_auc
        # Log "METRICS REVIEW REPORT"
        # Log "[architecture performance] (sorted by validation_precision)"
        # Per arch: "N. {ARCH} validation_precision=X ... PASS/FAIL (Min Req=X)"
        # Compute consensus metrics (final_predictions vs y_inference_binarized)

        # 5. Write metrics_summary.csv
        #    ARCH_CSV_ORDER determines row order
        #    Columns (must match chunk_20:288 exactly — 31 columns):
        #    Architecture,Phase,Loss,Epochs,Precision,Recall,AUC,F1,TP,FP,TN,FN,
        #    MaxPred,MeanPred,StdPred,PctAboveThresh,BestEpoch,TrainingTime,
        #    LabelThresh,ThresholdSource,HPO_Trials,HPO_Improvement,KeyHyperparams,
        #    TrainLoss,ValLoss,LossDelta,MCC,PRAUC,Specificity,BalancedAccuracy,
        #    PredictionThreshold
        #    Log: "[standardized metrics table]" + CSV header + 27 data rows (9 archs × 3 phases)
        #    Log: "[info] CSV saved to metrics_summary.csv"
        #    Log: "END OF METRICS REVIEW"

        # 6. Log phase timings:
        #    "[time] Total execution time: Xs"
        #    "Phase timings:"
        #    "   Pipeline Setup: Xs"   (DataSetup)
        #    "   Temporal Weighting: Xs"
        #    "   Phase Xa: Raw Feature Importance: Xs"
        #    "   Phase BE: Backward Elimination: Xs"
        #    "   Neural Ensemble: Xs"   (ModelTraining)
        #    "   Phase Xb: Temporal Precision Gap: Xs"
        #    "   Prediction Optimization: Xs"   (Inference)

        # 7. Log: "[pipeline] [info] Pipeline execution validated successfully"
        # 8. Print: "Stock Analysis Pipeline completed successfully!"

        return context

# ============================================================================
# Section 4: StateManager (formerly chunk_13) ~50 lines
# ============================================================================
class StateManager:
    """
    Context key-value store. Simplified from chunk_13 — removed
    unused feedback_loop and backtracking features that were never
    triggered in production (20+ iterations without activation).
    
    Note: StateManager is still instantiated in TemporalWeighting
    and ModelTraining but its methods are never called in the
    pipeline (confirmed by grep across all 24 chunk files). The simplified
    class preserves the constructor pattern while removing 100+ lines of
    dead code, including validate_state_manager().
    """
    def __init__(self):
        self.context = {}

    def set(self, key, value):
        self.context[key] = value

    def get(self, key, default=None):
        return self.context.get(key, default)

    def update(self, data: Dict):
        self.context.update(data)

    def keys(self):
        return self.context.keys()

# ============================================================================
# main() (~50 lines)
# ============================================================================
def main():
    config = validate_config_structure(CONFIG)
    orchestrator = PipelineOrchestrator(config)
    context = orchestrator.run()
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
```

**Why Phase 5 + model_loader are here**: Phase 5 depends on Phase 4's saved artifacts (`.keras` files, `.joblib` scalers, `.json` metadata). Moving it into `phases.py` would create a dependency where `phases.py` must import from `models.py` (to load .keras files). By keeping it in `pipeline.py`, the dependency graph stays acyclic: `pipeline.py → models.py` is fine (pipeline is the top-level entry point), but `phases.py → models.py` would create cycles when training.py also imports models.py.

**Remove (1,041 lines)**:
- `_reorder()` — inlined
- Unused feedback_loop logic from StateManager (3 methods never called)
- Backtracking functionality from StateManager (never triggered in 20+ iterations)
- Test `__main__` blocks from all 4 source files (~100 lines)
- `validate_pipeline_execution()` — inlined assertions
- `validate_state_manager()` — dead code
- Duplicate import statements across 4 files

---

## 4. Log-to-Code Map

Every log line in `pipeline_cpu_iter38.log` maps to its refactored source:

| iter38.log lines | Content | New File | Class/Function |
|---|---|---|---|
| 1 | CPU mode, TF-CUDA suppressed | `pipeline.py` | `PipelineOrchestrator.__init__` |
| 3 | CSV file path | `data_loader.py` | `DataManager._load_and_validate_csv` |
| 4 | 6,727,216r x 23c, 937.4 MB | `data_loader.py` | `DataManager._load_and_validate_csv` |
| 5 | Column names | `data_loader.py` | `DataManager._load_and_validate_csv` |
| 6-9 | Sample data rows | `data_loader.py` | `DataManager._load_and_validate_csv` |
| 10 | Sampling config | `pipeline.py` | `PipelineOrchestrator.__init__` |
| 11 | HPO config | `pipeline.py` | `PipelineOrchestrator.__init__` |
| 12 | Label thresholds | `pipeline.py` | `PipelineOrchestrator.__init__` |
| 14-23 | All config values | `pipeline.py` | `PipelineOrchestrator.__init__` |
| 24 | Log1p on 19 features | `data_loader.py` | `DataManager._apply_feature_engineering` |
| 25 | 5 ratio features added | `data_loader.py` | `DataManager._apply_feature_engineering` |
| 26 | Ratio features log1p'd | `data_loader.py` | `DataManager._apply_feature_engineering` |
| 27 | 4 poly interactions added | `data_loader.py` | `DataManager._apply_feature_engineering` |
| 28 | 4 RSI zones added | `data_loader.py` | `DataManager._apply_feature_engineering` |
| 29 | 21 → 34 features | `data_loader.py` | `DataManager._apply_feature_engineering` |
| 31 | Global winsor skipped | `phases.py` | `DataSetup.execute` |
| 32 | Temporal coverage | `logging.py` | `Logger.log_temporal_coverage` |
| 33-35 | Target distribution | `phases.py` | `DataSetup.execute` |
| 38-85 | Class dist. at 5 thresholds | `phases.py` | `DataSetup.execute` |
| 87-121 | Feature quality metrics | `logging.py` | `Logger.log_feature_quality_metrics` |
| 125 | 34 feature names | `phases.py` | `DataSetup.execute` |
| 127-130 | Train/val/inf split sizes | `phases.py` | `DataSetup.execute` |
| 131 | Phase 3 temporal weights | `phases.py` | `TemporalWeighting.execute` |
| 141-194 | Phase Xa feature importance | `phases.py` | `FeatureImportance.execute` |
| 195-239 | Phase BE elimination | `phases.py` | `FeaturePruning.execute` |
| 240-531 | Phase 4 data prep | `phase_4.py` | `ModelTraining.execute` |
| 540-570 | Section 1 baseline | `phase_4.py` | `_run_evaluation_section(section='section1')` |
| 571-990 | Section 2 threshold search | `phase_4.py` | `_run_evaluation_section(section='section2')` |
| 991-1020 | Section 3 HPO election | `phase_4.py` | `_run_hpo()` |
| 1021-1060 | Section 4 post-HPO | `phase_4.py` | `_run_evaluation_section(section='section4')` |
| 1061-1400 | Section 5 final training | `phase_4.py` | `_run_evaluation_section(section='section5')` |
| 1400-1800 | XGBoost coverage sweep | `evaluate.py` | `search_coverage_thresholds()` |
| 1800-1900 | Diagnostics | `phase_4.py` | `ModelTraining` SECTION 3 |
| 1900-2000 | Ensemble assembly | `phase_4.py` | `ModelTraining` SECTION 4 |
| 2000-2100 | Model saving | `phase_4.py` | `ModelTraining` SECTION 5 |
| 2271-2290 | Phase Xb temporal gap | `phases.py` | `TemporalPrecisionGap.execute` |
| 2291-2370 | Phase 5 inference | `pipeline.py` | `Inference.execute` |
| 2371-2390 | Consensus voting | `pipeline.py` | `Inference.execute` |
| 2408-2434 | Architecture ranking | `pipeline.py` | `PipelineOrchestrator.run` metrics review |
| 2435-2442 | Phase timings | `pipeline.py` | `PipelineOrchestrator.run` timing capture |

---

## 5. Dependency Graph

```
                 ┌────────────┐
                 │  config    │  (no external deps — pure dict + stdlib)
                 └─────┬──────┘
                       │
                 ┌─────▼──────┐
                 │  logging   │  (depends: config for LOG_VERBOSITY)
                 └─────┬──────┘
                       │
         ┌─────────────┼──────────────────┐
         │             │                   │
  ┌──────▼─────┐  ┌───▼────┐   ┌─────────▼─────────┐
  │data_loader │  │ models │   │     evaluate       │
  │(config,    │  │(config,│   │ (config, logging)  │
  │ logging)   │  │logging)│   └─────────┬──────────┘
  └──────┬─────┘  └───┬────┘             │
         │             │                  │
         └─────────────┼──────────────────┘
                       │
                 ┌─────▼──────┐
                 │  training  │
                 │ (config,   │
                 │  logging,  │
                 │  models,   │
                 │  evaluate) │
                 └─────┬──────┘
                       │
           ┌───────────┼────────────┐
           │                        │
    ┌──────▼─────┐         ┌───────▼────────┐
    │   phases   │         │    phase_4      │
    │ (config,   │         │ (config, logging,│
    │  logging,  │         │  data_loader,    │
    │  data_loader,│        │  models,         │
    │  evaluate, │         │  evaluate,       │
    │  training) │         │  training)       │
    └──────┬─────┘         └───────┬──────────┘
           │                       │
           └───────────┬───────────┘
                       │
                 ┌─────▼──────┐
                 │  pipeline  │
                 │ (all above)│
                 └────────────┘
```

**No circular dependencies.** Each file depends only on files listed above it.

**Import chain**:
```
pipeline.py → phases.py → training.py → models.py
                   → phase_4.py → training.py → models.py
                   → data_loader.py
                   → evaluate.py
                   → logging.py → config.py
```

---

## 6. Modularization Preserved

Every feature toggle from the current pipeline remains intact:

| Feature | Config Key | Toggle Location | What Happens When Off |
|---|---|---|---|
| Sampling | `USE_SAMPLING` | `data_loader.py:DataManager._apply_stratified_sampling` | All rows pass through unfiltered |
| Max rows | `SAMPLE_SIZE` | `data_loader.py:DataManager._apply_stratified_sampling` | Caps to `min(SAMPLE_SIZE, total)` |
| Feature analysis | `FEATURE_ANALYSIS_ENABLED` | `phases.py:FeatureImportance.execute` first line | Skip phase, return context unchanged |
| Feature importance methods | `FEATURE_IMPORTANCE_METHODS` | `phases.py:FeatureImportance.execute` | Filter active methods from the 6 available |
| Backward elimination | `BACKWARD_ELIMINATION_ENABLED` | `phases.py:FeaturePruning.execute` first line | Skip phase, return context unchanged |
| Temporal weighting | `USE_TEMPORAL_WEIGHTING` | `phases.py:TemporalWeighting.execute` first line + `pipeline.py:Inference.execute` STEP 3 | Phase 3: set `phase3_complete=True`, return unchanged. Phase 5: skip temporal recomputation, use `X_raw` directly |
| Architecture selection | `ACTIVE_ARCHITECTURES` (empty=all) | `phase_4.py:ModelTraining.execute` | Only specified archs are trained |
| HPO | `ENABLE_HYPERPARAM_OPTIMIZATION` | `phase_4.py:ModelTraining._run_hpo` gate | Skip HPO, carry Section 1 directly to Section 3 |
| HPO trials | `HYPERPARAM_OPTIMIZATION_TRIALS` | `training.py:HyperparameterOptimizer.optimize` | Optuna runs N trials |
| Post-HPO search | `ENABLE_POST_HPO_THRESHOLD_SEARCH` | `phase_4.py:ModelTraining._run_evaluation_section` | Skip Section 4, Section 3 results pass through |
| Model saving | `SAVE_TRAINED_MODELS` (default True) | `phase_4.py:ModelTraining.execute` SECTION 5 gate | No files written — Phase 5 no-ops with warning |
| Diagnostics | `FEATURE_STABILITY_ANALYSIS`, `TRACK_INFERENCE_LATENCY`, etc. | `phase_4.py:ModelTraining.execute` SECTION 3 gates | Individual diagnostic skipped |
| Coverage sweep | `PREDICTION_XGBOOST_PRECISION_TARGETING` | `phase_4.py:ModelTraining.execute` XGBoost section | Use standard F1-optimal threshold |
| Neural vs tree archs | `NEURAL_ARCHITECTURES`, `TREE_ARCHITECTURES` | `phase_4.py`, `phases.py` | Lists used for scaler gating, safeguard lookup |
| Log transform | `LOG_TRANSFORM_TARGET`, `LOG_TRANSFORM_FEATURES` | `data_loader.py:DataManager._apply_feature_engineering` | Feature/target values used raw |
| Focal loss | `USE_FOCAL_LOSS`, `FOCAL_LOSS_CONFIG` | `models.py:FocalLoss`, `training.py:ModelTrainer` | Uses BCE instead of focal loss |
| Validation split | `VAL_SPLIT_PERCENTAGE` | `phases.py:DataSetup.execute` | What fraction of dates go to val |
| Held-out dates | `TOP_DATES_HELD_OUT` | `phases.py:DataSetup.execute` | N newest dates for inference |
| Data path | `DATA_PATH` | `data_loader.py:DataManager._load_and_validate_csv` | Any CSV with matching schema |

---

## 7. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **Log format mismatch** | High — downstream parsers (shortmemory.txt) break | Medium | Keep metric keys, section tags, format strings identical. After each step: `diff <(grep metric_key current_log) <(grep metric_key new_log)` |
| **Phase 4 election gate bug** | High — wrong model chosen → different metrics | Low | Unit test the 3-branch HPO election (improved/not-improved/no-HPO) with frozen random seeds. Compare TP/FP counts against current chunk_18 output at each branch. |
| **Metric values differ** | High — invalidates iteration history | Medium | Freeze random seed (np.random.seed, tf.random.set_seed). Compare evaluate_at_threshold() output against current chunk_12 per metric. |
| **Phase 4 section refactoring** | High — 5 inline blocks → 1 parameterized function; each block has unique logic (S1 trains+searches, S2=HPO, S3=election, S4=re-evaluates, S5=retrains with temporal weights). Missing any branch changes every downstream metric. | Medium | After Phase 4 refactoring, compare per-architecture final metrics (val_precision, threshold, ensemble participants) against chunk_18 output. Use frozen RANDOM_SEED=42. |
| **Model builder signature mismatch** | Medium — different predictions | Low | Verify identical model.summary() output for each architecture. Compare weights after first training epoch. |
| **Feature engineering order wrong** | Medium — 34 feature columns different | Low | Unit test _apply_feature_engineering: assert output shape (184408, 34) and feature names identical. |
| **Data cleaning order changes** | Medium — row count different | Low | Unit test load_data: assert same number of NaN/zero rows removed. |
| **Consensus voting divergence** | High — Phase 5 output changes | Low | Verify min_votes and majority threshold logic matches current chunk_19 exactly. |
| **Removed code still needed** | Low — 5% chance | Low | Keep all dormant builders as stubs. Keep all validation as inlined assertions. Test mainline pipeline before removing any file. |
| **Dependency cycle introduced** | High — import error at startup | Low | Follow the dependency graph strictly. `pipeline.py` imports everything, `phases.py` does NOT import `pipeline.py`. |

### Verification Protocol

After each file is created:

```bash
# 1. Syntax check
python -c "import config; import logging; import data_loader; import models; import training; import evaluate; import phases; import phase_4; import pipeline"
# 2. Line count check
wc -l config.py logging.py data_loader.py models.py training.py evaluate.py phases.py phase_4.py pipeline.py
# 3. Log parity (requires test data)
#    Source filenames in [tags] change after 24→9 rename (e.g.
#    chunk_18_phase_4_ensemble.py → phase_4.py), so strip LN count
#    and source before comparing message content:
python pipeline.py | sed 's/^\[LN[0-9]*\] \[[^]]*\] \[[^]]*\] //' | sort > /tmp/new_messages.log
cat pipeline_cpu_iter38.log | sed 's/^\[LN[0-9]*\] \[[^]]*\] \[[^]]*\] //' | sort > /tmp/ref_messages.log
diff /tmp/new_messages.log /tmp/ref_messages.log
#    If diff is empty, all 2,491 log messages (by content) are identical.
#    Source tag mapping is verified separately via §4 Log-to-Code Map.
```

---

## 8. Execution Order

Each step builds on the previous. Steps can be verified independently.

| Step | New File | Source Lines | Target Lines | Key Structural Change | Dependencies |
|---|---|---|---|---|---|
| 1 | `config.py` | 665 | 200 | Remove test code, unused functions | None |
| 2 | `logging.py` | 370 | 200 | Remove unused validators | config |
| 3 | `data_loader.py` | 584+190 | 500 | Merge 2 files, unify interface | config, logging |
| 4 | `models.py` | 619+484+318+508 | 900 | Merge 4 files, unify builder signatures | config, logging |
| 5 | `training.py` | 534+401 | 450 | Merge trainer + HPO, namedtuple return | config, logging, models, evaluate |
| 6 | `evaluate.py` | 757+817 | 700 | Merge metrics + evaluator, eliminate layering | config, logging |
| 7 | `phases.py` | 72+335+182+139+654+281+269 | 700 | Consolidate 7 phases, inline feature importance | config, logging, data_loader, evaluate, training |
| 8 | `phase_4.py` | 2218 | 1400 | Parameterize 5 eval sections | config, logging, data_loader, models, evaluate, training |
| 9 | `pipeline.py` | 553+239+586+213 | 550 | Orchestrator + Inference + model_loader | All above |

**After all 9 steps**: Delete the 24 chunk files (keeping `iter10_reference_snapshot/` as historical reference).

---

## 9. Structural Summary

### 9 Files, ~5,900 Lines (from 24 files, 11,988 — 51% reduction)

| File | Target | Source Files | Source Lines | Saved |
|---|---|---|---|---|
| `config.py` | 200 | `chunk_01_config.py` | 665 | 465 |
| `logging.py` | 200 | `chunk_02_utils_logging.py` | 370 | 170 |
| `data_loader.py` | 500 | `chunk_05` + `chunk_07` | 774 | 274 |
| `models.py` | 900 | `chunk_08` + `chunk_09` + `chunk_10` + `chunk_11` | 1,929 | 1,029 |
| `training.py` | 450 | `chunk_14` + `chunk_21` | 935 | 485 |
| `evaluate.py` | 800 | `chunk_04` + `chunk_12` | 1,574 | 774 |
| `phases.py` | 700 | 7 chunk files | 1,932 | 1,232 |
| `phase_4.py` | 1,400 | `chunk_18_phase_4_ensemble.py` | 2,218 | 818 |
| `pipeline.py` | 550 | `chunk_20` + `chunk_22` + `chunk_19` + `chunk_13` | 1,591 | 1,041 |
| **Total** | **~5,900** | **24 files** | **11,988** | **~6,088** |

### Classes (14 total)

| File | Class | Role |
|---|---|---|
| `logging.py` | `Logger` | 12-method tagged logger |
| `config.py` | — | Stores `ARCH_CSV_ORDER` (CSV row order), all 19 toggles, 30+ config keys |
| `models.py` | `FocalLoss` | Alpha/gamma focal loss Keras layer |
| `models.py` | `SklearnModelWrapper` | Universal sklearn fit/predict/save/load wrapper |
| `training.py` | `KLAnnealingCallback` | VAE KL annealing (warmup=10, max_kl=0.1) |
| `training.py` | `ModelTrainer` | build_architecture + train_model dispatch |
| `training.py` | `HyperparameterOptimizer` | Optuna 3-trial HPO with stagnation stop |
| `evaluate.py` | `Evaluator` | CM + 24-metric evaluate_at_threshold + threshold search |
| `phases.py` | `BasePhase` | ABC with CONTEXT_CONSUMED/PRODUCED contracts |
| `phases.py` | `DataSetup` | CSV→FE→split→store |
| `phases.py` | `TemporalWeighting` | Temporal weights + features (gated by USE_TEMPORAL_WEIGHTING) |
| `phases.py` | `FeatureImportance` | 6-method importance analysis (gated by FEATURE_ANALYSIS_ENABLED) |
| `phases.py` | `FeaturePruning` | Per-arch proxy feature pruning (gated by BACKWARD_ELIMINATION_ENABLED) |
| `phases.py` | `TemporalPrecisionGap` | Recent vs older date precision gap |
| `phase_4.py` | `ModelTraining` | 5-section arch loop + ensemble + persistence + diagnostics |
| `pipeline.py` | `PipelineOrchestrator` | Phase sequencing, timings, metrics_summary.csv |
| `pipeline.py` | `Inference` | Load models, temporal recomputation, inference, consensus |
| `pipeline.py` | `StateManager` | Simplified context key-value store (dead in production) |

### Standalone Functions (~32)

| File | Functions |
|---|---|
| `config.py` | `validate_config_structure()` |
| `data_loader.py` | `extract_temporal_features()`, `apply_temporal_weighting_strategy()`, `validate_temporal_features()` |
| `models.py` | 11 active builders (`build_vae_model`, `build_dense_model`, etc.) + 14 dormant stubs (`build_tabnet_model`, etc.) |
| `evaluate.py` | 18 stateless metric fns + `safe_divide` + `search_coverage_thresholds` + `get_prediction_percentiles` + `format_diagnostic_string` + `inverse_log_transform` + `calculate_temporal_drift` + `calculate_permutation_importance` + `calculate_prediction_entropy` + `calculate_logit_compression` + `calculate_mutual_information` + `analyze_loss_distribution` + `calculate_ks_test` + `calculate_bhattacharyya_distance` |
| `pipeline.py` | `load_saved_models()` + `load_model_metadata()` + `load_scaler()` + `main()` |

### Config Keys (~80+)

All 80+ existing keys preserved. **2 additions:**
| Key | Default | Purpose |
|---|---|---|
| `RANDOM_SEED` | `42` | Reproducibility seed for all RNG |
| `USE_TEMPORAL_WEIGHTING` | `True` | Gate TemporalWeighting + Inference temporal recomputation |

### Feature Toggles (19 total)

| # | Toggle | Default | Gated In | Effect When False |
|---|---|---|---|---|
| 1 | `USE_SAMPLING` | `True` | `data_loader.py` | All rows pass unfiltered |
| 2 | `FEATURE_ANALYSIS_ENABLED` | `False` | `phases.py:FeatureImportance` | Skip feature importance analysis |
| 3 | `BACKWARD_ELIMINATION_ENABLED` | `False` | `phases.py:FeaturePruning` | Skip feature pruning |
| 4 | `USE_TEMPORAL_WEIGHTING` | `True` | `phases.py:TemporalWeighting` + `pipeline.py:Inference` | Skip temporal weighting + skip inference temporal recomputation |
| 5 | `ENABLE_HYPERPARAM_OPTIMIZATION` | `False` | `phase_4.py:ModelTraining` | Skip HPO section |
| 6 | `ENABLE_POST_HPO_THRESHOLD_SEARCH` | `True` | `phase_4.py:ModelTraining` | Skip Section 4 (default True to match IT39 — Section 4 always runs but always rejects; toggle kept for future ability to skip) |
| 7 | `SAVE_TRAINED_MODELS` | `True` | `phase_4.py:ModelTraining` | No files written to disk |
| 8 | `LOG_TRANSFORM_TARGET` | `False` | `data_loader.py` | Target used raw (no log1p) |
| 9 | `LOG_TRANSFORM_FEATURES` | `False` | `data_loader.py` | Features used raw (no log1p) |
| 10 | `USE_FOCAL_LOSS` | `False` | `training.py` | BCE instead of focal loss |
| 11 | `FEATURE_STABILITY_ANALYSIS` | `False` | `phase_4.py:ModelTraining` | Diagnostic skipped |
| 12 | `TRACK_INFERENCE_LATENCY` | `False` | `phase_4.py:ModelTraining` | Diagnostic skipped |
| 13 | `SLIDING_WINDOW_VALIDATION` | `False` | `phase_4.py:ModelTraining` | Diagnostic skipped |
| 14 | `PERMUTATION_IMPORTANCE` | `False` | `phase_4.py:ModelTraining` | Diagnostic skipped |
| 15 | `PREDICTION_XGBOOST_PRECISION_TARGETING` | `False` | `evaluate.py` | Use standard F1-optimal threshold |
| 16 | `ACTIVE_ARCHITECTURES` | `[]` (all) | `phase_4.py:ModelTraining` | Empty = all 9 architectures |
| 17 | `FEATURE_IMPORTANCE_METHODS` | `[]` (all) | `phases.py:FeatureImportance` | Filter from 6 available methods |
| 18 | `PER_ARCH_WINSORIZE` | `{}` | `data_loader.py` + `phase_4.py` | Skip per-arch winsorization |
| 19 | `FOCAL_LOSS_CONFIG` | `{}` | `models.py` | Dict: alpha/gamma params |

### `__main__` Test Blocks

**24 → 1** — only `pipeline.py` retains a test block for integration verification.

---

### Design Constraints & Gaps

| Gap | Severity | Details | Mitigation |
|---|---|---|---|
| Inference disk dependency | Medium | `SAVE_TRAINED_MODELS=True` required for inference. `ModelTraining` does not pass model objects via context. `Inference` gracefully no-ops if disk is empty. | Documented design constraint. If in-memory fallback is needed later, add `trained_models` to `ModelTraining` CONTEXT_PRODUCED. |
| Inference temporal weight sources | Low | (1) context `temporal_weights` (dead — never consumed), (2) `load_saved_models` return (removed — no longer returned), (3) STEP 5 recomputation (active — self-contained) | Cleaned up: sources 1 and 2 are removed. Only STEP 5 recomputation remains. See `USE_TEMPORAL_WEIGHTING` toggle. |
| No `validate_context()` in BasePhase despite being listed as planned | Low | Plan mentions `BasePhase.validate_context` (line 637, line 1389) but BasePhase stub (line 611-642) does not define it. | Add `validate_context()` method to BasePhase that verifies CONTEXT_CONSUMED keys exist and CONTEXT_PRODUCED keys were written. |
| `phase_4.py` at 1,400 lines — largest file, complex 5-section logic | Medium | 1,400 lines is 5x recommended max. Section parameterization reduces duplication but file is dense. | Future work: split into `phase_4_core.py` (execute + _run_hpo) and `phase_4_sections.py` (_run_evaluation_section helpers). |
| `load_saved_models()` type annotation | Low | Plan had `Dict[str, Dict]` but return contains mixed types (Dict, ndarray, List, str). | Fixed: now `Dict[str, Any]` and removed dead return keys `temporal_weights`, `feature_names`, `split_date`. |
| `TemporalPrecisionGap` lists `temporal_weights` in CONTEXT_CONSUMED but never uses it | Low | `chunk_XX_feature_analysis_b:29` — dead variable. The analysis is date-based only. | Remove `temporal_weights` from `TemporalPrecisionGap` CONTEXT_CONSUMED. |

`CONFIG_TYPES` note: Confirmed used by `validate_config_structure()` (`chunk_01:578` iterates over it). Plan correctly preserves this.

---

## Appendix: Current vs. Target Side-by-Side

| Dimension | Current | Target |
|---|---|---|
| Files | 24 | 9 |
| Lines | 11,988 | ~5,800 |
| `__main__` test blocks | 24 | 1 (in pipeline.py) |
| Phase validation functions | 14 (2 per phase) | 1 (BasePhase.validate_context) |
| Metric function duplication | 2 layers (standalone + wrapper) | 1 layer (direct calls) |
| Phase 4 eval sections | 5 inline blocks | 1 parameterized function |
| Model builder signatures | Inconsistent per file | Unified `build_model(config, input_dim, y, loss_fn)` |
| Feature importance methods | 14-method class | 5 helper functions inlined |
| Config access pattern | 3 direct imports + 20 injections | 3 direct imports + 20 injections (preserved) |
| Dependency cycle risk | Low (each chunk is separate) | Zero (strictly acyclic) |
| Log-to-source traceability | Scattered across 24 files | 1:1 mapping via Log-to-Code Map above |
| Time to understand pipeline | Hours (must read 12k lines) | Minutes (read 9 focused files, ~5.8k lines) |
