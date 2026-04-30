# Software Specification Requirements (SSR) - Fraud Detection Ensemble

**Version**: 3.0  
**Date**: 2026-04-15  
**Status**: Living Document - Update After Each Run  

---

# QUICK START GUIDE

## What Is This Document?
SPEC.md is a living document for the Fraud Detection Ensemble Pipeline. It records outputs (Section 1), documents the code (Section 2), maintains history (Section 3), and tracks permanent failures (Section 4).

## Document Structure

| Section | Type | Purpose |
|---------|------|---------|
| Section 1 | Dynamic (update after each run) | Results, metrics, actual values |
| Section 2 | Static (update only if code changes) | Functional specs, configurations |
| Section 3 | Dynamic (update after changes) | Version history, documentation |
| Section 4 | STATIC (NEVER changes) | Failed approaches with evidence |

## How to Use

1. **RUN CODE**: Execute pipeline, generate pipeline_cpu.log
2. **INPUT RESULTS**: Copy actual values from .log into Section 1 templates
3. **CHECK**: Compare actual vs target (e.g., Precision ≥ 0.40)
4. **UPDATE**: If code changed, update Sections 2-3
5. **RECORD FAILURES**: If failed, add to Section 4 with .log evidence

## Quick Lookups

| Need | Section |
|------|---------|
| Expected outputs | Section 1.1.x |
| Metrics measured | Section 1.3 |
| Code for architecture | Section 2.6, 3.3 |
| What failed before | Section 4.x |

---

# SECTION 1: Logging, Reporting, and Metrics

This section contains actual results populated AFTER each code run. Copy values from pipeline_cpu.log into the templates below.

## 1.0 Run Metadata

| Field | Value |
|-------|-------|
| Run Date | _ / _ / _ |
| Run ID / Description | |
| Total Duration | |
| Code Version | |
| Pipeline Status | Pending Review |

---

## 1.1 Runtime Logging

The pipeline produces comprehensive logging during execution. Each architecture in Phase 4 produces logging in 5 sections:

### Section 1: Baseline Diagnostics
Logged before threshold optimization:
- Prediction statistics: mean, std, min, max
- Percentage of positive predictions at threshold 0.5
- Warning if no predictions >= 0.5

**Source File**: chunk_14_models_trainer.py, chunk_18_phase_4_ensemble.py

Example:
```
[BASELINE] Before threshold optimization:
[BASELINE] Predictions: mean=0.0023, std=0.0156, min=0.0001, max=0.4523
[BASELINE] % positive predictions (Prediction_Threshold=0.5): 0.12%
```

### ACTUAL RESULTS - Run #

| Architecture | mean | std | min | max | % Positive | Notes |
|--------------|-----|-----|-----|-----|------------|-------|

### Section 2: Threshold Optimization Results
Logged for each of 11 thresholds (20.0 to 0.0, step -2.0):
- Train metrics: P, R, AUC, F1, TP, FP, TN, FN
- Val metrics: P, R, AUC, F1, TP, FP, TN, FN
- Optimal threshold selection

**Source File**: chunk_12_evaluation_evaluator.py

Example:
```
LightGBM t=20.0 | Train: P=0.8500 R=0.1200 AUC=0.9500 F1=0.2100 FN=45000 TN=120000 TP=5000 FP=880
LightGBM t=20.0 | Val:   P=0.8200 R=0.1100 AUC=0.9400 F1=0.1900 FN=15000 TN=40000 TP=1800 FP=395
LightGBM - OPTIMAL: label_threshold=12.0, Val P=0.7800
```

### ACTUAL RESULTS - Run #

| Architecture | Optimal Threshold | Val_Precision | Val_Recall | Val_AUC | Val_F1 | Pass? |
|--------------|-------------------|---------------|-----------|---------|--------|-------|

### Section 3: Hyperparameter Optimization (HPO)
Logged for each of 20 Optuna trials:
- Trial parameters tested
- Validation metrics: P, R, AUC, F1, TP, FP, TN, FN
- MaxPred, MeanPred values
- Rejection messages if MaxPred < 0.5

**Source File**: chunk_21_hyperparam_optimizer.py

Example:
```
Trial 1/20: n_estimators=500, num_leaves=31, learning_rate=0.05 → Val_P=0.4500 Val_R=0.0200 Val_AUC=0.7200 Val_F1=0.0380 Val_TP=180 Val_FP=220 Val_TN=39800 Val_FN=8920 Val_MaxPred=0.8500 Val_MeanPred=0.0045
```

### ACTUAL RESULTS - Run #

| Architecture | Best Trial# | Best Params | Val_Precision | Val_Recall | Val_AUC | Pass? |
|--------------|-------------|-------------|---------------|-----------|---------|-------|

### Section 4: Post-HPO Threshold Search
Logged after HPO completes:
- Pre-HPO vs Post-HPO comparison
- Final threshold selection

**Source File**: chunk_12_evaluation_evaluator.py

Example:
```
PRE-HPO threshold: t=12.0, Val P=0.7800
POST-HPO threshold: t=10.0, Val P=0.8200
POST-HPO improved: using t=10.0
```

### ACTUAL RESULTS - Run #

| Architecture | Pre-HPO Threshold | Pre-HPO P | Post-HPO Threshold | Post-HPO P | Improved? |
|--------------|-------------------|-----------|-------------------|-----------|-----------|

### Section 5: Final Model Summary
Logged at completion:
- All 17 metrics
- Source (section3 or section4)
- Training epochs

**Source File**: chunk_18_phase_4_ensemble.py

### ACTUAL RESULTS - Run #

| Architecture | Val_Precision | Val_Recall | Val_AUC | Val_F1 | Val_TP | Val_FP | TN | FN | MaxPred | MeanPred | Source |
|--------------|---------------|-----------|---------|--------|--------|--------|----|----|----------|-----------|--------|

## 1.2 Post-Execution Reports

### Metrics Summary
Generated: `metrics_summary.csv`

**Source File**: chunk_20_pipeline_main.py

| Column | Description |
|--------|-------------|
| Architecture | Model name |
| Val_Precision | Validation precision |
| Val_Recall | Validation recall |
| Val_F1 | Validation F1 score |
| Val_AUC | Validation AUC |
| Optimal_Threshold | Best label threshold |

### Feature Importance Report
Generated: `feature_importance_report.txt`, `feature_importance_report.csv`

**Source File**: chunk_XX_feature_importance.py

| Column | Description |
|--------|-------------|
| Feature | Feature name |
| Importance | Importance score |
| Rank | Feature ranking |

### Pipeline Execution Log
Generated: `pipeline_cpu.log`

**Source File**: chunk_02_utils_logging.py (all chunks route logging here)

Contains all runtime output from all phases.

---

## 1.3 Evaluation Metrics

The pipeline calculates and reports the following metrics:

### Primary Metrics

| Metric | Formula | Target | Actual Value | Pass/Fail | Run Date | Source File |
|--------|---------|--------|---------------|-----------|----------|-------------|
|--------|---------|--------|-------------|
| **Precision** | TP / (TP + FP) | ≥ 0.40 | chunk_04_utils_metrics.py |
| **Recall** | TP / (TP + FN) | Maximize | chunk_04_utils_metrics.py |
| **F1 Score** | 2 × (P × R) / (P + R) | Maximize | chunk_04_utils_metrics.py |
| **AUC** | Area under ROC curve | ≥ 0.70 | chunk_04_utils_metrics.py |
| **PR-AUC** | Area under Precision-Recall curve | Maximize | chunk_04_utils_metrics.py |
| **MCC** | Matthews Correlation Coefficient | -1 to 1 | chunk_04_utils_metrics.py |

### Confusion Matrix Components

| Component | Description | Actual Value | Run Date | Source File |
|-----------|-------------|--------------|----------|------------|
| TP | True Positives - Correctly predicted fraud | chunk_04_utils_metrics.py |
| FP | False Positives - Normal predicted as fraud | chunk_04_utils_metrics.py |
| TN | True Negatives - Correctly predicted normal | chunk_04_utils_metrics.py |
| FN | False Negatives - Fraud predicted as normal | chunk_04_utils_metrics.py |

### Prediction Statistics

| Statistic | Description | Actual Value | Run Date | Source File |
|-----------|-------------|--------------|----------|------------|
| pred_mean | Mean prediction value | chunk_14_models_trainer.py |
| pred_std | Standard deviation of predictions | chunk_14_models_trainer.py |
| pred_max | Maximum prediction value | chunk_14_models_trainer.py |
| pred_min | Minimum prediction value | chunk_14_models_trainer.py |

---

## 1.4 Acceptance Criteria

| ID | Criterion | Verification Method | FR Reference |
|----|-----------|---------------------|--------------|
| AC-01 | Pipeline executes without errors | Run `python chunk_20_pipeline_main.py` | Implements all FRs |
| AC-02 | All 10 architectures train successfully | Check logs | → FR-05 |
| AC-03 | Validation precision ≥ 0.40 | Check metrics output | → FR-06, FR-07, FR-11 |
| AC-04 | Models saved to ./saved_models/ | Directory inspection | → FR-09 |
| AC-05 | Predictions generated | Run predict.py | → FR-10 |
| AC-06 | Each chunk independently testable | Run individual chunk files | → All FRs |
| AC-07 | HPO completes 20 trials per architecture | Check chunk_21 logs | → FR-07 |
| AC-08 | Logging captures all 5 sections per architecture | Check phase 4 logs | → Section 1.1.1 to 1.1.5 |

---

## 1.5 Logging Source Reference

### Runtime Logging Sources

| Logging Section | Produces | Source File | Section 2 Reference |
|----------------|----------|------------|---------------------|
| 1.1.1 Baseline Diagnostics | Baseline prediction stats | chunk_14_models_trainer.py, chunk_18_phase_4_ensemble.py | → Section 2.4, Phase 4 |
| 1.1.2 Threshold Opt Results | Per-threshold P,R,F1,AUC | chunk_12_evaluation_evaluator.py | → Section 2.5, FR-06 |
| 1.1.3 HPO Trial Results | Trial parameters, val metrics | chunk_21_hyperparam_optimizer.py | → Section 2.5, FR-07 |
| 1.1.4 Post-HPO Threshold | Pre vs Post comparison | chunk_12_evaluation_evaluator.py | → Section 2.5, FR-06 |
| 1.1.5 Final Model Summary | 17 metrics per model | chunk_18_phase_4_ensemble.py | → Section 2.5, FR-05 |

### Post-Execution Report Sources

| Report | Generated By | Section 2 Reference |
|---------|--------------|---------------------|
| metrics_summary.csv | chunk_20_pipeline_main.py | → Section 2.4, Phase 5 |
| feature_importance_report.* | chunk_XX_feature_importance.py | → Feature analysis |
| pipeline_cpu.log | chunk_02_utils_logging.py (all chunks) | → All sections |

### Evaluation Metrics Sources

| Metric Category | Source File | Implements |
|----------------|------------|-------------|
| P, R, F1, AUC, PR-AUC, MCC | chunk_04_utils_metrics.py | FR-11 |
| Confusion Matrix | chunk_04_utils_metrics.py | FR-11 |
| Prediction Statistics | chunk_14_models_trainer.py | FR-05 |

---

## 1.6 Run Comparison

Use this table to track changes between runs:

| Metric | Previous Run | Current Run | Change | Trend |
|--------|------------|-------------|--------|-------|

# SECTION 2: Functionality

## 2.1 Project Overview

### Problem Statement

Financial fraud results in billions of dollars in losses annually. Traditional rule-based systems miss complex fraud patterns. This project addresses the challenge of detecting fraudulent financial transactions in highly imbalanced datasets (259:1 ratio) using ensemble machine learning.

### Project Details

| Aspect | Specification |
|--------|---------------|
| **Project Name** | Fraud Detection Ensemble Pipeline |
| **Project Type** | Machine Learning Pipeline |
| **Core Functionality** | Detect fraudulent financial transactions using ensemble of neural networks and gradient boosting models |
| **Target Domain** | Financial services / Fraud detection |
| **Language** | Python 3.12 |

---

## 2.2 System Context

| Component | Description |
|-----------|-------------|
| **Input** | CSV file with financial features + date + target |
| **Processing** | Data loading → Temporal weighting → Model training → Ensemble → Prediction |
| **Output** | Fraud predictions (binary + probability), model files, evaluation metrics |
| **Execution Mode** | CPU-only |
| **Mode** | Training vs Inference |

### Data Flow
```
CSV Input → Phase 1 (Data Loading) → Phase 3 (Temporal Weighting) → 
Phase 4 (Training/Ensemble) → Phase 5 (Prediction/Output)
```

---

## 2.3 Data Specifications

### Input Data Requirements

| Attribute | Requirement |
|-----------|--------------|
| **Input Format** | CSV with headers |
| **Date Format** | YYYYMMDD (integer or string) |
| **Target Column** | Binary fraud indicator (0/1) or continuous change value |
| **Dataset Size** | Approximately 6.7 million records |
| **Features** | 16-21 features (after pruning) |
| **Date Range** | 2022-03-01 to 2025-10-23 |
| **Class Imbalance** | 259:1 ratio (0.4% fraud) |

### Data Quality Requirements

| Requirement | Specification |
|-------------|---------------|
| **Null Handling** | Skip records with null values in critical columns |
| **Outlier Handling** | Flag extreme values (e.g., ChangeY > 32500) |
| **Duplicate Handling** | Remove duplicate records |
| **Date Validation** | Reject invalid date formats |

### Pruned Features

The following features were removed during preprocessing:
- `Market_Cap`, `Perf_Month`, `ATR`, `Perf_Week`, `Rel_Volume`

---

## 2.4 Pipeline Phases

| Phase | Description | Skip Note | Outputs | Logging Reference |
|-------|-------------|----------|---------|-------------------|
| Phase 1 | Data loading, preprocessing | - | X, y (continuous), dates | → See Section 1.1.1 |
| Phase 2 | Threshold optimization | ❌ Removed - merged into Phase 4 | N/A | N/A |
| Phase 3 | Temporal feature engineering | - | temporal_weights, dates | → See Section 1.1.1 |
| Phase 4a | Threshold optimization | - | optimal_threshold per architecture | → See Section 1.1.2, Section 1.3 |
| Phase 4b | Hyperparameter optimization | - | best_hyperparams per architecture | → See Section 1.1.3, Section 1.3 |
| Phase 4c | Ensemble creation | - | Combined predictions | → See Section 1.1.4, Section 1.3 |
| Phase 4d | Model persistence | - | ./saved_models/ | → See Section 1.2 |
| Phase 5 | Final evaluation | - | Predictions, metrics, rankings | → See Section 1.2, Section 1.3 |

---

## 2.5 Functional Requirements

| ID | Requirement | Priority | Implementation | Logging Reference |
|----|-------------|----------|----------------|----------------------|
| FR-01 | Load and preprocess CSV data with validation | Required | chunk_05_data_manager.py | → See Section 1.1.1 |
| FR-02 | Handle extreme class imbalance (259:1) using scale_pos_weight | Required | chunk_11_models_sklearn.py | → See Section 1.3 |
| FR-03 | Implement train/validation split (70%/30% temporal) | Required | chunk_16_phase_1_setup.py | → See Section 1.1.1 |
| FR-04 | Apply temporal weighting based on date | Required | chunk_07_data_temporal.py | → See Section 1.1.1 |
| FR-05 | Train multiple architectures: LightGBM, XGBoost, CatBoost, VAE, Dense, CNN, RNN, LSTM, Transformer | Required | chunk_08-11_models_*.py, chunk_18_phase_4_ensemble.py | → See Section 1.1.5 |
| FR-06 | Perform threshold optimization (11 thresholds: 20.0 to 0.0, step -2.0) | Required | chunk_12_evaluation_evaluator.py | → See Section 1.1.2, Section 1.3 |
| FR-07 | Perform hyperparameter optimization (20 Optuna trials) | Required | chunk_21_hyperparam_optimizer.py | → See Section 1.1.3, Section 1.3 |
| FR-08 | Create precision-weighted ensemble | Required | chunk_10_models_ensemble.py | → See Section 1.1.4, Section 1.3 |
| FR-09 | Save trained models to `./saved_models/` | Required | chunk_18_phase_4_ensemble.py | → See Section 1.2 |
| FR-10 | Generate predictions on new data | Required | predict.py, chunk_22_model_loader.py | → See Section 1.1.5, Section 1.3 |
| FR-11 | Evaluate model against precision threshold | Required | chunk_12_evaluation_evaluator.py | → See Section 1.3 |

---

## 2.6 Architecture Specifications

### Model Inventory

| Model | Type | Purpose | Status | Logging Reference |
|-------|------|---------|--------|-------------------|
| LightGBM | Gradient Boosting | Fastest, imbalance-aware | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| XGBoost | Gradient Boosting | Battle-tested | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| CatBoost | Gradient Boosting | Categorical feature handling | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| Boosting_Adaptive | Gradient Boosting | Adaptive boosting | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| VAE | Neural Network | Variational autoencoder | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| Dense | Neural Network | Feed-forward baseline | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| CNN | Neural Network | Convolutional feature extraction | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| RNN | Neural Network | Sequential pattern recognition | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| LSTM | Neural Network | Long-term dependencies | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| Transformer | Neural Network | Attention-based | ✅ Implemented | → Section 1.1.2, 1.1.3 |

---

## 2.7 Configuration

### Core Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| DATA_PATH | for_train_x_2025_10_24_clean.csv | Input CSV file |
| USE_SAMPLING | False | Use entire dataset |
| SAMPLE_SIZE | 99999999 | Use all samples |
| MIN_SAMPLES | 30 | Minimum samples required |
| TARGET_TYPE | continuous | Target type |
| DATE_COLUMN_INDEX | -1 | Auto-detect date column |
| TARGET_COLUMN_INDEX | -1 | Auto-detect target column |
| TEMPORAL_MULTIPLIER | 9.0 | Temporal weighting multiplier |
| INPUT_DIM | Auto-detected | Number of features |

### Threshold Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| FIRST_THRESHOLD | 20.0 | Starting label threshold |
| LAST_THRESHOLD | 0.0 | Ending label threshold |
| THRESHOLD_STEP | -2.0 | Label threshold increment |
| PREDICTION_THRESHOLD | 0.5 | Binary classification threshold |

### Ensemble Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ENSEMBLE_MIN_PRECISION | 0.40 | Minimum precision for ensemble |
| ENSEMBLE_WEIGHTING | precision_weighted | Weighting method |
| ENSEMBLE_VOTE_THRESHOLD | 0.5 | 2/4 models must agree |

### HPO Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ENABLE_HYPERPARAM_OPTIMIZATION | True | Enable HPO |
| HYPERPARAM_OPTIMIZATION_EPOCHS | 20 | Epochs per HPO trial |
| HPO_TRIALS | 20 | Number of Optuna trials |
| ENABLE_POST_HPO_THRESHOLD_SEARCH | True | Run threshold search after HPO |

### Validation Split

| Parameter | Value | Description |
|-----------|-------|-------------|
| VAL_SPLIT_PERCENTAGE | 0.30 | 30% validation split |
| TOP_DATES_HELD_OUT | 2 | Newest dates to hold out |

---

## 2.8 Hyperparameter Search Spaces

### LightGBM
```python
{
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': [100, 200, 500],
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [100, 200, 259],
    'min_child_samples': [50, 100, 200],
    'subsample': [0.7, 0.8, 0.9],
}
```

### XGBoost
```python
{
    'objective': 'binary:logistic',
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [100, 200, 259],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 0.8, 0.9],
}
```

### CatBoost
```python
{
    'iterations': [100, 200, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'auto_class_weights': ['Balanced', 'SqrtBalanced'],
    'l2_leaf_reg': [1, 3, 5],
}
```

### Neural Networks (Dense, VAE, CNN, RNN, LSTM, Transformer)
Each architecture has custom search space defined in `chunk_01_config.py` with parameters like:
- units, layers, dropout, learning_rate, epochs
- alpha, gamma (for focal loss)
- architecture-specific params (latent_dim, filters, etc.)

---

## 2.9 System Constraints

| Constraint | Specification | Notes |
|------------|---------------|-------|
| Platform | Linux (Ubuntu) | Tested on Ubuntu |
| Runtime | CPU-only | No GPU required |
| Python Version | 3.12 | - |
| Memory | ≤ 16GB RAM | Must handle 6.7M records |
| Storage (Models) | ~500MB | Trained model files |
| Storage (Logs) | ~10GB | During execution |

---

## 2.10 Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Extreme class imbalance | Model bias | scale_pos_weight, focal loss |
| Long training time | Cost overruns | GB models run first (faster) |
| Overfitting | Poor generalization | Temporal weights, validation split |
| Memory issues | Crashes | Chunked processing, memory utilities |
| HPO early termination | Missing optimal params | GB models excluded from filter |

---

## 2.11 Execution Order

| # | Architecture | Type | Expected Time |
|---|--------------|------|---------------|
| 1 | LightGBM | Gradient Boosting | ~1-2 hours |
| 2 | XGBoost | Gradient Boosting | ~2-3 hours |
| 3 | CatBoost | Gradient Boosting | ~3-4 hours |
| 4 | VAE | Neural Network | ~50 hours |
| 5 | Dense | Neural Network | ~37 hours |
| 6 | CNN | Neural Network | varies |
| 7 | RNN | Neural Network | varies |
| 8 | LSTM | Neural Network | varies |
| 9 | Transformer | Neural Network | varies |
| 10 | Boosting_Adaptive | Gradient Boosting | varies |

**Note**: All architectures run full searches (no early stopping) before ensemble selection. Each architecture runs Sections 1-5 of Phase 4.

---

# SECTION 3: Documentation

## 3.1 Version History

| Version | Date | Changes | Reason |
|---------|------|---------|--------|
| 1.0 | 2026-04-15 | Initial SSR - Added gradient boosting models | Initial specification |
| 1.1 | 2026-04-15 | Comprehensive improvements - added data quality specs, memory targets | Added specifications for data quality requirements |
| 1.2 | 2026-04-15 | Code synchronization - updated config values | Synced with actual code configuration |
| 2.0 | 2026-04-15 | Reorganized into 3 sections: Logging/Metrics, Functionality, Documentation | Improved document structure |
| 2.1 | 2026-04-15 | Added cross-references between Section 1 and Section 2 | Linked requirements to logging outputs |
| 2.2 | 2026-04-15 | Added cross-references between Section 2 and Section 3 | Linked requirements to code files |
| 2.3 | 2026-04-15 | Added cross-references between Section 1 and Section 3 | Complete bidirectional integration |
| 3.0 | 2026-04-15 | Living document updates - added actual value templates | Converted to dynamic document for post-run updates |

---

## 3.2 Cross-Reference Guide

### Functional Requirements to Logging Mapping

| FR | Produces | Metrics/Logging |
|----|----------|-----------------|
| FR-01 (Data Loading) | Section 1.1.1 | Baseline diagnostics |
| FR-02 (Class Imbalance) | Section 1.3 | Precision, Recall (affected by scale_pos_weight) |
| FR-03 (Train/Val Split) | Section 1.1.1 | Data loading logs |
| FR-04 (Temporal Weighting) | Section 1.1.1 | Temporal feature logs |
| FR-05 (Train Architectures) | Section 1.1.5 | Final model summary |
| FR-06 (Threshold Opt) | Section 1.1.2 | Per-threshold P, R, F1, AUC |
| FR-07 (HPO) | Section 1.1.3 | Trial parameters, validation metrics |
| FR-08 (Ensemble) | Section 1.1.4 | Post-HPO comparison, ensemble weights |
| FR-09 (Save Models) | Section 1.2 | Model files in ./saved_models/ |
| FR-10 (Predictions) | Section 1.1.5 | Prediction output |
| FR-11 (Evaluation) | Section 1.3 | P, R, F1, AUC, MCC |

### Pipeline Phase to Logging Mapping

| Phase | Logging Section | Metrics Produced |
|-------|-----------------|------------------|
| Phase 1 | 1.1.1 | Data stats, shape |
| Phase 3 | 1.1.1 | Temporal weights |
| Phase 4a | 1.1.2 | Threshold optimization results |
| Phase 4b | 1.1.3 | HPO trial results |
| Phase 4c | 1.1.4 | Post-HPO threshold search |
| Phase 4d | 1.2 | Saved model files |
| Phase 5 | 1.2, 1.3 | Final metrics, rankings |

### Acceptance Criteria Mapping

| AC | Verifies | Section 1 Reference |
|----|----------|---------------------|
| AC-01 | No errors | All sections |
| AC-02 | Training success | 1.1.1 to 1.1.5 |
| AC-03 | Precision ≥ 0.40 | 1.3 |
| AC-04 | Models saved | 1.2 |
| AC-05 | Predictions generated | 1.1.5 |
| AC-06 | Chunk testability | All FRs |
| AC-07 | HPO completes | 1.1.3 |
| AC-08 | Logging captured | 1.1.1 to 1.1.5 |

### Functional Requirements to Code File Mapping

| FR | Implemented By | Section 2 Reference |
|----|---------------|----------------|
| FR-01 (Data Loading) | chunk_05_data_manager.py | → Section 2.5 |
| FR-02 (Class Imbalance) | chunk_11_models_sklearn.py (LightGBM, XGBoost, CatBoost) | → Section 2.5 |
| FR-03 (Train/Val Split) | chunk_16_phase_1_setup.py | → Section 2.5 |
| FR-04 (Temporal Weighting) | chunk_07_data_temporal.py | → Section 2.5 |
| FR-05 (Train Architectures) | chunk_08_models_base.py, chunk_09_models_advanced.py, chunk_11_models_sklearn.py, chunk_14_models_trainer.py, chunk_18_phase_4_ensemble.py | → Section 2.5 |
| FR-06 (Threshold Opt) | chunk_12_evaluation_evaluator.py | → Section 2.5 |
| FR-07 (HPO) | chunk_21_hyperparam_optimizer.py | → Section 2.5 |
| FR-08 (Ensemble) | chunk_10_models_ensemble.py | → Section 2.5 |
| FR-09 (Save Models) | chunk_18_phase_4_ensemble.py | → Section 2.5 |
| FR-10 (Predictions) | chunk_22_model_loader.py, predict.py | → Section 2.5 |
| FR-11 (Evaluation) | chunk_12_evaluation_evaluator.py | → Section 2.5 |

---

## 3.2 Related Documentation

| File | Description | Location |
|------|-------------|----------|
| README.md | Pipeline architecture and usage guide | ./README.md |
| SPEC.md | This specification document | ./SPEC.md |
| AGENTS.md | Coding guidelines for this project | ../AGENTS.md |

---

## 3.3 File Inventory

### Source Code (Functionality)

| File | Purpose | FR Reference |
|------|---------|--------------|
| chunk_00_validation_framework.py | Centralized validation utilities | → All FRs (validates) |
| chunk_01_config.py | Configuration and constants | → All FRs (defines config) |
| chunk_02_utils_logging.py | Logger class and formatting | → All FRs (produces Section 1.1.x logs) |
| chunk_03_utils_memory.py | Memory management utilities | → All FRs (memory mgmt) |
| chunk_04_utils_metrics.py | Metric calculation utilities | → FR-06, FR-11 |
| chunk_05_data_manager.py | Data loading and management | → FR-01 |
| chunk_06_data_augmentation.py | Fraud case augmentation | → FR-01 |
| chunk_07_data_temporal.py | Temporal feature extraction | → FR-04 |
| chunk_08_models_base.py | Base neural architectures (VAE, Dense, CNN) | → FR-05 |
| chunk_09_models_advanced.py | Advanced architectures (Transformer, GNN, etc.) | → FR-05 |
| chunk_10_models_ensemble.py | Ensemble builders and aggregators | → FR-08 |
| chunk_11_models_sklearn.py | Scikit-learn model wrappers (LightGBM, XGBoost, CatBoost) | → FR-02, FR-05 |
| chunk_12_evaluation_evaluator.py | Model evaluation utilities | → FR-06, FR-11 |
| chunk_13_state_manager.py | Pipeline state management | → All FRs |
| chunk_14_models_trainer.py | Model training orchestration | → FR-05 |
| chunk_15_phase_base.py | Abstract base phase class | → All Phases |
| chunk_16_phase_1_setup.py | Phase 1: Pipeline setup | → FR-01, FR-03 |
| chunk_17_phase_3_temporal.py | Phase 3: Temporal weighting | → FR-04 |
| chunk_18_phase_4_ensemble.py | Phase 4: Neural ensemble training | → FR-05, FR-06, FR-07, FR-08, FR-09 |
| chunk_19_phase_5_optimization.py | Phase 5: Prediction optimization | → FR-10, FR-11 |
| chunk_20_pipeline_main.py | Main orchestrator | → All FRs |
| chunk_21_hyperparam_optimizer.py | Hyperparameter optimization (Optuna) | → FR-07 |
| chunk_22_model_loader.py | Model loading for predictions | → FR-10 |
| chunk_XX_*.py | Experimental/analysis chunks | → Various FRs |

### Utility Files

| File | Purpose | FR Reference |
|------|---------|--------------|
| predict.py | Prediction script for new data | → FR-10 |
| run_tests.py | Test runner | → All FRs (validation) |
| study9011_enhanced_final.py | Legacy monolithic reference | → All FRs |

### Documentation

| File | Purpose |
|------|---------|
| README.md | Project overview, architecture, usage |
| SPEC.md | This specification |

### Generated Files (Not in GitHub)

| File | Purpose |
|------|---------|
| pipeline_cpu.log | Runtime execution log |
| metrics_summary.csv | Aggregated metrics |
| feature_importance_report.txt/.csv | Feature analysis |
| saved_models/ | Trained model files |

---

## 3.4 Quality Attributes

| Attribute | Requirement |
|-----------|-------------|
| **Maintainability** | Modular chunked architecture (24 files) |
| **Testability** | Each chunk has self-test block |
| **Extensibility** | Easy to add new architectures |
| **Reliability** | Error handling and validation |
| **Performance** | Optimized for CPU execution |
| **Security** | No credentials in code, .gitignore enforced |

---

# SECTION 4: Failed Strategies & Approaches (STATIC - NEVER CHANGES)

This section is a PERMANENT record of failed approaches. Once entered, entries should NEVER be removed or modified. Future iterations should reference this section before attempting new strategies.

Each entry MUST include evidence from .log results and be tagged by date.

## 4.1 Failed Architecture Configurations

| Architecture | Configuration Tried | Failed Reason | Date Tagged | Evidence/Justification (from .log) | Notes |
|--------------|---------------------|----------------|--------------|---------------------------------------|-------|

## 4.2 Failed Hyperparameter Combinations

| Architecture | Hyperparameters | Why Failed | Date Tagged | Evidence/Justification (from .log) | What to Avoid |
|--------------|-----------------|------------|--------------|---------------------------------------|----------------|

## 4.3 Failed Improvement Strategies

| Strategy | Target Area | Failed Outcome | Date Tagged | Evidence/Justification (from .log) | Lessons Learned |
|----------|--------------|---------------|--------------|---------------------------------------|-----------------|

## 4.4 Recommended Avoidances (Based on Past Failures)

DO NOT attempt the following approaches in future iterations:

| Approach | Reason | Date Tagged | Evidence/Justification (from .log) |
|----------|--------|--------------|---------------------------------------|
| (Enter failed approaches here) | (Enter reason) | (Enter date) |

---

*Document generated: 2026-04-15*  
*Last updated: 2026-04-15*  
*Version: 3.0*
