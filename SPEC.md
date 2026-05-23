# Software Specification Requirements (SSR) - Fraud Detection Ensemble

**Version**: 3.5  
**Date**: 2026-05-19  
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
3. **CHECK**: Compare actual vs target (e.g., Precision ≥ 0.60)
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

## Table of Contents

| Section | Description |
|---------|-------------|
| [SECTION 1: Logging, Reporting, and Metrics](#section-1-logging-reporting-and-metrics) | Run results, metrics, runtime logs |
| [SECTION 2: Functionality](#section-2-functionality) | Static specs, architecture, configuration |
| [SECTION 3: Documentation](#section-3-documentation) | Version history, file inventory, cross-references |
| [SECTION 4: Failed Strategies & Approaches](#section-4-failed-strategies--approaches) | Permanent record of failed attempts |
| [IMPLEMENTATION SUMMARY](#implementation-summary---all-improvements-applied) | All improvements applied |

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

### Phase 1: Data Preparation
Phase 1 completes ALL preprocessing before training:
- Feature engineering (ratio features, winsorize, log-transform)
- Data split (train ~70%, val ~30%, inference newest date)
- Stores all splits in context for downstream phases
- Inference data: RAW, no temporal weighting
- Train/Val: temporal weighting applied (Phase 3)

**Source File**: chunk_16_phase_1_setup.py

**Context Fields**: X_train, X_val, X_inference, y_train_continuous, y_val_continuous, y_inference_continuous, dates_train, dates_val, dates_inference, feature_names

### Section 1: Baseline Diagnostics
Logged before threshold optimization:
- Prediction statistics: mean, std, min, max
- Percentage of positive predictions at threshold 0.5
- Warning if no predictions >= 0.5

**Source File**: chunk_14_models_trainer.py, chunk_18_phase_4_ensemble.py

Example (current format with tag reorder):
```
[ARCH_NAME] [BASELINE] Before threshold optimization:
[ARCH_NAME] [BASELINE] Predictions: mean=0.0023, std=0.0156, min=0.0001, max=0.4523
[ARCH_NAME] [BASELINE] % positive predictions (Prediction_Threshold=0.5): 0.12%
```

### ACTUAL RESULTS - Run #2026-05-19

| Architecture | mean | std | min | max | % Positive | Notes |
|--------------|-----|-----|-----|-----|------------|-------|

| Architecture | Optimal Threshold | Val_Precision | Val_Recall | Val_AUC | Val_TP | Val_FP | Notes |
|--------------|-------------------|---------------|-----------|---------|--------|--------|-------|
| CatBoost | 0.0 | **0.5381** | 0.4843 | 0.5195 | 27,034 | 23,209 | Best; stagnant trial 59+ |
| LightGBM | 4.0 | 0.2970 | 0.5734 | 0.6612 | 12,078 | 28,589 | Stagnant trial 90+ |
| XGBoost | 2.0 | 0.2527 | 0.4846 | 0.0000 | 0 | 0 | All CM zeros — severe overfit |
| Dense | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | All thresholds rejected |
| CNN | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | All thresholds rejected |
| RNN | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | All thresholds rejected |
| LSTM | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | All thresholds rejected |
| VAE | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | All thresholds rejected |
| Transformer | 2.0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | All thresholds rejected |

| Architecture | Pre-HPO Threshold | Pre-HPO P | Post-HPO Threshold | Post-HPO P | Improved? |
|--------------|-------------------|-----------|-------------------|-----------|-----------|
| CatBoost | (baseline) | 0.5370 | 0.0 | **0.5381** | Yes |
| LightGBM | (baseline) | 0.1802 | 4.0 | **0.1753** | No |
| XGBoost | (baseline) | 0.2527 | 2.0 | **0.2484** | No |

### ACTUAL RESULTS - Run #2026-05-11

| Architecture | mean | std | min | max | % Positive | Notes |
|--------------|-----|-----|-----|-----|------------|-------|

### Section 2: Threshold Optimization Results
Logged for each of 11 thresholds (20.0 to 0.0, step -2.0):
- Train metrics: 24 metrics (P, TP, TN, FP, FN, MaxPred, MeanPred, R, F1, AUC, Spec, FPR, F2, MCC, PRAUC, BalAcc, Brier, Kappa, Informedness, Markedness, Gini, OptThresh, StdPred, PctAboveThresh)
- Val metrics: Same 24 metrics
- Optimal threshold selection

**Source File**: chunk_12_evaluation_evaluator.py, chunk_18_phase_4_ensemble.py

Example:
```
LightGBM t=20.0 | Train: Train_P=0.8500 Train_TP=5000 Train_TN=120000 Train_FP=880 Train_FN=45000 Train_MaxPred=0.9500 Train_MeanPred=0.0045 Train_R=0.1200 Train_F1=0.2100 Train_AUC=0.9500 Train_Spec=0.9920 Train_FPR=0.0070 Train_F2=0.1800 Train_MCC=0.2100 Train_PRAUC=0.8500 Train_BalAcc=0.5600 Train_Brier=0.0040 Train_Kappa=0.1800 Train_Informedness=0.1200 Train_Markedness=0.8500 Train_Gini=0.9000 Train_OptThresh=0.5500 Train_StdPred=0.0300 Train_PctAboveThresh=0.50
LightGBM t=20.0 | Val:   Val_P=0.8200 Val_TP=1800 Val_TN=40000 Val_FP=395 Val_FN=15000 Val_MaxPred=0.9200 Val_MeanPred=0.0055 Val_R=0.1100 Val_F1=0.1900 Val_AUC=0.9400 Val_Spec=0.9900 Val_FPR=0.0100 Val_F2=0.1500 Val_MCC=0.1900 Val_PRAUC=0.8200 Val_BalAcc=0.5500 Val_Brier=0.0050 Val_Kappa=0.1600 Val_Informedness=0.1100 Val_Markedness=0.8200 Val_Gini=0.8800 Val_OptThresh=0.5200 Val_StdPred=0.0350 Val_PctAboveThresh=0.55
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
Trial 1/30: n_estimators=500, num_leaves=31, learning_rate=0.05 → Val_P=0.4500 Val_TP=180 Val_TN=39800 Val_FP=220 Val_FN=8920 Val_MaxPred=0.8500 Val_MeanPred=0.0045 Val_R=0.0200 Val_F1=0.0380 Val_AUC=0.7200 Val_Spec=0.9945 Val_FPR=0.0055 Val_F2=0.0280 Val_MCC=0.0250 Val_PRAUC=0.4500 Val_BalAcc=0.5100 Val_Brier=0.0045 Val_Kappa=0.0200 Val_Informedness=0.0200 Val_Markedness=0.4400 Val_Gini=0.4400 Val_OptThresh=0.5200
```

### ACTUAL RESULTS - Run #2026-05-11

| Architecture | Best Trial# | Best Params | Val_Precision | Val_Recall | Val_AUC | Notes |
|--------------|-------------|-------------|---------------|-----------|---------|-------|
| CatBoost | 59+ | iterations=200, depth=6, lr=0.1, auto_class_weights=Balanced, l2_leaf_reg=10 | 0.5381 | — | — | Stagnant; Maximize phase needed |
| LightGBM | 110+ | n_estimators=500, num_leaves=63, lr=0.1, scale_pos_weight=500, min_child_samples=100, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8 | 0.1753 | — | — | Stagnant; wider search needed |
| XGBoost | — | n_estimators=500, max_depth=7, lr=0.1, scale_pos_weight=500, min_child_weight=50, reg_alpha=0.5, reg_lambda=5.0, subsample=0.7 | 0.2527 | — | 0.0000 | Severely overfit; Train AUC=0.8806 → Val AUC=0.0000 |
| Dense | — | units=32, layers=2, dropout=0.3, lr=0.0005, epochs=15, alpha=1.0, gamma=3.0 | 0.3689 | — | — | HPO val P (from metadata); Phase 4 training failed |
| CNN | — | filters=32, kernel_size=7, dropout=0.1, lr=0.001, epochs=20, alpha=0.75, gamma=3.0 | 1.0000 | — | — | ARTIFACT — all TP=0, MaxPred=0.0042 |
| LSTM | — | lstm_units=32, dropout=0.05, lr=0.0005, epochs=15, alpha=0.75, gamma=2.0 | 1.0000 | — | — | ARTIFACT — all TP=0, MaxPred=0.0316 |
| RNN | — | units=16, dropout=0.05, lr=0.001, epochs=10, alpha=1.0, gamma=3.0 | 0.4970 | — | — | From metadata; Phase 4 training failed |
| VAE | — | latent_dim=64, lr=0.001, dropout=0.05, alpha=0.75, gamma=2.0 | 0.0000 | — | — | MaxPred<0.5, all trials rejected |
| Transformer | — | dim=32, heads=2, dropout=0.02, lr=0.0005, alpha=1.0, gamma=3.0 | 0.3787 | — | — | From metadata; Phase 4 training failed |

### Phase 5 Results - Run #2026-05-11 (CRASHED at chunk_19 line 272)

| Architecture | Inf_Precision | Inf_Recall | Inf_AUC | Inf_TP | Inf_FP | Inf_TN | Inf_FN | Notes |
|--------------|--------------|-----------|---------|--------|--------|--------|--------|-------|
| LightGBM | 0.1753 | 0.7455 | 0.7177 | 460 | 2164 | 3142 | 157 | Only arch with positive predictions; processed before crash |
| CatBoost | — | — | — | — | — | — | — | Crashed before processing |
| CNN | — | — | — | — | — | — | — | Crashed before processing |
| XGBoost | — | — | — | — | — | — | — | Crashed before processing |
| RNN | — | — | — | — | — | — | — | Crashed before processing |
| Transformer | — | — | — | — | — | — | — | Crashed before processing |
| Dense | — | — | — | — | — | — | — | Crashed before processing |
| VAE | — | — | — | — | — | — | — | Crashed before processing |
| LSTM | — | — | — | — | — | — | — | Crashed before processing |

**Crash**: `AttributeError: 'NoneType' object has no attribute 'columns'` at chunk_19_phase_5_optimization.py line 272. `df_with_all_cols` was None because context lookup failed. Fixed: added `if df_with_all_cols is not None` guard (Fix D). Pipeline failed at Phase 5; fixes A-D applied but not yet re-run.

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
| **Precision** | TP / (TP + FP) | ≥ 0.60 | chunk_04_utils_metrics.py |
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
| AC-03 | Validation precision ≥ 0.60 | Check metrics output | → FR-06, FR-07, FR-11 |
| AC-04 | Models saved to ./saved_models/ | Directory inspection | → FR-09 |
| AC-05 | Predictions generated | Run legacy files/predict.py | → FR-10 |
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
|----------------|-------------|-------------|
| P, R, F1, AUC, PR-AUC, MCC | chunk_04_utils_metrics.py | FR-11 |
| Confusion Matrix | chunk_04_utils_metrics.py | FR-11 |
| Prediction Statistics | chunk_14_models_trainer.py | FR-05 |

### Evaluator Methods Reference (chunk_12)

| Metric (Log Key) | Method Name | Used In Section | Returns |
|----------------|------------|----------------|----------|
| Val_P | calculate_precision() | SECTION 1-5, HPO Trials | Precision (0.0-1.0) |
| Val_R | calculate_recall() | SECTION 1-5, HPO Trials | Recall (0.0-1.0) |
| Val_AUC | calculate_auc() | SECTION 1-5, HPO Trials | AUC (0.0-1.0) |
| Val_F1 | calculate_f1() | SECTION 1-5, HPO Trials | F1 Score (0.0-1.0) |
| Val_TP | Extracted from CM | SECTION 1-5, HPO Trials | True Positives (count) |
| Val_FP | Extracted from CM | SECTION 1-5, HPO Trials | False Positives (count) |
| Val_TN | Extracted from CM | SECTION 1-5, HPO Trials | True Negatives (count) |
| Val_FN | Extracted from CM | SECTION 1-5, HPO Trials | False Negatives (count) |
| Val_MaxPred | pred.max() | SECTION 1-5 | Max prediction |
| Val_MeanPred | pred.mean() | SECTION 1-5 | Mean prediction |
| Val_StdPred | pred.std() | SECTION 1-5 | Std prediction |
| Val_PctAboveThresh | (pred>=0.5).mean() | SECTION 1-5 | % above threshold |
| Val_MCC | calculate_mcc() | SECTION 1-5, HPO Trials | MCC (-1 to 1) |
| Val_PRAUC | calculate_average_precision() | SECTION 1-5, HPO Trials | PR-AUC |
| Val_Spec | calculate_specificity() | SECTION 1-5, HPO Trials | Specificity |
| Val_BalAcc | calculate_balanced_accuracy() | SECTION 1-5, HPO Trials | Balanced Accuracy |
| Val_FPR | calculate_fpr() | SECTION 1-5, HPO Trials | False Positive Rate |
| Val_F2 | calculate_f2_score() | SECTION 1-5, HPO Trials | F2 Score |
| Val_Brier | calculate_brier_score() | SECTION 1-5, HPO Trials | Brier Score (lower=better) |
| Val_Kappa | calculate_kappa() | SECTION 1-5, HPO Trials | Cohen's Kappa |
| Val_Informedness | calculate_informedness() | SECTION 1-5, HPO Trials | Informedness |
| Val_Markedness | calculate_markedness() | SECTION 1-5, HPO Trials | Markedness |
| Val_Gini | calculate_gini() | SECTION 1-5, HPO Trials | Gini Coefficient |
| Val_OptThresh | calculate_optimal_threshold() | SECTION 1-5, HPO Trials | Optimal Threshold |

### Section Mapping

| Section | Description | Metrics Logged |
|---------|-------------|---------------|
| SECTION 1 | Baseline | All 24 standard Val_ metrics |
| SECTION 2 | Pre-HPO Threshold | All 24 standard Val_ metrics |
| HPO Trials | Hyperparameter Opt | All 24 standard Val_ metrics + hyperparams |
| SECTION 3 | HPO Results | All 24 standard Val_ metrics + Best hyperparams |
| SECTION 4 | Post-HPO Threshold | All 24 standard Val_ metrics + Source tag |
| SECTION 5 | Final Model | All 24 standard Val_ metrics + 18 Train_ metrics |
| Threshold Search (Train) | Per-threshold train | 8 Train_ metrics |
| Threshold Search (Val) | Per-threshold val | 8 Val_ metrics |

### Training Metrics

All 18 training metrics are reported in SECTION 5 (Final) with Train_ prefix:

| Metric | Prefix | Description | Formula |
|--------|--------|-------------|---------|
| Train_P | Train_ | Precision | TP / (TP + FP) |
| Train_TP | Train_ | True Positives | count(y=1 & pred=1) |
| Train_TN | Train_ | True Negatives | count(y=0 & pred=0) |
| Train_FP | Train_ | False Positives | count(y=0 & pred=1) |
| Train_FN | Train_ | False Negatives | count(y=1 & pred=0) |
| Train_MaxPred | Train_ | Max Probability | predictions.max() |
| Train_MeanPred | Train_ | Mean Probability | predictions.mean() |
| Train_R | Train_ | Recall (Sensitivity) | TP / (TP + FN) |
| Train_F1 | Train_ | F1 Score | 2*P*R / (P+R) |
| Train_AUC | Train_ | ROC-AUC | sklearn roc_auc_score |
| Train_Spec | Train_ | Specificity | TN / (TN + FP) |
| Train_FPR | Train_ | False Positive Rate | FP / (FP + TN) |
| Train_F2 | Train_ | F2 Score | 5*P*R / (4*P+R) |
| Train_MCC | Train_ | Matthews Corr Coef | sklearn matthews_corrcoef |
| Train_PRAUC | Train_ | Precision-Recall AUC | sklearn avg_precision_score |
| Train_BalAcc | Train_ | Balanced Accuracy | (Sens + Spec) / 2 |
| Train_StdPred | Train_ | Std Probability | predictions.std() |
| Train_PctAboveThresh | Train_ | % Above Threshold 0.5 | (pred >= 0.5).mean() * 100 |

### Validation Metrics

All 24 validation metrics are reported in SECTIONS 1-5 with Val_ prefix:

| Metric | Prefix | Description | Formula |
|--------|--------|-------------|---------|
| Val_P | Val_ | Precision | TP / (TP + FP) |
| Val_TP | Val_ | True Positives | count(y=1 & pred=1) |
| Val_TN | Val_ | True Negatives | count(y=0 & pred=0) |
| Val_FP | Val_ | False Positives | count(y=0 & pred=1) |
| Val_FN | Val_ | False Negatives | count(y=1 & pred=0) |
| Val_MaxPred | Val_ | Max Probability | predictions.max() |
| Val_MeanPred | Val_ | Mean Probability | predictions.mean() |
| Val_R | Val_ | Recall (Sensitivity) | TP / (TP + FN) |
| Val_F1 | Val_ | F1 Score | 2*P*R / (P+R) |
| Val_AUC | Val_ | ROC-AUC | sklearn roc_auc_score |
| Val_Spec | Val_ | Specificity | TN / (TN + FP) |
| Val_FPR | Val_ | False Positive Rate | FP / (FP + TN) |
| Val_F2 | Val_ | F2 Score | 5*P*R / (4*P+R) |
| Val_MCC | Val_ | Matthews Corr Coef | sklearn matthews_corrcoef |
| Val_PRAUC | Val_ | Precision-Recall AUC | sklearn avg_precision_score |
| Val_BalAcc | Val_ | Balanced Accuracy | (Sens + Spec) / 2 |
| Val_StdPred | Val_ | Std Probability | predictions.std() |
| Val_PctAboveThresh | Val_ | % Above Threshold 0.5 | (pred >= 0.5).mean() * 100 |
| Val_Brier | Val_ | Brier Score | mean((pred - y)^2) |
| Val_Kappa | Val_ | Cohen's Kappa | sklearn cohen_kappa_score |
| Val_Informedness | Val_ | Informedness | Sensitivity + Specificity - 1 |
| Val_Markedness | Val_ | Markedness | Precision + NPV - 1 |
| Val_Gini | Val_ | Gini Coefficient | 2 * AUC - 1 |
| Val_OptThresh | Val_ | Optimal Threshold | Youden's J (max TPR - FPR) |

### Inference Metrics

All 24 inference metrics are reported in Phase 5 with Inf_ prefix:

| Metric | Prefix | Description | Formula |
|--------|--------|-------------|---------|
| Inf_P | Inf_ | Precision | TP / (TP + FP) |
| Inf_TP | Inf_ | True Positives | count(y=1 & pred=1) |
| Inf_TN | Inf_ | True Negatives | count(y=0 & pred=0) |
| Inf_FP | Inf_ | False Positives | count(y=0 & pred=1) |
| Inf_FN | Inf_ | False Negatives | count(y=1 & pred=0) |
| Inf_MaxPred | Inf_ | Max Probability | predictions.max() |
| Inf_MeanPred | Inf_ | Mean Probability | predictions.mean() |
| Inf_StdPred | Inf_ | Std Probability | predictions.std() |
| Inf_PctAboveThresh | Inf_ | % Above Threshold 0.5 | (pred >= 0.5).mean() * 100 |
| Inf_R | Inf_ | Recall (Sensitivity) | TP / (TP + FN) |
| Inf_F1 | Inf_ | F1 Score | 2*P*R / (P+R) |
| Inf_AUC | Inf_ | ROC-AUC | sklearn roc_auc_score |
| Inf_Spec | Inf_ | Specificity | TN / (TN + FP) |
| Inf_FPR | Inf_ | False Positive Rate | FP / (FP + TN) |
| Inf_F2 | Inf_ | F2 Score | 5*P*R / (4*P+R) |
| Inf_MCC | Inf_ | Matthews Corr Coef | sklearn matthews_corrcoef |
| Inf_PRAUC | Inf_ | Precision-Recall AUC | sklearn avg_precision_score |
| Inf_BalAcc | Inf_ | Balanced Accuracy | (Sens + Spec) / 2 |
| Inf_Brier | Inf_ | Brier Score | mean((pred - y)^2) |
| Inf_Kappa | Inf_ | Cohen's Kappa | sklearn cohen_kappa_score |
| Inf_Informedness | Inf_ | Informedness | Sensitivity + Specificity - 1 |
| Inf_Markedness | Inf_ | Markedness | Precision + NPV - 1 |
| Inf_Gini | Inf_ | Gini Coefficient | 2 * AUC - 1 |
| Inf_OptThresh | Inf_ | Optimal Threshold | Youden's J (max TPR - FPR) |

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

| Phase | Description | Outputs | Logging Reference |
|-------|-------------|---------|-------------------|
| Phase 1 | Data loading, preprocessing, split (train/val/inference) | X_train, X_val, X_inference in context | → See Section 1.1 |
| Phase 2 | Threshold optimization | ❌ Removed - merged into Phase 4 | N/A |
| Phase 3 | Temporal weighting generation | temporal_weights in context | → See Section 1.1 |
| Phase Xa | Feature importance (11 thresholds, per-threshold pruning) | threshold_kept_indices dict, all 24 features kept in context['X'] | → Feature analysis |
| Phase 4a | Threshold optimization | optimal_threshold per architecture | → See Section 1.1.2, Section 1.3 |
| Phase 4b | Hyperparameter optimization (Optuna) | best_hyperparams per architecture | → See Section 1.1.3, Section 1.3 |
| Phase 4c | Ensemble creation | Combined predictions | → See Section 1.1.4, Section 1.3 |
| Phase 4d | Model persistence | ./saved_models/ | → See Section 1.2 |
| Phase Xb | Temporal precision gap analysis | Recent vs older precision gap | → Precision analysis |
| Phase 5 | Final inference (raw data, no temporal weighting) | Predictions, metrics, rankings | → See Section 1.2, Section 1.3 |

**Note**: Train/Val use temporal weighting; Inference uses RAW data from context (no weighting)

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
| FR-10 | Generate predictions on new data | Required | legacy files/predict.py, chunk_22_model_loader.py | → See Section 1.1.5, Section 1.3 |
| FR-11 | Evaluate model against precision threshold | Required | chunk_12_evaluation_evaluator.py | → See Section 1.3 |
| FR-12 | Apply log transform (Option C: sign * log1p(|y|)) to handle extreme target values | Improvement (Step 1) | chunk_05_data_manager.py, chunk_04_utils_metrics.py | → See Section 2.7 |

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
| LOG_TRANSFORM_TARGET | True | Apply log1p transform to target (Option C: sign * log1p(\|y\|)) |
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
| ENSEMBLE_MIN_PRECISION | 0.60 | Minimum precision for ensemble |
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

## 2.8 Hyperparameter Search Spaces (GIS (Global Iteration Strategy) Reconfiguration - May 13, 2026)

Root cause diagnosis: All 6 NNs produced **MaxPred << 0.5** (CNN: 0.0042, LSTM: 0.0316, RNN: 0.0661, VAE: 0.0919, Transformer: 0.0443) — search space too conservative. Trees stagnated due to small search space and missing key parameters (colsample, gamma). GIS Iteration 1 started with CatBoost 0.5381 → Maximize phase.

### CatBoost (Iteration 1: Maximize phase)
```python
{
    'iterations': [300, 400, 500],      # was [100, 200]
    'depth': [4, 5, 6],                 # was [6, 8] — shallower to reduce overfit
    'learning_rate': [0.03, 0.05, 0.08, 0.1],   # finer granularity, lower start
    'auto_class_weights': ['Balanced', 'SqrtBalanced'],  # keep
    'l2_leaf_reg': [1, 3, 5, 7],        # was [3, 5, 10] — lower reg option
}
```

### LightGBM (Iteration 1)
```python
{
    'n_estimators': [300, 500, 800],    # was [200, 500]
    'num_leaves': [31, 63, 127],        # was [31, 63]
    'learning_rate': [0.03, 0.05, 0.08],  # added lower LR
    'scale_pos_weight': [300, 400, 500, 700],  # was [400, 500]
    'min_child_samples': [50, 100, 200],  # was [100, 200]
    'reg_alpha': [0.01, 0.1, 0.5, 1.0],  # was [0.1, 0.5]
    'reg_lambda': [0.5, 1.0, 5.0, 10.0],  # was [1.0, 5.0]
    'subsample': [0.6, 0.7, 0.8, 0.9],  # was [0.7, 0.8]
    'colsample_bytree': [0.6, 0.8, 1.0],  # NEW — feature subsampling
    'min_split_gain': [0.0, 0.01, 0.1],  # NEW — min gain to split
}
```

### XGBoost (Iteration 1 — severe overfit: Train AUC 0.88 → Val AUC 0.00)
```python
{
    'n_estimators': [100, 200, 300, 500],  # added smaller to reduce overfit
    'max_depth': [3, 5, 7],             # added shallower depth
    'learning_rate': [0.01, 0.03, 0.05, 0.1],  # lower start, finer grid
    'scale_pos_weight': [200, 400, 500],  # lower min to reduce over-prediction
    'min_child_weight': [10, 50, 100, 200],  # was [50, 100, 200]
    'reg_alpha': [0.0, 0.1, 0.5, 1.0],  # wider L1
    'reg_lambda': [1.0, 5.0, 10.0],     # higher L2 for overfit control
    'subsample': [0.6, 0.7, 0.8],       # keep
    'colsample_bytree': [0.5, 0.7, 1.0],  # NEW — feature subsampling
    'gamma': [0, 0.1, 0.5],             # NEW — min split loss reduction
}
```

### Dense (Iteration 2 — MaxPred max 1.0 but all thresholds rejected)
```python
{
    'units': [64, 128, 256, 512, 1024],  # was [32, 64, 128, 256, 512]
    'layers': [2, 3, 4],                  # was [1, 2, 3] — deeper networks
    'dropout': [0.1, 0.2, 0.3, 0.4],     # was [0.05, 0.1, 0.2, 0.3]
    'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],  # finer granularity
    'epochs': [15, 20, 30, 40],          # was [8, 10, 12, 15, 20]
    'alpha': [1.0, 1.25, 1.5],           # was [0.5, 0.75, 1.0, 1.25]
    'gamma': [2.0, 2.5, 3.0, 4.0],      # was [1.0, 2.0, 2.5, 3.0, 3.5]
    'batch_size': [32, 64, 128, 256],    # NEW
    'activation': ['relu', 'leaky_relu', 'selu'],  # NEW
}
```

### CNN (Iteration 2 — MaxPred max 0.0042, completely broken)
```python
{
    'loss_function': ['binary_crossentropy', 'focal_loss'],
    'filters': [64, 128, 256, 512],    # was [32, 64, 128, 256]
    'kernel_size': [3, 5, 7, 11],      # was [3, 5, 7] — larger receptive field
    'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
    'learning_rate': [0.0005, 0.001, 0.002, 0.005, 0.01],  # wider range
    'epochs': [30, 50, 80, 100],        # much more training
    'alpha': [0.75, 1.0],
    'gamma': [2.0, 2.5, 3.0],
    'layers': [1, 2, 3],               # NEW — number of conv layers
    'pooling': ['max', 'avg', 'none'],  # NEW — pooling type
}
```

### RNN (Iteration 3 — MaxPred max 0.0661)
```python
{
    'loss_function': ['binary_crossentropy', 'focal_loss'],
    'units': [64, 128, 256],            # was [32, 64, 128]
    'dropout': [0.0, 0.05, 0.1],       # allow no dropout
    'learning_rate': [0.0005, 0.001, 0.002, 0.005],
    'epochs': [20, 30, 50],            # was [10, 15, 20, 30]
    'alpha': [0.75, 1.0, 1.25],
    'gamma': [2.0, 2.5, 3.0, 3.5],
    'layers': [1, 2],                   # NEW — number of RNN layers
}
```

### LSTM (Iteration 3 — MaxPred max 0.0316, completely broken)
```python
{
    'lstm_units': [32, 64, 128, 256],  # MAJOR expansion from [8, 16, 32]
    'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
    'learning_rate': [0.0005, 0.001, 0.002, 0.005],
    'epochs': [20, 30, 50],            # was [12, 15, 20, 25]
    'alpha': [0.75, 1.0],
    'gamma': [2.0, 2.5, 3.0],
    'layers': [1, 2],                  # NEW — number of LSTM layers
    'bidirectional': [True, False],     # NEW
}
```

### VAE (Iteration 4 — MaxPred max 0.0919, completely broken)
```python
{
    'loss_function': ['binary_crossentropy', 'focal_loss'],
    'latent_dim': [32, 64, 128, 256],  # added smaller dimension
    'learning_rate': [0.0005, 0.001, 0.002, 0.005],  # higher LR options
    'dropout': [0.0, 0.02, 0.05, 0.1],  # allow no dropout
    'epochs': [30, 50, 80],            # NEW — training epochs
    'alpha': [0.75, 1.0, 1.25],
    'gamma': [2.0, 2.5, 3.0],
    'encoder_layers': [1, 2, 3],      # NEW — encoder depth
    'decoder_layers': [1, 2, 3],      # NEW — decoder depth
}
```

### Transformer (Iteration 4 — MaxPred max 0.0443, very narrow)
```python
{
    'loss_function': ['binary_crossentropy'],  # keep (focal removed)
    'dim': [64, 128, 256],           # was [32, 64] — was 128 removed for NaN, restored with lower LR
    'heads': [2, 4, 8],              # was [1, 2]
    'dropout': [0.0, 0.05, 0.1, 0.2],  # allow no dropout
    'learning_rate': [0.00005, 0.0001, 0.0002],  # even lower for stability
    'epochs': [20, 30, 50],           # NEW — training epochs
    'alpha': [0.75, 1.0, 1.25],
    'gamma': [1.5, 2.0, 2.5, 3.0],
    'ff_dim': [64, 128, 256],         # NEW — feed-forward dimension
    'layers': [1, 2, 4],              # NEW — transformer layers
}
```

### HPO Control Parameters (updated May 13, 2026)
| Parameter | Was | Now | Rationale |
|-----------|-----|-----|-----------|
| `HPO_STAGNATION_THRESHOLD` | 30 | **50** | More room for exploration in wider search spaces |
| `HYPERPARAM_OPTIMIZATION_TRIALS` | 30 | **60** | 2x base trials per phase |
| `max_trials` (safety cap in chunk_21) | 500 | **1000** | Allow deeper maximization phases |

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

## 2.11 Execution Order (Discovery Sequence - May 11, 2026)

Discovery sequence for dataset understanding and precision optimization:

| # | Architecture | Group | Discovery Step | Expected Time |
|---|--------------|-------|----------------|---------------|
| 1 | CatBoost | Gradient Boosting | Step 1: Feature importance | ~3-4 hours |
| 2 | LightGBM | Gradient Boosting | Feature validation | ~1-2 hours |
| 3 | XGBoost | Gradient Boosting | Feature validation | ~2-3 hours |
| 4 | Dense | Neural Network | Step 2: Global interactions | ~37 hours |
| 5 | CNN | Neural Network | Step 3: Local patterns | varies |
| 6 | RNN | Neural Network | Step 4a: Temporal patterns | varies |
| 7 | LSTM | Neural Network | Step 4b: Temporal patterns | varies |
| 8 | VAE | Neural Network | Step 5: Latent structure | ~50 hours |
| 9 | Transformer | Neural Network | Step 6: Attention maps | varies |

**Mission**: Optimize val_precision ≥ 0.60 for each architecture independently.

**Note**: All architectures run full searches (unlimited trials until target met). Each architecture runs Sections 1-5 of Phase 4.

---

## 2.12 Iteration Strategy (May 11, 2026)

### Strategy Summary

| # | Decision | Value |
|---|----------|-------|
| 1 | Iteration method | **Grouped** - Run by architecture group |
| 2 | Trials per phase | **As deep as necessary** (no cap) |
| 3 | Stop condition | **P ≥ 0.60** (then move to next arch) |
| 4 | Config updates | **Yes** (incorporate findings between iterations) |

### Iteration Plan

| Iteration | Architectures | Target | Expected Insights |
|-----------|--------------|--------|-------------------|
| **1** | CatBoost→LightGBM→XGBoost | P ≥ 0.60 | Feature importance, tree depth |
| **2** | Dense→CNN | P ≥ 0.60 | Global vs local patterns |
| **3** | RNN→LSTM | P ≥ 0.60 | Temporal signal strength |
| **4** | VAE→Transformer | P ≥ 0.60 | Latent dimension, attention |
| **5** | Ensemble | P ≥ 0.60 | Combined precision |

### Deep Exploration Phases

| Phase | Trials | Condition to Proceed |
|-------|--------|---------------------|
| **Coarse** | 30 | Standard exploration |
| **Refine** | 30+ | Continue if P improving but < 0.60 |
| **Maximize** | Until P ≥ 0.60 | Stop immediately when target met |
| **Evidence** | 90+ | If stagnant, document and move on |

### Decision Rules

| Condition | Action |
|-----------|--------|
| P ≥ 0.60 AND TP > 0 | Save config, move to next architecture |
| P < 0.60 but improving | Continue with expanded search |
| P < 0.60, stagnant | Document findings, move to next arch |
| MaxPred < 0.3 | Architecture issue - document, move on |

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
| 3.1 | 2026-05-11 | Updated precision target to 0.60, discovery sequence execution order | Mission-focused optimization |
| 3.3 | 2026-05-13 | GIS hyperparameter reconfiguration — all 9 search spaces expanded, HPO thresholds raised (stagnation 30→50, trials 30→60, cap 500→1000), Phase 4 results logged (CatBoost 0.5381 best, 6 NNs broken), Phase 5 crash documented in Section 4.7, fixes A-D applied | Phase 4: CatBoost 0.5381, LightGBM 0.2970, XGBoost 0.2527; 6 NNs all MaxPred<<0.5; Phase 5 crashed line 272 — df_with_all_cols None guard fix |
| 3.4 | 2026-05-18 | GIS (Global Iteration Strategy) SUCCESS — CatBoost achieved 0.7204 inference precision (>0.60 target), Phase 5 fixes (KeyError: 'precision' → 'Inf_P', shape mismatch handling, df_filtered→n_inference, inference_date→dates_inference[0]), sample size reduced 368816→184408 | CatBoost: 0.7204 Inf_P; LightGBM: 0.2722; XGBoost: 0.2751; 5 NNs skipped (shape mismatch); sample reduced for faster runs |
| 3.5 | 2026-05-19 | Logging standardization (tag reorder, terminology), per-threshold feature pruning architecture, feature importance logging overhaul, HPO logging improvements (best trial tracking, [BEST TRIAL] format, [OPTIMAL] expanded), Section 2 redundant logs removed with stale y_val_binarized fix | Tag reorder: [BASELINE] {arch_tag}→{arch_tag} [BASELINE]; Phase Xa stores threshold_kept_indices dict; Phase 4+5 per-threshold feature slicing; _log_top_features()→_log_all_features(); 13 redundant log lines removed; stale-variable bug identified (dropped_indices on lines 187-188) |

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
| AC-03 | Precision ≥ 0.60 | 1.3 |
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
| FR-10 (Predictions) | chunk_22_model_loader.py, legacy files/predict.py | → Section 2.5 |
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
| legacy files/chunk_00_validation_framework.py | Centralized validation utilities | → All FRs (validates, moved to legacy) |
| chunk_01_config.py | Configuration and constants | → All FRs (defines config) |
| chunk_02_utils_logging.py | Logger class and formatting | → All FRs (produces Section 1.1.x logs) |
| legacy files/chunk_03_utils_memory.py | Memory management utilities | → All FRs (memory mgmt, moved to legacy) |
| chunk_04_utils_metrics.py | Metric calculation utilities | → FR-06, FR-11 |
| chunk_05_data_manager.py | Data loading and management | → FR-01 |
| legacy files/chunk_06_data_augmentation.py | Fraud case augmentation | → FR-01 (moved to legacy) |
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
| legacy files/predict.py | Prediction script for new data | → FR-10 (moved to legacy) |
| legacy files/run_tests.py | Test runner | → All FRs (validation, moved to legacy) |
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

## 4.1-4.4 Archived Failure Records

Detailed failure records (failed configs, hyperparameters, strategies, avoidances) are documented inline in Sections 4.5-4.7 below. Empty template tables were removed — add new entries following the format used in 4.5-4.7.

| Type | Location |
|------|----------|
| NN Prediction Range Failures | §4.5 |
| XGBoost Train-Val Gap | §4.6 |
| Phase 5 Crash | §4.7 |
## 4.5 NN Prediction Range Failures (May 11, 2026)

| Architecture | MaxPred | Root Cause | Fix Applied | Evidence |
|--------------|---------|-----------|-------------|-----------|
| CNN | 0.0042 | Filters [32-256], kernel [3-7], LR [0.001-0.005] too conservative | Expanded to [64-512], kernel 11, LR [0.0005-0.01], epochs [30-100], pooling, layers | pipeline_cpu.log lines 1184-1211 |
| LSTM | 0.0316 | lstm_units [8-32] too small, focal_loss removed | Expanded to [32-256], bidirectional, layers, epochs [20-50] | pipeline_cpu.log lines 1242-1271 |
| RNN | 0.0661 | units [32-128], LR [0.001-0.005] insufficient | Expanded to [64-256], epochs [20-50], layers | pipeline_cpu.log lines 1212-1241 |
| VAE | 0.0919 | latent_dim [64-128], LR [0.0005-0.0015] too narrow | Expanded to [32-256], LR [0.0005-0.005], encoder/decoder depth | pipeline_cpu.log lines 1272-1301 |
| Transformer | 0.0443 | dim [32-64], heads [1-2] insufficient | Expanded to [64-256], heads [2-8], ff_dim, layers | pipeline_cpu.log lines 1302-1331 |

All 5 NNs: every threshold rejected with "only N positive VALIDATION predictions (min=5)" — models cannot cross 0.5 threshold.

## 4.6 XGBoost Train-Val Gap (May 11, 2026)

| Metric | Train | Val | Gap |
|--------|-------|-----|-----|
| AUC | 0.8806 | 0.0000 | -0.8806 |
| Precision | 0.2134 | 0.2527 | +0.0393 |
| MaxPred | 0.9992 | 0.9979 | -0.0013 |
| MeanPred | 0.8179 | 0.3846 | -0.4333 |
| % Positive | 85.70% | 37.93% | -47.77% |
| TP | 45,811 | 0 | -45,811 |
| TN | 35,833 | 0 | -35,833 |

**Root Cause**: scale_pos_weight=500 + max_depth=7 + n_estimators=500 → extreme overfitting. Phase 4 threshold search produces TP=0 on validation due to optimal_threshold=2.0 producing zero TPs (all confusion matrix = 0). Train_P=0.2134 but Val_P=0.2527 only because Val predictions at threshold 0.5 yield 0 TP + 0 FP → precision undefined → zero_division=1.0 default, but confusion matrix shows all zeros. **Fix Applied**: Lower n_estimators [100-300], shallower depth [3-7], lower scale_pos_weight [200-500], higher regularization, add colsample_bytree, gamma. **Evidence**: pipeline_cpu.log lines 1140-1152.

## 4.7 Phase 5 Crash — df_with_all_cols AttributeError (May 11, 2026)

| Item | Value |
|------|-------|
| **Location** | chunk_19_phase_5_optimization.py line 272 |
| **Error** | `AttributeError: 'NoneType' object has no attribute 'columns'` |
| **Cause** | `df_with_all_cols = context.get('df_with_all_cols')` returned None; no guard around usage |
| **Fix Applied** | Added `if df_with_all_cols is not None` guard + `sorted_results = []` init before block (Fix A), added pruned_feature_indices lookup (Fix B), added SklearnModelWrapper in loader (Fix C), wrapped fraud output in None guard (Fix D) |
| **Evidence** | pipeline_cpu.log lines 1518-1527 |
| **Status** | Fixes A-D applied to chunk_19 and chunk_22; pipeline not yet re-run |

---

## Date: 2026-05-13

### GIS (Global Iteration Strategy) Hyperparameter Reconfiguration

**Root Cause**: All 6 NNs (CNN/LSTM/RNN/VAE/Transformer/Dense) produced MaxPred << 0.5 — search space too conservative. Gradient boosting trees stagnated due to small search spaces and missing key parameters.

#### All 9 Search Spaces Expanded (chunk_01_config.py lines 190-276)
| Architecture | Key Changes | Rationale |
|--------------|------------|-----------|
| CatBoost | iterations [300-500], depth [4-6] | Maximize phase — closer to 0.60 target |
| LightGBM | +colsample_bytree, +min_split_gain, num_leaves up to 127 | Wider regularization range |
| XGBoost | +colsample_bytree, +gamma, depth [3-7], lower spw | Combat severe train-val gap (AUC 0.88→0.00) |
| Dense | units [64-1024], layers [2-4], +batch_size, +activation | Deeper networks, larger capacity |
| CNN | filters [64-512], kernel 11, +pooling, +layers | Much larger capacity, more training |
| LSTM | lstm_units [32-256], +bidirectional, +layers | Major expansion from [8-32] |
| RNN | units [64-256], epochs [20-50], +layers | Larger RNNs |
| VAE | latent_dim [32-256], epochs, +encoder_layers, +decoder_layers | Architectural depth |
| Transformer | dim [64-256], heads [2-8], +ff_dim, +layers | Restore 128 dim with lower LR |

#### HPO Control Parameters Updated (chunk_01_config.py)
| Parameter | Was | Now |
|-----------|-----|-----|
| `HPO_STAGNATION_THRESHOLD` | 30 | **50** |
| `HYPERPARAM_OPTIMIZATION_TRIALS` | 30 | **60** |

#### max_trials Raised (chunk_21_hyperparam_optimizer.py)
| Parameter | Was | Now |
|-----------|-----|-----|
| Safety cap | 500 | **1000** |

#### Phase 5 Fixes A-D (chunk_19 + chunk_22)
- **Fix A**: initialized `sorted_results = []` before `if architecture_results:` block (chunk_19 line 328)
- **Fix B**: added pruned_feature_indices lookup for 24→19 pruning in Phase 5 (chunk_19 line 131)
- **Fix C**: patched model loader to detect sklearn archs and load via joblib + SklearnModelWrapper (chunk_22 lines 113-122)
- **Fix D**: wrapped fraud output section in `if df_with_all_cols is not None` guard (chunk_19 line 272)

### Files Modified
| File | Changes |
|------|---------|
| chunk_01_config.py | All 9 search spaces expanded, HPO thresholds raised |
| chunk_21_hyperparam_optimizer.py | max_trials 500→1000 |
| chunk_19_phase_5_optimization.py | Fixes A, B, D applied |
| chunk_22_model_loader.py | Fix C applied (SklearnModelWrapper) |
| SPEC.md | Sections 1.1, 2.8, 2.12, 3.1, 4, Implementation Summary updated |

### Validation
- All syntax verified with `python3 -m py_compile`
- Model loading verified: all 9 architectures load successfully
- sklearn models wrapped in SklearnModelWrapper with `_is_fitted = True`
- Phase 5 re-run pending — fixes not yet validated end-to-end

### GIS (Global Iteration Strategy) Execution Plan
| Iteration | Architectures | Phase | Target |
|-----------|--------------|-------|--------|
| 1 | CatBoost | Maximize | P ≥ 0.60 |
| 1 | LightGBM → XGBoost | Coarse/Refine | P ≥ 0.60 |
| 2 | Dense → CNN | Coarse/Refine | P ≥ 0.60 |
| 3 | RNN → LSTM | Coarse/Refine | P ≥ 0.60 |
| 4 | VAE → Transformer | Coarse/Refine | P ≥ 0.60 |
| 5 | Ensemble | Final | P ≥ 0.60 |

---

*Document generated: 2026-04-15*  
*Last updated: 2026-05-22*  
*Version: 3.6*

---

# IMPLEMENTATION SUMMARY - All Improvements Applied

## Date: 2026-05-04

### Step 1: Target Variable Transformation
- **LOG_TRANSFORM_TARGET**: True (Option C: sign * log1p(|y|))
- **Implementation**: chunk_05_data_manager.py - forward transform
- **Inverse transform**: chunk_04_utils_metrics.py (inverse_log_transform)
- **Usage**: chunk_18_phase_4_ensemble.py + chunk_19_phase_5_optimization.py

### Step 2: Diagnostics Enhancement
- Prediction percentiles (p1, p5, p10, p25, p50, p75, p90, p95, p99, max)
- Prediction histogram (20 bins)
- Loss distribution analysis
- Functions added to chunk_04_utils_metrics.py

### Step 3: Imbalance Handling
- **DYNAMIC_CLASS_WEIGHTS**: True
- **PREDICTION_THRESHOLD_SEARCH**: True (searches 0.1-0.5)
- **CALIBRATE_PREDICTIONS**: False (available for future use)
- **scale_pos_weight**: Wired for LightGBM, XGBoost, CatBoost

### Step 4: Feature Engineering
- **WINSORIZE_FEATURES**: True (1%/99% percentile clipping)
- **ADD_RATIO_FEATURES**: True (Price_to_52W_High, Volume_to_Avg_Volume, Price_to_52W_Low)
- **LOG_TRANSFORM_FEATURES**: True
- **HIGHLY_SKEWED_FEATURES**: [0, 1, 4, 5]

### New Diagnostics Items (Items 1-4)
- **FEATURE_STABILITY_ANALYSIS**: True (train/val correlation)
- **TRACK_INFERENCE_LATENCY**: True (ms per sample)
- **SLIDING_WINDOW_VALIDATION**: True (temporal validation)
- **PERMUTATION_IMPORTANCE**: True (all 9 models)
- **MIN_DATES_THRESHOLD**: 30 (raises ValueError if below)

### Advanced Diagnostics Added (Ready for Implementation)
- Prediction Entropy (model certainty)
- Logit Compression (max vs mean ratio)
- Kolmogorov-Smirnov Test (Power Features)
- Bhattacharyya Distance (class separation)
- Signal-to-Noise Ratio (temporal signal)
- Mutual Information Score (non-linear relationships)
- Population Stability Index (PSI) (distribution drift)

### Naming Standardization
- y_train_raw → y_train_continuous
- y_val_raw → y_val_continuous
- y_val_optimal → y_val_binarized
- Applied in chunk_18_phase_4_ensemble.py + chunk_19_phase_5_optimization.py

### TensorFlow Warning Suppression
- **TF_CPP_MIN_LOG_LEVEL**: '3' (suppress all CUDA/cuFFT/cuDNN/cuBLAS errors on CPU)
- **legacy files/run_pipeline_cpu.sh**: Wrapper script using system Python (TF 2.20.0, moved to legacy)

### Files Modified
| File | Changes |
|------|---------|
| chunk_01_config.py | 20+ config additions |
| chunk_04_utils_metrics.py | Diagnostic functions |
| chunk_05_data_manager.py | Forward transform + feature engineering |
| chunk_11_models_sklearn.py | scale_pos_weight + y_train parameter |
| chunk_14_models_trainer.py | Pass y_train to sklearn builders |
| chunk_18_phase_4_ensemble.py | All diagnostics + threshold search |
| chunk_19_phase_5_optimization.py | Inverse transform in evaluation |
| SPEC.md | Documentation updated |

### Validation Requirements
- **30 unique dates minimum** for diagnostics (Items 1, 2, 4)
- Raises ValueError if below threshold

### Model Support
- All 9 architectures get diagnostics: Dense, LightGBM, XGBoost, CatBoost, VAE, CNN, RNN, LSTM, Transformer

### Phase 1-3: Advanced Diagnostics (Implemented)
| Function | Purpose | Lines |
|----------|---------|-------|
| calculate_prediction_entropy | Model uncertainty (0=certain, higher=uncertain) | 317-335 |
| calculate_logit_compression | Confidence ratio (max/mean) | 338-351 |
| calculate_ks_test | Kolmogorov-Smirnov distribution test | 354-381 |
| calculate_bhattacharyya_distance | Class separation measure | 384-402 |
| calculate_snr | Signal-to-Noise ratio per time segment | 405-431 |
| calculate_mutual_information | Non-linear dependency (pred vs labels) | 434-447 |
| calculate_psi | Population Stability Index (drift detection) | 450-484 |

### Per-Architecture Advanced Diagnostics
- Applied to all 9 architectures in permutation importance loop (chunk_18:1176-1200)
- Output format: `[ARCH_NAME] [ADVANCED] Entropy=X.XXXX, LogitComp=X.XX, MI=X.XXXX`
- Output format: `[ARCH_NAME] [ADVANCED] KS-stat=X.XXXX (interpretation), Bhattacharyya=X.XXXX`

## Date: 2026-05-05

### Bug Fixes for Boosting Models
- Fixed `time` scoping error: Removed redundant import in chunk_18_phase_4_ensemble.py
- Fixed sklearn detection: Added sklearn detection in chunk_14_models_trainer.py
- Fixed predict(): Returns probabilities (not class labels), accepts **kwargs
- Fixed val_binary_preds scope: Calculate from val_pred

### LightGBM Hyperparameter Improvements
- scale_pos_weight: 259 → 400
- num_leaves: 31 → 63
- learning_rate: 0.05 → 0.1
- min_child_samples: 100 → 200
- Added: reg_alpha=0.1, reg_lambda=1.0, max_depth=5

### New Metrics Added
- calculate_specificity(), calculate_fpr(), calculate_f2_score() in chunk_12
- Extra metrics logging: Specificity, FPR, F2 Score, bucket distribution

### Additional Files Modified
| File | Changes |
|------|---------|
| chunk_11_models_sklearn.py | predict() returns probabilities + **kwargs + improved defaults |
| chunk_12_evaluation_evaluator.py | Added specificity, FPR, F2 methods + removed max_pos_ratio |
| chunk_14_models_trainer.py | Added sklearn detection |
| chunk_18_phase_4_ensemble.py | Removed redundant import + fixed metrics logging + minlength=11 + patience=5 |

### Architecture-Specific HPO Improvements (May 5, 2026)
- HPO Trials: 20 → 30 for all architectures
- LightGBM: scale_pos_weight [400,500], min_child [100,200], added reg_alpha/reg_lambda
- XGBoost: scale_pos_weight [400,500], min_child [50,100,200], added reg_alpha/reg_lambda
- CatBoost: iterations [100,200] (reduced), l2_leaf_reg [3,5,10] (extended)

---

## All Implementation Items Complete
