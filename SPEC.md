# Software Specification Requirements (SSR) - Stock Analysis Ensemble

**Version**: 3.17  
**Date**: 2026-06-01  
**Status**: Living Document - Update After Each Run  

---

# QUICK START GUIDE

## What Is This Document?
SPEC.md is a living document for the Stock Analysis Ensemble Pipeline. It records outputs (Section 1), documents the code (Section 2), maintains history (Section 3), and tracks permanent failures (Section 4).

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
[ARCH_NAME] [BASELINE] BEFORE_THRESHOLD_OPTIMIZATION:
[ARCH_NAME] [BASELINE] Predictions: mean=0.0023, std=0.0156, min=0.0001, max=0.4523
[ARCH_NAME] [BASELINE] % positive predictions (Prediction_Threshold=0.5): 0.12%
```

### ACTUAL RESULTS - Run #2026-05-19

| Architecture | mean | std | min | max | % Positive | Notes |
|--------------|-----|-----|-----|-----|------------|-------|

| Architecture | Optimal Threshold | VALIDATION_Precision | VALIDATION_Recall | VALIDATION_AUC | VALIDATION_TP | VALIDATION_FP | Notes |
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

| Architecture | Pre-HYPERPARAMETER_OPTIMIZATION Threshold | Pre-HYPERPARAMETER_OPTIMIZATION PRECISION | Post-HYPERPARAMETER_OPTIMIZATION Threshold | Post-HYPERPARAMETER_OPTIMIZATION PRECISION | Improved? |
|--------------|-------------------|-----------|-------------------|-----------|-----------|
| CatBoost | (baseline) | 0.5370 | 0.0 | **0.5381** | Yes |
| LightGBM | (baseline) | 0.1802 | 4.0 | **0.1753** | No |
| XGBoost | (baseline) | 0.2527 | 2.0 | **0.2484** | No |

### ACTUAL RESULTS - Run #2026-06-01 (Iter 2 — Tier 2 active)

**S5 Final Validation Precision:**

| Architecture | VALIDATION_PRECISION | Source | Notes |
|--------------|---------------|--------|-------|
| CNN | **0.6537** | section4 | Dramatic improvement; learned meaningful predictions |
| LightGBM | 0.5308 | section4 | HPO improved 0.1367→0.5306, post-HPO tuned t=0.0 |
| Transformer | 0.5299 | section4 | hpo did NOT improve (0.0360 ≤ 0.0417) but post-HPO threshold search found t=0.0 |
| Dense | 0.5285 | section4 | HPO improved 0.1571→0.3158, post-HPO tuned t=0.0 |
| CatBoost | 0.5283 | section3 | hpo improved 0.5280→0.5306 but post-HPO did not improve further |
| XGBoost | **0.4899** | section4* | **BUG**: S4→S5 model carry-forward. hpo NOT improved (0.0670 ≤ 0.0797) → threshold_opt_model used. Post-HPO found P=0.5629 for HPO model at t=0.0 but model is wrong. Post-fix estimate: 0.5629. |
| LSTM | 0.4853 | section4 | hpo improved 0.0321→0.0536. Lower than S4 (0.5357) due to PREDICTION_THRESHOLD=0.55 vs S4's 0.5. Correct model used (hpo_best_model). |
| VAE | 0.4838 | section4 | hpo improved 0.0000→0.0212. Lower than S4 (0.5270) due to PREDICTION_THRESHOLD=0.55 vs S4's 0.5. Correct model used (hpo_best_model). |
| RNN | 0.0000 | section3 | Collapsed; all thresholds rejected |

**Decision Gate**: Top 1 P = 0.6537 ≥ 0.56 → **ENTER OPTIMIZE PHASE** ✅

**S4→S5 Bug**: Only XGBoost is affected. Section 4 finds HPO model achieves P=0.5629 at t=0.0, but Section 5 selects `threshold_opt_model` because `hpo_improved=False`. The pre-HPO model (trained at t=20.0) cannot generalize to t=0.0 labels. Applies to architectures where `hpo_improved=False` AND `optimal_threshold ≠ post_hpo_thresh`. Post-fix: 6/9 archs would pass 0.52 filter.

### ACTUAL RESULTS - Run #2026-05-11

| Architecture | mean | std | min | max | % Positive | Notes |
|--------------|-----|-----|-----|-----|------------|-------|

### Section 2: Threshold Optimization Results
Logged for each of 3 thresholds (20.0, 10.0, 0.0):
- Train metrics: 24 metrics (PRECISION, TRUE_POSITIVES, TRUE_NEGATIVES, FALSE_POSITIVES, FALSE_NEGATIVES, MAX_PREDICTION, MEAN_PREDICTION, RECALL, F1_SCORE, AUC, SPECIFICITY, FALSE_POSITIVE_RATE, F2_SCORE, MCC, PRAUC, BALANCED_ACCURACY, Brier, Kappa, Informedness, Markedness, Gini, OPTIMAL_THRESHOLD, STD_PREDICTION, PCT_ABOVE_THRESHOLD)
- Val metrics: Same 24 metrics
- Optimal threshold selection

**Source File**: chunk_12_evaluation_evaluator.py, chunk_18_phase_4_ensemble.py

Example:
```
LightGBM t=20.0 | TRAIN: TRAIN_PRECISION=0.8500 TRAIN_TRUE_POSITIVES=5000 TRAIN_TRUE_NEGATIVES=120000 TRAIN_FALSE_POSITIVES=880 TRAIN_FALSE_NEGATIVES=45000 TRAIN_MAX_PREDICTION=0.9500 TRAIN_MEAN_PREDICTION=0.0045 TRAIN_RECALL=0.1200 TRAIN_F1_SCORE=0.2100 TRAIN_AUC=0.9500 TRAIN_SPECIFICITY=0.9920 TRAIN_FALSE_POSITIVE_RATE=0.0070 TRAIN_F2_SCORE=0.1800 TRAIN_MCC=0.2100 TRAIN_PRAUC=0.8500 TRAIN_BALANCED_ACCURACY=0.5600 TRAIN_Brier=0.0040 TRAIN_Kappa=0.1800 TRAIN_Informedness=0.1200 TRAIN_Markedness=0.8500 TRAIN_Gini=0.9000 TRAIN_OPTIMAL_THRESHOLD=0.5500 TRAIN_STD_PREDICTION=0.0300 TRAIN_PCT_ABOVE_THRESHOLD=0.50
LightGBM t=20.0 | VALIDATION:   VALIDATION_PRECISION=0.8200 VALIDATION_TRUE_POSITIVES=1800 VALIDATION_TRUE_NEGATIVES=40000 VALIDATION_FALSE_POSITIVES=395 VALIDATION_FALSE_NEGATIVES=15000 VALIDATION_MAX_PREDICTION=0.9200 VALIDATION_MEAN_PREDICTION=0.0055 VALIDATION_RECALL=0.1100 VALIDATION_F1_SCORE=0.1900 VALIDATION_AUC=0.9400 VALIDATION_SPECIFICITY=0.9900 VALIDATION_FALSE_POSITIVE_RATE=0.0100 VALIDATION_F2_SCORE=0.1500 VALIDATION_MCC=0.1900 VALIDATION_PRAUC=0.8200 VALIDATION_BALANCED_ACCURACY=0.5500 VALIDATION_Brier=0.0050 VALIDATION_Kappa=0.1600 VALIDATION_Informedness=0.1100 VALIDATION_Markedness=0.8200 VALIDATION_Gini=0.8800 VALIDATION_OPTIMAL_THRESHOLD=0.5200 VALIDATION_STD_PREDICTION=0.0350 VALIDATION_PCT_ABOVE_THRESHOLD=0.55
LightGBM - OPTIMAL: label_threshold=12.0, VALIDATION_PRECISION=0.7800
```

### ACTUAL RESULTS - Run #

| Architecture | Optimal Threshold | VALIDATION_Precision | VALIDATION_Recall | VALIDATION_AUC | VALIDATION_F1 | Pass? |
|--------------|-------------------|---------------|-----------|---------|--------|-------|

### Section 3: Hyperparameter Optimization (HPO)
Logged for each of 20 Optuna trials:
- Trial parameters tested
- Validation metrics: PRECISION, RECALL, AUC, F1_SCORE, TRUE_POSITIVES, FALSE_POSITIVES, TRUE_NEGATIVES, FALSE_NEGATIVES
- MaxPred, MeanPred values
- Rejection messages if MaxPred < 0.5

**Source File**: chunk_21_hyperparam_optimizer.py

Example:
```
Trial 1/30: n_estimators=500, num_leaves=31, learning_rate=0.05 → VALIDATION_PRECISION=0.4500 VALIDATION_TRUE_POSITIVES=180 VALIDATION_TRUE_NEGATIVES=39800 VALIDATION_FALSE_POSITIVES=220 VALIDATION_FALSE_NEGATIVES=8920 VALIDATION_MAX_PREDICTION=0.8500 VALIDATION_MEAN_PREDICTION=0.0045 VALIDATION_RECALL=0.0200 VALIDATION_F1_SCORE=0.0380 VALIDATION_AUC=0.7200 VALIDATION_SPECIFICITY=0.9945 VALIDATION_FALSE_POSITIVE_RATE=0.0055 VALIDATION_F2_SCORE=0.0280 VALIDATION_MCC=0.0250 VALIDATION_PRAUC=0.4500 VALIDATION_BALANCED_ACCURACY=0.5100 VALIDATION_Brier=0.0045 VALIDATION_Kappa=0.0200 VALIDATION_Informedness=0.0200 VALIDATION_Markedness=0.4400 VALIDATION_Gini=0.4400 VALIDATION_OPTIMAL_THRESHOLD=0.5200
```

### ACTUAL RESULTS - Run #2026-05-11

| Architecture | Best Trial# | Best Params | VALIDATION_Precision | VALIDATION_Recall | VALIDATION_AUC | Notes |
|--------------|-------------|-------------|---------------|-----------|---------|-------|
| CatBoost | 59+ | iterations=200, depth=6, lr=0.1, auto_class_weights=SqrtBalanced, l2_leaf_reg=10 | 0.5381 | — | — | Stagnant; Maximize phase needed |
| LightGBM | 110+ | n_estimators=500, num_leaves=63, lr=0.1, class_weight=balanced, min_child_samples=100, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8 | 0.1753 | — | — | Stagnant; wider search needed |
| XGBoost | — | n_estimators=500, max_depth=7, lr=0.1, scale_pos_weight=500, min_child_weight=50, reg_alpha=0.5, reg_lambda=5.0, subsample=0.7 | 0.2527 | — | 0.0000 | Severely overfit; Train AUC=0.8806 → Val AUC=0.0000 |
| Dense | — | units=32, layers=2, dropout=0.3, lr=0.0005, epochs=15, alpha=1.0, gamma=3.0 | 0.3689 | — | — | HPO val P (from metadata); Phase 4 training failed |
| CNN | — | filters=32, kernel_size=7, dropout=0.1, lr=0.001, epochs=20, alpha=0.75, gamma=3.0 | 1.0000 | — | — | ARTIFACT — all TP=0, MaxPred=0.0042 |
| LSTM | — | lstm_units=32, dropout=0.05, lr=0.0005, epochs=15, alpha=0.75, gamma=2.0 | 1.0000 | — | — | ARTIFACT — all TP=0, MaxPred=0.0316 |
| RNN | — | units=16, dropout=0.05, lr=0.001, epochs=10, alpha=1.0, gamma=3.0 | 0.4970 | — | — | From metadata; Phase 4 training failed |
| VAE | — | latent_dim=64, lr=0.001, dropout=0.05, alpha=0.75, gamma=2.0 | 0.0000 | — | — | MaxPred<0.5, all trials rejected |
| Transformer | — | dim=32, heads=2, dropout=0.02, lr=0.0005, alpha=1.0, gamma=3.0 | 0.3787 | — | — | From metadata; Phase 4 training failed |

### Objective / Loss Functions Per Active Architecture

| Architecture | Loss Function | Imbalance Handling | Source |
|---|---|---|---|
| CatBoost | Logloss | `auto_class_weights='SqrtBalanced'` | chunk_11 |
| LightGBM | binary (log loss) | `class_weight='balanced'` | chunk_11 |
| XGBoost | `binary:logistic` | `scale_pos_weight` (dynamic neg/pos) | chunk_11 |
| DENSE | BCE / FocalLoss (via FOCAL_LOSS_CONFIG) | `class_weight` from trainer + optional FocalLoss | chunk_08 |
| CNN | FocalLoss(α=0.75, γ=1.5) | Built-in FocalLoss + `class_weight` from trainer | chunk_08 |
| RNN | BCE / FocalLoss (via FOCAL_LOSS_CONFIG) | `class_weight` from trainer + optional FocalLoss | chunk_08 |
| LSTM | BCE / FocalLoss (via FOCAL_LOSS_CONFIG) | `class_weight` from trainer + optional FocalLoss | chunk_08 |
| VAE | FocalLoss(α=0.75, γ=1.5) | Built-in FocalLoss + `class_weight` from trainer | chunk_08 |
| Transformer | FocalLoss(α=0.75, γ=1.5) | Built-in FocalLoss + `class_weight` from trainer | chunk_09 |

Notes:
- All neural architectures receive `class_weight={0: weight, 1: weight}` at `.fit()` time via `chunk_14` trainer (`compute_class_weight('balanced')`)
- FOCAL_LOSS_CONFIG enables per-architecture FocalLoss override at compile time (config key in `chunk_01_config.py:161-168`)
- Architectures with FocalLoss as default: CNN, VAE, Transformer (α=0.75, γ=1.5), Dense (α=0.5, γ=1.0), RNN (α=0.5, γ=1.0), LSTM (α=0.75, γ=1.5)
- Architectures with `loss_function` in HPO space: CNN, Dense, RNN, LSTM, VAE (Transformer has FocalLoss default but HPO only offers BCE)

### Phase 5 Results - Run #2026-05-11 (CRASHED at chunk_19 line 272)

| Architecture | INFERENCE_PRECISION | INFERENCE_RECALL | INFERENCE_AUC | INFERENCE_TRUE_POSITIVES | INFERENCE_FALSE_POSITIVES | INFERENCE_TRUE_NEGATIVES | INFERENCE_FALSE_NEGATIVES | Notes |
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

### Section 4: Post-HYPERPARAMETER_OPTIMIZATION Threshold Search
Logged after HPO completes:
- Pre-HYPERPARAMETER_OPTIMIZATION vs Post-HYPERPARAMETER_OPTIMIZATION comparison
- Final threshold selection

**Source File**: chunk_12_evaluation_evaluator.py

Example:
```
PRE-HYPERPARAMETER_OPTIMIZATION threshold: t=12.0, VALIDATION_PRECISION=0.7800
POST-HYPERPARAMETER_OPTIMIZATION threshold: t=10.0, VALIDATION_PRECISION=0.8200
POST-HYPERPARAMETER_OPTIMIZATION improved: using t=10.0
```

### ACTUAL RESULTS - Run #

| Architecture | Pre-HYPERPARAMETER_OPTIMIZATION Threshold | Pre-HYPERPARAMETER_OPTIMIZATION PRECISION | Post-HYPERPARAMETER_OPTIMIZATION Threshold | Post-HYPERPARAMETER_OPTIMIZATION PRECISION | Improved? |
|--------------|-------------------|-------------------|-------------------|-------------------|-----------|

### Section 5: Final Model Summary
Logged at completion:
- All 17 metrics
- Source (section3 or section4)
- Training epochs

**Source File**: chunk_18_phase_4_ensemble.py

### ACTUAL RESULTS - Run #

| Architecture | VALIDATION_PRECISION | VALIDATION_RECALL | VALIDATION_AUC | VALIDATION_F1_SCORE | VALIDATION_TRUE_POSITIVES | VALIDATION_FALSE_POSITIVES | VALIDATION_TRUE_NEGATIVES | VALIDATION_FALSE_NEGATIVES | VALIDATION_MAX_PREDICTION | VALIDATION_MEAN_PREDICTION | Source |
|--------------|---------------|-----------|---------|--------|--------|--------|----|----|----------|-----------|--------|

## 1.2 Post-Execution Reports

### Metrics Summary
Generated: `metrics_summary.csv`

**Source File**: chunk_20_pipeline_main.py

| Column | Description |
|--------|-------------|
| Architecture | Model name |
| VALIDATION_PRECISION | Validation precision |
| VALIDATION_RECALL | Validation recall |
| VALIDATION_F1_SCORE | Validation F1 score |
| VALIDATION_AUC | Validation AUC |
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
| TP | True Positives - Correctly predicted signal | chunk_04_utils_metrics.py |
| FP | False Positives - Non-signal predicted as signal | chunk_04_utils_metrics.py |
| TN | True Negatives - Correctly predicted normal | chunk_04_utils_metrics.py |
| FN | False Negatives - Signal predicted as non-signal | chunk_04_utils_metrics.py |

### Prediction Statistics

| Statistic | Description | Actual Value | Run Date | Source File |
|-----------|-------------|--------------|----------|------------|
| PREDICTION_MEAN | Mean prediction value | chunk_14_models_trainer.py |
| PREDICTION_STANDARD_DEVIATION | Standard deviation of predictions | chunk_14_models_trainer.py |
| PREDICTION_MAX | Maximum prediction value | chunk_14_models_trainer.py |
| PREDICTION_MIN | Minimum prediction value | chunk_14_models_trainer.py |

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
| 1.1.4 Post-HYPERPARAMETER_OPTIMIZATION Threshold | Pre vs Post comparison | chunk_12_evaluation_evaluator.py | → Section 2.5, FR-06 |
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
| VALIDATION_PRECISION | calculate_precision() | SECTION 1-5, HPO Trials | Precision (0.0-1.0) |
| VALIDATION_RECALL | calculate_recall() | SECTION 1-5, HPO Trials | Recall (0.0-1.0) |
| VALIDATION_AUC | calculate_auc() | SECTION 1-5, HPO Trials | AUC (0.0-1.0) |
| VALIDATION_F1_SCORE | calculate_f1() | SECTION 1-5, HPO Trials | F1 Score (0.0-1.0) |
| VALIDATION_TRUE_POSITIVES | Extracted from CM | SECTION 1-5, HPO Trials | True Positives (count) |
| VALIDATION_FALSE_POSITIVES | Extracted from CM | SECTION 1-5, HPO Trials | False Positives (count) |
| VALIDATION_TRUE_NEGATIVES | Extracted from CM | SECTION 1-5, HPO Trials | True Negatives (count) |
| VALIDATION_FALSE_NEGATIVES | Extracted from CM | SECTION 1-5, HPO Trials | False Negatives (count) |
| VALIDATION_MAX_PREDICTION | pred.max() | SECTION 1-5 | Max prediction |
| VALIDATION_MEAN_PREDICTION | pred.mean() | SECTION 1-5 | Mean prediction |
| VALIDATION_STD_PREDICTION | pred.std() | SECTION 1-5 | Std prediction |
| VALIDATION_PCT_ABOVE_THRESHOLD | (pred>=0.5).mean() | SECTION 1-5 | % above threshold |
| VALIDATION_MCC | calculate_mcc() | SECTION 1-5, HPO Trials | MCC (-1 to 1) |
| VALIDATION_PRAUC | calculate_average_precision() | SECTION 1-5, HPO Trials | PR-AUC |
| VALIDATION_SPECIFICITY | calculate_specificity() | SECTION 1-5, HPO Trials | Specificity |
| VALIDATION_BALANCED_ACCURACY | calculate_balanced_accuracy() | SECTION 1-5, HPO Trials | Balanced Accuracy |
| VALIDATION_FALSE_POSITIVE_RATE | calculate_fpr() | SECTION 1-5, HPO Trials | False Positive Rate |
| VALIDATION_F2_SCORE | calculate_f2_score() | SECTION 1-5, HPO Trials | F2 Score |
| VALIDATION_Brier | calculate_brier_score() | SECTION 1-5, HPO Trials | Brier Score (lower=better) |
| VALIDATION_Kappa | calculate_kappa() | SECTION 1-5, HPO Trials | Cohen's Kappa |
| VALIDATION_Informedness | calculate_informedness() | SECTION 1-5, HPO Trials | Informedness |
| VALIDATION_Markedness | calculate_markedness() | SECTION 1-5, HPO Trials | Markedness |
| VALIDATION_Gini | calculate_gini() | SECTION 1-5, HPO Trials | Gini Coefficient |
| VALIDATION_OPTIMAL_THRESHOLD | calculate_optimal_threshold() | SECTION 1-5, HPO Trials | Optimal Threshold |

### Section Mapping

| Section | Description | Metrics Logged |
|---------|-------------|---------------|
| SECTION_1_BASELINE | Baseline | All 24 standard VALIDATION_ metrics |
| SECTION_2_HYPERPARAMETER_OPTIMIZATION_SEARCH | Pre-HYPERPARAMETER_OPTIMIZATION Threshold | All 24 standard VALIDATION_ metrics |
| HPO Trials | Hyperparameter Opt | All 24 standard VALIDATION_ metrics + hyperparams |
| SECTION_3_PRE_HYPERPARAMETER_OPTIMIZATION | HPO Results | All 24 standard VALIDATION_ metrics + Best hyperparams |
| SECTION_4_POST_HYPERPARAMETER_OPTIMIZATION | Post-HYPERPARAMETER_OPTIMIZATION Threshold | All 24 standard VALIDATION_ metrics + Source tag |
| SECTION_5_FINAL | Final Model | All 24 standard VALIDATION_ metrics + 18 TRAIN_ metrics |
| Threshold Search (Train) | Per-threshold train | 8 TRAIN_ metrics |
| Threshold Search (Val) | Per-threshold val | 8 VALIDATION_ metrics |

### Training Metrics

All 18 training metrics are reported in SECTION 5 (Final) with TRAIN_ prefix:

| Metric | Prefix | Description | Formula |
|--------|--------|-------------|---------|
| TRAIN_PRECISION | TRAIN_ | Precision | TP / (TP + FP) |
| TRAIN_TRUE_POSITIVES | TRAIN_ | True Positives | count(y=1 & pred=1) |
| TRAIN_TRUE_NEGATIVES | TRAIN_ | True Negatives | count(y=0 & pred=0) |
| TRAIN_FALSE_POSITIVES | TRAIN_ | False Positives | count(y=0 & pred=1) |
| TRAIN_FALSE_NEGATIVES | TRAIN_ | False Negatives | count(y=1 & pred=0) |
| TRAIN_MAX_PREDICTION | TRAIN_ | Max Probability | predictions.max() |
| TRAIN_MEAN_PREDICTION | TRAIN_ | Mean Probability | predictions.mean() |
| TRAIN_RECALL | TRAIN_ | Recall (Sensitivity) | TP / (TP + FN) |
| TRAIN_F1_SCORE | TRAIN_ | F1 Score | 2*P*R / (P+R) |
| TRAIN_AUC | TRAIN_ | ROC-AUC | sklearn roc_auc_score |
| TRAIN_SPECIFICITY | TRAIN_ | Specificity | TN / (TN + FP) |
| TRAIN_FALSE_POSITIVE_RATE | TRAIN_ | False Positive Rate | FP / (FP + TN) |
| TRAIN_F2_SCORE | TRAIN_ | F2 Score | 5*P*R / (4*P+R) |
| TRAIN_MCC | TRAIN_ | Matthews Corr Coef | sklearn matthews_corrcoef |
| TRAIN_PRAUC | TRAIN_ | Precision-Recall AUC | sklearn avg_precision_score |
| TRAIN_BALANCED_ACCURACY | TRAIN_ | Balanced Accuracy | (Sens + Spec) / 2 |
| TRAIN_STD_PREDICTION | TRAIN_ | Std Probability | predictions.std() |
| TRAIN_PCT_ABOVE_THRESHOLD | TRAIN_ | % Above Threshold 0.5 | (pred >= 0.5).mean() * 100 |

### Validation Metrics

All 24 validation metrics are reported in SECTIONS 1-5 with VALIDATION_ prefix:

| Metric | Prefix | Description | Formula |
|--------|--------|-------------|---------|
| VALIDATION_PRECISION | VALIDATION_ | Precision | TP / (TP + FP) |
| VALIDATION_TRUE_POSITIVES | VALIDATION_ | True Positives | count(y=1 & pred=1) |
| VALIDATION_TRUE_NEGATIVES | VALIDATION_ | True Negatives | count(y=0 & pred=0) |
| VALIDATION_FALSE_POSITIVES | VALIDATION_ | False Positives | count(y=0 & pred=1) |
| VALIDATION_FALSE_NEGATIVES | VALIDATION_ | False Negatives | count(y=1 & pred=0) |
| VALIDATION_MAX_PREDICTION | VALIDATION_ | Max Probability | predictions.max() |
| VALIDATION_MEAN_PREDICTION | VALIDATION_ | Mean Probability | predictions.mean() |
| VALIDATION_RECALL | VALIDATION_ | Recall (Sensitivity) | TP / (TP + FN) |
| VALIDATION_F1_SCORE | VALIDATION_ | F1 Score | 2*P*R / (P+R) |
| VALIDATION_AUC | VALIDATION_ | ROC-AUC | sklearn roc_auc_score |
| VALIDATION_SPECIFICITY | VALIDATION_ | Specificity | TN / (TN + FP) |
| VALIDATION_FALSE_POSITIVE_RATE | VALIDATION_ | False Positive Rate | FP / (FP + TN) |
| VALIDATION_F2_SCORE | VALIDATION_ | F2 Score | 5*P*R / (4*P+R) |
| VALIDATION_MCC | VALIDATION_ | Matthews Corr Coef | sklearn matthews_corrcoef |
| VALIDATION_PRAUC | VALIDATION_ | Precision-Recall AUC | sklearn avg_precision_score |
| VALIDATION_BALANCED_ACCURACY | VALIDATION_ | Balanced Accuracy | (Sens + Spec) / 2 |
| VALIDATION_STD_PREDICTION | VALIDATION_ | Std Probability | predictions.std() |
| VALIDATION_PCT_ABOVE_THRESHOLD | VALIDATION_ | % Above Threshold 0.5 | (pred >= 0.5).mean() * 100 |
| VALIDATION_Brier | VALIDATION_ | Brier Score | mean((pred - y)^2) |
| VALIDATION_Kappa | VALIDATION_ | Cohen's Kappa | sklearn cohen_kappa_score |
| VALIDATION_Informedness | VALIDATION_ | Informedness | Sensitivity + Specificity - 1 |
| VALIDATION_Markedness | VALIDATION_ | Markedness | Precision + NPV - 1 |
| VALIDATION_Gini | VALIDATION_ | Gini Coefficient | 2 * AUC - 1 |
| VALIDATION_OPTIMAL_THRESHOLD | VALIDATION_ | Optimal Threshold | Youden's J (max TPR - FPR) |

### Inference Metrics

All 24 inference metrics are reported in Phase 5 with INFERENCE_ prefix:

| Metric | Prefix | Description | Formula |
|--------|--------|-------------|---------|
| INFERENCE_PRECISION | INFERENCE_ | Precision | TP / (TP + FP) |
| INFERENCE_TRUE_POSITIVES | INFERENCE_ | True Positives | count(y=1 & pred=1) |
| INFERENCE_TRUE_NEGATIVES | INFERENCE_ | True Negatives | count(y=0 & pred=0) |
| INFERENCE_FALSE_POSITIVES | INFERENCE_ | False Positives | count(y=0 & pred=1) |
| INFERENCE_FALSE_NEGATIVES | INFERENCE_ | False Negatives | count(y=1 & pred=0) |
| INFERENCE_MAX_PREDICTION | INFERENCE_ | Max Probability | predictions.max() |
| INFERENCE_MEAN_PREDICTION | INFERENCE_ | Mean Probability | predictions.mean() |
| INFERENCE_STD_PREDICTION | INFERENCE_ | Std Probability | predictions.std() |
| INFERENCE_PCT_ABOVE_THRESHOLD | INFERENCE_ | % Above Threshold 0.5 | (pred >= 0.5).mean() * 100 |
| INFERENCE_RECALL | INFERENCE_ | Recall (Sensitivity) | TP / (TP + FN) |
| INFERENCE_F1_SCORE | INFERENCE_ | F1 Score | 2*P*R / (P+R) |
| INFERENCE_AUC | INFERENCE_ | ROC-AUC | sklearn roc_auc_score |
| INFERENCE_SPECIFICITY | INFERENCE_ | Specificity | TN / (TN + FP) |
| INFERENCE_FALSE_POSITIVE_RATE | INFERENCE_ | False Positive Rate | FP / (FP + TN) |
| INFERENCE_F2_SCORE | INFERENCE_ | F2 Score | 5*P*R / (4*P+R) |
| INFERENCE_MCC | INFERENCE_ | Matthews Corr Coef | sklearn matthews_corrcoef |
| INFERENCE_PRAUC | INFERENCE_ | Precision-Recall AUC | sklearn avg_precision_score |
| INFERENCE_BALANCED_ACCURACY | INFERENCE_ | Balanced Accuracy | (Sens + Spec) / 2 |
| INFERENCE_Brier | INFERENCE_ | Brier Score | mean((pred - y)^2) |
| INFERENCE_Kappa | INFERENCE_ | Cohen's Kappa | sklearn cohen_kappa_score |
| INFERENCE_Informedness | INFERENCE_ | Informedness | Sensitivity + Specificity - 1 |
| INFERENCE_Markedness | INFERENCE_ | Markedness | Precision + NPV - 1 |
| INFERENCE_Gini | INFERENCE_ | Gini Coefficient | 2 * AUC - 1 |
| INFERENCE_OPTIMAL_THRESHOLD | INFERENCE_ | Optimal Threshold | Youden's J (max TPR - FPR) |

---

## 1.6 Run Comparison

Use this table to track changes between runs:

| Metric | Previous Run | Current Run | Change | Trend |
|--------|------------|-------------|--------|-------|

# SECTION 2: Functionality

## 2.1 Project Overview

### Problem Statement

Stock markets generate massive datasets spanning multiple regimes — bull/bear cycles, high/low volatility periods, sector rotations — each exhibiting different dynamics. Traditional single-model approaches miss regime-specific signals. This project addresses the challenge of identifying stock strength signals in highly imbalanced datasets (259:1 ratio) using ensemble machine learning.

### Project Details

| Aspect | Specification |
|--------|---------------|
| **Project Name** | Stock Analysis Ensemble Pipeline |
| **Project Type** | Machine Learning Pipeline |
| **Core Functionality** | Identify stock strength signals using ensemble of neural networks and gradient boosting models |
| **Target Domain** | Financial markets / Stock analysis |
| **Language** | Python 3.12 |

---

## 2.2 System Context

| Component | Description |
|-----------|-------------|
| **Input** | CSV file with financial features + date + target |
| **Processing** | Data loading → Temporal weighting → Model training → Ensemble → Prediction |
| **Output** | Strength signal predictions (binary + probability), model files, evaluation metrics |
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
| **Target Column** | Binary strength signal indicator (0/1) or continuous change value |
| **Dataset Size** | Approximately 6.7 million records |
| **Features** | 16-21 features (after pruning) |
| **Date Range** | 2022-03-01 to 2025-10-23 |
| **Class Imbalance** | 259:1 ratio (0.4% signal) |

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
| Phase Xa | Feature importance (3 thresholds, per-threshold pruning) | threshold_kept_indices dict, all 24 features kept in context['X'] | → Feature analysis |
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
| FR-02 | Handle extreme class imbalance (259:1) using class weights | Required | chunk_11_models_sklearn.py | → See Section 1.3 |
| FR-03 | Implement train/validation split (70%/30% temporal) | Required | chunk_16_phase_1_setup.py | → See Section 1.1.1 |
| FR-04 | Apply temporal weighting based on date | Required | chunk_07_data_temporal.py | → See Section 1.1.1 |
| FR-05 | Train multiple architectures: LightGBM, XGBoost, CatBoost, VAE, Dense, CNN, RNN, LSTM, Transformer | Required | chunk_08-11_models_*.py, chunk_18_phase_4_ensemble.py | → See Section 1.1.5 |
| FR-06 | Perform threshold optimization (3 thresholds: 20.0, 10.0, 0.0) | Required | chunk_12_evaluation_evaluator.py | → See Section 1.1.2, Section 1.3 |
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
| THRESHOLD_STEP | -10.0 | Label threshold increment (reduced to 3 steps: 20→10→0) |
| PREDICTION_THRESHOLD | 0.55 | Binary classification threshold (raised from 0.5 for GIS Tier 1) |

### Ensemble Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ENSEMBLE_MIN_PRECISION | 0.53 | Minimum precision for ensemble (raised from 0.52 for GIS Tier 3 — tighter ensemble filter) |
| ENSEMBLE_WEIGHTING | uniform | Weighting method (changed from precision_weighted to uniform for GIS Tier 1 — prevents CatBoost dominance) |
| ENSEMBLE_VOTE_THRESHOLD | 0.67 | Models must agree threshold (raised from 0.5 for GIS Tier 1 — tighter consensus) |
| FALLBACK_ARCHITECTURE | VAE | Highest val precision fallback (changed from RNN for GIS Tier 1) |

### HPO Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ENABLE_HYPERPARAM_OPTIMIZATION | True | Enable HPO |
| HYPERPARAM_OPTIMIZATION_EPOCHS | 5 | Epochs per HPO trial |
| HPO_TRIALS | 20 | Number of Optuna trials |
| ENABLE_POST_HPO_THRESHOLD_SEARCH | True | Run threshold search after HPO |

### Safe Guard Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| WINSORIZE_PERCENTILE_LOW | 2 | Lower percentile for winsorization (raised from 1 for GIS Tier 2) |
| WINSORIZE_PERCENTILE_HIGH | 95 | Upper percentile for winsorization (lowered from 98 for GIS Tier 3 — tighter outlier removal) |
| MIN_POSITIVE_PERCENTAGE | 0.01 | 1% of samples must be positive (raised from 0.5% for GIS Tier 2) |
| MIN_POSITIVE_ABSOLUTE | 100 | Absolute floor for positive predictions (raised from 50 for GIS Tier 2) |
| MIN_PRECISION_OVER_BASELINE | 0.02 | Precision must beat baseline by 2% (raised from 1% for GIS Tier 2) |
| MIN_POS_PRED_RATIO | 0.001 | Min 0.1% of predictions must be positive (raised from 0.01% for GIS Tier 2) |
| MAX_POS_PRED_RATIO | 0.60 | Max 60% of predictions can be positive (lowered from 70% for GIS Tier 2) |

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

### LightGBM (Iteration 1) — `scale_pos_weight` removed in 3.8 (uses `class_weight='balanced'`)
```python
{
    'n_estimators': [300, 500, 800],    # was [200, 500]
    'num_leaves': [31, 63, 127],        # was [31, 63]
    'learning_rate': [0.03, 0.05, 0.08],  # added lower LR
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
| `HYPERPARAM_OPTIMIZATION_TRIALS` | 30 | **5** | Reduced from 30 to 5 for faster pipeline runs |
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
| Extreme class imbalance | Model bias | class_weight, auto_class_weights, focal loss |
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
| 3.4 | 2026-05-18 | GIS (Global Iteration Strategy) SUCCESS — CatBoost achieved 0.7204 inference precision (>0.60 target), Phase 5 fixes (KeyError: 'precision' → 'Inf_P' [now INFERENCE_PRECISION per Category A rename], shape mismatch handling, df_filtered→n_inference, inference_date→dates_inference[0]), sample size reduced 368816→184408 | CatBoost: 0.7204 INFERENCE_PRECISION; LightGBM: 0.2722; XGBoost: 0.2751; 5 NNs skipped (shape mismatch); sample reduced for faster runs |
| 3.5 | 2026-05-19 | Logging standardization (tag reorder, terminology), per-threshold feature pruning architecture, feature importance logging overhaul, HPO logging improvements (best trial tracking, [BEST TRIAL] format, [OPTIMAL] expanded), Section 2 redundant logs removed with stale y_val_binarized fix | Tag reorder: [BASELINE] {arch_tag}→{arch_tag} [BASELINE]; Phase Xa stores threshold_kept_indices dict; Phase 4+5 per-threshold feature slicing; _log_top_features()→_log_all_features(); 13 redundant log lines removed; stale-variable bug identified (dropped_indices on lines 187-188) |
| 3.7 | 2026-05-25 | `[label_threshold_search]` block deleted (redundant with `[diagnostic]`), per-threshold diagnostic expanded to full 24-metric train+val set (matches section format), `pred_*`→`prediction_*` (pred_mean, pred_std, pred_min, pred_max), `val_stdpred`/`val_pctabovethresh` standardized → `validation_standard_deviation_prediction`/`validation_percentage_above_threshold` across all 4 sections, trailing comma removed from `LABEL_THRESHOLD=20.0,` | Remove redundant threshold search logging, standardize remaining shorthand metrics, match diagnostic format to section format |
| 3.8 | 2026-05-25 | LightGBM: `scale_pos_weight`→`class_weight='balanced'` for auto class balancing; CatBoost: `auto_class_weights` default `'Balanced'`→`'SqrtBalanced'` for safer weight scaling on imbalanced data; LightGBM HPO space pruned (`scale_pos_weight` removed) | Eliminate manual weight calculation, reduce overfitting risk from extreme class weights |
| 3.9 | 2026-05-25 | DENSE, RNN, LSTM builders: added `FOCAL_LOSS_CONFIG` checks for per-architecture FocalLoss support; Dense and LSTM HPO spaces: added `loss_function` param for BCE/FocalLoss toggle | Enable FocalLoss on remaining active neural architectures for precision-focused training |
| 3.10 | 2026-05-25 | Reduced threshold search from 11 to 3 increments (20→10→0, step -10.0); centralized fallback constants in `chunk_01_config`; unified mismatched fallbacks across all phases; reduced HPO trials from 30 to 5 per architecture with `DEFAULT_HPO_TRIALS=10` fallback; reduced HPO epochs per trial from 20 to 5; moved `Label_Thresholds` log from analyzer to pipeline init for earlier log position; moved `Temporal Coverage` from 4 `print()` lines to single `self.log()` right after data loading; removed blank line from pipeline init; consolidated per-architecture predicted-signal data rows into single intersection table (logged after all architectures, Ticker_id present in every arch); CSV Phase value `Val`→`Validation` | Faster pipeline runs, consistent defaults, cleaner log ordering |
| 3.11 | 2026-05-31 | VAE serialization fix: `@keras.saving.register_keras_serializable` decorator on `VAEClassifier`, `get_config()` expanded to save all 5 params, `from_config()` classmethod added | Keras 3 `load_model()` fails for Model subclass without proper serialization support |
| 3.12 | 2026-05-31 | GIS Precision Lever Plan audit: 4 categories of proposed changes found ineffective or wrong. **Removed**: all 12 FOCAL_LOSS_CONFIG α/γ changes (zero effect — Section 2 uses BCE, HPO overrides from search space, Transformer has no focal loss option); MIN_ENSEMBLE_SIZE 5→4 (controls tree count, not ensemble filtering); HPO_MIN_POSITIVE_PERCENTAGE/ABSOLUTE dict changes (dead keys — defined but never read). **Corrected**: HIGHLY_SKEWED_FEATURES list (missed features 10/12). **Revised plan**: 14 changes across 3 tiers, all zero-runtime. Full audit documented in shortmemory.txt §GIS Precision Lever Action Plan — Revised. | Code audit revealed FOCAL_LOSS_CONFIG is a fallback-only value overridden by HPO search space — config changes had no pipeline effect |
| 3.13 | 2026-06-01 | **Pipeline Iter 1 run completed** (Tier 1 applied). Best val P: LightGBM 0.5329 (+0.0179 vs baseline). Best inference P: CatBoost 0.7213 (+0.0120). 5 of 9 archs passed ensemble filter (0.52 threshold). **Decision gate**: improved but < 0.56 → proceed to Tier 2. **Tier 2 plan formulated**: 7 config changes from GIS Precision Lever audit. | Iter 1: 4/9 archs improved, best 0.5329 (LightGBM); XGBoost/RNN/LSTM collapsed (non-deterministic HPO); VAE regressed from 0.5416→0.4842. Tier 2 tightens safeguard gates + winsorization to raise precision floor. |
| 3.14 | 2026-06-01 | **Tier 2 config applied + Iter 2 run + S4→S5 bug discovered**. Winsorization tightened (1→2, 99→98). Safeguard gates raised (MIN_POSITIVE_PERCENTAGE 0.005→0.01, MIN_POSITIVE_ABSOLUTE 50→100, MIN_PRECISION_OVER_BASELINE 0.01→0.02, MIN_POS_PRED_RATIO 0.0001→0.001, MAX_POS_PRED_RATIO 0.70→0.60). 3 of 7 structurally dead (overridden by SKLEARN/NEURAL_SAFEGUARDS). **Iter 2**: CNN dramatically improved to 0.6537 (best). 5/9 archs pass ensemble filter. **Decision gate**: CNN 0.6537 ≥ 0.56 → ENTER OPTIMIZE PHASE. **Bug discovered**: S4→S5 model carry-forward in `chunk_18_phase_4_ensemble.py:789-809`. Section 4 finds HPO model achieves better precision at a different label_threshold, but Section 5 selects `threshold_opt_model` (pre-HPO) when `hpo_improved=False`. Affects XGBoost (S5=0.4899, should be 0.5629). **Tier 3 partial**: WINSORIZE_PERCENTILE_HIGH 98→95, ENSEMBLE_MIN_PRECISION 0.52→0.53. **Plan**: apply Tier 4 code fix + remaining Tier 3 config. | Iter 2: 5/9 pass, CNN 0.6537 leads. XGBoost bugged. S4→S5 carry-forward analyzed per-architecture. RNN collapsed. Decision gate triggers optimization phase. Next: apply code fix + remaining config tweaks. |
| 3.16 | 2026-06-01 | **Tier 3 FocalLoss alpha expansion + Section 1 all-thresholds-rejected bug fix**. Config: added `focal_loss` to Transformer HPO search space; expanded alpha ranges to include `[0.25, 0.5]` for all neural architectures (Dense, CNN, RNN, LSTM, VAE, Transformer); lowered static FOCAL_LOSS_CONFIG defaults to `alpha=0.25, gamma=2.0` for all neural archs except VAE (kept 0.75/1.5). Bug fix: `chunk_18_phase_4_ensemble.py` lines 412-461 — detected empty `all_results` from `find_optimal_threshold` (when all 3 label thresholds rejected), logs baseline metrics instead of all-zeros. Affected Dense/RNN/LSTM in Iter 3 (LN366/LN521/LN592). | Tier 3: lower alpha values (0.25, 0.5) enable FocalLoss to penalize false positives more aggressively in HPO. Bug fix: clean logging when baseline model predicts < 0.5 at highest threshold. |
| 3.17 | 2026-06-01 | **Tier 7: temporal sample_weight added to model training**. `chunk_14_models_trainer.py`: `train_model()` accepts `sample_weight`, splits with train_test_split, passes to `model.fit()`. `_train_sklearn_model()` accepts `sample_weight`, passes to sklearn. `chunk_11_models_sklearn.py`: `SklearnModelWrapper.fit()` forwards `**kwargs` to underlying estimator. `chunk_18_phase_4_ensemble.py`: all 4 final training calls pass `sample_weight=np.sqrt(weights_train)`. Combined with existing sqrt feature scaling, total temporal multiplier = 9x (was 3x from features only). | Tier 7: temporal sample_weight closes the loop — `weights_train`/`weights_val` were computed but never passed to the loss function. Phase Xb showed 0/9 architectures had positive temporal gap at 3x. 9x aims to drive meaningful recency focus. |
| 3.18 | 2026-06-03 | **Cross-phase model propagation fix** — 3 code bugs in `chunk_18_phase_4_ensemble.py`: (1) line 692 `model_for_post_hpo` now uses `threshold_opt_model` when `not hpo_improved` instead of always preferring `hpo_best_model` — Section 4 post-HPO threshold search now runs on the elected Section 3 model. (2) line 711 improvement comparison changed from `> hpo_val_precision` to `> section3_precision` — Section 4 correctly compares against the elected branch's precision (was using HPO precision even on Branch 1). (3) line 775 override guard updated similarly. Documentation: model propagation chain documented in README.md §Phase-to-Phase Model Propagation Logic; cardinal rule expanded with (d) idempotent evaluation violation and three-branch logic in longmemory.txt; all 6 new bugs (F15, B, F1, F2, G, H) catalogued in shortmemory.txt; SPEC.md version history updated. | Fixes buggy model carry-forward on Branch 1 (HPO didn't improve). Section 4 was running post-HPO threshold search on the rejected HPO model instead of the elected pre-HPO model, and comparing against the wrong precision metric. The model propagation chain is now correctly enforced at every gate. |
| 3.19 | 2026-06-04 | **Bug I fix (consensus→final_predictions) + Bug J fix (S3 aux metric carry-over) + min_votes 6→5**. Bug I: final_predictions built from consensus vote (≥5 archs agree) instead of single best arch (VAE artifact P=1.0000, TP=0). Bug J: S3 auxiliary metrics (spec/FPR/F2/MCC/balacc/kappa/informedness/markedness) carried over from S1/baseline on Branch 1/3 instead of recalculated from section3_pred. min_votes lowered from 6 to 5 because CatBoost dropped to 0.5296 (<0.53), leaving only 5 archs in Phase 5. Pipeline Iter 4 run: best inference P=0.7341 (Dense), up from P=0.0000 (VAE artifact). | Fixes two functional failures: consensus voting now actually drives output; S3 auxiliary metrics are self-consistent with confusion matrix on carried-over branches. min_votes change ensures consensus can be reached with typical 5 loaded archs. |
| 3.20 | 2026-06-08 | **Metric log standardization — 31 bare/unprefixed lines fixed across 6 files**. Added missing `inference_`, `VALIDATION_`, `validation_` tags. Expanded abbreviations: `P`→`precision`, `R`→`recall`, `TP`→`true_positives`, `FP`→`false_positives`, `FN`→`false_negatives`, `TN`→`true_negatives`, `HPO`→`hyperparameter_optimization`, `f1 Score`→`f1`, `best_precision=`→`validation_best_precision=`, `train_P=`→`train_precision=`, `val_p=`→`validation_precision=`, `Recent P`→`validation_recent_precision`, `Older P`→`validation_older_precision`, `Pre-HPO P`/`Post-HPO P`→`Pre_hyperparameter_optimization_validation_precision`/`Post_hyperparameter_optimization_validation_precision`. Files: chunk_20 (15 lines), chunk_18 (9), chunk_21 (4), chunk_19 (2), chunk_XX_phase_b (2), chunk_12 (1). | Completes Category N/O abbreviation expansion from G11 refactoring plan. All metric log lines across the pipeline now carry proper train_/VALIDATION_/validation_/inference_ prefix with full metric names. |
| 3.21 | 2026-06-08 | **Two functional error fixes from iter6 review**. Error 1: post-pipeline `ValueError: mix of binary and continuous targets` — `final_predictions` were float probabilities consumed by `precision_score` expecting binary `{0,1}`. Fixed at source (`chunk_19_phase_5_optimization.py:443`: `.predict()` binarized at 0.5) plus defensive binarization at both `precision_score` call sites in `chunk_20_pipeline_main.py:135-138,586-587`. Error 2 (Error 3 from analysis): ensemble evaluated at wrong label threshold — `max(final_thresholds)` selected Dense's 20.0 when only VAE (threshold 0.0) passed the precision filter. Fixed in `chunk_18_phase_4_ensemble.py`: threshold computed after filtering loop using **mode** of passing architectures' thresholds instead of `max()` over all archs. | Fixes two functional errors found in `pipeline_cpu_iter6.log` — both pre-existed in iter5 (not caused by metric standardization cosmetic changes). |
| 3.22 | 2026-06-08 | **Log cosmetics cleanup: removed 13 separator lines (=====/-----) from chunk_XX_feature_importance.py (7) and chunk_20_pipeline_main.py (6); removed "..." from 3 log lines; removed redundant histogram log (LN119) — all 20 bins now inline via `format_diagnostic_string`; removed "before_threshold_optimization:" and "Pipeline Complete!" lines. Metric naming: LN120 `prediction_threshold`→`prediction_binary_split` with explicit `VALIDATION predictions at LABEL_THRESHOLD=20.0`; chunk_19 table headers expanded `FN/TN/TP/FP`→`inference_false_negatives/inference_true_negatives/inference_true_positives/inference_false_positives` with `inference_` prefix on all metric columns; `sorted by Val precision`→`sorted by validation_precision`. Added `LABEL_THRESHOLD=` context line before baseline diagnostics. **Global TF progress bar suppression**: added `tf.keras.utils.disable_interactive_logging()` + `logging.getLogger('tensorflow').setLevel(logging.ERROR)` in chunk_20 pipeline startup to prevent TF progress bars (━━━━━, 0s/step) from bleeding into log output. | Pure cosmetic cleanup (separators, ellipsis, empty lines, redundant logging) plus global TF suppression — no functional changes. Files: chunk_18 (8 edits), chunk_19 (4 edits), chunk_20 (9 edits), chunk_04 (1 edit), chunk_XX_feature_importance (8 edits). |


---

## 3.2 Cross-Reference Guide

### Functional Requirements to Logging Mapping

| FR | Produces | Metrics/Logging |
|----|----------|-----------------|
| FR-01 (Data Loading) | Section 1.1.1 | Baseline diagnostics |
| FR-02 (Class Imbalance) | Section 1.3 | Precision, Recall (affected by class weight configuration) |
| FR-03 (Train/Val Split) | Section 1.1.1 | Data loading logs |
| FR-04 (Temporal Weighting) | Section 1.1.1 | Temporal feature logs |
| FR-05 (Train Architectures) | Section 1.1.5 | Final model summary |
| FR-06 (Threshold Opt) | Section 1.1.2 | Per-threshold P, R, F1, AUC |
| FR-07 (HPO) | Section 1.1.3 | Trial parameters, validation metrics |
| FR-08 (Ensemble) | Section 1.1.4 | Post-HYPERPARAMETER_OPTIMIZATION comparison, ensemble weights |
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
| Phase 4c | 1.1.4 | Post-HYPERPARAMETER_OPTIMIZATION threshold search |
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
| legacy files/chunk_06_data_augmentation.py | Signal case augmentation | → FR-01 (moved to legacy) |
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

**Root Cause**: scale_pos_weight=500 + max_depth=7 + n_estimators=500 → extreme overfitting. Phase 4 threshold search produces TP=0 on validation due to optimal_threshold=2.0 producing zero TPs (all confusion matrix = 0). TRAIN_PRECISION=0.2134 but VALIDATION_PRECISION=0.2527 only because Val predictions at threshold 0.5 yield 0 TP + 0 FP → precision undefined → zero_division=1.0 default, but confusion matrix shows all zeros. **Fix Applied**: Lower n_estimators [100-300], shallower depth [3-7], lower scale_pos_weight [200-500], higher regularization, add colsample_bytree, gamma. **Evidence**: pipeline_cpu.log lines 1140-1152.

## 4.7 Phase 5 Crash — df_with_all_cols AttributeError (May 11, 2026)

| Item | Value |
|------|-------|
| **Location** | chunk_19_phase_5_optimization.py line 272 |
| **Error** | `AttributeError: 'NoneType' object has no attribute 'columns'` |
| **Cause** | `df_with_all_cols = context.get('df_with_all_cols')` returned None; no guard around usage |
| **Fix Applied** | Added `if df_with_all_cols is not None` guard + `sorted_results = []` init before block (Fix A), added pruned_feature_indices lookup (Fix B), added SklearnModelWrapper in loader (Fix C), wrapped prediction output in None guard (Fix D) |
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
| `HYPERPARAM_OPTIMIZATION_TRIALS` | 30 | **5** |

#### max_trials Raised (chunk_21_hyperparam_optimizer.py)
| Parameter | Was | Now |
|-----------|-----|-----|
| Safety cap | 500 | **1000** |

#### Phase 5 Fixes A-D (chunk_19 + chunk_22)
- **Fix A**: initialized `sorted_results = []` before `if architecture_results:` block (chunk_19 line 328)
- **Fix B**: added pruned_feature_indices lookup for 24→19 pruning in Phase 5 (chunk_19 line 131)
- **Fix C**: patched model loader to detect sklearn archs and load via joblib + SklearnModelWrapper (chunk_22 lines 113-122)
- **Fix D**: wrapped prediction output section in `if df_with_all_cols is not None` guard (chunk_19 line 272)

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
*Last updated: 2026-06-01*  
*Version: 3.17*

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
- **class_weight**: LightGBM uses `'balanced'` (auto class balancing); **scale_pos_weight**: XGBoost (dynamic ratio), CatBoost (dynamic ratio, unused in constructor)

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
- LightGBM: class_weight='balanced', min_child [100,200], added reg_alpha/reg_lambda
- XGBoost: scale_pos_weight [400,500], min_child [50,100,200], added reg_alpha/reg_lambda
- CatBoost: iterations [100,200] (reduced), l2_leaf_reg [3,5,10] (extended)

---

## All Implementation Items Complete

---

## PROJECT_LEXICON

Complete reference of all cosmetic log labels, section tags, metric keys, and abbreviations used across the pipeline. Labels follow lowercase_snake_case convention unless noted. Exceptions that stay UPPER_SNAKE_CASE: VALIDATION_PRECISION, VALIDATION_TRUE_POSITIVES, VALIDATION_TRUE_NEGATIVES, LABEL_THRESHOLD, [HYPERPARAMETER_OPTIMIZATION_SEARCH], TRIAL, OPTIMAL. Title-case metrics unchanged: Brier, Kappa, Informedness, Markedness, Gini.

### G1: Section / Status Tags

**Log Section Tags** (bracketed identifiers in output)

| Tag | Description | Source File(s) |
|-----|-------------|---------------|
| `[section_1_baseline]` | Baseline threshold search | chunk_18 |
| `[SECTION_2_HYPERPARAMETER_OPTIMIZATION_SEARCH]` | Pre-HPO threshold search | chunk_18 |
| `[section_3_pre_hyperparameter_optimization]` | Pre-HPO evaluation | chunk_18 |
| `[section_4_post_hyperparameter_optimization]` | Post-HPO evaluation | chunk_18 |
| `[section_5_final]` | Final training summary | chunk_18 |
| `[baseline]` | Baseline model evaluation | chunk_18 |
| `[label_threshold_OPTIMAL]` | Optimal threshold found | chunk_18 |
| `[final]` | Final training step | chunk_18 |
| `[post_hyperparameter_optimization]` | Post-HPO results | chunk_18 |
| `[pre_hyperparameter_optimization]` | Pre-HPO results | chunk_18 |
| `[diagnostic]` | Diagnostic info (expanded from `[diag]`) | chunk_18 |
| `[diagnostic-hyperparameter_optimization]` | Diagnostic HPO info (expanded from `[diag-hpo]`) | chunk_18 |
| `[diagnostic-ensemble]` | Diagnostic ensemble info (expanded from `[diag-ensemble]`) | chunk_18 |
| `[HYPERPARAMETER_OPTIMIZATION_SEARCH]` | HPO search header | chunk_21 |
| `[best_trial]` | Best HPO trial result | chunk_21 |
| `[cross_threshold]` | Cross-threshold feature analysis | chunk_XX |
| `[class_distribution]` | Class distribution section | chunk_02, chunk_16 |
| `[data_split]` | Data split configuration | chunk_16 |
| `[statistics]` | Statistics section | chunk_02, chunk_16 |
| `[date]` | Date coverage section | chunk_02 |
| `[feature_engineering]` | Feature engineering step | chunk_05 |
| `[running]` | Pipeline running status | chunk_20 |
| `[passed]` | Test/step passed | chunk_20 |
| `[skipped]` | Skipped iteration | chunk_20 |
| `[timing]` | Timing measurement | chunk_20 |
| `[error]` | Error status | chunk_20 |
| `[warning]` | Warning status | chunk_20 |
| `[ok]` | Status passed | chunk_12, chunk_18 |
| `[rejected]` | Threshold rejected | chunk_12, chunk_18 |
| `[phase_1_5]` | Phase 1.5 diagnostics | chunk_02, chunk_04 |

**Inline Status / Warning Labels**

| Label | Context | File(s) |
|-------|---------|---------|
| `sanity_check:` | Data validation warnings | chunk_16 |
| `feature_quality_analysis:` | Feature statistics header | chunk_02 |
| `temporal_coverage:` | Date coverage header | chunk_02 |
| `before_threshold_optimization:` | Baseline diagnostics before threshold search | chunk_18 |
| `info` | Informational log level | All |
| `warning` | Warning log level | All |
| `running` | Pipeline running | chunk_20 |

### G2: Model / Architecture Names

| Abbreviation | Full Name | Log Reference | Doc Reference |
|-------------|-----------|---------------|---------------|
| CatBoost | Categorical Boosting | log:108,120,167,182,557,587,599 | SPEC:590,659-668 |
| LightGBM | Light Gradient Boosting Machine | log:109,121,141,168,181,590,670 | SPEC:590,670-684 |
| XGBoost | Extreme Gradient Boosting | log:110,122,169,184,591,686-700 | SPEC:591,686-700 |
| RF | Random Forest | log:~110 area | SPEC:593,832 |
| CNN | Convolutional Neural Network | log:112,1155-1260+ | SPEC:596,717-731 |
| RNN | Recurrent Neural Network | log:113 area | SPEC:597,734-745 |
| LSTM | Long Short-Term Memory | log:114 area | SPEC:598,748-759 |
| VAE | Variational Autoencoder | log:115 area | SPEC:594,762-774 |
| Dense | Dense (Fully-Connected) Neural Network | log:111,170,1107-1154 | SPEC:595,703-715 |
| GBM | Gradient Boosting Machine | log:direct matches | SPEC:590-593,832-834 |
| Transformer | Attention-based Neural Network | log:1302-1331 | SPEC:599,776-790 |

### G3: Metric Keys (all-prefix variants)

**Base metric keys** — each appears with TRAIN_, VALIDATION_, and INFERENCE_ prefix.

| Old Key | Current Key | Description | Source File(s) |
|---------|-------------|-------------|----------------|
| P | PRECISION | Precision score | chunk_04, chunk_12 |
| R | RECALL | Recall / sensitivity | chunk_04, chunk_12 |
| F1 | F1_SCORE | F1 score (harmonic mean) | chunk_04, chunk_12 |
| F2 | F2_SCORE | F2 score (recall weighted 2x) | chunk_04, chunk_12 |
| Spec | SPECIFICITY | Specificity / true negative rate | chunk_04, chunk_12 |
| FPR | FALSE_POSITIVE_RATE | False positive rate | chunk_04, chunk_12 |
| TP | TRUE_POSITIVES | True positives count | chunk_04, chunk_12 |
| FP | FALSE_POSITIVES | False positives count | chunk_04, chunk_12 |
| TN | TRUE_NEGATIVES | True negatives count | chunk_04, chunk_12 |
| FN | FALSE_NEGATIVES | False negatives count | chunk_04, chunk_12 |
| MaxPred | MAX_PREDICTION | Maximum prediction value | chunk_12, chunk_14 |
| MeanPred | MEAN_PREDICTION | Mean prediction value | chunk_12, chunk_14 |
| StdPred | STD_PREDICTION | Std deviation of predictions | chunk_12, chunk_14 |
| PctAboveThresh | PCT_ABOVE_THRESHOLD | % predictions above 0.5 | chunk_12, chunk_14 |
| OptThresh | OPTIMAL_THRESHOLD | Optimal threshold (Youden's J) | chunk_12 |
| BalAcc | BALANCED_ACCURACY | Balanced accuracy | chunk_12 |
| AUC | AUC | ROC AUC (kept as-is) | chunk_04, chunk_12 |
| MCC | MCC | Matthews correlation coeff (kept as-is) | chunk_04, chunk_12 |
| PRAUC | PRAUC | PR AUC (kept as-is) | chunk_04, chunk_12 |
| Brier | Brier | Brier score (Title-case) | chunk_04, chunk_12 |
| Kappa | Kappa | Cohen's kappa (Title-case) | chunk_04, chunk_12 |
| Informedness | Informedness | Sensitivity + Specificity - 1 (Title-case) | chunk_04, chunk_12 |
| Markedness | Markedness | Precision + NPV - 1 (Title-case) | chunk_04, chunk_12 |
| Gini | Gini | Gini coefficient (Title-case) | chunk_04, chunk_12 |

**Prefix conventions for metric keys in log output:**

| Prefix | Usage | Section(s) |
|--------|-------|------------|
| `train_` | Training split metrics | section_5_final |
| `validation_` | Validation split metrics | sections 1-5, HPO trials, threshold search |
| `inference_` | Inference/prediction metrics | phase 5 |

### G4: Data / Configuration Labels

**Data Loading & Stats**

| Label | Description | File(s) |
|-------|-------------|---------|
| `data_loaded:` | Dataset load confirmation | chunk_16, chunk_05 |
| `total_samples:` | Total sample count | chunk_16, chunk_02 |
| `signal_cases:` | Signal case count | chunk_16, chunk_02 |
| `normal_cases:` | Normal case count | chunk_16, chunk_02 |
| `imbalance_ratio:` | Class imbalance ratio | chunk_16, chunk_02 |
| `data_concentration:` | Post-concentration sample count | chunk_16 |
| `data_preprocessing_complete:` | Feature count after preprocessing | chunk_16 |
| `min:` / `max:` / `mean:` / `median:` / `standard_deviation:` | Target value distribution stats | chunk_16 |
| `date_range:` / `unique_dates:` | Date coverage | chunk_02 |

**Data Split**

| Label | Description | File(s) |
|-------|-------------|---------|
| `dataset_split_ratio:` | Train/validation split ratio | chunk_18 |
| `train:` / `validation:` / `inference:` / `total:` | Data split summary headers | chunk_16 |
| `target_column:` / `date_split:` / `remaining_dates:` | Column/split metadata | chunk_18 |
| `validation_split:` / `training_split:` | Split date counts | chunk_18 |
| `validation_set:` / `training_set:` | Set row counts | chunk_18 |
| `train_positives:` / `validation_positives:` | Per-threshold positive counts | chunk_18 |
| `sampling:` | Sampling config (size / enabled / forced) | chunk_20 |
| `temporal_weights:` | Temporal weight stats (min / max / mean) | chunk_17 |

**Configuration / Threshold Labels**

| Label | Description | File(s) |
|-------|-------------|---------|
| `LABEL_THRESHOLD` | Label binarization threshold | chunk_12, chunk_18, chunk_19, chunk_21 |
| `prediction_binary_split` | Binary split threshold (0.5) | chunk_12, chunk_18, chunk_19 |
| `hyperparameters` | Hyperparameter dict | chunk_18 |
| `hyperparameter_optimization` | HPO trials/config | chunk_18, chunk_20, chunk_21 |
| `ChangeY` | Target column name | chunk_16 |

### G5: Prediction Distribution Labels

| Label | Description | File(s) |
|-------|-------------|---------|
| `predictions:` | Prediction stats header (mean / standard_deviation / min / max) | chunk_18, chunk_19 |
| `train_predictions:` | Train prediction stats | chunk_18 |
| `validation_predictions:` | Validation prediction stats | chunk_18 |
| `inference_predictions:` | Inference prediction stats | chunk_19 |
| `prediction_mean` | Per-threshold mean prediction | chunk_12 |
| `prediction_standard_deviation` | Per-threshold std prediction | chunk_12 |
| `prediction_min` | Per-threshold min prediction | chunk_12 |
| `prediction_max` | Per-threshold max prediction | chunk_12 |
| `percentage_positive_at_prediction_binary_split` | % positive predictions at 0.5 split | chunk_12 |
| `percentiles` | Prediction percentile distribution | chunk_04 |
| `histogram` | Prediction histogram bins | chunk_04 |

### G6: Report / Table Headers

| Header | Description | File(s) |
|--------|-------------|---------|
| `feature_importance_analysis_report` | Feature importance analysis header | chunk_XX |
| `consolidated_ranking (1 = most_important)` | Consolidated feature ranking | chunk_XX |
| `top_features_per_method` | Top features per method | chunk_XX |
| `spearman_correlation` | Spearman correlation method | chunk_XX |
| `tree_importance (rf+gbm)` | Tree-based importance method | chunk_XX |
| `permutation_importance` | Permutation importance method | chunk_XX |
| `neural_weight_magnitude` | Neural weight importance method | chunk_XX |
| `shap_values` | SHAP importance method | chunk_XX |
| `ablation_study (auc)` | Ablation study method | chunk_XX |
| `runtime:` / `correlation:` / `tree:` | Method runtime/type labels | chunk_XX |
| `permutation:` / `neural:` / `shap:` / `ablation:` | Method runtime/type labels | chunk_XX |
| `architecture ranking (by validation precision)` | Architecture ranking header | chunk_18 |
| `hyperparameter_optimization impact summary` | HPO impact header | chunk_18 |
| `training time summary` | Training time header | chunk_18 |
| `final prediction results (sorted by precision)` | Final results header | chunk_19 |
| `features:` / `samples:` | Feature importance input dimensions | chunk_XX |
| `borderline:` | Features pruned in some thresholds | chunk_XX |
| `best_for_recent_signals:` / `worst_for_recent_signals:` / `gap:` | Temporal precision gap analysis | chunk_XX |

### G7: Hyperparameter Names

| Abbreviation | Full Name | Log Reference | Doc Reference |
|-------------|-----------|---------------|---------------|
| lr | Learning rate | log:557,600,852,977,1152 | SPEC:664,676,690 |
| alpha | Focal Loss alpha / reg_alpha | log:HPO trial lines | SPEC:710,726,770 |
| gamma | Focal Loss gamma / tree gamma | log:HPO trial lines | SPEC:711,727,771 |
| dropout | Dropout rate | log:HPO trial lines | SPEC:707,723,767 |
| latent_dim | Latent dimension (VAE) | log:557,747,971,1150 | SPEC:765 |
| epochs | Number of training epochs | log:HPO trial lines | SPEC:709,725,740 |
| units | Dense/RNN units | log:1102,1152 | SPEC:705,737 |
| layers | Number of NN layers | log:1102,1152 | SPEC:706 |
| filters | Number of CNN filters | log:CNN trial lines | SPEC:721 |
| kernel_size | CNN kernel size | log:CNN trial lines | SPEC:722 |
| pooling | CNN pooling type (max/avg/none) | log:CNN trial lines | SPEC:729 |
| batch_size | Mini-batch size | log:Dense trial lines | SPEC:712 |
| activation | Activation function | log:Dense trial lines | SPEC:713 |
| n_estimators | Number of trees | log:XGB tree trials | SPEC:673,689 |
| max_depth | Maximum tree depth | log:XGB tree trials | SPEC:690 |
| class_weight | Class weight mode (LightGBM, 'balanced') | log:LightGBM trial lines | SPEC:676 |
| scale_pos_weight | Positive class weight ratio (XGBoost, CatBoost) | log:XGB trial lines | SPEC:692 |
| min_child_weight | Min child weight (XGBoost) | log:XGB trial lines | SPEC:693 |
| reg_alpha | L1 regularization on weights | log:tree trial lines | SPEC:678,694 |
| reg_lambda | L2 regularization on weights | log:tree trial lines | SPEC:679,695 |
| subsample | Row sampling ratio | log:tree trial lines | SPEC:680,696 |
| colsample_bytree | Column sampling ratio per tree | log:tree trial lines | SPEC:681,697 |
| num_leaves | Max leaves (LightGBM) | log:LightGBM HP | SPEC:674 |
| min_child_samples | Min samples per leaf (LightGBM) | log:LightGBM HP | SPEC:677 |
| min_split_gain | Minimum split gain (LightGBM) | log:LightGBM HP | SPEC:682 |
| l2_leaf_reg | L2 leaf regularization (CatBoost) | log:557,587,599 | SPEC:666 |
| auto_class_weights | Automatic class weight mode | log:557,587 | SPEC:665 |
| SqrtBalanced | Square-root balanced class weighting | log:557,587 | SPEC:665 |
| loss_function | Loss function name | log:CNN trial lines | — |
| binary_crossentropy | Binary cross-entropy loss (BCE) | log:trial lines | — |
| focal_loss | Focal loss function | log:~1211+ | — |

### G8: Financial Feature Names

| Feature | Description | Log Reference |
|---------|-------------|---------------|
| SMA50/20/200 | Simple Moving Average (50/20/200 period) | log:159-270 |
| ATR | Average True Range | log:159-270 |
| RSI_14 | Relative Strength Index (14 period) | log:159-270 |
| Market_Cap | Market Capitalization | log:throughout |
| Perf_Half_Y | 6-month performance | log:throughout |
| Perf_Quarter | 3-month performance | log:throughout |
| Perf_Month | 1-month performance | log:throughout |
| Perf_Week | 1-week performance | log:throughout |
| Perf_Year | 1-year performance | log:throughout |
| Perf_YTD | Year-to-date performance | log:throughout |
| Prev_Close | Previous closing price | log:throughout |
| Avg_Volume | Average volume | log:throughout |
| Rel_Volume | Relative volume | log:throughout |
| Change | Price change | log:throughout |
| Price | Current price | log:throughout |
| Volume | Trading volume | log:throughout |
| Price_to_52W_High | Price / 52-week high ratio | log:throughout |
| Price_to_52W_Low | Price / 52-week low ratio | log:throughout |
| 52W_High | 52-week high price | log:throughout |
| 52W_Low | 52-week low price | log:throughout |
| Ticker_id | Ticker identifier (dropped) | log:throughout |
| Volume_to_Avg_Volume | Volume / average volume ratio | log:throughout |

### G9: HPO / Phase Labels

| Label | Description | File(s) |
|-------|-------------|---------|
| `progress:` | HPO search progress counter | chunk_21 |
| `TRIAL:` | Individual HPO trial counter | chunk_21 |
| `target:` | HPO target metric | chunk_21 |
| `best_precision:` | Best precision tracking during HPO | chunk_21 |
| `phase_5_total_time:` | Phase 5 total runtime | chunk_19 |
| `data_points_evaluated:` | Phase 5 data count | chunk_19 |
| `search` | HPO search phase | chunk_21 |
| `OPTIMAL` | Optimal label threshold found | chunk_12 |
| `prediction_buckets` | Prediction bucket distribution | chunk_18 |

### G10: Terms NOT in pipeline_cpu.log (documentation only)

CI/CD, GPU, GNN, JSON, HTML, WSL, TF/TensorFlow, sklearn, FR, AC, SSR, SOP, GIS, KS, MI, PSI, SNR, LogitComp, Bhattacharyya, Transformer (as doc term), BCE, NaN, CUDA, cuDNN, cuBLAS, cuFFT, CSV

### G11: Cosmetic Label Refactoring — Implementation Plan

**Date**: 2026-05-24
**Scope**: Lowercase all cosmetic log labels across the pipeline. Dict keys unchanged.
**Exceptions** (stay UPPER_SNAKE_CASE): VALIDATION_PRECISION, VALIDATION_TRUE_POSITIVES, VALIDATION_TRUE_NEGATIVES, LABEL_THRESHOLD, [HYPERPARAMETER_OPTIMIZATION_SEARCH], TRIAL, OPTIMAL.

**Execution Categories:**

| Cat | Description | Count | Safety Pattern |
|-----|-------------|-------|----------------|
| A | Bracket tags: `[BASELINE]`→`[baseline]` (except `[HYPERPARAMETER_OPTIMIZATION_SEARCH]`) | ~38 | `replaceAll` — globally unique |
| B | TRAIN_ metric labels: `TRAIN_PRECISION=`→`train_precision=` | 25 | `={ ` suffix — never matches `'KEY'` dict keys |
| C | INFERENCE_ metric labels: `INFERENCE_PRECISION=`→`inference_precision=` | 24 | Same `={ ` pattern; 18 strings dual-used with dict keys, safe |
| D | VALIDATION_ non-exceptions: `VALIDATION_RECALL=`→`validation_recall=` | 20 (3 stay) | Values are variables/attributes, not dict keys |
| E | Standalone/base metric labels: `PRECISION=`→`precision=` | ~15 | `={ ` pattern only in f-string context |
| F | Colon-terminated: `DATA_LOADED:`→`data_loaded:` | ~50 | `replaceAll` — globally unique |
| G | Inline stats: `mean=0.0023` (already handled by parent label change) | ~12 | Covered by B/D/F cascading |
| H | PREDICTION_ diagnostic: `PREDICTION_MEAN=`→`prediction_mean=` | 6 | Unique to chunk_12:508 |
| I | Feature importance: `SPEARMAN_CORRELATION`→`spearman_correlation` | 9 | Unique to chunk_XX |
| J | HPO labels: `PROGRESS:`→`progress:`, `Trial`→`TRIAL`, `OPTIMAL` stays | 8 | `TRIAL` and `OPTIMAL` are UPPER exceptions |
| K | CSV/table headers: `MCC`→`mcc`, `Precision`→`precision` | ~15 | Standalone strings |
| L | Standalone one-offs: `SAMPLES`→`samples`, `FEATURES`→`features` | ~20 | Unique per file |
| M | Prose: `Class Distribution`→`class distribution` | ~25 | Initial caps → lowercase |
| N | Abbreviation expansion: `[diag]`→`[diagnostic]`, `[diag-hpo]`→`[diagnostic-hyperparameter_optimization]`, `[diag-ensemble]`→`[diagnostic-ensemble]`, `[post-hpo]`→`[post hyperparameter_optimization]`, `PredBuckets:`→`prediction_buckets:`, `Percentiles:`→`percentiles:`, `Hist:`→`histogram:`, plus inline prose labels | 16 | chunk_04 (2), chunk_18 (14) |
| O | HPO label expansion + Phase logging removal: `[HPO]`/`[PRE-HPO]`→`[HYPERPARAMETER_OPTIMIZATION]`/`[PRE-HYPERPARAMETER_OPTIMIZATION]`, `hpo:`→`hyperparameter_optimization:`, all HPO/POST-HPO prose expanded, `Phase #:` removed, `[pass]`/`[time]`/`Running`/`Starting` deleted, `label_threshold=`→`LABEL_THRESHOLD=` in chunk_12 (6×), `Skipping`→`skipping` (2×), `P=`→`precision=`, `%pos@0.5=`→`percent_positive_at_0_5=`, `Train:`→`train:`, `Val:`→`validation:`, `[label_threshold_optimal]`→`[label_threshold_OPTIMAL]` | ~49 | chunk_12 (11), chunk_16 (4), chunk_17 (3), chunk_18 (~25), chunk_19 (4), chunk_20 (10), chunk_01/02/05/07 (4 print deletions) |
| P | Per-threshold diagnostic standardization: `pred_*`→`prediction_*`, lightweight (6) → full (24 train + 24 val), `[label_threshold_search]` block deleted (4 lines), `val_stdpred`/`val_pctabovethresh` standardization (4 sections × 2 labels each) | ~35 | chunk_12 (1 line replacement), chunk_18 (4 lines deleted + 8 labels expanded) |

**Files affected**: 19 chunk files
**Highest risk**: chunk_18 (~200 changes), chunk_19 (~50 changes), chunk_20 (~60 changes)
**Verification**: grep zero-survivors + py_compile all files
