# Software Specification Requirements (SSR) - Stock Analysis Ensemble

**Version**: 3.31  
**Date**: 2026-06-10  
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

### Section 5: Final Model Summary
Logged at completion:
- All 17 metrics
- Source (section3 or section4)
- Training epochs

**Source File**: chunk_18_phase_4_ensemble.py

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

### Metrics CSV Schema

Full CSV schema for `metrics_summary.csv`, generated by chunk_20_pipeline_main.py at end of each run:

```csv
Architecture,Phase,Loss,Epochs,Precision,Recall,AUC,F1,TP,FP,TN,FN,MaxPred,MeanPred,StdPred,PctAboveThresh,BestEpoch,TrainingTime,LabelThresh,HPO_Trials,HPO_Improvement,KeyHyperparams,TrainLoss,ValLoss,LossDelta,MCC,PRAUC,Specificity,BalancedAccuracy,PredictionThreshold
```

| Field | Description | Source |
|-------|-------------|--------|
| Architecture | Model name | - |
| Phase | Train, Val, or Inference | - |
| Loss | Loss function used (binary_crossentropy or focal_loss) | HPO metadata |
| Epochs | Number of training epochs | Training config |
| Precision | TP / (TP + FP) | calculate_precision() |
| Recall | TP / (TP + FN) | calculate_recall() |
| AUC | ROC-AUC score | calculate_auc() |
| F1 | F1 score | calculate_f1() |
| TP | True Positives | Confusion matrix |
| FP | False Positives | Confusion matrix |
| TN | True Negatives | Confusion matrix |
| FN | False Negatives | Confusion matrix |
| MaxPred | Maximum prediction value | Model output |
| MeanPred | Mean prediction value | Model output |
| StdPred | Standard deviation of predictions | Model output |
| PctAboveThresh | Percentage above threshold | (predictions >= threshold).mean() * 100 |
| BestEpoch | Best epoch from early stopping | Training history |
| TrainingTime | Time taken to train | time.time() |
| LabelThresh | Label binarization threshold | optimal threshold |
| HPO_Trials | Number of HPO trials | Config |
| HPO_Improvement | Precision improvement from HPO | best - baseline |
| KeyHyperparams | Key hyperparameters | String of key=value |
| TrainLoss | Final training loss | Training history |
| ValLoss | Final validation loss | Training history |
| LossDelta | Train - Val loss | train_loss - val_loss |
| MCC | Matthews Correlation Coefficient | (TP*TN - FP*FN) / sqrt(...) |
| PRAUC | Precision-Recall AUC | average_precision_score |
| Specificity | True Negative Rate | TN / (TN + FP) |
| BalancedAccuracy | Average of Recall + Specificity | (R + Spec) / 2 |
| Brier | Brier Score | (y - proba)² mean (lower=better) |
| Kappa | Cohen's Kappa | Inter-rater reliability |
| Informedness | Informedness | R + Spec - 1 |
| Markedness | Markedness | P + Spec - 1 |
| Gini | Gini Coefficient | 2 * AUC - 1 |
| OptThresh | Optimal Threshold | Threshold maximizing Youden's J |
| PredictionThreshold | Binary decision threshold | Fixed at 0.5 |

**Data Sources**: Train/Val metrics from `arch_final_metrics` context key; Inference metrics from `architecture_results` context key.

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
| AC-02 | All 9 architectures train successfully | Check logs | → FR-05 |
| AC-03 | Validation precision ≥ 0.60 | Check metrics output | → FR-06, FR-07, FR-11 |
| AC-04 | Models saved to ./saved_models/ | Directory inspection | → FR-09 |
| AC-05 | Predictions generated | Run legacy files/predict.py | → FR-10 |
| AC-06 | Each chunk independently testable | Run individual chunk files | → All FRs |
| AC-07 | HPO completes 20 trials per architecture | Check chunk_21 logs | → FR-07 |
| AC-08 | Logging captures all 5 sections per architecture | Check phase 4 logs | → Section 1.1.1 to 1.1.5 |

### Additional Acceptance Criteria

| ID | Criterion | Test Method |
|----|-----------|--------------|
| AC-9 | Pipeline completes without crash | Run full pipeline |
| AC-10 | All 9 architectures train successfully | Check logs |
| AC-11 | Ensemble precision ≥ 0.60 OR fallback used | Check metrics |
| AC-12 | Minimum 50 positive predictions | Check threshold validation |
| AC-13 | Models saved to ./saved_models/ | Verify directory |
| AC-14 | Signal predictions CSV generated | Check output file |
| AC-15 | Recall ≥ 10% for ensemble | Check metrics |

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

### Metrics by Pipeline Phase

| Phase | P | R | AUC | F1 | TP/FP/TN/FN | MCC | PR-AUC | Spec | BalAcc |
|-------|---|---|-----|-----|-------------|-----|--------|------|--------|
| Phase 4 Train | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phase 4 Val | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phase 5 Inference | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Phase Xb (Temporal) | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | - | - |
| HPO Trials | ✅ | ✅ | ✅ | ✅ | ✅ | - | - | - | - |
| Enhanced Metrics Table | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

#### Phase Details

- **Phase 4** (chunk_18): Train + Val metrics per architecture → Architecture ranking by VALIDATION_PRECISION
- **Phase 5** (chunk_19): Inference metrics per architecture → Final prediction results sorted by Precision
- **Phase Xb** (chunk_XX_phase_feature_analysis_b.py): Recent vs Older period metrics (P, R, AUC, F1, TP/FP/TN/FN)
- **HPO** (chunk_21): Per-trial metrics + Best trial summary
- **Enhanced Metrics Table** (chunk_20): CSV file with all 30 metrics

#### Standard ML Metrics Definition

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Minimize false alarms |
| **Recall** | TP / (TP + FN) | Catch actual signals |
| **AUC-ROC** | Area under ROC | Ranking ability |
| **F1 Score** | 2P×R / (P+R) | Balance P and R |
| **MCC** | (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Best for imbalanced data |
| **PR-AUC** | Area under Precision-Recall curve | Average precision score |
| **Specificity** | TN / (TN + FP) | True negative rate |
| **Balanced Accuracy** | (Recall + Specificity) / 2 | Average of sens/spec |

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

All 24 validation metrics are reported in SECTIONS 1-5 with VALIDATION_ prefix. Note: in the code, some validation keys appear in lowercase (`validation_false_positives`, `validation_false_negatives`) while others are uppercase (`VALIDATION_PRECISION`, `VALIDATION_TRUE_POSITIVES`). The tables below document both variants.

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

## 1.7 Metrics Pipeline Flow

### Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   TRAINING      │     │  VALIDATION     │     │   INFERENCE     │
│   (Phase 4)    │     │   (Phase 4)     │     │   (Phase 5)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │  X_train        │     │  X_val          │     │  X_inference    │
  │  y_train_raw   │     │  y_val_raw      │     │  y_raw          │
  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
           │                      │                      │
           ▼                      ▼                      ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  LABEL THRESHOLD CONVERSION (applied at each stage)             │
  │  y_train_binary = (y_train_raw >= optimal_label_threshold)      │
  │  y_val_binary   = (y_val_raw   >= optimal_label_threshold)     │
  │  y_inference    = (y_raw        >= optimal_label_threshold)     │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  MODEL TRAINING                                                  │
  │  model.fit(X_train, y_train_binary)                             │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  PREDICTIONS (probabilities [0,1])                              │
  │  train_pred = model.predict(X_train)                            │
  │  val_pred   = model.predict(X_val)                              │
  │  inf_pred   = model.predict(X_inference)                        │
  └────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  BINARY CONVERSION (Prediction Binary Split = 0.5)              │
  │  train_binary = (train_pred >= 0.5)                              │
  │  val_binary   = (val_pred   >= 0.5)                              │
  │  inf_binary   = (inf_pred   >= 0.5)                             │
  └────────┬─────────────────────────┬──────────────────┬────────────┘
           ▼                         ▼                  ▼
  ┌─────────────────┐     ┌──────────────────┐  ┌──────────────────┐
  │  TRAIN METRICS  │     │  VAL METRICS     │  │ INFERENCE        │
  │  (logged only)  │     │  (selection)     │  │ METRICS (output) │
  └─────────────────┘     └──────────────────┘  └──────────────────┘
```

### Detailed Flow by Stage

| Stage | Step | Code | Description |
|-------|------|------|-------------|
| **Training** (Phase 4) | Input | `X_train, y_train_raw` | 167,142 samples |
| | Label Transform | `y_train_optimal = (y_train_raw >= 2.0).astype(int)` | Convert continuous to binary |
| | Train | `model.fit(X_train, y_train_binary)` | Train model |
| | Predict | `train_pred = model.predict(X_train)` | Raw probabilities |
| | Binary | `train_binary = (train_pred >= 0.5).astype(int)` | Binary predictions |
| | Metrics | `calculate_precision(y_train_optimal, train_binary)` | **Train metrics** |
| **Validation** (Phase 4) | Input | `X_val, y_val_raw` | 71,014 samples |
| | Label Transform | `y_val_optimal = (y_val_raw >= 2.0).astype(int)` | Same threshold as train |
| | Predict | `val_pred = model.predict(X_val)` | Raw probabilities |
| | Binary | `val_binary = (val_pred >= 0.5).astype(int)` | Binary predictions |
| | Metrics | `calculate_precision(y_val_optimal, val_binary)` | **Val metrics** |
| | Usage | Selected for ensemble if P >= 0.40 | Architecture selection |
| **Inference** (Phase 5) | Input | `X_inference, y_raw` | 5,921 samples (newest date) |
| | Label Transform | `y_binary = (y_raw >= 2.0).astype(int)` | Same threshold as training |
| | Predict | `inf_pred = model.predict(X_inference)` | Raw probabilities |
| | Binary | `inf_binary = (inf_pred >= 0.5).astype(int)` | Binary predictions |
| | Metrics | `calculate_metrics(y_binary, inf_binary, inf_pred)` | **Inference metrics** |
| | Usage | Final signal predictions output | Ranked by precision |

### Key Points

| Aspect | Detail |
|--------|--------|
| **Label Threshold** | Same across all stages (2.0 default) |
| **Prediction Binary Split** | Same across all stages (0.5) |
| **Data Split** | TRAIN: 70%, VALIDATION: 30% of training data |
| **Inference Data** | Second newest date (held out from training) |
| **Metrics Used** | TRAIN: logged only; VALIDATION: selection; INFERENCE: final |

### Metrics Storage

| Stage | Where Stored | Used For |
|-------|-------------|----------|
| **Train** | `arch_final_metrics['TRAINING_PRECISION']` | Overfitting detection |
| **Val** | `arch_final_metrics['VALIDATION_PRECISION', 'VALIDATION_RECALL', 'VALIDATION_AUC', 'VALIDATION_F1_SCORE']` | Ensemble selection, auto-tune |
| **Inference** | `architecture_results[]` | Final report, signal output |

# SECTION 2: Functionality

## 2.1 Project Overview

See [README.md §Overview](./README.md#overview).

---

## 2.2 System Context

See [README.md §System Context](./README.md#system-context) for system context and data flow.

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

### Observed Data Characteristics

- ChangeY range: -99.75 to 32,500 (extreme outliers present)
- ~49% signal rate at label_threshold=0.0
- AUC ~0.5 for most architectures (features have low discriminative power)
- Focal loss used to address class imbalance

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

### Phase 4 Evaluation Sections (Model Propagation Logic)

Each architecture passes through 5 evaluation sections. At every gate the **better result propagates forward** — if a later phase doesn't improve, the prior phase's model+threshold is carried over unchanged.

```
Section 1 (threshold search across multiple label thresholds)
  │  best threshold → threshold_opt_model + optimal_threshold
  ▼
Section 2 (HPO — Optuna Bayesian optimization)
  │  best trial → hpo_best_model + hpo_val_precision
  ▼
Section 3 (election gate)
  ├─ HPO improved → use HPO model           [HYPERPARAMETER_OPTIMIZATION]
  └─ HPO did not improve → carry over S1    [PRE-HYPERPARAMETER_OPTIMIZATION]
  │
  ▼
Section 4 (post-HPO threshold search)
  ├─ post-HPO precision > S3 precision → adopt new threshold   [section4]
  └─ post-HPO did not improve → carry over S3 threshold+model  [section3]
  │
  ▼
Section 5 FINAL (uses Section 4's elected model+threshold)
```

#### Section 1 — Threshold Search
- Train model with **default hyperparameters** at each label threshold (20→10→0)
- Evaluate at prediction binary split 0.5 at each label threshold
- **Output**: `optimal_threshold` (label threshold with best val precision), `threshold_opt_model` (model trained at that threshold), all per-threshold metrics stored in `all_results`

#### Section 2 — Hyperparameter Optimization (HPO)
- Run Optuna Bayesian optimization (5–30 trials) using the `optimal_threshold` from Section 1 with the same 0.5 prediction binary split
- **Output**: `hpo_best_model` + `hpo_val_precision` + `best_hyperparams`
- HPO trials that fail MaxPred, TP, or min-precision gates are rejected silently; the surviving best trial is the "HPO best"

#### Section 3 — HPO Election Gate
Compare HPO precision vs pre-HPO precision at prediction binary split 0.5:

- **Branch 1** (HPO did NOT improve — 7 archs: CatBoost through Transformer): The pre-HPO model (`threshold_opt_model`) is the best. Model and threshold are identical to Section 1. **No re-evaluation** — copy all metrics from Section 1's `section2_TP/FP/TN/FN/AUC/F1/R/pred` directly. Tag: `[PRE-HYPERPARAMETER_OPTIMIZATION]`

- **Branch 2** (HPO improved — 2 archs: LSTM, VAE): The HPO model (`hpo_best_model`) is better. Re-evaluate on validation data and compute precision from own TP/FP. Tag: `[HYPERPARAMETER_OPTIMIZATION]`

- **Branch 3** (all HPO trials rejected): No HPO model exists. Use Section 1 baseline metrics; compute precision from `baseline_cm` TP/FP. Tag: `[PRE-HYPERPARAMETER_OPTIMIZATION]`

- **Safety net**: `section3_precision = section3_TP / (section3_TP + section3_FP)` recalculated after all branches to guarantee self-consistency.

#### Section 4 — Post-HPO Threshold Search
- `model_for_post_hpo` = the model elected in Section 3 (`threshold_opt_model` if HPO didn't improve, `hpo_best_model` if it did)
- Run a second threshold search (same label thresholds) using `retrain_model=False` (inference-only, no retraining per threshold)
- **Decision**: if `post_hpo_prec > section3_precision`: adopt `final_threshold = post_hpo_thresh`, `threshold_source = 'section4'`. Else: keep `final_threshold = optimal_threshold` (Section 1) and the elected model
- If post-HPO was not elected for S5, Section 4 overrides its own logged metrics to match whatever model S5 will actually use (prevents cross-contamination in the log)

#### Section 5 — Final Evaluation
- Use Section 4's elected model + `final_threshold`
- Evaluate on validation data for the final log line
- Evaluate on inference data (newest held-out date) for production predictions
- The same model that produced Section 4's metrics must be used here — silent model swaps between S4 and S5 are a functional failure

#### Ensemble Assembly (after Section 5)
- All architectures with `VAL_PRECISION ≥ ENSEMBLE_MIN_PRECISION` (0.53) are eligible for the ensemble
- Uniform averaging (each eligible architecture gets equal weight)
- Fallback: if no architecture meets 0.53, use the highest-precision arch alone

> **Note**: Actual value is 0.53 (GIS Tier 3). The code comment in chunk_18 was previously 0.40 but the config value was raised. Uniform weighting (not precision-weighted) was adopted in GIS Tier 1 to prevent any single architecture from dominating the vote.

### Normalization Scope

- `StandardScaler` is applied only to NN architectures (CNN, RNN, LSTM, Dense, VAE, Transformer)
- Gradient boosting models (CatBoost, LightGBM, XGBoost) are tree-based and scale-invariant — skip normalization
- Normalization must be applied at the **caller level** (chunk_18, chunk_12), NOT inside `train_model()`, to ensure both `model.fit()` and `model.predict()` receive consistent scaled data

### HPO Architecture-Specific Objectives

Each architecture uses a different objective function during Bayesian optimization, tuned to its prediction range and failure mode:

| Architecture | Objective Function | Rationale |
|--------------|-------------------|-----------|
| CatBoost, LightGBM, XGBoost | precision | Standard — well-calibrated trees |
| VAE | precision | Standard — falls through to else branch |
| Dense | precision * log(TP + 1) | Balances precision and TP count |
| CNN, RNN, LSTM, Transformer | precision * MaxPred | Push predictions toward 0.5 threshold |

Implementation details: Dense uses `balanced_score = precision * np.log(tp + 1 + 1e-6)`; CNN/RNN/LSTM/Transformer use `balanced_score = precision * max_pred` (RNN additionally rejects trials with TP < 100); tree archs and VAE use `balanced_score = precision`.

### Cross-Phase Variable Hygiene

#### Pattern 1: Dead Variables (Initialized but Never Updated)
Variables like `hpo_TP`, `hpo_FP`, `hpo_TN`, `hpo_FN` were initialized at the top of the architecture
loop but never recomputed from HPO predictions. Section 5 FINAL then read these zeros instead of actual
confusion matrix values. **Fix**: Always compute dependent variables immediately after their source.

#### Pattern 2: Stale Binary Array After Prediction Reassignment
When `val_pred` is reassigned (e.g., inside a fallback path), all derived arrays (`val_binary`,
`val_cm`, `val_precision`) must also be recomputed. Otherwise, metrics come from mismatched
data sources.

#### Pattern 3: Variable Leakage Between Architectures
Variables like `train_cm` that are conditionally assigned in one architecture but not another
retain the previous architecture's values. **Fix**: Always initialize loop-scoped variables before
the architecture loop, or explicitly set defaults in every code path.

#### Pattern 4: Feature Dimension Mismatch in Ensemble Evaluation
Models trained on pruned features (e.g., 19 of 24) crash when the ensemble passes full-feature
`X_val` to each model's `predict()`. **Fix**: Use pre-computed predictions (which already have
correct feature pruning) instead of re-predicting with raw data.

#### Pattern 5: Misaligned Parallel Arrays in Cross-Phase Analysis
When `val_dates` and `val_y_raw` are truncated for alignment, `val_predictions` (from a third
data source) must also be truncated to the same length.

#### Pattern 6: Normalization Inconsistency Between Training and Inference
When `StandardScaler` is fitted inside `train_model()`, the model learns on normalized data
but callers pass un-normalized data to `.predict()`. The same data must be normalized
consistently for both `model.fit()` and `model.predict()`.

**Fix**: Normalize at the **caller level** before both train and predict, not inside
`train_model()`. In chunk_18, normalize `X_train_opt`/`X_val_opt` immediately after
feature pruning. In chunk_12, normalize `X_train_t`/`X_val_t` immediately after
per-threshold feature selection. Only apply to neural network architectures (trees
are scale-invariant).

#### Pattern 7: HPO Metrics Variables Initialized But Never Populated
Variables `hpo_R`, `hpo_AUC`, `hpo_F1` initialized at the top of the architecture loop but never updated from HPO trial results. Section 5 FINAL then reads zero values for `val_auc` and `val_f1`. **Fix**: Populate immediately after HPO completes, or compute from best trial data before the Section 5 block.

#### Pattern 8: Conditional Path Silently Zeros Train Predictions
When `hpo_improved==True` AND `threshold_source=='section3'` (6 of 8 architectures), the `else` branch hardcodes `train_pred = np.zeros(n_train)` and assigns zero metrics instead of computing from the actual retrained model. Downstream code receives `TRAINING_PRECISION=0`, `TRAINING_RECALL=0`, etc. **Fix**: Always compute train predictions from the retrained model regardless of improvement status; the `else` branch should only exist for truly exceptional paths.

#### Pattern 9: Architecture Build Failure Silently Falls Back to Different Architecture
`try-catch` in `build_vae_model` (chunk_08_models_base.py) catches all exceptions (including `"A KerasTensor cannot be used as input to a TensorFlow function"`) and returns a Dense model. No warning is logged that the VAE architecture was not actually built. Downstream code stores results under `'VAE'` key but metrics are from a Dense model. **Fix**: Log a WARNING when fallback occurs; consider propagating the error or raising a custom exception for cleaner separation.

#### Pattern 10: Re-evaluating Unchanged Model Produces Different TP/FP
Section 3 Branch 1 re-evaluates the pre-HPO model on `X_val_opt` at the same `optimal_threshold` as Section 1, but gets different TP/FP counts (CATBOAST: 7344→11135 TP, 6483→10206 FP). Root cause is unknown (possibly data seeding, TF graph state, or feature pruning inconsistency). The fix eliminates the re-evaluation entirely: when the model and label threshold are identical to Section 1 (HPO didn't improve), copy all metrics from Section 1's `section2_*` variables instead of re-running `.predict()`.

**Fix**: At each phase-to-phase gate, the model propagation chain must be respected:
  Section 1 (threshold search) → Section 1 label_threshold_optimal → Section 2 HPO → Section 3 (carry-over if no improvement) → Section 4 (carry-over if no improvement) → Section 5.
Always use the elected model from the prior phase. Never re-evaluate a model that hasn't changed. If re-evaluation is unavoidable (model did change), compute precision from own TP/FP, never copy a stale precision from a different evaluation.

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

### Edge Cases

| Scenario | Handling |
|----------|----------|
| No architectures meet precision threshold | Use fallback (RNN) |
| All predictions negative | Log warning, output empty |
| HPO worse than baseline | Keep pre-HPO model |
| Precision = 0 | REJECT threshold, try next |
| Insufficient positive predictions | REJECT threshold, try next |

---

## 2.6 Architecture Specifications

### Model Inventory

| Model | Type | Purpose | Status | Logging Reference |
|-------|------|---------|--------|-------------------|
| LightGBM | Gradient Boosting | Fastest, imbalance-aware | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| XGBoost | Gradient Boosting | Battle-tested | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| CatBoost | Gradient Boosting | Categorical feature handling | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| Boosting_Adaptive | Gradient Boosting | Adaptive boosting | ⬜ Registered (not active in pipeline) | → Section 1.1.2, 1.1.3 |
| VAE | Neural Network | Variational autoencoder | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| Dense | Neural Network | Feed-forward baseline | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| CNN | Neural Network | Convolutional feature extraction | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| RNN | Neural Network | Sequential pattern recognition | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| LSTM | Neural Network | Long-term dependencies | ✅ Implemented | → Section 1.1.2, 1.1.3 |
| Transformer | Neural Network | Attention-based | ✅ Implemented | → Section 1.1.2, 1.1.3 |

### Architecture Groups

**Gradient Boosting** (LightGBM, XGBoost, CatBoost)
- Native class imbalance handling via tree parameters (`class_weight='balanced'`, `auto_class_weights`)
- Produces well-calibrated probabilities
- HPO: tree params (n_estimators, depth, learning_rate)
- Safeguards: SKLEARN_SAFEGUARDS

**Neural Networks** (CNN, RNN, LSTM, Dense, VAE, Transformer)
- Focal loss for imbalance (alpha, gamma parameters)
- Predictions clustered near zero (< 5% range)
- HPO: network params + loss params
- Safeguards: NEURAL_SAFEGUARDS

### Per-Architecture Parameter Impact

The tables below document which HPO parameters have the strongest effect on each architecture. HIGHEST = primary lever for improvement; HIGH = strong influence; MEDIUM = secondary tuning.

#### VAE

| Category | Parameter | Config Key | Range | Impact |
|----------|-----------|------------|-------|--------|
| Architecture | Latent Dim | latent_dim | [32, 64, 128, 256] | HIGH |
| | Dropout | dropout | [0.0, 0.02, 0.05, 0.1] | HIGH |
| | Encoder Layers | encoder_layers | [1, 2, 3] | MEDIUM |
| | Decoder Layers | decoder_layers | [1, 2, 3] | MEDIUM |
| Loss | **Loss Function** | loss_function | [bce, focal_loss] | HIGHEST |
| | Focal Alpha | alpha | [0.25, 0.5, 0.75, 1.0, 1.25] | HIGH |
| | Focal Gamma | gamma | [2.0, 2.5, 3.0] | HIGHEST |
| Optimizer | Learning Rate | learning_rate | [0.0005, 0.001, 0.002, 0.005] | HIGH |
| Training | Epochs | epochs | [30, 50, 80] | MEDIUM |

#### CNN

| Category | Parameter | Config Key | Range | Impact |
|----------|-----------|------------|-------|--------|
| Architecture | Filters | filters | [64, 128, 256, 512] | HIGH |
| | Kernel Size | kernel_size | [3, 5, 7, 11] | HIGH |
| | Dropout | dropout | [0.0, 0.05, 0.1, 0.2] | HIGH |
| | Conv Layers | layers | [1, 2, 3] | MEDIUM |
| | Pooling | pooling | [max, avg, none] | MEDIUM |
| Loss | **Loss Function** | loss_function | [bce, focal_loss] | HIGHEST |
| | Focal Alpha | alpha | [0.25, 0.5, 0.75, 1.0] | HIGH |
| | Focal Gamma | gamma | [2.0, 2.5, 3.0] | HIGHEST |
| Optimizer | Learning Rate | learning_rate | [0.0005, 0.001, 0.002, 0.005, 0.01] | HIGH |
| Training | Epochs | epochs | [30, 50, 80, 100] | HIGH |

#### RNN

| Category | Parameter | Config Key | Range | Impact |
|----------|-----------|------------|-------|--------|
| Architecture | Units | units | [64, 128, 256] | HIGH |
| | Dropout | dropout | [0.0, 0.05, 0.1] | HIGH |
| | RNN Layers | layers | [1, 2] | MEDIUM |
| Loss | **Loss Function** | loss_function | [bce, focal_loss] | HIGHEST |
| | Focal Alpha | alpha | [0.25, 0.5, 0.75, 1.0, 1.25] | HIGH |
| | Focal Gamma | gamma | [2.0, 2.5, 3.0, 3.5] | HIGHEST |
| Optimizer | Learning Rate | learning_rate | [0.0005, 0.001, 0.002, 0.005] | HIGH |
| Training | Epochs | epochs | [20, 30, 50] | HIGH |

#### LSTM

| Category | Parameter | Config Key | Range | Impact |
|----------|-----------|------------|-------|--------|
| Architecture | LSTM Units | lstm_units | [32, 64, 128, 256] | HIGH |
| | Dropout | dropout | [0.0, 0.05, 0.1, 0.2] | HIGH |
| | LSTM Layers | layers | [1, 2] | MEDIUM |
| | Bidirectional | bidirectional | [True, False] | MEDIUM |
| Loss | **Loss Function** | loss_function | [bce, focal_loss] | HIGHEST |
| | Focal Alpha | alpha | [0.25, 0.5, 0.75, 1.0] | HIGH |
| | Focal Gamma | gamma | [2.0, 2.5, 3.0] | HIGHEST |
| Optimizer | Learning Rate | learning_rate | [0.0005, 0.001, 0.002, 0.005] | HIGH |
| Training | Epochs | epochs | [20, 30, 50] | HIGH |

#### Transformer

| Category | Parameter | Config Key | Range | Impact |
|----------|-----------|------------|-------|--------|
| Architecture | Embedding Dim | dim | [64, 128, 256] | HIGH |
| | Attention Heads | heads | [2, 4, 8] | HIGH |
| | Feed-Forward Dim | ff_dim | [64, 128, 256] | MEDIUM |
| | Dropout | dropout | [0.0, 0.05, 0.1, 0.2] | HIGH |
| | Transformer Layers | layers | [1, 2, 4] | HIGH |
| Loss | **Loss Function** | loss_function | [bce, focal_loss] | HIGHEST |
| | Focal Alpha | alpha | [0.25, 0.5, 0.75, 1.0, 1.25] | HIGH |
| | Focal Gamma | gamma | [1.5, 2.0, 2.5, 3.0] | HIGHEST |
| Optimizer | Learning Rate | learning_rate | [0.00005, 0.0001, 0.0002] | HIGH |
| Training | Epochs | epochs | [20, 30, 50] | HIGH |

#### Dense

| Category | Parameter | Config Key | Range | Impact |
|----------|-----------|------------|-------|--------|
| Architecture | Layers | layers | [2, 3, 4] | HIGH |
| | Units | units | [64, 128, 256, 512, 1024] | HIGH |
| | Dropout | dropout | [0.1, 0.2, 0.3, 0.4] | HIGH |
| | Activation | activation | [relu, leaky_relu, selu] | MEDIUM |
| | Batch Size | batch_size | [32, 64, 128, 256] | MEDIUM |
| Loss | **Loss Function** | loss_function | [bce, focal_loss] | HIGHEST |
| | Focal Alpha | alpha | [0.25, 0.5, 0.75, 1.0, 1.25, 1.5] | HIGH |
| | Focal Gamma | gamma | [2.0, 2.5, 3.0, 4.0] | HIGHEST |
| Optimizer | Learning Rate | learning_rate | [0.0001, 0.0003, 0.0005, 0.001] | HIGH |
| Training | Epochs | epochs | [15, 20, 30, 40] | HIGH |

---

## 2.7 Configuration

### Core Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| DATA_PATH | for_train_x_2025_10_24_clean.csv | Input CSV file |
| USE_SAMPLING | True | Enable sampling for faster testing |
| SAMPLE_SIZE | 184408 | ~25 dates worth (~2.7% of dataset) |
| MIN_SAMPLES | 30 | Minimum samples required |
| TARGET_TYPE | continuous | Target type |
| LOG_TRANSFORM_TARGET | False | Apply log1p transform to target (disabled May 5, 2026 — use raw ChangeY values) |
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
| PREDICTION_THRESHOLD | 0.5 | Binary classification split threshold (kept at 0.5; not raised to 0.55 as initially planned in GIS Tier 1) |

### Ensemble Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ENSEMBLE_MIN_PRECISION | 0.53 | Minimum precision for ensemble (raised from 0.52 for GIS Tier 3 — tighter ensemble filter) |
| ENSEMBLE_WEIGHTING | uniform | Uniform averaging (changed from precision_weighted for GIS Tier 1 — prevents CatBoost dominance) |
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
| MIN_PRECISION_OVER_BASELINE | 0.02 | Precision must beat baseline by 2% (raised from 1% for GIS Tier 2) |
| MIN_POS_PRED_RATIO | 0.001 | Min 0.1% of predictions must be positive (raised from 0.01% for GIS Tier 2) |
| MAX_POS_PRED_RATIO | 0.60 | Max 60% of predictions can be positive (lowered from 70% for GIS Tier 2) |
| SKLEARN_SAFEGUARDS | dict | Arch-specific safeguard overrides for sklearn models (MIN_PRECISION_OVER_BASELINE=0.01, MIN_POSITIVE_PERCENTAGE=0.001, MIN_POSITIVE_ABSOLUTE=10) |
| NEURAL_SAFEGUARDS | dict | Arch-specific safeguard overrides for neural models (MIN_POSITIVE_PERCENTAGE=0, MIN_POSITIVE_ABSOLUTE=5, PATIENCE=10) |

### Validation Split

| Parameter | Value | Description |
|-----------|-------|-------------|
| VAL_SPLIT_PERCENTAGE | 0.30 | 30% validation split |
| TOP_DATES_HELD_OUT | 2 | Newest dates to hold out |

### Temporal Precision Gap Analysis (Phase Xb)

| Parameter | Value | Description |
|-----------|-------|-------------|
| TEMPORAL_GAP_N_DAYS | 3 | Number of recent dates in each tail (overrides TAIL_FRACTION if > 0) |
| TEMPORAL_GAP_TAIL_FRACTION | 0.33 | Tail fraction fallback when N_DAYS <= 0 |

### Model Persistence

| Parameter | Value | Description |
|-----------|-------|-------------|
| SAVE_TRAINED_MODELS | True | Save best models after training |
| MODELS_PATH | ./saved_models | Model output directory |
| FEATURE_ANALYSIS_REPORT_PATH | ./feature_importance_report.txt | Analysis output path |

### Logging & Verbosity

| Parameter | Value | Description |
|-----------|-------|-------------|
| LOG_VERBOSITY | 2 | Log level (0=quiet, 2=verbose) |
| VERBOSE_TENSORFLOW_LOGGING | False | TF internal log suppression |
| VERBOSE_PROCESSING_LOGGING | False | Chunk-level log suppression |

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

See [README.md §Prerequisites](./README.md#prerequisites) for system constraints.

---## 2.10 Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Extreme class imbalance | Model bias | class_weight, auto_class_weights, focal loss |
| Long training time | Cost overruns | GB models run first (faster) |
| Overfitting | Poor generalization | Temporal weights, validation split |
| Memory issues | Crashes | Chunked processing, memory utilities |
| HPO early termination | Missing optimal params | GB models excluded from filter |


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
| 3.23 | 2026-06-08 | **Config consolidation — zero-risk fallback elimination (Phase A + C) + Phase B (CONFIG_FEATURE_ANALYSIS merge)**. Converted 31 `config.get(KEY, fallback)`→`config[KEY]` where fallback == CONFIG value (dead code — key guaranteed by `validate_config_structure`). Removed 12 stale imports across 7 files. Phase A: added 6 CONFIG keys (`ABLATON_THRESHOLD`, `CORRELATION_THRESHOLDS`, `TREE_ESTIMATORS`, `PERMUTATION_REPEATS`, `SHAP_SAMPLE_SIZE`, `PATIENCE`), added 12 missing entries to REQUIRED_CONFIG_KEYS + CONFIG_TYPES (defense-in-depth). Phase B: deleted `CONFIG_FEATURE_ANALYSIS` standalone dict from `chunk_XX_feature_importance.py`, updated `__init__` to use passed config directly, updated `__main__` test with explicit config. Kept `PREDICTION_THRESHOLD_DEFAULT` (used by `chunk_04` function signatures). Files: chunk_01 (add keys + types + required), chunk_XX_feature_importance (remove dict + test), 7 consumer files. | Dead-code cleanup — zero behavioral change. All 88+ configurables now centralized in chunk_01_config.py (single source of truth). |
| 3.24 | 2026-06-09 | **Visual icon removal (✓/✗→PASS/FAIL) + log terminology cleanup.** chunk_18 LN926-LN934: `✓`→`PASS (Minimum Validation_Precision Required=0.53)`, `✗ (below threshold)`→`FAIL (Minimum Validation_Precision Required=0.53)`. LN925: `min precision`→`minimum validation_precision`. Consistent with chunk_20 PASS/FAIL format. Files: chunk_18 (2 edits). | Pure cosmetic — no functional change. Verified clean in iter8.log. |
| 3.25 | 2026-06-09 | **Config consolidation — Step 1: full registration.** Added 11 missing CONFIG_TYPES entries (`MIN_PRECISION_OVER_BASELINE`, `MIN_POS_PRED_RATIO`, `MAX_POS_PRED_RATIO`, `HPO_MIN_POSITIVE_PERCENTAGE`, `HPO_MIN_POSITIVE_ABSOLUTE`, `SKLEARN_SAFEGUARDS`, `NEURAL_SAFEGUARDS`, `ENABLE_POST_HPO_THRESHOLD_SEARCH`, `TOP_DATES_HELD_OUT`, `ENSEMBLE_VOTE_THRESHOLD`, `HYPERPARAM_SEARCH_SPACE`). Added 26 missing entries to REQUIRED_CONFIG_KEYS (the 11 above + `FORCE_SAMPLING`, `LOG_VERBOSITY`, `kernel_sizes`, `layers`, `heads`, `dim`, `cnn_filters`, `lstm_units`, `MIN_ENSEMBLE_SIZE`, `MAX_TRAINING_ATTEMPTS`, `VERBOSE_TENSORFLOW_LOGGING`, `VERBOSE_PROCESSING_LOGGING`, `USE_FOCAL_LOSS`, `FOCAL_LOSS_ALPHA`, `FOCAL_LOSS_GAMMA`). Removed duplicate `TEMPORAL_MULTIPLIER` in CONFIG_TYPES. CONFIG dict now has 89 keys, all registered as required + type-checked. Files: chunk_01 (only). Verified clean in iter9.log. | Defense-in-depth — every CONFIG key now validated. No behavioral change (all keys exist with correct types). Sets stage for Step 2: converting all remaining config.get(fallback) calls to direct config[] access. |
| 3.26 | 2026-06-09 | **Config consolidation — Step 2: all remaining config.get(KEY, fallback) → config[KEY].** Converted ~80 calls across 11 consumer files where fallback ≠ CONFIG value. All target keys guaranteed by validate_config_structure (registered in Step 1). Includes MATCH (zero-risk) conversions missed in Phase A-C. Import cleanup: removed 5 stale imports (DEFAULT_THRESHOLD_STEP ×4, DEFAULT_HPO_TRIALS ×1). Dead code removal: deleted 4 unused module-level constants from chunk_01_config (DEFAULT_FIRST_THRESHOLD, DEFAULT_LAST_THRESHOLD, DEFAULT_THRESHOLD_STEP, DEFAULT_HPO_TRIALS). Kept PREDICTION_THRESHOLD_DEFAULT (still used by chunk_04/chunk_21 function signatures). CONFIG dict: 89 keys, 0 remaining config.get(fallback) calls for pipeline config keys. Files: chunk_01, chunk_02, chunk_05, chunk_12, chunk_14, chunk_16, chunk_17, chunk_18, chunk_19, chunk_20, chunk_21, chunk_XX_feature_importance, chunk_XX_phase_feature_analysis_b. | Completes config consolidation. Single source of truth — all pipeline config reads go through direct config[KEY] access. Fail-fast on missing keys. Ready for pipeline re-run (iter10). |
| 3.27 | 2026-06-09 | **Best-model fallback for zero-models-above-threshold edge case.** iter10 crashed (`AssertionError: Missing final_metrics`) because ALL 9 architectures scored below `ENSEMBLE_MIN_PRECISION=0.53` (best CNN at 0.5276). Phase 4 model-saving loop at `chunk_18:1805` skipped every model — only metadata saved to `./saved_models/`. Phase 5 found no `.keras` files → `validate_pipeline_execution` raised AssertionError. Fix: track `best_fallback_model/arch/prec` during the skip loop; after loop, if `arch_names_to_save` is empty, save the best-performing model unconditionally. Metadata loop now iterates `arch_names_to_save` instead of re-filtering `arch_names`. Final log message conditional: shows model count or fallback warning. **Key finding**: iter10 crash was NOT a regression from config consolidation — pre-existing edge case confirmed by comparing iter9 (CNN passed at 0.5330) vs iter10 (CNN failed at 0.5276). All `0.53` threshold values existed before and after consolidation. | Fixes pre-existing edge case exposed in iter10: zero architectures meeting the ensemble precision threshold. Pipeline now degrades gracefully by saving the best available model. |
| 3.28 | 2026-06-09 | **GIS tier documentation restructured + config alignment + TOP_DATES_HELD_OUT code fix + expanded ranking table + temporal gap config.** Added §2.13 GIS Tier Reference. Fixed stale config values in §2.7. Removed dead config key `ENSEMBLE_VOTE_THRESHOLD`. Updated §2.12 to 5-trial cap. Wired `TOP_DATES_HELD_OUT` into Phase 1 and Phase 4. Expanded ARCHITECTURE RANKING to all 24 validation metrics. Added `TEMPORAL_GAP_N_DAYS`/`TEMPORAL_GAP_TAIL_FRACTION` config keys for Phase Xb temporal precision gap analysis (replaced hardcoded 0.67/0.33 split). Temporal Gap log now shows exact date ranges and specifies validation set. | GIS tier consolidation, config-spec sync, dead key removal, strategy doc alignment, TOP_DATES_HELD_OUT code fix, expanded ranking metrics, temporal gap config. |
| 3.29 | 2026-06-10 | **Removed broken auto-apply from METRICS REVIEW + standardized log formats across pipeline.** Deleted `[recommended actions]`, `[auto-apply]`, and `[ADDITIONAL AUTO-TUNE RULES]` from chunk_20:346-516 (~170 lines). Standardized Phase 5 `FINAL PREDICTION RESULTS` from pipe-delimited to `key=value` (chunk_19:400-416). Converted `architecture: X | inference_precision: Y` to `X inference_precision=Y` (chunk_19:307). Standardized chunk_12 threshold evaluation from mixed `|`/`:`/`=` to pure `key=value` (chunk_12:389-393). Converted `inference_precision: {val}` to `inference_precision={val}` in chunk_20 Final Results section (8 lines, both consensus and fallback paths). All `key: value` data pairs across pipeline now use consistent `key=value`. | GIS alignment + log format standardization. All data key-value pairs use `key=value` convention. |
| 3.30 | 2026-06-10 | **Dataset CSV preview after LN4 + feature importance log standardization + remaining arch log cleanup + column reorder.** Added header, newest-date first/last ticker, oldest-date first/last ticker lines after Dataset shape in `chunk_20_pipeline_main.py` (LN5-LN9). CSV columns reordered: `date` first, `Ticker_id` last. Standardized `chunk_XX_feature_importance.py` method headers (`Method 1/6: Name (...)`→`method=1 ...`), consolidated ranking (`#N Feature=val | ...`→`rank=N Feature=val ...`), consolidated pruning, cross-threshold summary (Python list→comma-separated `key=val`). Standardized `chunk_18_phase_4_ensemble.py` (`[arch] loss: val | pred_threshold: val`→`arch loss=val pred_threshold=val`). | Completes log format standardization — all pipeline data key=value pairs use consistent space-separated format. Dataset preview aids quick human inspection. |
| 3.31 | 2026-06-10 | **Feature stability & permutation importance: removed top-N limits; diagnostic format overhaul; timing bug fix; report label cleanup; config dead-key removal.** `chunk_18`: Feature Stability and Permutation Importance now list ALL features (removed `[:5]`/`[:10]` slices). `chunk_04_utils_metrics.py`: `percentiles: p1=...` → `cumulative binary_split_predictions (1%) ≤ ...`, removed `|` before histogram, replaced 20-bin histogram with round-threshold `binary_split_predictions distribution: 48% ≤ 0.01, ...`. `chunk_XX_feature_importance.py`: fixed bug where individual method timings were always 0.0s (missing `results['correlation_timing']` → `self.timings['correlation']` copy); collapsed 6 timing report lines into 1 `key=value` line; `RF+GBM` → `Random Forest + Gradient Boosting`; removed blank line after header; removed `nsmallest(5)` → shows all features; renamed `TOP FEATURES PER METHOD` → `FEATURE IMPORTANCE RANKING PER METHOD`. `chunk_12`: `[reject] Skipping threshold ... (min=5)` → `[reject] threshold=... action=skipped reason=insufficient_positive_predictions positive_predictions=... minimum_positive_required=...`. **Config Step 1**: removed deprecated `MIN_POSITIVE_PREDICTIONS` (static 1000) — replaced with dynamic safeguard read in chunk_18 HPO/final threshold search. **Step 2**: removed top-level `MIN_POSITIVE_PERCENTAGE`/`MIN_POSITIVE_ABSOLUTE` duplicates (already in SKLEARN_SAFEGUARDS/NEURAL_SAFEGUARDS) — removed dead `config.get(fallback)` patterns in evaluator. **Step 3**: removed `HPO_MIN_POSITIVE_PERCENTAGE`/`HPO_MIN_POSITIVE_ABSOLUTE` dead keys (defined but never read) — deleted from config dict, REQUIRED_CONFIG_KEYS, CONFIG_TYPES. **Step 4**: fixed stale safeguard comments referencing removed values. | Config deduplication (4 keys removed) + timing bug fix + remaining diagnostic standardization. All safeguard min-positive values now read from arch-specific dicts. Changelog entry consolidation across live session edits. |

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

### Non-Functional Requirements

| Category | Requirement | Target |
|----------|-------------|--------|
| **Performance** | Pipeline runtime | <2 hours |
| **Performance** | Phase 4 training | ~27000s |
| **Scalability** | Sample size support | ≥250K |
| **Reliability** | Error handling | Graceful degradation |
| **Reproducibility** | Random seeds | Fixed where possible |
| **Maintainability** | Configuration-driven | No hardcoded params |

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
*Last updated: 2026-06-10*  
*Version: 3.31*


## PROJECT_LEXICON

Complete reference of all cosmetic log labels, section tags, metric keys, and abbreviations used across the pipeline. Labels follow lowercase_snake_case convention unless noted. Exceptions that stay UPPER_SNAKE_CASE: VALIDATION_PRECISION, VALIDATION_TRUE_POSITIVES, VALIDATION_TRUE_NEGATIVES, LABEL_THRESHOLD, [HYPERPARAMETER_OPTIMIZATION_SEARCH], TRIAL, OPTIMAL. Title-case metrics unchanged: Brier, Kappa, Informedness, Markedness, Gini.

### G1: Section / Status Tags

**Log Section Tags** (bracketed identifiers in output)

| Tag | Description | Source File(s) |
|-----|-------------|---------------|
| `[section 1] [baseline]` | Baseline threshold search | chunk_18 |
| `[section 2] [HYPERPARAMETER_OPTIMIZATION_SEARCH]` | Pre-HPO threshold search | chunk_18 |
| `[section 3] [HYPERPARAMETER_OPTIMIZATION]` / `[section 3] [PRE-HYPERPARAMETER_OPTIMIZATION]` | Pre-HPO evaluation (HPO improved / did not improve) | chunk_18 |
| `[section 4] [post hyperparameter_optimization]` | Post-HPO evaluation | chunk_18 |
| `[section 5] [final]` | Final training summary | chunk_18 |
| `[baseline]` | Baseline model evaluation | chunk_18 |
| `[label_threshold_optimal]` | Optimal threshold found | chunk_18 |
| `[final]` | Final training step | chunk_18 |
| `[post hyperparameter_optimization]` | Post-HPO results | chunk_18 |
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

| `[skipped]` | Skipped iteration | chunk_20 |
| `[timing]` | Timing measurement | chunk_20 |
| `[error]` | Error status | chunk_20 |
| `[warning]` | Warning status | chunk_20 |
| `[ok]` | Status passed | chunk_12, chunk_18 |
| `[reject]` | Threshold rejected | chunk_12, chunk_18 |
| `[phase_1_5]` | Phase 1.5 diagnostics | chunk_02, chunk_04 |

**Inline Status / Warning Labels**

| Label | Context | File(s) |
|-------|---------|---------|
| `sanity_check:` | Data validation warnings | chunk_16 |
| `feature_quality_analysis:` | Feature statistics header | chunk_02 |
| `temporal_coverage:` | Date coverage header | chunk_02 |

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
| `cumulative binary_split_predictions (N%) ≤ X.XX` | Cumulative density below threshold (replaced `percentiles`) | chunk_04 |
| `binary_split_predictions distribution: N% ≤ X.XX, ...` | Round-threshold density distribution (replaced 20-bin histogram) | chunk_04 |

### G6: Report / Table Headers

| Header | Description | File(s) |
|--------|-------------|---------|
| `feature_importance_analysis_report` | Feature importance analysis header | chunk_XX |
| `consolidated_ranking (rank=N Feature=val ...)` | Consolidated feature ranking (space-separated key=value) | chunk_XX |
| `feature_importance_ranking_per_method` | Per-method feature importance header (lists all features) | chunk_XX |
| `spearman` | Spearman correlation method name | chunk_XX |
| `tree_importance (Random Forest + Gradient Boosting)` | Tree-based importance method | chunk_XX |
| `permutation_importance` | Permutation importance method | chunk_XX |
| `neural_weight_magnitude` | Neural weight importance method | chunk_XX |
| `shap_values` | SHAP importance method | chunk_XX |
| `ablation_study (auc)` | Ablation study method | chunk_XX |
| `method_runtime:` | Consolidated single-line timing (`method_runtime_total=X.Xs`) | chunk_XX |
| `architecture ranking (by validation precision)` | Architecture ranking header | chunk_18 |
| `hyperparameter_optimization impact summary` | HPO impact header | chunk_18 |
| `training time summary` | Training time header | chunk_18 |
| `final prediction results (sorted by precision)` | Final results header | chunk_19 |
| `features:` / `samples:` | Feature importance input dimensions | chunk_XX |
| `cross_threshold (always_pruned=..., never_pruned=..., borderline=...)` | Cross-threshold feature summary (space-separated key=value) | chunk_XX |
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

