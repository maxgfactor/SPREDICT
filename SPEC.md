# Software Specification Requirements (SSR) — Dataset Classification Ensemble

**Version**: 3.51  
**Date**: 2026-06-17  
**Status**: Living Document - Update After Each Run  

---

# QUICK START GUIDE

## What Is This Document?
SPEC.md is a living document for the Dataset Classification Ensemble Pipeline. It defines logging formats and metrics (Section 1), documents the code and architecture (Section 2), maintains history (Section 3), and tracks permanent failures (Section 4). GIS strategy details are in [GIS.md](./GIS.md). Run results are archived in shortmemory.txt.

## Document Structure

| Section | Type | Purpose |
|---------|------|---------|
| Section 1 | Static format definitions + metrics | Log format, output schemas, metric formulas |
| Section 2 | Static (update only if code changes) | Functional specs, configurations |
| Section 3 | Dynamic (update after changes) | Version history, file inventory, NFRs |
| Section 4 | STATIC (NEVER changes) | Failed approaches with evidence |

## How to Use

1. **RUN CODE**: Execute pipeline, generate pipeline_cpu.log
2. **REVIEW RESULTS**: Check pipeline_cpu.log for metrics output
3. **ARCHIVE**: Copy run results to shortmemory.txt for future reference
4. **CHECK**: Compare actual vs target (e.g., Precision ≥ 0.60 at per-architecture calibrated threshold — see GIS.md §1)
5. **UPDATE**: If code changed, update Sections 2-3
6. **RECORD FAILURES**: If strategy failed, add to Section 4 with .log evidence

> **Mission Target**: Each architecture targets per-architecture calibrated precision ≥ 0.60 at maximal recall. See [GIS.md §1](./GIS.md#1-overview).

## Quick Lookups

| Need | Section |
|------|---------|
| GIS strategy details | [GIS.md](./GIS.md) |
| Log format reference | Section 1.1 |
| Metrics definitions | Section 1.3 |
| Code for architecture | Section 2.6, 3.4 |
| What failed before | Section 4.x |

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [SECTION 1: Logging, Reporting, and Metrics](#section-1-logging-reporting-and-metrics) | Log formats, output schemas, metric definitions |
| [SECTION 2: Functionality](#section-2-functionality) | Static specs, architecture, configuration |
| [SECTION 3: Documentation](#section-3-documentation) | Version history, file inventory, NFRs |
| [SECTION 4: Failed Strategies & Approaches](#section-4-failed-strategies--approaches) | Permanent record of failed attempts |

---

# SECTION 1: Logging, Reporting, and Metrics

This section documents the logging format, output file schemas, and metrics definitions. Run results are archived in shortmemory.txt.

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

| Metric | Formula | Target | Source File |
|--------|---------|--------|-------------|
| **Precision** | TP / (TP + FP) | ≥ 0.60* | chunk_04_utils_metrics.py |

\* At per-architecture calibrated prediction threshold — see [GIS.md §1](./GIS.md#1-overview).
| **Recall** | TP / (TP + FN) | Maximize | chunk_04_utils_metrics.py |
| **F1 Score** | 2 × (P × R) / (P + R) | Maximize | chunk_04_utils_metrics.py |
| **AUC** | Area under ROC curve | ≥ 0.70 | chunk_04_utils_metrics.py |
| **PR-AUC** | Area under Precision-Recall curve | Maximize | chunk_04_utils_metrics.py |
| **MCC** | Matthews Correlation Coefficient | -1 to 1 | chunk_04_utils_metrics.py |

### Confusion Matrix Components

| Component | Description | Source File |
|-----------|-------------|-------------|
| TP | True Positives - Correctly predicted signal | chunk_04_utils_metrics.py |
| FP | False Positives - Non-signal predicted as signal | chunk_04_utils_metrics.py |
| TN | True Negatives - Correctly predicted normal | chunk_04_utils_metrics.py |
| FN | False Negatives - Signal predicted as non-signal | chunk_04_utils_metrics.py |

### Prediction Statistics

| Statistic | Description | Source File |
|-----------|-------------|-------------|
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
| AC-11 | Inference precision ≥ 0.60 OR fallback used | Check metrics |
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

## 1.6 Metrics Pipeline Flow

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
| **Training** (Phase 4) | Input | `X_train, y_train_raw` | N_train samples (varies by run) |
| | Label Transform | `y_train_optimal = (y_train_raw >= 2.0).astype(int)` | Convert continuous to binary |
| | Train | `model.fit(X_train, y_train_binary)` | Train model |
| | Predict | `train_pred = model.predict(X_train)` | Raw probabilities |
| | Binary | `train_binary = (train_pred >= 0.5).astype(int)` | Binary predictions |
| | Metrics | `calculate_precision(y_train_optimal, train_binary)` | **Train metrics** |
| **Validation** (Phase 4) | Input | `X_val, y_val_raw` | N_val samples (varies by run) |
| | Label Transform | `y_val_optimal = (y_val_raw >= 2.0).astype(int)` | Same threshold as train |
| | Predict | `val_pred = model.predict(X_val)` | Raw probabilities |
| | Binary | `val_binary = (val_pred >= 0.5).astype(int)` | Binary predictions |
| | Metrics | `calculate_precision(y_val_optimal, val_binary)` | **Val metrics** |
| | Usage | Selected for ensemble if val precision >= 0.53 | Architecture selection |
| **Inference** (Phase 5) | Input | `X_inference, y_raw` | N_inference samples (varies by run) |
| | Label Transform | `y_binary = (y_raw >= 2.0).astype(int)` | Same threshold as training |
| | Predict | `inf_pred = model.predict(X_inference)` | Raw probabilities |
| | Binary | `inf_binary = (inf_pred >= 0.5).astype(int)` | Binary predictions |
| | Metrics | `calculate_metrics(y_binary, inf_binary, inf_pred)` | **Inference metrics** |
| | Usage | Final signal predictions output | Ranked by inference precision |

### Key Points

| Aspect | Detail |
|--------|--------|
| **Label Threshold** | Same across all stages (varies by architecture, 0.0–20.0) |
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
| Phase Xb | Temporal validation precision gap analysis | Recent vs older validation precision gap | → Precision analysis |
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
- HPO trials that fail MaxPred, TP, or minimum validation precision gates are rejected silently; the surviving best trial is the "HPO best"

#### Section 3 — HPO Election Gate
Compare HPO validation precision vs pre-HPO validation precision at prediction binary split 0.5:

- **Branch 1** (HPO did NOT improve — 7 archs: CatBoost through Transformer): The pre-HPO model (`threshold_opt_model`) is the best. Model and threshold are identical to Section 1. **No re-evaluation** — copy all metrics from Section 1's `section2_TP/FP/TN/FN/AUC/F1/R/pred` directly. Tag: `[PRE-HYPERPARAMETER_OPTIMIZATION]`

- **Branch 2** (HPO improved — 2 archs: LSTM, VAE): The HPO model (`hpo_best_model`) is better. Re-evaluate on validation data and compute validation precision from own TP/FP. Tag: `[HYPERPARAMETER_OPTIMIZATION]`

- **Branch 3** (all HPO trials rejected): No HPO model exists. Use Section 1 baseline metrics; compute validation precision from `baseline_cm` TP/FP. Tag: `[PRE-HYPERPARAMETER_OPTIMIZATION]`

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
- Fallback: if no architecture meets 0.53, use the highest-validation-precision arch alone

> See GIS.md §2 (Stage 3: Filter) for the tier rationale behind these ensemble values.

### Normalization Scope

- `StandardScaler` is applied only to architectures in `NEURAL_ARCHITECTURES` (config key)
- Architectures in `TREE_ARCHITECTURES` are tree-based and scale-invariant — skip normalization
- Normalization must be applied at the **caller level** (chunk_18, chunk_12), NOT inside `train_model()`, to ensure both `model.fit()` and `model.predict()` receive consistent scaled data

### HPO Architecture-Specific Objectives

Each architecture uses a different objective function during Bayesian optimization, tuned to its prediction range and failure mode:

| Architecture | Objective Function | Rationale |
|--------------|-------------------|-----------|
| CatBoost, LightGBM, XGBoost | precision | Standard — well-calibrated trees |
| VAE | precision | Standard — falls through to else branch |
| Dense | precision * log(TP + 1) | Balances precision and TP count |
| Archs in `MAXPRED_OBJECTIVE_ARCHS` | precision * MaxPred | Push predictions toward 0.5 threshold |

Implementation details: Dense uses `balanced_score = precision * np.log(tp + 1 + 1e-6)`; archs in `MAXPRED_OBJECTIVE_ARCHS` use `balanced_score = precision * max_pred` (RNN additionally rejects trials with TP < 100); tree archs and VAE use `balanced_score = precision`.

### Architecture Classification Groups

All architecture grouping is centralized in `chunk_01_config.py` as CONFIG keys:

| Key | Value | Purpose |
|-----|-------|---------|
| `NEURAL_ARCHITECTURES` | `['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']` | StandardScaler gating, neural safeguard lookup |
| `TREE_ARCHITECTURES` | `['CatBoost', 'LightGBM', 'XGBoost']` | Sklearn safeguard lookup |
| `MAXPRED_OBJECTIVE_ARCHS` | `['CNN', 'RNN', 'LSTM', 'Transformer']` | HPO objective: precision * MaxPred |
| `ARCH_CSV_ORDER` | All 9 archs in display order | metrics_summary.csv row ordering |
| `HPO_RETRAIN_EPOCHS` | `{'Dense':15, 'VAE':15, 'CNN':15}` | Fast HPO threshold retrain epochs (fallback=3) |
| `FINAL_TRAIN_EPOCHS` | `{'Dense':15, 'VAE':30, 'CNN':20, 'LSTM':20, 'Transformer':20}` | Final training epochs (fallback=3) |

Adding a new architecture to a group = 1 config change, not N scattered list updates. All 6 keys registered in `CONFIG_KEYS` and `CONFIG_TYPES`.

### Cross-Phase Variable Hygiene

Ten bug patterns were identified and fixed during development (Patterns 1–10, archived in shortmemory.txt). Key principles:

- **Compute dependent variables immediately after source**: Never initialize loop vars at the top of a block if they'll be overwritten downstream (Patterns 1, 7).
- **Re-evaluate after reassignment**: When `val_pred` is reassigned, all derived arrays (`val_binary`, `val_cm`, `val_precision`) must be recomputed (Pattern 2).
- **Initialize loop-scoped variables explicitly**: Conditional assignment in one arch iteration leaks values to the next (Pattern 3).
- **Use pre-computed predictions for ensemble**: Models trained on pruned features crash on full-feature data (Pattern 4).
- **Align parallel arrays**: When `val_dates` is truncated, `val_predictions` must match (Pattern 5).
- **Normalize at caller level**: Both `model.fit()` and `model.predict()` must receive consistently scaled data (Pattern 6).
- **Compute train predictions from retrained model**: Don't hardcode zeros in fallback branches (Pattern 8).
- **Log architecture build failures**: Don't silently substitute Dense for VAE (Pattern 9).
- **Never re-evaluate an unchanged model**: Copy S1 metrics directly instead of re-running `.predict()` (Pattern 10).

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
| FR-11 | Evaluate model against validation precision threshold | Required | chunk_12_evaluation_evaluator.py | → See Section 1.3 |
| FR-12 | Apply log transform (Option C: sign * log1p(|y|)) to handle extreme target values | Improvement (Step 1) | chunk_05_data_manager.py, chunk_04_utils_metrics.py | → See Section 2.7 |

### Edge Cases

| Scenario | Handling |
|----------|----------|
| No architectures meet validation precision threshold | Use fallback (RNN) |
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
| TARGET_COLUMN_INDEX | -1 | Auto-detect target column |
| TEMPORAL_MULTIPLIER | 9.0 | Temporal weighting multiplier |
| INPUT_DIM | Auto-detected | Number of features |

### Threshold Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| FIRST_THRESHOLD | 20.0 | Starting label threshold |
| LAST_THRESHOLD | 0.0 | Ending label threshold |
| THRESHOLD_STEP | -10.0 | Label threshold increment (reduced to 3 steps: 20→10→0) |
| PREDICTION_THRESHOLD | 0.5 | Binary classification split threshold (see GIS.md) |

### Ensemble Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ENSEMBLE_MIN_PRECISION | 0.53 | Minimum validation precision for ensemble (see GIS.md §2 Stage 3) |
| ENSEMBLE_WEIGHTING | uniform | Uniform averaging (see GIS.md §2 Stage 3) |
| FALLBACK_ARCHITECTURE | VAE | Highest val precision fallback (see GIS.md §2 Stage 3) |

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
| WINSORIZE_PERCENTILE_LOW | 2 | Lower percentile for winsorization (see GIS.md §2 Stage 5) |
| WINSORIZE_PERCENTILE_HIGH | 95 | Upper percentile for winsorization (see GIS.md §2 Stage 5) |
| MIN_PRECISION_OVER_BASELINE | 0.02 | Precision must beat baseline by 2% (see GIS.md §2 Stage 5) |
| MIN_POS_PRED_RATIO | 0.001 | Min 0.1% of predictions must be positive (see GIS.md §2 Stage 5) |
| MAX_POS_PRED_RATIO | 0.60 | Max 60% of predictions can be positive (see GIS.md §2 Stage 5) |
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

---

## 2.8 Hyperparameter Search Spaces

Each architecture's HPO search space is defined in `chunk_01_config.py`. Runtime-impact tags, footnotes, and the HPO control parameter table are maintained in **GIS.md §7**.

---

## 2.9 GIS (Global Iteration Strategy)

GIS is an iterative optimization framework that tunes pipeline configuration parameters across successive runs, driving each architecture toward a per-architecture calibrated precision threshold achieving P ≥ 0.60 — see [GIS.md §1](./GIS.md#1-overview).

```
Run pipeline → Evaluate pipeline_cpu.log → Analyze gaps → Adjust config → Re-run
```

→ For full detail (iteration log, precision lever dimensions, search spaces, execution plan): see [GIS.md](./GIS.md)

---

## 2.10 System Constraints

See [README.md §Prerequisites](./README.md#prerequisites) for system constraints.

---

## 2.11 Risks and Mitigations

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
| 1.0–2.3 | 2026-04-15 | Initial SSR + reorganization into 3 sections (Logging, Functionality, Documentation) + bidirectional cross-references | Initial spec, improved structure |
| 3.0 | 2026-04-15 | Living document conversion — added actual value templates for post-run updates | Dynamic document for pipeline runs |
| 3.1–3.3 | 2026-05-11–13 | Precision target 0.60, GIS hyperparameter reconfiguration (all 9 search spaces expanded), HPO thresholds raised, Phase 5 crash fix (A-D) | Mission focus, fix search space stagnation |
| 3.4–3.5 | 2026-05-18–19 | Phase 5 fixes, logging standardization (tag reorder, terminology), feature importance overhaul, per-threshold feature pruning | GIS Iter 1 improvements |
| 3.7–3.10 | 2026-05-25 | Threshold search cleanup (11→3 steps), LightGBM scale_pos_weight→class_weight, FocalLoss on neural archs, faster pipeline (HPO 30→5 trials) | Redundant log removal, pipeline speed |
| 3.11–3.12 | 2026-05-31 | VAE serialization fix, GIS Precision Lever Plan audit (14 changes across 3 tiers) | Keras 3 compatibility, code audit |
| 3.13–3.14 | 2026-06-01 | Tier 1–2 config applied (winsorization, safeguard gates), S4→S5 model carry-forward bug discovered | Precision floor improvements, bug discovery |
| 3.16–3.17 | 2026-06-01 | Tier 3 FocalLoss alpha expansion, Tier 7 temporal sample weights added to training | FocalLoss tuning, recency focus |
| 3.18–3.19 | 2026-06-03–04 | Cross-phase model propagation fix (3 code bugs), consensus voting fix (Bug I/J), min_votes 6→5 | Model carry-forward correctness |
| 3.20–3.27 | 2026-06-08–09 | Metric log standardization (31 lines), 2 functional error fixes, log cosmetics, config consolidation (Steps 1–2, 89 keys), best-model fallback edge case | Format consistency, config single-source-of-truth |
| 3.28–3.31 | 2026-06-09–10 | GIS tier docs restructured, config alignment, auto-apply removal, log format standardization, dataset preview, feature stability/diagnostic overhaul, timing bug fix, config dead-key removal (4 keys) | Spec-code sync, cleanup |
| 3.52 | 2026-06-20 | Mission re-oriented: per-architecture calibrated threshold targeting P ≥ 0.60 at max recall. XGBoost freeze deprecated. CNN degeneracy investigation queued. | Mission scope updated |
| 3.32 | 2026-06-12 | SPEC restructured: stale run results archived to shortmemory, version history consolidated (40→12 entries), HPO search-space evolution comments removed, cross-phase hygiene patterns summarized (details archived), PROJECT_LEXICON G2–G10 moved to shortmemory, empty template tables deleted | SPEC document focus — remove stale/dynamic/duplicative content |
| 3.33 | 2026-06-12 | GIS info extracted to standalone GIS.md (428 lines). SPEC retains only GIS overview (§2.9) and cross-references. Inline GIS tier evolution notes stripped from config descriptions. §4.7 hyperparameter reconfiguration detail moved to GIS.md §6. | GIS strategy documentation — standalone reference, cleaner SPEC |
| 3.34 | 2026-06-12 | §2.8 stale alpha/loss_function values synced across all 9 search spaces, runtime tags added to all params, footnotes added (FOCAL_LOSS_CONFIG override, XGBoost scale_pos_weight removal) | Spec-code alignment after GIS extraction |
| 3.35 | 2026-06-12 | §2.8 search spaces + HPO table + footnotes moved to GIS.md §7; §2.9 GIS overview collapsed to brief summary with cross-reference | SPEC → informational; GIS → actionable tracking |
| 3.36 | 2026-06-15 | GIS Iter 1 (full 9-arch run after HPO bug fixes): 5/9 archs pass ensemble filter (≥0.53 val_p), best val_p=0.5473 (Transformer), 5/5 loaded archs hit inf P ≥ 0.60, ensemble P=0.6525/R=0.020. HPO dimension bugs 2–8 validated fixed. GIS.md Key Results updated, Iteration Log + Execution Plan revised. shortmemory.txt appended with iter1 results. | Full production run after Jun 14 bug fixes; entering Phase A zero-runtime lever sweep |
| 3.37 | 2026-06-15 | GIS Iter 2 (permutation-only FI): Phase Xa 844s→169s (−80%), total time 34,726s→13,354s (−61.5%). 3/9 archs pass ensemble filter. 0/9 meet val_p≥0.60. CNN best val_p=0.5369, CNN best inf P=0.6755, ensemble P=0.6816/R=0.2452. Iter naming restarted: old runs tagged (legacy), iter18→iter1, iter19→iter2. | Permutation-only config applied; entering Phase A zero-runtime lever sweep (A1: WINSORIZE_PERCENTILE_LOW 2→3) |
| 3.38 | 2026-06-15 | Iter 2 results **invalidated**: permutation-only FI caused broken feature pruning (quick dense NN → baseline_auc≈0.50 → all permutation importances zero → flat ranking → positional slicing). Feature set was determined by column index, not importance. Dense/LSTM collapses are artifacts. Reverted FEATURE_IMPORTANCE_METHODS to all 6. Iter 2 repurposed as A1: WINSORIZE_PERCENTILE_LOW 2→3. | Bug discovered; permutation-only FI removed; A1 sweep replaces invalidated run |
| 3.39 | 2026-06-15 | GIS Iter 2 (A1: WINSORIZE_PERCENTILE_LOW 2→3): Total 14,169s (−59% vs iter1, primarily CNN time fix). Best val P 0.5476 (VAE, flat vs iter1 0.5473). Ensemble val P 0.5341→0.5561 (+4.1%). 5/9 pass filter (roster shuffled: RNN/LightGBM in, CatBoost/CNN out). Ensemble inf P 0.7552 / R 0.0140 (conservative consensus). Best individual inf P RNN 0.7078. Temporal drift −66.7%. Feature pruning set changed: RSI_14 replaced Perf_YTD. | A1 applied; minimal val P impact; considering A2 or A3 |
| 3.40 | 2026-06-15 | GIS Iter 3 (A3+A4 bundle): HIGHLY_SKEWED_FEATURES [0,1,4,5]→[] (disable log1p transforms). MIN_POS_PRED_RATIO 0.001→0.0005, MAX_POS_PRED_RATIO 0.60→0.65 (gate relaxation). A1 stays (WINSORIZE_LOW=3). **A3 collapsed RNN/Transformer/LightGBM** (RNN 0.5357→0.0581, Transformer 0.5339→0.0617). A4 helped Dense (0.5322→0.5441) and CNN (0.5289→0.5403). Best val P 0.5449 (regression). Ensemble 3/9 pass. Dense shows +0.2273 temporal gap (first positive gap). | A3 too aggressive — reverting. A4 beneficial — keeping. |
| 3.41 | 2026-06-15 | Reverted A3: HIGHLY_SKEWED_FEATURES restored to [0,1,4,5] (log1p transforms re-enabled). A1 (WINSORIZE_LOW=3) and A4 (MIN_POS_PRED_RATIO=0.0005, MAX_POS_PRED_RATIO=0.65) kept. Iter 4 pending. | A3 reverted; A4 retained; iter4 planned |
| 3.42 | 2026-06-15 | GIS Iter 4 (A3 reverted, A4+A1 kept): Total 12,826s. Best val P 0.5443 (Transformer). 5/9 pass filter (Transformer, VAE, Dense, LSTM, LightGBM). LSTM entered ensemble (0.5365). Transformer fully recovered (0.0617→0.5443). RNN still collapsed (0.0615). CNN dropped out (0.5403→0.5252). Dense inf P 0.6810 (best). Ensemble inf P 0.6511 / R 0.0832. 0 degenerate members. 0/9 positive temporal gap (Dense lost iter3's +0.2273 — was A3 artifact). | Val P ceiling ~0.544 confirmed. Phase A nearly exhausted. |
| 3.43 | 2026-06-15 | GIS Iter 5 (A2: WINSORIZE_PERCENTILE_HIGH 95→97): Total 17,820s (+39%). Best val P RNN 0.5454 (fully recovered from 0.0615). Dense collapsed to 0.1931, LSTM collapsed to 0.0648. Only 2/9 pass filter (RNN, VAE). Ensemble inf P 0.6495 / R 0.7310 (best recall ever). 0/9 positive temporal gap. Phase A definitively exhausted. | A2 highly arch-specific. Phase A complete. → Phase B: per-arch winsorization. |
| 3.44 | 2026-06-15 | Phase B planned: per-architecture winsorization. Move from Phase 1 global to Phase 4 per-arch. RNN HIGH=97, Dense/LSTM HIGH=92, VAE HIGH=97, rest 95. Pipeline code change required (chunk_16 + chunk_18). | Per-arch winsorization planned for iter6 |
| 3.45 | 2026-06-16 | **GIS Iter 6 (Phase B: per-architecture winsorization)**: Total 13,715s. 5 files modified (chunk_01, chunk_05, chunk_16, chunk_18, chunk_19). Pipeline validated end-to-end. **Best val P Transformer 0.5505** (new record, first above 0.55). 3/9 pass filter (LSTM 0.5305, VAE 0.5449, Transformer 0.5505). LSTM recovered from collapse (iter5 0.0648→0.5305). Dense partially recovered (0.1931→0.5203, still below 0.53). Best inf P VAE 0.6541. Ensemble inf P 0.5417 / R 0.0203 (288 consensus rows). Phase B validated — per-arch winsorization was key to breaking 0.55 ceiling. | Phase B complete; evaluate Dense pathology for Phase C |
| 3.46 | 2026-06-17 | **Iter7 code fixes (§8e)**: Fix 1 — XGBoost `scale_pos_weight` HPO control: HPO-provided value now takes priority over `DYNAMIC_CLASS_WEIGHTS`. Added `[1,10,50,100,259]` to XGBoost HPO space. Fix 2 — Neural safeguard relaxation: `MIN_PRECISION_OVER_BASELINE` reduced from 0.02→0.01 for all 6 NN archs (matching SKLEARN_SAFEGUARDS). Fixes RNN threshold mismatch (Section 1 rejected LT=0.0 by 0.0002). RNN now passes LT=0.0 baseline check, HPO runs at correct threshold. | Run iter7 to validate both fixes |
| 3.47 | 2026-06-17 | **Architecture list centralization**: 6 new CONFIG keys (`NEURAL_ARCHITECTURES`, `TREE_ARCHITECTURES`, `MAXPRED_OBJECTIVE_ARCHS`, `ARCH_CSV_ORDER`, `HPO_RETRAIN_EPOCHS`, `FINAL_TRAIN_EPOCHS`). 23 hardcoded architecture lists across 5 files relocated to `chunk_01_config.py`. Replaced lists in `chunk_12_evaluation_evaluator.py` (5 refs), `chunk_18_phase_4_ensemble.py` (8 refs), `chunk_19_phase_5_optimization.py` (1 ref), `chunk_20_pipeline_main.py` (1 ref), `chunk_21_hyperparam_optimizer.py` (3 refs). `ARCH_CSV_ORDER` expanded from 6→9 archs (trees were missing from CSV output). | Single-source-of-truth for architecture groups; zero hardcoded arch lists remain in logic gates |
| 3.48 | 2026-06-17 | **GIS Iter 7 (Fix 1+2 + arch centralization + CNN/XGBoost root cause fixes)**: Total 14,773s, 9/9 archs trained, no crashes. Fix 1 (XGBoost scale_pos_weight HPO) partially successful — HPO trial used weight=100, val P 0.0867→0.0924, root cause deeper (early stopping). Fix 2 (neural safeguard) fully successful — RNN progressed to HPO (0.5101→0.5275). Architecture list centralization flawless — zero config reference errors. CNN unexpectedly recovered from pathological 0.0000 to 0.5339 (1st place). Best val P CNN 0.5339. 2/9 pass 0.53 filter (Dense 0.5303, CNN 0.5339). Ensemble val P 0.5273 / R 0.7050. Best inf P ever: Dense 0.6739, CNN 0.6695. CNN focal_loss+class_weight conflict diagnosed and fixed (model._is_focal flag). XGBoost root cause diagnosed: missing eval_set+early_stopping_rounds; fix applied (validation_data forwarding, TREE_EARLY_STOPPING_ROUNDS=10). | Fix 1+2 validated; CNN/XGBoost root causes found and fixed; iter8 pending |
| 3.49 | 2026-06-17 | **Section 4 LIVING RECORD refresh**: Replaced pre-GIS failure records (§4.5-4.7, May 2026) with 18 GIS-era entries across §4.1 Preprocessing (6), §4.2 Model Training (6), §4.3 Strategic Approaches (3), §4.4 False Signals (3). Old records archived to shortmemory.txt under `SPEC.md SECTION 4 ARCHIVE`. Header updated: "STATIC - NEVER CHANGES" → "LIVING RECORD — ARCHIVE TRACEABLE". Subtitle updated to domain-agnostic: "Dataset Classification Ensemble". | Focus Section 4 on GIS-era findings; archive historical records with traceability |
| 3.50 | 2026-06-18 | **GIS Iter 8 (All 5 fixes) + Fix 4 repair + Fix 5 completion**: Total 18,071s. VAE 0.5409 best val P (reclaims #1). 3/9 pass 0.53 filter (VAE, Dense, CNN). CNN 0.6818 best inf P (new record). Ensemble P 0.6898 / R 0.2917. Fix 3 (CNN focal_loss) **validated** — all 6 NNs selected focal_loss in HPO, no collapse. Fix 4 (XGBoost early stopping) **BROKEN** — early_stopping_rounds passed to .fit() crashes all XGBoost/LightGBM HPO trials. Fix 5 completed — last 2 hardcoded arch lists replaced. Fix 4 repair applied: early_stopping_rounds moved to tree constructors (chunk_11), removed from fit_kwargs (chunk_14), injected into HPO trial params (chunk_21). GIS.md up-revved to v1.7 with §8g Phase C lever catalog. | Fix 3 fully validated; Fix 4 repaired; iter9 pending |
| 3.51 | 2026-06-18 | **Comprehensive lever audit trail**: Added ~30 per-cycle log lines across 8 files (chunk_05, chunk_12, chunk_14, chunk_18, chunk_19, chunk_20, chunk_XX_feature_analysis_a). Every config lever is now directly verifiable from the .log at the pipeline stage where it takes effect. Startup config dump logs all lever values. Per-arch safeguard summary (1×/arch), early_stopping_rounds + scale_pos_weight source per tree training, class_weight skip reason per NN training, ensemble weighting scheme, winsor bounds at inference with percentiles + clip range, FI method active count. GIS.md §8h catalogs 7 previously undocumented levers (L1–L7). All 10 blind spots closed; .log is now the single source of truth for lever state. | All levers traceable in .log; no new logic or data paths |

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

## 3.3 Related Documentation

| File | Description | Location |
|------|-------------|----------|
| README.md | Pipeline architecture and usage guide | ./README.md |
| SPEC.md | This specification document | ./SPEC.md |
| AGENTS.md | Coding guidelines for this project | ../AGENTS.md |

---

## 3.4 File Inventory

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

## 3.5 Quality Attributes

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

# SECTION 4: Failed Strategies & Approaches (LIVING RECORD — ARCHIVE TRACEABLE)

This section records failed approaches discovered during GIS iterations. Entries must be tagged by iteration and cite evidence. Old records are archived to shortmemory.txt (search `SPEC.md SECTION 4 ARCHIVE`) — previous entries are never deleted without an archive record. Future iterations should reference this section before attempting new strategies.

## 4.1 Preprocessing Strategy Failures

| Failure | Iter | Strategy | Empirical Result | Resolution | Evidence |
|---------|------|----------|----------------|------------|----------|
| A3: Remove log1p transforms | 3 | Disable log1p on 4 skewed features (`HIGHLY_SKEWED_FEATURES` `[0,1,4,5]→[]`) | RNN 0.5357→0.0581, Transformer 0.5339→0.0617, LightGBM collapsed. 5/9→3/9 pass filter. | Reverted A3 (iter4) | GIS §8b |
| A2: Global winsorize HIGH 95→97 | 5 | Raise right-tail clip from 95→97 globally | Dense −0.3436, LSTM −0.4717 collapsed. RNN recovered (+0.4839). Contradictory effects across architectures. | Per-arch winsorization (Phase B, iter6) | GIS §8c |
| Permutation-only FI | 2 (inv) | Replace all 6 FI methods with permutation-only to save ~13min | Quick dense NN AUC≈0.50 at all thresholds → every feature `auc_drop≈0` → identical rank → positional slicing by column index. Iter2 invalidated. | Reverted to all 6 FI methods. Documented broken-pruning caveat. | GIS §8a |
| Global winsorization paradigm | 1–5 | Apply same LOW/HIGH percentiles to all architectures uniformly | Same preprocessing cannot serve all 9 architectures: some gain, others collapse. Architecture-specific sensitivity confirmed by A2 and iterative evidence. | Per-arch winsorization (Phase B, iter6) | GIS §8c |
| Phase A lever-only strategy | 1–5 | Zero-runtime config levers alone can find near-optimal config | All 4 levers swept (A1–A4). Best val P 0.5505 (barely above baseline 0.5473). None broke 0.55 structurally. Ceiling required runtime-impacting changes. | Abandoned Phase A; Phase B runtime levers initiated | GIS §8b |
| A1: WINSORIZE_LOW 2→3 | 2 | Tighter left-tail clip to reduce noise | 0.5473→0.5476 (+0.0003). Functionally neutral — ceiling unchanged. | Kept as default (no-op benefit) | GIS §5 iter2 |

## 4.2 Model Training & HPO Design Failures

| Failure | Iter | Strategy | Empirical Result | Resolution | Evidence |
|---------|------|----------|----------------|------------|----------|
| CNN focal + class_weight combined | 4–6 | Apply both focal_loss (gamma=2) and `class_weight='balanced'` (259x) simultaneously | Contradictory gradients: focal down-weights well-classified samples via gamma, class weight demands 259x attention to positives. Network outputs ~0.5 for all inputs. CNN val P 0.0000 for 3 consecutive iterations. | `model._is_focal` flag stored at HPO build; `class_weight` skipped when `_is_focal=True` | GIS §8f Fix 3 |
| XGBoost no early stopping | 6–7 | Trust sklearn wrapper defaults; `eval_set` not forwarded to fit | All 500 `n_estimators` trained unchecked. Train PRAUC 0.9093 vs val 0.0850. XGBoost val P stuck at 0.0867–0.0924 across 2 iterations. | `validation_data` forwarding; `eval_set=[(X_val, y_val)]` + `early_stopping_rounds=10` | GIS §8f Fix 4 |
| XGBoost scale_pos_weight locked out | 6 | `DYNAMIC_CLASS_WEIGHTS=True` overrides HPO-provided `scale_pos_weight` | HPO cannot tune class balance. All 5 trials worse than baseline (best 0.0648 vs 0.0867). HPO always discarded. | Priority: HPO > dynamic > default | GIS §8e Fix 1 |
| Neural safeguard 2% universal | 6 | `MIN_PRECISION_OVER_BASELINE=0.02` applied to all architectures uniformly | RNN LT=0.0 rejected by 0.0002 margin (0.5198 ≤ 0.5200). HPO ran at LT=20.0 where 3/5 trials produced TP=0. | Reduced to 0.01 for all 6 NN architectures | GIS §8e Fix 2 |
| 12 dead HPO dimensions (bugs 2–8) | 1–5 (latent) | Builder code ignores HPO param dict; trains at default params | 5 NN architectures: units, layers, activation, pooling, bidirectional, ff_dim all dead dimensions. Architectures unknowingly trained at default values for entire legacy period. | All parameters wired in builder code (Jun 14 fix) | GIS §5 fixes 14–16 |
| 5 HPO trials insufficient for stable ranking | 6→7 | Coarse 5-trial exploration assumed sufficient for reliable architecture ranking | Transformer 0.5505→0.5216 with zero config change — pure stochastic variance. Ranking not reproducible at 5 trials. | Phase C lever (pending) | GIS §1.5 P7 |

## 4.3 Strategic Approach Failures

| Failure | Iter Range | Strategy | Empirical Result | Resolution | Evidence |
|---------|-----------|----------|----------------|------------|----------|
| Temporal weighting as value-add | 1–7 | Weight recent training data higher; temporal signal improves predictions | Zero positive temporal gap across all 7 iterations. Dense iter3's +0.2273 was A3 artifact (disappeared when log1p restored). No architecture demonstrated real recency signal. | Strategy produced zero measurable benefit | GIS §5 all iters |
| ≤3 ensemble members for robust inference | 3, 6, 7 | Uniform consensus among 2–3 passing architectures is sufficient | Without ≥4 members, consensus loses buffering: CNN all-positive, Dense all-negative (iter3). Inference P degrades at low filter counts. Ensemble P 0.5417–0.6494 vs 0.7552 at 5-member ensemble (iter2). | Maintain ≥4 if possible; accept limitation | GIS §1.5 P5 |
| 0.60 validation precision assumed attainable | 1–7 | Mission target set at ≥0.60 val P per architecture; Phase A + B levers will close gap | 7 iterations, 0 architectures reach 0.60. Best ever: 0.5505 (Transformer, iter6). 6/9 architectures never pass 0.53 ensemble filter. Ceiling appears structural. | Target under review; Phase C pending | GIS §1 Key Results |

## 4.4 False Signals & Interpretation Errors

| Signal | Iter | What Appeared to Happen | What Was Actually Happening | Resolution | Evidence |
|--------|------|------------------------|---------------------------|------------|----------|
| Dense temporal gap +0.2273 | 3 | Dense appeared to learn recency signal under gate relaxation (A4) | A3 distribution-shift artifact: log1p removal altered feature distribution, creating spurious temporal pattern. Gap disappeared when log1p restored in iter4. | Classified as artifact in P4 pattern | GIS §1.5 P4 |
| A4 benefit to Dense/CNN | 3 | Gate relaxation (`MAX_POS_PRED_RATIO 0.60→0.65`) improved Dense (+0.0119) and CNN (+0.0114) | iter4 with A3 reverted and A4 still active: Dense −0.0031 vs iter2. The iter3 "benefit" was A3 artifact from altered feature distribution, not A4. | A4 kept but recognized as marginal; benefit was illusory | GIS §1.5 P3/P4 |
| CNN recovery appeared to validate Fix 3 | 7 | CNN 0.0000→0.5339 (1st place) after focal_loss + class_weight fix applied | HPO may have coincidentally chosen non-focal-loss trials; recovery may be A4 gate relaxation (excluding MaxPred>0.65 in iters 4–6). Fix is correct but iter7 alone does not prove it. | iter8 needed for definitive validation | GIS §1.5 P6 |

---

### Archive Note

The original §4.5 (NN Prediction Range Failures), §4.6 (XGBoost Train-Val Gap), §4.7 (Phase 5 Crash), and GIS Hyperparameter Reconfiguration (May 2026) have been archived to `shortmemory.txt` under `SPEC.md SECTION 4 ARCHIVE` to keep this section focused on findings from the GIS iteration era (iter1–7 onward).

*Document generated: 2026-04-15*  
*Last updated: 2026-06-17*  
*Version: 3.51*


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



