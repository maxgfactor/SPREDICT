"""
evaluate.py — Evaluation Metrics, Evaluator Class, Threshold Search, Diagnostics
Refactored from chunk_04_utils_metrics.py + chunk_12_evaluation_evaluator.py (2026-08-07).
Section 1: stateless metric functions (extracted from chunk_12 Evaluator methods).
Section 2: Evaluator class (metric orchestration + threshold search).
Section 3: XGBoost coverage sweep (extracted from inline chunk_18:1132-1194).
Section 4: diagnostic / analysis functions (from chunk_04).
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, cohen_kappa_score, roc_curve, fbeta_score
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler  # NN-only normalization (trees are scale-invariant)

from config import PREDICTION_THRESHOLD_DEFAULT


# ============================================================================
# Section 1: Stateless Metric Functions
# SOURCE CORRECTION: these 16 metric functions were methods of the Evaluator
# class in chunk_12 (lines 31-198). They are extracted to top-level functions
# here. safe_divide() is NEW (plan helper — does not exist in source).
# Each function: (y_true, y_pred/y_proba) -> float
# ============================================================================

def safe_divide(numerator: float, denominator: float) -> float:
    """NEW helper — division guard. Returns 0.0 on zero/empty denominator."""
    try:
        if denominator is None or float(denominator) == 0:
            return 0.0
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision = TP / (TP + FP)."""
    try:
        return float(precision_score(y_true, y_pred, average='binary', zero_division=0))
    except Exception:
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return safe_divide(tp, tp + fp)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall = TP / (TP + FN)."""
    try:
        return float(recall_score(y_true, y_pred, average='binary', zero_division=0))
    except Exception:
        return 0.0


def calculate_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """AUC score. Returns 0.0 on failure."""
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return 0.0


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2PR / (P + R)."""
    try:
        return float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    except Exception:
        p = calculate_precision(y_true, y_pred)
        r = calculate_recall(y_true, y_pred)
        return safe_divide(2 * p * r, p + r)


def calculate_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient."""
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return 0.0


def calculate_average_precision(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Precision-Recall AUC (Average Precision)."""
    try:
        return float(average_precision_score(y_true, y_proba))
    except Exception:
        return 0.0


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity (True Negative Rate) = TN / (TN + FP)."""
    try:
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return safe_divide(tn, tn + fp)
    except Exception:
        return 0.0


def calculate_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Positive Rate = 1 - Specificity."""
    try:
        return 1.0 - calculate_specificity(y_true, y_pred)
    except Exception:
        return 0.0


def calculate_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F2 score (beta=2)."""
    try:
        return float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    except Exception:
        p = calculate_precision(y_true, y_pred)
        r = calculate_recall(y_true, y_pred)
        return safe_divide(5 * p * r, 4 * p + r)


def calculate_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score for probability calibration (lower=better)."""
    try:
        return float(brier_score_loss(y_true, y_proba))
    except Exception:
        return 0.0


def calculate_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa inter-rater reliability."""
    try:
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return 0.0


def calculate_informedness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Informedness = Recall + Specificity - 1."""
    try:
        return calculate_recall(y_true, y_pred) + calculate_specificity(y_true, y_pred) - 1.0
    except Exception:
        return 0.0


def calculate_markedness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Markedness = Precision + Specificity - 1."""
    try:
        return calculate_precision(y_true, y_pred) + calculate_specificity(y_true, y_pred) - 1.0
    except Exception:
        return 0.0


def calculate_gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Gini coefficient = max(0, 2 * AUC - 1)."""
    try:
        return max(0.0, 2 * calculate_auc(y_true, y_proba) - 1)
    except Exception:
        return 0.0


def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced Accuracy = (Recall + Specificity) / 2."""
    try:
        return (calculate_recall(y_true, y_pred) + calculate_specificity(y_true, y_pred)) / 2.0
    except Exception:
        return 0.0


def calculate_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J (TPR - FPR). Returns threshold value."""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        return float(thresholds[optimal_idx]) if len(thresholds) > 0 else 0.5
    except Exception:
        return 0.5


# ============================================================================
# Section 2: Evaluator Class
# From chunk_12_evaluation_evaluator.py — metric orchestration.
# The 16 core metric methods delegate to the stateless Section 1 functions.
# ============================================================================

class Evaluator:
    """Comprehensive model evaluation utility with defensive programming"""

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger

    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_precision(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"precision_score failed: {e}", 'warning')
            return 0.0

    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_recall(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"recall_score failed: {e}", 'warning')
            return 0.0

    def calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_f1(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"f1_score failed: {e}", 'warning')
            return 0.0

    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_specificity(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_specificity failed: {e}", 'warning')
            return 0.0

    def calculate_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_fpr(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_fpr failed: {e}", 'warning')
            return 0.0

    def calculate_f2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_f2_score(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"f2_score calculation failed: {e}", 'warning')
            return 0.0

    def calculate_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            return calculate_brier_score(y_true, y_proba)
        except Exception as e:
            if self.logger: self.logger.log(f"brier_score_loss failed: {e}", 'warning')
            return 0.0

    def calculate_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_kappa(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"cohen_kappa_score failed: {e}", 'warning')
            return 0.0

    def calculate_informedness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_informedness(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_informedness failed: {e}", 'warning')
            return 0.0

    def calculate_markedness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_markedness(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_markedness failed: {e}", 'warning')
            return 0.0

    def calculate_gini(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        try:
            return calculate_gini(y_true, y_scores)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_gini failed: {e}", 'warning')
            return 0.0

    def calculate_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        try:
            return calculate_optimal_threshold(y_true, y_scores)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_optimal_threshold failed: {e}", 'warning')
            return 0.5

    def calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_mcc(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_mcc failed: {e}", 'warning')
            return 0.0

    def calculate_average_precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        try:
            return calculate_average_precision(y_true, y_scores)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_average_precision failed: {e}", 'warning')
            return 0.0

    def calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return calculate_balanced_accuracy(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_balanced_accuracy failed: {e}", 'warning')
            return 0.0

    def calculate_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        try:
            return calculate_auc(y_true, y_scores)
        except Exception as e:
            if self.logger: self.logger.log(f"roc_auc_score failed: {e}", 'warning')
            return 0.0

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Returns (TP, FP, TN, FN) counts."""
        try:
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        except Exception as e:
            if self.logger: self.logger.log(f"confusion matrix calculation failed: {e}", 'warning')
            return {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    def evaluate_at_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Compute ALL 24 metrics at once.
        Returns dict with keys: PRECISION, TRUE_POSITIVES, FALSE_POSITIVES,
        TRUE_NEGATIVES, FALSE_NEGATIVES, MAX_PREDICTION, MEAN_PREDICTION,
        RECALL, F1_SCORE, AUC, SPECIFICITY, FALSE_POSITIVE_RATE, F2_SCORE,
        MCC, PRAUC, BALANCED_ACCURACY, Brier, Kappa, Informedness, Markedness,
        Gini, OPTIMAL_THRESHOLD, STD_PREDICTION, PCT_ABOVE_THRESHOLD
        """
        try:
            if not np.all(np.isfinite(y_true)):
                if self.logger: self.logger.log(f"y_true contains NaN/Inf values", 'warning')
                return {
                    'P': 0.0, 'R': 0.0, 'F1': 0.0, 'AUC': 0.0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                    'Spec': 0.0, 'FPR': 0.0, 'F2': 0.0, 'MCC': 0.0, 'PRAUC': 0.0, 'BalAcc': 0.0, 'Brier': 0.0, 'Kappa': 0.0,
                    'Informedness': 0.0, 'Markedness': 0.0, 'Gini': 0.0, 'OptThresh': 0.0, 'MaxPred': 0.0, 'MeanPred': 0.0, 'StdPred': 0.0, 'PctAboveThresh': 0.0
                }
            if not np.all(np.isfinite(y_pred_proba)):
                if self.logger: self.logger.log(f"y_pred_proba contains NaN/Inf values", 'warning')
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0, posinf=1.0, neginf=0.0)
                y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

            unique_true = np.unique(y_true)
            if len(unique_true) < 2:
                if self.logger: self.logger.log(f"Only one class present in y_true: {unique_true}", 'warning')

            pred_threshold = self.config['PREDICTION_THRESHOLD']
            y_pred_binary = (y_pred_proba >= pred_threshold).astype(int)

            max_pred = float(y_pred_proba.max()) if len(y_pred_proba) > 0 else 0.0
            mean_pred = float(y_pred_proba.mean()) if len(y_pred_proba) > 0 else 0.0
            std_pred = float(y_pred_proba.std()) if len(y_pred_proba) > 0 else 0.0
            pct_above = float(np.mean(y_pred_binary)) * 100.0 if len(y_pred_binary) > 0 else 0.0

            spec = calculate_specificity(y_true, y_pred_binary)
            fpr = calculate_fpr(y_true, y_pred_binary)
            f2 = calculate_f2_score(y_true, y_pred_binary)
            mcc = calculate_mcc(y_true, y_pred_binary)
            prauc = calculate_average_precision(y_true, y_pred_proba)
            balacc = calculate_balanced_accuracy(y_true, y_pred_binary)
            brier = calculate_brier_score(y_true, y_pred_proba)
            kappa = calculate_kappa(y_true, y_pred_binary)
            informedness = calculate_informedness(y_true, y_pred_binary)
            markedness = calculate_markedness(y_true, y_pred_binary)
            gini = calculate_gini(y_true, y_pred_proba)
            opt_thresh = calculate_optimal_threshold(y_true, y_pred_proba)

            return {
                'P': calculate_precision(y_true, y_pred_binary),
                'R': calculate_recall(y_true, y_pred_binary),
                'F1': calculate_f1(y_true, y_pred_binary),
                'AUC': calculate_auc(y_true, y_pred_proba),
                'TP': int(np.sum((y_pred_binary == 1) & (y_true == 1))),
                'TN': int(np.sum((y_pred_binary == 0) & (y_true == 0))),
                'FP': int(np.sum((y_pred_binary == 1) & (y_true == 0))),
                'FN': int(np.sum((y_pred_binary == 0) & (y_true == 1))),
                'Spec': spec,
                'FPR': fpr,
                'F2': f2,
                'MCC': mcc,
                'PRAUC': prauc,
                'BalAcc': balacc,
                'Brier': brier,
                'Kappa': kappa,
                'Informedness': informedness,
                'Markedness': markedness,
                'Gini': gini,
                'OptThresh': opt_thresh,
                'MaxPred': max_pred,
                'MeanPred': mean_pred,
                'StdPred': std_pred,
                'PctAboveThresh': pct_above,
            }
        except Exception as e:
            if self.logger: self.logger.log(f"evaluate_at_threshold failed: {e}", 'warning')
            return {
                'P': 0.0, 'R': 0.0, 'F1': 0.0, 'AUC': 0.0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'Spec': 0.0, 'FPR': 0.0, 'F2': 0.0, 'MCC': 0.0, 'PRAUC': 0.0, 'BalAcc': 0.0, 'Brier': 0.0, 'Kappa': 0.0,
                'Informedness': 0.0, 'Markedness': 0.0, 'Gini': 0.0, 'OptThresh': 0.0, 'MaxPred': 0.0, 'MeanPred': 0.0, 'StdPred': 0.0, 'PctAboveThresh': 0.0
            }

    def find_optimal_threshold(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               model, model_trainer, arch_name: str,
                               thresholds: np.ndarray, patience: int = 5,
                               retrain_model: bool = True,
                               threshold_feature_indices: Dict = None) -> Tuple[float, float, List[Dict], Any]:
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
        best_thresh = thresholds[0]
        best_precision = 0.0
        best_trained_model = None
        last_trained_model = None
        results = []
        no_improve_count = 0

        for thresh in thresholds:
            if threshold_feature_indices is not None:
                thresh_key = round(float(thresh), 1)
                feat_idx = threshold_feature_indices.get(thresh_key)
                if feat_idx is not None:
                    X_train_t = X_train[:, feat_idx]
                    X_val_t = X_val[:, feat_idx]
                else:
                    X_train_t = X_train
                    X_val_t = X_val
            else:
                X_train_t = X_train
                X_val_t = X_val

            if arch_name in self.config['NEURAL_ARCHITECTURES']:
                scaler = StandardScaler()
                X_train_t = scaler.fit_transform(X_train_t)
                X_val_t = scaler.transform(X_val_t)

            y_train_binary = (y_train >= thresh).astype(int)
            y_val_binary = (y_val >= thresh).astype(int)

            train_class_0 = int(np.sum(y_train_binary == 0))
            train_class_1 = int(np.sum(y_train_binary == 1))
            val_class_0 = int(np.sum(y_val_binary == 0))
            val_class_1 = int(np.sum(y_val_binary == 1))

            train_positive_found = train_class_1 > 0
            val_positive_found = val_class_1 > 0

            arch_tag = f"[{arch_name.upper()}]"
            if self.logger: self.logger.log(
                f"{arch_tag} label_threshold={thresh:.1f} "
                f"train_below_prediction_binary_split={train_class_0:,} "
                f"train_above_prediction_binary_split={train_class_1:,} "
                f"train_positive_found={str(train_positive_found).lower()} "
                f"val_below_prediction_binary_split={val_class_0:,} "
                f"val_above_prediction_binary_split={val_class_1:,} "
                f"val_positive_found={str(val_positive_found).lower()}", 'info')

            try:
                if model is not None and not retrain_model:
                    trained = model
                elif model is not None:
                    if hasattr(model, 'sklearn_model'):
                        trained, _ = model_trainer._train_sklearn_model(model, X_train_t, y_train_binary)
                    else:
                        epochs = self.config['HPO_RETRAIN_EPOCHS'].get(arch_name, 3)
                        trained, _ = model_trainer.train_model(model, X_train_t, y_train_binary, epochs=epochs, verbose=0)
                else:
                    fresh_model = model_trainer.build_architecture(arch_name, X_train_t.shape[1])
                    if hasattr(fresh_model, 'sklearn_model'):
                        trained, _ = model_trainer._train_sklearn_model(fresh_model, X_train_t, y_train_binary)
                    else:
                        epochs = self.config['HPO_RETRAIN_EPOCHS'].get(arch_name, 3)
                        trained, _ = model_trainer.train_model(fresh_model, X_train_t, y_train_binary, epochs=epochs, verbose=0)
            except Exception as e:
                if self.logger: self.logger.log(f"Training failed for {arch_name} at threshold {thresh}: {e}", 'warning')
                continue
            last_trained_model = trained

            try:
                train_pred = trained.predict(X_train_t, verbose=0).flatten()
                val_pred = trained.predict(X_val_t, verbose=0).flatten()

                if not np.all(np.isfinite(train_pred)):
                    if self.logger: self.logger.log(f"{arch_tag} train_pred contains NaN/Inf at LABEL_THRESHOLD={thresh}", 'warning')
                    train_pred = np.nan_to_num(train_pred, nan=0.0, posinf=1.0, neginf=0.0)
                if not np.all(np.isfinite(val_pred)):
                    if self.logger: self.logger.log(f"{arch_tag} val_pred contains NaN/Inf at LABEL_THRESHOLD={thresh}", 'warning')
                    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0)

                train_pred = np.clip(train_pred, 1e-7, 1 - 1e-7)
                val_pred = np.clip(val_pred, 1e-7, 1 - 1e-7)
            except Exception as e:
                if self.logger: self.logger.log(f"Prediction failed for {arch_name} at threshold {thresh}: {e}", 'warning')
                continue

            train_metrics = self.evaluate_at_threshold(y_train_binary, train_pred)
            val_metrics = self.evaluate_at_threshold(y_val_binary, val_pred)

            if arch_name in self.config['TREE_ARCHITECTURES']:
                sklearn_safeguards = self.config['SKLEARN_SAFEGUARDS']
                min_positive_percentage = sklearn_safeguards['MIN_POSITIVE_PERCENTAGE']
                min_positive_absolute = sklearn_safeguards['MIN_POSITIVE_ABSOLUTE']
            elif arch_name in self.config['NEURAL_ARCHITECTURES']:
                neural_safeguards = self.config['NEURAL_SAFEGUARDS']
                min_positive_percentage = neural_safeguards['MIN_POSITIVE_PERCENTAGE']
                min_positive_absolute = neural_safeguards['MIN_POSITIVE_ABSOLUTE']
                patience = neural_safeguards['PATIENCE']
            else:
                min_positive_percentage = 0.01
                min_positive_absolute = 100
                patience = self.config['PATIENCE']
            n_samples = len(y_val_binary)
            min_positive_predictions = max(min_positive_absolute, int(n_samples * min_positive_percentage))

            val_total_positive_preds = val_metrics['TP'] + val_metrics['FP']
            if val_total_positive_preds < min_positive_predictions:
                if self.logger: self.logger.log(f"[reject] threshold={thresh:.1f} action=skipped reason=insufficient_positive_predictions positive_predictions={val_total_positive_preds} minimum_positive_required={min_positive_predictions}", 'info')
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.logger: self.logger.log(f"{arch_name}: Early stopping at threshold {thresh:.1f} (no improvement for {patience} iterations)", 'info')
                    break
                continue

            if arch_name in self.config['TREE_ARCHITECTURES']:
                sklearn_safeguards = self.config['SKLEARN_SAFEGUARDS']
                min_pos_ratio = sklearn_safeguards.get('MIN_POS_PRED_RATIO', self.config['MIN_POS_PRED_RATIO'])
                max_pos_ratio = sklearn_safeguards.get('MAX_POS_PRED_RATIO', self.config['MAX_POS_PRED_RATIO'])
                if arch_name == 'XGBoost':
                    max_pos_ratio = 0.48
            elif arch_name in self.config['NEURAL_ARCHITECTURES']:
                neural_safeguards = self.config.get('NEURAL_SAFEGUARDS', {})
                min_pos_ratio = neural_safeguards.get('MIN_POS_PRED_RATIO', self.config['MIN_POS_PRED_RATIO'])
                max_pos_ratio = neural_safeguards.get('MAX_POS_PRED_RATIO', self.config['MAX_POS_PRED_RATIO'])
            else:
                min_pos_ratio = self.config['MIN_POS_PRED_RATIO']
                max_pos_ratio = self.config['MAX_POS_PRED_RATIO']
            pos_pred_ratio = val_total_positive_preds / len(y_val_binary)
            if pos_pred_ratio < min_pos_ratio or pos_pred_ratio > max_pos_ratio:
                if self.logger: self.logger.log(f"{arch_tag} [reject] skipping LABEL_THRESHOLD={thresh:.1f}: pos_pred_ratio={pos_pred_ratio:.2%} outside [{min_pos_ratio:.0%}, {max_pos_ratio:.0%}]", 'info')
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.logger: self.logger.log(f"{arch_tag} Early stopping at threshold {thresh:.1f}", 'info')
                    break
                continue

            baseline_precision = min(y_val_binary.mean(), 0.5)
            if arch_name in self.config['TREE_ARCHITECTURES']:
                sklearn_safeguards = self.config['SKLEARN_SAFEGUARDS']
                min_improvement = sklearn_safeguards.get('MIN_PRECISION_OVER_BASELINE', 0.05)
            elif arch_name in self.config['NEURAL_ARCHITECTURES']:
                neural_safeguards = self.config.get('NEURAL_SAFEGUARDS', {})
                min_improvement = neural_safeguards.get('MIN_PRECISION_OVER_BASELINE', self.config['MIN_PRECISION_OVER_BASELINE'])
            else:
                min_improvement = self.config['MIN_PRECISION_OVER_BASELINE']

            if not hasattr(self, '_safeguard_logged'):
                self._safeguard_logged = set()
            if arch_name not in self._safeguard_logged:
                self._safeguard_logged.add(arch_name)
                safe_type = 'tree' if arch_name in self.config['TREE_ARCHITECTURES'] else (
                    'neural' if arch_name in self.config['NEURAL_ARCHITECTURES'] else 'global')
                if self.logger:
                    self.logger.log(
                        f"[{arch_name}] Safeguards: type={safe_type}, "
                        f"min_precision_over_baseline={min_improvement:.4f}, "
                        f"min_pos_pct={min_positive_percentage:.4f}, "
                        f"min_pos_abs={min_positive_absolute}, "
                        f"patience={patience}",
                        'info'
                    )

            current_precision = val_metrics['P']
            if current_precision <= baseline_precision + min_improvement:
                if self.logger: self.logger.log(f"{arch_tag} [reject] skipping LABEL_THRESHOLD={thresh:.1f}: validation_precision={current_precision:.4f} <= baseline+{min_improvement:.4f}={baseline_precision+min_improvement:.4f}", 'info')
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.logger: self.logger.log(f"{arch_name}: Early stopping at threshold {thresh:.1f}", 'info')
                    break
                continue

            if self.logger: self.logger.log(f"{arch_tag} [diagnostic] LABEL_THRESHOLD={thresh:.1f} train_precision={train_metrics['P']:.4f} train_true_positives={train_metrics['TP']} train_true_negatives={train_metrics['TN']} train_false_positives={train_metrics['FP']} train_false_negatives={train_metrics['FN']} train_max_prediction={train_metrics['MaxPred']:.4f} train_mean_prediction={train_metrics['MeanPred']:.4f} train_recall={train_metrics['R']:.4f} train_f1={train_metrics['F1']:.4f} train_auc={train_metrics['AUC']:.4f} train_specificity={train_metrics['Spec']:.4f} train_false_positive_rate={train_metrics['FPR']:.4f} train_f2={train_metrics['F2']:.4f} train_mcc={train_metrics['MCC']:.4f} train_prauc={train_metrics['PRAUC']:.4f} train_balanced_accuracy={train_metrics['BalAcc']:.4f} train_brier={train_metrics['Brier']:.4f} train_kappa={train_metrics['Kappa']:.4f} train_informedness={train_metrics['Informedness']:.4f} train_markedness={train_metrics['Markedness']:.4f} train_gini={train_metrics['Gini']:.4f} train_optimal_threshold={train_metrics['OptThresh']:.4f} train_standard_deviation_prediction={train_metrics['StdPred']:.4f} train_percentage_above_threshold={train_metrics['PctAboveThresh']:.2f}", 'info')
            if self.logger: self.logger.log(f"{arch_tag} [diagnostic] LABEL_THRESHOLD={thresh:.1f} VALIDATION_PRECISION={val_metrics['P']:.4f} VALIDATION_TRUE_POSITIVES={val_metrics['TP']} VALIDATION_TRUE_NEGATIVES={val_metrics['TN']} validation_false_positives={val_metrics['FP']} validation_false_negatives={val_metrics['FN']} validation_max_prediction={val_metrics['MaxPred']:.4f} validation_mean_prediction={val_metrics['MeanPred']:.4f} validation_recall={val_metrics['R']:.4f} validation_f1={val_metrics['F1']:.4f} validation_auc={val_metrics['AUC']:.4f} validation_specificity={val_metrics['Spec']:.4f} validation_false_positive_rate={val_metrics['FPR']:.4f} validation_f2={val_metrics['F2']:.4f} validation_mcc={val_metrics['MCC']:.4f} validation_prauc={val_metrics['PRAUC']:.4f} validation_balanced_accuracy={val_metrics['BalAcc']:.4f} validation_brier={val_metrics['Brier']:.4f} validation_kappa={val_metrics['Kappa']:.4f} validation_informedness={val_metrics['Informedness']:.4f} validation_markedness={val_metrics['Markedness']:.4f} validation_gini={val_metrics['Gini']:.4f} validation_optimal_threshold={val_metrics['OptThresh']:.4f} validation_standard_deviation_prediction={val_metrics['StdPred']:.4f} validation_percentage_above_threshold={val_metrics['PctAboveThresh']:.2f}", 'info')

            result = {
                'threshold': thresh,
                'train': train_metrics,
                'val': val_metrics,
            }
            results.append(result)

            current_precision = val_metrics['P']
            if current_precision > best_precision:
                best_precision = current_precision
                best_thresh = thresh
                best_trained_model = trained
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                if self.logger: self.logger.log(f"{arch_name}: Early stopping at threshold {thresh:.1f} (no improvement for {patience} iterations)", 'info')
                break

        return best_thresh, best_precision, results, best_trained_model if best_trained_model is not None else last_trained_model

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Full metric dict (chunk_12:572). LIVE — called by Phase 4 Section 5."""
        metrics = {}

        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"accuracy_score failed: {e}", 'warning')
            metrics['accuracy'] = 0.0

        try:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=self.config['ZERO_DIVISION_MODE'])
        except Exception as e:
            if self.logger: self.logger.log(f"precision_score failed: {e}", 'warning')
            metrics['precision'] = 0.0

        try:
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=self.config['ZERO_DIVISION_MODE'])
        except Exception as e:
            if self.logger: self.logger.log(f"recall_score failed: {e}", 'warning')
            metrics['recall'] = 0.0

        try:
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=self.config['ZERO_DIVISION_MODE'])
        except Exception as e:
            if self.logger: self.logger.log(f"f1_score failed: {e}", 'warning')
            metrics['f1'] = 0.0

        try:
            if y_pred_proba is not None:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"roc_auc_score failed: {e}", 'warning')
            metrics['auc'] = 0.0

        try:
            if y_pred_proba is not None:
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            else:
                metrics['average_precision'] = average_precision_score(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"average_precision_score failed: {e}", 'warning')
            metrics['average_precision'] = 0.0

        try:
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"matthews_corrcoef failed: {e}", 'warning')
            metrics['mcc'] = 0.0

        try:
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['specificity'] = specificity
        except Exception as e:
            if self.logger: self.logger.log(f"specificity calculation failed: {e}", 'warning')
            metrics['specificity'] = 0.0

        try:
            recall = metrics.get('recall', 0.0)
            specificity = metrics.get('specificity', 0.0)
            metrics['balanced_accuracy'] = (recall + specificity) / 2.0
        except Exception as e:
            if self.logger: self.logger.log(f"balanced_accuracy calculation failed: {e}", 'warning')
            metrics['balanced_accuracy'] = 0.0

        return metrics


# ============================================================================
# Section 3: XGBoost Coverage Sweep
# NEW function — extracted from inline chunk_18:1132-1194 (gated by
# PREDICTION_XGBOOST_PRECISION_TARGETING for arch XGBoost). Behavior matches
# chunk_18:1132-1194 exactly (search_results, chosen_rate, F1-optimal fallback).
# ============================================================================

def search_coverage_thresholds(y_true: np.ndarray, y_proba: np.ndarray,
                               coverage_rates: List[float],
                               target_precision: float,
                               max_coverage: float,
                               config: Dict,
                               evaluator: Optional[Evaluator] = None,
                               logger=None,
                               arch_name: str = 'XGBoost',
                               min_positive_percentage: float = 0.001,
                               min_positive_absolute: int = 10) -> float:
    """
    XGBoost coverage sweep. Always called for XGBoost in Phase 4 Section 5.
    9 coverage rates from config: [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
                                   0.10, 0.25, 0.50].
    For each rate: find prediction threshold that achieves that coverage.

    Gated by PREDICTION_XGBOOST_PRECISION_TARGETING toggle:
    - True:  try precision-targeted thresholds first; if none reach target → F1-optimal fallback
    - False: skip precision targeting, return F1-optimal threshold directly

    In both cases returns a threshold (either precision-targeted or F1-optimal).
    """
    pred_threshold = config['PREDICTION_THRESHOLD']

    if not config.get('PREDICTION_XGBOOST_PRECISION_TARGETING', False) or arch_name != 'XGBoost':
        return pred_threshold

    if evaluator is None:
        evaluator = Evaluator(config, logger=logger)

    y_val_binary_search = (y_true >= 0.5).astype(int) if y_true.dtype.kind == 'f' else y_true

    sorted_preds = np.sort(y_proba.flatten())
    n_val = len(sorted_preds)
    if n_val == 0:
        return pred_threshold

    best_pred_threshold = pred_threshold
    best_f1 = 0.0
    search_results = []
    chosen_rate = None

    for rate in sorted(coverage_rates):
        if rate > max_coverage:
            break
        k = max(1, int(n_val * (1.0 - rate)))
        pred_thresh = sorted_preds[min(k, n_val - 1)]

        val_binary_test = (y_proba >= pred_thresh).astype(int)

        if arch_name in config['TREE_ARCHITECTURES']:
            _sg = config['SKLEARN_SAFEGUARDS']
        elif arch_name in config['NEURAL_ARCHITECTURES']:
            _sg = config['NEURAL_SAFEGUARDS']
        else:
            _sg = {'MIN_POSITIVE_ABSOLUTE': min_positive_absolute, 'MIN_POSITIVE_PERCENTAGE': min_positive_percentage}
        _min_pos = max(_sg['MIN_POSITIVE_ABSOLUTE'], int(n_val * _sg['MIN_POSITIVE_PERCENTAGE']))

        if val_binary_test.sum() >= _min_pos:
            precision = evaluator.calculate_precision(y_val_binary_search, val_binary_test)
            recall = evaluator.calculate_recall(y_val_binary_search, val_binary_test)
            f1 = evaluator.calculate_f1(y_val_binary_search, val_binary_test)
            search_results.append((rate, pred_thresh, precision, recall, f1))

            if f1 > best_f1:
                best_f1 = f1

            if precision >= target_precision and chosen_rate is None:
                chosen_rate = rate
                best_pred_threshold = pred_thresh

    if search_results:
        if logger: logger.log(f"   [diagnostic] xgboost_coverage_sweep: tested {len(search_results)} coverage rates", 'info')
        for r in search_results[-5:]:
            if logger: logger.log(
                f"   [diag]   coverage={r[0]:.4f} thresh={r[1]:.4f}: "
                f"precision={r[2]:.4f} recall={r[3]:.4f} f1={r[4]:.4f}", 'info')
        if chosen_rate is None:
            best_result = max(search_results, key=lambda x: x[4])
            best_pred_threshold = best_result[1]
            if logger: logger.log(
                f"   [diagnostic] precision_target={target_precision:.2f} not met — "
                f"falling back to F1-optimal threshold: {best_pred_threshold:.4f} "
                f"(coverage={best_result[0]:.4f}, f1={best_result[4]:.4f})", 'info')
        else:
            if logger: logger.log(
                f"   [diagnostic] precision_target={target_precision:.2f} met at "
                f"coverage={chosen_rate:.4f}, threshold={best_pred_threshold:.4f}", 'info')

    return best_pred_threshold


# ============================================================================
# Section 4: Diagnostic / Analysis Functions
# All from chunk_04_utils_metrics.py. Used in Phase 4 SECTION 3 (Diagnostics).
# ============================================================================

def get_prediction_percentiles(predictions: np.ndarray) -> Dict[str, float]:
    """Compute percentiles (p1, p5, p10, p25, p50, p75, p90, p95, p99, max)."""
    if len(predictions) == 0:
        return {'p1': 0.0, 'p5': 0.0, 'p10': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    p = np.percentile(predictions, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        'p1': float(p[0]), 'p5': float(p[1]), 'p10': float(p[2]), 'p25': float(p[3]),
        'p50': float(p[4]), 'p75': float(p[5]), 'p90': float(p[6]), 'p95': float(p[7]),
        'p99': float(p[8]), 'max': float(predictions.max())
    }


def get_round_threshold_density(predictions: np.ndarray) -> str:
    """Internal helper for format_diagnostic_string() (chunk_04:97)."""
    if len(predictions) == 0:
        return ""
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    total = len(predictions)
    parts = []
    for t in thresholds:
        pct = (predictions <= t).sum() / total * 100
        parts.append(f"{pct:.0f}% ≤ {t}")
    pct_above = (predictions >= 0.50).sum() / total * 100
    parts.append(f"{pct_above:.0f}% ≥ 0.50")
    return ", ".join(parts)


def format_diagnostic_string(predictions: np.ndarray, prefix: str = "") -> str:
    """Format comprehensive diagnostic string for predictions (chunk_04:111)."""
    if len(predictions) == 0:
        return f"{prefix} No predictions"

    stats = get_prediction_percentiles(predictions)

    result = f"{prefix} cumulative binary_split_predictions (1%) ≤ {stats['p1']:.4f}, (5%) ≤ {stats['p5']:.4f}, (10%) ≤ {stats['p10']:.4f}, (25%) ≤ {stats['p25']:.4f}, (50%) ≤ {stats['p50']:.4f}, (75%) ≤ {stats['p75']:.4f}, (90%) ≤ {stats['p90']:.4f}, (95%) ≤ {stats['p95']:.4f}, (99%) ≤ {stats['p99']:.4f}, max={stats['max']:.4f}"

    density_str = get_round_threshold_density(predictions)
    if density_str:
        result += f" binary_split_predictions distribution: {density_str}"

    return result


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """Inverse of Option C sign-transform (sign * log1p(|y|))."""
    sign = np.sign(y_log)
    magnitude_log = np.abs(y_log)
    magnitude = np.expm1(magnitude_log)
    return sign * magnitude


def calculate_temporal_drift(segment_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Compare precision/recall across train/val/inference segments."""
    if not segment_metrics or len(segment_metrics) < 2:
        return {'drift_detected': False, 'interpretation': 'Insufficient segments for drift analysis'}

    segments_order = ['early', 'mid', 'late']
    available_segments = [s for s in segments_order if s in segment_metrics]

    if len(available_segments) < 2:
        return {'drift_detected': False, 'interpretation': 'Need at least 2 segments for drift analysis'}

    first_segment = segment_metrics[available_segments[0]]
    last_segment = segment_metrics[available_segments[-1]]

    precision_change = last_segment.get('precision', 0) - first_segment.get('precision', 0)
    recall_change = last_segment.get('recall', 0) - first_segment.get('recall', 0)
    f1_change = last_segment.get('f1', 0) - first_segment.get('f1', 0)

    precision_pct_change = (precision_change / max(first_segment.get('precision', 0.001), 0.001)) * 100

    if abs(precision_pct_change) > 20:
        interpretation = f"Significant precision drift: {precision_pct_change:+.1f}% change"
        drift_detected = True
    elif abs(precision_pct_change) > 10:
        interpretation = f"Moderate precision drift: {precision_pct_change:+.1f}% change"
        drift_detected = True
    else:
        interpretation = f"Stable precision: {precision_pct_change:+.1f}% change"
        drift_detected = False

    if precision_change < 0 and recall_change < 0:
        interpretation += " (performance degrading over time)"
    elif precision_change > 0 and recall_change > 0:
        interpretation += " (performance improving over time)"

    precision_values = [segment_metrics[s].get('precision', 0) for s in available_segments]
    stability_score = 1.0 - (np.std(precision_values) / max(np.mean(precision_values), 0.001)) if len(precision_values) > 1 else 1.0

    return {
        'drift_detected': drift_detected,
        'precision_change': float(precision_change),
        'recall_change': float(recall_change),
        'f1_change': float(f1_change),
        'precision_pct_change': float(precision_pct_change),
        'interpretation': interpretation,
        'stability_score': float(stability_score),
        'segments_analyzed': len(available_segments)
    }


def calculate_permutation_importance(model: Any, X: np.ndarray, y_true: np.ndarray,
                                     scoring_metric: str = 'precision',
                                     n_iterations: int = 5,
                                     pred_threshold: float = PREDICTION_THRESHOLD_DEFAULT,
                                     logger: Callable = None) -> Dict[int, float]:
    """Shuffle each feature, measure precision drop."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    try:
        y_pred = model.predict(X, verbose=0).flatten()
        y_binary = (y_pred >= pred_threshold).astype(int)

        if scoring_metric == 'precision':
            baseline = precision_score(y_true, y_binary, zero_division=0)
        elif scoring_metric == 'recall':
            baseline = recall_score(y_true, y_binary, zero_division=0)
        elif scoring_metric == 'f1':
            baseline = f1_score(y_true, y_binary, zero_division=0)
        elif scoring_metric == 'auc':
            baseline = roc_auc_score(y_true, y_pred)
        else:
            baseline = precision_score(y_true, y_binary, zero_division=0)

        importance = {}
        n_features = X.shape[1]

        for i in range(n_features):
            scores = []
            for _ in range(n_iterations):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])

                y_pred_perm = model.predict(X_permuted, verbose=0).flatten()
                y_binary_perm = (y_pred_perm >= pred_threshold).astype(int)

                if scoring_metric == 'precision':
                    score = precision_score(y_true, y_binary_perm, zero_division=0)
                elif scoring_metric == 'recall':
                    score = recall_score(y_true, y_binary_perm, zero_division=0)
                elif scoring_metric == 'f1':
                    score = f1_score(y_true, y_binary_perm, zero_division=0)
                elif scoring_metric == 'auc':
                    score = roc_auc_score(y_true, y_pred_perm)
                else:
                    score = precision_score(y_true, y_binary_perm, zero_division=0)

                scores.append(score)

            importance[i] = float(baseline - np.mean(scores))

        return importance

    except Exception as e:
        if logger:
            logger(f"Warning: Permutation importance calculation failed: {e}", 'warning')
        else:
            print(f"Warning: Permutation importance calculation failed: {e}")
        return {}


def calculate_prediction_entropy(predictions: np.ndarray) -> float:
    """Entropy of prediction distribution."""
    try:
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        entropy = -np.sum(predictions * np.log(predictions) + (1 - predictions) * np.log(1 - predictions))
        return float(entropy)
    except Exception as e:
        return 0.0


def calculate_logit_compression(predictions: np.ndarray) -> float:
    """Measure of prediction concentration."""
    try:
        mask = predictions > 0.1
        if mask.sum() == 0:
            return 0.0
        positive_preds = predictions[mask]
        max_pred = positive_preds.max()
        mean_pred = positive_preds.mean()
        return float(max_pred / max(mean_pred, 1e-10))
    except Exception as e:
        return 0.0


def calculate_ks_test(positive_preds: np.ndarray, negative_preds: np.ndarray) -> Dict[str, Any]:
    """Kolmogorov-Smirnov test: max separation between class predictions."""
    try:
        from scipy.stats import ks_2samp
        if len(positive_preds) < 2 or len(negative_preds) < 2:
            return {'ks_stat': 0.0, 'p_value': 1.0, 'interpretation': 'insufficient_data'}
        ks_stat, p_value = ks_2samp(positive_preds, negative_preds)
        if ks_stat > 0.5:
            interpretation = 'excellent_separation'
        elif ks_stat > 0.3:
            interpretation = 'good_separation'
        elif ks_stat > 0.1:
            interpretation = 'moderate_separation'
        else:
            interpretation = 'poor_separation'
        return {'ks_stat': float(ks_stat), 'p_value': float(p_value), 'interpretation': interpretation}
    except Exception as e:
        return {'ks_stat': 0.0, 'p_value': 1.0, 'interpretation': f'error: {e}'}


def calculate_bhattacharyya_distance(positive_preds: np.ndarray, negative_preds: np.ndarray) -> float:
    """Bhattacharyya distance between class prediction distributions."""
    try:
        if len(positive_preds) < 2 or len(negative_preds) < 2:
            return 0.0
        hist_pos, bin_edges = np.histogram(positive_preds, bins=20, density=True)
        hist_neg, _ = np.histogram(negative_preds, bins=bin_edges, density=True)
        hist_pos = hist_pos + 1e-10
        hist_neg = hist_neg + 1e-10
        bc = np.sum(np.sqrt(hist_pos * hist_neg))
        bh_dist = -np.log(bc)
        return float(bh_dist)
    except Exception as e:
        return 0.0


def calculate_mutual_information(predictions: np.ndarray, y_true: np.ndarray, threshold: float = PREDICTION_THRESHOLD_DEFAULT) -> float:
    """MI between predictions and labels."""
    try:
        from sklearn.metrics import mutual_info_score
        pred_binned = (predictions > threshold).astype(int)
        mi = mutual_info_score(y_true, pred_binned)
        return float(mi)
    except Exception as e:
        return 0.0
