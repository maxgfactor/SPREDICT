"""
Chunk 12: Evaluation - Evaluator
Comprehensive model evaluation utility
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, cohen_kappa_score, roc_curve
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler  # NN-only normalization (trees are scale-invariant)
from chunk_01_config import PREDICTION_THRESHOLD_DEFAULT


class Evaluator:
    """Comprehensive model evaluation utility with defensive programming"""
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger
    
    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision score with comprehensive error handling
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Precision score (0.0 if fails)
        """
        try:
            return precision_score(y_true, y_pred, average='binary', zero_division=1.0)
        except Exception as e:
            if self.logger: self.logger.log(f"precision_score failed: {e}", 'warning')
            # Fallback to manual calculation
            try:
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                return float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            except Exception as e2:
                if self.logger: self.logger.log(f"Manual precision calculation failed: {e2}", 'warning')
                return 0.0
    
    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall score with comprehensive error handling
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Recall score (0.0 if fails)
        """
        try:
            return recall_score(y_true, y_pred, average='binary', zero_division=1.0)
        except Exception as e:
            if self.logger: self.logger.log(f"recall_score failed: {e}", 'warning')
            return 0.0
    
    def calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate F1 score with comprehensive error handling
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (0.0 if fails)
        """
        try:
            return f1_score(y_true, y_pred, average='binary', zero_division=1.0)
        except Exception as e:
            if self.logger: self.logger.log(f"f1_score failed: {e}", 'warning')
            return 0.0
    
    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        try:
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_specificity failed: {e}", 'warning')
            return 0.0
    
    def calculate_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate False Positive Rate"""
        try:
            return 1 - self.calculate_specificity(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_fpr failed: {e}", 'warning')
            return 0.0
    
    def calculate_f2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F2 score (beta=2)"""
        try:
            from sklearn.metrics import fbeta_score
            return fbeta_score(y_true, y_pred, beta=2.0, zero_division=1.0)
        except Exception as e:
            if self.logger: self.logger.log(f"f2_score calculation failed: {e}", 'warning')
            return 0.0
    
    def calculate_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Brier score for probability calibration (lower=better)"""
        try:
            return brier_score_loss(y_true, y_proba)
        except Exception as e:
            if self.logger: self.logger.log(f"brier_score_loss failed: {e}", 'warning')
            return 0.0
    
    def calculate_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cohen's kappa inter-rater reliability"""
        try:
            return cohen_kappa_score(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"cohen_kappa_score failed: {e}", 'warning')
            return 0.0
    
    def calculate_informedness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Informedness = Recall + Specificity - 1"""
        try:
            r = self.calculate_recall(y_true, y_pred)
            spec = self.calculate_specificity(y_true, y_pred)
            return r + spec - 1.0
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_informedness failed: {e}", 'warning')
            return 0.0
    
    def calculate_markedness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Markedness = Precision + Specificity - 1"""
        try:
            p = self.calculate_precision(y_true, y_pred)
            spec = self.calculate_specificity(y_true, y_pred)
            return p + spec - 1.0
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_markedness failed: {e}", 'warning')
            return 0.0
    
    def calculate_gini(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Gini coefficient = 2 * AUC - 1"""
        try:
            auc = self.calculate_auc(y_true, y_scores)
            return max(0, 2 * auc - 1)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_gini failed: {e}", 'warning')
            return 0.0
    
    def calculate_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Find threshold that maximizes Youden's J"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            return float(thresholds[optimal_idx]) if len(thresholds) > 0 else 0.5
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_optimal_threshold failed: {e}", 'warning')
            return 0.5
    
    def calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Matthews Correlation Coefficient"""
        try:
            return matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_mcc failed: {e}", 'warning')
            return 0.0
    
    def calculate_average_precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Precision-Recall AUC (Average Precision)"""
        try:
            return average_precision_score(y_true, y_scores)
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_average_precision failed: {e}", 'warning')
            return 0.0
    
    def calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Balanced Accuracy = (Recall + Specificity) / 2"""
        try:
            r = self.calculate_recall(y_true, y_pred)
            spec = self.calculate_specificity(y_true, y_pred)
            return (r + spec) / 2.0
        except Exception as e:
            if self.logger: self.logger.log(f"calculate_balanced_accuracy failed: {e}", 'warning')
            return 0.0
    
    def calculate_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Calculate AUC score with comprehensive error handling
        
        Args:
            y_true: True labels
            y_scores: Prediction probabilities or scores
            
        Returns:
            AUC score (0.0 if fails)
        """
        try:
            return float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            if self.logger: self.logger.log(f"roc_auc_score failed: {e}", 'warning')
            return 0.0
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        Calculate confusion matrix values
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
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
        Calculate all metrics using PREDICTION_THRESHOLD for binarizing predictions
        
        Args:
            y_true: True binary labels (already binarized using label threshold)
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with P, R, F1, AUC, TP, TN, FP, FN
        """
        try:
            # SANITY CHECK: Validate inputs
            if not np.all(np.isfinite(y_true)):
                if self.logger: self.logger.log(f"y_true contains NaN/Inf values", 'warning')
                return {
                    'P': 0.0, 'R': 0.0, 'F1': 0.0, 'AUC': 0.0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                    'Spec': 0.0, 'FPR': 0.0, 'F2': 0.0, 'MCC': 0.0, 'PRAUC': 0.0, 'BalAcc': 0.0, 'Brier': 0.0, 'Kappa': 0.0,
                    'Informedness': 0.0, 'Markedness': 0.0, 'Gini': 0.0, 'OptThresh': 0.0, 'MaxPred': 0.0, 'MeanPred': 0.0, 'StdPred': 0.0, 'PctAboveThresh': 0.0
                }
            if not np.all(np.isfinite(y_pred_proba)):
                if self.logger: self.logger.log(f"y_pred_proba contains NaN/Inf values", 'warning')
                # Fix NaN/Inf instead of returning zeros
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0, posinf=1.0, neginf=0.0)
                y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            
            # SANITY CHECK: Ensure both classes exist in y_true
            unique_true = np.unique(y_true)
            if len(unique_true) < 2:
                if self.logger: self.logger.log(f"Only one class present in y_true: {unique_true}", 'warning')
            
            # Use PREDICTION_THRESHOLD from config (default 0.5) for converting predictions to binary
            pred_threshold = self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)
            y_pred_binary = (y_pred_proba >= pred_threshold).astype(int)
            
            # Calculate prediction distribution metrics
            max_pred = float(y_pred_proba.max()) if len(y_pred_proba) > 0 else 0.0
            mean_pred = float(y_pred_proba.mean()) if len(y_pred_proba) > 0 else 0.0
            std_pred = float(y_pred_proba.std()) if len(y_pred_proba) > 0 else 0.0
            pct_above = float(np.mean(y_pred_binary)) * 100.0 if len(y_pred_binary) > 0 else 0.0
            
            # Calculate additional metrics (16 metrics - full set)
            spec = self.calculate_specificity(y_true, y_pred_binary)
            fpr = self.calculate_fpr(y_true, y_pred_binary)
            f2 = self.calculate_f2_score(y_true, y_pred_binary)
            mcc = self.calculate_mcc(y_true, y_pred_binary)
            prauc = self.calculate_average_precision(y_true, y_pred_proba)
            balacc = self.calculate_balanced_accuracy(y_true, y_pred_binary)
            brier = self.calculate_brier_score(y_true, y_pred_proba)
            kappa = self.calculate_kappa(y_true, y_pred_binary)
            informedness = self.calculate_informedness(y_true, y_pred_binary)
            markedness = self.calculate_markedness(y_true, y_pred_binary)
            gini = self.calculate_gini(y_true, y_pred_proba)
            opt_thresh = self.calculate_optimal_threshold(y_true, y_pred_proba)
            
            return {
                # Core metrics (8)
                'P': self.calculate_precision(y_true, y_pred_binary),
                'R': self.calculate_recall(y_true, y_pred_binary),
                'F1': self.calculate_f1(y_true, y_pred_binary),
                'AUC': self.calculate_auc(y_true, y_pred_proba),
                'TP': int(np.sum((y_pred_binary == 1) & (y_true == 1))),
                'TN': int(np.sum((y_pred_binary == 0) & (y_true == 0))),
                'FP': int(np.sum((y_pred_binary == 1) & (y_true == 0))),
                'FN': int(np.sum((y_pred_binary == 0) & (y_true == 1))),
                # Additional metrics (16 - full standard set)
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
        Train and evaluate model at multiple thresholds
        
        Args:
            X_train, y_train: Training data (all features)
            X_val, y_val: Validation data (all features)
            model: Pre-trained model (or None for fresh training)
            model_trainer: ModelTrainer instance
            arch_name: Name of architecture
            thresholds: Array of thresholds to test (decreasing order)
            patience: Early stopping patience
            retrain_model: If True, retrain model for each threshold. If False, use model for inference only (for POST-HPO).
            threshold_feature_indices: Dict mapping threshold -> kept feature indices for per-threshold pruning
        
        Returns:
            (optimal_threshold, best_precision, all_results, best_model)
        """
        best_thresh = thresholds[0]
        best_precision = 0.0
        best_trained_model = None  # Store the best trained model
        last_trained_model = None  # Track last trained model even if rejected by safeguards (fallback for architectures with weak default params)
        results = []
        no_improve_count = 0
        
        for thresh in thresholds:
            # Select features for this threshold
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
            
            # Normalize features for NNs only (trees scale-invariant). Applied at caller level,
            # NOT inside train_model(), to keep fit() and predict() data consistent.
            if arch_name in ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']:
                scaler = StandardScaler()
                X_train_t = scaler.fit_transform(X_train_t)
                X_val_t = scaler.transform(X_val_t)
            
            # Convert labels using threshold (labels >= threshold are positive)
            y_train_binary = (y_train >= thresh).astype(int)
            y_val_binary = (y_val >= thresh).astype(int)
            
            # Report class presence to catch UndefinedMetricWarning early
            train_class_0 = int(np.sum(y_train_binary == 0))
            train_class_1 = int(np.sum(y_train_binary == 1))
            val_class_0 = int(np.sum(y_val_binary == 0))
            val_class_1 = int(np.sum(y_val_binary == 1))
            
            train_status = "[ok]" if train_class_1 > 0 else "[warning]"
            val_status = "[ok]" if val_class_1 > 0 else "[warning]"
            
            arch_tag = f"[{arch_name.upper()}]"
            if self.logger: self.logger.log(f"{arch_tag} LABEL_THRESHOLD={thresh:.1f} | Train: 0={train_class_0:,},1={train_class_1:,} {train_status} | Val: 0={val_class_0:,},1={val_class_1:,} {val_status}", 'info')
            
            # Use model for inference only (no retraining) - for POST-HPO threshold search
            # Or train model for each threshold - for Section 2 pre-HPO threshold search
            try:
                if model is not None and not retrain_model:
                    # POST-HPO: Use pre-trained model for inference only (no retraining)
                    trained = model
                elif model is not None:
                    # Section 2: Use provided model but retrain with new labels
                    # Just train with the new labels
                    if hasattr(model, 'sklearn_model'):
                        trained, _ = model_trainer._train_sklearn_model(model, X_train_t, y_train_binary)
                    else:
                        epochs = 15 if arch_name in ['Dense', 'VAE', 'CNN'] else 3
                        trained, _ = model_trainer.train_model(model, X_train_t, y_train_binary, epochs=epochs, verbose=0)
                else:
                    # No model provided - build fresh model (original behavior)
                    fresh_model = model_trainer.build_architecture(arch_name, X_train_t.shape[1])
                    if hasattr(fresh_model, 'sklearn_model'):
                        trained, _ = model_trainer._train_sklearn_model(fresh_model, X_train_t, y_train_binary)
                    else:
                        # Dense, VAE, and CNN need more epochs to learn effectively
                        epochs = 15 if arch_name in ['Dense', 'VAE', 'CNN'] else 3
                        trained, _ = model_trainer.train_model(fresh_model, X_train_t, y_train_binary, epochs=epochs, verbose=0)
            except Exception as e:
                if self.logger: self.logger.log(f"Training failed for {arch_name} at threshold {thresh}: {e}", 'warning')
                continue
            last_trained_model = trained  # Always capture last-trained model regardless of safeguard acceptance
            
            # Get predictions
            try:
                train_pred = trained.predict(X_train_t, verbose=0).flatten()
                val_pred = trained.predict(X_val_t, verbose=0).flatten()
                
                # SANITY CHECK: Validate and fix predictions
                if not np.all(np.isfinite(train_pred)):
                    if self.logger: self.logger.log(f"{arch_tag} train_pred contains NaN/Inf at LABEL_THRESHOLD={thresh}", 'warning')
                    train_pred = np.nan_to_num(train_pred, nan=0.0, posinf=1.0, neginf=0.0)
                if not np.all(np.isfinite(val_pred)):
                    if self.logger: self.logger.log(f"{arch_tag} val_pred contains NaN/Inf at LABEL_THRESHOLD={thresh}", 'warning')
                    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Additional safety: clip predictions to valid probability range
                train_pred = np.clip(train_pred, 1e-7, 1 - 1e-7)
                val_pred = np.clip(val_pred, 1e-7, 1 - 1e-7)
            except Exception as e:
                if self.logger: self.logger.log(f"Prediction failed for {arch_name} at threshold {thresh}: {e}", 'warning')
                continue
            
            # Calculate all metrics for train (compare binary predictions to binary labels)
            train_metrics = self.evaluate_at_threshold(y_train_binary, train_pred)
            
            # Calculate all metrics for validation (compare binary predictions to binary labels)
            val_metrics = self.evaluate_at_threshold(y_val_binary, val_pred)
            
            # Reject degenerate solutions: require minimum positive predictions
            # to prevent precision gaming (e.g., predicting almost nothing → artificially high P)
            # Dynamic calculation: max(MIN_POSITIVE_ABSOLUTE, n_samples * MIN_POSITIVE_PERCENTAGE)
            # Architecture-specific thresholds (May 6, 2026)
            if arch_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                sklearn_safeguards = self.config.get('SKLEARN_SAFEGUARDS', {})
                min_positive_percentage = sklearn_safeguards.get('MIN_POSITIVE_PERCENTAGE', self.config.get('MIN_POSITIVE_PERCENTAGE', 0.005))
                min_positive_absolute = sklearn_safeguards.get('MIN_POSITIVE_ABSOLUTE', self.config.get('MIN_POSITIVE_ABSOLUTE', 50))
            elif arch_name in ['VAE', 'Dense', 'CNN', 'RNN', 'LSTM', 'Transformer']:
                neural_safeguards = self.config.get('NEURAL_SAFEGUARDS', {})
                min_positive_percentage = neural_safeguards.get('MIN_POSITIVE_PERCENTAGE', self.config.get('MIN_POSITIVE_PERCENTAGE', 0.005))
                min_positive_absolute = neural_safeguards.get('MIN_POSITIVE_ABSOLUTE', self.config.get('MIN_POSITIVE_ABSOLUTE', 50))
                patience = neural_safeguards.get('PATIENCE', self.config.get('PATIENCE', 5))
            else:
                min_positive_percentage = self.config.get('MIN_POSITIVE_PERCENTAGE', 0.005)
                min_positive_absolute = self.config.get('MIN_POSITIVE_ABSOLUTE', 50)
                patience = self.config.get('PATIENCE', 5)
            n_samples = len(y_val_binary)
            min_positive_predictions = max(min_positive_absolute, int(n_samples * min_positive_percentage))
            
            val_total_positive_preds = val_metrics['TP'] + val_metrics['FP']
            if val_total_positive_preds < min_positive_predictions:
                if self.logger: self.logger.log(f"[reject] Skipping threshold {thresh:.1f}: only {val_total_positive_preds} positive VALIDATION predictions (min={min_positive_predictions})", 'info')
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.logger: self.logger.log(f"{arch_name}: Early stopping at threshold {thresh:.1f} (no improvement for {patience} iterations)", 'info')
                    break
                continue
            
            # Safeguard 2: Reject if positive prediction ratio is too extreme
            min_pos_ratio = self.config.get('MIN_POS_PRED_RATIO', 0.01)
            max_pos_ratio = self.config.get('MAX_POS_PRED_RATIO', 0.70)
            pos_pred_ratio = val_total_positive_preds / len(y_val_binary)
            if pos_pred_ratio < min_pos_ratio or pos_pred_ratio > max_pos_ratio:
                if self.logger: self.logger.log(f"{arch_tag} [reject] skipping LABEL_THRESHOLD={thresh:.1f}: pos_pred_ratio={pos_pred_ratio:.2%} outside [{min_pos_ratio:.0%}, {max_pos_ratio:.0%}]", 'info')
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.logger: self.logger.log(f"{arch_tag} Early stopping at threshold {thresh:.1f}", 'info')
                    break
                continue
            
            # Safeguard 3: Reject if precision doesn't beat baseline by minimum margin
            # Prevents convergence to degenerate solutions where P ≈ base rate
            baseline_precision = min(y_val_binary.mean(), 0.5)  # Cap at 0.5 to avoid degenerate rejections at high baseline prevalence (t=0.0)
            # Check for sklearn architecture-specific overrides
            if arch_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                sklearn_safeguards = self.config.get('SKLEARN_SAFEGUARDS', {})
                min_improvement = sklearn_safeguards.get('MIN_PRECISION_OVER_BASELINE', 0.05)
            else:
                min_improvement = self.config.get('MIN_PRECISION_OVER_BASELINE', 0.05)
            current_precision = val_metrics['P']
            if current_precision <= baseline_precision + min_improvement:
                if self.logger: self.logger.log(f"{arch_tag} [reject] skipping LABEL_THRESHOLD={thresh:.1f}: precision={current_precision:.4f} <= baseline+{min_improvement:.4f}={baseline_precision+min_improvement:.4f}", 'info')
                no_improve_count += 1
                if no_improve_count >= patience:
                    if self.logger: self.logger.log(f"{arch_name}: Early stopping at threshold {thresh:.1f}", 'info')
                    break
                continue
            
            # === PER-THRESHOLD DIAGNOSTICS ===
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
                best_trained_model = trained  # Store the best model
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                if self.logger: self.logger.log(f"{arch_name}: Early stopping at threshold {thresh:.1f} (no improvement for {patience} iterations)", 'info')
                break
        
        return best_thresh, best_precision, results, best_trained_model if best_trained_model is not None else last_trained_model
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        except:
            metrics['accuracy'] = 0.0
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=1.0)
        except:
            metrics['precision'] = 0.0
        
        try:
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=1.0)
        except:
            metrics['recall'] = 0.0
        
        try:
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=1.0)
        except:
            metrics['f1'] = 0.0
        
        # AUC (requires probabilities or at least 2 classes)
        try:
            if y_pred_proba is not None:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc'] = 0.0
        
        # Average Precision
        try:
            if y_pred_proba is not None:
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            else:
                metrics['average_precision'] = average_precision_score(y_true, y_pred)
        except:
            metrics['average_precision'] = 0.0
        
        # Matthews Correlation Coefficient (MCC)
        try:
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['mcc'] = 0.0
        
        # Specificity (True Negative Rate)
        try:
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['specificity'] = specificity
        except:
            metrics['specificity'] = 0.0
        
        # Balanced Accuracy
        try:
            recall = metrics.get('recall', 0.0)
            specificity = metrics.get('specificity', 0.0)
            metrics['balanced_accuracy'] = (recall + specificity) / 2.0
        except:
            metrics['balanced_accuracy'] = 0.0
        
        return metrics
    
    def assess_learning(self, loss_history: List[float], 
                       prc_history: Optional[List[float]], 
                       patience_epochs: int) -> Dict[str, Any]:
        """
        Analyze if the model actually learned during training
        
        Args:
            loss_history: Training loss history
            prc_history: Precision-Recall AUC history
            patience_epochs: Number of epochs to wait for improvement
            
        Returns:
            Learning assessment dictionary
        """
        issues = []
        
        if not loss_history or len(loss_history) < 5:
            return {'learned': False, 'issues': ['insufficient_history']}
        
        # Check loss convergence
        early_loss = np.mean(loss_history[:len(loss_history)//5])
        late_loss = np.mean(loss_history[-len(loss_history)//5:])
        loss_reduction = (early_loss - late_loss) / early_loss if early_loss > 0 else 0
        
        if loss_reduction < 0.05:
            issues.append("poor_convergence")
        
        # Check PRC improvement
        if prc_history and len(prc_history) > 1:
            prc_gain = prc_history[-1] - prc_history[0]
            min_improvement = 0.001 if len(prc_history) < 50 else 0.005
            if prc_gain < min_improvement:
                issues.append("poor_prc_gain")
        
        # Check for training instability
        if len(loss_history) >= 10:
            loss_std = np.std(loss_history[-10:])
            loss_mean = np.mean(loss_history[-10:])
            if loss_mean > 0 and loss_std / loss_mean > 0.1:
                issues.append("training_instability")
        
        return {'learned': len(issues) == 0, 'issues': issues}
    
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform cross-validation with adaptive folds and error handling
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Labels
            cv_folds: Number of CV folds (default: 3)
            
        Returns:
            Cross-validation results dictionary
        """
        if cv_folds is None:
            cv_folds = 3
        
        try:
            precision_scorer = 'precision_weighted'
            
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=precision_scorer)
            
            return {
                'cv_scores': scores.tolist(),
                'mean_score': float(np.mean(scores)) if len(scores) > 0 else 0.0,
                'std_score': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'folds': cv_folds
            }
        except Exception as e:
            if self.logger: self.logger.log(f"Cross-validation failed: {e}", 'warning')
            return {
                'cv_scores': [0.0],
                'mean_score': 0.0,
                'std_score': 0.0,
                'folds': cv_folds
            }


def validate_evaluator_instance(evaluator: Evaluator) -> bool:
    """
    Ensure Evaluator has all required methods
    
    Args:
        evaluator: Evaluator instance to validate
        
    Returns:
        True if valid
    """
    required_methods = [
        'calculate_precision', 'calculate_metrics',
        'assess_learning', 'cross_validate'
    ]
    
    for method in required_methods:
        assert hasattr(evaluator, method), f"Evaluator missing method: {method}"
        assert callable(getattr(evaluator, method)), f"Evaluator.{method} is not callable"
    
    return True


def validate_evaluator_output(metrics: Dict, required_keys: Optional[List[str]] = None) -> bool:
    """
    Ensure metrics dict has required keys
    
    Args:
        metrics: Metrics dictionary
        required_keys: List of required metric names
        
    Returns:
        True if valid
    """
    if required_keys is None:
        required_keys = ['precision', 'recall', 'f1', 'auc']
    
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"{key} must be numeric"
        assert 0 <= metrics[key] <= 1, f"{key} must be in [0,1], got {metrics[key]}"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing Evaluator...")
    
    config = {}
    evaluator = Evaluator(config)
    
    # Validate instance
    validate_evaluator_instance(evaluator)
    print("[pass] Evaluator instance validated")
    
    # Create test data
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.15, 0.6, 0.55, 0.8, 0.9, 0.85, 0.3, 0.25])
    
    # Test precision
    precision = evaluator.calculate_precision(y_true, y_pred)
    print(f"[pass] precision: {precision:.4f}")
    
    # Test metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
    print(f"[pass] Metrics: {metrics}")
    validate_evaluator_output(metrics)
    print("[pass] Metrics output validated")
    
    # Test learning assessment
    loss_history = [0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30]
    prc_history = [0.5, 0.6, 0.7, 0.75, 0.78, 0.80]
    assessment = evaluator.assess_learning(loss_history, prc_history, 10)
    print(f"[pass] Learning assessment: {assessment}")
    
    print("\n[pass] All Evaluator tests passed")