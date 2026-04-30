"""
Chunk 12: Evaluation - Evaluator
Comprehensive model evaluation utility
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score


class Evaluator:
    """Comprehensive model evaluation utility with defensive programming"""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
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
            print(f"Warning: precision_score failed: {e}")
            # Fallback to manual calculation
            try:
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                return float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            except Exception as e2:
                print(f"Warning: Manual precision calculation failed: {e2}")
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
            print(f"Warning: recall_score failed: {e}")
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
            print(f"Warning: f1_score failed: {e}")
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
            print(f"Warning: roc_auc_score failed: {e}")
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
            print(f"Warning: confusion matrix calculation failed: {e}")
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
                print(f"Warning: y_true contains NaN/Inf values")
                return {'P': 0.0, 'R': 0.0, 'F1': 0.0, 'AUC': 0.0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            if not np.all(np.isfinite(y_pred_proba)):
                print(f"Warning: y_pred_proba contains NaN/Inf values")
                # Fix NaN/Inf instead of returning zeros
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0, posinf=1.0, neginf=0.0)
                y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            
            # SANITY CHECK: Ensure both classes exist in y_true
            unique_true = np.unique(y_true)
            if len(unique_true) < 2:
                print(f"Warning: Only one class present in y_true: {unique_true}")
            
            # Use PREDICTION_THRESHOLD from config (default 0.5) for converting predictions to binary
            pred_threshold = self.config.get('PREDICTION_THRESHOLD', 0.5)
            y_pred_binary = (y_pred_proba >= pred_threshold).astype(int)
            return {
                'P': self.calculate_precision(y_true, y_pred_binary),
                'R': self.calculate_recall(y_true, y_pred_binary),
                'F1': self.calculate_f1(y_true, y_pred_binary),
                'AUC': self.calculate_auc(y_true, y_pred_proba),
                'TP': int(np.sum((y_pred_binary == 1) & (y_true == 1))),
                'TN': int(np.sum((y_pred_binary == 0) & (y_true == 0))),
                'FP': int(np.sum((y_pred_binary == 1) & (y_true == 0))),
                'FN': int(np.sum((y_pred_binary == 0) & (y_true == 1))),
            }
        except Exception as e:
            print(f"Warning: evaluate_at_threshold failed: {e}")
            return {'P': 0.0, 'R': 0.0, 'F1': 0.0, 'AUC': 0.0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    def find_optimal_threshold(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               model, model_trainer, arch_name: str,
                               thresholds: np.ndarray, patience: int = 5,
                               retrain_model: bool = True) -> Tuple[float, float, List[Dict], Any]:
        """
        Train and evaluate model at multiple thresholds
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model: Pre-trained model (or None for fresh training)
            model_trainer: ModelTrainer instance
            arch_name: Name of architecture
            thresholds: Array of thresholds to test (decreasing order)
            patience: Early stopping patience
            retrain_model: If True, retrain model for each threshold. If False, use model for inference only (for POST-HPO).
        
        Returns:
            (optimal_threshold, best_precision, all_results, best_model)
            
        Returns:
            (optimal_threshold, best_precision, all_results)
        """
        best_thresh = thresholds[0]
        best_precision = 0.0
        best_trained_model = None  # Store the best trained model
        results = []
        no_improve_count = 0
        
        for thresh in thresholds:
            # Convert labels using threshold (labels >= threshold are positive)
            y_train_binary = (y_train >= thresh).astype(int)
            y_val_binary = (y_val >= thresh).astype(int)
            
            # Report class presence to catch UndefinedMetricWarning early
            train_class_0 = int(np.sum(y_train_binary == 0))
            train_class_1 = int(np.sum(y_train_binary == 1))
            val_class_0 = int(np.sum(y_val_binary == 0))
            val_class_1 = int(np.sum(y_val_binary == 1))
            
            train_status = "[OK]" if train_class_1 > 0 else "[WARNING]"
            val_status = "[OK]" if val_class_1 > 0 else "[WARNING]"
            
            print(f"   {arch_name} t={thresh:.1f} | Train: 0={train_class_0:,},1={train_class_1:,} {train_status} | Val: 0={val_class_0:,},1={val_class_1:,} {val_status}")
            
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
                        trained, _ = model_trainer._train_sklearn_model(model, X_train, y_train_binary)
                    else:
                        epochs = 15 if arch_name in ['Dense', 'VAE', 'CNN'] else 3
                        trained, _ = model_trainer.train_model(model, X_train, y_train_binary, epochs=epochs, verbose=0)
                else:
                    # No model provided - build fresh model (original behavior)
                    fresh_model = model_trainer.build_architecture(arch_name, X_train.shape[1])
                    if hasattr(fresh_model, 'sklearn_model'):
                        trained, _ = model_trainer._train_sklearn_model(fresh_model, X_train, y_train_binary)
                    else:
                        # Dense, VAE, and CNN need more epochs to learn effectively
                        epochs = 15 if arch_name in ['Dense', 'VAE', 'CNN'] else 3
                        trained, _ = model_trainer.train_model(fresh_model, X_train, y_train_binary, epochs=epochs, verbose=0)
            except Exception as e:
                print(f"Warning: Training failed for {arch_name} at threshold {thresh}: {e}")
                continue
            
            # Get predictions
            try:
                train_pred = trained.predict(X_train, verbose=0).flatten()
                val_pred = trained.predict(X_val, verbose=0).flatten()
                
                # SANITY CHECK: Validate and fix predictions
                if not np.all(np.isfinite(train_pred)):
                    print(f"Warning: train_pred contains NaN/Inf for {arch_name} at t={thresh}")
                    train_pred = np.nan_to_num(train_pred, nan=0.0, posinf=1.0, neginf=0.0)
                if not np.all(np.isfinite(val_pred)):
                    print(f"Warning: val_pred contains NaN/Inf for {arch_name} at t={thresh}")
                    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Additional safety: clip predictions to valid probability range
                train_pred = np.clip(train_pred, 1e-7, 1 - 1e-7)
                val_pred = np.clip(val_pred, 1e-7, 1 - 1e-7)
            except Exception as e:
                print(f"Warning: Prediction failed for {arch_name} at threshold {thresh}: {e}")
                continue
            
            # Calculate all metrics for train (compare binary predictions to binary labels)
            train_metrics = self.evaluate_at_threshold(y_train_binary, train_pred)
            
            # Calculate all metrics for validation (compare binary predictions to binary labels)
            val_metrics = self.evaluate_at_threshold(y_val_binary, val_pred)
            
            # Reject degenerate solutions: require minimum positive predictions
            # to prevent precision gaming (e.g., predicting almost nothing → artificially high P)
            # Dynamic calculation: max(MIN_POSITIVE_ABSOLUTE, n_samples * MIN_POSITIVE_PERCENTAGE)
            # Architecture-specific HPO thresholds (Apr 4, 2026)
            hpo_pct_config = self.config.get('HPO_MIN_POSITIVE_PERCENTAGE', {})
            hpo_abs_config = self.config.get('HPO_MIN_POSITIVE_ABSOLUTE', {})
            
            # Use architecture-specific if available, otherwise use defaults
            min_positive_percentage = hpo_pct_config.get(arch_name, self.config.get('MIN_POSITIVE_PERCENTAGE', 0.005))
            min_positive_absolute = hpo_abs_config.get(arch_name, self.config.get('MIN_POSITIVE_ABSOLUTE', 50))
            n_samples = len(y_val_binary)
            min_positive_predictions = max(min_positive_absolute, int(n_samples * min_positive_percentage))
            
            val_total_positive_preds = val_metrics['TP'] + val_metrics['FP']
            if val_total_positive_preds < min_positive_predictions:
                print(f"   [REJECT] Skipping threshold {thresh:.1f}: only {val_total_positive_preds} positive VALIDATION predictions (min={min_positive_predictions})")
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"   {arch_name}: Early stopping at threshold {thresh:.1f} (no improvement for {patience} iterations)")
                    break
                continue
            
            # Safeguard 2: Reject if positive prediction ratio is too extreme
            min_pos_ratio = self.config.get('MIN_POS_PRED_RATIO', 0.01)
            max_pos_ratio = self.config.get('MAX_POS_PRED_RATIO', 0.70)
            pos_pred_ratio = val_total_positive_preds / len(y_val_binary)
            if pos_pred_ratio < min_pos_ratio or pos_pred_ratio > max_pos_ratio:
                print(f"   [REJECT] Skipping t={thresh:.1f}: pos_pred_ratio={pos_pred_ratio:.2%} outside [{min_pos_ratio:.0%}, {max_pos_ratio:.0%}]")
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"   {arch_name}: Early stopping at threshold {thresh:.1f}")
                    break
                continue
            
            # Safeguard 3: Reject if precision doesn't beat baseline by minimum margin
            # Prevents convergence to degenerate solutions where P ≈ base rate
            baseline_precision = y_val_binary.mean()
            min_improvement = self.config.get('MIN_PRECISION_OVER_BASELINE', 0.05)
            current_precision = val_metrics['P']
            if current_precision <= baseline_precision + min_improvement:
                print(f"   [REJECT] Skipping t={thresh:.1f}: P={current_precision:.4f} <= baseline+{min_improvement:.4f}={baseline_precision+min_improvement:.4f}")
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"   {arch_name}: Early stopping at threshold {thresh:.1f}")
                    break
                continue
            
            # === PER-THRESHOLD DIAGNOSTICS ===
            val_pos_pct = (val_pred >= 0.5).mean() * 100
            print(f"   [DIAG] t={thresh:.1f}: pred_mean={val_pred.mean():.4f}, pred_std={val_pred.std():.4f}, pred_min={val_pred.min():.4f}, pred_max={val_pred.max():.4f}, %pos@0.5={val_pos_pct:.2f}%")
            
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
                print(f"   {arch_name}: Early stopping at threshold {thresh:.1f} (no improvement for {patience} iterations)")
                break
        
        return best_thresh, best_precision, results, best_trained_model
    
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
            print(f"Warning: Cross-validation failed: {e}")
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
    print("[PASS] Evaluator instance validated")
    
    # Create test data
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.15, 0.6, 0.55, 0.8, 0.9, 0.85, 0.3, 0.25])
    
    # Test precision
    precision = evaluator.calculate_precision(y_true, y_pred)
    print(f"[PASS] Precision: {precision:.4f}")
    
    # Test metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
    print(f"[PASS] Metrics: {metrics}")
    validate_evaluator_output(metrics)
    print("[PASS] Metrics output validated")
    
    # Test learning assessment
    loss_history = [0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30]
    prc_history = [0.5, 0.6, 0.7, 0.75, 0.78, 0.80]
    assessment = evaluator.assess_learning(loss_history, prc_history, 10)
    print(f"[PASS] Learning assessment: {assessment}")
    
    print("\n[PASS] All Evaluator tests passed")