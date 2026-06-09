"""
Chunk 21: Hyperparameter Optimization
Bayesian hyperparameter optimization using Optuna
"""

import optuna
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from chunk_01_config import PREDICTION_THRESHOLD_DEFAULT

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization using Optuna"""
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger
        self.n_trials = config['HYPERPARAM_OPTIMIZATION_TRIALS']
        self.epochs = config['HYPERPARAM_OPTIMIZATION_EPOCHS']
        self.search_space = config['HYPERPARAM_SEARCH_SPACE']
    
    def optimize(
        self,
        arch_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_builder: Callable,
        train_func: Callable,
        pred_threshold: float = PREDICTION_THRESHOLD_DEFAULT,
        label_threshold: float = 20.0
    ) -> Tuple[Dict[str, Any], Any, float]:
        """
        Run Bayesian optimization to find best hyperparameters
        
        Args:
            arch_name: Architecture name (e.g., 'Dense', 'VAE')
            X_train: Training features
            y_train: Training labels (binary)
            X_val: Validation features
            y_val: Validation labels (binary)
            model_builder: Function to build model with given hyperparams
            train_func: Function to train model
            pred_threshold: Prediction threshold for binary conversion
            
        Returns:
            Tuple of (best_hyperparameters, best_trained_model, best_validation_precision)
        """
        space = self.search_space.get(arch_name, {})
        
        if not space:
            if self.logger: self.logger.log(f"   No search space defined for {arch_name}, using defaults", 'warning')
            return {}, None, 0.0
        
        arch_tag = f"[{arch_name.upper()}]"
        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search] Running Bayesian optimization ({self.n_trials} trials)...", 'info')
        
        best_model = None
        best_precision = 0.0
        
        class Objective:
            def __init__(self, config, space, model_builder, train_func, X_train, y_train, X_val, y_val, epochs, pred_threshold, total_trials, arch_name='Dense', logger=None, label_threshold=20.0):
                self.config = config
                self.logger = logger
                self.space = space
                self.model_builder = model_builder
                self.train_func = train_func
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
                self.epochs = epochs
                self.pred_threshold = pred_threshold
                self.total_trials = total_trials
                self.arch_name = arch_name
                self.label_threshold = label_threshold
                self.trial_count = 0
                self.best_trial_model = None
                self.best_trial_precision = 0.0
                self.best_trial_recall = 0.0
                self.best_trial_auc = 0.0
                self.best_trial_f1 = 0.0
                self.best_trial_tp = 0
                self.best_trial_fp = 0
                self.best_trial_tn = 0
                self.best_trial_fn = 0
                self.best_trial_max_pred = 0.0
                self.best_trial_mean_pred = 0.0
                self.best_trial_std_pred = 0.0
                self.best_trial_pct_above = 0.0
                self.best_trial_spec = 0.0
                self.best_trial_fpr = 0.0
                self.best_trial_f2 = 0.0
                self.best_trial_mcc = 0.0
                self.best_trial_prauc = 0.0
                self.best_trial_balacc = 0.0
                self.best_trial_brier = 0.0
                self.best_trial_kappa = 0.0
                self.best_trial_informedness = 0.0
                self.best_trial_markedness = 0.0
                self.best_trial_gini = 0.0
                self.best_trial_opt_thresh = 0.0
                self.best_trial_params = {}
            
            def __call__(self, trial):
                # Increment trial counter
                self.trial_count = getattr(self, 'trial_count', 0) + 1
                trial_number = self.trial_count
                arch_tag = f"[{self.arch_name.upper()}]"
                
                hyperparams = {}
                for param_name, param_values in self.space.items():
                    hyperparams[param_name] = trial.suggest_categorical(param_name, param_values)
                
                try:
                    model = self.model_builder(hyperparams)
                    trained, _ = self.train_func(
                        model, self.X_train, self.y_train, 
                        validation_data=(self.X_val, self.y_val),
                        epochs=self.epochs, verbose=0
                    )
                    
                    y_pred = trained.predict(self.X_val, verbose=0).flatten()
                    y_pred_binary = (y_pred >= self.pred_threshold).astype(int)
                    
                    # Calculate all metrics
                    precision = precision_score(self.y_val, y_pred_binary, zero_division=0)
                    recall = recall_score(self.y_val, y_pred_binary, zero_division=0)
                    auc = roc_auc_score(self.y_val, y_pred) if len(np.unique(self.y_val)) > 1 else 0.5
                    f1 = f1_score(self.y_val, y_pred_binary, zero_division=0)
                    
                    # Calculate confusion matrix components
                    tp = int(np.sum((y_pred_binary == 1) & (self.y_val == 1)))
                    fp = int(np.sum((y_pred_binary == 1) & (self.y_val == 0)))
                    tn = int(np.sum((y_pred_binary == 0) & (self.y_val == 0)))
                    fn = int(np.sum((y_pred_binary == 0) & (self.y_val == 1)))
                    
                    # Calculate prediction distribution metrics (for TP optimization)
                    max_pred = float(y_pred.max())
                    mean_pred = float(y_pred.mean())
                    std_pred = float(y_pred.std())
                    pct_above = float((y_pred >= self.pred_threshold).mean() * 100)
                    
                    # Min TP constraint for RNN: reject trials with very low TP (< 100).
                    # TP=100 ensures statistically meaningful positive rate (~0.06% of val set).
                    if self.arch_name == 'RNN' and tp < 100:
                        if self.logger: self.logger.log(f"   TRIAL {trial_number}/{self.total_trials}: REJECTED - true_positives={tp} < 100 (min true_positives threshold)", 'warning')
                        return 0.0
                    
                    # TP-balanced objective for Dense: maximize precision * log(TP + 1)
                    # This balances high precision with reasonable TP counts (Apr 4, 2026)
                    if self.arch_name == 'Dense':
                        # Use log-scaled TP to prevent precision from dominating
                        balanced_score = precision * np.log(tp + 1 + 1e-6)
                    elif self.arch_name in ['CNN', 'RNN', 'LSTM', 'Transformer']:
                        # MaxPred-prioritized objective for CNN/RNN/LSTM/Transformer: maximize precision * MaxPred
                        # All have 100% TP=0 or collapse risk, so we optimize for pushing predictions toward threshold (Apr 4, 2026)
                        balanced_score = precision * max_pred
                    else:
                        balanced_score = precision
                    
                    # Print trial progress in real-time with all metrics (16 metrics unified)
                    params_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
                    
                    # Calculate extended metrics for this trial
                    trial_spec = trial_fpr = trial_f2 = trial_mcc = 0.0
                    trial_prauc = trial_balacc = trial_brier = 0.0
                    trial_kappa = trial_informedness = trial_markedness = 0.0
                    trial_gini = trial_opt_thresh = 0.0
                    try:
                        trial_binary = (y_pred.flatten() >= self.pred_threshold).astype(int)
                        from chunk_12_evaluation_evaluator import Evaluator
                        evaluator = Evaluator(self.config)
                        trial_spec = evaluator.calculate_specificity(self.y_val, trial_binary)
                        trial_fpr = evaluator.calculate_fpr(self.y_val, trial_binary)
                        trial_f2 = evaluator.calculate_f2_score(self.y_val, trial_binary)
                        trial_mcc = evaluator.calculate_mcc(self.y_val, trial_binary)
                        trial_prauc = evaluator.calculate_average_precision(self.y_val, y_pred.flatten())
                        trial_balacc = evaluator.calculate_balanced_accuracy(self.y_val, trial_binary)
                        trial_brier = evaluator.calculate_brier_score(self.y_val, y_pred.flatten())
                        trial_kappa = evaluator.calculate_kappa(self.y_val, trial_binary)
                        trial_informedness = evaluator.calculate_informedness(self.y_val, trial_binary)
                        trial_markedness = evaluator.calculate_markedness(self.y_val, trial_binary)
                        trial_gini = evaluator.calculate_gini(self.y_val, y_pred.flatten())
                        trial_opt_thresh = evaluator.calculate_optimal_threshold(self.y_val, y_pred.flatten())
                        
                        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search baseline] LABEL_THRESHOLD={self.label_threshold:.1f}", 'info')
                        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search baseline] VALIDATION_PRECISION={precision:.4f} VALIDATION_TRUE_POSITIVES={tp} VALIDATION_TRUE_NEGATIVES={tn} validation_false_positives={fp} validation_false_negatives={fn} validation_max_prediction={max_pred:.4f} validation_mean_prediction={mean_pred:.4f} validation_recall={recall:.4f} validation_f1={f1:.4f} validation_auc={auc:.4f} validation_specificity={trial_spec:.4f} validation_false_positive_rate={trial_fpr:.4f} validation_f2={trial_f2:.4f} validation_mcc={trial_mcc:.4f} validation_prauc={trial_prauc:.4f} validation_balanced_accuracy={trial_balacc:.4f} validation_brier={trial_brier:.4f} validation_kappa={trial_kappa:.4f} validation_informedness={trial_informedness:.4f} validation_markedness={trial_markedness:.4f} validation_gini={trial_gini:.4f} validation_optimal_threshold={trial_opt_thresh:.4f} validation_standard_deviation_prediction={std_pred:.4f} validation_percentage_above_threshold={pct_above:.2f}", 'info')
                        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search baseline] TRIAL {trial_number}/{self.total_trials}: {params_str}", 'info')
                    except Exception as e:
                        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search] LABEL_THRESHOLD={self.label_threshold:.1f}", 'info')
                        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search] VALIDATION_PRECISION={precision:.4f} VALIDATION_TRUE_POSITIVES={tp} VALIDATION_TRUE_NEGATIVES={tn} validation_false_positives={fp} validation_false_negatives={fn} validation_max_prediction={max_pred:.4f} validation_mean_prediction={mean_pred:.4f} validation_recall={recall:.4f} validation_f1={f1:.4f} validation_auc={auc:.4f} validation_standard_deviation_prediction={std_pred:.4f} validation_percentage_above_threshold={pct_above:.2f}", 'info')
                        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search] TRIAL {trial_number}/{self.total_trials}: {params_str}", 'info')
                    
                    # Track best model - use architecture-specific balanced score (Apr 4, 2026)
                    if self.arch_name == 'Dense':
                        if balanced_score > getattr(self, 'best_trial_balanced_score', 0):
                            self.best_trial_balanced_score = balanced_score
                            self.best_trial_precision = precision
                            self.best_trial_recall = recall
                            self.best_trial_auc = auc
                            self.best_trial_f1 = f1
                            self.best_trial_tp = tp
                            self.best_trial_fp = fp
                            self.best_trial_tn = tn
                            self.best_trial_fn = fn
                            self.best_trial_max_pred = max_pred
                            self.best_trial_mean_pred = mean_pred
                            self.best_trial_std_pred = std_pred
                            self.best_trial_pct_above = pct_above
                            self.best_trial_spec = trial_spec
                            self.best_trial_fpr = trial_fpr
                            self.best_trial_f2 = trial_f2
                            self.best_trial_mcc = trial_mcc
                            self.best_trial_prauc = trial_prauc
                            self.best_trial_balacc = trial_balacc
                            self.best_trial_brier = trial_brier
                            self.best_trial_kappa = trial_kappa
                            self.best_trial_informedness = trial_informedness
                            self.best_trial_markedness = trial_markedness
                            self.best_trial_gini = trial_gini
                            self.best_trial_opt_thresh = trial_opt_thresh
                            self.best_trial_params = hyperparams
                            self.best_trial_model = trained
                    elif self.arch_name in ['CNN', 'RNN', 'LSTM', 'Transformer']:
                        # For CNN/RNN/LSTM/Transformer: track by precision * MaxPred
                        if balanced_score > getattr(self, 'best_trial_balanced_score', 0):
                            self.best_trial_balanced_score = balanced_score
                            self.best_trial_precision = precision
                            self.best_trial_recall = recall
                            self.best_trial_auc = auc
                            self.best_trial_f1 = f1
                            self.best_trial_tp = tp
                            self.best_trial_fp = fp
                            self.best_trial_tn = tn
                            self.best_trial_fn = fn
                            self.best_trial_max_pred = max_pred
                            self.best_trial_mean_pred = mean_pred
                            self.best_trial_std_pred = std_pred
                            self.best_trial_pct_above = pct_above
                            self.best_trial_spec = trial_spec
                            self.best_trial_fpr = trial_fpr
                            self.best_trial_f2 = trial_f2
                            self.best_trial_mcc = trial_mcc
                            self.best_trial_prauc = trial_prauc
                            self.best_trial_balacc = trial_balacc
                            self.best_trial_brier = trial_brier
                            self.best_trial_kappa = trial_kappa
                            self.best_trial_informedness = trial_informedness
                            self.best_trial_markedness = trial_markedness
                            self.best_trial_gini = trial_gini
                            self.best_trial_opt_thresh = trial_opt_thresh
                            self.best_trial_params = hyperparams
                            self.best_trial_model = trained
                    else:
                        if precision > self.best_trial_precision:
                            self.best_trial_precision = precision
                            self.best_trial_recall = recall
                            self.best_trial_auc = auc
                            self.best_trial_f1 = f1
                            self.best_trial_tp = tp
                            self.best_trial_fp = fp
                            self.best_trial_tn = tn
                            self.best_trial_fn = fn
                            self.best_trial_max_pred = max_pred
                            self.best_trial_mean_pred = mean_pred
                            self.best_trial_std_pred = std_pred
                            self.best_trial_pct_above = pct_above
                            self.best_trial_spec = trial_spec
                            self.best_trial_fpr = trial_fpr
                            self.best_trial_f2 = trial_f2
                            self.best_trial_mcc = trial_mcc
                            self.best_trial_prauc = trial_prauc
                            self.best_trial_balacc = trial_balacc
                            self.best_trial_brier = trial_brier
                            self.best_trial_kappa = trial_kappa
                            self.best_trial_informedness = trial_informedness
                            self.best_trial_markedness = trial_markedness
                            self.best_trial_gini = trial_gini
                            self.best_trial_opt_thresh = trial_opt_thresh
                            self.best_trial_params = hyperparams
                            self.best_trial_model = trained
                    
                    # Return architecture-specific score (Apr 4, 2026)
                    # Dense: precision * log(TP+1), CNN/RNN/LSTM/Transformer: precision * MaxPred, others: precision
                    if self.arch_name == 'Dense':
                        return balanced_score
                    elif self.arch_name in ['CNN', 'RNN', 'LSTM', 'Transformer']:
                        return balanced_score
                    else:
                        return precision
                except Exception as e:
                    if self.logger: self.logger.log(f"      TRIAL {trial_number} failed: {e}", 'warning')
                    return 0.0
        
        objective = Objective(
            self.config, space, model_builder, train_func,
            X_train, y_train, X_val, y_val,
            self.epochs, pred_threshold, self.n_trials, arch_name,
            logger=self.logger, label_threshold=label_threshold
        )
        
        # Custom loop for unlimited trials until target met (May 11, 2026)
        target_precision = self.config.get('HPO_TARGET_PRECISION', 0.60)
        continue_until_target = self.config['HPO_CONTINUE_UNTIL_TARGET']
        stagnation_threshold = self.config['HPO_STAGNATION_THRESHOLD']
        max_trials = 1000  # Safety cap (raised from 500, May 13, 2026)
        
        study = optuna.create_study(direction='maximize')
        
        trial_number = 0
        best_precision_seen = 0.0
        best_trial_number = 0
        no_improve_count = 0
        phase_start_precision = 0.0
        
        while True:
            trial_number += 1
            
            # Run one trial
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            
            # Get current best
            current_precision = objective.best_trial_precision
            current_tp = objective.best_trial_tp
            
            # Check: Target met?
            if current_precision >= target_precision and current_tp > 0:
                if self.logger: self.logger.log(f"   TARGET MET at trial {trial_number}: validation_precision={current_precision:.4f} >= {target_precision}", 'info')
                break
            
            # Progress logging every 10 trials
            if trial_number % 10 == 0:
                if self.logger: self.logger.log(f"   progress: TRIAL {trial_number} | validation_best_precision={current_precision:.4f} (trial {best_trial_number}) | target={target_precision}", 'info')
            
            # Check: Phase transition (reset stagnation every 30 trials)
            if trial_number % 30 == 0:
                # Phase transition - reset stagnation counter
                if current_precision > phase_start_precision:
                    phase_start_precision = current_precision
                no_improve_count = 0
            
            # Check: Stagnation?
            if current_precision > best_precision_seen:
                best_precision_seen = current_precision
                best_trial_number = trial_number
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Stop if stagnant
            if no_improve_count >= stagnation_threshold:
                if self.logger: self.logger.log(f"   stopped: No improvement for {stagnation_threshold} trials (validation_best_precision={best_precision_seen:.4f})", 'info')
                break
            
            # Continue if enabled and not target met
            if not continue_until_target and trial_number >= self.n_trials:
                break
            
            # Safety cap
            if trial_number >= max_trials:
                if self.logger: self.logger.log(f"   safety stop: Reached {max_trials} trials (validation_best_precision={best_precision_seen:.4f})", 'warning')
                break
        
        best_params = study.best_params
        best_precision = study.best_value  # balanced_score (used by Optuna)
        best_model = objective.best_trial_model

        arch_tag = f"[{arch_name.upper()}]"
        lt = objective.label_threshold
        o = objective
        if self.logger: self.logger.log(f"[section 2] {arch_tag} [best trial] LABEL_THRESHOLD={lt:.1f}", 'info')
        if self.logger: self.logger.log(f"[section 2] {arch_tag} [best trial] VALIDATION_PRECISION={o.best_trial_precision:.4f} VALIDATION_TRUE_POSITIVES={o.best_trial_tp} VALIDATION_TRUE_NEGATIVES={o.best_trial_tn} validation_false_positives={o.best_trial_fp} validation_false_negatives={o.best_trial_fn} validation_max_prediction={o.best_trial_max_pred:.4f} validation_mean_prediction={o.best_trial_mean_pred:.4f} validation_recall={o.best_trial_recall:.4f} validation_f1={o.best_trial_f1:.4f} validation_auc={o.best_trial_auc:.4f} validation_specificity={o.best_trial_spec:.4f} validation_false_positive_rate={o.best_trial_fpr:.4f} validation_f2={o.best_trial_f2:.4f} validation_mcc={o.best_trial_mcc:.4f} validation_prauc={o.best_trial_prauc:.4f} validation_balanced_accuracy={o.best_trial_balacc:.4f} validation_brier={o.best_trial_brier:.4f} validation_kappa={o.best_trial_kappa:.4f} validation_informedness={o.best_trial_informedness:.4f} validation_markedness={o.best_trial_markedness:.4f} validation_gini={o.best_trial_gini:.4f} validation_optimal_threshold={o.best_trial_opt_thresh:.4f} validation_standard_deviation_prediction={o.best_trial_std_pred:.4f} validation_percentage_above_threshold={o.best_trial_pct_above:.2f}", 'info')
        params_str = ", ".join([f"{k}={v}" for k, v in o.best_trial_params.items()])
        if self.logger: self.logger.log(f"[section 2] {arch_tag} [best trial] {params_str}", 'info')
        
        return best_params, best_model, best_precision
    
    def get_search_space_summary(self) -> str:
        """Get summary of search spaces for all architectures"""
        summary = []
        for arch, space in self.search_space.items():
            total_combinations = 1
            for param_values in space.values():
                total_combinations *= len(param_values)
            summary.append(f"  {arch}: {total_combinations:,} combinations")
        return "\n".join(summary)


def create_hyperparameter_optimizer(config: Dict) -> HyperparameterOptimizer:
    """
    Factory function to create hyperparameter optimizer
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HyperparameterOptimizer instance
    """
    return HyperparameterOptimizer(config)


if __name__ == "__main__":
    print("Testing HyperparameterOptimizer...")
    
    config = {
        'HYPERPARAM_OPTIMIZATION_TRIALS': 5,
        'HYPERPARAM_OPTIMIZATION_EPOCHS': 2,
        'HYPERPARAM_SEARCH_SPACE': {
            'Dense': {
                'units': [32, 64],
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001],
            },
        }
    }
    
    optimizer = HyperparameterOptimizer(config)
    print(f"Optimizer created successfully")
    print(f"Search space summary:\n{optimizer.get_search_space_summary()}")
    
    print("\n[pass] HyperparameterOptimizer tests passed")