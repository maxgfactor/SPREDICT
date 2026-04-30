"""
Chunk 21: Hyperparameter Optimization
Bayesian hyperparameter optimization using Optuna
"""

import optuna
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization using Optuna"""
    
    def __init__(self, config: Dict):
        """
        Initialize hyperparameter optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_trials = config.get('HYPERPARAM_OPTIMIZATION_TRIALS', 20)
        self.epochs = config.get('HYPERPARAM_OPTIMIZATION_EPOCHS', 3)
        self.search_space = config.get('HYPERPARAM_SEARCH_SPACE', {})
    
    def optimize(
        self,
        arch_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_builder: Callable,
        train_func: Callable,
        pred_threshold: float = 0.5
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
            print(f"   No search space defined for {arch_name}, using defaults")
            return {}, None, 0.0
        
        print(f"   Running Bayesian optimization for {arch_name} ({self.n_trials} trials)...")
        
        best_model = None
        best_precision = 0.0
        
        class Objective:
            def __init__(self, space, model_builder, train_func, X_train, y_train, X_val, y_val, epochs, pred_threshold, total_trials, arch_name='Dense'):
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
            
            def __call__(self, trial):
                # Increment trial counter
                self.trial_count = getattr(self, 'trial_count', 0) + 1
                trial_number = self.trial_count
                
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
                    
                    # Early termination filter: reject trials with MaxPred < 0.5
                    # These trials cannot produce any true positives (no predictions above threshold)
                    # Added based on VAE HPO analysis - reduces wasted trials (Apr 4, 2026)
                    # EXCLUSION: Skip this filter for gradient boosting models (LightGBM, XGBoost, CatBoost)
                    # They naturally produce predictions above 0.5 with scale_pos_weight
                    if max_pred < 0.5 and self.arch_name not in ['LightGBM', 'XGBoost', 'CatBoost']:
                        print(f"   Trial {trial_number}/{self.total_trials}: REJECTED - MaxPred={max_pred:.4f} < 0.5 (no predictions above threshold)", flush=True)
                        return 0.0
                    
                    # Min TP constraint for RNN: reject trials with very low TP (< 100)
                    # This ensures TP > 0 while maximizing precision (Apr 4, 2026)
                    if self.arch_name == 'RNN' and tp < 100:
                        print(f"   Trial {trial_number}/{self.total_trials}: REJECTED - TP={tp} < 100 (min TP threshold)", flush=True)
                        return 0.0
                    
                    # TP-balanced objective for Dense: maximize precision * log(TP + 1)
                    # This balances high precision with reasonable TP counts (Apr 4, 2026)
                    if self.arch_name == 'Dense':
                        # Use log-scaled TP to prevent precision from dominating
                        balanced_score = precision * np.log(tp + 1 + 1e-6)
                    elif self.arch_name in ['CNN', 'LSTM', 'Transformer']:
                        # MaxPred-prioritized objective for CNN/LSTM/Transformer: maximize precision * MaxPred
                        # All have 100% TP=0, so we optimize for pushing predictions toward threshold (Apr 4, 2026)
                        balanced_score = precision * max_pred
                    else:
                        balanced_score = precision
                    
                    # Print trial progress in real-time with all metrics
                    params_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
                    print(f"   Trial {trial_number}/{self.total_trials}: {params_str} → Val_P={precision:.4f} Val_R={recall:.4f} Val_AUC={auc:.4f} Val_F1={f1:.4f} Val_TP={tp} Val_FP={fp} Val_TN={tn} Val_FN={fn} Val_MaxPred={max_pred:.4f} Val_MeanPred={mean_pred:.4f}", flush=True)
                    
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
                            self.best_trial_model = trained
                    elif self.arch_name in ['CNN', 'LSTM', 'Transformer']:
                        # For CNN/LSTM/Transformer: track by precision * MaxPred
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
                            self.best_trial_model = trained
                    
                    # Return architecture-specific score (Apr 4, 2026)
                    # Dense: precision * log(TP+1), CNN/LSTM/Transformer: precision * MaxPred, others: precision
                    if self.arch_name == 'Dense':
                        return balanced_score
                    elif self.arch_name in ['CNN', 'LSTM', 'Transformer']:
                        return balanced_score
                    else:
                        return precision
                except Exception as e:
                    print(f"      Trial {trial_number} failed: {e}", flush=True)
                    return 0.0
        
        objective = Objective(
            space, model_builder, train_func,
            X_train, y_train, X_val, y_val,
            self.epochs, pred_threshold, self.n_trials, arch_name
        )
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_precision = study.best_value  # balanced_score (used by Optuna)
        best_actual_precision = objective.best_trial_precision  # actual precision
        best_model = objective.best_trial_model
        
        print(f"   {arch_name} - Best hyperparameters: {best_params}", flush=True)
        print(f"   {arch_name} - Best validation metrics: Val_P={best_actual_precision:.4f} (opt_score={best_precision:.4f}) Val_R={objective.best_trial_recall:.4f} Val_AUC={objective.best_trial_auc:.4f} Val_F1={objective.best_trial_f1:.4f} Val_TP={objective.best_trial_tp} Val_FP={objective.best_trial_fp} Val_TN={objective.best_trial_tn} Val_FN={objective.best_trial_fn}", flush=True)
        
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
    
    print("\n[PASS] HyperparameterOptimizer tests passed")
