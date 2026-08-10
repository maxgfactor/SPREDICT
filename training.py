"""
training.py — Model Training
Refactored from chunk_14_models_trainer.py + chunk_21_hyperparam_optimizer.py (2026-08-07).
Model training orchestration and hyperparameter optimization.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, Any, List, NamedTuple, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

import optuna
from config import PREDICTION_THRESHOLD_DEFAULT
from models import SamplingLayer

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Section 1: Callbacks
# ============================================================================

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """Ramp KL weight from near-zero to max over warmup epochs to prevent posterior collapse."""
    def __init__(self, kl_weight_var, warmup_epochs=10, max_kl_weight=1.0):
        super().__init__()
        self.kl_weight_var = kl_weight_var
        self.warmup_epochs = warmup_epochs
        self.max_kl_weight = max_kl_weight

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            w = self.max_kl_weight * (epoch + 1) / self.warmup_epochs
        else:
            w = self.max_kl_weight
        self.kl_weight_var.assign(w)


# TrainResult namedtuple (replaces loose tuple returns from chunk_14)
class TrainResult(NamedTuple):
    model: Any
    history: Dict
    train_loss: float
    val_loss: float
    training_time: float


# ============================================================================
# Section 2: ModelTrainer
# ============================================================================

class ModelTrainer:
    """Orchestrates model training and architecture building"""
    
    def __init__(self, config: Dict, logger=None, evaluator=None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            logger: Logger instance (optional)
            evaluator: Evaluator instance (required)
        """
        self.config = config
        self.logger = logger
        self.evaluator = evaluator
        
        assert self.evaluator is not None, "Evaluator required for threshold optimization"
    
    def get_loss_function(self):
        """
        Get the loss function based on config settings.
        
        Returns:
            Loss function (string or BinaryFocalCrossentropy)
        """
        if self.config['USE_FOCAL_LOSS']:
            from tensorflow.keras.losses import BinaryFocalCrossentropy
            alpha = self.config['FOCAL_LOSS_ALPHA']
            gamma = self.config['FOCAL_LOSS_GAMMA']
            return BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)
        return 'binary_crossentropy'
    
    def build_architecture(self, arch_name: str, input_dim: int, y_train: np.ndarray = None) -> Any:
        """
        Build specific neural architecture
        
        Args:
            arch_name: Architecture name
            input_dim: Input dimension
            y_train: Training labels for dynamic class weight (optional, for sklearn models)
            
        Returns:
            Built model
        """
        # Import architecture builders from models.py
        from models import (
            build_vae_model, build_cnn_model, build_rnn_model, build_lstm_model, build_dense_model,
            build_transformer_model, build_tabnet_model, build_gnn_sage_model,
            build_gnn_gat_model, build_hybrid_cnn_lstm_model, build_hybrid_transformer_gnn_model,
            build_stacking_meta_model, build_bagging_random_forest_model,
            build_extra_trees_ensemble_model, build_boosting_adaptive_model,
            build_isolation_forest_model, build_oneclass_svm_model, build_svm_model,
            build_lightgbm_model, build_xgboost_model, build_catboost_model,
            calculate_dynamic_class_weight
        )
        
        # Architecture mapping
        builders = {
            'VAE': build_vae_model,
            'CNN': build_cnn_model,
            'RNN': build_rnn_model,
            'LSTM': build_lstm_model,
            'Dense': build_dense_model,
            'Transformer': build_transformer_model,
            'TabNet': build_tabnet_model,
            'GNN_SAGE': build_gnn_sage_model,
            'GNN_GAT': build_gnn_gat_model,
            'CNN_LSTM_Hybrid': build_hybrid_cnn_lstm_model,
            'Transformer_GNN_Hybrid': build_hybrid_transformer_gnn_model,
            'Stacking_Meta': build_stacking_meta_model,
            'Bagging_RandomForest': build_bagging_random_forest_model,
            'ExtraTrees_Ensemble': build_extra_trees_ensemble_model,
            'Boosting_Adaptive': build_boosting_adaptive_model,
            'Isolation_Forest': build_isolation_forest_model,
            'OneClass_SVM': build_oneclass_svm_model,
            'SVM': build_svm_model,
            'LightGBM': build_lightgbm_model,
            'XGBoost': build_xgboost_model,
            'CatBoost': build_catboost_model,
        }
        
        # Get builder
        builder = builders.get(arch_name)
        
        # Get loss function based on config
        loss_fn = self.get_loss_function()
        
        if builder is None:
            raise ValueError(f"Unknown architecture '{arch_name}'")
        
        try:
            # Handle sklearn models differently
            if arch_name in ['Isolation_Forest', 'OneClass_SVM', 'SVM', 
                           'Bagging_RandomForest', 'ExtraTrees_Ensemble', 'LightGBM',
                           'XGBoost', 'CatBoost']:
                return builder(self.config, input_dim, y_train)
            else:
                return builder(self.config, input_dim, loss_fn)
        except Exception as e:
            raise RuntimeError(f"Failed to build {arch_name}: {e}") from e
    
    def build_architecture_with_params(self, arch_name: str, input_dim: int, 
                                       hyperparams: Dict) -> tf.keras.Model:
        """
        Build architecture with custom hyperparameters
        
        Args:
            arch_name: Architecture name
            input_dim: Input dimension
            hyperparams: Dictionary of hyperparameters to override defaults
            
        Returns:
            Built model
        """
        from models import (
            build_vae_model, build_cnn_model, build_rnn_model, build_lstm_model, build_dense_model,
            build_transformer_model, build_boosting_adaptive_model,
            build_lightgbm_model, build_xgboost_model, build_catboost_model
        )
        
        builders = {
            'Dense': build_dense_model,
            'VAE': build_vae_model,
            'CNN': build_cnn_model,
            'RNN': build_rnn_model,
            'LSTM': build_lstm_model,
            'Transformer': build_transformer_model,
            'Boosting_Adaptive': build_boosting_adaptive_model,
            'LightGBM': build_lightgbm_model,
            'XGBoost': build_xgboost_model,
            'CatBoost': build_catboost_model,
        }
        
        builder = builders.get(arch_name)
        
        # Get loss function based on config
        loss_fn = self.get_loss_function()
        
        if builder is None:
            self.logger.log(f"Warning: Unknown architecture '{arch_name}', using fallback", 'warning')
            return build_dense_model(self.config, input_dim, loss_fn)
        
        try:
            # Merge config with hyperparams (hyperparams override config)
            merged_config = {**self.config, **hyperparams}
            
            # NEW: Handle loss_function parameter (binary_crossentropy vs focal_loss)
            loss_function = hyperparams.get('loss_function', 'binary_crossentropy')
            
            if loss_function == 'focal_loss':
                # Enable FocalLoss and merge alpha/gamma into FOCAL_LOSS_CONFIG
                arch_key = arch_name
                if 'FOCAL_LOSS_CONFIG' not in merged_config:
                    merged_config['FOCAL_LOSS_CONFIG'] = {}
                if arch_key not in merged_config['FOCAL_LOSS_CONFIG']:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key] = {'enabled': True}
                else:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key]['enabled'] = True
                
                # Use alpha/gamma from hyperparams or defaults
                if 'alpha' in hyperparams:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key]['alpha'] = hyperparams['alpha']
                else:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key]['alpha'] = merged_config['FOCAL_LOSS_CONFIG'][arch_key].get('alpha', 0.75)
                    
                if 'gamma' in hyperparams:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key]['gamma'] = hyperparams['gamma']
                else:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key]['gamma'] = merged_config['FOCAL_LOSS_CONFIG'][arch_key].get('gamma', 2.0)
                
                # Use binary_crossentropy as loss_fn (model builder will apply FocalLoss)
                effective_loss_fn = 'binary_crossentropy'
            else:
                # Disable FocalLoss for binary_crossentropy
                arch_key = arch_name
                if 'FOCAL_LOSS_CONFIG' not in merged_config:
                    merged_config['FOCAL_LOSS_CONFIG'] = {}
                if arch_key not in merged_config['FOCAL_LOSS_CONFIG']:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key] = {'enabled': False}
                else:
                    merged_config['FOCAL_LOSS_CONFIG'][arch_key]['enabled'] = False
                
                effective_loss_fn = 'binary_crossentropy'
            
            if arch_name in ['Boosting_Adaptive']:
                return builder(merged_config, input_dim)
            else:
                model = builder(merged_config, input_dim, effective_loss_fn)
                model._is_focal = (loss_function == 'focal_loss')
                return model
        except Exception as e:
            raise RuntimeError(f"Failed to build {arch_name} with params {hyperparams}: {e}") from e
    
    def train_model(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
                   validation_data: Optional[Tuple] = None,
                   epochs: int = 50, batch_size: int = 32,
                   verbose: int = 0,
                   sample_weight: Optional[np.ndarray] = None) -> Tuple[tf.keras.Model, Dict]:
        """
        Train TensorFlow/Keras model
        
        Args:
            model: Model to train
            X: Training features
            y: Training labels
            validation_data: Validation tuple (X_val, y_val)
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Tuple of (trained_model, history)
        """
        # Check if sklearn model - route to sklearn trainer
        if hasattr(model, 'sklearn_model'):
            return self._train_sklearn_model(model, X, y, validation_data=validation_data)
        
        # Create validation split if not provided
        if validation_data is None:
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                    X, y, sample_weight, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                sw_train = None
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data
            sw_train = sample_weight
        
        # Compute class weights for balanced training. Prevents (~99% negative) class bias.
        # Skip when focal_loss is active — focal loss handles class imbalance internally.
        # Combining both creates contradictory gradients (score modulation vs 259x weight).
        # When sample_weight is provided, merge class_weight into it — Keras rejects both separately.
        if getattr(model, '_is_focal', False):
            class_weight_dict = None
            self.logger.log(f"   class_weight=None (focal_loss active — skipping balanced class_weight)", 'info')
        elif sw_train is not None:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            cw = compute_class_weight('balanced', classes=classes, y=y_train)
            cw_map = {cls: w for cls, w in zip(classes, cw)}
            sw_train = sw_train * np.array([cw_map[y] for y in y_train])
            class_weight_dict = None
            self.logger.log(f"   class_weight=balanced merged into sample_weight (Keras rejects both simultaneously)", 'info')
        else:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            cw = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, cw))
            self.logger.log(f"   class_weight=balanced (no focal_loss, no sample_weight)", 'info')
        
        # Handle models that expect 3D input
        model_input_shape = getattr(model, 'input_shape', None)
        if model_input_shape is not None and len(model_input_shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Build callbacks
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
        ]
        
        # If VAE model (has SamplingLayer), inject KL annealing callback
        sampling_layer = next((l for l in model.layers if isinstance(l, SamplingLayer)), None)
        if sampling_layer is not None:
            callbacks.append(KLAnnealingCallback(
                sampling_layer.kl_weight, warmup_epochs=10, max_kl_weight=0.1
            ))
        
        # Train model with increased patience for better convergence
        # Only pass class_weight or sample_weight when non-None to avoid Keras validation
        fit_kwargs = {
            'validation_data': (X_val, y_val),
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
            'callbacks': callbacks,
        }
        if class_weight_dict is not None:
            fit_kwargs['class_weight'] = class_weight_dict
        if sw_train is not None:
            fit_kwargs['sample_weight'] = sw_train

        history = model.fit(X_train, y_train, **fit_kwargs)
        
        return model, history.history
    
    def _train_sklearn_model(self, model_wrapper, X: np.ndarray, 
                            y: np.ndarray,
                            sample_weight: Optional[np.ndarray] = None,
                            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[Any, Dict]:
        """
        Train scikit-learn model
        
        Args:
            model_wrapper: SklearnModelWrapper instance
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            validation_data: Optional (X_val, y_val) tuple for early stopping
            
        Returns:
            Tuple of (trained_model, dummy_history)
        """
        fit_kwargs = {}
        esr = self.config.get('TREE_EARLY_STOPPING_ROUNDS', 10)
        self.logger.log(f"   _train_sklearn_model: early_stopping_rounds={esr}, "
                        f"eval_set={'provided' if validation_data is not None else 'None'}", 'info')
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        if validation_data is not None:
            fit_kwargs['eval_set'] = [validation_data]
            if hasattr(model_wrapper, 'sklearn_model') and hasattr(model_wrapper.sklearn_model, 'set_params'):
                model_wrapper.sklearn_model.set_params(verbose=-1, early_stopping_rounds=esr)
        # Dynamic scale_pos_weight for XGBoost (threshold-dependent — recalculated per training call)
        if hasattr(model_wrapper, 'sklearn_model') and 'XGB' in type(model_wrapper.sklearn_model).__name__:
            pos = np.sum(y == 1)
            neg = np.sum(y == 0)
            if pos > 0:
                spw = neg / pos
            else:
                spw = 1.0
            model_wrapper.sklearn_model.set_params(scale_pos_weight=spw)
        model_wrapper.fit(X, y, **fit_kwargs)
        
        # Log tree model params for lever audit
        if hasattr(model_wrapper.sklearn_model, 'get_params'):
            p = model_wrapper.sklearn_model.get_params()
            mt = type(model_wrapper.sklearn_model).__name__
            if 'XGB' in mt or 'CatBoost' in mt:
                v = p.get('scale_pos_weight', 'N/A')
                self.logger.log(f"   scale_pos_weight={v} (model={mt})", 'info')
            if 'LGBM' in mt:
                v = p.get('class_weight', 'N/A')
                self.logger.log(f"   class_weight={v} (model={mt})", 'info')
        
        # Create dummy history for consistency
        history = {
            'loss': [0.5],  # Dummy values
            'val_loss': [0.5],
            'accuracy': [0.5],
            'val_accuracy': [0.5]
        }
        
        return model_wrapper, history


# ============================================================================
# Section 3: HyperparameterOptimizer
# ============================================================================

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
            Normal path returns a 6-tuple (best_params, best_model, best_precision,
            raw_best_model, raw_best_precision, raw_best_params). The frozen-XGBoost
            early-return path returns a 3-tuple (best_params, best_model, best_precision).
            Caller must branch on len() == 6 vs 3 (chunk_18:615-618).
        """
        space = self.search_space.get(arch_name, {})
        
        if not space:
            if self.logger: self.logger.log(f"   No search space defined for {arch_name}, using defaults", 'warning')
            return {}, None, 0.0
        
        arch_tag = f"[{arch_name.upper()}]"
        if self.logger: self.logger.log(f"[section 2] {arch_tag} [hyperparameter_optimization search] Running Bayesian optimization ({self.n_trials} trials)...", 'info')
        
        # XGBoost freeze gate (GIS §10) — skip HPO, use iter10 winning params
        frozen_params = self.config.get('XGBOOST_FROZEN_PARAMS', {})
        if frozen_params.get('skip_hpo', False) and arch_name == 'XGBoost':
            hyperparams = frozen_params.get('hyperparams', {}) or {}
            if hyperparams:
                self.logger.log(f"[section 2] {arch_tag} [frozen] Using iter10 winning hyperparams (skip_hpo=True)", 'info')
                model = model_builder(hyperparams)
                trained, _ = train_func(model, X_train, y_train, validation_data=(X_val, y_val))
                preds = trained.predict(X_val, verbose=0).flatten()
                binary = (preds >= pred_threshold).astype(int)
                precision = precision_score(y_val, binary, zero_division=1.0)
                return hyperparams, trained, precision
        
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
                self.best_trial_balanced_score = 0.0
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
                self.raw_best_model = None
                self.raw_best_precision = 0.0
                self.raw_best_params = {}
            
            def __call__(self, trial):
                try:
                    # Increment trial counter
                    self.trial_count = getattr(self, 'trial_count', 0) + 1
                    trial_number = self.trial_count
                    arch_tag = f"[{self.arch_name.upper()}]"
                    
                    hyperparams = {}
                    for param_name, param_values in self.space.items():
                        hyperparams[param_name] = trial.suggest_categorical(param_name, param_values)

                    
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
                    
                    # Reject degenerate near-constant predictions for all architectures.
                    # When std_pred < 0.005, the model predicts the same value for everything
                    # and any apparent precision is by chance (class-base-rate gaming).
                    if std_pred < 0.005:
                        if self.logger: self.logger.log(f"   TRIAL {trial_number}/{self.total_trials}: REJECTED - degenerate predictions std_pred={std_pred:.4f} < 0.005", 'warning')
                        return 0.0
                    
                    # Reject near-constant positive predictions at low label thresholds.
                    # At LT=0.0 (~50% base rate), a model predicting "always positive" achieves
                    # recall~1.0 with precision barely above base rate — no real discrimination.
                    # This catches the pattern without affecting models with legitimate high recall.
                    label_threshold = getattr(self, 'label_threshold', 20.0)
                    if label_threshold < 5.0 and recall > 0.95 and (precision - self.y_val.mean()) < 0.01:
                        if self.logger: self.logger.log(f"   TRIAL {trial_number}/{self.total_trials}: REJECTED - near-constant positive predictions at LT={label_threshold:.1f} (recall={recall:.4f}, precision={precision:.4f}, base_rate={self.y_val.mean():.4f})", 'warning')
                        return 0.0
                    
                    # TP-balanced objective for all NNs: maximize precision * log(TP + 1)
                    # This balances high precision with reasonable TP counts
                    if self.arch_name in ['Dense', 'CNN', 'RNN', 'LSTM', 'Transformer']:
                        balanced_score = precision * np.log(tp + 1 + 1e-6)
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
                        from evaluate import Evaluator
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
                    
                    # Also track by raw precision for Section 3 comparison
                    if precision > self.raw_best_precision:
                        self.raw_best_precision = precision
                        self.raw_best_model = trained
                        self.raw_best_params = hyperparams
                    
                    return balanced_score
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
        
        return best_params, best_model, best_precision, objective.raw_best_model, objective.raw_best_precision, objective.raw_best_params
    
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
