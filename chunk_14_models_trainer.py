"""
Chunk 14: Models - Trainer
Model training orchestration and management
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, Any, List
from sklearn.model_selection import train_test_split


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
        if self.config.get('USE_FOCAL_LOSS', False):
            from tensorflow.keras.losses import BinaryFocalCrossentropy
            alpha = self.config.get('FOCAL_LOSS_ALPHA', 0.7)
            gamma = self.config.get('FOCAL_LOSS_GAMMA', 2.0)
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
        # Import architecture builders
        from chunk_08_models_base import (
            build_vae_model, build_cnn_model, build_rnn_model, build_lstm_model, build_dense_model
        )
        from chunk_09_models_advanced import (
            build_transformer_model, build_tabnet_model, build_gnn_sage_model,
            build_gnn_gat_model, build_hybrid_cnn_lstm_model, build_hybrid_transformer_gnn_model
        )
        from chunk_10_models_ensemble import (
            build_stacking_meta_model, build_bagging_random_forest_model,
            build_extra_trees_ensemble_model, build_boosting_adaptive_model
        )
        from chunk_11_models_sklearn import (
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
        from chunk_08_models_base import (
            build_vae_model, build_cnn_model, build_rnn_model, build_lstm_model, build_dense_model
        )
        from chunk_09_models_advanced import (
            build_transformer_model
        )
        from chunk_10_models_ensemble import (
            build_boosting_adaptive_model
        )
        from chunk_11_models_sklearn import (
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
                return builder(merged_config, input_dim, effective_loss_fn)
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
            return self._train_sklearn_model(model, X, y)
        
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
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, cw))
        
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
        from chunk_08_models_base import SamplingLayer
        sampling_layer = next((l for l in model.layers if isinstance(l, SamplingLayer)), None)
        if sampling_layer is not None:
            callbacks.append(KLAnnealingCallback(
                sampling_layer.kl_weight, warmup_epochs=10, max_kl_weight=1.0
            ))
        
        # Train model with increased patience for better convergence
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            sample_weight=sw_train,
            verbose=verbose,
            callbacks=callbacks,
        )
        
        return model, history.history
    
    def _train_sklearn_model(self, model_wrapper, X: np.ndarray, 
                            y: np.ndarray,
                            sample_weight: Optional[np.ndarray] = None) -> Tuple[Any, Dict]:
        """
        Train scikit-learn model
        
        Args:
            model_wrapper: SklearnModelWrapper instance
            X: Training features
            y: Training labels
            
        Returns:
            Tuple of (trained_model, dummy_history)
        """
        if sample_weight is not None:
            model_wrapper.fit(X, y, sample_weight=sample_weight)
        else:
            model_wrapper.fit(X, y)
        
        # Create dummy history for consistency
        history = {
            'loss': [0.5],  # Dummy values
            'val_loss': [0.5],
            'accuracy': [0.5],
            'val_accuracy': [0.5]
        }
        
        return model_wrapper, history
    
    def train_multiple_architectures(self, X: np.ndarray, y: np.ndarray,
                                    architecture_names: List[str],
                                    input_dim: int) -> List[Any]:
        """
        Train multiple architectures and return trained models
        
        Args:
            X: Training features
            y: Training labels
            architecture_names: List of architecture names to train
            input_dim: Input dimension
            
        Returns:
            List of trained models
        """
        trained_models = []
        
        for arch_name in architecture_names:
            print(f"Training {arch_name}...")
            try:
                model = self.build_architecture(arch_name, input_dim)
                
                # Check if sklearn model
                if hasattr(model, 'sklearn_model'):
                    trained_model, _ = self._train_sklearn_model(model, X, y)
                else:
                    trained_model, _ = self.train_model(model, X, y)
                
                trained_models.append(trained_model)
                print(f"[pass] {arch_name} trained successfully")
                
            except Exception as e:
                print(f"[error] {arch_name} training failed: {e}")
                continue
        
        return trained_models


def validate_training_output(model: Any, history: Dict, X: np.ndarray, 
                            y: np.ndarray) -> bool:
    """
    Ensure training produced valid results
    
    Args:
        model: Trained model
        history: Training history
        X: Features
        y: Labels
        
    Returns:
        True if valid
    """
    assert model is not None, "Model cannot be None after training"
    assert isinstance(history, dict), "History must be dict"
    assert 'loss' in history, "History must contain loss"
    
    # Test model can predict
    sample_size = min(5, len(X))
    X_sample = X[:sample_size]
    
    try:
        if hasattr(model, 'sklearn_model'):
            predictions = model.predict(X_sample)
        elif hasattr(model, 'predict'):
            predictions = model.predict(X_sample)
        else:
            raise AssertionError("Model has no predict method")
        
        assert len(predictions) == sample_size, f"Prediction length mismatch: {len(predictions)} vs {sample_size}"
        
    except Exception as e:
        raise AssertionError(f"Model failed to predict: {e}")
    
    return True


def validate_trainer_instance(trainer: ModelTrainer) -> bool:
    """
    Ensure ModelTrainer has required dependencies
    
    Args:
        trainer: ModelTrainer instance
        
    Returns:
        True if valid
    """
    assert trainer.evaluator is not None, "ModelTrainer requires Evaluator"
    assert hasattr(trainer, 'build_architecture'), "Missing build_architecture method"
    assert hasattr(trainer, 'train_model'), "Missing train_model method"
    assert hasattr(trainer, '_train_sklearn_model'), "Missing _train_sklearn_model method"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing ModelTrainer...")
    
    from chunk_12_evaluation_evaluator import Evaluator
    from chunk_02_utils_logging import Logger
    
    config = {
        'latent_dim': 32,
        'units': 64,
        'dropout': 0.1,
        'cnn_filters': 64,
        'lstm_units': 32,
        'heads': 4,
        'dim': 64,
        'MIN_ENSEMBLE_SIZE': 5
    }
    
    evaluator = Evaluator(config)
    logger = Logger({'LOG_VERBOSITY': 0})
    trainer = ModelTrainer(config, logger=logger, evaluator=evaluator)
    
    # Validate instance
    validate_trainer_instance(trainer)
    print("[pass] Trainer instance validated")
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    
    # Test building architecture
    print("\nTesting architecture building...")
    model = trainer.build_architecture('Dense', 10)
    print(f"[pass] Model built: {type(model)}")
    
    # Test training
    print("\nTesting model training...")
    trained_model, history = trainer.train_model(
        model, X, y, epochs=5, batch_size=16, verbose=0
    )
    validate_training_output(trained_model, history, X, y)
    print(f"[pass] Model trained, history keys: {list(history.keys())}")
    
    print("\n[pass] All ModelTrainer tests passed")