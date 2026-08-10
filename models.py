"""
models.py — Model Builders
Refactored from chunk_08_models_base.py + chunk_09_models_advanced.py +
chunk_10_models_ensemble.py + chunk_11_models_sklearn.py (2026-08-07).
"""
# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import tensorflow as tf
import keras
from typing import Dict, List, Optional, Callable, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM, SVC

# ============================================================================
# Section 1: Loss Functions
# ============================================================================

@keras.saving.register_keras_serializable()
class FocalLoss:
    """
    Focal Loss for handling class imbalance in binary classification.
    
    Formula: FL(pt) = -α(1-pt)^γ log(pt)
    
    Args:
        alpha (float): Weight for positive class (0.5 = balanced)
        gamma (float): Focusing parameter (1.0 = standard, higher = focus on hard samples)
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 1.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weight for positive class (signal)
            gamma: Focusing parameter (reduces loss for easy samples)
        """
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
            
        Returns:
            Focal loss scalar value
        """
        import tensorflow as tf
        
        # Binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Prediction probability for true class
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Focal weight: (1 - pt)^γ
        focal_weight = tf.pow(1 - pt, self.gamma)
        
        # Apply focal weighting with alpha
        loss = tf.reduce_mean(self.alpha * focal_weight * bce)
        
        return loss
    
    def get_config(self):
        """Return configuration for serialization."""
        return {'alpha': self.alpha, 'gamma': self.gamma}
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)


# ============================================================================
# Section 2: Neural Architecture Builders
# ============================================================================

@keras.saving.register_keras_serializable(package='VAE')
class SamplingLayer(tf.keras.layers.Layer):
    """Reparameterization sampling with KL regularization for VAE (Keras 3 compatible).
    
    Uses an adjustable KL weight (kl_weight Variable) that a KLAnnealingCallback
    can ramp from near-zero to 1.0 over warmup epochs.
    """
    def __init__(self, initial_kl_weight=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.initial_kl_weight = initial_kl_weight
        self.kl_weight = self.add_weight(
            name='kl_weight',
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_kl_weight),
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.keras.ops.clip(z_log_var, -5.0, 2.0)
        epsilon = tf.keras.random.normal(shape=tf.keras.ops.shape(z_mean))
        z = z_mean + tf.keras.ops.exp(0.5 * z_log_var) * epsilon
        kl_loss = -0.5 * tf.keras.ops.mean(tf.keras.ops.sum(
            1 + z_log_var - tf.keras.ops.square(z_mean) - tf.keras.ops.exp(z_log_var), axis=-1))
        self.add_loss(self.kl_weight * kl_loss)
        return z

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config['initial_kl_weight'] = self.initial_kl_weight
        return config


@keras.saving.register_keras_serializable(package='VAE')
class VAEClassifier(tf.keras.Model):
    """VAE classifier with decoder + reconstruction loss and KL annealing.
    
    Architecture:
    - Encoder (N layers, configurable via encoder_layers): 256 → 128 → 64
    - Latent space: z_mean, z_log_var → sampling → z
    - Classifier (from sampled z): 64 → 32 → 1 (sigmoid)
    - Decoder (N layers, configurable via decoder_layers): 64 → 128 → 256 → input_dim (linear)
    - Total loss: classifier_loss + 0.1 * MSE_reconstruction + kl_weight * KL
    """
    
    def __init__(self, config: Dict, input_dim: int, loss_fn: str = 'binary_crossentropy'):
        super().__init__()
        self.input_dim = input_dim
        self.loss_fn = loss_fn
        self.latent_dim = config.get('latent_dim', 64)
        self.dropout_rate = config.get('dropout', 0.1)
        
        # Encoder - configurable depth via encoder_layers HPO param
        self.num_encoder_layers = config.get('encoder_layers', 2)
        dropout_rate = self.dropout_rate
        num_encoder_layers = self.num_encoder_layers
        encoder_widths = [256, 128, 64]
        self.encoder_blocks = []
        for i in range(min(num_encoder_layers, len(encoder_widths))):
            self.encoder_blocks.append([
                tf.keras.layers.Dense(encoder_widths[i], activation='relu', kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dropout_rate),
            ])
        
        # Latent space
        self.z_mean_dense = tf.keras.layers.Dense(self.latent_dim, name='z_mean')
        self.z_log_var_dense = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')
        self.sampling = SamplingLayer(name='z')
        
        # Classifier head from sampled latent
        self.clf_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.clf_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.clf_dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.clf_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.clf_output = tf.keras.layers.Dense(1, activation='sigmoid', name='signal_output')
        
        # Decoder - configurable depth via decoder_layers HPO param
        self.num_decoder_layers = max(config.get('decoder_layers', 2), 2)
        num_decoder_layers = self.num_decoder_layers
        decoder_widths = [64, 128, 256]
        self.decoder_blocks = []
        for i in range(min(num_decoder_layers, len(decoder_widths))):
            self.decoder_blocks.append([
                tf.keras.layers.Dense(decoder_widths[i], activation='relu', kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dropout_rate),
            ])
        self.dec_output = tf.keras.layers.Dense(self.input_dim, activation='linear', name='reconstruction')
    
    def call(self, inputs):
        # Encoder
        x = inputs
        for block in self.encoder_blocks:
            x = block[0](x)
            x = block[1](x)
            x = block[2](x)
        
        # Latent space parameters
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        
        # Sampling with KL regularization (loss added inside SamplingLayer.call)
        z = self.sampling([z_mean, z_log_var])
        
        # Classifier head
        c = self.clf_dense1(z)
        c = self.clf_dropout1(c)
        c = self.clf_dense2(c)
        c = self.clf_dropout2(c)
        classification = self.clf_output(c)
        
        # Decoder (reconstruction)
        d = z
        for block in self.decoder_blocks:
            d = block[0](d)
            d = block[1](d)
            d = block[2](d)
        reconstruction = self.dec_output(d)
        
        # Reconstruction loss (internal — not a model output)
        recon_loss = tf.keras.ops.mean(tf.keras.ops.square(inputs - reconstruction))
        self.add_loss(0.1 * recon_loss)
        
        return classification
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'loss_fn': self.loss_fn,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout_rate,
            'encoder_layers': self.num_encoder_layers,
            'decoder_layers': self.num_decoder_layers,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            config={
                'latent_dim': config.get('latent_dim', 64),
                'dropout': config.get('dropout', 0.1),
                'encoder_layers': config.get('encoder_layers', 2),
                'decoder_layers': config.get('decoder_layers', 3),
            },
            input_dim=config['input_dim'],
            loss_fn=config['loss_fn'],
        )


@keras.saving.register_keras_serializable(package='chunk_09_models_advanced', name='ExpandDimsLayer')
class ExpandDimsLayer(keras.layers.Layer):
    """Custom layer to expand dimensions - serializable without Lambda issues"""
    
    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config


def build_vae_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build VAE classifier with decoder + reconstruction loss and KL annealing.
    
    Architecture:
    - Encoder (N layers, configurable via encoder_layers): 256 → 128 → 64
    - Latent space: z_mean, z_log_var → sampling → z
    - Classifier (from sampled z): 64 → 32 → 1 (sigmoid)
    - Decoder (N layers, configurable via decoder_layers): 64 → 128 → 256 → input_dim (linear)
    - Total loss: classifier_loss + 0.1 * MSE_reconstruction + kl_weight * KL
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (number of features)
        loss: Loss function for classifier (default: binary_crossentropy)
        
    Returns:
        Compiled VAE model (single-output: signal prediction)
    """
    model = VAEClassifier(config, input_dim, loss)
    
    # Build all sub-layers via dry-run forward pass on a symbolic input
    dummy_input = tf.keras.Input(shape=(input_dim,))
    model(dummy_input)
    
    # Use focal loss if configured for this architecture (per-arch config)
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('VAE', {})
    default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('VAE', 0.0005)
    if arch_config.get('enabled', False):
        try:
            alpha = arch_config.get('alpha', 0.5)
            gamma = arch_config.get('gamma', 1.0)
            clf_loss = FocalLoss(alpha=alpha, gamma=gamma)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', default_lr)
            )
            model.compile(
                optimizer=optimizer,
                loss=clf_loss,
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            model._is_focal = True  # Prevent train_model from adding redundant class_weight on top of FocalLoss
        except Exception as e:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', default_lr)
            )
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', default_lr)
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    
    return model


def build_cnn_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build CNN model for 1D feature data.
    
    Config keys: filters, kernel_size, layers, pooling, dropout, learning_rate
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (number of features)
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled CNN model
    """
    filters = config.get('filters', 64)
    dropout = config.get('dropout', 0.1)
    kernel_size = config.get('kernel_size', 5)
    num_layers = config.get('layers', 3)
    pooling = config.get('pooling', 'global_avg')
    
    inputs = tf.keras.Input(shape=(input_dim, 1))
    
    x = inputs
    for i in range(num_layers):
        x = tf.keras.layers.Conv1D(filters * (2 ** i), kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if i < num_layers - 1:
            x = tf.keras.layers.Dropout(dropout)(x)
    
    if pooling == 'global_max':
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif pooling == 'flatten':
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense classifier
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Use focal loss if configured for this architecture (per-arch config)
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('CNN', {})
    default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('CNN', 0.001)
    if arch_config.get('enabled', False):
        try:
            alpha = arch_config.get('alpha', 0.5)
            gamma = arch_config.get('gamma', 1.0)
            clf_loss = FocalLoss(alpha=alpha, gamma=gamma)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', default_lr)
            )
            model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        except Exception as e:
            # Fallback to standard loss if focal loss fails
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', default_lr)
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', default_lr)
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_rnn_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build RNN model (bidirectional LSTM).
    
    Config keys: units, layers, dropout, learning_rate
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled RNN model
    """
    default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('RNN', 0.001)
    units = config.get('units', 64)
    num_layers = config.get('layers', 2)
    dropout = max(config.get('dropout', 0.1), 0.1)
    learning_rate = config.get('learning_rate', default_lr)
    
    inputs = tf.keras.Input(shape=(input_dim, 1))
    
    x = inputs
    for i in range(num_layers):
        return_seq = i < num_layers - 1
        lstm_units = max(units // (2 ** i), 8)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=return_seq)
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('RNN', {})
    if arch_config.get('enabled', False):
        try:
            clf_loss = FocalLoss(alpha=arch_config.get('alpha', 0.5), gamma=arch_config.get('gamma', 1.0))
            model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        except Exception:
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_lstm_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build LSTM model with dedicated architecture (separate from RNN).
    
    Config keys: lstm_units, layers, bidirectional, dropout, learning_rate
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (number of features)
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled LSTM model
    """
    default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('LSTM', 0.001)
    lstm_units = config.get('lstm_units', 64)
    num_layers = config.get('layers', 2)
    bidirectional = config.get('bidirectional', False)
    dropout = config.get('dropout', 0.1)
    learning_rate = config.get('learning_rate', default_lr)
    
    inputs = tf.keras.Input(shape=(input_dim, 1))
    
    x = inputs
    for i in range(num_layers):
        return_seq = i < num_layers - 1
        units = max(lstm_units // (2 ** i), 8)
        lstm_layer = tf.keras.layers.LSTM(units, return_sequences=return_seq)
        if bidirectional:
            lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)
        x = lstm_layer(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('LSTM', {})
    if arch_config.get('enabled', False):
        try:
            clf_loss = FocalLoss(alpha=arch_config.get('alpha', 0.5), gamma=arch_config.get('gamma', 1.0))
            model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        except Exception:
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_dense_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build simple dense neural network

    Config keys: units, layers, dropout, activation, learning_rate

    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)

    Returns:
        Compiled dense model
    """
    hidden_units = config.get('units', 64)
    num_layers = config.get('layers', 2)
    activation = config.get('activation', 'relu')
    dropout_rate = config.get('dropout', 0.2)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for _ in range(num_layers):
        x = tf.keras.layers.Dense(hidden_units, activation=activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('Dense', 0.001)
    opt = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', default_lr))

    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('Dense', {})
    if arch_config.get('enabled', False):
        try:
            clf_loss = FocalLoss(alpha=arch_config.get('alpha', 0.5), gamma=arch_config.get('gamma', 1.0))
            model.compile(optimizer=opt, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        except Exception:
            model.compile(optimizer=opt, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    else:
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])

    return model


def build_transformer_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build Transformer/Attention model for tabular data.
    
    Config keys: heads, dim, ff_dim, layers, dropout, learning_rate
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled Transformer model
    """
    try:
        heads = config.get('heads', 2)
        dim = config.get('dim', 64)
        dropout = config.get('dropout', 0.2)
        num_layers = config.get('layers', 1)
        ff_dim = config.get('ff_dim', dim * 2)
        
        inputs = tf.keras.Input(shape=(input_dim,))
        
        x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
        x = tf.keras.layers.Dense(dim)(x)
        
        for _ in range(num_layers):
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=heads, key_dim=dim // heads, dropout=dropout
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn_output)
            
            ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
            ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
            ff_output = tf.keras.layers.Dense(dim)(ff_output)
            ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
            x = tf.keras.layers.LayerNormalization()(x + ff_output)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Stronger classifier head
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Use focal loss if configured for this architecture (per-arch config)
        arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('Transformer', {})
        default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('Transformer', 0.0001)
        if arch_config.get('enabled', False):
            try:
                alpha = arch_config.get('alpha', 0.5)
                gamma = arch_config.get('gamma', 1.0)
                clf_loss = FocalLoss(alpha=alpha, gamma=gamma)
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.get('learning_rate', default_lr)
                )
                model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
            except Exception as e:
                # Fallback to standard loss if focal loss fails
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.get('learning_rate', default_lr)
                )
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', default_lr)
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        
        return model
    except Exception as e:
        print(f"Transformer creation failed: {e}, using fallback")
        return build_dense_fallback(config, input_dim, loss)


def build_dense_fallback(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Dense fallback model when other architectures fail.
    
    NOTE: Not dormant — this is the live fallback path of the active
    build_transformer_model (and was kept functional for that reason).
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function
        
    Returns:
        Compiled Dense model
    """
    return build_dense_model(config, input_dim, loss)


# ============================================================================
# Section 3: Tree Architecture Builders
# ============================================================================

def calculate_dynamic_class_weight(y: np.ndarray, config: Dict) -> float:
    """
    Calculate scale_pos_weight from actual class distribution.
    
    Args:
        y: Binary labels (0 or 1)
        config: Configuration dictionary
        
    Returns:
        scale_pos_weight value
    """
    if not config.get('DYNAMIC_CLASS_WEIGHTS', False):
        return config.get('scale_pos_weight', 259)
    
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    
    if pos_count == 0:
        return 259.0
    
    weight = neg_count / pos_count
    return float(weight)


def build_lightgbm_model(config: Dict, input_dim: int, y_train: np.ndarray = None):
    """
    Build LightGBM classifier
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (unused but kept for API consistency)
        y_train: Training labels for dynamic class weight calculation (optional)
        
    Returns:
        SklearnModelWrapper wrapping LGBMClassifier
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=config.get('n_estimators', 1000),  # Increased from 500
        num_leaves=config.get('num_leaves', 127),  # Increased from 63
        learning_rate=config.get('learning_rate', 0.05),  # Decreased from 0.1
        class_weight='balanced',
        min_child_samples=config.get('min_child_samples', 100),  # Decreased from 200
        subsample=config.get('subsample', 0.8),
        colsample_bytree=config.get('colsample_bytree', 0.8),
        reg_alpha=config.get('reg_alpha', 0.1),
        reg_lambda=config.get('reg_lambda', 1.0),
        max_depth=config.get('max_depth', 8),  # Increased from 5
        min_split_gain=config.get('min_split_gain', 0.0),
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    
    return SklearnModelWrapper(model)


def build_xgboost_model(config: Dict, input_dim: int, y_train: np.ndarray = None):
    """
    Build XGBoost classifier
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (unused but kept for API consistency)
        y_train: Kept for API compatibility (scale_pos_weight set dynamically in trainer per threshold)
        
    Returns:
        SklearnModelWrapper wrapping XGBClassifier
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=config.get('n_estimators', 1000),  # Increased from 500
        max_depth=config.get('max_depth', 8),  # Increased from 5
        learning_rate=config.get('learning_rate', 0.03),  # Decreased from 0.05
        scale_pos_weight=1,  # Overridden per training call in _train_sklearn_model
        min_child_weight=config.get('min_child_weight', 1),
        subsample=config.get('subsample', 0.8),
        colsample_bytree=config.get('colsample_bytree', 0.8),
        gamma=config.get('gamma', 0),
        reg_alpha=config.get('reg_alpha', 0),
        reg_lambda=config.get('reg_lambda', 1),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
    )
    
    return SklearnModelWrapper(model)


def build_catboost_model(config: Dict, input_dim: int, y_train: np.ndarray = None):
    """
    Build CatBoost classifier
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension (unused but kept for API consistency)
        y_train: Training labels for dynamic class weight calculation (optional)
        
    Returns:
        SklearnModelWrapper wrapping CatBoostClassifier
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    # Calculate dynamic weight if enabled and y_train provided
    # CatBoost uses scale_pos_weight (not available in all versions) or auto_class_weights
    if y_train is not None and config.get('DYNAMIC_CLASS_WEIGHTS', False):
        scale_pos_weight = calculate_dynamic_class_weight(y_train, config)
        # For CatBoost, use calculated weight if supported, else fallback to Balanced
        auto_weights = 'Scaled' if hasattr(CatBoostClassifier, 'scale_pos_weight') else 'Balanced'
    else:
        scale_pos_weight = config.get('scale_pos_weight', 259)
        auto_weights = config.get('auto_class_weights', 'SqrtBalanced')
    
    model = CatBoostClassifier(
        iterations=config.get('iterations', 1000),  # Increased from 500
        depth=config.get('depth', 8),  # Increased from 6
        learning_rate=config.get('learning_rate', 0.03),  # Decreased from 0.05
        auto_class_weights=auto_weights,
        l2_leaf_reg=config.get('l2_leaf_reg', 3),
        random_state=42,
        verbose=False,
        thread_count=-1,
    )
    
    return SklearnModelWrapper(model)


# ============================================================================
# Section 4: Ensemble Builders
# ============================================================================

def build_stacking_meta_model(config: Dict, input_dim: int,
                             base_models: Optional[List] = None) -> tf.keras.Model:
    """
    Build stacking meta-learner model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        base_models: List of base models (not used in this simplified version)
        
    Returns:
        Compiled meta-learner model
    """
    units = config.get('units', 64)
    dropout = config.get('dropout', 0.1)
    
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(units, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Dense(units // 2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_precision_ensemble(models: List, val_preds_matrix: np.ndarray,
                             ensemble_name: str = "ensemble",
                             precision_weights: List[float] = None,
                             features_per_model: List[List[int]] = None,
                         logger: Callable = None) -> Callable:
    """
    Create precision-optimized ensemble from trained models
    
    Args:
        models: List of trained models
        val_preds_matrix: Matrix of validation predictions (n_models x n_samples)
        ensemble_name: Name for the ensemble
        precision_weights: Optional list of precision values for weighted averaging
        features_per_model: Optional list of feature indices per model for pruning
        
    Returns:
        Ensemble callable that takes X and returns predictions
    """
    if not models:
        # Return dummy ensemble that returns zeros
        def dummy_ensemble(X):
            return np.zeros(len(X))
        return dummy_ensemble
    
    # Determine weighting strategy
    use_precision_weights = precision_weights is not None and len(precision_weights) == len(models)
    
    def ensemble_predict(X):
        predictions = []
        for i, model in enumerate(models):
            try:
                X_i = X[:, features_per_model[i]] if features_per_model is not None and i < len(features_per_model) and features_per_model[i] is not None else X
                if hasattr(model, 'sklearn_model'):
                    pred = model.predict_proba(X_i)[:, 1]
                elif hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_i)[:, 1]
                else:
                    pred = model.predict(X_i).flatten()
                predictions.append(pred)
            except Exception as e:
                if logger:
                    logger(f"Warning: Model prediction failed: {e}", 'warning')
                else:
                    print(f"Warning: Model prediction failed: {e}")
                continue
        
        if not predictions:
            return np.zeros(len(X))
        
        if use_precision_weights:
            # Precision-weighted averaging
            # Weight = precision_i / sum(precision)
            total_weight = sum(precision_weights)
            if total_weight > 0:
                weighted_preds = np.zeros_like(predictions[0])
                for pred, weight in zip(predictions, precision_weights):
                    weighted_preds += pred * (weight / total_weight)
                return weighted_preds
            else:
                # Fallback to simple average
                return np.mean(predictions, axis=0)
        else:
            # Simple averaging ensemble
            return np.mean(predictions, axis=0)
    
    return ensemble_predict


# ============================================================================
# Section 5: SklearnModelWrapper
# ============================================================================

class SklearnModelWrapper:
    """Wrapper to make sklearn models compatible with TensorFlow interface"""
    
    def __init__(self, sklearn_model):
        """
        Initialize wrapper
        
        Args:
            sklearn_model: Scikit-learn model instance
        """
        self.sklearn_model = sklearn_model
        self._is_fitted = False
    
    def fit(self, X, y=None, **kwargs):
        """
        Fit the model
        
        Args:
            X: Features
            y: Labels (optional for unsupervised models)
            
        Returns:
            self
        """
        if hasattr(self.sklearn_model, 'fit'):
            if y is not None:
                self.sklearn_model.fit(X, y, **kwargs)
            else:
                self.sklearn_model.fit(X)
        self._is_fitted = True
        return self
    
    def predict(self, X, **kwargs):
        """
        Predict probabilities (for compatibility with TensorFlow interface)
        
        Args:
            X: Features
            **kwargs: Extra arguments (ignored for sklearn compatibility)
            
        Returns:
            Probability array (n_samples, 1)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        # Use predict_proba for probability predictions if available
        if hasattr(self.sklearn_model, 'predict_proba'):
            proba = self.sklearn_model.predict_proba(X)
            # Return probability of positive class (column 1)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1:2]  # Shape (n_samples, 1)
            return proba
        else:
            # Fallback: convert class labels to probabilities
            preds = self.sklearn_model.predict(X)
            return preds.astype(np.float32).reshape(-1, 1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability array (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        if hasattr(self.sklearn_model, 'predict_proba'):
            return self.sklearn_model.predict_proba(X)
        else:
            # For models without predict_proba, convert predictions to pseudo-probabilities
            preds = self.predict(X)
            
            # Handle IsolationForest (-1 for outliers, 1 for inliers)
            if isinstance(self.sklearn_model, IsolationForest):
                # Convert to [0, 1] probabilities
                # -1 (outlier/signal) -> high probability
                # 1 (inlier/normal) -> low probability
                proba = np.zeros((len(preds), 2))
                proba[:, 1] = (preds == -1).astype(float)  # Signal probability
                proba[:, 0] = 1 - proba[:, 1]  # Normal probability
                return proba
            else:
                # Default: binary classification
                proba = np.zeros((len(preds), 2))
                proba[:, 1] = preds  # Assume preds are in [0, 1] or {0, 1}
                proba[:, 0] = 1 - proba[:, 1]
                return proba
    
    def __call__(self, X):
        """
        Make callable like TensorFlow models
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class
    
    def save(self, filepath):
        """Save sklearn model to file using joblib
        
        Args:
            filepath: Path to save the model (with .joblib extension)
        """
        import joblib
        joblib.dump(self.sklearn_model, filepath)
    
    @staticmethod
    def load(filepath):
        """Load sklearn model from file using joblib
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            SklearnModelWrapper: Wrapped sklearn model
        """
        import joblib
        model = joblib.load(filepath)
        return SklearnModelWrapper(model)
    
    def decision_function(self, X):
        """
        Get decision function scores
        
        Args:
            X: Features
            
        Returns:
            Decision scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        if hasattr(self.sklearn_model, 'decision_function'):
            return self.sklearn_model.decision_function(X)
        else:
            return self.predict_proba(X)[:, 1]


# ============================================================================
# Section 6: Dormant Builders (stubs with NotImplementedError)
# ============================================================================
# These builders are registered in training.py's dispatch table but not in
# ACTIVE_ARCHITECTURES by default. If activated via config, they raise
# NotImplementedError pointing to the pre-refactoring source files.

def build_tabnet_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'TabNet' is dormant in the refactored pipeline. "
        f"See chunk_09_models_advanced.py for the original implementation."
    )

def build_gnn_sage_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'GNN_SAGE' is dormant in the refactored pipeline. "
        f"See chunk_09_models_advanced.py for the original implementation."
    )

def build_gnn_gat_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'GNN_GAT' is dormant in the refactored pipeline. "
        f"See chunk_09_models_advanced.py for the original implementation."
    )

def build_hybrid_cnn_lstm_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'CNN_LSTM_Hybrid' is dormant in the refactored pipeline. "
        f"See chunk_09_models_advanced.py for the original implementation."
    )

def build_hybrid_transformer_gnn_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'Transformer_GNN_Hybrid' is dormant in the refactored pipeline. "
        f"See chunk_09_models_advanced.py for the original implementation."
    )

def build_simple_attention_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'Simple_Attention' is dormant in the refactored pipeline. "
        f"See chunk_09_models_advanced.py for the original implementation."
    )

def build_isolation_forest_model(config: Dict, input_dim: int = None, y_train: np.ndarray = None):
    raise NotImplementedError(
        f"Architecture 'Isolation_Forest' is dormant in the refactored pipeline. "
        f"See chunk_11_models_sklearn.py for the original implementation."
    )

def build_oneclass_svm_model(config: Dict, input_dim: int = None, y_train: np.ndarray = None):
    raise NotImplementedError(
        f"Architecture 'OneClass_SVM' is dormant in the refactored pipeline. "
        f"See chunk_11_models_sklearn.py for the original implementation."
    )

def build_svm_model(config: Dict, input_dim: int = None, y_train: np.ndarray = None):
    raise NotImplementedError(
        f"Architecture 'SVM' is dormant in the refactored pipeline. "
        f"See chunk_11_models_sklearn.py for the original implementation."
    )

def build_cnn_feature_extractor(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'CNN_Feature_Extractor' is dormant in the refactored pipeline. "
        f"See chunk_08_models_base.py for the original implementation."
    )

def build_bagging_random_forest_model(config: Dict, input_dim: int, y_train: np.ndarray = None):
    raise NotImplementedError(
        f"Architecture 'Bagging_RandomForest' is dormant in the refactored pipeline. "
        f"See chunk_10_models_ensemble.py for the original implementation."
    )

def build_extra_trees_ensemble_model(config: Dict, input_dim: int, y_train: np.ndarray = None):
    raise NotImplementedError(
        f"Architecture 'ExtraTrees_Ensemble' is dormant in the refactored pipeline. "
        f"See chunk_10_models_ensemble.py for the original implementation."
    )

def build_boosting_adaptive_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy'):
    raise NotImplementedError(
        f"Architecture 'Boosting_Adaptive' is dormant in the refactored pipeline. "
        f"See chunk_10_models_ensemble.py for the original implementation."
    )
