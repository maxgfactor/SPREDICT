"""
Chunk 08: Models - Base
Base neural network architectures
"""

import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Optional


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
            from chunk_11_models_sklearn import FocalLoss
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
            from chunk_11_models_sklearn import FocalLoss
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


def build_cnn_feature_extractor(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build CNN Feature Extractor + Dense Classifier (two-stage approach).
    
    Architecture:
    - Stage 1: CNN feature extraction (without classification head)
    - Stage 2: Dense classifier on CNN features
    
    This separates feature learning from classification, allowing each
    to specialize in its task.
    
    Args:
        config: Configuration with 'cnn_filters', 'dropout', 'kernel_sizes'
        input_dim: Input dimension
        loss: Loss function
        
    Returns:
        Compiled CNN-Feature model
    """
    try:
        filters = config.get('filters', 64)
        dropout = config.get('dropout', 0.2)
        kernel_size = config.get('kernel_size', 5)
        
        # Stage 1: CNN Feature Extractor
        inputs = tf.keras.Input(shape=(input_dim, 1))
        
        # Convolutional feature extraction
        x = tf.keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Conv1D(filters * 2, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(filters * 2, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Feature extraction output (before pooling)
        feature_maps = x
        
        # Stage 2: Dense Classifier on features
        features = tf.keras.layers.GlobalAveragePooling1D()(feature_maps)
        features = tf.keras.layers.Dense(256, activation='relu')(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.Dropout(dropout)(features)
        
        features = tf.keras.layers.Dense(128, activation='relu')(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.Dropout(dropout)(features)
        
        features = tf.keras.layers.Dense(64, activation='relu')(features)
        features = tf.keras.layers.Dropout(dropout)(features)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(features)
        
        model = tf.keras.Model(inputs, outputs, name='cnn_feature_classifier')
        
        # Use focal loss if configured for this architecture (per-arch config)
        arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('CNN', {})
        default_lr = config.get('DEFAULT_LEARNING_RATES', {}).get('CNN', 0.001)
        if arch_config.get('enabled', False):
            try:
                from chunk_11_models_sklearn import FocalLoss
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
        print(f"CNN Feature Extractor creation failed: {e}, using fallback")
        return build_cnn_model(config, input_dim, loss)


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
            from chunk_11_models_sklearn import FocalLoss
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
            from chunk_11_models_sklearn import FocalLoss
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
            from chunk_11_models_sklearn import FocalLoss
            clf_loss = FocalLoss(alpha=arch_config.get('alpha', 0.5), gamma=arch_config.get('gamma', 1.0))
            model.compile(optimizer=opt, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        except Exception:
            model.compile(optimizer=opt, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    else:
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])

    return model


def validate_model_output(model: tf.keras.Model, input_dim: int) -> bool:
    """
    Validate a compiled model can accept input and produce output
    
    Args:
        model: Model to validate
        input_dim: Expected input dimension
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(model, tf.keras.Model), f"Model must be tf.keras.Model, got {type(model)}"
    assert model.optimizer is not None, "Model not compiled (no optimizer)"
    assert model.loss is not None, "Model missing loss function"
    
    # Test forward pass
    test_input = np.random.randn(1, input_dim).astype(np.float32)
    
    # Handle models that expect 3D input (CNN, RNN)
    if hasattr(model, 'input_shape') and len(model.input_shape) == 3:
        test_input = test_input.reshape(1, input_dim, 1)
    
    try:
        output = model(test_input)
        output_np = output.numpy() if hasattr(output, 'numpy') else np.array(output)
        
        assert output_np.shape == (1, 1), f"Output shape mismatch: expected (1, 1), got {output_np.shape}"
        assert np.all(np.isfinite(output_np)), "Model output contains non-finite values"
        
    except Exception as e:
        raise AssertionError(f"Model failed forward pass: {e}")
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing base models...")
    
    config = {
        'latent_dim': 32,
        'filters': 64,
        'lstm_units': 32,
        'dropout': 0.1
    }
    input_dim = 37
    
    # Test VAE
    print("\nTesting VAE...")
    vae = build_vae_model(config, input_dim)
    validate_model_output(vae, input_dim)
    print(f"[pass] VAE validated: {vae.count_params()} params")
    
    # Test CNN
    print("\nTesting CNN...")
    cnn = build_cnn_model(config, input_dim)
    validate_model_output(cnn, input_dim)
    print(f"[pass] CNN validated: {cnn.count_params()} params")
    
    # Test RNN
    print("\nTesting RNN...")
    rnn = build_rnn_model(config, input_dim)
    validate_model_output(rnn, input_dim)
    print(f"[pass] RNN validated: {rnn.count_params()} params")
    
    # Test Dense
    print("\nTesting Dense...")
    dense = build_dense_model(config, input_dim)
    validate_model_output(dense, input_dim)
    print(f"[pass] Dense validated: {dense.count_params()} params")
    
    print("\n[pass] All base model tests passed")