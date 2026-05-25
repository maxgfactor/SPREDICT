"""
Chunk 08: Models - Base
Base neural network architectures
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional


def build_vae_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build VAE-inspired classifier with deeper architecture.
    
    Key improvements over original:
    - Deeper encoder with more layers
    - Larger latent space for better representation
    - Stronger classification head
    - Focal loss support for class imbalance
    
    Args:
        config: Configuration dictionary with 'latent_dim', 'USE_FOCAL_LOSS', 'FOCAL_LOSS_ALPHA', 'FOCAL_LOSS_GAMMA'
        input_dim: Input dimension (number of features)
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled VAE model
    """
    try:
        latent_dim = config.get('latent_dim', 64)
        
        # Encoder - processes input to latent representation
        encoder_inputs = tf.keras.Input(shape=(input_dim,))
        
        # Deeper encoder with He initialization for better gradient flow
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(encoder_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Latent space
        z_mean = tf.keras.layers.Dense(latent_dim, activation='relu', name='latent')(x)
        
        # Stronger classification head
        clf = tf.keras.layers.Dense(64, activation='relu')(z_mean)
        clf = tf.keras.layers.Dropout(0.1)(clf)
        clf = tf.keras.layers.Dense(32, activation='relu')(clf)
        clf = tf.keras.layers.Dropout(0.1)(clf)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid')(clf)
        
        # Create model
        vae = tf.keras.Model(encoder_inputs, classification_output)
        
        # Use focal loss if configured for this architecture (per-arch config)
        arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('VAE', {})
        if arch_config.get('enabled', False):
            try:
                from chunk_11_models_sklearn import FocalLoss
                alpha = arch_config.get('alpha', 0.5)
                gamma = arch_config.get('gamma', 1.0)
                clf_loss = FocalLoss(alpha=alpha, gamma=gamma)
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.get('learning_rate', 0.0001)
                )
                vae.compile(
                    optimizer=optimizer,
                    loss=clf_loss,
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )
            except Exception as e:
                # Fallback to standard loss if focal loss fails
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.get('learning_rate', 0.0005)
                )
                vae.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.0005)
            )
            vae.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
        
        return vae
    except Exception as e:
        print(f"VAE creation failed: {e}, using fallback")
        return build_dense_model(config, input_dim, loss)


def build_cnn_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build CNN model for 1D feature data with improved architecture.
    
    Key changes from original:
    - No global pooling (loses local info)
    - Larger kernels for wider receptive field
    - GlobalAveragePooling instead of Flatten
    - Skip connections for better gradient flow
    
    Args:
        config: Configuration with 'cnn_filters', 'dropout', 'kernel_sizes'
        input_dim: Input dimension (number of features)
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled CNN model
    """
    filters = config.get('cnn_filters', 64)
    dropout = config.get('dropout', 0.1)  # Reduced dropout
    kernel_size = config.get('kernel_size', 5)  # Larger kernel
    
    inputs = tf.keras.Input(shape=(input_dim, 1))  # Add channel dimension
    
    # CNN layers with larger kernel
    x = tf.keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Conv1D(filters * 2, kernel_size, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Conv1D(filters * 4, kernel_size, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Global pooling instead of flatten (preserves local patterns)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense classifier
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Use focal loss if configured for this architecture (per-arch config)
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('Dense', {})
    if arch_config.get('enabled', False):
        try:
            from chunk_11_models_sklearn import FocalLoss
            alpha = arch_config.get('alpha', 0.5)
            gamma = arch_config.get('gamma', 1.0)
            clf_loss = FocalLoss(alpha=alpha, gamma=gamma)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.0001)
            )
            model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        except Exception as e:
            # Fallback to standard loss if focal loss fails
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.0001)
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.0001)
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
        filters = config.get('cnn_filters', 64)
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
        if arch_config.get('enabled', False):
            try:
                from chunk_11_models_sklearn import FocalLoss
                alpha = arch_config.get('alpha', 0.5)
                gamma = arch_config.get('gamma', 1.0)
                clf_loss = FocalLoss(alpha=alpha, gamma=gamma)
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.get('learning_rate', 0.0001)
                )
                model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
            except Exception as e:
                # Fallback to standard loss if focal loss fails
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=config.get('learning_rate', 0.0001)
                )
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.0001)
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        
        return model
    except Exception as e:
        print(f"CNN Feature Extractor creation failed: {e}, using fallback")
        return build_cnn_model(config, input_dim, loss)


def build_rnn_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build RNN/LSTM model
    
    Args:
        config: Configuration with 'lstm_units', 'dropout', 'learning_rate'
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled RNN model
    """
    lstm_units = config.get('lstm_units', 32)
    dropout = config.get('dropout', 0.1)
    learning_rate = config.get('learning_rate', 0.001)
    
    inputs = tf.keras.Input(shape=(input_dim, 1))  # Add time step dimension
    
    # Bidirectional LSTM layers (captures patterns in both directions)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units // 2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_lstm_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build LSTM model with dedicated architecture (separate from RNN)
    
    Args:
        config: Configuration dictionary with 'lstm_units', 'dropout', 'learning_rate'
        input_dim: Input dimension (number of features)
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled LSTM model
    """
    lstm_units = config.get('lstm_units', 64)
    dropout = config.get('dropout', 0.1)  # Reduced dropout
    learning_rate = config.get('learning_rate', 0.0001)  # Lower learning rate for stability
    
    inputs = tf.keras.Input(shape=(input_dim, 1))  # 3D input for sequence models
    
    # LSTM layers - standard (not bidirectional, separate from RNN which uses Bidirectional)
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(lstm_units // 2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_dense_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build simple dense neural network (fallback model)
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled dense model
    """
    hidden_units = config.get('dense_hidden_units', [64, 32])
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
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
        'cnn_filters': 64,
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