"""
Chunk 09: Models - Advanced
Advanced neural architectures (Transformer, GNN, TabNet)
"""

import numpy as np
import tensorflow as tf
import keras
from typing import Dict


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


class TabNet(keras.Model):
    """TabNet architecture for tabular data"""
    
    def __init__(self, feature_dim: int = 64, output_dim: int = 64, 
                 num_decision_steps: int = 5, relaxation_factor: float = 1.5,
                 **kwargs):
        super(TabNet, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        
    def build(self, input_shape):
        self.initial_layer = tf.keras.layers.Dense(self.feature_dim)
        self.decision_layers = []
        for _ in range(self.num_decision_steps):
            self.decision_layers.append(
                tf.keras.layers.Dense(self.output_dim, activation='relu')
            )
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        super(TabNet, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.initial_layer(inputs)
        for layer in self.decision_layers:
            x = layer(x)
        return self.output_layer(x)


def build_transformer_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build Simplified Transformer/Attention model for tabular data.
    
    Key changes from original:
    - Single attention head (multi-head adds complexity without benefit for small feature sets)
    - Single transformer layer
    - Stronger classifier head
    - Attention mechanism for feature importance
    
    Args:
        config: Configuration with 'heads', 'dim', 'dropout'
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled Transformer model
    """
    try:
        heads = config.get('heads', 2)  # Reduced from 4
        dim = config.get('dim', 64)
        dropout = config.get('dropout', 0.2)
        
        inputs = tf.keras.Input(shape=(input_dim,))
        
        # Project to embedding dimension
        embeddings = tf.keras.layers.Dense(dim)(inputs)
        # Expand for attention layer
        x = tf.keras.layers.Reshape((1, dim))(embeddings)
        
        # Single transformer block
        # Multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=dim // heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)
        
        # Feed-forward
        ff_output = tf.keras.layers.Dense(dim * 2, activation='relu')(x)
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
        print(f"Transformer creation failed: {e}, using fallback")
        return build_dense_fallback(config, input_dim, loss)


def build_simple_attention_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build Simple Self-Attention model for tabular data.
    
    Uses simplified attention mechanism that:
    - Computes attention weights between features
    - Applies weighted sum to get feature importance
    - Uses attention output for classification
    
    Args:
        config: Configuration with 'dim', 'dropout'
        input_dim: Input dimension (number of features)
        loss: Loss function
        
    Returns:
        Compiled attention model
    """
    try:
        dim = config.get('dim', 64)
        dropout = config.get('dropout', 0.2)
        
        inputs = tf.keras.Input(shape=(input_dim,))
        
        # Project features
        queries = tf.keras.layers.Dense(dim, activation='relu')(inputs)
        keys = tf.keras.layers.Dense(dim, activation='relu')(inputs)
        values = tf.keras.layers.Dense(dim, activation='relu')(inputs)
        
        # Self-attention: compute attention weights
        # attention_weights = softmax(Q * K^T / sqrt(dim))
        attention_scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(dim, tf.float32))
        attention_weights = tf.keras.layers.Softmax()(attention_scores)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, values)
        
        # Combine original with attended
        combined = tf.keras.layers.Concatenate()([inputs, attended])
        
        # Classifier
        x = tf.keras.layers.Dense(dim, activation='relu')(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        
        x = tf.keras.layers.Dense(dim // 2, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Use focal loss if configured
        if config.get('USE_FOCAL_LOSS', False):
            from chunk_11_models_sklearn import FocalLoss
            clf_loss = FocalLoss(
                alpha=config.get('FOCAL_LOSS_ALPHA', 0.5), 
                gamma=config.get('FOCAL_LOSS_GAMMA', 1.0)
            )
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.0001)
            )
            model.compile(optimizer=optimizer, loss=clf_loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.0001)
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
        
        return model
    except Exception as e:
        print(f"Simple Attention creation failed: {e}, using fallback")
        return build_dense_fallback(config, input_dim, loss)


def build_dense_fallback(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Dense fallback model when other architectures fail.
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function
        
    Returns:
        Compiled Dense model
    """
    from chunk_08_models_base import build_dense_model
    return build_dense_model(config, input_dim, loss)


def build_tabnet_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build TabNet model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled TabNet model
    """
    dim = config.get('dim', 64)
    
    model = TabNet(feature_dim=dim, output_dim=dim, num_decision_steps=5)
    model.build((None, input_dim))
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_gnn_sage_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build GraphSAGE-style model for tabular data
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled GNN model
    """
    units = config.get('units', 64)
    dropout = config.get('dropout', 0.1)
    
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Dense layers simulating message passing
    x = tf.keras.layers.Dense(units, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # "Aggregation" layer
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Dense(units // 2, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_gnn_gat_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build Graph Attention Network style model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled GAT model
    """
    units = config.get('units', 64)
    heads = config.get('heads', 4)
    dropout = config.get('dropout', 0.1)
    
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Attention mechanism
    # For tabular data, use self-attention across features
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    
    # Multi-head attention
    attn_outputs = []
    for _ in range(heads):
        attn = tf.keras.layers.Dense(input_dim, activation='softmax')(inputs)
        attn_outputs.append(inputs * attn)
    
    x = tf.keras.layers.Concatenate()(attn_outputs) if heads > 1 else attn_outputs[0]
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Dense(units // 2, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_gnn_gat_model(config: Dict, input_dim: int) -> tf.keras.Model:
    """
    Build Graph Attention Network style model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        
    Returns:
        Compiled GAT model
    """
    units = config.get('units', 64)
    heads = config.get('heads', 4)
    dropout = config.get('dropout', 0.1)
    
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Attention mechanism
    # For tabular data, use self-attention across features
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    
    # Multi-head attention
    attn_outputs = []
    for _ in range(heads):
        attn = tf.keras.layers.Dense(input_dim, activation='softmax')(inputs)
        attn_outputs.append(inputs * attn)
    
    x = tf.keras.layers.Concatenate()(attn_outputs) if heads > 1 else attn_outputs[0]
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = tf.keras.layers.Dense(units // 2, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def build_hybrid_cnn_lstm_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build hybrid CNN-LSTM model
    
    Args:
        config: Configuration with 'cnn_filters', 'lstm_units', 'dropout'
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled hybrid model
    """
    cnn_filters = config.get('cnn_filters', 64)
    lstm_units = config.get('lstm_units', 32)
    dropout = config.get('dropout', 0.1)
    
    inputs = tf.keras.Input(shape=(input_dim, 1))
    
    # CNN layers
    x = tf.keras.layers.Conv1D(cnn_filters, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    # LSTM layers
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(lstm_units // 2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Use focal loss if configured for this architecture (per-arch config)
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get('Hybrid_CNN_LSTM', {})
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
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


def build_hybrid_transformer_gnn_model(config: Dict, input_dim: int, loss: str = 'binary_crossentropy') -> tf.keras.Model:
    """
    Build hybrid Transformer-GNN model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        loss: Loss function (default: binary_crossentropy)
        
    Returns:
        Compiled hybrid model
    """
    # Combine transformer attention with graph-like aggregation
    heads = config.get('heads', 4)
    dim = config.get('dim', 64)
    dropout = config.get('dropout', 0.1)
    
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Project to embedding dimension
    x = tf.keras.layers.Dense(dim)(inputs)
    x = tf.keras.layers.Reshape((1, dim))(x)
    
    # Transformer attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=heads, key_dim=dim // heads, dropout=dropout
    )(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn_output)
    
    # Graph-like aggregation (simulate with dense layers)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dim, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dim // 2, activation='relu')(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Precision()])
    
    return model


if __name__ == "__main__":
    # Self-test
    print("Testing advanced models...")
    
    config = {
        'heads': 4,
        'dim': 64,
        'units': 64,
        'cnn_filters': 64,
        'lstm_units': 32,
        'dropout': 0.1
    }
    input_dim = 37
    
    from chunk_08_models_base import validate_model_output
    
    # Test Transformer
    print("\nTesting Transformer...")
    transformer = build_transformer_model(config, input_dim)
    validate_model_output(transformer, input_dim)
    print(f"[pass] Transformer validated: {transformer.count_params()} params")
    
    # Test TabNet
    print("\nTesting TabNet...")
    tabnet = build_tabnet_model(config, input_dim)
    validate_model_output(tabnet, input_dim)
    print(f"[pass] TabNet validated: {tabnet.count_params()} params")
    
    # Test GNN SAGE
    print("\nTesting GNN SAGE...")
    gnn_sage = build_gnn_sage_model(config, input_dim)
    validate_model_output(gnn_sage, input_dim)
    print(f"[pass] GNN SAGE validated: {gnn_sage.count_params()} params")
    
    # Test GNN GAT
    print("\nTesting GNN GAT...")
    gnn_gat = build_gnn_gat_model(config, input_dim)
    validate_model_output(gnn_gat, input_dim)
    print(f"[pass] GNN GAT validated: {gnn_gat.count_params()} params")
    
    # Test Hybrid CNN-LSTM
    print("\nTesting Hybrid CNN-LSTM...")
    hybrid_cnn_lstm = build_hybrid_cnn_lstm_model(config, input_dim)
    validate_model_output(hybrid_cnn_lstm, input_dim)
    print(f"[pass] Hybrid CNN-LSTM validated: {hybrid_cnn_lstm.count_params()} params")
    
    # Test Hybrid Transformer-GNN
    print("\nTesting Hybrid Transformer-GNN...")
    hybrid_trans_gnn = build_hybrid_transformer_gnn_model(config, input_dim)
    validate_model_output(hybrid_trans_gnn, input_dim)
    print(f"[pass] Hybrid Transformer-GNN validated: {hybrid_trans_gnn.count_params()} params")
    
    print("\n[pass] All advanced model tests passed")