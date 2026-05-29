"""
Chunk 10: Models - Ensemble
Ensemble model builders and aggregators
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Callable, Optional
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


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


def build_bagging_random_forest_model(config: Dict, input_dim: int):
    """
    Build Bagging Random Forest model (sklearn wrapper)
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        
    Returns:
        Sklearn model wrapper
    """
    from sklearn.ensemble import RandomForestClassifier
    
    n_estimators = config.get('MIN_ENSEMBLE_SIZE', 5)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    return SklearnModelWrapper(rf)


def build_extra_trees_ensemble_model(config: Dict, input_dim: int):
    """
    Build Extra Trees ensemble model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        
    Returns:
        Sklearn model wrapper
    """
    n_estimators = config.get('MIN_ENSEMBLE_SIZE', 5)
    
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    return SklearnModelWrapper(et)


def build_boosting_adaptive_model(config: Dict, input_dim: int) -> tf.keras.Model:
    """
    Build adaptive boosting model
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension
        
    Returns:
        Compiled boosting model
    """
    units = config.get('units', 64)
    dropout = config.get('dropout', 0.1)
    
    # Simplified boosting: deep ensemble network
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Multiple "weak learners" (parallel paths)
    weak_learners = []
    for i in range(5):
        x = tf.keras.layers.Dense(units // 2, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        weak_learners.append(x)
    
    # Combine weak learners
    if len(weak_learners) > 1:
        combined = tf.keras.layers.Average()(weak_learners)
    else:
        combined = weak_learners[0]
    
    # Final adjustment
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


class SklearnModelWrapper:
    """Wrapper to make sklearn models compatible with TensorFlow interface"""
    
    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model
        self._is_fitted = False
    
    def fit(self, X, y, **kwargs):
        """Fit the model"""
        self.sklearn_model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X, **kwargs):
        """Predict class labels"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        return self.sklearn_model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        return self.sklearn_model.predict_proba(X)
    
    def __call__(self, X):
        """Make callable like TensorFlow models"""
        proba = self.predict_proba(X)
        return proba[:, 1]  # Return probability of positive class


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


def validate_ensemble_output(ensemble: Callable, X: np.ndarray, 
                            expected_length: int) -> bool:
    """
    Validate ensemble produces valid predictions
    
    Args:
        ensemble: Ensemble callable
        X: Input features
        expected_length: Expected number of predictions
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    assert callable(ensemble), "Ensemble must be callable"
    
    predictions = ensemble(X)
    
    assert isinstance(predictions, np.ndarray), \
        f"Ensemble must return np.ndarray, got {type(predictions)}"
    
    assert len(predictions) == expected_length, \
        f"Prediction length mismatch: expected {expected_length}, got {len(predictions)}"
    
    assert np.all((predictions >= 0) & (predictions <= 1)), \
        f"Predictions must be probabilities [0,1], got [{predictions.min()}, {predictions.max()}]"
    
    assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
    
    # Test determinism
    predictions2 = ensemble(X)
    assert np.allclose(predictions, predictions2), "Ensemble not deterministic"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing ensemble models...")
    
    config = {
        'units': 64,
        'dropout': 0.1,
        'MIN_ENSEMBLE_SIZE': 5
    }
    input_dim = 37
    
    # Test Bagging RF
    print("\nTesting Bagging Random Forest...")
    bagging_rf = build_bagging_random_forest_model(config, input_dim)
    print(f"[pass] Bagging RF created")
    
    # Test Extra Trees
    print("\nTesting Extra Trees...")
    extra_trees = build_extra_trees_ensemble_model(config, input_dim)
    print(f"[pass] Extra Trees created")
    
    # Test Boosting
    print("\nTesting Boosting...")
    boosting = build_boosting_adaptive_model(config, input_dim)
    from chunk_08_models_base import validate_model_output
    validate_model_output(boosting, input_dim)
    print(f"[pass] Boosting validated: {boosting.count_params()} params")
    
    # Test Stacking
    print("\nTesting Stacking Meta...")
    stacking = build_stacking_meta_model(config, input_dim)
    validate_model_output(stacking, input_dim)
    print(f"[pass] Stacking validated: {stacking.count_params()} params")
    
    # Test ensemble creation
    print("\nTesting ensemble creation...")
    models = [boosting, stacking]
    val_preds = np.random.rand(len(models), 100)
    ensemble = create_precision_ensemble(models, val_preds, "test_ensemble")
    
    X_test = np.random.randn(100, input_dim).astype(np.float32)
    validate_ensemble_output(ensemble, X_test, 100)
    print(f"[pass] Ensemble validated")
    
    print("\n[pass] All ensemble model tests passed")