"""
Chunk 11: Models - Sklearn
Scikit-learn model wrappers and builders
"""

import numpy as np
from typing import Dict
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM, SVC
import keras


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


def build_isolation_forest_model(config: Dict):
    """
    Build Isolation Forest model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SklearnModelWrapper wrapping IsolationForest
    """
    n_estimators = config.get('MIN_ENSEMBLE_SIZE', 5)
    
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination='auto',
        random_state=42,
        n_jobs=-1
    )
    
    return SklearnModelWrapper(iso_forest)


def build_oneclass_svm_model(config: Dict):
    """
    Build One-Class SVM model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SklearnModelWrapper wrapping OneClassSVM
    """
    svm = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.1  # Expected proportion of outliers
    )
    
    return SklearnModelWrapper(svm)


def build_svm_model(config: Dict):
    """
    Build standard SVM classifier
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SklearnModelWrapper wrapping SVC
    """
    svm = SVC(
        kernel='rbf',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    
    return SklearnModelWrapper(svm)


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
        y_train: Training labels for dynamic class weight calculation (optional)
        
    Returns:
        SklearnModelWrapper wrapping XGBClassifier
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    # Calculate dynamic weight if enabled and y_train provided
    if y_train is not None and config.get('DYNAMIC_CLASS_WEIGHTS', False):
        scale_pos_weight = calculate_dynamic_class_weight(y_train, config)
    else:
        scale_pos_weight = config.get('scale_pos_weight', 259)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=config.get('n_estimators', 1000),  # Increased from 500
        max_depth=config.get('max_depth', 8),  # Increased from 5
        learning_rate=config.get('learning_rate', 0.03),  # Decreased from 0.05
        scale_pos_weight=scale_pos_weight,
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


def validate_sklearn_model(model_wrapper: SklearnModelWrapper, X: np.ndarray, 
                          y: np.ndarray = None) -> bool:
    """
    Validate sklearn model wrapper interface
    
    Args:
        model_wrapper: Model wrapper to validate
        X: Features for testing
        y: Labels for testing (optional)
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(model_wrapper, SklearnModelWrapper), \
        f"Model must be SklearnModelWrapper, got {type(model_wrapper)}"
    
    # Test fit
    if y is not None:
        model_wrapper.fit(X, y)
    else:
        model_wrapper.fit(X)
    
    # Test predict
    predictions = model_wrapper.predict(X)
    assert len(predictions) == len(X), \
        f"Prediction length mismatch: {len(predictions)} vs {len(X)}"
    
    # Test predict_proba - skip for One-Class SVM (no proper probability output)
    model_type = type(model_wrapper.sklearn_model).__name__
    if model_type != 'OneClassSVM':
        proba = model_wrapper.predict_proba(X)
        assert proba.shape == (len(X), 2), \
            f"Probability shape mismatch: {proba.shape} vs ({len(X)}, 2)"
        assert np.all((proba >= 0) & (proba <= 1)), \
            f"Probabilities out of range: [{proba.min()}, {proba.max()}]"
        assert np.allclose(proba.sum(axis=1), 1.0), \
            "Probabilities don't sum to 1"
    
    # Test __call__
    call_output = model_wrapper(X)
    assert isinstance(call_output, np.ndarray), \
        f"__call__ must return np.ndarray, got {type(call_output)}"
    assert len(call_output) == len(X), \
        f"__call__ length mismatch: {len(call_output)} vs {len(X)}"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing sklearn models...")
    
    config = {
        'MIN_ENSEMBLE_SIZE': 5
    }
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Test Isolation Forest
    print("\nTesting Isolation Forest...")
    iso_forest = build_isolation_forest_model(config)
    validate_sklearn_model(iso_forest, X)
    print("[pass] Isolation Forest validated")
    
    # Test One-Class SVM
    print("\nTesting One-Class SVM...")
    oc_svm = build_oneclass_svm_model(config)
    validate_sklearn_model(oc_svm, X)  # Unsupervised, no labels needed
    print("[pass] One-Class SVM validated")
    
    # Test Standard SVM
    print("\nTesting Standard SVM...")
    svm = build_svm_model(config)
    validate_sklearn_model(svm, X, y)
    print("[pass] Standard SVM validated")
    
    print("\n[pass] All sklearn model tests passed")


# =============================================================================
# FOCAL LOSS CLASS
# =============================================================================

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