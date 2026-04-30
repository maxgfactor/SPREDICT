"""
Validation Framework for CI/CD Pipeline Chunks
Centralized validation utilities for input/output contracts
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


class ValidationError(Exception):
    """Raised when validation fails between pipeline chunks"""
    pass


class ChunkValidator:
    """Validates inputs/outputs between pipeline chunks"""
    
    @staticmethod
    def validate_array(name: str, arr: np.ndarray, expected_shape: Optional[Tuple] = None, 
                       allow_none: bool = False, allow_empty: bool = False) -> bool:
        """
        Validate numpy array structure
        
        Args:
            name: Name of the array for error messages
            arr: Array to validate
            expected_shape: Expected shape tuple (None for any dimension)
            allow_none: Whether None is allowed
            allow_empty: Whether empty arrays are allowed
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if arr is None:
            if not allow_none:
                raise ValidationError(f"{name}: Cannot be None")
            return True
            
        if not isinstance(arr, np.ndarray):
            raise ValidationError(f"{name}: Expected np.ndarray, got {type(arr)}")
            
        if not allow_empty and len(arr) == 0:
            raise ValidationError(f"{name}: Cannot be empty")
            
        if expected_shape is not None:
            if len(arr.shape) != len(expected_shape):
                raise ValidationError(
                    f"{name}: Dimension mismatch. Expected {len(expected_shape)}D, got {len(arr.shape)}D"
                )
            
            for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValidationError(
                        f"{name}: Shape mismatch at dim {i}. Expected {expected}, got {actual}"
                    )
                    
        return True
    
    @staticmethod
    def validate_context(context: Dict, required_keys: List[str]) -> bool:
        """
        Validate context dictionary has required keys
        
        Args:
            context: Dictionary to validate
            required_keys: List of required key names
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(context, dict):
            raise ValidationError(f"Context must be dict, got {type(context)}")
            
        missing = [k for k in required_keys if k not in context]
        if missing:
            raise ValidationError(f"Context missing required keys: {missing}")
            
        return True
    
    @staticmethod
    def validate_config(config: Dict, required_keys: List[str]) -> bool:
        """
        Validate CONFIG has required keys
        
        Args:
            config: Configuration dictionary
            required_keys: List of required key names
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(config, dict):
            raise ValidationError(f"CONFIG must be dict, got {type(config)}")
            
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValidationError(f"CONFIG missing required keys: {missing}")
            
        return True
    
    @staticmethod
    def validate_binary_labels(y: np.ndarray, name: str = "labels") -> bool:
        """
        Validate array contains only binary values (0, 1)
        
        Args:
            y: Array to validate
            name: Name for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(y, np.ndarray):
            raise ValidationError(f"{name}: Expected np.ndarray, got {type(y)}")
            
        unique = np.unique(y)
        invalid = set(unique) - {0, 1}
        if invalid:
            raise ValidationError(f"{name}: Must be binary (0, 1), got {invalid}")
            
        return True
    
    @staticmethod
    def validate_probabilities(preds: np.ndarray, name: str = "predictions") -> bool:
        """
        Validate array contains valid probabilities [0, 1]
        
        Args:
            preds: Array to validate
            name: Name for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(preds, np.ndarray):
            raise ValidationError(f"{name}: Expected np.ndarray, got {type(preds)}")
            
        if np.any(preds < 0) or np.any(preds > 1):
            min_val, max_val = preds.min(), preds.max()
            raise ValidationError(f"{name}: Must be in [0, 1], got [{min_val}, {max_val}]")
            
        if not np.all(np.isfinite(preds)):
            raise ValidationError(f"{name}: Contains non-finite values")
            
        return True
    
    @staticmethod
    def validate_metrics_dict(metrics: Dict, required_keys: List[str] = None) -> bool:
        """
        Validate metrics dictionary structure
        
        Args:
            metrics: Metrics dictionary
            required_keys: List of required metric names
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(metrics, dict):
            raise ValidationError(f"Metrics must be dict, got {type(metrics)}")
            
        if required_keys:
            missing = [k for k in required_keys if k not in metrics]
            if missing:
                raise ValidationError(f"Metrics missing required keys: {missing}")
        
        # Validate all values are numeric and in [0, 1] if they look like scores
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    raise ValidationError(f"Metric '{key}' is not finite: {value}")
                    
        return True
    
    @staticmethod
    def validate_phase_completion_flags(context: Dict, phase_nums: List[int]) -> bool:
        """
        Validate that specified phases are marked as complete
        
        Args:
            context: Pipeline context
            phase_nums: List of phase numbers to check
            
        Returns:
            True if all phases complete
            
        Raises:
            ValidationError: If any phase not complete
        """
        for phase_num in phase_nums:
            flag = f'phase{phase_num}_complete'
            if not context.get(flag):
                raise ValidationError(f"Phase {phase_num} not complete (missing {flag})")
                
        return True


def validate_data_contract(X: np.ndarray, y: np.ndarray, dates: np.ndarray, 
                          min_samples: int = 30) -> bool:
    """
    Validate the standard data contract (X, y, dates)
    
    Args:
        X: Feature matrix
        y: Labels
        dates: Temporal data
        min_samples: Minimum required samples
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    validator = ChunkValidator()
    
    # Validate types and dimensions
    validator.validate_array("X", X, expected_shape=(None, None))
    validator.validate_array("y", y, expected_shape=(None,))
    validator.validate_array("dates", dates, expected_shape=(None,))
    
    # Validate consistent lengths
    if not (len(X) == len(y) == len(dates)):
        raise ValidationError(
            f"Length mismatch: X={len(X)}, y={len(y)}, dates={len(dates)}"
        )
    
    # Validate minimum samples
    if len(X) < min_samples:
        raise ValidationError(f"Insufficient samples: {len(X)} < {min_samples}")
    
    # Validate y is binary
    validator.validate_binary_labels(y, "y")
    
    # Validate dates are reasonable (YYYYMMDD format)
    if dates.min() < 19000000:
        raise ValidationError(f"Dates too old: {dates.min()}")
    if dates.max() > 21000000:
        raise ValidationError(f"Dates too futuristic: {dates.max()}")
    
    return True


def validate_model_contract(model: Any, input_dim: int, test_input: Optional[np.ndarray] = None) -> bool:
    """
    Validate a compiled model can accept input and produce output
    
    Args:
        model: Model to validate
        input_dim: Expected input dimension
        test_input: Optional specific test input
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ValidationError("TensorFlow not available for model validation")
    
    if not isinstance(model, tf.keras.Model):
        raise ValidationError(f"Model must be tf.keras.Model, got {type(model)}")
    
    if model.optimizer is None:
        raise ValidationError("Model not compiled (no optimizer)")
    
    if model.loss is None:
        raise ValidationError("Model missing loss function")
    
    # Test forward pass
    if test_input is None:
        test_input = np.random.randn(1, input_dim).astype(np.float32)
    
    try:
        output = model(test_input)
        output_np = output.numpy() if hasattr(output, 'numpy') else np.array(output)
        
        if output_np.shape != (1, 1):
            raise ValidationError(f"Output shape mismatch: expected (1, 1), got {output_np.shape}")
            
        if not np.all(np.isfinite(output_np)):
            raise ValidationError("Model output contains non-finite values")
            
    except Exception as e:
        raise ValidationError(f"Model failed forward pass: {e}")
    
    return True


def validate_ensemble_contract(ensemble: callable, X: np.ndarray, expected_length: int) -> bool:
    """
    Validate an ensemble produces valid predictions
    
    Args:
        ensemble: Ensemble callable
        X: Input features
        expected_length: Expected number of predictions
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not callable(ensemble):
        raise ValidationError("Ensemble must be callable")
    
    try:
        predictions = ensemble(X)
    except Exception as e:
        raise ValidationError(f"Ensemble failed to predict: {e}")
    
    if not isinstance(predictions, np.ndarray):
        raise ValidationError(f"Ensemble must return np.ndarray, got {type(predictions)}")
    
    if len(predictions) != expected_length:
        raise ValidationError(
            f"Prediction length mismatch: expected {expected_length}, got {len(predictions)}"
        )
    
    # Validate probabilities
    ChunkValidator.validate_probabilities(predictions, "ensemble predictions")
    
    # Test determinism
    predictions2 = ensemble(X)
    if not np.allclose(predictions, predictions2):
        raise ValidationError("Ensemble not deterministic")
    
    return True