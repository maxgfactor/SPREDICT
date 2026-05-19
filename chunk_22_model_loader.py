"""
Chunk 22: Model Loader Utility
Load trained models from disk for prediction
"""

import tensorflow as tf
import os
import json
from typing import Dict, Optional, Tuple, Any
from chunk_02_utils_logging import Logger


def load_model(arch_name: str, models_path: str = './saved_models') -> tf.keras.Model:
    """
    Load a trained model from disk.
    
    Args:
        arch_name: Architecture name (e.g., 'RNN', 'LSTM')
        models_path: Path to saved models directory
        
    Returns:
        Loaded Keras model
    """
    model_path = f"{models_path}/{arch_name}_model.keras"
    # Load with safe_mode=False to allow Lambda layers (needed for Transformer)
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    return model


def load_preprocessing_params(models_path: str = './saved_models') -> Dict[str, Any]:
    """
    Load preprocessing parameters needed for inference.
    
    Args:
        models_path: Path to saved models directory
        
    Returns:
        Dictionary containing:
        - temporal_weights: dict mapping dates to weights
        - feature_names: list of feature column names
        - all_dates: dict mapping dates to indices
        - split_date: string date used for train/val split
    """
    params = {}
    
    # Load temporal weights
    tw_path = os.path.join(models_path, 'temporal_weights.json')
    if os.path.exists(tw_path):
        with open(tw_path, 'r') as f:
            params['temporal_weights'] = json.load(f)
    
    # Load feature names
    fn_path = os.path.join(models_path, 'feature_names.json')
    if os.path.exists(fn_path):
        with open(fn_path, 'r') as f:
            params['feature_names'] = json.load(f)
    
    # Load all dates
    ad_path = os.path.join(models_path, 'all_dates.json')
    if os.path.exists(ad_path):
        with open(ad_path, 'r') as f:
            params['all_dates'] = json.load(f)
    
    # Load split date
    sd_path = os.path.join(models_path, 'split_date.txt')
    if os.path.exists(sd_path):
        with open(sd_path, 'r') as f:
            params['split_date'] = f.read().strip()
    
    return params


def load_model_metadata(arch_name: str, models_path: str = './saved_models') -> Optional[Dict]:
    """
    Load metadata for a specific architecture.
    
    Args:
        arch_name: Architecture name (e.g., 'RNN', 'LSTM')
        models_path: Path to saved models directory
        
    Returns:
        Dictionary containing optimal_threshold, best_hyperparams, best_val_precision
        or None if metadata file doesn't exist
    """
    metadata_path = os.path.join(models_path, f'{arch_name}_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def load_models_with_metadata(models_path: str = './saved_models', logger=None) -> Tuple[Dict[str, tf.keras.Model], Dict[str, Dict]]:
    """
    Load all saved models along with their metadata.
    
    Args:
        models_path: Path to saved models directory
        
    Returns:
        Tuple of (models_dict, metadata_dict)
        - models_dict: {arch_name: model}
        - metadata_dict: {arch_name: {optimal_threshold, best_hyperparams, best_val_precision}}
    """
    models = {}
    metadata = {}
    sklearn_archs = {'CatBoost', 'LightGBM', 'XGBoost'}
    if logger is None:
        from chunk_02_utils_logging import Logger
        logger = Logger({'LOG_VERBOSITY': 1})

    if os.path.exists(models_path):
        for filename in os.listdir(models_path):
            if filename.endswith('_model.keras'):
                arch_name = filename.replace('_model.keras', '')
                filepath = os.path.join(models_path, filename)

                if arch_name in sklearn_archs:
                    try:
                        import joblib
                        from chunk_11_models_sklearn import SklearnModelWrapper
                        sklearn_model = joblib.load(filepath)
                        wrapped = SklearnModelWrapper(sklearn_model)
                        wrapped._is_fitted = True
                        models[arch_name] = wrapped
                        logger.log(f"Loaded {arch_name} model (sklearn)", 'info')
                    except Exception as e:
                        logger.log(f"Failed to load {arch_name} model: {e}", 'warning')
                else:
                    try:
                        models[arch_name] = load_model(arch_name, models_path)
                        logger.log(f"Loaded {arch_name} model", 'info')
                    except Exception as e:
                        logger.log(f"Failed to load {arch_name} model: {e}", 'warning')

                # Load metadata for this architecture
                meta = load_model_metadata(arch_name, models_path)
                if meta:
                    metadata[arch_name] = meta
                    logger.log(f"Loaded {arch_name} metadata: Label_Threshold={meta.get('optimal_threshold')}", 'info')
    
    return models, metadata


def load_all_models(models_path: str = './saved_models', logger=None) -> dict:
    """
    Load all available trained models from disk.
    
    Args:
        models_path: Path to saved models directory
        
    Returns:
        Dictionary of {arch_name: model}
    """
    models = {}
    if logger is None:
        from chunk_02_utils_logging import Logger
        logger = Logger({'LOG_VERBOSITY': 1})
    if os.path.exists(models_path):
        for filename in os.listdir(models_path):
            if filename.endswith('_model.keras'):
                arch_name = filename.replace('_model.keras', '')
                try:
                    models[arch_name] = load_model(arch_name, models_path)
                    logger.log(f"Loaded {arch_name} model", 'info')
                except Exception as e:
                    logger.log(f"Failed to load {arch_name} model: {e}", 'warning')
    
    return models


def get_best_model(models_path: str = './saved_models') -> tuple:
    """
    Get the best performing model based on saved precision metrics.
    
    Args:
        models_path: Path to saved models directory
        
    Returns:
        Tuple of (arch_name, model, precision)
    """
    best_precision = 0.0
    best_model = None
    best_arch = None
    
    # Check for precision metrics file
    metrics_path = os.path.join(models_path, 'model_precision.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        for arch, precision in metrics.items():
            if precision > best_precision:
                best_precision = precision
                best_arch = arch
        
        if best_arch:
            best_model = load_model(best_arch, models_path)
    
    return best_arch, best_model, best_precision


if __name__ == '__main__':
    # Test loading
    models = load_all_models()
    print(f"Loaded models: {list(models.keys())}")
