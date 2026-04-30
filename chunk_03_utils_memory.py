"""
Chunk 03: Utilities - Memory Management
Memory-efficient operations and chunking utilities
"""

import numpy as np
import pandas as pd
import psutil
import os
from typing import List, Optional, Callable, Any


def check_memory_usage(operation_name: str = "operation", memory_limit_gb: float = 70) -> dict:
    """
    Check current memory usage and return status
    
    Args:
        operation_name: Name of operation for logging
        memory_limit_gb: Memory limit in GB
        
    Returns:
        Dictionary with memory status
    """
    assert isinstance(operation_name, str), "operation_name must be string"
    assert memory_limit_gb > 0, "memory_limit_gb must be positive"
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        
        return {
            'memory_gb': memory_gb,
            'memory_limit_gb': memory_limit_gb,
            'is_critical': memory_gb > memory_limit_gb,
            'operation': operation_name
        }
    except Exception as e:
        return {
            'memory_gb': 0,
            'memory_limit_gb': memory_limit_gb,
            'is_critical': False,
            'operation': operation_name,
            'error': str(e)
        }


def get_optimal_chunk_size(total_rows: int, feature_count: int,
                          memory_limit_gb: float = 62, 
                          safety_factor: float = 0.85) -> int:
    """
    Calculate optimal chunk size for memory-efficient operations
    
    Args:
        total_rows: Total number of rows
        feature_count: Number of features
        memory_limit_gb: Memory limit in GB
        safety_factor: Safety factor (0-1)
        
    Returns:
        Optimal chunk size
    """
    assert total_rows > 0, "total_rows must be positive"
    assert feature_count > 0, "feature_count must be positive"
    assert 0 < safety_factor <= 1, "safety_factor must be in (0, 1]"
    
    # Estimate memory per row (8 bytes per float64 * feature_count)
    bytes_per_row = 8 * feature_count
    memory_limit_bytes = memory_limit_gb * (1024 ** 3) * safety_factor
    
    # Calculate chunk size
    chunk_size = int(memory_limit_bytes / bytes_per_row)
    
    # Don't exceed total rows
    chunk_size = min(chunk_size, total_rows)
    
    # Minimum chunk size of 1000
    chunk_size = max(chunk_size, 1000)
    
    return chunk_size


def memory_efficient_vstack(arrays: List[np.ndarray], 
                           operation_name: str = "array combination",
                           max_memory_gb: float = 62) -> np.ndarray:
    """
    Stack arrays vertically with memory monitoring
    
    Args:
        arrays: List of arrays to stack
        operation_name: Name for logging
        max_memory_gb: Maximum memory to use
        
    Returns:
        Stacked array
    """
    assert isinstance(arrays, list), "arrays must be a list"
    assert len(arrays) > 0, "arrays list cannot be empty"
    assert all(isinstance(arr, np.ndarray) for arr in arrays), "All elements must be np.ndarray"
    
    # Check memory before operation
    mem_status = check_memory_usage(operation_name, max_memory_gb)
    
    if mem_status['is_critical']:
        raise MemoryError(f"Memory critical before {operation_name}: {mem_status['memory_gb']:.2f} GB")
    
    try:
        result = np.vstack(arrays)
        return result
    except MemoryError as e:
        raise MemoryError(f"Memory error during {operation_name}: {e}")


def memory_efficient_df_to_array(df: pd.DataFrame, 
                                chunk_size: Optional[int] = None,
                                operation_name: str = "DataFrame to array conversion") -> np.ndarray:
    """
    Convert DataFrame to numpy array with memory monitoring
    
    Args:
        df: DataFrame to convert
        chunk_size: Optional chunk size
        operation_name: Name for logging
        
    Returns:
        Numpy array
    """
    assert isinstance(df, pd.DataFrame), "df must be pd.DataFrame"
    assert len(df) > 0, "DataFrame cannot be empty"
    
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(len(df), len(df.columns))
    
    # For small dataframes, convert directly
    if len(df) <= chunk_size:
        return df.to_numpy(dtype=np.float32)
    
    # For large dataframes, convert in chunks
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunks.append(chunk.to_numpy(dtype=np.float32))
    
    return memory_efficient_vstack(chunks, operation_name)


def chunked_fit_transform(transformer: Any, X: np.ndarray, 
                         chunk_size: Optional[int] = None,
                         operation_name: str = "sklearn operation") -> np.ndarray:
    """
    Apply sklearn transform in chunks to manage memory
    
    Args:
        transformer: Sklearn transformer with fit_transform method
        X: Input data
        chunk_size: Optional chunk size
        operation_name: Name for logging
        
    Returns:
        Transformed array
    """
    assert hasattr(transformer, 'fit_transform'), "transformer must have fit_transform method"
    assert isinstance(X, np.ndarray), "X must be np.ndarray"
    assert len(X) > 0, "X cannot be empty"
    
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(len(X), X.shape[1] if X.ndim > 1 else 1)
    
    # For small datasets, process directly
    if len(X) <= chunk_size:
        return transformer.fit_transform(X)
    
    # For large datasets, process in chunks
    # First fit on a sample
    sample_size = min(chunk_size, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    transformer.fit(X[sample_indices])
    
    # Then transform in chunks
    chunks = []
    for i in range(0, len(X), chunk_size):
        chunk = X[i:i+chunk_size]
        chunks.append(transformer.transform(chunk))
    
    return memory_efficient_vstack(chunks, operation_name)


def estimate_memory_usage(n_samples: int, n_features: int, 
                         dtype_bytes: int = 8) -> float:
    """
    Estimate memory usage in GB for given data size
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        dtype_bytes: Bytes per element
        
    Returns:
        Estimated memory in GB
    """
    total_bytes = n_samples * n_features * dtype_bytes
    return total_bytes / (1024 ** 3)


def validate_memory_function(func: Callable, test_input: Any, 
                            expected_output_type: type) -> bool:
    """
    Test memory utility returns expected type
    
    Args:
        func: Function to test
        test_input: Input to function
        expected_output_type: Expected return type
        
    Returns:
        True if valid
    """
    result = func(test_input)
    assert isinstance(result, expected_output_type), (
        f"Expected {expected_output_type}, got {type(result)}"
    )
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing memory utilities...")
    
    # Test check_memory_usage
    mem_status = check_memory_usage("test")
    assert isinstance(mem_status, dict)
    assert 'memory_gb' in mem_status
    print(f"[PASS] Memory check: {mem_status['memory_gb']:.3f} GB")
    
    # Test get_optimal_chunk_size
    chunk_size = get_optimal_chunk_size(100000, 50)
    assert isinstance(chunk_size, int)
    assert chunk_size > 0
    print(f"[PASS] Optimal chunk size for 100Kx50: {chunk_size}")
    
    # Test memory_efficient_vstack
    arrays = [np.random.randn(100, 10) for _ in range(5)]
    result = memory_efficient_vstack(arrays, "test vstack")
    assert result.shape == (500, 10)
    print(f"[PASS] vstack result shape: {result.shape}")
    
    # Test memory_efficient_df_to_array
    df = pd.DataFrame(np.random.randn(1000, 20))
    arr = memory_efficient_df_to_array(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1000, 20)
    print(f"[PASS] DataFrame conversion shape: {arr.shape}")
    
    # Test chunked_fit_transform
    from sklearn.preprocessing import StandardScaler
    X = np.random.randn(5000, 10)
    scaler = StandardScaler()
    X_scaled = chunked_fit_transform(scaler, X, chunk_size=1000)
    assert X_scaled.shape == X.shape
    print(f"[PASS] Chunked transform shape: {X_scaled.shape}")
    
    print("\n[PASS] All memory utility tests passed")