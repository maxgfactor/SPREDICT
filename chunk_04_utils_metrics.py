"""
Chunk 04: Utilities - Metrics
Metric calculation utilities with defensive programming
"""

import numpy as np
import warnings
from typing import List, Optional, Dict, Any


def safe_average_precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Wrapper for average_precision_score that handles edge cases
    
    Args:
        y_true: True labels
        y_pred: Predicted scores or labels
        
    Returns:
        Average precision score (0.0 if fails)
    """
    try:
        from sklearn.metrics import average_precision_score
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No positive class found in y_true")
            return average_precision_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: average_precision_score failed: {e}")
        return 0.0


def assess_model_learning(loss_history: List[float], 
                         prc_history: Optional[List[float]], 
                         patience_epochs: int) -> Dict[str, Any]:
    """
    Analyze if the model actually learned during training
    
    Args:
        loss_history: Training loss history
        prc_history: Precision-Recall AUC history
        patience_epochs: Number of epochs to wait for improvement
        
    Returns:
        Dictionary with learning assessment
    """
    issues = []
    
    if not loss_history or len(loss_history) < 5:
        return {'learned': False, 'issues': ['insufficient_history']}
    
    # Check loss convergence
    early_loss = np.mean(loss_history[:len(loss_history)//5])
    late_loss = np.mean(loss_history[-len(loss_history)//5:])
    loss_reduction = (early_loss - late_loss) / early_loss if early_loss > 0 else 0
    
    if loss_reduction < 0.05:
        issues.append("poor_convergence")
    
    # Check PRC improvement
    if prc_history and len(prc_history) > 1:
        prc_gain = prc_history[-1] - prc_history[0]
        min_improvement = 0.001 if len(prc_history) < 50 else 0.005
        if prc_gain < min_improvement:
            issues.append("poor_prc_gain")
    
    # Check for training instability
    if len(loss_history) >= 10:
        loss_std = np.std(loss_history[-10:])
        loss_mean = np.mean(loss_history[-10:])
        if loss_mean > 0 and loss_std / loss_mean > 0.1:
            issues.append("training_instability")
    
    return {'learned': len(issues) == 0, 'issues': issues}


def format_metric_value(value: float, is_percentage: bool = True) -> str:
    """
    Format metric value to 1 decimal place
    
    Args:
        value: Value to format
        is_percentage: Whether to format as percentage
        
    Returns:
        Formatted string
    """
    if is_percentage:
        return f"{value:.1%}"
    else:
        return f"{value:.1f}"


def get_trend_indicator(current: float, previous: Optional[float], 
                       threshold: float = 0.01) -> str:
    """
    Get trend indicator based on current vs previous value
    
    Args:
        current: Current value
        previous: Previous value (None for no previous)
        threshold: Change threshold
        
    Returns:
        Trend indicator string
    """
    if previous is None:
        return ""
    
    change = (current - previous) / previous if previous != 0 else 0
    
    if change > threshold:
        return "[UP]"
    elif change < -threshold:
        return "[DOWN]"
    else:
        return "->"


def format_phase_1_5_standardized(precision_value: float, prc_value: float,
                                 iterations_completed: int) -> str:
    """
    Standardized formatter for Phase 1.5 results
    
    Args:
        precision_value: Precision score
        prc_value: Precision-Recall AUC
        iterations_completed: Number of iterations
        
    Returns:
        Formatted report string
    """
    precision_formatted = format_metric_value(precision_value, True)
    prc_formatted = format_metric_value(prc_value, False)
    context_info = f"OBJECTIVE: MAXIMIZE, ITERATIONS: {iterations_completed}"
    
    report = f"[PHASE_1_5] PRECISION: {precision_formatted} ({context_info}) → optimization complete | PRC: {prc_formatted}"
    return report


def format_standard_metric_report(phase_name: str, primary_metric_type: str,
                                 primary_value: float,
                                 secondary_metric_type: Optional[str] = None,
                                 secondary_value: Optional[float] = None,
                                 target: float = 0.95,
                                 previous_primary: Optional[float] = None) -> str:
    """
    Standardized metric reporting for phases
    
    Args:
        phase_name: Phase name
        primary_metric_type: Type of primary metric
        primary_value: Primary metric value
        secondary_metric_type: Type of secondary metric
        secondary_value: Secondary metric value
        target: Target value
        previous_primary: Previous primary value
        
    Returns:
        Formatted report string
    """
    is_percentage = primary_metric_type == "PRECISION"
    primary_formatted = format_metric_value(primary_value, is_percentage)
    
    if primary_metric_type == "PRECISION":
        progress = (primary_value - target) / target * 100
        progress_str = f" (TARGET: {target:.1%}, PROGRESS: {progress:+.1f}%)"
    else:
        progress_str = ""
    
    trend = get_trend_indicator(primary_value, previous_primary)
    
    report = f"[{phase_name}] {primary_metric_type}: {primary_formatted}{progress_str}{trend}"
    
    if secondary_metric_type and secondary_value is not None:
        secondary_formatted = format_metric_value(secondary_value, secondary_metric_type == "PRECISION")
        report += f" | {secondary_metric_type}: {secondary_formatted}"
    
    return report


def calculate_precision_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                                    threshold: float) -> float:
    """
    Calculate precision at a specific threshold
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        threshold: Threshold to apply
        
    Returns:
        Precision score
    """
    y_pred = (y_scores > threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def validate_metric_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """
    Ensure metric inputs are valid
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or scores
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(y_true, np.ndarray), "y_true must be np.ndarray"
    assert isinstance(y_pred, np.ndarray), "y_pred must be np.ndarray"
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    assert set(np.unique(y_true)).issubset({0, 1}), "y_true must be binary"
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing metrics utilities...")
    
    # Test safe_average_precision_score
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    score = safe_average_precision_score(y_true, y_scores)
    assert 0 <= score <= 1
    print(f"[PASS] Average precision: {score:.4f}")
    
    # Test assess_model_learning
    loss_history = [0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30]
    prc_history = [0.5, 0.6, 0.7, 0.75, 0.78, 0.80]
    result = assess_model_learning(loss_history, prc_history, 10)
    assert 'learned' in result
    assert 'issues' in result
    print(f"[PASS] Learning assessment: {result}")
    
    # Test format_metric_value
    assert format_metric_value(0.955, True) == "95.5%"
    assert format_metric_value(0.955, False) == "1.0"
    print("[PASS] Metric formatting works")
    
    # Test format_phase_1_5_standardized
    report = format_phase_1_5_standardized(0.95, 0.87, 50)
    assert "PHASE_1_5" in report
    assert "95.0%" in report
    print(f"[PASS] Phase report: {report}")
    
    # Test calculate_precision_at_threshold
    precision = calculate_precision_at_threshold(y_true, y_scores, 0.5)
    assert 0 <= precision <= 1
    print(f"[PASS] Precision at threshold 0.5: {precision:.4f}")
    
    print("\n[PASS] All metrics utility tests passed")