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


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """
    Inverse of Option C sign-transform (sign * log1p(|y|)).
    
    Args:
        y_log: Transformed values (sign * log1p(|y|))
        
    Returns:
        Original scale values
        
    Example:
        y_log = -4.61 → sign=-1, magnitude_log=4.61 → expm1(4.61)=99.75 → -99.75
    """
    sign = np.sign(y_log)
    magnitude_log = np.abs(y_log)
    magnitude = np.expm1(magnitude_log)
    return sign * magnitude


def get_prediction_percentiles(predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive prediction percentiles for diagnostics.
    
    Args:
        predictions: Array of prediction values
        
    Returns:
        Dictionary with percentile values
    """
    if len(predictions) == 0:
        return {'p1': 0.0, 'p5': 0.0, 'p10': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    p = np.percentile(predictions, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        'p1': float(p[0]), 'p5': float(p[1]), 'p10': float(p[2]), 'p25': float(p[3]),
        'p50': float(p[4]), 'p75': float(p[5]), 'p90': float(p[6]), 'p95': float(p[7]),
        'p99': float(p[8]), 'max': float(predictions.max())
    }


def get_prediction_histogram(predictions: np.ndarray, num_bins: int = 20) -> Dict[str, Any]:
    """
    Calculate histogram of prediction values for diagnostics.
    
    Args:
        predictions: Array of prediction values
        num_bins: Number of histogram bins
        
    Returns:
        Dictionary with histogram data
    """
    if len(predictions) == 0:
        return {'bins': [], 'counts': []}
    hist, bin_edges = np.histogram(predictions, bins=num_bins)
    return {
        'bin_edges': bin_edges.tolist(),
        'counts': hist.tolist(),
        'num_bins': num_bins
    }


def format_diagnostic_string(predictions: np.ndarray, prefix: str = "") -> str:
    """
    Format comprehensive diagnostic string for predictions.
    
    Args:
        predictions: Array of prediction values
        prefix: Prefix for log message (e.g., "[baseline]", "[hpo]")
        
    Returns:
        Formatted diagnostic string
    """
    if len(predictions) == 0:
        return f"{prefix} No predictions"
    
    stats = get_prediction_percentiles(predictions)
    hist = get_prediction_histogram(predictions)
    
    # Format percentiles
    result = f"{prefix} percentiles: p1={stats['p1']:.4f}, p5={stats['p5']:.4f}, p10={stats['p10']:.4f}, p25={stats['p25']:.4f}, p50={stats['p50']:.4f}, p75={stats['p75']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}, p99={stats['p99']:.4f}, max={stats['max']:.4f}"
    
    # Add histogram info (first and last 3 bins)
    if len(hist['counts']) > 0:
        result += f" | histogram: bins[{hist['counts'][0]},{hist['counts'][1]},{hist['counts'][2]}...{hist['counts'][-3]},{hist['counts'][-2]},{hist['counts'][-1]}]"
    
    return result


def analyze_loss_distribution(loss_history: List[float]) -> Dict[str, float]:
    """
    Analyze loss distribution during training.
    
    Args:
        loss_history: List of loss values per epoch
        
    Returns:
        Dictionary with loss statistics
    """
    if not loss_history or len(loss_history) == 0:
        return {'current': 0.0, 'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'epochs': 0}
    
    arr = np.array(loss_history)
    return {
        'current': float(arr[-1]) if len(arr) > 0 else 0.0,
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'epochs': len(arr)
    }


def calibrate_predictions(predictions: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
    """
    Apply isotonic regression for probability calibration.
    
    Args:
        predictions: Raw model predictions (probabilities)
        y_binary: Binary ground truth labels for calibration
        
    Returns:
        Calibrated predictions
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds='clip')
        calibrated = ir.fit_transform(predictions, y_binary)
        return calibrated
    except Exception as e:
        print(f"Warning: Calibration failed: {e}")
        return predictions


def apply_temperature_scaling(predictions: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling for probability calibration.
    
    Args:
        predictions: Raw predictions (probabilities 0-1)
        temperature: Temperature parameter (>1 softens, <1 sharpens)
        
    Returns:
        Temperature-scaled predictions
    """
    if predictions.max() <= 1.0 and predictions.min() >= 0.0:
        logits = np.log(np.clip(predictions, 1e-10, 1-1e-10) / np.clip(1-predictions, 1e-10, 1-1e-10))
        scaled_logits = logits / temperature
        return 1 / (1 + np.exp(-scaled_logits))
    return predictions


def calculate_temporal_drift(segment_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Calculate temporal drift from segment metrics.
    
    Args:
        segment_metrics: Dictionary with keys 'early', 'mid', 'late', each containing
                         precision/recall metrics from temporal segments.
                         
    Returns:
        Dictionary with drift analysis including precision_change, recall_change,
        interpretation, and stability_score.
    """
    if not segment_metrics or len(segment_metrics) < 2:
        return {'drift_detected': False, 'interpretation': 'Insufficient segments for drift analysis'}
    
    segments_order = ['early', 'mid', 'late']
    available_segments = [s for s in segments_order if s in segment_metrics]
    
    if len(available_segments) < 2:
        return {'drift_detected': False, 'interpretation': 'Need at least 2 segments for drift analysis'}
    
    first_segment = segment_metrics[available_segments[0]]
    last_segment = segment_metrics[available_segments[-1]]
    
    precision_change = last_segment.get('precision', 0) - first_segment.get('precision', 0)
    recall_change = last_segment.get('recall', 0) - first_segment.get('recall', 0)
    f1_change = last_segment.get('f1', 0) - first_segment.get('f1', 0)
    
    precision_pct_change = (precision_change / max(first_segment.get('precision', 0.001), 0.001)) * 100
    
    if abs(precision_pct_change) > 20:
        interpretation = f"Significant precision drift: {precision_pct_change:+.1f}% change"
        drift_detected = True
    elif abs(precision_pct_change) > 10:
        interpretation = f"Moderate precision drift: {precision_pct_change:+.1f}% change"
        drift_detected = True
    else:
        interpretation = f"Stable precision: {precision_pct_change:+.1f}% change"
        drift_detected = False
    
    if precision_change < 0 and recall_change < 0:
        interpretation += " (performance degrading over time)"
    elif precision_change > 0 and recall_change > 0:
        interpretation += " (performance improving over time)"
    
    precision_values = [segment_metrics[s].get('precision', 0) for s in available_segments]
    stability_score = 1.0 - (np.std(precision_values) / max(np.mean(precision_values), 0.001)) if len(precision_values) > 1 else 1.0
    
    return {
        'drift_detected': drift_detected,
        'precision_change': float(precision_change),
        'recall_change': float(recall_change),
        'f1_change': float(f1_change),
        'precision_pct_change': float(precision_pct_change),
        'interpretation': interpretation,
        'stability_score': float(stability_score),
        'segments_analyzed': len(available_segments)
    }


def calculate_permutation_importance(model: Any, X: np.ndarray, y_true: np.ndarray,
                                     scoring_metric: str = 'precision',
                                     n_iterations: int = 5,
                                     pred_threshold: float = 0.5) -> Dict[int, float]:
    """
    Calculate permutation importance for each feature.
    
    Args:
        model: Trained model with predict() method
        X: Feature matrix (n_samples, n_features)
        y_true: True binary labels
        scoring_metric: Metric to use ('precision', 'recall', 'f1', 'auc')
        n_iterations: Number of permutations per feature
        pred_threshold: Threshold for converting probabilities to binary
        
    Returns:
        Dictionary with feature indices as keys and importance scores as values
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    try:
        # Get baseline predictions
        y_pred = model.predict(X).flatten()
        y_binary = (y_pred >= pred_threshold).astype(int)
        
        # Calculate baseline score
        if scoring_metric == 'precision':
            baseline = precision_score(y_true, y_binary, zero_division=0)
        elif scoring_metric == 'recall':
            baseline = recall_score(y_true, y_binary, zero_division=0)
        elif scoring_metric == 'f1':
            baseline = f1_score(y_true, y_binary, zero_division=0)
        elif scoring_metric == 'auc':
            baseline = roc_auc_score(y_true, y_pred)
        else:
            baseline = precision_score(y_true, y_binary, zero_division=0)
        
        # Calculate importance for each feature
        importance = {}
        n_features = X.shape[1]
        
        for i in range(n_features):
            scores = []
            for _ in range(n_iterations):
                # Shuffle feature
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                
                # Get predictions with shuffled feature
                y_pred_perm = model.predict(X_permuted).flatten()
                y_binary_perm = (y_pred_perm >= pred_threshold).astype(int)
                
                # Calculate score
                if scoring_metric == 'precision':
                    score = precision_score(y_true, y_binary_perm, zero_division=0)
                elif scoring_metric == 'recall':
                    score = recall_score(y_true, y_binary_perm, zero_division=0)
                elif scoring_metric == 'f1':
                    score = f1_score(y_true, y_binary_perm, zero_division=0)
                elif scoring_metric == 'auc':
                    score = roc_auc_score(y_true, y_pred_perm)
                else:
                    score = precision_score(y_true, y_binary_perm, zero_division=0)
                
                scores.append(score)
            
            # Importance = baseline - permuted score (higher = more important)
            importance[i] = float(baseline - np.mean(scores))
        
        return importance
    
    except Exception as e:
        print(f"Warning: Permutation importance calculation failed: {e}")
        return {}


def calculate_prediction_entropy(predictions: np.ndarray) -> float:
    """
    Calculate entropy of predictions (measure of model uncertainty).
    
    Args:
        predictions: Array of prediction probabilities
        
    Returns:
        float: Entropy value (0 = certain, higher = more uncertain)
    """
    try:
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        entropy = -np.sum(predictions * np.log(predictions) + (1 - predictions) * np.log(1 - predictions))
        return float(entropy)
    except Exception as e:
        return 0.0


def calculate_logit_compression(predictions: np.ndarray) -> float:
    """
    Calculate logit compression ratio (model confidence measure).
    
    Args:
        predictions: Array of prediction probabilities
        
    Returns:
        float: max/mean ratio (higher = more confident)
    """
    try:
        mask = predictions > 0.1
        if mask.sum() == 0:
            return 0.0
        positive_preds = predictions[mask]
        max_pred = positive_preds.max()
        mean_pred = positive_preds.mean()
        return float(max_pred / max(mean_pred, 1e-10))
    except Exception as e:
        return 0.0


def calculate_ks_test(positive_preds: np.ndarray, negative_preds: np.ndarray) -> Dict[str, Any]:
    """
    Kolmogorov-Smirnov test for distribution separation between positive and negative predictions.
    
    Args:
        positive_preds: Predictions for positive class samples
        negative_preds: Predictions for negative class samples
        
    Returns:
        Dictionary with ks_stat, p_value, interpretation
    """
    try:
        from scipy.stats import ks_2samp
        if len(positive_preds) < 2 or len(negative_preds) < 2:
            return {'ks_stat': 0.0, 'p_value': 1.0, 'interpretation': 'insufficient_data'}
        ks_stat, p_value = ks_2samp(positive_preds, negative_preds)
        if ks_stat > 0.5:
            interpretation = 'excellent_separation'
        elif ks_stat > 0.3:
            interpretation = 'good_separation'
        elif ks_stat > 0.1:
            interpretation = 'moderate_separation'
        else:
            interpretation = 'poor_separation'
        return {'ks_stat': float(ks_stat), 'p_value': float(p_value), 'interpretation': interpretation}
    except Exception as e:
        return {'ks_stat': 0.0, 'p_value': 1.0, 'interpretation': f'error: {e}'}


def calculate_bhattacharyya_distance(positive_preds: np.ndarray, negative_preds: np.ndarray) -> float:
    """
    Calculate Bhattacharyya distance for class separation measure.
    
    Args:
        positive_preds: Predictions for positive class samples
        negative_preds: Predictions for negative class samples
        
    Returns:
        float: Bhattacharyya distance (higher = better separation)
    """
    try:
        if len(positive_preds) < 2 or len(negative_preds) < 2:
            return 0.0
        hist_pos, bin_edges = np.histogram(positive_preds, bins=20, density=True)
        hist_neg, _ = np.histogram(negative_preds, bins=bin_edges, density=True)
        hist_pos = hist_pos + 1e-10
        hist_neg = hist_neg + 1e-10
        bc = np.sum(np.sqrt(hist_pos * hist_neg))
        bh_dist = -np.log(bc)
        return float(bh_dist)
    except Exception as e:
        return 0.0


def calculate_snr(predictions: np.ndarray, dates: np.ndarray) -> Dict[str, float]:
    """
    Calculate Signal-to-Noise ratio per time segment.
    
    Args:
        predictions: Array of predictions
        dates: Array of date values for temporal segmentation
        
    Returns:
        Dictionary with SNR per segment (early, mid, late)
    """
    try:
        unique_dates = np.unique(dates)
        if len(unique_dates) < 3:
            return {'early': 0.0, 'mid': 0.0, 'late': 0.0, 'overall': 0.0}
        seg_size = len(unique_dates) // 3
        segments = {}
        for seg_name, seg_range in [('early', slice(0, seg_size)), ('mid', slice(seg_size, seg_size*2)), ('late', slice(seg_size*2, None))]:
            seg_dates = unique_dates[seg_range]
            seg_mask = np.isin(dates, seg_dates)
            seg_preds = predictions[seg_mask]
            if len(seg_preds) > 0 and np.std(seg_preds) > 1e-10:
                segments[seg_name] = float(np.mean(seg_preds) / np.std(seg_preds))
            else:
                segments[seg_name] = 0.0
        if len(predictions) > 0 and np.std(predictions) > 1e-10:
            segments['overall'] = float(np.mean(predictions) / np.std(predictions))
        else:
            segments['overall'] = 0.0
        return segments
    except Exception as e:
        return {'early': 0.0, 'mid': 0.0, 'late': 0.0, 'overall': 0.0}


def calculate_mutual_information(predictions: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate mutual information between predictions and true labels.
    
    Args:
        predictions: Array of predictions
        y_true: Binary ground truth labels
        
    Returns:
        float: Mutual information score
    """
    try:
        from sklearn.metrics import mutual_info_score
        pred_binned = (predictions > 0.5).astype(int)
        mi = mutual_info_score(y_true, pred_binned)
        return float(mi)
    except Exception as e:
        return 0.0


def calculate_psi(actual: np.ndarray, expected: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate Population Stability Index (PSI) for distribution drift detection.
    
    Args:
        actual: Actual/predicted distribution
        expected: Expected/baseline distribution
        n_bins: Number of bins for bucketing
        
    Returns:
        Dictionary with psi_value, drift_status, interpretation
    """
    try:
        if len(actual) == 0 or len(expected) == 0:
            return {'psi_value': 0.0, 'drift_status': 'insufficient_data', 'interpretation': 'No data'}
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        actual_hist, _ = np.histogram(actual, bins=bins)
        expected_hist, _ = np.histogram(expected, bins=bins)
        actual_pct = (actual_hist + 1e-10) / actual_hist.sum()
        expected_pct = (expected_hist + 1e-10) / expected_hist.sum()
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi_value = np.sum(psi_values)
        if psi_value < 0.1:
            drift_status = 'stable'
            interpretation = 'No significant drift'
        elif psi_value < 0.2:
            drift_status = 'moderate_drift'
            interpretation = 'Moderate distribution shift'
        else:
            drift_status = 'significant_drift'
            interpretation = 'Significant distribution shift'
        return {'psi_value': float(psi_value), 'drift_status': drift_status, 'interpretation': interpretation}
    except Exception as e:
        return {'psi_value': 0.0, 'drift_status': 'error', 'interpretation': str(e)}


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
        return "[up]"
    elif change < -threshold:
        return "[down]"
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
    
    report = f"[phase_1_5] PRECISION: {precision_formatted} ({context_info}) → optimization complete | PRC: {prc_formatted}"
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
    print(f"[pass] Average precision: {score:.4f}")
    
    # Test assess_model_learning
    loss_history = [0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30]
    prc_history = [0.5, 0.6, 0.7, 0.75, 0.78, 0.80]
    result = assess_model_learning(loss_history, prc_history, 10)
    assert 'learned' in result
    assert 'issues' in result
    print(f"[pass] Learning assessment: {result}")
    
    # Test format_metric_value
    assert format_metric_value(0.955, True) == "95.5%"
    assert format_metric_value(0.955, False) == "1.0"
    print("[pass] Metric formatting works")
    
    # Test format_phase_1_5_standardized
    report = format_phase_1_5_standardized(0.95, 0.87, 50)
    assert "PHASE_1_5" in report
    assert "95.0%" in report
    print(f"[pass] Phase report: {report}")
    
    # Test calculate_precision_at_threshold
    precision = calculate_precision_at_threshold(y_true, y_scores, 0.5)
    assert 0 <= precision <= 1
    print(f"[pass] precision at threshold 0.5: {precision:.4f}")
    
    print("\n[pass] All metrics utility tests passed")