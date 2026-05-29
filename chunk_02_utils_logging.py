"""
Chunk 02: Utilities - Logging
Logger class and formatting utilities

## Purpose
Provides standardized logging across the fraud detection pipeline with automatic
source code references for easy troubleshooting.

## Source References
All log messages automatically include the source filename in brackets [filename.py]
for easy traceability and debugging.

## Usage
self.logger.log("message", "info")  # Output: [chunk_XX_filename.py] [info] message
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class Logger:
    """Logging utility for fraud detection pipeline"""
    
    log_count = 0
    
    def __init__(self, config: Dict):
        """
        Initialize logger
        
        Args:
            config: Configuration dictionary with LOG_VERBOSITY
        """
        self.verbosity = config.get('LOG_VERBOSITY', 1)  # 0=quiet, 1=normal, 2=verbose
    
    def log(self, message: str, level: str = 'info', source: str = None):
        """
        Log a message with automatic source code reference
        
        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error')
            source: Optional custom source label. If not provided, auto-detects from caller.
                    Can use custom labels like "PHASE5" or "CONFIG" for grouped output.
        """
        if level == 'error' or self.verbosity >= 1:
            Logger.log_count += 1
            # Auto-detect source file if not provided
            if source is None:
                try:
                    import inspect
                    # Get caller's frame
                    frame = inspect.currentframe()
                    if frame is not None:
                        caller_frame = frame.f_back
                        if caller_frame is not None:
                            # Get filename
                            filename = caller_frame.f_code.co_filename
                            # Extract base filename (e.g., "chunk_19_phase_5_optimization.py")
                            source = os.path.basename(filename)
                        else:
                            source = "logger"
                    else:
                        source = "logger"
                except Exception:
                    source = "logger"
            
            # Format: [log_count] [source] [level] message
            level_prefix = {"info": "[info]", "warning": "[warning]", "error": "[error]"}.get(level, "[info]")
            print(f"[LN{Logger.log_count:<7}] [{source}] {level_prefix} {message}", flush=True)
    
    def format_metric(self, value: float, is_percentage: bool = True) -> str:
        """
        Standardize to 1 decimal place
        
        Args:
            value: Numeric value to format
            is_percentage: Whether to format as percentage
            
        Returns:
            Formatted string
        """
        if is_percentage:
            return f"{value:.1%}"  # 5.5%, 95.0%
        else:
            return f"{value:.1f}"   # 0.1, 1.1 (for PRC)
    
    def get_trend_indicator(self, current: float, previous: Optional[float], 
                           threshold: float = 0.01) -> str:
        """
        Consistent trend calculation
        
        Args:
            current: Current value
            previous: Previous value (None if no previous)
            threshold: Change threshold for trend detection
            
        Returns:
            Trend indicator string
        """
        if previous is None:
            return ""
        change = (current - previous) / previous if previous != 0 else 0
        if change > threshold:
            return " [up]"  # Improving
        elif change < -threshold:
            return " [down]"  # Declining
        else:
            return " →"   # Stable
    
    def format_phase_1_5_standardized(self, precision_value: float, prc_value: float,
                                     iterations_completed: int) -> str:
        """
        Standardized formatter for PHASE 2: PRECISION first, PRC second
        
        Args:
            precision_value: Precision score
            prc_value: Precision-Recall AUC
            iterations_completed: Number of iterations
            
        Returns:
            Formatted report string
        """
        precision_formatted = self.format_metric(precision_value, True)
        prc_formatted = self.format_metric(prc_value, False)
        context_info = f"OBJECTIVE: MAXIMIZE, ITERATIONS: {iterations_completed}"
        
        report = f"[phase_1_5] PRECISION: {precision_formatted} ({context_info}) → optimization complete | PRC: {prc_formatted}"
        return report
    
    def format_standard_metric_report(self, phase_name: str, primary_metric_type: str,
                                     primary_value: float,
                                     secondary_metric_type: Optional[str] = None,
                                     secondary_value: Optional[float] = None,
                                     target: float = 0.95,
                                     previous_primary: Optional[float] = None) -> str:
        """
        Standardized metric reporting for PHASE 2, 3, 4
        
        Args:
            phase_name: Name of the phase
            primary_metric_type: Type of primary metric (e.g., "PRECISION")
            primary_value: Primary metric value
            secondary_metric_type: Type of secondary metric
            secondary_value: Secondary metric value
            target: Target value for progress calculation
            previous_primary: Previous primary value for trend
            
        Returns:
            Formatted report string
        """
        is_percentage = primary_metric_type == "PRECISION"
        primary_formatted = self.format_metric(primary_value, is_percentage)
        
        if primary_metric_type == "PRECISION":
            progress = (primary_value - target) / target * 100
            progress_str = f" (TARGET: {target:.1%}, PROGRESS: {progress:+.1f}%)"
        else:
            progress_str = ""
        
        trend = self.get_trend_indicator(primary_value, previous_primary)
        
        report = f"[{phase_name}] {primary_metric_type}: {primary_formatted}{progress_str}{trend}"
        
        if secondary_metric_type and secondary_value is not None:
            secondary_formatted = self.format_metric(
                secondary_value, secondary_metric_type == "PRECISION"
            )
            report += f" | {secondary_metric_type}: {secondary_formatted}"
        
        return report
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def log_feature_quality_metrics(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Log comprehensive feature quality statistics
        
        Args:
            X: Feature matrix
            feature_names: Optional list of feature names
        """
        if self.verbosity < 2:
            return
        
        print("[stat] Feature Quality Analysis:")
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        for i, name in enumerate(feature_names):  # Show all features
            feature_data = X[:, i]
            stats = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'skewness': float(self._calculate_skewness(feature_data)),
                'kurtosis': float(self._calculate_kurtosis(feature_data)),
                'missing': float(np.isnan(feature_data).sum() / len(feature_data))
            }
            print(f"  {name}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, "
                  f"skew={stats['skewness']:.3f}, kurt={stats['kurtosis']:.3f}, "
                  f"missing={stats['missing']:.1%}")
        
        if X.shape[1] > 25:
            print(f"  ... and {X.shape[1] - 25} more features")
    
    def log_class_distribution(self, y: np.ndarray):
        """
        Log class distribution and imbalance metrics
        
        Args:
            y: Binary labels array
        """
        if self.verbosity < 1:
            return
        
        fraud_rate = float(np.mean(y))
        total_samples = len(y)
        fraud_count = int(np.sum(y))
        normal_count = total_samples - fraud_count
        
        imbalance_ratio = (max(fraud_count, normal_count) / min(fraud_count, normal_count)
                          if min(fraud_count, normal_count) > 0 else float('inf'))
        
        print("[class] Class Distribution:")
        print(f"  Total samples: {total_samples}")
        print(f"  Fraud cases: {fraud_count} ({fraud_rate:.1%})")
        print(f"  Normal cases: {normal_count} ({1-fraud_rate:.1%})")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    def log_temporal_coverage(self, dates: np.ndarray):
        """
        Log temporal distribution and coverage metrics
        
        Args:
            dates: Array of dates (YYYYMMDD format)
        """
        if self.verbosity < 1:
            return
        
        dates_numeric = pd.to_numeric(dates, errors='coerce')
        valid_dates = dates_numeric[~np.isnan(dates_numeric)]
        
        if len(valid_dates) == 0:
            self.log("[warning] No valid dates found", 'warning')
            return
        
        self.log(f"[date] Temporal Coverage:  Date range: {int(valid_dates.min())} to {int(valid_dates.max())}  Unique dates: {len(np.unique(valid_dates))}  Total samples: {len(valid_dates)}", 'info')
    
    def log_system_performance(self, phase_timings: Dict[str, float], 
                              total_time: float, memory_usage: float):
        """
        Log system performance metrics
        
        Args:
            phase_timings: Dictionary of phase names to execution times
            total_time: Total execution time
            memory_usage: Memory usage in GB
        """
        print("\n[time] Performance Metrics:")
        print(f"  Total execution time: {total_time:.2f}s")
        for phase, timing in phase_timings.items():
            print(f"  {phase}: {timing:.2f}s")
        if memory_usage > 0:
            print(f"  Peak memory usage: {memory_usage:.2f} GB")
    
    def log_data_flow_metrics(self, data_flow_stages: Dict[str, Dict]):
        """
        Log data flow metrics across pipeline stages
        
        Args:
            data_flow_stages: Dictionary of stage names to metrics
        """
        print("\n[stat] Data Flow Metrics:")
        for stage, metrics in data_flow_stages.items():
            print(f"  {stage}: {metrics}")
    
    def log_final_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Log comprehensive final evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        print("\n[target] Final Evaluation:")
        print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"  precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  f1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
        
        try:
            auc = roc_auc_score(y_true, y_pred)
            print(f"  auc: {auc:.4f}")
        except ValueError:
            print("  auc: N/A (only one class present)")


def validate_logger_instance(logger: Logger) -> bool:
    """
    Ensure Logger has all required methods
    
    Args:
        logger: Logger instance to validate
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    required_methods = [
        'log', 'format_metric', 'get_trend_indicator',
        'format_phase_1_5_standardized', 'format_standard_metric_report',
        'log_class_distribution', 'log_feature_quality_metrics',
        'log_temporal_coverage', 'log_system_performance',
        'log_data_flow_metrics', 'log_final_evaluation'
    ]
    
    for method in required_methods:
        assert hasattr(logger, method), f"Logger missing method: {method}"
        assert callable(getattr(logger, method)), f"Logger.{method} is not callable"
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing Logger...")
    
    config = {'LOG_VERBOSITY': 2}
    logger = Logger(config)
    
    # Validate instance
    validate_logger_instance(logger)
    
    # Test methods
    logger.log("Test message", "info")
    print(f"Format metric (0.955): {logger.format_metric(0.955, True)}")
    print(f"Trend indicator: {logger.get_trend_indicator(0.9, 0.8)}")
    
    # Test with sample data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    dates = np.random.randint(20220101, 20230101, 100)
    
    logger.log_feature_quality_metrics(X)
    logger.log_class_distribution(y)
    logger.log_temporal_coverage(dates)
    
    print("\n[pass] All Logger tests passed")