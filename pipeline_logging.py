"""
pipeline_logging.py — Logging Utility
Refactored from chunk_02_utils_logging.py (2026-08-07).
Logger class and formatting utilities.

## Purpose
Provides standardized logging across the stock analysis pipeline with automatic
source code references for easy troubleshooting.

## Source References
All log messages automatically include the source filename in brackets [filename.py]
for easy traceability and debugging.

## Usage
self.logger.log("message", "info")  # Output: [filename.py] [info] message
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class Logger:
    """Logging utility for stock analysis pipeline"""
    
    log_count = 0
    
    def __init__(self, config: Dict):
        """
        Initialize logger
        
        Args:
            config: Configuration dictionary with LOG_VERBOSITY
        """
        self.verbosity = config['LOG_VERBOSITY']  # 0=quiet, 1=normal, 2=verbose
    
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
                            # Extract base filename (e.g., "pipeline_logging.py")
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
        
        signal_rate = float(np.mean(y))
        total_samples = len(y)
        signal_count = int(np.sum(y))
        normal_count = total_samples - signal_count
        
        imbalance_ratio = (max(signal_count, normal_count) / min(signal_count, normal_count)
                          if min(signal_count, normal_count) > 0 else float('inf'))
        
        print("[class] Class Distribution:")
        print(f"  Total samples: {total_samples}")
        print(f"  Signal cases: {signal_count} ({signal_rate:.1%})")
        print(f"  Normal cases: {normal_count} ({1-signal_rate:.1%})")
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
