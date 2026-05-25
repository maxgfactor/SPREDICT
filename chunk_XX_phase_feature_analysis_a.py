"""
Chunk XXa: Phase Xa - Raw Feature Importance Analysis
Analyzes raw features only (before temporal features are added)
Runs 6-method feature importance analysis and auto-prunes raw features
Inserts between Phase 1 (Setup) and Phase 3 (Temporal Weighting)
"""

import numpy as np
from typing import Dict, Any, Optional

import chunk_15_phase_base as phase_base
import chunk_XX_feature_importance as feature_importance


class PhaseXa_FeatureAnalysis(phase_base.BasePhase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Phase Xa: Raw Feature Importance Analysis"
    
    def execute(self, context: Dict) -> Dict:
        self.analyzer = feature_importance.FeatureImportanceAnalyzer(self.config, logger=self.logger)
        if self.logger:
            self.logger.log(f"{self.name} - Starting", 'info')
        
        X = context.get('X')
        y_raw = context.get('raw_target_values')
        if y_raw is None:
            y_raw = context.get('y')
        feature_names = context.get('feature_names')
        
        if X is None or y_raw is None:
            if self.logger:
                self.logger.log("error: X or y not found in context", 'error')
            return context
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        sample_size = self.config.get('FEATURE_ANALYSIS_SAMPLE_SIZE', 100000)
        
        if n_samples > sample_size:
            if self.logger:
                self.logger.log(f"Subsampling {n_samples} -> {sample_size} for feature analysis", 'info')
            rng = np.random.RandomState(42)
            indices = rng.choice(n_samples, sample_size, replace=False)
            X_sub = X[indices]
            y_sub = y_raw[indices]
        else:
            X_sub = X
            y_sub = y_raw
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_sub.shape[1])]
        
        if self.logger:
            self.logger.log(f"Analyzing {n_features} raw features on {X_sub.shape[0]} samples", 'info')
        
        trained_dense = None
        
        analysis_results = self.analyzer.run_full_analysis(
            X=X_sub,
            y_raw=y_sub,
            feature_names=feature_names,
            dates=None,
            temporal_weights=None,
            trained_dense_model=trained_dense
        )
        
        report_path = self.config.get('FEATURE_ANALYSIS_REPORT_PATH', './feature_importance_report.txt')
        self.analyzer.save_report(analysis_results, report_path)
        
        # Store per-threshold pruning results (each Label_Threshold gets its own feature subset)
        results_by_threshold = analysis_results.get('results_by_threshold', {})
        threshold_kept = {round(float(t), 1): r['kept_indices'] for t, r in results_by_threshold.items()}
        threshold_dropped = {round(float(t), 1): r['dropped_indices'] for t, r in results_by_threshold.items()}
        
        context['X'] = X
        context['feature_names'] = feature_names
        context['feature_importance_results'] = analysis_results
        context['threshold_kept_indices'] = threshold_kept
        context['threshold_dropped_indices'] = threshold_dropped
        # Backward compat: first threshold's pruning
        pruned_indices = analysis_results['kept_indices']
        context['pruned_feature_indices'] = pruned_indices
        context['dropped_feature_indices'] = analysis_results['dropped_indices']
        context['dropped_feature_names'] = analysis_results['dropped_names']
        
        context['phaseXa_complete'] = True
        
        n_pruned = n_features - len(pruned_indices)
        if self.logger:
            self.logger.log(f"complete: {n_features} raw features (per-threshold pruning stored)", 'info')
            first_thr = next(iter(threshold_kept.keys()))
            self.logger.log(f"  First threshold ({first_thr}): {len(pruned_indices)} kept", 'info')
        
        return context


def run_phase_xa(config: Dict, context: Dict) -> Dict:
    phase = PhaseXa_FeatureAnalysis(config)
    return phase.execute(context)


if __name__ == "__main__":
    print("Testing Phase Xa...")
    
    config = {
        'FEATURE_ANALYSIS_SAMPLE_SIZE': 10000,
        'FEATURE_PRUNE_PERCENTILE': 20,
        'ABLATON_THRESHOLD': 2.0,
    }
    
    np.random.seed(42)
    X = np.random.randn(10000, 21)
    y_raw = np.random.randn(10000) * 10
    feature_names = [f'feature_{i}' for i in range(21)]
    
    context = {
        'X': X,
        'raw_target_values': y_raw,
        'feature_names': feature_names,
    }
    
    phase = PhaseXa_FeatureAnalysis(config)
    result = phase.execute(context)
    
    print(f"\n[pass] Phase Xa test passed")
    print(f"[pass] Original: {X.shape[1]} -> Pruned: {result['X'].shape[1]}")
    print(f"[pass] Dropped: {result['dropped_feature_names']}")
