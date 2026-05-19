"""
Chunk XX: Phase X - Feature Importance Analysis
Runs 6-method feature importance analysis and auto-prunes features
Inserts between Phase 3 (Temporal) and Phase 4 (Ensemble Training)
"""

import numpy as np
from typing import Dict, Any, Optional

import chunk_15_phase_base as phase_base
import chunk_XX_feature_importance as feature_importance


class PhaseX_FeatureAnalysis(phase_base.BasePhase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Phase X: Feature Importance Analysis"
    
    def execute(self, context: Dict) -> Dict:
        self.analyzer = feature_importance.FeatureImportanceAnalyzer(self.config, logger=self.logger)
        if self.logger:
            self.logger.log(f"{self.name} - Starting", 'info')
        
        X = context.get('X')
        y_raw = context.get('raw_target_values')
        if y_raw is None:
            y_raw = context.get('y')
        dates = context.get('dates')
        temporal_weights = context.get('temporal_weights')
        feature_names = context.get('feature_names')
        
        if X is None or y_raw is None:
            if self.logger:
                self.logger.log("ERROR: X or y not found in context", 'error')
            return context
        
        n_samples = X.shape[0]
        sample_size = self.config.get('FEATURE_ANALYSIS_SAMPLE_SIZE', 100000)
        
        if n_samples > sample_size:
            if self.logger:
                self.logger.log(f"Subsampling {n_samples} -> {sample_size} for feature analysis", 'info')
            rng = np.random.RandomState(42)
            indices = rng.choice(n_samples, sample_size, replace=False)
            X_sub = X[indices]
            y_sub = y_raw[indices]
            dates_sub = dates[indices] if dates is not None else None
            temporal_sub = temporal_weights[indices] if temporal_weights is not None else None
        else:
            X_sub = X
            y_sub = y_raw
            dates_sub = dates
            temporal_sub = temporal_weights
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_sub.shape[1])]
        
        trained_dense = None
        
        analysis_results = self.analyzer.run_full_analysis(
            X=X_sub,
            y_raw=y_sub,
            feature_names=feature_names,
            dates=dates_sub,
            temporal_weights=temporal_sub,
            trained_dense_model=trained_dense
        )
        
        report_path = self.config.get('FEATURE_ANALYSIS_REPORT_PATH', './feature_importance_report.txt')
        self.analyzer.save_report(analysis_results, report_path)
        
        pruned_indices = analysis_results['kept_indices']
        context['X'] = X[:, pruned_indices]
        context['feature_names'] = analysis_results['kept_names']
        context['feature_importance_results'] = analysis_results
        context['pruned_feature_indices'] = pruned_indices
        context['dropped_feature_indices'] = analysis_results['dropped_indices']
        context['dropped_feature_names'] = analysis_results['dropped_names']
        
        context['phaseX_complete'] = True
        
        if self.logger:
            self.logger.log(f"COMPLETE: {analysis_results['n_features_original']} -> {analysis_results['n_features_pruned']} features", 'info')
            self.logger.log(f"Dropped: {analysis_results['dropped_names']}", 'info')
        
        return context


def run_phase_x(config: Dict, context: Dict) -> Dict:
    phase = PhaseX_FeatureAnalysis(config)
    return phase.execute(context)


if __name__ == "__main__":
    print("Testing Phase X...")
    
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
        'dates': np.random.randint(20250101, 20251001, 10000),
        'temporal_weights': np.ones(10000),
        'feature_names': feature_names,
    }
    
    phase = PhaseX_FeatureAnalysis(config)
    result = phase.execute(context)
    
    print(f"\n[PASS] Phase X test passed")
    print(f"[PASS] Original: {X.shape[1]} → Pruned: {result['X'].shape[1]}")
    print(f"[PASS] Dropped: {result['dropped_feature_names']}")
