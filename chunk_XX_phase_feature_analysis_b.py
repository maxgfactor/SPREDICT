"""
Chunk XXb: Phase Xb - Temporal Precision Gap Analysis
Analyzes which architectures best predict RECENT fraud vs OLDER fraud
Compares precision on most recent dates vs older dates in validation set
Goal: Measure temporal weighting effectiveness and identify architectures that excel at recent fraud prediction
Runs after Phase 4 (which stores validation predictions in context)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

import chunk_15_phase_base as phase_base


class PhaseXb_TemporalCorrelation(phase_base.BasePhase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Phase Xb: Temporal Precision Gap Analysis"
    
    def execute(self, context: Dict) -> Dict:
        print("=" * 80)
        print(f"{self.name} - Starting")
        print("=" * 80)
        
        val_predictions = context.get('val_predictions')
        val_dates = context.get('val_dates')
        val_y_raw = context.get('val_y_raw')
        arch_names = context.get('arch_names', [])
        temporal_weights = context.get('temporal_weights')
        
        if val_predictions is None or len(val_predictions) == 0:
            print("[PhaseXb] No validation predictions found in context")
            print(f"[PhaseXb] Phase 4 must store val_predictions for this analysis")
            print(f"{self.name} - SKIPPED")
            context['phaseXb_complete'] = True
            return context
        
        if val_dates is None or val_y_raw is None:
            print("[PhaseXb] Validation dates or target values not found")
            print(f"{self.name} - SKIPPED")
            context['phaseXb_complete'] = True
            return context
        
        if len(arch_names) == 0:
            print("[PhaseXb] No architecture names found")
            print(f"{self.name} - SKIPPED")
            context['phaseXb_complete'] = True
            return context
        
        label_threshold = self.config.get('FIRST_THRESHOLD', 2.0)
        pred_threshold = self.config.get('PREDICTION_THRESHOLD', 0.5)
        
        print(f"[PhaseXb] Analyzing temporal precision gap for {len(arch_names)} architectures")
        print(f"[PhaseXb] Label threshold: {label_threshold}, Prediction threshold: {pred_threshold}")
        
        unique_dates = np.unique(val_dates)
        n_dates = len(unique_dates)
        
        if n_dates < 3:
            print(f"[PhaseXb] WARNING: Only {n_dates} unique dates in validation - splitting may be unreliable")
        
        recent_date_cutoff = unique_dates[int(n_dates * 0.67)]
        older_date_cutoff = unique_dates[int(n_dates * 0.33)]
        
        is_recent = np.isin(val_dates, unique_dates[unique_dates >= recent_date_cutoff])
        is_older = np.isin(val_dates, unique_dates[unique_dates <= older_date_cutoff])
        
        fraud_mask = (val_y_raw >= label_threshold).astype(int)
        
        n_recent_fraud_total = int(np.sum(fraud_mask[is_recent]))
        n_older_fraud_total = int(np.sum(fraud_mask[is_older]))
        
        print(f"[PhaseXb] Date split: {n_dates} unique dates")
        print(f"[PhaseXb] Recent dates (>={recent_date_cutoff}): {np.sum(is_recent):,} samples, {n_recent_fraud_total:,} fraud cases")
        print(f"[PhaseXb] Older dates (<={older_date_cutoff}): {np.sum(is_older):,} samples, {n_older_fraud_total:,} fraud cases")
        print()
        
        results = []
        for i, (arch_name, preds) in enumerate(zip(arch_names, val_predictions)):
            preds = np.asarray(preds).flatten()
            
            recent_fraud = fraud_mask[is_recent] == 1
            older_fraud = fraud_mask[is_older] == 1
            
            recent_preds = (preds[is_recent] >= pred_threshold).astype(int)
            older_preds = (preds[is_older] >= pred_threshold).astype(int)
            
            n_rf = int(np.sum(recent_fraud))
            n_of = int(np.sum(older_fraud))
            
            # Calculate all metrics for recent period
            recent_tp = int(np.sum((recent_preds == 1) & (recent_fraud == 1)))
            recent_fp = int(np.sum((recent_preds == 1) & (recent_fraud == 0)))
            recent_tn = int(np.sum((recent_preds == 0) & (recent_fraud == 0)))
            recent_fn = int(np.sum((recent_preds == 0) & (recent_fraud == 1)))
            
            recent_precision = recent_tp / (recent_tp + recent_fp) if (recent_tp + recent_fp) > 0 else 0.0
            recent_recall = recent_tp / (recent_tp + recent_fn) if (recent_tp + recent_fn) > 0 else 0.0
            recent_f1 = 2 * recent_precision * recent_recall / (recent_precision + recent_recall) if (recent_precision + recent_recall) > 0 else 0.0
            
            # Calculate all metrics for older period
            older_tp = int(np.sum((older_preds == 1) & (older_fraud == 1)))
            older_fp = int(np.sum((older_preds == 1) & (older_fraud == 0)))
            older_tn = int(np.sum((older_preds == 0) & (older_fraud == 0)))
            older_fn = int(np.sum((older_preds == 0) & (older_fraud == 1)))
            
            older_precision = older_tp / (older_tp + older_fp) if (older_tp + older_fp) > 0 else 0.0
            older_recall = older_tp / (older_tp + older_fn) if (older_tp + older_fn) > 0 else 0.0
            older_f1 = 2 * older_precision * older_recall / (older_precision + older_recall) if (older_precision + older_recall) > 0 else 0.0
            
            # Calculate AUC (using raw predictions, not binary)
            try:
                from sklearn.metrics import roc_auc_score
                recent_auc = roc_auc_score(fraud_mask[is_recent], preds[is_recent]) if len(np.unique(fraud_mask[is_recent])) > 1 else 0.5
                older_auc = roc_auc_score(fraud_mask[is_older], preds[is_older]) if len(np.unique(fraud_mask[is_older])) > 1 else 0.5
            except:
                recent_auc = 0.0
                older_auc = 0.0
            
            gap = recent_precision - older_precision
            
            if gap > 0.10:
                interpretation = "Strong recency"
            elif gap > 0.05:
                interpretation = "Mild improvement"
            elif gap > 0.0:
                interpretation = "Slight improvement"
            elif gap > -0.05:
                interpretation = "No change"
            elif gap > -0.10:
                interpretation = "Mild decline"
            else:
                interpretation = "Significant decline"
            
            results.append({
                'architecture': arch_name,
                'recent_precision': recent_precision,
                'recent_recall': recent_recall,
                'recent_auc': recent_auc,
                'recent_f1': recent_f1,
                'recent_tp': recent_tp,
                'recent_fp': recent_fp,
                'recent_tn': recent_tn,
                'recent_fn': recent_fn,
                'older_precision': older_precision,
                'older_recall': older_recall,
                'older_auc': older_auc,
                'older_f1': older_f1,
                'older_tp': older_tp,
                'older_fp': older_fp,
                'older_tn': older_tn,
                'older_fn': older_fn,
                'gap': gap,
                'n_recent_fraud': n_rf,
                'n_older_fraud': n_of,
                'interpretation': interpretation,
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('gap', ascending=False)
        
        print("=" * 100)
        print(f"{self.name} Report - FULL METRICS")
        print("=" * 100)
        print(f"{'Architecture':<12} {'Recent P':>8} {'Recent R':>8} {'Recent AUC':>10} {'Recent F1':>8} {'Older P':>8} {'Older R':>8} {'Older AUC':>10} {'Older F1':>8} {'Gap':>6}")
        print("-" * 100)
        
        for _, row in results_df.iterrows():
            gap_str = f"{row['gap']:+.2f}"
            print(f"{row['architecture']:<12} {row['recent_precision']:>8.4f} {row['recent_recall']:>8.4f} {row['recent_auc']:>10.4f} {row['recent_f1']:>8.4f} {row['older_precision']:>8.4f} {row['older_recall']:>8.4f} {row['older_auc']:>10.4f} {row['older_f1']:>8.4f} {gap_str:>6}")
        
        print("=" * 80)
        
        best_arch = results_df.iloc[0]
        worst_arch = results_df.iloc[-1]
        
        print(f"\n[PhaseXb] BEST for recent fraud: {best_arch['architecture']} (Gap: {best_arch['gap']:+.4f})")
        print(f"[PhaseXb] WORST for recent fraud: {worst_arch['architecture']} (Gap: {worst_arch['gap']:+.4f})")
        
        positive_gap_count = int(np.sum(results_df['gap'] > 0.05))
        print(f"[PhaseXb] {positive_gap_count}/{len(results_df)} architectures show positive temporal precision gap")
        
        if positive_gap_count == 0:
            print(f"[PhaseXb] WARNING: No architecture shows meaningful improvement on recent fraud")
            print(f"[PhaseXb] Consider: stronger temporal weighting or different architectures")
        
        context['temporal_precision_gap'] = results_df.to_dict('records')
        context['best_recency_architecture'] = best_arch['architecture']
        context['worst_recency_architecture'] = worst_arch['architecture']
        context['phaseXb_complete'] = True
        
        print(f"\n{self.name} - COMPLETE")
        
        return context


def run_phase_xb(config: Dict, context: Dict) -> Dict:
    phase = PhaseXb_TemporalCorrelation(config)
    return phase.execute(context)


if __name__ == "__main__":
    print("Testing Phase Xb...")
    
    np.random.seed(42)
    n_samples = 10000
    
    dates = np.sort(np.random.choice(range(20251001, 20251022), n_samples))
    
    y_raw = np.random.randn(n_samples) * 10
    y_raw = np.clip(y_raw, -50, 100)
    
    arch_names = ['Dense', 'VAE', 'CNN', 'RNN', 'LSTM', 'Transformer']
    val_predictions = [np.random.rand(n_samples) * 0.3 + 0.2 + i * 0.02 for i in range(6)]
    
    for i in range(len(val_predictions)):
        mask_recent = dates >= 20251015
        val_predictions[i][mask_recent] += 0.05
    
    context = {
        'val_predictions': val_predictions,
        'val_dates': dates,
        'val_y_raw': y_raw,
        'arch_names': arch_names,
    }
    
    config = {
        'FIRST_THRESHOLD': 2.0,
        'PREDICTION_THRESHOLD': 0.5,
    }
    
    phase = PhaseXb_TemporalCorrelation(config)
    result = phase.execute(context)
    
    print(f"\n[PASS] Phase Xb test passed")
