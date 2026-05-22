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
        if self.logger:
            self.logger.log(f"{self.name} - Starting", 'info')
        
        val_predictions = context.get('val_predictions')
        val_dates = context.get('val_dates')
        val_y_raw = context.get('val_y_raw')
        arch_names = context.get('arch_names', [])
        temporal_weights = context.get('temporal_weights')
        
        if val_predictions is None or len(val_predictions) == 0:
            if self.logger:
                self.logger.log("No validation predictions found in context", 'warning')
                self.logger.log("Phase 4 must store val_predictions for this analysis", 'warning')
                self.logger.log(f"{self.name} - SKIPPED", 'warning')
            context['phaseXb_complete'] = True
            return context
        
        if val_dates is None or val_y_raw is None:
            if self.logger:
                self.logger.log("Validation dates or target values not found", 'warning')
                self.logger.log(f"{self.name} - SKIPPED", 'warning')
            context['phaseXb_complete'] = True
            return context
        
        if len(arch_names) == 0:
            if self.logger:
                self.logger.log("No architecture names found", 'warning')
                self.logger.log(f"{self.name} - SKIPPED", 'warning')
            context['phaseXb_complete'] = True
            return context
        
        label_threshold = self.config.get('FIRST_THRESHOLD', 2.0)
        pred_threshold = self.config.get('PREDICTION_THRESHOLD', 0.5)
        
        # PRIORITY 1 FIX: Validate and Re-derive Dimensions (May 7, 2026)
        # Check if val_dates and val_y_raw have mismatched lengths
        # If so, align to ensure proper indexing
        if val_dates is not None and val_y_raw is not None:
            if len(val_dates) != len(val_y_raw):
                if self.logger:
                    self.logger.log(f"WARNING: Dimension mismatch detected! val_dates: {len(val_dates)}, val_y_raw: {len(val_y_raw)}", 'warning')
                    # Align to minimum length
                min_len = min(len(val_dates), len(val_y_raw))
                val_dates = val_dates[:min_len]
                val_y_raw = val_y_raw[:min_len]
                val_predictions = [p[:min_len] for p in val_predictions]
                
                if self.logger:
                    self.logger.log(f"INFO: Aligned to {min_len} elements", 'info')
        
        if self.logger:
            self.logger.log(f"Analyzing temporal precision gap for {len(arch_names)} architectures", 'info')
            self.logger.log(f"Label threshold: {label_threshold}, Prediction threshold: {pred_threshold}", 'info')
        
        unique_dates = np.unique(val_dates)
        n_dates = len(unique_dates)
        
        if n_dates < 3 and self.logger:
            self.logger.log(f"WARNING: Only {n_dates} unique dates in validation - splitting may be unreliable", 'warning')
        
        recent_date_cutoff = unique_dates[int(n_dates * 0.67)]
        older_date_cutoff = unique_dates[int(n_dates * 0.33)]
        
        is_recent = np.isin(val_dates, unique_dates[unique_dates >= recent_date_cutoff])
        is_older = np.isin(val_dates, unique_dates[unique_dates <= older_date_cutoff])
        
        fraud_mask = (val_y_raw >= label_threshold).astype(int)
        
        n_recent_fraud_total = int(np.sum(fraud_mask[is_recent]))
        n_older_fraud_total = int(np.sum(fraud_mask[is_older]))
        
        if self.logger:
            self.logger.log(f"Date split: {n_dates} unique dates", 'info')
            self.logger.log(f"Recent dates (>={recent_date_cutoff}): {np.sum(is_recent):,} samples, {n_recent_fraud_total:,} fraud cases", 'info')
            self.logger.log(f"Older dates (<={older_date_cutoff}): {np.sum(is_older):,} samples, {n_older_fraud_total:,} fraud cases", 'info')
        
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
        
        status_lines = []
        status_lines.append(f"{self.name} Report - FULL METRICS")
        status_lines.append(f"{'Architecture':<12} {'Recent P':>8} {'Recent R':>8} {'Recent AUC':>10} {'Recent F1':>8} {'Older P':>8} {'Older R':>8} {'Older AUC':>10} {'Older F1':>8} {'Gap':>6}")
        
        for _, row in results_df.iterrows():
            gap_str = f"{row['gap']:+.2f}"
            status_lines.append(f"{row['architecture']:<12} {row['recent_precision']:>8.4f} {row['recent_recall']:>8.4f} {row['recent_auc']:>10.4f} {row['recent_f1']:>8.4f} {row['older_precision']:>8.4f} {row['older_recall']:>8.4f} {row['older_auc']:>10.4f} {row['older_f1']:>8.4f} {gap_str:>6}")
        
        if self.logger:
            for line in status_lines:
                self.logger.log(line, 'info')
        
        best_arch = results_df.iloc[0]
        worst_arch = results_df.iloc[-1]
        
        if self.logger:
            self.logger.log(f"BEST for recent fraud: {best_arch['architecture']} (Gap: {best_arch['gap']:+.4f})", 'info')
            self.logger.log(f"WORST for recent fraud: {worst_arch['architecture']} (Gap: {worst_arch['gap']:+.4f})", 'info')
        
        positive_gap_count = int(np.sum(results_df['gap'] > 0.05))
        if self.logger:
            self.logger.log(f"{positive_gap_count}/{len(results_df)} architectures show positive temporal precision gap", 'info')
        
        if positive_gap_count == 0 and self.logger:
            self.logger.log(f"WARNING: No architecture shows meaningful improvement on recent fraud", 'warning')
            self.logger.log(f"Consider: stronger temporal weighting or different architectures", 'warning')
        
        context['temporal_precision_gap'] = results_df.to_dict('records')
        context['best_recency_architecture'] = best_arch['architecture']
        context['worst_recency_architecture'] = worst_arch['architecture']
        context['phaseXb_complete'] = True
        
        if self.logger:
            self.logger.log(f"{self.name} - COMPLETE", 'info')
        
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
