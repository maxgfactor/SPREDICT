"""
Chunk 20: Pipeline Main
Main orchestrator for the fraud detection pipeline
"""

import os
import sys
import time
import warnings

# CPU Mode Configuration - Disabled GPU (2026-02-28)
if sys.platform == 'linux':
    # GPU/CUDA paths commented out - CPU mode
    # cuda_paths = [
    #     '/usr/lib/wsl/lib',
    #     '/usr/local/cuda-13.0/lib64',
    #     '/usr/local/cuda-12.5/lib64',
    #     '/lib/x86_64-linux-gnu',
    # ]
    # existing_path = os.environ.get('LD_LIBRARY_PATH', '')
    # cuda_path_str = ':'.join([p for p in cuda_paths if os.path.exists(p)])
    # os.environ['LD_LIBRARY_PATH'] = cuda_path_str + ':' + existing_path if existing_path else cuda_path_str
    
    # Set CUDA home (prefer CUDA 12.5 for better compatibility)
    # if os.path.exists('/usr/local/cuda-12.5'):
    #     os.environ['CUDA_HOME'] = '/usr/local/cuda-12.5'
    # elif os.path.exists('/usr/local/cuda-13.0'):
    #     os.environ['CUDA_HOME'] = '/usr/local/cuda-13.0'
    
    # Force CPU mode
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Suppress TensorFlow CPU optimization warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from typing import Dict, List, Type

from chunk_01_config import CONFIG, validate_config_structure
from chunk_02_utils_logging import Logger
from chunk_15_phase_base import BasePhase
from chunk_16_phase_1_setup import Phase1_PipelineSetup, validate_phase1_output
from chunk_17_phase_3_temporal import Phase3_TemporalWeighting, validate_phase3_output
from chunk_18_phase_4_ensemble import Phase4_NeuralEnsemble, validate_phase4_output
from chunk_19_phase_5_optimization import Phase5_PredictionOptimization, validate_phase5_output
from chunk_XX_phase_feature_analysis_a import PhaseXa_FeatureAnalysis
from chunk_XX_phase_feature_analysis_b import PhaseXb_TemporalCorrelation


class PipelineOrchestrator:
    """Orchestrates the complete fraud detection pipeline"""
    
    def __init__(self, config: Dict):
        """
        Initialize orchestrator
        
        Args:
            config: Configuration dictionary
        """
        # Validate config
        validate_config_structure(config)
        self.config = config
        self.logger = Logger(config)
        self.phase_timings = {}
    
    def run(self) -> Dict:
        """
        Run the complete pipeline
        
        Returns:
            Final context with all results
        """
        start_time = time.time()
        
        print("[RUNNING] Starting Fraud Detection Pipeline...")
        print(f"   Data path: {self.config['DATA_PATH']}")
        print(f"   Sample size: {self.config['SAMPLE_SIZE']}")
        print()
        
        # Phase execution sequence
        phase_sequence = [
            ('Phase 1: Pipeline Setup', Phase1_PipelineSetup, validate_phase1_output),
            ('Phase 3: Temporal Weighting', Phase3_TemporalWeighting, validate_phase3_output),
            ('Phase Xa: Raw Feature Importance', PhaseXa_FeatureAnalysis, None),
            ('Phase 4: Neural Ensemble', Phase4_NeuralEnsemble, validate_phase4_output),
            ('Phase Xb: Temporal Precision Gap', PhaseXb_TemporalCorrelation, None),
            ('Phase 5: Prediction Optimization', Phase5_PredictionOptimization, validate_phase5_output),
        ]
        
        # Initialize context
        context = {}
        
        # Execute phases in sequence
        for phase_name, PhaseClass, validator in phase_sequence:
            phase_start = time.time()
            print(f"\n{'='*60}")
            print(f"Running {phase_name}")
            print('='*60)
            
            try:
                # Create and execute phase
                phase = PhaseClass(self.config)
                result = phase.execute(context)
                
                # Validate phase output
                try:
                    if validator is not None:
                        validator(result)
                        print(f"[PASS] {phase_name} validation passed")
                    else:
                        print(f"[SKIP] {phase_name} validation skipped (no validator)")
                except AssertionError as e:
                    print(f"[WARNING]  {phase_name} validation warning: {e}")
                
                # Update context with phase results
                context.update(result)
                
                # Record timing
                phase_time = time.time() - phase_start
                self.phase_timings[phase_name] = phase_time
                print(f"[TIME]  {phase_name} completed in {phase_time:.2f}s")
                
            except Exception as e:
                print(f"[ERROR] {phase_name} failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Pipeline failed at {phase_name}") from e
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Log final summary
        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print('='*60)
        print(f"[TIME] Total execution time: {total_time:.2f}s")
        print("\nPhase timings:")
        for phase, timing in self.phase_timings.items():
            print(f"   {phase}: {timing:.2f}s")
        
        # Log final metrics
        if 'final_metrics' in context:
            metrics = context['final_metrics']
            # Handle both dict and list formats
            if isinstance(metrics, list) and len(metrics) > 0:
                # Use the first (best) architecture's metrics
                metrics = metrics[0]
            if isinstance(metrics, dict):
                print(f"\n[STAT] Final Results:")
                print(f"   Precision: {metrics.get('precision', 0):.4f}")
                print(f"   Recall: {metrics.get('recall', 0):.4f}")
                print(f"   F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"   AUC: {metrics.get('auc', 0):.4f}")
        
        # =========================================================================
        # METRICS REVIEW FRAMEWORK
        # =========================================================================
        print(f"\n{'='*60}")
        print("METRICS REVIEW REPORT")
        print('='*60)
        
        # Get architecture metrics from Phase 4
        arch_metrics = context.get('arch_final_metrics', [])
        
        if arch_metrics:
            # Sort by precision (descending)
            sorted_metrics = sorted(arch_metrics, key=lambda x: x.get('P', 0), reverse=True)
            
            print(f"\n[ARCHITECTURE PERFORMANCE] (sorted by Val Precision)")
            print("-" * 60)
            
            ensemble_threshold = self.config.get('ENSEMBLE_MIN_PRECISION', 0.40)
            
            for i, m in enumerate(sorted_metrics, 1):
                arch = m.get('arch', 'Unknown')
                p = m.get('P', 0)
                r = m.get('R', 0)
                auc = m.get('AUC', 0)
                tp = m.get('TP', 0)
                fp = m.get('FP', 0)
                
                status = "✓" if p >= ensemble_threshold else "✗"
                print(f"{i}. {arch:15s} P={p:.4f} R={r:.4f} AUC={auc:.4f} TP={tp:5d} FP={fp:5d} {status}")
            
            # Identify issues
            print(f"\n[ISSUES IDENTIFIED]")
            print("-" * 60)
            
            issues = []
            for m in sorted_metrics:
                arch = m.get('arch', 'Unknown')
                p = m.get('P', 0)
                tp = m.get('TP', 0)
                fp = m.get('FP', 0)
                pred_total = tp + fp
                
                if p == 0:
                    issues.append(f"- {arch}: PRECISION=0 (model failing completely)")
                elif p < ensemble_threshold:
                    issues.append(f"- {arch}: PRECISION={p:.4f} below threshold {ensemble_threshold}")
                elif pred_total < 100:
                    issues.append(f"- {arch}: Too few predictions ({pred_total})")
            
            if issues:
                for issue in issues:
                    print(issue)
            else:
                print("- No issues found")
            
            # =========================================================================
            # STANDARDIZED METRICS TABLE (CSV FORMAT)
            # =========================================================================
            print(f"\n[STANDARDIZED METRICS TABLE]")
            print("-" * 60)
            
            # Get inference metrics from Phase 5
            inference_metrics = context.get('architecture_results', [])
            
            # Get train/val metrics from Phase 4
            train_val_metrics = context.get('arch_final_metrics', [])
            
            # Build CSV output with enhanced fields
            csv_lines = []
            csv_lines.append("Architecture,Phase,Loss,Epochs,Precision,Recall,AUC,F1,TP,FP,TN,FN,MaxPred,MeanPred,StdPred,PctAboveThresh,BestEpoch,TrainingTime,LabelThresh,ThresholdSource,HPO_Trials,HPO_Improvement,KeyHyperparams,TrainLoss,ValLoss,LossDelta,MCC,PRAUC,Specificity,BalancedAccuracy,PredictionThreshold")
            
            # Known architecture order
            arch_order = ['VAE', 'Dense', 'CNN', 'RNN', 'LSTM', 'Transformer']
            
            # Loss function mapping (from HPO metadata or config)
            loss_map = {}
            for m in train_val_metrics:
                arch = m.get('arch', '')
                loss_map[arch] = 'binary_crossentropy'  # Default assumption (most are now BCE)
            
            # Get epochs from train_val_metrics
            epochs_map = {}
            for m in train_val_metrics:
                arch = m.get('arch', '')
                epochs_map[arch] = m.get('epochs_trained', 0)
            
            for arch in arch_order:
                # Find train/val metrics
                train_m = next((m for m in train_val_metrics if m.get('arch') == arch), None)
                
                # Find inference metrics
                inf_m = next((m for m in inference_metrics if m.get('architecture') == arch), None)
                
                # Get loss and epochs
                loss = loss_map.get(arch, 'unknown')
                epochs = epochs_map.get(arch, 0) if train_m else 0
                
                # Train metrics
                if train_m:
                    csv_lines.append(
                        f"{arch},Train,{loss},{epochs},"
                        f"{train_m.get('train_P', 0):.4f},"
                        f"{train_m.get('train_R', 0):.4f},"
                        f"{train_m.get('train_AUC', 0):.4f},"
                        f"{train_m.get('train_F1', 0):.4f},"
                        f"{train_m.get('train_TP', 0)},"
                        f"{train_m.get('train_FP', 0)},"
                        f"{train_m.get('train_TN', 0)},"
                        f"{train_m.get('train_FN', 0)},"
                        f"{train_m.get('val_max_pred', 0):.4f},"
                        f"{train_m.get('train_mean_pred', 0):.4f},"
                        f"{train_m.get('train_std_pred', 0):.4f},"
                        f"N/A,"
                        f"{train_m.get('best_epoch', 0)},"
                        f"{train_m.get('training_time', 0):.1f},"
                        f"{train_m.get('final_label_threshold', train_m.get('optimal_label_threshold', 0)):.1f},"
                        f"{train_m.get('threshold_source', 'pre_hpo')},"
                        f"{train_m.get('hpo_trials', 0)},"
                        f"{train_m.get('hpo_improvement', 0):.4f},"
                        f"{train_m.get('key_hyperparams', 'N/A')},"
                        f"{train_m.get('train_loss', 0):.4f},"
                        f"{train_m.get('val_loss', 0):.4f},"
                        f"{train_m.get('loss_delta', 0):.4f},"
                        f"{train_m.get('train_mcc', 0):.4f},"
                        f"{train_m.get('train_prauc', 0):.4f},"
                        f"{train_m.get('train_specificity', 0):.4f},"
                        f"{train_m.get('train_balanced_acc', 0):.4f},"
                        f"0.5"
                    )
                else:
                    csv_lines.append(f"{arch},Train,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
                
                # Val metrics
                if train_m:
                    csv_lines.append(
                        f"{arch},Val,{loss},{epochs},"
                        f"{train_m.get('P', 0):.4f},"
                        f"{train_m.get('R', 0):.4f},"
                        f"{train_m.get('AUC', 0):.4f},"
                        f"{train_m.get('F1', 0):.4f},"
                        f"{train_m.get('TP', 0)},"
                        f"{train_m.get('FP', 0)},"
                        f"{train_m.get('TN', 0)},"
                        f"{train_m.get('FN', 0)},"
                        f"{train_m.get('val_max_pred', 0):.4f},"
                        f"{train_m.get('val_mean_pred', 0):.4f},"
                        f"{train_m.get('val_std_pred', 0):.4f},"
                        f"N/A,"
                        f"{train_m.get('best_epoch', 0)},"
                        f"{train_m.get('training_time', 0):.1f},"
                        f"{train_m.get('final_label_threshold', train_m.get('optimal_label_threshold', 0)):.1f},"
                        f"{train_m.get('threshold_source', 'pre_hpo')},"
                        f"{train_m.get('hpo_trials', 0)},"
                        f"{train_m.get('hpo_improvement', 0):.4f},"
                        f"{train_m.get('key_hyperparams', 'N/A')},"
                        f"{train_m.get('train_loss', 0):.4f},"
                        f"{train_m.get('val_loss', 0):.4f},"
                        f"{train_m.get('loss_delta', 0):.4f},"
                        f"{train_m.get('val_mcc', 0):.4f},"
                        f"{train_m.get('val_prauc', 0):.4f},"
                        f"{train_m.get('val_specificity', 0):.4f},"
                        f"{train_m.get('val_balanced_acc', 0):.4f},"
                        f"0.5"
                    )
                else:
                    csv_lines.append(f"{arch},Val,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
                
                # Inference metrics
                if inf_m:
                    csv_lines.append(
                        f"{arch},Inference,{loss},N/A,"
                        f"{inf_m.get('precision', 0):.4f},"
                        f"{inf_m.get('recall', 0):.4f},"
                        f"{inf_m.get('auc', 0):.4f},"
                        f"{inf_m.get('f1', 0):.4f},"
                        f"{inf_m.get('tp', 0)},"
                        f"{inf_m.get('fp', 0)},"
                        f"{inf_m.get('tn', 0)},"
                        f"{inf_m.get('fn', 0)},"
                        f"{inf_m.get('max_pred', 0):.4f},"
                        f"{inf_m.get('mean_pred', 0):.4f},"
                        f"{inf_m.get('std_pred', 0):.4f},"
                        f"{inf_m.get('pct_above_thresh', 0):.2f},"
                        f"N/A,"
                        f"N/A,"
                        f"{inf_m.get('label_threshold', 0):.1f},"
                        f"N/A,"
                        f"N/A,"
                        f"N/A,"
                        f"N/A,"
                        f"{inf_m.get('inf_mcc', 0):.4f},"
                        f"{inf_m.get('inf_prauc', 0):.4f},"
                        f"{inf_m.get('inf_specificity', 0):.4f},"
                        f"{inf_m.get('inf_balanced_acc', 0):.4f},"
                        f"{inf_m.get('pred_threshold', 0.5):.2f}"
                    )
                else:
                    csv_lines.append(f"{arch},Inference,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
            
            # Print CSV
            for line in csv_lines:
                print(line)
            
            # Save to file
            import os
            csv_filename = 'metrics_summary.csv'
            with open(csv_filename, 'w') as f:
                f.write('\n'.join(csv_lines))
            print(f"\n[INFO] CSV saved to {csv_filename}")
            
            # Generate recommendations
            print(f"\n[RECOMMENDED ACTIONS]")
            print("-" * 60)
            
            # Check for common issues
            low_precision_archs = [m.get('arch') for m in sorted_metrics if m.get('P', 0) < ensemble_threshold and m.get('P', 0) > 0]
            zero_precision_archs = [m.get('arch') for m in sorted_metrics if m.get('P', 0) == 0]
            
            if zero_precision_archs:
                print(f"1. For {', '.join(zero_precision_archs)}: Try binary_crossentropy instead of FocalLoss")
            
            if low_precision_archs:
                print(f"2. For {', '.join(low_precision_archs)}: Tune alpha/gamma in FocalLoss or lower prediction threshold")
            
            # Best performer
            best_arch = sorted_metrics[0].get('arch', 'Unknown') if sorted_metrics else 'None'
            best_p = sorted_metrics[0].get('P', 0) if sorted_metrics else 0
            print(f"3. Use {best_arch} as primary (P={best_p:.4f})")
            
            print(f"\n[PARAMETER TUNING PRIORITY]")
            print("-" * 60)
            print("Priority 1: Loss Function (BCE vs FocalLoss) - highest impact")
            print("Priority 2: FocalLoss Gamma - controls selectivity")
            print("Priority 3: Learning Rate - convergence quality")
            print("Priority 4: Dropout - capacity vs regularization")
            
            # =========================================================================
            # AUTO-APPLY CONFIG RECOMMENDATIONS
            # =========================================================================
            print(f"\n[AUTO-APPLY] Analyzing config changes...")
            
            # Get current config
            config = self.config
            changes_applied = []
            
            # Save backup of current config
            import json
            import os
            from datetime import datetime
            
            config_backup_file = 'config_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
            with open(config_backup_file, 'w') as f:
                json.dump({k: str(v)[:100] for k, v in config.items()}, f, indent=2)
            print(f"[BACKUP] Config saved to {config_backup_file}")
            
            # Auto-apply logic
            hpo_space = config.get('HYPERPARAM_SEARCH_SPACE', {})
            
            for m in sorted_metrics:
                arch = m.get('arch', 'Unknown')
                p = m.get('P', 0)
                r = m.get('R', 0)
                tp = m.get('TP', 0)
                fp = m.get('FP', 0)
                pred_total = tp + fp
                
                # Skip if precision is already above threshold
                if p >= ensemble_threshold:
                    print(f"[SKIP] {arch}: P={p:.4f} >= {ensemble_threshold} (working)")
                    continue
                
                # Get current HPO config for this architecture
                arch_hpo = hpo_space.get(arch, {})
                current_loss = arch_hpo.get('loss_function', ['binary_crossentropy', 'focal_loss'])
                
                # Rule 1: If precision = 0, switch to binary_crossentropy
                if p == 0:
                    if 'binary_crossentropy' in current_loss and 'focal_loss' in current_loss:
                        # Keep both in HPO, let next run try both
                        print(f"[AUTO] {arch}: P=0 - Will try both BCE and FocalLoss in next run")
                        changes_applied.append(f"{arch}: P=0 - keeping both loss options")
                    else:
                        # Add binary_crossentropy to options
                        if 'focal_loss' in str(current_loss):
                            if isinstance(current_loss, list):
                                new_loss = current_loss + ['binary_crossentropy']
                            else:
                                new_loss = ['binary_crossentropy', 'focal_loss']
                            hpo_space[arch]['loss_function'] = new_loss
                            changes_applied.append(f"{arch}: Added BCE to loss options (was: {current_loss})")
                            print(f"[AUTO] {arch}: Added BCE to loss options")
                
                # Rule 2: If precision below threshold but > 0, adjust alpha/gamma
                elif p < ensemble_threshold and p > 0:
                    current_alpha = arch_hpo.get('alpha', [0.5])
                    current_gamma = arch_hpo.get('gamma', [1.0])
                    
                    # Expand alpha range to include higher values
                    if isinstance(current_alpha, list):
                        max_alpha = max(current_alpha)
                        if max_alpha < 1.5:
                            new_alpha = current_alpha + [min(1.5, max_alpha + 0.25)]
                            hpo_space[arch]['alpha'] = new_alpha
                            changes_applied.append(f"{arch}: Expanded alpha to {new_alpha}")
                            print(f"[AUTO] {arch}: Expanded alpha range to {new_alpha}")
                    
                    # Expand gamma range to include lower values
                    if isinstance(current_gamma, list):
                        min_gamma = min(current_gamma)
                        if min_gamma > 1.0:
                            new_gamma = [max(0.5, min_gamma - 0.5)] + current_gamma
                            hpo_space[arch]['gamma'] = new_gamma
                            changes_applied.append(f"{arch}: Expanded gamma to {new_gamma}")
                            print(f"[AUTO] {arch}: Expanded gamma range to {new_gamma}")
            
            # Update config with new HPO space
            if changes_applied:
                config['HYPERPARAM_SEARCH_SPACE'] = hpo_space
                
                # Save updated config
                config_file = 'chunk_01_config.py'
                print(f"\n[SUCCESS] Applied {len(changes_applied)} config changes:")
                for change in changes_applied:
                    print(f"   - {change}")
                
                # Note: Manual update of chunk_01_config.py required
                print(f"\n[NOTE] Please manually update {config_file} with:")
                print("   HPO space has been adjusted based on results")
                print("   Review the changes above and re-run pipeline")
            else:
                print(f"[INFO] No config changes needed (all architectures working)")
            
            # =========================================================================
            # ADDITIONAL AUTO-TUNE RULES (PRINTED ONLY)
            # =========================================================================
            print(f"\n[ADDITIONAL AUTO-TUNE RULES] (printed only)")
            print("-" * 60)
            
            additional_rules_triggered = []
            
            for m in sorted_metrics:
                arch = m.get('arch', 'Unknown')
                p = m.get('P', 0)
                r = m.get('R', 0)
                auc = m.get('AUC', 0)
                train_p = m.get('train_P', 0)
                epochs_trained = m.get('epochs_trained', 0)
                tp = m.get('TP', 0)
                fp = m.get('FP', 0)
                pred_total = tp + fp
                
                # Rule 1: Check epochs trained (patience issue)
                if epochs_trained > 0 and epochs_trained <= 3:
                    msg = f"Increase patience for {arch} (only {epochs_trained} epochs)"
                    print(f"[RULE1] {msg}")
                    additional_rules_triggered.append(msg)
                
                # Rule 2: Check train/val precision gap (overfitting)
                if train_p > 0 and p > 0 and (train_p - p) > 0.1:
                    gap = train_p - p
                    msg = f"Expand dropout range for {arch} (train_P={train_p:.4f} >> val_P={p:.4f}, gap={gap:.4f})"
                    print(f"[RULE2] {msg}")
                    additional_rules_triggered.append(msg)
                
                # Rule 3: Check recall (minimum coverage)
                if r > 0 and r < 0.05:
                    msg = f"Expand label_threshold search for {arch} (recall={r:.4f} too low, need >=0.05)"
                    print(f"[RULE3] {msg}")
                    additional_rules_triggered.append(msg)
                
                # Rule 4: Check AUC (barely better than random)
                if auc > 0 and auc < 0.55:
                    msg = f"Flag {arch} for feature engineering review (AUC={auc:.4f} barely above random 0.50)"
                    print(f"[RULE4] {msg}")
                    additional_rules_triggered.append(msg)
                
                # Rule 5: Check positive prediction rate (too few predictions)
                if pred_total > 0 and pred_total < 50:
                    pct = (pred_total / len(sorted_metrics)) * 100 if len(sorted_metrics) > 0 else 0
                    msg = f"Increase predictions for {arch} (only {pred_total} positive predictions)"
                    print(f"[RULE5] {msg}")
                    additional_rules_triggered.append(msg)
            
            if not additional_rules_triggered:
                print("[INFO] No additional auto-tune rules triggered")
            
        else:
            print("[WARNING] No architecture metrics found in context")
        
        print(f"\n{'='*60}")
        print("END OF METRICS REVIEW")
        print('='*60)
        
        return context


def validate_pipeline_execution(context: Dict) -> bool:
    """
    Validate complete pipeline executed successfully
    
    Args:
        context: Final pipeline context
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    # Check all phase completion flags
    required_phases = [1, 3, 4, 5]
    optional_phases = ['Xa', 'Xb']
    for phase_num in required_phases:
        flag = f'phase{phase_num}_complete'
        assert context.get(flag) == True, f"Phase {phase_num} not completed (missing {flag})"
    for phase_str in optional_phases:
        flag = f'phase{phase_str}_complete'
        if context.get(flag) is None:
            print(f"[WARNING] {flag} not set (may be optional)")
    
    # Check final outputs present
    assert 'final_metrics' in context, "Missing final_metrics"
    assert 'final_predictions' in context, "Missing final_predictions"
    
    # Validate metrics quality
    metrics = context['final_metrics']
    if isinstance(metrics, list) and len(metrics) > 0:
        metrics = metrics[0]  # Use first (best) architecture's metrics
    precision = metrics.get('precision', 0)
    assert precision >= 0, "Precision cannot be negative"
    assert precision <= 1, "Precision cannot exceed 1"
    
    # Validate predictions
    predictions = context['final_predictions']
    assert predictions is not None, "No predictions generated"
    assert len(predictions) > 0, "Predictions are empty"
    
    return True


def main(config: Dict = None) -> Dict:
    """
    Main entry point for the pipeline
    
    Args:
        config: Optional custom configuration (uses default CONFIG if None)
        
    Returns:
        Final pipeline context
    """
    if config is None:
        config = CONFIG
    
    # Configure TensorFlow - CPU mode (GPU disabled)
    # GPU detection commented out - forcing CPU
    try:
        import tensorflow as tf
        # Force CPU mode - CUDA_VISIBLE_DEVICES already set to ''
        print("[INFO] Running in CPU mode")
    except Exception as e:
        print(f"[WARNING] TensorFlow configuration: {e}")
    
    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(config)
    context = orchestrator.run()
    
    # Validate final result
    validate_pipeline_execution(context)
    print("\n[PASS] Pipeline execution validated successfully")
    
    return context


if __name__ == "__main__":
    # Run pipeline with default configuration
    try:
        result = main()
        print("\n" + "="*60)
        print("Fraud Detection Pipeline completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] CRITICAL ERROR: Data file not found")
        print(f"{e}")
        print("\n[FIX] SOLUTION: Please ensure your fraud data CSV file exists.")
        import sys
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n[ERROR] DATA VALIDATION ERROR: {e}")
        print("\n[FIX] SOLUTION: Please check your CSV file format and data quality")
        import sys
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"\n[ERROR] PIPELINE ERROR: {e}")
        import sys
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)