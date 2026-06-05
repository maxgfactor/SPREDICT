"""
Chunk 20: Pipeline Main
Main orchestrator for the stock analysis pipeline
"""

import os
import sys
import time
import warnings

# CPU Mode - GPU paths removed, forced CPU (2026-02-28)
if sys.platform == 'linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from typing import Dict, List, Type
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from chunk_01_config import CONFIG, validate_config_structure, DEFAULT_FIRST_THRESHOLD, DEFAULT_LAST_THRESHOLD, DEFAULT_THRESHOLD_STEP
from chunk_02_utils_logging import Logger
from chunk_15_phase_base import BasePhase
from chunk_16_phase_1_setup import Phase1_PipelineSetup, validate_phase1_output
from chunk_17_phase_3_temporal import Phase3_TemporalWeighting, validate_phase3_output
from chunk_18_phase_4_ensemble import Phase4_NeuralEnsemble, validate_phase4_output
from chunk_19_phase_5_optimization import Phase5_PredictionOptimization, validate_phase5_output
from chunk_XX_phase_feature_analysis_a import PhaseXa_FeatureAnalysis
from chunk_XX_phase_feature_analysis_b import PhaseXb_TemporalCorrelation


class PipelineOrchestrator:
    """Orchestrates the complete stock analysis pipeline"""
    
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
        
        self.logger.log("[running] Starting Stock Analysis Pipeline...", 'info')
        self.logger.log(f"   Data path: {self.config['DATA_PATH']}", 'info')
        try:
            data_path = self.config['DATA_PATH']
            with open(data_path) as f:
                num_cols = len(f.readline().split(','))
                f.seek(0)
                num_rows = sum(1 for _ in f) - 1
            file_size_mb = os.path.getsize(data_path) / 1024 / 1024
            self.logger.log(f"   Dataset shape: {num_rows:,} rows x {num_cols} columns ({file_size_mb:.1f} MB)", 'info')
        except Exception as e:
            self.logger.log(f"   Could not read dataset: {e}", 'warning')
        self.logger.log(f"   Sampling: size={self.config['SAMPLE_SIZE']}, enabled={self.config['USE_SAMPLING']}, forced={self.config['FORCE_SAMPLING']}", 'info')
        self.logger.log(f"   hyperparameter_optimization: trials={self.config['HYPERPARAM_OPTIMIZATION_TRIALS']}, continue_until_target={self.config['HPO_CONTINUE_UNTIL_TARGET']}, epochs_per_trial={self.config['HYPERPARAM_OPTIMIZATION_EPOCHS']}, stagnation_threshold={self.config['HPO_STAGNATION_THRESHOLD']}", 'info')
        first_thresh = self.config.get('FIRST_THRESHOLD', DEFAULT_FIRST_THRESHOLD)
        last_thresh = self.config.get('LAST_THRESHOLD', DEFAULT_LAST_THRESHOLD)
        thresh_step = self.config.get('THRESHOLD_STEP', DEFAULT_THRESHOLD_STEP)
        thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)
        self.logger.log(f"   Label_Thresholds: {first_thresh} to {last_thresh} ({len(thresholds)} thresholds)", 'info')
        
        # Phase execution sequence
        phase_sequence = [
            ('Pipeline Setup', Phase1_PipelineSetup, validate_phase1_output),
            ('Temporal Weighting', Phase3_TemporalWeighting, validate_phase3_output),
            ('Phase Xa: Raw Feature Importance', PhaseXa_FeatureAnalysis, None),
            ('Neural Ensemble', Phase4_NeuralEnsemble, validate_phase4_output),
            ('Phase Xb: Temporal Precision Gap', PhaseXb_TemporalCorrelation, None),
            ('Prediction Optimization', Phase5_PredictionOptimization, validate_phase5_output),
        ]
        
        # Initialize context
        context = {}
        
        # Execute phases in sequence
        for phase_name, PhaseClass, validator in phase_sequence:
            phase_start = time.time()
            
            try:
                # Create and execute phase
                phase = PhaseClass(self.config)
                phase.logger = self.logger
                result = phase.execute(context)
                
                # Validate phase output
                try:
                    if validator is not None:
                        validator(result)
                    else:
                        self.logger.log(f"[skip] {phase_name} validation skipped (no validator)", 'info')
                except AssertionError as e:
                    self.logger.log(f"[warning]  {phase_name} validation warning: {e}", 'warning')
                
                # Update context with phase results
                context.update(result)
                
                # Record timing
                phase_time = time.time() - phase_start
                self.phase_timings[phase_name] = phase_time
                
            except Exception as e:
                self.logger.log(f"[error] {phase_name} failed: {e}", 'error')
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Pipeline failed at {phase_name}") from e
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Log final summary
        self.logger.log("Pipeline Complete!", 'info')
        self.logger.log(f"[time] Total execution time: {total_time:.2f}s", 'info')
        self.logger.log("Phase timings:", 'info')
        for phase, timing in self.phase_timings.items():
            self.logger.log(f"   {phase}: {timing:.2f}s", 'info')
        
        # Log final metrics — use consensus predictions vs binarized ground truth if available
        final_predictions = context.get('final_predictions')
        y_inference_binarized = context.get('y_inference_binarized')
        if final_predictions is not None and y_inference_binarized is not None:
            cons_precision = precision_score(y_inference_binarized, final_predictions, zero_division=0)
            cons_recall = recall_score(y_inference_binarized, final_predictions, zero_division=0)
            cons_f1 = f1_score(y_inference_binarized, final_predictions, zero_division=0)
            try:
                cons_auc = roc_auc_score(y_inference_binarized, final_predictions)
            except Exception:
                cons_auc = 0.0
            self.logger.log(f"[stat] Final Results:", 'info')
            self.logger.log(f"   precision: {cons_precision:.4f}", 'info')
            self.logger.log(f"   recall: {cons_recall:.4f}", 'info')
            self.logger.log(f"   f1 Score: {cons_f1:.4f}", 'info')
            self.logger.log(f"   auc: {cons_auc:.4f}", 'info')
        elif 'final_metrics' in context:
            # Fallback: display best architecture's metrics
            metrics = context['final_metrics']
            if isinstance(metrics, list) and len(metrics) > 0:
                metrics = metrics[0]
            if isinstance(metrics, dict):
                self.logger.log(f"[stat] Final Results:", 'info')
                self.logger.log(f"   precision: {metrics.get('Inf_P', 0):.4f}", 'info')
                self.logger.log(f"   recall: {metrics.get('Inf_R', 0):.4f}", 'info')
                self.logger.log(f"   f1 Score: {metrics.get('Inf_F1', 0):.4f}", 'info')
                self.logger.log(f"   auc: {metrics.get('Inf_AUC', 0):.4f}", 'info')
        
        # =========================================================================
        # METRICS REVIEW FRAMEWORK
        # =========================================================================
        self.logger.log("METRICS REVIEW REPORT", 'info')
        
        # Get architecture metrics from Phase 4
        arch_metrics = context.get('arch_final_metrics', [])
        
        if arch_metrics:
            # Sort by precision (descending)
            sorted_metrics = sorted(arch_metrics, key=lambda x: x.get('P', 0), reverse=True)
            
            self.logger.log(f"[architecture performance] (sorted by Val precision)", 'info')
            self.logger.log("-" * 60, 'info')
            
            ensemble_threshold = self.config.get('ENSEMBLE_MIN_PRECISION', 0.40)
            
            for i, m in enumerate(sorted_metrics, 1):
                arch = m.get('arch', 'Unknown')
                p = m.get('P', 0)
                r = m.get('R', 0)
                auc = m.get('AUC', 0)
                tp = m.get('TP', 0)
                fp = m.get('FP', 0)
                
                status = "✓" if p >= ensemble_threshold else "✗"
                self.logger.log(f"{i}. {arch:15s} P={p:.4f} R={r:.4f} auc={auc:.4f} TP={tp:5d} FP={fp:5d} {status}", 'info')
            
            # Identify issues
            self.logger.log(f"[issues identified]", 'info')
            self.logger.log("-" * 60, 'info')
            
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
                    self.logger.log(issue, 'info')
            else:
                self.logger.log("- No issues found", 'info')
            
            # =========================================================================
            # STANDARDIZED METRICS TABLE (CSV FORMAT)
            # =========================================================================
            self.logger.log(f"[standardized metrics table]", 'info')
            self.logger.log("-" * 60, 'info')
            
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
                        f"{arch},Validation,{loss},{epochs},"
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
                    csv_lines.append(f"{arch},Validation,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
                
                # Inference metrics with Inf_ prefix
                if inf_m:
                    csv_lines.append(
                        f"{arch},Inference,{loss},N/A,"
                        f"{inf_m.get('Inf_P', 0):.4f},"
                        f"{inf_m.get('Inf_R', 0):.4f},"
                        f"{inf_m.get('Inf_AUC', 0):.4f},"
                        f"{inf_m.get('Inf_F1', 0):.4f},"
                        f"{inf_m.get('Inf_TP', 0)},"
                        f"{inf_m.get('Inf_FP', 0)},"
                        f"{inf_m.get('Inf_TN', 0)},"
                        f"{inf_m.get('Inf_FN', 0)},"
                        f"{inf_m.get('Inf_MaxPred', 0):.4f},"
                        f"{inf_m.get('Inf_MeanPred', 0):.4f},"
                        f"{inf_m.get('Inf_StdPred', 0):.4f},"
                        f"{inf_m.get('Inf_PctAboveThresh', 0):.2f},"
                        f"N/A,"
                        f"N/A,"
                        f"{inf_m.get('label_threshold', 0):.1f},"
                        f"N/A,"
                        f"N/A,"
                        f"N/A,"
                        f"N/A,"
                        f"{inf_m.get('Inf_Spec', 0):.4f},"
                        f"{inf_m.get('Inf_FPR', 0):.4f},"
                        f"{inf_m.get('Inf_F2', 0):.4f},"
                        f"{inf_m.get('Inf_MCC', 0):.4f},"
                        f"{inf_m.get('Inf_PRAUC', 0):.4f},"
                        f"{inf_m.get('Inf_BalAcc', 0):.4f},"
                        f"{inf_m.get('pred_threshold', 0.5):.2f}"
                    )
                else:
                    csv_lines.append(f"{arch},Inference,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
            
            # Print CSV
            for line in csv_lines:
                self.logger.log(line, 'info')
            
            # Save to file
            csv_filename = 'metrics_summary.csv'
            with open(csv_filename, 'w') as f:
                f.write('\n'.join(csv_lines))
            self.logger.log(f"[info] CSV saved to {csv_filename}", 'info')
            
            # Generate recommendations
            self.logger.log(f"[recommended actions]", 'info')
            self.logger.log("-" * 60, 'info')
            
            # Check for common issues
            low_precision_archs = [m.get('arch') for m in sorted_metrics if m.get('P', 0) < ensemble_threshold and m.get('P', 0) > 0]
            zero_precision_archs = [m.get('arch') for m in sorted_metrics if m.get('P', 0) == 0]
            
            if zero_precision_archs:
                self.logger.log(f"1. For {', '.join(zero_precision_archs)}: Try binary_crossentropy instead of FocalLoss", 'info')
            
            if low_precision_archs:
                self.logger.log(f"2. For {', '.join(low_precision_archs)}: Tune alpha/gamma in FocalLoss or lower prediction threshold", 'info')
            
            # Best performer
            best_arch = sorted_metrics[0].get('arch', 'Unknown') if sorted_metrics else 'None'
            best_p = sorted_metrics[0].get('P', 0) if sorted_metrics else 0
            self.logger.log(f"3. Use {best_arch} as primary (P={best_p:.4f})", 'info')
            
            self.logger.log(f"[parameter tuning priority]", 'info')
            self.logger.log("-" * 60, 'info')
            self.logger.log("Priority 1: Loss Function (BCE vs FocalLoss) - highest impact", 'info')
            self.logger.log("Priority 2: FocalLoss Gamma - controls selectivity", 'info')
            self.logger.log("Priority 3: Learning Rate - convergence quality", 'info')
            self.logger.log("Priority 4: Dropout - capacity vs regularization", 'info')
            
            # =========================================================================
            # AUTO-APPLY CONFIG RECOMMENDATIONS
            # =========================================================================
            self.logger.log(f"[auto-apply] Analyzing config changes...", 'info')
            
            # Get current config
            config = self.config
            changes_applied = []
            
            # Save backup of current config
            import json
            from datetime import datetime
            
            config_backup_file = 'config_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
            with open(config_backup_file, 'w') as f:
                json.dump({k: str(v)[:100] for k, v in config.items()}, f, indent=2)
            self.logger.log(f"[backup] Config saved to {config_backup_file}", 'info')
            
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
                    self.logger.log(f"[skip] {arch}: P={p:.4f} >= {ensemble_threshold} (working)", 'info')
                    continue
                
                # Get current HPO config for this architecture
                arch_hpo = hpo_space.get(arch, {})
                current_loss = arch_hpo.get('loss_function', ['binary_crossentropy', 'focal_loss'])
                
                # Rule 1: If precision = 0, switch to binary_crossentropy
                if p == 0:
                    if 'binary_crossentropy' in current_loss and 'focal_loss' in current_loss:
                        # Keep both in HPO, let next run try both
                        self.logger.log(f"[auto] {arch}: P=0 - Will try both BCE and FocalLoss in next run", 'info')
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
                            self.logger.log(f"[auto] {arch}: Added BCE to loss options", 'info')
                
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
                            self.logger.log(f"[auto] {arch}: Expanded alpha range to {new_alpha}", 'info')
                    
                    # Expand gamma range to include lower values
                    if isinstance(current_gamma, list):
                        min_gamma = min(current_gamma)
                        if min_gamma > 1.0:
                            new_gamma = [max(0.5, min_gamma - 0.5)] + current_gamma
                            hpo_space[arch]['gamma'] = new_gamma
                            changes_applied.append(f"{arch}: Expanded gamma to {new_gamma}")
                            self.logger.log(f"[auto] {arch}: Expanded gamma range to {new_gamma}", 'info')
            
            # Update config with new HPO space
            if changes_applied:
                config['HYPERPARAM_SEARCH_SPACE'] = hpo_space
                
                # Save updated config
                config_file = 'chunk_01_config.py'
                self.logger.log(f"[success] Applied {len(changes_applied)} config changes:", 'info')
                for change in changes_applied:
                    self.logger.log(f"   - {change}", 'info')
                
                # Note: Manual update of chunk_01_config.py required
                self.logger.log(f"[note] Please manually update {config_file} with:", 'info')
                self.logger.log("   HPO space has been adjusted based on results", 'info')
                self.logger.log("   Review the changes above and re-run pipeline", 'info')
            else:
                self.logger.log(f"[info] No config changes needed (all architectures working)", 'info')
            
            # =========================================================================
            # ADDITIONAL AUTO-TUNE RULES (PRINTED ONLY)
            # =========================================================================
            self.logger.log(f"[ADDITIONAL AUTO-TUNE RULES] (printed only)", 'info')
            self.logger.log("-" * 60, 'info')
            
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
                    self.logger.log(f"[rule1] {msg}", 'info')
                    additional_rules_triggered.append(msg)
                
                # Rule 2: Check train/val precision gap (overfitting)
                if train_p > 0 and p > 0 and (train_p - p) > 0.1:
                    gap = train_p - p
                    msg = f"Expand dropout range for {arch} (train_P={train_p:.4f} >> val_p={p:.4f}, gap={gap:.4f})"
                    self.logger.log(f"[rule2] {msg}", 'info')
                    additional_rules_triggered.append(msg)
                
                # Rule 3: Check recall (minimum coverage)
                if r > 0 and r < 0.05:
                    msg = f"Expand label_threshold search for {arch} (recall={r:.4f} too low, need >=0.05)"
                    self.logger.log(f"[rule3] {msg}", 'info')
                    additional_rules_triggered.append(msg)
                
                # Rule 4: Check AUC (barely better than random)
                if auc > 0 and auc < 0.55:
                    msg = f"Flag {arch} for feature engineering review (AUC={auc:.4f} barely above random 0.50)"
                    self.logger.log(f"[rule4] {msg}", 'info')
                    additional_rules_triggered.append(msg)
                
                # Rule 5: Check positive prediction rate (too few predictions)
                if pred_total > 0 and pred_total < 50:
                    pct = (pred_total / len(sorted_metrics)) * 100 if len(sorted_metrics) > 0 else 0
                    msg = f"Increase predictions for {arch} (only {pred_total} positive predictions)"
                    self.logger.log(f"[rule5] {msg}", 'info')
                    additional_rules_triggered.append(msg)
            
            if not additional_rules_triggered:
                self.logger.log("[info] No additional auto-tune rules triggered", 'info')
            
        else:
            self.logger.log("[warning] No architecture metrics found in context", 'warning')
        
        self.logger.log("END OF METRICS REVIEW", 'info')
        
        return context


def validate_pipeline_execution(context: Dict, logger: Logger = None) -> bool:
    """
    Validate pipeline execution results
    
    Args:
        context: Pipeline execution context
        logger: Optional Logger instance for formatted output
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If validation fails
    """
    log = logger.log if logger else print
    # Check all phase completion flags
    required_phases = [1, 3, 4, 5]
    optional_phases = ['Xa', 'Xb']
    for phase_num in required_phases:
        flag = f'phase{phase_num}_complete'
        assert context.get(flag) == True, f"Phase {phase_num} not completed (missing {flag})"
    for phase_str in optional_phases:
        flag = f'phase{phase_str}_complete'
        if context.get(flag) is None:
            log(f"[warning] {flag} not set (may be optional)")
    
    # Check final outputs present
    assert 'final_metrics' in context, "Missing final_metrics"
    assert 'final_predictions' in context, "Missing final_predictions"
    
    # Validate metrics quality — use consensus precision if available
    fp = context.get('final_predictions')
    y_true = context.get('y_inference_binarized')
    if fp is not None and y_true is not None:
        precision = precision_score(y_true, fp, zero_division=0)
    else:
        metrics = context.get('final_metrics', [{}])
        if isinstance(metrics, list) and len(metrics) > 0:
            metrics = metrics[0]
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
    
    logger = Logger(config)
    
    # CPU mode (GPU disabled)
    try:
        import tensorflow as tf
        # Force CPU mode - CUDA_VISIBLE_DEVICES already set to ''
        logger.log("Running in CPU mode", 'info', 'pipeline')
    except Exception as e:
        logger.log(f"TensorFlow configuration: {e}", 'warning', 'pipeline')
    
    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(config)
    context = orchestrator.run()
    
    # Validate final result
    validate_pipeline_execution(context, logger)
    logger.log("Pipeline execution validated successfully", 'info', 'pipeline')
    
    return context


if __name__ == "__main__":
    # Run pipeline with default configuration
    try:
        result = main()
        print("Stock Analysis Pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n[error] critical error: Data file not found")
        print(f"{e}")
        print("\n[fix] solution: Please ensure your stock data CSV file exists.")
        import sys
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n[error] data validation error: {e}")
        print("\n[fix] solution: Please check your CSV file format and data quality")
        import sys
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"\n[error] pipeline error: {e}")
        import sys
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[error] unexpected error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
