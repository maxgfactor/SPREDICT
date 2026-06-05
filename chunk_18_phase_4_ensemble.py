"""
Chunk 18: Phase 4 - Neural Ensemble
Ensemble model training phase

## Purpose
Phase 4 orchestrates the complete model training pipeline including:
1. Finding optimal classification thresholds
2. Hyperparameter optimization via Optuna
3. Creating precision-weighted ensembles
4. Persisting trained models for later use

## Key Responsibilities
- Data preparation with temporal weighting
- Per-architecture threshold optimization
- Bayesian hyperparameter optimization
- Ensemble creation and evaluation
- Model serialization to ./saved_models/

## Dependencies
- Input: X, y (continuous), dates, temporal_weights from Phase 3
- Output: Trained models, optimal_thresholds, best_hyperparams for Phase 5

## Pipeline Flow
Phase 4 → Phase 5
(RNN/LSTM models + thresholds are passed to Phase 5 for final predictions)
"""

import numpy as np
import time
import sys
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler  # NN-only normalization (trees are scale-invariant)
from chunk_01_config import DEFAULT_FIRST_THRESHOLD, DEFAULT_LAST_THRESHOLD, DEFAULT_THRESHOLD_STEP, PREDICTION_THRESHOLD_DEFAULT

from chunk_15_phase_base import BasePhase
from chunk_02_utils_logging import Logger
from chunk_12_evaluation_evaluator import Evaluator
from chunk_14_models_trainer import ModelTrainer
from chunk_13_state_manager import StateManager
from chunk_10_models_ensemble import create_precision_ensemble, validate_ensemble_output
from chunk_21_hyperparam_optimizer import HyperparameterOptimizer
from chunk_04_utils_metrics import (
    inverse_log_transform,
    get_prediction_percentiles,
    get_prediction_histogram,
    format_diagnostic_string,
    analyze_loss_distribution,
    calculate_temporal_drift,
    calculate_permutation_importance,
    calculate_prediction_entropy,
    calculate_logit_compression,
    calculate_ks_test,
    calculate_bhattacharyya_distance,
    calculate_snr,
    calculate_mutual_information,
    calculate_psi
)


class Phase4_NeuralEnsemble(BasePhase):
    """Phase 4: Neural ensemble training and optimization"""
    
    def __init__(self, config: Dict):
        """
        Initialize Phase 4
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = Logger(config)
        self.evaluator = Evaluator(config, logger=self.logger)
        self.model_trainer = ModelTrainer(config, logger=self.logger, evaluator=self.evaluator)
        self.state_manager = StateManager()
        self.hyperparam_optimizer = HyperparameterOptimizer(config, logger=self.logger)
    
    def _validate_diagnostics_requirements(self, dates: np.ndarray) -> None:
        """
        Validate minimum date requirements for diagnostic features.
        Raises ValueError if below minimum threshold.
        
        Args:
            dates: Array of date values
            
        Raises:
            ValueError: If diagnostic features enabled but insufficient dates
        """
        min_dates = self.config.get('MIN_DATES_THRESHOLD', 30)
        unique_dates = len(np.unique(dates))
        
        if self.config.get('FEATURE_STABILITY_ANALYSIS', False):
            if unique_dates < min_dates:
                raise ValueError(f"Feature stability analysis requires at least {min_dates} unique dates, got {unique_dates}")
        
        if self.config.get('TRACK_INFERENCE_LATENCY', False):
            if unique_dates < min_dates:
                raise ValueError(f"Inference latency tracking requires at least {min_dates} unique dates, got {unique_dates}")
        
        if self.config.get('SLIDING_WINDOW_VALIDATION', False):
            if unique_dates < min_dates:
                raise ValueError(f"Sliding window validation requires at least {min_dates} unique dates, got {unique_dates}")
    
    def execute(self, context: Dict) -> Dict:
        """
        Execute Phase 4: Train ensemble models
        
        Args:
            context: Pipeline context with data from Phase 3
            
        Returns:
            Updated context with trained ensemble
        """
        # ============================================================================
        # SECTION 1: Data Preparation
        # ============================================================================
        # - Extract data from context (X, y, dates, temporal_weights)
        # - Apply temporal weighting to features (sqrt to avoid over-amplification)
        # - Time-based train/val split: top 15 most recent dates = validation
        # - Log class distribution at key thresholds
        
        # Validate input
        self._validate_input(context)
        
        X = context['X']
        y_binary = context['y']  # Binary target from Phase 1
        y_raw = context.get('raw_target_values', y_binary)  # Raw continuous target
        dates = context.get('dates', np.zeros(len(X)))
        temporal_weights = context.get('temporal_weights', np.ones(len(X)))
        
        # Validate diagnostic requirements (Items 1, 2, 4)
        self._validate_diagnostics_requirements(dates)
        
        # Get metadata for reporting
        target_col = context.get('raw_target_column', 'unknown')
        feature_names = context.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
        threshold_kept_indices = context.get('threshold_kept_indices', {})
        
        # Log metadata
        self.logger.log(f"Target column: {target_col}", 'info')
        self.logger.log(f"Features ({len(feature_names)}): {feature_names}", 'info')
        if threshold_kept_indices:
            self.logger.log(f"Per-threshold feature pruning loaded for {len(threshold_kept_indices)} thresholds", 'info')
        
        self.logger.log(f"Training ensemble on {len(X)} samples", 'info')
        
        # Apply temporal weighting to features
        # Weight samples by multiplying features (simplified approach)
        X_weighted = X * np.sqrt(temporal_weights[:, np.newaxis])
        
        # Time-based train/validation split:
        # - Always exclude top 2 newest dates (Inference + Held Out)
        # - Split remaining dates by configurable percentage
        unique_dates = np.unique(dates)
        
        # Get split percentage from config (default 0.30 = 30%)
        val_split_pct = self.config.get('VAL_SPLIT_PERCENTAGE', 0.30)
        
        # Exclude top 2 newest dates
        if len(unique_dates) >= 2:
            top_2_dates = unique_dates[-2:]
        else:
            top_2_dates = []
        
        # Get remaining dates (exclude top 2)
        remaining_dates = unique_dates[:-2] if len(unique_dates) >= 2 else unique_dates
        
        # Calculate number of dates for validation and training
        n_remaining = len(remaining_dates)
        n_val = int(n_remaining * val_split_pct)
        n_train = n_remaining - n_val
        
        # Select validation dates (most recent of remaining dates)
        if n_val > 0:
            val_dates = remaining_dates[-n_val:]
        else:
            val_dates = []
        
        # Select training dates (older dates)
        if n_train > 0:
            train_dates = remaining_dates[:n_train]
        else:
            train_dates = []
        
        # Create masks
        val_mask = np.isin(dates, val_dates)
        train_mask = np.isin(dates, train_dates)
        
        X_train = X_weighted[train_mask]
        X_val = X_weighted[val_mask]
        y_train_continuous = y_raw[train_mask]
        y_val_continuous = y_raw[val_mask]
        weights_train = temporal_weights[train_mask]
        weights_val = temporal_weights[val_mask]
        
        # Log row counts as metrics
        n_train_rows = len(X_train)
        n_val_rows = len(X_val)
        n_dates_train = len(np.unique(dates[train_mask]))
        n_dates_val = len(np.unique(dates[val_mask]))
        
        self.logger.log(f"Date split: {len(unique_dates)} total, top 2 held out", 'info')
        self.logger.log(f"  Remaining dates: {n_remaining} (after excluding top 2)", 'info')
        self.logger.log(f"  Validation split: {val_split_pct:.0%} = {n_val} dates", 'info')
        self.logger.log(f"  Training split: {1-val_split_pct:.0%} = {n_train} dates", 'info')
        self.logger.log(f"Training set: {n_train_rows:,} rows, {n_dates_train} unique dates", 'info')
        self.logger.log(f"Validation set: {n_val_rows:,} rows, {n_dates_val} unique dates", 'info')
        
        # SANITY CHECK: Validate train/val split
        total_rows = n_train + n_val
        train_pct = n_train / total_rows * 100
        val_pct = n_val / total_rows * 100
        self.logger.log(f"[stat] split ratio: train={train_pct:.1f}% | validation={val_pct:.1f}%", 'info')
        
        # Get threshold config values
        first_threshold = self.config.get('FIRST_THRESHOLD', DEFAULT_FIRST_THRESHOLD)
        last_threshold = self.config.get('LAST_THRESHOLD', DEFAULT_LAST_THRESHOLD)
        threshold_step = self.config.get('THRESHOLD_STEP', DEFAULT_THRESHOLD_STEP)
        
        # SANITY CHECK: Show class distribution at key thresholds
        # Use same thresholds as the search range for coherence
        thresholds = np.arange(first_threshold, last_threshold + threshold_step, threshold_step)
        self.logger.log(f"[stat] Class Distribution at Label Thresholds ({first_threshold} to {last_threshold}, step {threshold_step}):", 'info')
        y_val_continuous = y_raw[val_mask]  # Get raw values for validation
        for label_threshold in thresholds:
            if y_raw is not None:
                train_pos = int(np.sum(y_train_continuous >= label_threshold))
                val_pos = int(np.sum(y_val_continuous >= label_threshold))
                train_pct = train_pos / n_train_rows * 100 if n_train_rows > 0 else 0
                val_pct = val_pos / n_val_rows * 100 if n_val_rows > 0 else 0
                status = "[ok]" if train_pos > 0 and val_pos > 0 else "[warning]"
                self.logger.log(f"   LABEL_THRESHOLD={label_threshold:>5.1f}: train_positives={train_pos:,} ({train_pct:>5.2f}%) | validation_positives={val_pos:,} ({val_pct:>5.2f}%) {status}", 'info')
        
        # Define architectures to train (Discovery sequence - May 11, 2026)
        # Gradient Boosting first (CatBoost→LightGBM→XGBoost) for feature insights
        # Then Neural Networks (Dense→CNN→RNN→LSTM→VAE→Transformer) per discovery sequence
        architectures = [
            'CatBoost', 'LightGBM', 'XGBoost', 'Dense', 'CNN', 'RNN', 'LSTM', 'VAE', 'Transformer',
        ]
        
        self.logger.log(f"[section 1] [baseline] Training {len(architectures)} architectures", 'info')
        
        # Define threshold range from config
        thresholds = np.arange(first_threshold, last_threshold + threshold_step, threshold_step)
        self.logger.log(f"[section 1] [baseline] Testing {len(thresholds)} thresholds per architecture ({first_threshold} to {last_threshold}, step {threshold_step})", 'info')
        
        # Get first threshold's kept features for baseline diagnostic
        first_thr_key = round(float(thresholds[0]), 1)
        first_kept = threshold_kept_indices.get(first_thr_key)
        if first_kept is None:
            first_kept = list(range(X.shape[1]))
        first_kept = list(first_kept)
        
        # ============================================================================
        # SECTION 2: Threshold Optimization Loop
        # ============================================================================
        # For each architecture (RNN, LSTM):
        # - Iterate through thresholds (FIRST_THRESHOLD → LAST_THRESHOLD)
        # - Train model at each threshold using evaluator.find_optimal_threshold()
        # - Select threshold with highest validation precision
        # - Uses early stopping (patience=17) to avoid unnecessary iterations
        #
        # Output: optimal_threshold per architecture (e.g., RNN=22.1, LSTM=21.3)
        
        # Train models with threshold optimization
        trained_models = []
        val_predictions = []
        optimal_thresholds = []  # Track optimal threshold for each architecture
        arch_names = []  # Track architecture names
        best_hyperparams_list = []  # Track best hyperparams per architecture
        best_val_precision_list = []  # Track best validation precision per architecture
        arch_kept_indices = []  # Track kept feature indices per architecture (for Phase 5)
        arch_scalers = {}  # Track StandardScaler per NN architecture (for Phase 5 inference)
        final_thresholds = []  # Track final (post-HPO) threshold per architecture
        
        # Track timing and metrics for summary
        arch_training_times = []  # Time per architecture
        arch_final_metrics = []  # Final metrics per architecture
        pre_hpo_precisions = []  # Pre-HPO precision for HPO impact summary
        post_hpo_precisions = []  # Post-HPO precision for HPO impact summary
        
        pred_threshold = self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)
        
        for arch_name in architectures:
            try:
                arch_tag = f"[{arch_name.upper()}]"
                arch_start_time = time.time()
                self.logger.log(f"[section 1] {arch_tag} [baseline] Training with threshold optimization...", 'info')
                
                # === BASELINE DIAGNOSTICS: Get prediction stats before any threshold optimization ===
                baseline_y_train = (y_train_continuous >= thresholds[0]).astype(int)  # Use first threshold
                X_train_bl = X_train[:, first_kept]
                X_val_bl = X_val[:, first_kept]
                baseline_model = self.model_trainer.build_architecture(arch_name, X_train_bl.shape[1], y_train_continuous)
                baseline_model, _ = self.model_trainer.train_model(baseline_model, X_train_bl, baseline_y_train, epochs=1, verbose=0)
                if baseline_model is None:
                    raise ValueError(f"[fatal] {arch_name} baseline_model is None after training")
                baseline_pred = baseline_model.predict(X_val_bl, verbose=0).flatten()
                
                self.logger.log(f"[section 1] {arch_tag} [baseline] before_threshold_optimization:", 'info')
                self.logger.log(f"[section 1] {arch_tag} [baseline] predictions: mean={baseline_pred.mean():.4f}, std={baseline_pred.std():.4f}, min={baseline_pred.min():.4f}, max={baseline_pred.max():.4f}", 'info')
                self.logger.log(f"[section 1] {arch_tag} [baseline] " + format_diagnostic_string(baseline_pred, ""), 'info')
                hist = get_prediction_histogram(baseline_pred, 20)
                self.logger.log(f"[section 1] {arch_tag} [baseline] Histogram bins: {hist['counts'][:5]} ... {hist['counts'][-5:]}", 'info')
                self.logger.log(f"[section 1] {arch_tag} [baseline] % positive predictions (prediction_threshold={pred_threshold:.2f}): {((baseline_pred >= pred_threshold).mean() * 100):.2f}%", 'info')
                if ((baseline_pred >= pred_threshold).sum() == 0):
                    self.logger.log(f"[section 1] {arch_tag} [baseline] [warning] No predictions >= {pred_threshold}!", 'warning')
                
                # Calculate baseline precision for HPO comparison
                # NOTE: This calculates ACCURACY (predictions == labels), NOT precision
                # We need to calculate actual precision from confusion matrix below
                y_val_binarized = (y_val_continuous >= thresholds[0]).astype(int)
                baseline_binary = (baseline_pred >= pred_threshold).astype(int)
                baseline_accuracy = float((baseline_binary == y_val_binarized).sum()) / max(len(y_val_binarized), 1) if baseline_binary.sum() > 0 else 0.0
                
                # Calculate full baseline metrics
                baseline_cm = self.evaluator.calculate_confusion_matrix(y_val_binarized, baseline_binary)
                baseline_precision = float(baseline_cm['TP']) / max(baseline_cm['TP'] + baseline_cm['FP'], 1) if (baseline_cm['TP'] + baseline_cm['FP']) > 0 else 0.0
                baseline_recall = self.evaluator.calculate_recall(y_val_binarized, baseline_binary)
                baseline_auc = self.evaluator.calculate_auc(y_val_binarized, baseline_pred)
                baseline_f1 = self.evaluator.calculate_f1(y_val_binarized, baseline_binary)
                
                # Get baseline model hyperparameters
                baseline_hyperparams = {}
                try:
                    if hasattr(baseline_model, 'hyperparams'):
                        baseline_hyperparams = baseline_model.hyperparams
                    elif hasattr(baseline_model, 'get_config'):
                        config = baseline_model.get_config()
                        baseline_hyperparams = {
                            'loss_function': config.get('loss', 'binary_crossentropy'),
                            'learning_rate': config.get('learning_rate', 0.001),
                            'dropout': config.get('dropout', 0.0),
                        }
                except:
                    pass
                
                # Section 1: Baseline Summary
                loss_fn = baseline_hyperparams.get('loss_function', 'binary_crossentropy')
                lr = baseline_hyperparams.get('learning_rate', 0.001)
                dropout = baseline_hyperparams.get('dropout', 0.0)
                alpha = baseline_hyperparams.get('alpha', 1.0)
                gamma = baseline_hyperparams.get('gamma', 1.0)
                latent_dim = baseline_hyperparams.get('latent_dim', 0)
                
                # Build hyperparam string
                hpo_str = f"loss={loss_fn}, lr={lr}"
                if dropout > 0:
                    hpo_str += f", dropout={dropout}"
                if alpha != 1.0:
                    hpo_str += f", alpha={alpha}"
                if gamma != 1.0:
                    hpo_str += f", gamma={gamma}"
                if latent_dim > 0:
                    hpo_str += f", latent_dim={latent_dim}"
                
                baseline_binary = (baseline_pred.flatten() >= pred_threshold).astype(int)
                baseline_spec = self.evaluator.calculate_specificity(y_val_binarized, baseline_binary)
                baseline_fpr = self.evaluator.calculate_fpr(y_val_binarized, baseline_binary)
                baseline_f2 = self.evaluator.calculate_f2_score(y_val_binarized, baseline_binary)
                baseline_mcc = self.evaluator.calculate_mcc(y_val_binarized, baseline_binary)
                baseline_prauc = self.evaluator.calculate_average_precision(y_val_binarized, baseline_pred.flatten())
                baseline_balacc = self.evaluator.calculate_balanced_accuracy(y_val_binarized, baseline_binary)
                baseline_brier = self.evaluator.calculate_brier_score(y_val_binarized, baseline_pred.flatten())
                baseline_kappa = self.evaluator.calculate_kappa(y_val_binarized, baseline_binary)
                baseline_informedness = self.evaluator.calculate_informedness(y_val_binarized, baseline_binary)
                baseline_markedness = self.evaluator.calculate_markedness(y_val_binarized, baseline_binary)
                baseline_gini = self.evaluator.calculate_gini(y_val_binarized, baseline_pred.flatten())
                baseline_opt_thresh = self.evaluator.calculate_optimal_threshold(y_val_binarized, baseline_pred.flatten())
                
                self.logger.log(f"[section 1] {arch_tag} [baseline] LABEL_THRESHOLD={thresholds[0]:.1f},", 'info')
                self.logger.log(f"[section 1] {arch_tag} [baseline] VALIDATION_PRECISION={baseline_precision:.4f} VALIDATION_TRUE_POSITIVES={baseline_cm['TP']} VALIDATION_TRUE_NEGATIVES={baseline_cm['TN']} validation_false_positives={baseline_cm['FP']} validation_false_negatives={baseline_cm['FN']} "
                             f"validation_max_prediction={baseline_pred.max():.4f} validation_mean_prediction={baseline_pred.mean():.4f} "
                             f"validation_recall={baseline_recall:.4f} validation_f1={baseline_f1:.4f} validation_auc={baseline_auc:.4f} "
                             f"validation_specificity={baseline_spec:.4f} validation_false_positive_rate={baseline_fpr:.4f} validation_f2={baseline_f2:.4f} validation_mcc={baseline_mcc:.4f} validation_prauc={baseline_prauc:.4f} validation_balanced_accuracy={baseline_balacc:.4f} "
                             f"validation_brier={baseline_brier:.4f} validation_kappa={baseline_kappa:.4f} validation_informedness={baseline_informedness:.4f} validation_markedness={baseline_markedness:.4f} validation_gini={baseline_gini:.4f} validation_optimal_threshold={baseline_opt_thresh:.4f} "
                             f"validation_standard_deviation_prediction={baseline_pred.std():.4f} validation_percentage_above_threshold={(baseline_pred >= pred_threshold).mean() * 100:.2f}", 'info')
                self.logger.log(f"[section 1] {arch_tag} [baseline] prediction_binary_split={pred_threshold:.2f} {hpo_str}", 'info')
                
                # Run threshold optimization
                optimal_threshold, best_prec, all_results, threshold_opt_model = self.evaluator.find_optimal_threshold(
                    X_train, y_train_continuous, X_val, y_val_continuous,
                    None, self.model_trainer, arch_name,
                    thresholds, patience=5,
                    threshold_feature_indices=threshold_kept_indices
                )
                if threshold_opt_model is None:
                    self.logger.log(f"[warning] {arch_name}: threshold_opt_model is None after find_optimal_threshold — falling back to last-trained model", 'warning')
                    # Fall through: outer exception handler will log and add placeholder metrics
                
                # Store pre-HPO precision for HPO impact tracking
                pre_hpo_precisions.append(best_prec)
                
                # Track optimal threshold for ensemble evaluation
                optimal_thresholds.append(optimal_threshold)
                
                # Slice X to optimal threshold's feature set for remaining sections (HPO, POST-HPO, Final)
                opt_thr_key = round(float(optimal_threshold), 1)
                opt_kept = threshold_kept_indices.get(opt_thr_key, list(range(X_train.shape[1])))
                opt_kept = list(opt_kept)
                X_train_opt = X_train[:, opt_kept]
                X_val_opt = X_val[:, opt_kept]
                arch_kept_indices.append(opt_kept)
                
                # Normalize features for NNs only (trees scale-invariant). Applied at caller level,
                # NOT inside train_model(), to keep fit() and predict() data consistent.
                if arch_name in ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']:
                    scaler = StandardScaler()
                    X_train_opt = scaler.fit_transform(X_train_opt)
                    X_val_opt = scaler.transform(X_val_opt)
                    arch_scalers[arch_name] = scaler
                
                # Log ALL results for each threshold - 4-line blocks per threshold
                # Log optimal threshold (4-line [optimal] block matching per-threshold format)
                optimal_result = next((r for r in all_results if r['threshold'] == optimal_threshold), None)
                all_thresholds_rejected = optimal_result is None and not all_results
                
                if all_thresholds_rejected:
                    self.logger.log(f"[section 1] {arch_tag} [label_threshold_optimal] [warning] All label thresholds rejected — using baseline metrics at {thresholds[0]:.1f}", 'warning')
                    optimal_threshold = thresholds[0]
                    tr = {
                        'P': baseline_precision, 'TP': baseline_cm['TP'], 'TN': baseline_cm['TN'],
                        'FP': baseline_cm['FP'], 'FN': baseline_cm['FN'],
                        'R': baseline_recall, 'AUC': baseline_auc, 'F1': baseline_f1,
                        'MaxPred': float(baseline_pred.max()), 'MeanPred': float(baseline_pred.mean()),
                        'Spec': baseline_spec, 'FPR': baseline_fpr, 'F2': baseline_f2, 'MCC': baseline_mcc,
                        'PRAUC': baseline_prauc, 'BalAcc': baseline_balacc, 'Brier': baseline_brier,
                        'Kappa': baseline_kappa, 'Informedness': baseline_informedness,
                        'Markedness': baseline_markedness, 'Gini': baseline_gini,
                        'OptThresh': baseline_opt_thresh, 'StdPred': float(baseline_pred.std()),
                        'PctAboveThresh': float((baseline_pred >= pred_threshold).mean() * 100),
                    }
                    v = dict(tr)
                    pre_hpo_val_precision = baseline_precision
                    pre_hpo_val_recall = baseline_recall
                    section2_P = baseline_precision
                    section2_R = baseline_recall
                    section2_AUC = baseline_auc
                    section2_F1 = baseline_f1
                    section2_TP = baseline_cm['TP']
                    section2_FP = baseline_cm['FP']
                    section2_TN = baseline_cm['TN']
                    section2_FN = baseline_cm['FN']
                    section2_max_pred = float(baseline_pred.max())
                    section2_mean_pred = float(baseline_pred.mean())
                    section2_pred = baseline_pred
                    section2_spec = v.get('Spec', 0.0)
                    section2_fpr = v.get('FPR', 0.0)
                    section2_f2 = v.get('F2', 0.0)
                    section2_mcc = v.get('MCC', 0.0)
                    section2_prauc = v.get('PRAUC', 0.0)
                    section2_balacc = v.get('BalAcc', 0.0)
                    section2_brier = v.get('Brier', 0.0)
                    section2_kappa = v.get('Kappa', 0.0)
                    section2_informedness = v.get('Informedness', 0.0)
                    section2_markedness = v.get('Markedness', 0.0)
                    section2_gini = v.get('Gini', 0.0)
                    section2_opt_thresh = v.get('OptThresh', 0.0)
                    section2_std_pred = v.get('StdPred', float(baseline_pred.std()))
                    section2_pct_above = v.get('PctAboveThresh', float((baseline_pred >= pred_threshold).mean() * 100))
                else:
                    tr, v = optimal_result['train'], optimal_result['val']
                    pre_hpo_val_precision = best_prec
                    
                    section2_val = optimal_result.get('val', {})
                    section2_P = section2_val.get('P', 0.0)
                    section2_R = section2_val.get('R', 0.0)
                    
                    pre_hpo_val_recall = section2_R
                    section2_AUC = section2_val.get('AUC', 0.0)
                    section2_F1 = section2_val.get('F1', 0.0)
                    section2_TP = section2_val.get('TP', 0)
                    section2_FP = section2_val.get('FP', 0)
                    section2_TN = section2_val.get('TN', 0)
                    section2_FN = section2_val.get('FN', 0)
                    
                    if threshold_opt_model is not None:
                        section2_pred = threshold_opt_model.predict(X_val_opt, verbose=0).flatten()
                        section2_max_pred = section2_pred.max()
                        section2_mean_pred = section2_pred.mean()
                        if self.config.get('LOG_VERBOSITY', 0) >= 2:
                            self.logger.log(f"[section 1] {arch_tag} [label_threshold_optimal] " + format_diagnostic_string(section2_pred, ""), 'info')
                    else:
                        section2_pred = np.zeros(len(X_val_opt))
                        section2_max_pred = 0.0
                        section2_mean_pred = 0.0
                    section2_spec = v.get('Spec', 0.0)
                    section2_fpr = v.get('FPR', 0.0)
                    section2_f2 = v.get('F2', 0.0)
                    section2_mcc = v.get('MCC', 0.0)
                    section2_prauc = v.get('PRAUC', 0.0)
                    section2_balacc = v.get('BalAcc', 0.0)
                    section2_brier = v.get('Brier', 0.0)
                    section2_kappa = v.get('Kappa', 0.0)
                    section2_informedness = v.get('Informedness', 0.0)
                    section2_markedness = v.get('Markedness', 0.0)
                    section2_gini = v.get('Gini', 0.0)
                    section2_opt_thresh = v.get('OptThresh', 0.0)
                    section2_std_pred = v.get('StdPred', 0.0)
                    section2_pct_above = v.get('PctAboveThresh', 0.0)
                
                self.logger.log(f"[section 1] {arch_tag} [label_threshold_optimal] LABEL_THRESHOLD={optimal_threshold:.1f}", 'info')
                self.logger.log(f"[section 1] {arch_tag} [label_threshold_optimal] train_precision={tr.get('P', 0.0):.4f} train_true_positives={tr.get('TP', 0)} train_true_negatives={tr.get('TN', 0)} train_false_positives={tr.get('FP', 0)} train_false_negatives={tr.get('FN', 0)} train_max_prediction={tr.get('MaxPred', 0.0):.4f} train_mean_prediction={tr.get('MeanPred', 0.0):.4f} train_recall={tr.get('R', 0.0):.4f} train_f1={tr.get('F1', 0.0):.4f} train_auc={tr.get('AUC', 0.0):.4f} train_specificity={tr.get('Spec', 0.0):.4f} train_false_positive_rate={tr.get('FPR', 0.0):.4f} train_f2={tr.get('F2', 0.0):.4f} train_mcc={tr.get('MCC', 0.0):.4f} train_prauc={tr.get('PRAUC', 0.0):.4f} train_balanced_accuracy={tr.get('BalAcc', 0.0):.4f} train_brier={tr.get('Brier', 0.0):.4f} train_kappa={tr.get('Kappa', 0.0):.4f} train_informedness={tr.get('Informedness', 0.0):.4f} train_markedness={tr.get('Markedness', 0.0):.4f} train_gini={tr.get('Gini', 0.0):.4f} train_optimal_threshold={tr.get('OptThresh', 0.0):.4f} train_standard_deviation_prediction={tr.get('StdPred', 0.0):.4f} train_percentage_above_threshold={tr.get('PctAboveThresh', 0.0):.2f}", 'info')
                self.logger.log(f"[section 1] {arch_tag} [label_threshold_optimal] VALIDATION_PRECISION={v.get('P', 0.0):.4f} VALIDATION_TRUE_POSITIVES={v.get('TP', 0)} VALIDATION_TRUE_NEGATIVES={v.get('TN', 0)} validation_false_positives={v.get('FP', 0)} validation_false_negatives={v.get('FN', 0)} validation_max_prediction={v.get('MaxPred', 0.0):.4f} validation_mean_prediction={v.get('MeanPred', 0.0):.4f} validation_recall={v.get('R', 0.0):.4f} validation_f1={v.get('F1', 0.0):.4f} validation_auc={v.get('AUC', 0.0):.4f} validation_specificity={v.get('Spec', 0.0):.4f} validation_false_positive_rate={v.get('FPR', 0.0):.4f} validation_f2={v.get('F2', 0.0):.4f} validation_mcc={v.get('MCC', 0.0):.4f} validation_prauc={v.get('PRAUC', 0.0):.4f} validation_balanced_accuracy={v.get('BalAcc', 0.0):.4f} validation_brier={v.get('Brier', 0.0):.4f} validation_kappa={v.get('Kappa', 0.0):.4f} validation_informedness={v.get('Informedness', 0.0):.4f} validation_markedness={v.get('Markedness', 0.0):.4f} validation_gini={v.get('Gini', 0.0):.4f} validation_optimal_threshold={v.get('OptThresh', 0.0):.4f} validation_standard_deviation_prediction={v.get('StdPred', 0.0):.4f} validation_percentage_above_threshold={v.get('PctAboveThresh', 0.0):.2f}", 'info')
                self.logger.log(f"[section 1] {arch_tag} [label_threshold_optimal] prediction_binary_split={pred_threshold:.2f}", 'info')
                
                # Define binary labels BEFORE hyperparameter optimization
                y_train_optimal = (y_train_continuous >= optimal_threshold).astype(int)
                y_val_binarized = (y_val_continuous >= optimal_threshold).astype(int)
                
                # ============================================================================
                # SECTION 3: Hyperparameter Optimization (Optuna)
                # ============================================================================
                # If ENABLE_HYPERPARAM_OPTIMIZATION is True:
                # - Run Optuna Bayesian optimization (20 trials, 20 epochs each)
                # - Search space: units, dropout, learning_rate
                # - Evaluate at 0.5 threshold (model outputs probabilities 0-1)
                # - Compare HPO precision vs pre-HPO precision at optimal_threshold
                # - Use HPO model ONLY if it improves; otherwise use pre-HPO model
                #
                # This is computationally expensive (~4 hours for RNN, ~6.5 for LSTM on CPU)
                
                # Run hyperparameter optimization if enabled
                enable_hyperparam = self.config.get('ENABLE_HYPERPARAM_OPTIMIZATION', True)
                best_hyperparams = {}
                hpo_best_model = None
                hpo_best_precision = 0.0
                hpo_improved = False
                hpo_val_precision = 0.0  # Default initialization
                
                # Default hyperparams for logging (used when HPO disabled or sklearn models)
                hpo_loss = 'N/A'
                hpo_lr = 0.0
                hpo_dropout = 0.0
                hpo_alpha = 0.0
                hpo_gamma = 0.0
                hpo_latent_dim = 0
                # Default HPO metrics (used when HPO disabled or sklearn models)
                hpo_R = 0.0
                hpo_AUC = 0.0
                hpo_F1 = 0.0
                hpo_TP = 0
                hpo_FP = 0
                hpo_TN = 0
                hpo_FN = 0
                # Default HPO prediction array (used when HPO disabled or sklearn models)
                hpo_val_pred = np.zeros(len(X_val_opt))  # Empty predictions default
                
                if enable_hyperparam:
                    sys.stdout.flush()
                    
                    # Create model builder with custom hyperparams (using optimal threshold's features)
                    def model_builder_with_params(hyperparams):
                        return self.model_trainer.build_architecture_with_params(
                            arch_name, X_train_opt.shape[1], hyperparams
                        )
                    
                    best_hyperparams, hpo_best_model, hpo_best_precision = self.hyperparam_optimizer.optimize(
                        arch_name=arch_name,
                        X_train=X_train_opt,
                        y_train=y_train_optimal,
                        X_val=X_val_opt,
                        y_val=y_val_binarized,
                        model_builder=model_builder_with_params,
                        train_func=self.model_trainer.train_model,
                        pred_threshold=pred_threshold,  # Use config-driven threshold for HPO
                        label_threshold=optimal_threshold
                    )
                    
                    # Extract hyperparams from best_hyperparams for logging
                    if best_hyperparams:
                        hpo_loss = best_hyperparams.get('loss_function', 'binary_crossentropy')
                        hpo_lr = best_hyperparams.get('learning_rate', 0.001)
                        hpo_dropout = best_hyperparams.get('dropout', 0.1)
                        hpo_alpha = best_hyperparams.get('alpha', 0.0)
                        hpo_gamma = best_hyperparams.get('gamma', 0.0)
                        hpo_latent_dim = best_hyperparams.get('latent_dim', 0)
                    
                    # Evaluate HPO model using prediction threshold from config
                    hpo_post_precision = 0.0  # Default
                    if hpo_best_model is not None:
                        hpo_val_pred = hpo_best_model.predict(X_val_opt, verbose=0).flatten()
                        if self.config.get('LOG_VERBOSITY', 0) >= 2:
                            self.logger.log(f"   [diagnostic-hyperparameter_optimization] " + format_diagnostic_string(hpo_val_pred, ""), 'info')
                        
                        # Search for best prediction threshold if enabled
                        if self.config.get('PREDICTION_THRESHOLD_SEARCH', False):
                            best_pred_threshold = 0.5
                            best_f1 = 0.0
                            for pred_thresh in np.arange(
                                self.config.get('PREDICTION_THRESHOLD_MIN', 0.1),
                                self.config.get('PREDICTION_THRESHOLD_MAX', 0.5) + 0.001,
                                self.config.get('PREDICTION_THRESHOLD_STEP', 0.05)
                            ):
                                hpo_binary_test = (hpo_val_pred >= pred_thresh).astype(int)
                                if hpo_binary_test.sum() >= self.config.get('MIN_POSITIVE_PREDICTIONS', 100):
                                    f1 = self.evaluator.calculate_f1(y_val_binarized, hpo_binary_test)
                                    if f1 > best_f1:
                                        best_f1 = f1
                                        best_pred_threshold = pred_thresh
                            pred_threshold = best_pred_threshold
                            self.logger.log(f"   [diagnostic] best_prediction_threshold_(hyperparameter_optimization): {pred_threshold:.2f} (f1={best_f1:.4f})", 'info')
                        else:
                            pred_threshold = self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)
                        
                        # Use prediction threshold 0.5 for fair comparison with Section 2
                        # (Section 2 uses 0.5, so using searched threshold would be unfair)
                        hpo_val_binary = (hpo_val_pred >= pred_threshold).astype(int)
                        hpo_val_precision = self.evaluator.calculate_precision(y_val_binarized, hpo_val_binary)
                        hpo_cm = self.evaluator.calculate_confusion_matrix(y_val_binarized, hpo_val_binary)
                        hpo_TP = hpo_cm['TP']
                        hpo_FP = hpo_cm['FP']
                        hpo_TN = hpo_cm['TN']
                        hpo_FN = hpo_cm['FN']
                        hpo_post_precision = hpo_val_precision
                        hpo_R = self.evaluator.calculate_recall(y_val_binarized, hpo_val_binary)
                        hpo_AUC = self.evaluator.calculate_auc(y_val_binarized, hpo_val_pred)
                        hpo_F1 = self.evaluator.calculate_f1(y_val_binarized, hpo_val_binary)
                        # Compare HPO vs pre-HPO precision at optimal_threshold
                        if hpo_val_precision > pre_hpo_val_precision:
                            hpo_improved = True
                            self.logger.log(f"   hyperparameter_optimization improved: {hpo_val_precision:.4f} > {pre_hpo_val_precision:.4f}", 'info')
                        else:
                            self.logger.log(f"   hyperparameter_optimization did NOT improve: {hpo_val_precision:.4f} <= {pre_hpo_val_precision:.4f} - using pre hyperparameter_optimization model", 'info')
                    
                    post_hpo_precisions.append(hpo_post_precision)
                else:
                    self.logger.log(f"   Hyperparameter optimization disabled, using defaults", 'info')
                
                # Section 3: HPO Summary
                # Determine which model to report based on HPO improvement
                model_label = "hyperparameter_optimization" if hpo_improved else "pre-hyperparameter_optimization"
                
                # Get predictions and metrics from appropriate model
                if not hpo_improved and threshold_opt_model is not None:
                    # HPO didn't improve - carry over Section 2 metrics (model+threshold unchanged)
                    # Do NOT re-evaluate: model and threshold are identical to Section 1,
                    # and re-evaluating produces different TP/FP counts (see longmemory.txt Pattern 10)
                    section3_pred = section2_pred
                    section3_R = section2_R
                    section3_AUC = section2_AUC
                    section3_F1 = section2_F1
                    section3_TP = section2_TP
                    section3_FP = section2_FP
                    section3_TN = section2_TN
                    section3_FN = section2_FN
                    section3_max_pred = section2_max_pred
                    section3_mean_pred = section2_mean_pred
                    section3_precision = pre_hpo_val_precision
                    section3_binary = (section3_pred.flatten() >= pred_threshold).astype(int)
                    section3_spec = section2_spec
                    section3_fpr = section2_fpr
                    section3_f2 = section2_f2
                    section3_mcc = section2_mcc
                    section3_prauc = section2_prauc
                    section3_balacc = section2_balacc
                    section3_brier = section2_brier
                    section3_kappa = section2_kappa
                    section3_informedness = section2_informedness
                    section3_markedness = section2_markedness
                    section3_gini = section2_gini
                    section3_opt_thresh = section2_opt_thresh
                    section3_std_pred = section2_std_pred
                    section3_pct_above = section2_pct_above
                    section3_metrics_carried = True
                elif hpo_best_model is not None:
                    # HPO improved - use HPO model predictions
                    section3_pred = hpo_best_model.predict(X_val_opt, verbose=0).flatten()
                    section3_binary = (section3_pred >= pred_threshold).astype(int)
                    section3_cm = self.evaluator.calculate_confusion_matrix(y_val_binarized, section3_binary)
                    section3_R = self.evaluator.calculate_recall(y_val_binarized, section3_binary)
                    section3_AUC = self.evaluator.calculate_auc(y_val_binarized, section3_pred)
                    section3_F1 = self.evaluator.calculate_f1(y_val_binarized, section3_binary)
                    section3_TP = section3_cm['TP']
                    section3_FP = section3_cm['FP']
                    section3_TN = section3_cm['TN']
                    section3_FN = section3_cm['FN']
                    section3_max_pred = section3_pred.max()
                    section3_mean_pred = section3_pred.mean()
                    section3_precision = hpo_val_precision
                    section3_metrics_carried = False
                else:
                    # All HPO trials rejected - use pre-HPO (Section 2) baseline metrics
                    self.logger.log(f"   [warning] All HPO trials rejected for {arch_name} - using Section 2 metrics", 'warning')
                    section3_pred = baseline_pred
                    section3_R = baseline_recall
                    section3_AUC = baseline_auc
                    section3_F1 = baseline_f1
                    section3_TP = baseline_cm['TP']
                    section3_FP = baseline_cm['FP']
                    section3_TN = baseline_cm['TN']
                    section3_FN = baseline_cm['FN']
                    section3_max_pred = baseline_pred.max()
                    section3_mean_pred = baseline_pred.mean()
                    section3_precision = pre_hpo_val_precision
                    section3_binary = (section3_pred.flatten() >= pred_threshold).astype(int)
                    section3_spec = baseline_spec
                    section3_fpr = baseline_fpr
                    section3_f2 = baseline_f2
                    section3_mcc = baseline_mcc
                    section3_prauc = baseline_prauc
                    section3_balacc = baseline_balacc
                    section3_brier = baseline_brier
                    section3_kappa = baseline_kappa
                    section3_informedness = baseline_informedness
                    section3_markedness = baseline_markedness
                    section3_gini = baseline_gini
                    section3_opt_thresh = baseline_opt_thresh
                    section3_std_pred = float(section3_pred.flatten().std()) if len(section3_pred.flatten()) > 0 else 0.0
                    section3_pct_above = (section3_pred.flatten() >= pred_threshold).mean() * 100 if len(section3_pred.flatten()) > 0 else 0.0
                    section3_metrics_carried = True
                
                # Safety net: recalculate precision from own TP/FP (guarantees self-consistency)
                if (section3_TP + section3_FP) > 0:
                    section3_precision = section3_TP / (section3_TP + section3_FP)
                
                section3_binary = (section3_pred.flatten() >= pred_threshold).astype(int)
                if not section3_metrics_carried:
                    section3_spec = self.evaluator.calculate_specificity(y_val_binarized, section3_binary)
                    section3_fpr = self.evaluator.calculate_fpr(y_val_binarized, section3_binary)
                    section3_f2 = self.evaluator.calculate_f2_score(y_val_binarized, section3_binary)
                    section3_mcc = self.evaluator.calculate_mcc(y_val_binarized, section3_binary)
                    section3_prauc = self.evaluator.calculate_average_precision(y_val_binarized, section3_pred.flatten())
                    section3_balacc = self.evaluator.calculate_balanced_accuracy(y_val_binarized, section3_binary)
                    section3_brier = self.evaluator.calculate_brier_score(y_val_binarized, section3_pred.flatten())
                    section3_kappa = self.evaluator.calculate_kappa(y_val_binarized, section3_binary)
                    section3_informedness = self.evaluator.calculate_informedness(y_val_binarized, section3_binary)
                    section3_markedness = self.evaluator.calculate_markedness(y_val_binarized, section3_binary)
                    section3_gini = self.evaluator.calculate_gini(y_val_binarized, section3_pred.flatten())
                    section3_opt_thresh = self.evaluator.calculate_optimal_threshold(y_val_binarized, section3_pred.flatten())
                
                arch_tag = f"[{arch_name.upper()}]"
                s3_label = f"[{model_label.upper()}]"
                section3_std_pred = float(section3_pred.flatten().std()) if len(section3_pred.flatten()) > 0 else 0.0
                section3_pct_above = (section3_pred.flatten() >= pred_threshold).mean() * 100 if len(section3_pred.flatten()) > 0 else 0.0
                self.logger.log(f"[section 3] {arch_tag} {s3_label} LABEL_THRESHOLD={optimal_threshold:.1f},", 'info')
                self.logger.log(f"[section 3] {arch_tag} {s3_label} VALIDATION_PRECISION={section3_precision:.4f} VALIDATION_TRUE_POSITIVES={section3_TP} VALIDATION_TRUE_NEGATIVES={section3_TN} validation_false_positives={section3_FP} validation_false_negatives={section3_FN} "
                             f"validation_max_prediction={section3_max_pred:.4f} validation_mean_prediction={section3_mean_pred:.4f} "
                             f"validation_recall={section3_R:.4f} validation_f1={section3_F1:.4f} validation_auc={section3_AUC:.4f} "
                             f"validation_specificity={section3_spec:.4f} validation_false_positive_rate={section3_fpr:.4f} validation_f2={section3_f2:.4f} validation_mcc={section3_mcc:.4f} validation_prauc={section3_prauc:.4f} validation_balanced_accuracy={section3_balacc:.4f} "
                             f"validation_brier={section3_brier:.4f} validation_kappa={section3_kappa:.4f} validation_informedness={section3_informedness:.4f} validation_markedness={section3_markedness:.4f} validation_gini={section3_gini:.4f} validation_optimal_threshold={section3_opt_thresh:.4f} "
                             f"validation_standard_deviation_prediction={section3_std_pred:.4f} validation_percentage_above_threshold={section3_pct_above:.2f}", 'info')
                self.logger.log(f"[section 3] {arch_tag} {s3_label} prediction_binary_split={pred_threshold:.2f} hyperparams=(loss={hpo_loss} lr={hpo_lr} dropout={hpo_dropout} alpha={hpo_alpha} gamma={hpo_gamma} latent_dim={hpo_latent_dim}) improved={hpo_improved} {best_hyperparams}", 'info')
                
                # ============================================================================
                # SECTION 4: Post-HPO Threshold Search (Apr 5, 2026)
                # ============================================================================
                # Run a second threshold search AFTER HPO using the HPO-optimized model
                # This finds the optimal threshold specifically for the HPO model
                # Compare pre-HPO vs post-HPO threshold precision and use the better one
                
                enable_post_hpo = self.config.get('ENABLE_POST_HPO_THRESHOLD_SEARCH', True)
                final_threshold = optimal_threshold
                threshold_source = 'section3'
                post_hpo_prec = 0.0  # Default value
                post_hpo_thresh = optimal_threshold  # Default
                post_hpo_results = []  # Default
                
                # Use the best model available: HPO model if exists, else threshold-optimized model
                model_for_post_hpo = threshold_opt_model if not hpo_improved else (hpo_best_model if hpo_best_model is not None else threshold_opt_model)
                
                if enable_post_hpo and model_for_post_hpo is not None:
                    self.logger.log(f"   Running post hyperparameter_optimization threshold search for {arch_name}...", 'info')
                    
                    # Use the trained model (either HPO-improved or pre-HPO)
                    # POST-hpo: use original HPO model WITHOUT retraining for each threshold
                    post_hpo_thresh, post_hpo_prec, post_hpo_results, _ = \
                        self.evaluator.find_optimal_threshold(
                            X_train, y_train_continuous, X_val, y_val_continuous,
                            model_for_post_hpo, self.model_trainer, arch_name,
                            thresholds, patience=999, retrain_model=False,
                            threshold_feature_indices=threshold_kept_indices
                        )
                    
                    # Compare HPO model threshold vs post-HPO threshold precision
                    self.logger.log(f"   hyperparameter_optimization model threshold search: LABEL_THRESHOLD={optimal_threshold:.1f}, VALIDATION_PRECISION={hpo_val_precision:.4f}", 'info')
                    self.logger.log(f"   post hyperparameter_optimization threshold search: LABEL_THRESHOLD={post_hpo_thresh:.1f}, VALIDATION_PRECISION={post_hpo_prec:.4f}", 'info')
                    
                    if post_hpo_prec > section3_precision:
                        final_threshold = post_hpo_thresh
                        threshold_source = 'section4'
                        # Re-slice features for the new final_threshold
                        # Section 4's find_optimal_threshold evaluated internally
                        # with threshold_feature_indices[post_hpo_thresh], so
                        # Section 5 must use the same feature set
                        new_thr_key = round(float(final_threshold), 1)
                        new_kept = threshold_kept_indices.get(new_thr_key, list(range(X_train.shape[1])))
                        new_kept = list(new_kept)
                        if new_kept != opt_kept:
                            X_train_opt = X_train[:, new_kept]
                            X_val_opt = X_val[:, new_kept]
                            opt_kept = new_kept
                            arch_kept_indices[-1] = new_kept
                            if arch_name in ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']:
                                scaler = StandardScaler()
                                X_train_opt = scaler.fit_transform(X_train_opt)
                                X_val_opt = scaler.transform(X_val_opt)
                                arch_scalers[arch_name] = scaler
                        self.logger.log(f"   post hyperparameter_optimization improved: using LABEL_THRESHOLD={post_hpo_thresh:.1f}", 'info')
                    else:
                        self.logger.log(f"   post hyperparameter_optimization no improvement: keeping LABEL_THRESHOLD={optimal_threshold:.1f}", 'info')
                
                # Section 4: Post-HPO Summary (17 metrics)
                # Get hyperparams from best_hyperparams (same as Section 3)
                s4_loss = best_hyperparams.get('loss_function', 'binary_crossentropy') if best_hyperparams else 'binary_crossentropy'
                s4_lr = best_hyperparams.get('learning_rate', 0.001) if best_hyperparams else 0.001
                s4_dropout = best_hyperparams.get('dropout', 0.0) if best_hyperparams else 0.0
                s4_alpha = best_hyperparams.get('alpha', 1.0) if best_hyperparams else 1.0
                s4_gamma = best_hyperparams.get('gamma', 1.0) if best_hyperparams else 1.0
                s4_latent_dim = best_hyperparams.get('latent_dim', 64) if best_hyperparams else 64
                
                # Get metrics from post_hpo_results at post_hpo_thresh
                section4_result = next((r for r in post_hpo_results if r['threshold'] == post_hpo_thresh), None) if post_hpo_results else None
                section4_val = section4_result.get('val', {}) if section4_result else {}
                s4_P = post_hpo_prec
                s4_R = section4_val.get('R', 0.0)
                s4_AUC = section4_val.get('AUC', 0.0)
                s4_F1 = section4_val.get('F1', 0.0)
                s4_TP = section4_val.get('TP', 0)
                s4_FP = section4_val.get('FP', 0)
                s4_TN = section4_val.get('TN', 0)
                s4_FN = section4_val.get('FN', 0)
                
                # Get MaxPred/MeanPred from model
                if model_for_post_hpo is not None:
                    s4_pred = model_for_post_hpo.predict(X_val_opt, verbose=0).flatten()
                    if self.config.get('LOG_VERBOSITY', 0) >= 2:
                        self.logger.log(f"[section 4] {arch_tag} [post hyperparameter_optimization] [diagnostic] " + format_diagnostic_string(s4_pred, ""), 'info')
                    s4_max_pred = s4_pred.max()
                    s4_mean_pred = s4_pred.mean()
                    # Default auxiliary prediction source: the post-HPO model
                    s4_aux_pred = s4_pred
                else:
                    s4_max_pred = 0.0
                    s4_mean_pred = 0.0
                    s4_aux_pred = np.array([])
                
                # If post-HPO model is not elected in S5, override metrics to match the actual elected model
                if post_hpo_prec == 0.0 or not (
                    model_for_post_hpo is not None
                    and hpo_best_model is not None
                    and threshold_source == 'section4'
                    and post_hpo_prec > section3_precision
                ):
                    if hpo_best_model is not None and hpo_improved:
                        # S5 will use HPO model — log its validation metrics
                        s4_P = hpo_val_precision
                        s4_R = hpo_R
                        s4_AUC = hpo_AUC
                        s4_F1 = hpo_F1
                        s4_TP = hpo_TP
                        s4_FP = hpo_FP
                        s4_TN = hpo_TN
                        s4_FN = hpo_FN
                        s4_max_pred = hpo_val_pred.max() if len(hpo_val_pred) > 0 else 0.0
                        s4_mean_pred = hpo_val_pred.mean() if len(hpo_val_pred) > 0 else 0.0
                        s4_aux_pred = hpo_val_pred
                    else:
                        # S5 will use Section 1 threshold-opt model — log its validation metrics
                        s4_P = pre_hpo_val_precision
                        s4_R = pre_hpo_val_recall
                        s4_AUC = section2_AUC
                        s4_F1 = section2_F1
                        s4_TP = section2_TP
                        s4_FP = section2_FP
                        s4_TN = section2_TN
                        s4_FN = section2_FN
                        s4_max_pred = section2_pred.max() if len(section2_pred) > 0 else 0.0
                        s4_mean_pred = section2_pred.mean() if len(section2_pred) > 0 else 0.0
                        s4_aux_pred = section2_pred
                
                s4_binary = (s4_aux_pred.flatten() >= pred_threshold).astype(int)
                s4_spec = self.evaluator.calculate_specificity(y_val_binarized, s4_binary)
                s4_fpr = self.evaluator.calculate_fpr(y_val_binarized, s4_binary)
                s4_f2 = self.evaluator.calculate_f2_score(y_val_binarized, s4_binary)
                s4_mcc = self.evaluator.calculate_mcc(y_val_binarized, s4_binary)
                s4_prauc = self.evaluator.calculate_average_precision(y_val_binarized, s4_aux_pred.flatten())
                s4_balacc = self.evaluator.calculate_balanced_accuracy(y_val_binarized, s4_binary)
                s4_brier = self.evaluator.calculate_brier_score(y_val_binarized, s4_aux_pred.flatten())
                s4_kappa = self.evaluator.calculate_kappa(y_val_binarized, s4_binary)
                s4_informedness = self.evaluator.calculate_informedness(y_val_binarized, s4_binary)
                s4_markedness = self.evaluator.calculate_markedness(y_val_binarized, s4_binary)
                s4_gini = self.evaluator.calculate_gini(y_val_binarized, s4_aux_pred.flatten())
                s4_opt_thresh = self.evaluator.calculate_optimal_threshold(y_val_binarized, s4_aux_pred.flatten())
                
                s4_std_pred = float(s4_aux_pred.flatten().std()) if len(s4_aux_pred.flatten()) > 0 else 0.0
                s4_pct_above = (s4_aux_pred.flatten() >= pred_threshold).mean() * 100 if len(s4_aux_pred.flatten()) > 0 else 0.0
                self.logger.log(f"[section 4] {arch_tag} [post hyperparameter_optimization] LABEL_THRESHOLD={final_threshold:.1f},", 'info')
                self.logger.log(f"[section 4] {arch_tag} [post hyperparameter_optimization] VALIDATION_PRECISION={s4_P:.4f} VALIDATION_TRUE_POSITIVES={s4_TP} VALIDATION_TRUE_NEGATIVES={s4_TN} validation_false_positives={s4_FP} validation_false_negatives={s4_FN} "
                             f"validation_max_prediction={s4_max_pred:.4f} validation_mean_prediction={s4_mean_pred:.4f} "
                             f"validation_recall={s4_R:.4f} validation_f1={s4_F1:.4f} validation_auc={s4_AUC:.4f} "
                             f"validation_specificity={s4_spec:.4f} validation_false_positive_rate={s4_fpr:.4f} validation_f2={s4_f2:.4f} validation_mcc={s4_mcc:.4f} validation_prauc={s4_prauc:.4f} validation_balanced_accuracy={s4_balacc:.4f} "
                             f"validation_brier={s4_brier:.4f} validation_kappa={s4_kappa:.4f} validation_informedness={s4_informedness:.4f} validation_markedness={s4_markedness:.4f} validation_gini={s4_gini:.4f} validation_optimal_threshold={s4_opt_thresh:.4f} "
                             f"validation_standard_deviation_prediction={s4_std_pred:.4f} validation_percentage_above_threshold={s4_pct_above:.2f}", 'info')
                self.logger.log(f"[section 4] {arch_tag} [post hyperparameter_optimization] prediction_binary_split={pred_threshold:.2f} hyperparams=(loss={s4_loss} lr={s4_lr} dropout={s4_dropout} alpha={s4_alpha} gamma={s4_gamma} latent_dim={s4_latent_dim}) source={threshold_source}", 'info')
                
                # Update binary labels with final threshold
                y_train_optimal = (y_train_continuous >= final_threshold).astype(int)
                y_val_binarized = (y_val_continuous >= final_threshold).astype(int)
                
                # ============================================================================
                # SECTION 5: Model Selection & Final Training
                # ============================================================================
                # Three scenarios:
                # 1. HPO improved: Use HPO best model with optimized hyperparameters
                # 2. HPO didn't improve: Use pre-HPO threshold-optimized model
                # 3. No hpo: Use default hyperparameters
                #
                # PLUS: Post-HPO threshold discovery: when Section 4 finds the HPO model
                # performs better at a different label_threshold than optimal_threshold,
                # use the HPO model regardless of hpo_improved status.
                #
                # Final model produces probabilities (0-1) from model.predict()
                
                # Check if post-HPO threshold search found a better operating point.
                # When post_hpo_prec exceeds Section 3's elected precision, the HPO model
                # generalizes better at the new threshold — use it even when hpo_improved=False.
                use_post_hpo_model = (
                    model_for_post_hpo is not None
                    and hpo_best_model is not None
                    and threshold_source == 'section4'
                    and post_hpo_prec > section3_precision
                )
                
                if use_post_hpo_model:
                    # Use the HPO model evaluated at post_hpo_thresh
                    trained_model = model_for_post_hpo
                    train_epochs = best_hyperparams.get('epochs', 3) if best_hyperparams else 3
                    training_history = {}
                    self.logger.log(
                        f"   Using post-HPO model at LABEL_THRESHOLD={final_threshold:.1f} "
                        f"(post_hpo_prec={post_hpo_prec:.4f} > "
                        f"hpo={hpo_val_precision:.4f}, pre_hpo={pre_hpo_val_precision:.4f})",
                        'info'
                    )
                # Use the best model: HPO only if it actually improved at optimal_threshold
                elif hpo_best_model is not None and hpo_improved:
                    # Use HPO model directly WITHOUT retraining
                    trained_model = hpo_best_model
                    train_epochs = best_hyperparams.get('epochs', 3) if best_hyperparams else 3
                    training_history = {}
                    self.logger.log(f"   Using Hyperparameter_Optimized model directly (no change to Label_Threshold)", 'info')
                else:
                    # Only retrain if NOT using HPO model
                    # Default epochs value
                    train_epochs = 3
                    
                    if best_hyperparams and not hpo_improved and threshold_opt_model is not None:
                        # HPO didn't improve - use the model from threshold optimization (pre-HPO model)
                        # Do NOT retrain - the threshold_opt_model already achieved the best precision
                        trained_model = threshold_opt_model
                        train_epochs = 3
                        training_history = {}
                        self.logger.log(f"   Using threshold-optimized model (HPO did not improve)", 'info')
                    elif best_hyperparams:
                        model = self.model_trainer.build_architecture_with_params(
                            arch_name, X_train_opt.shape[1], best_hyperparams
                        )
                        self.logger.log(f"   Using optimized hyperparameters: {best_hyperparams}", 'info')
                        # Use architecture-specific epochs if specified
                        train_epochs = best_hyperparams.get('epochs', 3)
                        if hasattr(model, 'sklearn_model'):
                            trained_model, _ = self.model_trainer._train_sklearn_model(
                                model, X_train_opt, y_train_optimal, sample_weight=np.sqrt(weights_train)
                            )
                            training_history = {}
                        else:
                            trained_model, training_history = self.model_trainer.train_model(
                                model, X_train_opt, y_train_optimal, epochs=train_epochs, verbose=0,
                                sample_weight=np.sqrt(weights_train)
                            )
                    else:
                        model = self.model_trainer.build_architecture(arch_name, X_train_opt.shape[1], y_train_continuous)
                        # Use architecture-specific epochs (increased for failing architectures)
                        arch_epochs = {'Dense': 15, 'VAE': 30, 'CNN': 20, 'LSTM': 20, 'Transformer': 20}
                        train_epochs = arch_epochs.get(arch_name, 3)
                        if hasattr(model, 'sklearn_model'):
                            trained_model, _ = self.model_trainer._train_sklearn_model(
                                model, X_train_opt, y_train_optimal, sample_weight=np.sqrt(weights_train)
                            )
                            training_history = {}
                        else:
                            trained_model, training_history = self.model_trainer.train_model(
                                model, X_train_opt, y_train_optimal, epochs=train_epochs, verbose=0,
                                sample_weight=np.sqrt(weights_train)
                            )
                
                # Get predictions and final metrics
                # Model outputs probabilities (0-1), so use 0.5 for binary conversion
                # (not optimal_threshold which is used for creating binary labels during training)
                if trained_model is None:
                    raise ValueError(f"[fatal] {arch_name} training failed: trained_model is None")
                
                train_pred = trained_model.predict(X_train_opt, verbose=0).flatten()
                val_pred = trained_model.predict(X_val_opt, verbose=0).flatten()
                
                # Validate predictions are in valid probability range [0,1]
                if np.any(np.isnan(train_pred)) or np.any(np.isnan(val_pred)):
                    self.logger.log(f"   [warning] NaN values detected in predictions!", 'warning')
                if train_pred.min() < 0 or train_pred.max() > 1:
                    self.logger.log(f"   [warning] Predictions outside [0,1] range! min={train_pred.min():.4f}, max={train_pred.max():.4f}", 'warning')
                
                # Log prediction distribution statistics
                self.logger.log(f"   Train predictions: mean={train_pred.mean():.4f}, std={train_pred.std():.4f}, min={train_pred.min():.4f}, max={train_pred.max():.4f}", 'info')
                self.logger.log(f"   validation predictions:   mean={val_pred.mean():.4f}, std={val_pred.std():.4f}, min={val_pred.min():.4f}, max={val_pred.max():.4f}", 'info')
                # Detect near-constant predictions (e.g., VAE decoder collapse)
                if train_pred.std() < 1e-6 or val_pred.std() < 1e-6:
                    self.logger.log(f"   [warning] Near-constant predictions: train_std={train_pred.std():.6f}, val_std={val_pred.std():.6f}", 'warning')
                if self.config.get('LOG_VERBOSITY', 0) >= 2:
                    self.logger.log(f"[section 5] {arch_tag} [final] [diagnostic] Train " + format_diagnostic_string(train_pred, ""), 'info')
                    self.logger.log(f"[section 5] {arch_tag} [final] [diagnostic] validation   " + format_diagnostic_string(val_pred, ""), 'info')
                
                # Search for best prediction threshold if enabled
                if self.config.get('PREDICTION_THRESHOLD_SEARCH', False):
                    y_train_binary = (y_train_continuous >= optimal_threshold).astype(int)
                    y_val_binary_search = (y_val_continuous >= optimal_threshold).astype(int)
                    
                    best_pred_threshold = 0.5
                    best_f1 = 0.0
                    search_results = []
                    
                    for pred_thresh in np.arange(
                        self.config.get('PREDICTION_THRESHOLD_MIN', 0.1),
                        self.config.get('PREDICTION_THRESHOLD_MAX', 0.5) + 0.001,
                        self.config.get('PREDICTION_THRESHOLD_STEP', 0.05)
                    ):
                        val_binary_test = (val_pred >= pred_thresh).astype(int)
                        if val_binary_test.sum() >= self.config.get('MIN_POSITIVE_PREDICTIONS', 100):
                            f1 = self.evaluator.calculate_f1(y_val_binary_search, val_binary_test)
                            precision = self.evaluator.calculate_precision(y_val_binary_search, val_binary_test)
                            recall = self.evaluator.calculate_recall(y_val_binary_search, val_binary_test)
                            search_results.append((pred_thresh, precision, recall, f1))
                            if f1 > best_f1:
                                best_f1 = f1
                                best_pred_threshold = pred_thresh
                    
                    if search_results:
                        self.logger.log(f"   [diagnostic] prediction_threshold_search: tested {len(search_results)} thresholds", 'info')
                        top_results = sorted(search_results, key=lambda x: x[3], reverse=True)[:3]
                        for r in top_results:
                            self.logger.log(f"   [diag]   thresh={r[0]:.2f}: VALIDATION_PRECISION={r[1]:.4f} validation_recall={r[2]:.4f} validation_f1={r[3]:.4f}", 'info')
                        self.logger.log(f"   [diagnostic] best_prediction_threshold: {best_pred_threshold:.2f} (validation_f1={best_f1:.4f})", 'info')
                    
                    pred_threshold = best_pred_threshold
                else:
                    pred_threshold = self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)
                
                train_binary = (train_pred >= pred_threshold).astype(int)
                val_binary = (val_pred >= pred_threshold).astype(int)
                
                # Log prediction class distribution
                train_pos_pct = train_binary.mean() * 100
                val_pos_pct = val_binary.mean() * 100
                self.logger.log(f"   Train predictions: {train_pos_pct:.2f}% positive predictions ({train_binary.sum():,} / {len(train_binary):,})", 'info')
                self.logger.log(f"   validation predictions:   {val_pos_pct:.2f}% positive predictions ({val_binary.sum():,} / {len(val_binary):,})", 'info')
                
                # Check for all-zero predictions (early warning)
                if train_binary.sum() == 0:
                    self.logger.log(f"   [warning] Model predicts ALL NEGATIVES on training data!", 'warning')
                if val_binary.sum() == 0:
                    self.logger.log(f"   [warning] Model predicts ALL NEGATIVES on validation data!", 'warning')
                
                # If Section 4 was rejected (all thresholds rejected), use Section 3 results directly
                # This ensures consistency when no improvement found in Section 4
                if threshold_source == 'section3':
                    self.logger.log(f"   [info] Section 4 all rejected - using Section 3 results for Section 5", 'info')
                    # FIX: Use pre-HPO precision when HPO didn't improve (ensures best metrics)
                    # FIX: val_cm moved inside each branch so P/R/F1 match TP/FP/TN/FN (Bug F1)
                    if not hpo_improved:
                        val_precision = pre_hpo_val_precision
                        val_recall = pre_hpo_val_recall
                        val_auc = section2_AUC
                        val_f1 = section2_F1
                        val_cm = {'TP': section2_TP, 'FP': section2_FP, 'TN': section2_TN, 'FN': section2_FN}
                    else:
                        val_precision = hpo_val_precision
                        val_recall = hpo_R
                        val_auc = hpo_AUC
                        val_f1 = hpo_F1
                        val_cm = {'TP': hpo_TP, 'FP': hpo_FP, 'TN': hpo_TN, 'FN': hpo_FN}
                    # Also use pre-HPO predictions when HPO didn't improve
                    if not hpo_improved and threshold_opt_model is not None:
                        val_pred = threshold_opt_model.predict(X_val_opt, verbose=0).flatten()
                        train_pred = threshold_opt_model.predict(X_train_opt, verbose=0).flatten()
                        # FIX: Update val_binary to match swapped val_pred (Bug F2)
                        val_binary = (val_pred >= pred_threshold).astype(int)
                        # Calculate train metrics from pre-HPO predictions
                        train_binary = (train_pred >= pred_threshold).astype(int)
                        train_cm = self.evaluator.calculate_confusion_matrix(y_train_optimal, train_binary)
                        train_precision = self.evaluator.calculate_precision(y_train_optimal, train_binary)
                        train_recall = self.evaluator.calculate_recall(y_train_optimal, train_binary)
                        train_auc = self.evaluator.calculate_auc(y_train_optimal, train_pred)
                        train_f1 = self.evaluator.calculate_f1(y_train_optimal, train_binary)
                    else:
                        val_pred = hpo_val_pred if len(hpo_val_pred) > 0 else np.zeros(len(X_val))
                        # FIX: Update val_binary to match swapped val_pred (Bug F2)
                        val_binary = (val_pred >= pred_threshold).astype(int)
                        train_cm = self.evaluator.calculate_confusion_matrix(y_train_optimal, train_binary)
                        s5_max_pred = val_pred.max() if len(val_pred) > 0 else 0.0
                        s5_mean_pred = val_pred.mean() if len(val_pred) > 0 else 0.0
                        train_precision = self.evaluator.calculate_precision(y_train_optimal, train_binary)
                        train_recall = self.evaluator.calculate_recall(y_train_optimal, train_binary)
                        train_auc = self.evaluator.calculate_auc(y_train_optimal, train_pred)
                        train_f1 = self.evaluator.calculate_f1(y_train_optimal, train_binary)
                else:
                    # Calculate final metrics normally (original logic)
                    train_cm = self.evaluator.calculate_confusion_matrix(y_train_optimal, train_binary)
                    val_cm = self.evaluator.calculate_confusion_matrix(y_val_binarized, val_binary)
                    train_precision = self.evaluator.calculate_precision(y_train_optimal, train_binary)
                    val_precision = self.evaluator.calculate_precision(y_val_binarized, val_binary)
                    train_recall = self.evaluator.calculate_recall(y_train_optimal, train_binary)
                    val_recall = self.evaluator.calculate_recall(y_val_binarized, val_binary)
                    train_f1 = self.evaluator.calculate_f1(y_train_optimal, train_binary)
                    val_f1 = self.evaluator.calculate_f1(y_val_binarized, val_binary)
                    train_auc = self.evaluator.calculate_auc(y_train_optimal, train_pred)
                    val_auc = self.evaluator.calculate_auc(y_val_binarized, val_pred)
                    s5_max_pred = val_pred.max() if len(val_pred) > 0 else 0.0
                    s5_mean_pred = val_pred.mean() if len(val_pred) > 0 else 0.0
                    
                    # Apply inverse log transform if configured (convert predictions back to original scale)
                    if self.config.get('LOG_TRANSFORM_TARGET', False):
                        train_pred_original = inverse_log_transform(train_pred)
                        val_pred_original = inverse_log_transform(val_pred)
                    else:
                        train_pred_original = train_pred
                        val_pred_original = val_pred
                    
                    # Calculate new metrics (MCC, PR-AUC, Specificity, BalancedAccuracy)
                    train_metrics = self.evaluator.calculate_metrics(y_train_optimal, train_binary, train_pred_original)
                    val_metrics = self.evaluator.calculate_metrics(y_val_binarized, val_binary, val_pred_original)
                    
                    train_mcc = train_metrics.get('mcc', 0.0)
                    val_mcc = val_metrics.get('mcc', 0.0)
                    train_prauc = train_metrics.get('average_precision', 0.0)
                    val_prauc = val_metrics.get('average_precision', 0.0)
                    train_specificity = train_metrics.get('specificity', 0.0)
                    val_specificity = val_metrics.get('specificity', 0.0)
                    train_balanced_acc = train_metrics.get('balanced_accuracy', 0.0)
                    val_balanced_acc = val_metrics.get('balanced_accuracy', 0.0)
                
                # Apply inverse log transform if configured (convert predictions back to original scale)
                if self.config.get('LOG_TRANSFORM_TARGET', False):
                    train_pred_original = inverse_log_transform(train_pred)
                    val_pred_original = inverse_log_transform(val_pred)
                else:
                    train_pred_original = train_pred
                    val_pred_original = val_pred
                
                # Calculate new metrics (MCC, PR-AUC, Specificity, BalancedAccuracy)
                train_metrics = self.evaluator.calculate_metrics(y_train_optimal, train_binary, train_pred_original)
                val_metrics = self.evaluator.calculate_metrics(y_val_binarized, val_binary, val_pred_original)
                
                # Ensure val_binary_preds is defined for SECTION 5 metrics calculation
                val_binary_preds = (val_pred >= pred_threshold).astype(int)
                
                train_mcc = train_metrics.get('mcc', 0.0)
                val_mcc = val_metrics.get('mcc', 0.0)
                train_prauc = train_metrics.get('average_precision', 0.0)
                val_prauc = val_metrics.get('average_precision', 0.0)
                train_specificity = train_metrics.get('specificity', 0.0)
                val_specificity = val_metrics.get('specificity', 0.0)
                train_balanced_acc = train_metrics.get('balanced_accuracy', 0.0)
                val_balanced_acc = val_metrics.get('balanced_accuracy', 0.0)
                
                train_fpr = self.evaluator.calculate_fpr(y_train_optimal, train_binary)
                train_f2 = self.evaluator.calculate_f2_score(y_train_optimal, train_binary)
                train_max_pred = float(train_pred_original.max()) if len(train_pred_original) > 0 else 0.0
                train_mean_pred = float(train_pred_original.mean()) if len(train_pred_original) > 0 else 0.0
                train_std_pred = float(train_pred_original.std()) if len(train_pred_original) > 0 else 0.0
                train_pct_above_thresh = (train_pred_original >= pred_threshold).mean() * 100 if len(train_pred_original) > 0 else 0.0
                
                val_fpr = self.evaluator.calculate_fpr(y_val_binarized, val_binary_preds)
                val_f2 = self.evaluator.calculate_f2_score(y_val_binarized, val_binary_preds)
                val_std_pred = float(val_pred.std()) if len(val_pred) > 0 else 0.0
                val_pct_above_thresh = (val_pred >= pred_threshold).mean() * 100 if len(val_pred) > 0 else 0.0
                
                train_brier = self.evaluator.calculate_brier_score(y_train_optimal, train_pred_original.flatten())
                val_brier = self.evaluator.calculate_brier_score(y_val_binarized, val_pred.flatten())
                train_kappa = self.evaluator.calculate_kappa(y_train_optimal, train_binary)
                val_kappa = self.evaluator.calculate_kappa(y_val_binarized, val_binary_preds)
                train_informedness = self.evaluator.calculate_informedness(y_train_optimal, train_binary)
                val_informedness = self.evaluator.calculate_informedness(y_val_binarized, val_binary_preds)
                train_markedness = self.evaluator.calculate_markedness(y_train_optimal, train_binary)
                val_markedness = self.evaluator.calculate_markedness(y_val_binarized, val_binary_preds)
                train_gini = self.evaluator.calculate_gini(y_train_optimal, train_pred_original.flatten())
                val_gini = self.evaluator.calculate_gini(y_val_binarized, val_pred.flatten())
                train_opt_thresh = self.evaluator.calculate_optimal_threshold(y_train_optimal, train_pred_original.flatten())
                val_opt_thresh = self.evaluator.calculate_optimal_threshold(y_val_binarized, val_pred.flatten())
                
                # Prediction bucket distribution
                pred_bins = np.digitize(val_pred.flatten(), np.arange(0.1, 1.0, 0.1))
                bucket_counts = np.bincount(pred_bins, minlength=11)
                self.logger.log(f"   [diagnostic] prediction_buckets: 0-10:{bucket_counts[1]}, 10-20:{bucket_counts[2]}, 20-30:{bucket_counts[3]}, 30-40:{bucket_counts[4]}, 40-50:{bucket_counts[5]}, 50-60:{bucket_counts[6]}, 60-70:{bucket_counts[7]}, 70-80:{bucket_counts[8]}, 80-90:{bucket_counts[9]}, 90-100:{bucket_counts[10]}", 'info')
                
                # Section 5: Final Summary (16 metrics)
                # Get hyperparams from best_hyperparams (same as Section 3/4)
                s5_loss = best_hyperparams.get('loss_function', 'binary_crossentropy') if best_hyperparams else 'binary_crossentropy'
                s5_lr = best_hyperparams.get('learning_rate', 0.001) if best_hyperparams else 0.001
                s5_dropout = best_hyperparams.get('dropout', 0.0) if best_hyperparams else 0.0
                s5_alpha = best_hyperparams.get('alpha', 1.0) if best_hyperparams else 1.0
                s5_gamma = best_hyperparams.get('gamma', 1.0) if best_hyperparams else 1.0
                s5_latent_dim = best_hyperparams.get('latent_dim', 64) if best_hyperparams else 64
                
                # Get MaxPred/MeanPred from val_pred
                s5_max_pred = val_pred.max() if len(val_pred) > 0 else 0.0
                s5_mean_pred = val_pred.mean() if len(val_pred) > 0 else 0.0
                
                # Use val_pred as final_pred for SECTION 5 metrics
                final_pred = val_pred
                
                s5_binary = (final_pred.flatten() >= pred_threshold).astype(int)
                s5_spec = self.evaluator.calculate_specificity(y_val_binarized, s5_binary)
                s5_fpr = self.evaluator.calculate_fpr(y_val_binarized, s5_binary)
                s5_f2 = self.evaluator.calculate_f2_score(y_val_binarized, s5_binary)
                s5_mcc = self.evaluator.calculate_mcc(y_val_binarized, s5_binary)
                s5_prauc = self.evaluator.calculate_average_precision(y_val_binarized, final_pred.flatten())
                s5_balacc = self.evaluator.calculate_balanced_accuracy(y_val_binarized, s5_binary)
                s5_brier = self.evaluator.calculate_brier_score(y_val_binarized, final_pred.flatten())
                s5_kappa = self.evaluator.calculate_kappa(y_val_binarized, s5_binary)
                s5_informedness = self.evaluator.calculate_informedness(y_val_binarized, s5_binary)
                s5_markedness = self.evaluator.calculate_markedness(y_val_binarized, s5_binary)
                s5_gini = self.evaluator.calculate_gini(y_val_binarized, final_pred.flatten())
                s5_opt_thresh = self.evaluator.calculate_optimal_threshold(y_val_binarized, final_pred.flatten())
                
                self.logger.log(f"[section 5] {arch_tag} [final] LABEL_THRESHOLD={final_threshold:.1f},", 'info')
                self.logger.log(f"[section 5] {arch_tag} [final] train_precision={train_precision:.4f} train_true_positives={train_cm['TP']} train_true_negatives={train_cm['TN']} train_false_positives={train_cm['FP']} train_false_negatives={train_cm['FN']} train_max_prediction={train_max_pred:.4f} train_mean_prediction={train_mean_pred:.4f} train_recall={train_recall:.4f} train_f1={train_f1:.4f} train_auc={train_auc:.4f} train_specificity={train_specificity:.4f} train_false_positive_rate={train_fpr:.4f} train_f2={train_f2:.4f} train_mcc={train_mcc:.4f} train_prauc={train_prauc:.4f} train_balanced_accuracy={train_balanced_acc:.4f} train_standard_deviation_prediction={train_std_pred:.4f} train_percentage_above_threshold={train_pct_above_thresh:.2f} train_brier={train_brier:.4f} train_kappa={train_kappa:.4f} train_informedness={train_informedness:.4f} train_markedness={train_markedness:.4f} train_gini={train_gini:.4f} train_optimal_threshold={train_opt_thresh:.4f}", 'info')
                self.logger.log(f"[section 5] {arch_tag} [final] VALIDATION_PRECISION={val_precision:.4f} VALIDATION_TRUE_POSITIVES={val_cm['TP']} VALIDATION_TRUE_NEGATIVES={val_cm['TN']} validation_false_positives={val_cm['FP']} validation_false_negatives={val_cm['FN']} "
                             f"validation_max_prediction={s5_max_pred:.4f} validation_mean_prediction={s5_mean_pred:.4f} "
                             f"validation_recall={val_recall:.4f} validation_f1={val_f1:.4f} validation_auc={val_auc:.4f} "
                             f"validation_specificity={s5_spec:.4f} validation_false_positive_rate={s5_fpr:.4f} validation_f2={s5_f2:.4f} validation_mcc={s5_mcc:.4f} validation_prauc={s5_prauc:.4f} validation_balanced_accuracy={s5_balacc:.4f} "
                             f"validation_brier={s5_brier:.4f} validation_kappa={s5_kappa:.4f} validation_informedness={s5_informedness:.4f} validation_markedness={s5_markedness:.4f} validation_gini={s5_gini:.4f} validation_optimal_threshold={s5_opt_thresh:.4f} "
                             f"validation_standard_deviation_prediction={val_std_pred:.4f} validation_percentage_above_threshold={val_pct_above_thresh:.2f}", 'info')
                self.logger.log(f"[section 5] {arch_tag} [final] prediction_binary_split={pred_threshold:.2f} hyperparams=(loss={s5_loss} lr={s5_lr} dropout={s5_dropout} alpha={s5_alpha} gamma={s5_gamma} latent_dim={s5_latent_dim}) (using HPO model directly)", 'info')
                
                # Log additional configuration details for analysis
                loss_fn = best_hyperparams.get('loss_function', 'binary_crossentropy') if best_hyperparams else 'binary_crossentropy'
                self.logger.log(f"   [{arch_name}] loss: {loss_fn} | pred_threshold: {pred_threshold} | epochs: {train_epochs}", 'info')
                
                # Log key hyperparameters if available
                if best_hyperparams:
                    hpo_keys = ['learning_rate', 'dropout', 'alpha', 'gamma', 'latent_dim', 'filters', 'units', 'lstm_units', 'dim', 'heads', 'kernel_size', 'layers']
                    hpo_str = ', '.join([f"{k}={v}" for k, v in best_hyperparams.items() if k in hpo_keys])
                    if hpo_str:
                        self.logger.log(f"   [{arch_name}] hyperparameter_optimization: {hpo_str}", 'info')
                
                sys.stdout.flush()
                
                # Extract training history metrics
                train_loss = 0.0
                val_loss = 0.0
                best_epoch = 0
                if training_history and isinstance(training_history, dict):
                    if 'loss' in training_history and len(training_history['loss']) > 0:
                        train_loss = float(training_history['loss'][-1])
                    if 'val_loss' in training_history and len(training_history['val_loss']) > 0:
                        val_loss = float(training_history['val_loss'][-1])
                    # Best epoch is typically when early stopping restored best weights
                    best_epoch = len(training_history.get('loss', []))
                
                loss_delta = train_loss - val_loss if (train_loss > 0 and val_loss > 0) else 0.0
                
                # Calculate HPO improvement (with error handling)
                hpo_improvement = 0.0
                try:
                    if hpo_best_precision > 0 and baseline_precision > 0:
                        hpo_improvement = hpo_best_precision - baseline_precision
                except Exception:
                    hpo_improvement = 0.0
                
                # Build key hyperparameters string
                key_hp = []
                if best_hyperparams:
                    for k in ['units', 'lstm_units', 'filters', 'layers', 'dim', 'heads', 'dropout', 'learning_rate', 'latent_dim']:
                        if k in best_hyperparams:
                            key_hp.append(f"{k}={best_hyperparams[k]}")
                key_hyperparams_str = ','.join(key_hp) if key_hp else 'default'
                
                # Store final metrics for summary table
                arch_final_metrics.append({
                    'arch': arch_name,
                    'optimal_label_threshold': final_threshold,
                    'P': val_precision,
                    'R': val_recall,
                    'AUC': val_auc,
                    'F1': val_f1,
                    'FN': val_cm['FN'],
                    'TN': val_cm['TN'],
                    'TP': val_cm['TP'],
                    'FP': val_cm['FP'],
                    'train_P': train_precision,
                    'train_R': train_recall,
                    'train_AUC': train_auc,
                    'train_F1': train_f1,
                    'train_TP': train_cm['TP'],
                    'train_FP': train_cm['FP'],
                    'train_TN': train_cm['TN'],
                    'train_FN': train_cm['FN'],
                    'epochs_trained': train_epochs,
                    # NEW: Prediction distribution
                    'train_mean_pred': float(train_pred.mean()) if len(train_pred) > 0 else 0.0,
                    'train_std_pred': float(train_pred.std()) if len(train_pred) > 0 else 0.0,
                    'val_mean_pred': float(val_pred.mean()) if len(val_pred) > 0 else 0.0,
                    'val_std_pred': float(val_pred.std()) if len(val_pred) > 0 else 0.0,
                    'val_max_pred': float(val_pred.max()) if len(val_pred) > 0 else 0.0,
                    # NEW: Training dynamics
                    'best_epoch': best_epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'loss_delta': loss_delta,
                    # NEW: HPO details
                    'hpo_trials': 20,
                    'hpo_improvement': hpo_improvement,
                    'key_hyperparams': key_hyperparams_str,
                    # NEW: Enhanced metrics
                    'val_mcc': val_mcc,
                    'val_prauc': val_prauc,
                    'val_specificity': val_specificity,
                    'val_balanced_acc': val_balanced_acc,
                    'train_mcc': train_mcc,
                    'train_prauc': train_prauc,
                    'train_specificity': train_specificity,
                    'train_balanced_acc': train_balanced_acc,
                    # NEW: Post-HPO threshold search
                    'threshold_source': threshold_source,
                    'final_label_threshold': final_threshold,
                })
                
                # Track per-architecture data for Phase 5
                arch_names.append(arch_name)
                best_hyperparams_list.append(best_hyperparams if best_hyperparams else {})
                best_val_precision_list.append(val_precision)
                final_thresholds.append(final_threshold)
                
                trained_models.append(trained_model)
                val_predictions.append(val_pred)
                
                # Track training time
                arch_time = time.time() - arch_start_time
                arch_training_times.append(arch_time)
                self.logger.log(f"   {arch_name} training time: {arch_time:.1f}s", 'info')
                
                self.logger.log(f"[pass] {arch_name} trained successfully", 'info')
            
            except Exception as e:
                self.logger.log(f"[error] {arch_name} training failed: {e}", 'warning')
                # Add placeholder metrics so the table is not empty
                arch_final_metrics.append({
                    'arch': arch_name,
                    'optimal_label_threshold': 2.0,
                    'P': 0.0, 'R': 0.0, 'AUC': 0.0, 'F1': 0.0,
                    'FN': 0, 'TN': 0, 'TP': 0, 'FP': 0,
                    'train_P': 0.0, 'train_R': 0.0, 'train_AUC': 0.0, 'train_F1': 0.0,
                    'train_TP': 0, 'train_FP': 0, 'train_TN': 0, 'train_FN': 0,
                    'epochs_trained': 0,
                    'val_max_pred': 0.0, 'train_mean_pred': 0.0, 'train_std_pred': 0.0,
                    'val_mean_pred': 0.0, 'val_std_pred': 0.0,
                    'best_epoch': 0, 'train_loss': 0.0, 'val_loss': 0.0, 'loss_delta': 0.0,
                    'hpo_trials': 0, 'hpo_improvement': 0.0, 'key_hyperparams': 'failed',
                })
                continue
        
        # ============================================================================
        # DIAGNOSTICS: Feature Stability, Inference Latency, Sliding Window (Items 1, 2, 4)
        # ============================================================================
        unique_dates = np.unique(dates)
        
        # Item 1: Feature Stability Analysis
        if self.config.get('FEATURE_STABILITY_ANALYSIS', False):
            self.logger.log("Running Feature Stability Analysis...", 'info')
            try:
                train_dates_mask = dates < np.median(unique_dates)
                val_dates_mask = dates >= np.median(unique_dates)
                
                # PRIORITY 2 FIX: Validate dimension alignment (May 7, 2026)
                # Ensure X and y_binary have matching dimensions
                min_samples = min(X.shape[0], y_binary.shape[0])
                X_fs = X[:min_samples]
                y_binary_fs = y_binary[:min_samples]
                train_dates_mask = train_dates_mask[:min_samples]
                val_dates_mask = val_dates_mask[:min_samples]
                
                if train_dates_mask.sum() > 0 and val_dates_mask.sum() > 0:
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # Binarize y_binary for classifier (line 1150 expects integer 0/1)
                    optimal_threshold = self.config.get('FIRST_THRESHOLD', 2.0)
                    y_binary_binarized = (y_binary_fs >= optimal_threshold).astype(int)
                    
                    stability_results = {}
                    for i, fname in enumerate(feature_names):
                        rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
                        rf.fit(X_fs[train_dates_mask][::100], y_binary_binarized[train_dates_mask][::100])
                        train_importance = rf.feature_importances_[0]
                        
                        if X_fs[val_dates_mask].shape[0] > 0:
                            # Ensure equal dimensions after masking
                            min_train = train_dates_mask.sum()
                            min_val = val_dates_mask.sum()
                            min_size = min(min_train, min_val)
                            
                            if min_size > 0:
                                # Get indices for equal-sized samples
                                train_idx = np.where(train_dates_mask)[0][:min_size]
                                val_idx = np.where(val_dates_mask)[0][:min_size]
                                
                                # Calculate correlation with equal-sized arrays
                                if X_fs[train_idx, i].shape[0] > 0 and X_fs[val_idx, i].shape[0] > 0:
                                    corr = np.corrcoef(X_fs[train_idx, i], X_fs[val_idx, i])[0, 1]
                                    stability_results[fname] = {'importance': train_importance, 'stability': abs(corr) if not np.isnan(corr) else 0.0}
                                else:
                                    stability_results[fname] = {'importance': train_importance, 'stability': 0.0}
                            else:
                                stability_results[fname] = {'importance': train_importance, 'stability': 0.0}
                    
                    sorted_stability = sorted(stability_results.items(), key=lambda x: x[1]['stability'], reverse=True)
                    self.logger.log(f"   Feature Stability (top 5):", 'info')
                    for fname, vals in sorted_stability[:5]:
                        self.logger.log(f"      {fname}: stability={vals['stability']:.3f}, importance={vals['importance']:.3f}", 'info')
                else:
                    self.logger.log("   [warning] Insufficient data for feature stability analysis", 'warning')
            except Exception as e:
                self.logger.log(f"   [error] Feature stability analysis failed: {e}", 'warning')
        
        # Item 2: Inference Latency
        if self.config.get('TRACK_INFERENCE_LATENCY', False):
            self.logger.log("Measuring Inference Latency...", 'info')
            try:
                sample_size = min(self.config.get('INFERENCE_LATENCY_SAMPLE_SIZE', 10000), len(X_val))
                sample_idx = np.random.choice(len(X_val), sample_size, replace=False)
                
                if trained_models and trained_models[0] is not None:
                    start_time = time.time()
                    _ = trained_models[0].predict(X_val[sample_idx], verbose=0)
                    elapsed = (time.time() - start_time) / sample_size * 1000
                    self.logger.log(f"   Inference latency: {elapsed:.3f} ms/sample ({sample_size} samples)", 'info')
                else:
                    self.logger.log("   [warning] No trained model available for latency measurement", 'warning')
            except Exception as e:
                self.logger.log(f"   [error] Latency measurement failed: {e}", 'warning')
        
        # Save original val_mask before SWV block (may overwrite it)
        orig_val_mask = val_mask.copy()
        
        # Item 4: Sliding Window Validation
        if self.config.get('SLIDING_WINDOW_VALIDATION', False):
            self.logger.log("Running Sliding Window Validation...", 'info')
            try:
                n_dates = len(unique_dates)
                split_idx = int(n_dates * 0.7)
                
                train_dates = unique_dates[:split_idx]
                val_dates = unique_dates[split_idx:]
                
                train_mask = np.isin(dates, train_dates)
                val_mask = np.isin(dates, val_dates)
                
                # PRIORITY 2 FIX: Validate dimension alignment for sliding window (May 7, 2026)
                # Ensure arrays have matching dimensions
                min_samples = min(X.shape[0], y_binary.shape[0], len(dates))
                if X.shape[0] != y_binary.shape[0] or X.shape[0] != len(dates):
                    self.logger.log(f"   [warning] Dimension mismatch: X={X.shape[0]}, y_binary={y_binary.shape[0]}, dates={len(dates)}. Aligning.", 'warning')
                    X = X[:min_samples]
                    y_binary = y_binary[:min_samples]
                    dates = dates[:min_samples]
                    train_mask = train_mask[:min_samples]
                    val_mask = val_mask[:min_samples]
                
                # Binarize y_binary for classifier metrics (line 1212+ expects integer 0/1)
                optimal_threshold = self.config.get('FIRST_THRESHOLD', 2.0)
                y_binary_binarized = (y_binary >= optimal_threshold).astype(int)
                
                X_train_sw = X[train_mask]
                y_train_sw = y_binary_binarized[train_mask]
                X_val_sw = X[val_mask]
                y_val_sw = y_binary_binarized[val_mask]
                
                if X_train_sw.shape[0] > 100 and X_val_sw.shape[0] > 100 and trained_models and trained_models[0] is not None:
                    sw_model = trained_models[0]
                    sw_val_pred = sw_model.predict(X_val_sw, verbose=0).flatten()
                    
                    pred_threshold = self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)
                    sw_val_binary = (sw_val_pred >= pred_threshold).astype(int)
                    
                    sw_precision = self.evaluator.calculate_precision(y_val_sw, sw_val_binary)
                    sw_recall = self.evaluator.calculate_recall(y_val_sw, sw_val_binary)
                    sw_f1 = self.evaluator.calculate_f1(y_val_sw, sw_val_binary)
                    
                    self.logger.log(f"   sliding window: train_samples={X_train_sw.shape[0]}, validation_samples={X_val_sw.shape[0]}", 'info')
                    self.logger.log(f"   Sliding Window validation: precision={sw_precision:.4f} recall={sw_recall:.4f} f1={sw_f1:.4f}", 'info')
                    
                    if len(val_dates) >= 3:
                        seg_size = len(val_dates) // 3
                        segment_metrics = {}
                        
                        for seg_name, seg_range in [('early', slice(0, seg_size)), ('mid', slice(seg_size, seg_size*2)), ('late', slice(seg_size*2, None))]:
                            seg_dates = val_dates[seg_range]
                            seg_mask = np.isin(dates, seg_dates)
                            if seg_mask.sum() > 0:
                                seg_pred = sw_model.predict(X[seg_mask], verbose=0).flatten()
                                seg_binary = (seg_pred >= pred_threshold).astype(int)
                                # PRIORITY 4 FIX: Use binarized y_binary not continuous y_binary
                                seg_y_binary = y_binary_binarized[seg_mask]
                                segment_metrics[seg_name] = {
                                    'precision': self.evaluator.calculate_precision(seg_y_binary, seg_binary),
                                    'recall': self.evaluator.calculate_recall(seg_y_binary, seg_binary),
                                    'f1': self.evaluator.calculate_f1(seg_y_binary, seg_binary)
                                }
                        
                        if len(segment_metrics) >= 2:
                            drift = calculate_temporal_drift(segment_metrics)
                            self.logger.log(f"   temporal drift: {drift.get('interpretation', 'N/A')}", 'info')
                            self.logger.log(f"   drift: stability_score={drift.get('stability_score', 0):.3f}", 'info')
                else:
                    self.logger.log("   [warning] Insufficient data for sliding window validation", 'warning')
            except Exception as e:
                self.logger.log(f"   [error] Sliding window validation failed: {e}", 'warning')
        
        # Permutation Importance on all trained models
        if self.config.get('PERMUTATION_IMPORTANCE', False):
            self.logger.log("Running Permutation Importance on all trained models...", 'info')
            
            # Use validation data
            pred_threshold = self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)
            
            for i, model in enumerate(trained_models):
                if model is None:
                    continue
                
                arch_name = arch_names[i] if i < len(arch_names) else f"Model_{i}"
                
                try:
                    X_val_arch = X_val[:, arch_kept_indices[i]] if i < len(arch_kept_indices) else X_val
                    if arch_name in ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']:
                        scaler = arch_scalers.get(arch_name)
                        if scaler is not None:
                            X_val_arch = scaler.transform(X_val_arch)
                    perm_importance = calculate_permutation_importance(
                        model, X_val_arch, y_val_binarized,
                        scoring_metric='precision', n_iterations=5,
                        pred_threshold=pred_threshold
                    )
                    
                    if perm_importance:
                        sorted_importance = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
                        self.logger.log(f"   {arch_name} Permutation Importance (top 10):", 'info')
                        
                        for feat_idx, score in sorted_importance[:10]:
                            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
                            self.logger.log(f"      {feat_name}: {score:.4f}", 'info')
                    else:
                        self.logger.log(f"   {arch_name}: Calculation returned empty", 'warning')
                        
                except Exception as e:
                    self.logger.log(f"   [warning] Permutation importance failed for {arch_name}: {e}", 'warning')

                try:
                    X_val_arch = X_val[:, arch_kept_indices[i]] if i < len(arch_kept_indices) else X_val
                    if arch_name in ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']:
                        scaler = arch_scalers.get(arch_name)
                        if scaler is not None:
                            X_val_arch = scaler.transform(X_val_arch)
                    val_pred = model.predict(X_val_arch, verbose=0).flatten()
                    positive_mask = y_val_binarized == 1
                    negative_mask = y_val_binarized == 0
                    positive_preds = val_pred[positive_mask] if positive_mask.any() else np.array([])
                    negative_preds = val_pred[negative_mask] if negative_mask.any() else np.array([])

                    entropy = calculate_prediction_entropy(val_pred)
                    logit_comp = calculate_logit_compression(val_pred)
                    mi = calculate_mutual_information(val_pred, y_val_binarized, threshold=pred_threshold)

                    ks_result = {'ks_stat': 0.0, 'interpretation': 'N/A'}
                    bh_dist = 0.0
                    if len(positive_preds) > 1 and len(negative_preds) > 1:
                        ks_result = calculate_ks_test(positive_preds, negative_preds)
                        bh_dist = calculate_bhattacharyya_distance(positive_preds, negative_preds)

                    self.logger.log(f"   {arch_name} [advanced] entropy={entropy:.4f}, LogitComp={logit_comp:.2f}, MI={mi:.4f}", 'info')
                    self.logger.log(f"   {arch_name} [advanced] KS-stat={ks_result.get('ks_stat', 0):.4f} ({ks_result.get('interpretation', 'N/A')}), bhattacharyya={bh_dist:.4f}", 'info')

                except Exception as e:
                    self.logger.log(f"   [warning] Advanced diagnostics failed for {arch_name}: {e}", 'warning')

        # Use the optimal threshold from architecture training for ensemble evaluation
        if optimal_thresholds:
            best_ensemble_threshold = max(final_thresholds) if final_thresholds else max(optimal_thresholds)
        else:
            best_ensemble_threshold = self.config.get('FIRST_THRESHOLD', 24.9)
        
        # Initialize ensemble variables with defaults
        ensemble_min_precision = self.config.get('ENSEMBLE_MIN_PRECISION', 0.40)
        ensemble_weighting = self.config.get('ENSEMBLE_WEIGHTING', 'precision_weighted')
        filtered_models = []
        filtered_precisions = []
        filtered_arch_names = []
        filtered_kept_indices = []
        
        if not trained_models:
            self.logger.log("No models trained successfully, using fallback", 'warning')
            # Create dummy ensemble
            def dummy_ensemble(X):
                return np.full(len(X), y_binary.mean() if 'y_binary' in dir() else 0.5)
            ensemble = dummy_ensemble
            ensemble_precision = 0.0
        else:
            # ============================================================================
            # SECTION 5: Ensemble Creation
            # ============================================================================
            # - Filter architectures by ENSEMBLE_MIN_PRECISION (>0.40)
            # - Use precision-weighted averaging (higher precision = higher weight)
            # - Evaluate ensemble at best_ensemble_threshold
            # - Note: Ensemble is for reference; Phase 5 uses individual models
            
            # Filter architectures by minimum precision
            self.logger.log(f"Filtering architectures by min precision > {ensemble_min_precision:.2f}:", 'info')
            for i, (arch_name, model) in enumerate(zip(arch_names, trained_models)):
                if model is not None and i < len(best_val_precision_list):
                    val_prec = best_val_precision_list[i]
                    if val_prec > ensemble_min_precision:
                        filtered_models.append(model)
                        filtered_precisions.append(val_prec)
                        filtered_arch_names.append(arch_name)
                        filtered_kept_indices.append(arch_kept_indices[i] if i < len(arch_kept_indices) else None)
                        self.logger.log(f"  {arch_name}: precision={val_prec:.4f} ✓", 'info')
                    else:
                        self.logger.log(f"  {arch_name}: precision={val_prec:.4f} ✗ (below threshold)", 'info')
            
            # Check for fallback
            if not filtered_models:
                self.logger.log(f"No architectures met min precision > {ensemble_min_precision:.2f}, using fallback", 'warning')
                fallback_arch = self.config.get('FALLBACK_ARCHITECTURE', 'RNN')
                for i, arch_name in enumerate(arch_names):
                    if arch_name == fallback_arch and trained_models[i] is not None:
                        filtered_models.append(trained_models[i])
                        filtered_precisions.append(best_val_precision_list[i] if i < len(best_val_precision_list) else 0.0)
                        filtered_arch_names.append(arch_name)
                        filtered_kept_indices.append(arch_kept_indices[i] if i < len(arch_kept_indices) else None)
                        self.logger.log(f"  Fallback {arch_name}: precision={filtered_precisions[-1]:.4f}", 'info')
                        break
            
            # Create precision-weighted ensemble
            if filtered_models:
                val_preds_matrix = np.array(val_predictions)
                # Use precision weights if weighting is enabled
                weights_to_use: List[float] = filtered_precisions if ensemble_weighting == 'precision_weighted' else None
                ensemble = create_precision_ensemble(
                    filtered_models, 
                    val_preds_matrix, 
                    "main_ensemble",
                    precision_weights=weights_to_use,
                    features_per_model=filtered_kept_indices,
                    logger=self.logger.log
                )
                self.logger.log(f"ensemble: {len(filtered_models)} architectures ({', '.join(filtered_arch_names)})", 'info')
                if ensemble_weighting == 'precision_weighted':
                    self.logger.log(f"  Using precision-weighted averaging", 'info')
            else:
                ensemble = create_precision_ensemble(trained_models, np.array(val_predictions), "main_ensemble")
            
            # Evaluate ensemble on validation set using optimal threshold
            # Use the optimal threshold from threshold search for both labels and predictions
            opt_threshold = best_ensemble_threshold
            y_val_binary = (y_val_continuous >= opt_threshold).astype(int)
            
            # Use pre-computed val_predictions (correct feature pruning already applied)
            if filtered_models and filtered_precisions:
                pred_indices = [i for i, name in enumerate(arch_names) if name in filtered_arch_names]
                ensemble_preds = np.array([val_predictions[i] for i in pred_indices])
                w = np.array(filtered_precisions)
                w = w / w.sum()
                predictions = np.average(ensemble_preds, axis=0, weights=w)
            else:
                predictions = np.mean(val_predictions, axis=0)
            if self.config.get('LOG_VERBOSITY', 0) >= 2:
                self.logger.log(f"   [diagnostic-ensemble] " + format_diagnostic_string(predictions, ""), 'info')
            binary_preds = (predictions >= self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT)).astype(int)
            ensemble_precision = self.evaluator.calculate_precision(y_val_binary, binary_preds)
            
            self.logger.log(f"Ensemble precision (validation): {ensemble_precision:.4f} (LABEL_THRESHOLD={opt_threshold:.1f}, prediction_binary_split={self.config.get('PREDICTION_THRESHOLD', PREDICTION_THRESHOLD_DEFAULT):.2f})", 'info')
        
        # Update context with per-architecture data for Phase 5
        # Use first model's threshold as fallback for ensemble threshold
        ensemble_threshold = final_thresholds[0] if final_thresholds else (optimal_thresholds[0] if optimal_thresholds else 0.5)
        
        # Determine ensemble callable (fallback if not created)
        ensemble_callable = None
        if 'ensemble' in dir() and callable(ensemble):
            ensemble_callable = ensemble
        else:
            # Create a simple averaging ensemble as fallback
            def fallback_ensemble(X):
                preds = []
                for i, m in enumerate(trained_models):
                    if m is not None:
                        X_i = X[:, arch_kept_indices[i]] if i < len(arch_kept_indices) and arch_kept_indices[i] is not None else X
                        p = m.predict(X_i, verbose=0).flatten()
                        preds.append(p)
                if preds:
                    return np.mean(preds, axis=0)
                return np.zeros(len(X))
            ensemble_callable = fallback_ensemble
        
        context.update({
            'models': trained_models,
            'arch_names': arch_names,
            'optimal_thresholds': final_thresholds if final_thresholds else optimal_thresholds,
            'best_hyperparams_list': best_hyperparams_list,
            'best_val_precision_list': best_val_precision_list,
            'final_ensemble': ensemble_callable,
            'ensemble_precision': ensemble_precision,
            'optimal_label_threshold': ensemble_threshold,
            'phase4_complete': True,
            'val_predictions': val_predictions,
            'val_dates': dates[orig_val_mask],
            'val_y_raw': y_val_continuous,
            # Ensemble configuration for Phase 5
            'ensemble_min_precision': ensemble_min_precision,
            'ensemble_participants': filtered_arch_names if filtered_arch_names else arch_names,
            'ensemble_precision_weights': filtered_precisions if filtered_precisions else best_val_precision_list,
        })
        
        # =========================================================================
        # SUMMARY TABLES
        # =========================================================================
        
        # 1. Architecture Ranking Table (sorted by Val Precision)
        if arch_final_metrics:
            # Sort by precision descending
            sorted_metrics = sorted(arch_final_metrics, key=lambda x: x['P'], reverse=True)
            
            self.logger.log("ARCHITECTURE RANKING (by Validation precision)", 'info')
            self.logger.log(
                f"{'Rank':>4} | {'Arch':<7} | {'Thresh':>6} | {'validation_precision':>8} | {'validation_recall':>8} | {'validation_auc':>8} | "
                f"{'FN':>5} | {'TN':>6} | {'TP':>4} | {'FP':>4}",
                'info'
            )
            
            for rank, m in enumerate(sorted_metrics, 1):
                self.logger.log(
                    f"{rank:>4} | {m['arch']:<7} | {m['optimal_label_threshold']:>6.1f} | {m['P']:>8.4f} | "
                    f"{m['R']:>8.4f} | {m['AUC']:>8.4f} | {m['FN']:>5} | {m['TN']:>6} | "
                    f"{m['TP']:>4} | {m['FP']:>4}",
                    'info'
                )
            
            # 2. HPO Impact Summary
            if pre_hpo_precisions and post_hpo_precisions and len(pre_hpo_precisions) == len(post_hpo_precisions):
                self.logger.log("HPO IMPACT SUMMARY", 'info')
                self.logger.log(
                    f"{'Architecture':<12} | {'Pre-HPO P':>10} | {'Post-HPO P':>10} | {'Improved?':<9}",
                    'info'
                )
                
                for i, arch in enumerate(arch_names):
                    pre_p = pre_hpo_precisions[i] if i < len(pre_hpo_precisions) else 0.0
                    post_p = post_hpo_precisions[i] if i < len(post_hpo_precisions) else 0.0
                    improved = "Yes" if post_p > pre_p else "No"
                    self.logger.log(
                        f"{arch:<12} | {pre_p:>10.4f} | {post_p:>10.4f} | {improved:<9}",
                        'info'
                    )
                
            # 3. Training Time Summary
            if arch_training_times:
                self.logger.log("TRAINING TIME SUMMARY", 'info')
                for i, arch in enumerate(arch_names):
                    self.logger.log(f"   {arch}: {arch_training_times[i]:.1f}s", 'info')
                total_time = sum(arch_training_times)
                self.logger.log(f"   total: {total_time:.1f}s", 'info')
            
            # ============================================================================
            # SECTION 6: Model Persistence
            # ============================================================================
            # Save trained models to ./saved_models/ for later predictions:
            # - {ARCH}_model.keras: Trained Keras model (RNN_model.keras, LSTM_model.keras)
            # - temporal_weights.json: Weights used for temporal feature scaling
            # - feature_names.json: List of feature column names
            # - split_date.txt: Date used for train/val split
            # - all_dates.json: All unique dates for temporal feature extraction
            #
            # These files are loaded by legacy files/predict.py for inference on new data
            
            # Save trained models and preprocessing parameters
            save_models = self.config.get('SAVE_TRAINED_MODELS', True)
            models_path = self.config.get('MODELS_PATH', './saved_models')
            
            if save_models and trained_models:
                import os
                import json
                import shutil
                import joblib
                
                # Clean stale models from previous runs to avoid metadata contamination (Bug 3 fix)
                if os.path.exists(models_path):
                    shutil.rmtree(models_path)
                
                # Create fresh models directory
                os.makedirs(models_path, exist_ok=True)
                
                # Save each trained model (only those meeting precision threshold — C3 fix)
                for i, (model, arch_name) in enumerate(zip(trained_models, arch_names)):
                    if model is not None:
                        # Skip low-precision models — they'd degrade Phase 5 consensus
                        val_prec = best_val_precision_list[i] if i < len(best_val_precision_list) else 0.0
                        if val_prec < ensemble_min_precision:
                            self.logger.log(f"   Skipping {arch_name} (precision {val_prec:.4f} < {ensemble_min_precision})", 'info')
                            continue
                        model_path = os.path.join(models_path, f'{arch_name}_model.keras')
                        model.save(model_path)
                        self.logger.log(f"   Saved {arch_name} model to {model_path}", 'info')
                    # Save scaler for NN architectures (Bug 2 fix: persist for Phase 5 inference)
                    if arch_name in arch_scalers:
                        scaler_path = os.path.join(models_path, f'{arch_name}_scaler.joblib')
                        joblib.dump(arch_scalers[arch_name], scaler_path)
                        self.logger.log(f"   Saved {arch_name} scaler to {scaler_path}", 'info')
                
                # Save temporal weights (used for preprocessing)
                if 'temporal_weights' in context:
                    temporal_weights_path = os.path.join(models_path, 'temporal_weights.json')
                    # Convert numpy array to serializable format
                    tw = context['temporal_weights']
                    if hasattr(tw, 'tolist'):
                        tw = tw.tolist()
                    with open(temporal_weights_path, 'w') as f:
                        json.dump(tw, f)
                    self.logger.log(f"   Saved temporal weights to {temporal_weights_path}", 'info')
                
                # Save feature names
                if 'feature_names' in context:
                    feature_names_path = os.path.join(models_path, 'feature_names.json')
                    with open(feature_names_path, 'w') as f:
                        json.dump(context['feature_names'], f)
                    self.logger.log(f"   Saved feature names to {feature_names_path}", 'info')
                
                # Save split date for preprocessing
                unique_dates = np.unique(context.get('dates', []))
                if len(unique_dates) > 0:
                    split_date = str(unique_dates[-17])  # First date of validation set (excludes top 2 newest)
                    split_date_path = os.path.join(models_path, 'split_date.txt')
                    with open(split_date_path, 'w') as f:
                        f.write(split_date)
                    self.logger.log(f"   Saved split date to {split_date_path}", 'info')
                
                # Save all validation dates (for temporal feature extraction on new data)
                if len(unique_dates) > 0:
                    all_dates_path = os.path.join(models_path, 'all_dates.json')
                    all_dates = {str(d): float(i) for i, d in enumerate(unique_dates)}
                    with open(all_dates_path, 'w') as f:
                        json.dump(all_dates, f)
                    self.logger.log(f"   Saved all training dates to {all_dates_path}", 'info')
                
                # Save metadata for each architecture (optimal_threshold, best_hyperparams)
                # This allows Phase 5 to load models and know which threshold/hyperparams to use
                for i, arch_name in enumerate(arch_names):
                    if i < len(optimal_thresholds) and i < len(best_hyperparams_list) and i < len(final_thresholds):
                        # Skip metadata for low-precision models (matching model save filter — C3 fix)
                        val_prec = best_val_precision_list[i] if i < len(best_val_precision_list) else 0.0
                        if val_prec < ensemble_min_precision:
                            continue
                        metadata = {
                            'optimal_threshold': float(final_thresholds[i]),
                            'best_hyperparams': best_hyperparams_list[i] if best_hyperparams_list[i] else {},
                            'best_val_precision': float(best_val_precision_list[i]) if i < len(best_val_precision_list) else 0.0,
                            'kept_feature_indices': list(arch_kept_indices[i]) if i < len(arch_kept_indices) else []
                        }
                        metadata_path = os.path.join(models_path, f'{arch_name}_metadata.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)
                        self.logger.log(f"   Saved {arch_name} metadata to {metadata_path}", 'info')
                
                self.logger.log(f"   All models and parameters saved to {models_path}", 'info')
        
        # Add architecture metrics to context for metrics review
        context['arch_final_metrics'] = arch_final_metrics
        
        return context
    
    def _validate_input(self, context: Dict):
        """
        Validate Phase 4 input requirements
        
        Args:
            context: Input context
            
        Raises:
            ValueError: If validation fails
        """
        if not context.get('phase1_complete'):
            raise ValueError("Phase 1 must complete before Phase 4")
        if not context.get('phase3_complete'):
            raise ValueError("Phase 3 must complete before Phase 4")
        
        required = ['X', 'y']
        for key in required:
            if key not in context:
                raise ValueError(f"Phase 4 missing required input: {key}")


def validate_phase4_input(context: Dict) -> bool:
    """
    Ensure Phase 4 has required inputs
    
    Args:
        context: Input context
        
    Returns:
        True if valid
    """
    assert context.get('phase1_complete') == True, "Phase 1 must be complete"
    assert context.get('phase3_complete') == True, "Phase 3 must be complete"
    required = ['X', 'y', 'temporal_weights']
    for key in required:
        assert key in context, f"Phase 4 missing required input: {key}"
    return True


def validate_phase4_output(context: Dict) -> bool:
    """
    Validate Phase 4 output
    
    Args:
        context: Output context from Phase 4
        
    Returns:
        True if valid
    """
    required = ['models', 'final_ensemble', 'ensemble_precision', 'optimal_label_threshold', 'phase4_complete']
    for key in required:
        assert key in context, f"Phase 4 missing output: {key}"
    
    # Validate ensemble is callable
    assert callable(context['final_ensemble']), "final_ensemble must be callable"
    
    # Test ensemble produces valid predictions
    X_sample = context['X'][:5]
    test_preds = context['final_ensemble'](X_sample)
    assert len(test_preds) == 5, "Ensemble prediction length mismatch"
    
    # Validate precision is in valid range
    assert 0 <= context['ensemble_precision'] <= 1, "Precision out of range"
    # Note: optimal_label_threshold is a label threshold (e.g., 19.3), not a prediction threshold (0-1)
    # So we don't validate its range here
    assert context['phase4_complete'] == True
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing Phase4_NeuralEnsemble...")
    
    # Create test config
    config = {
        'latent_dim': 32,
        'units': 64,
        'dropout': 0.1,
        'cnn_filters': 64,
        'lstm_units': 32,
        'heads': 4,
        'dim': 64,
        'MIN_ENSEMBLE_SIZE': 5,
        'LOG_VERBOSITY': 0
    }
    
    # Create mock context from Phase 3
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    context = {
        'X': np.random.randn(n_samples, n_features).astype(np.float32),
        'y': np.random.randint(0, 2, n_samples),
        'dates': np.random.randint(20220101, 20230101, n_samples),
        'temporal_weights': np.ones(n_samples),
        'temporal_features': {'weights': np.ones(n_samples)},
        'phase1_complete': True,
        'phase3_complete': True
    }
    
    # Run Phase 4
    phase4 = Phase4_NeuralEnsemble(config)
    result = phase4.execute(context.copy())
    
    print(f"[pass] Phase 4 executed successfully")
    print(f"   Models trained: {len(result['models'])}")
    print(f"   Ensemble precision: {result['ensemble_precision']:.4f}")
    
    # Validate
    validate_phase4_output(result)
    
    print("\n[pass] Phase4_NeuralEnsemble tests passed")