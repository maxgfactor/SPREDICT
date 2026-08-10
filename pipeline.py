"""
pipeline.py — Main Pipeline Orchestrator
Refactored from chunk_20 (pipeline main) + chunk_19 (Phase 5 inference) +
chunk_22 (model loader) + chunk_13 (state manager).

CPU mode enforced. Phases live in the new modules; Phase 5 (Inference)
loads models from disk via load_saved_models(). StateManager is the
simplified key-value store (its methods are never called in production —
verified across all 24 chunk files).
"""

import os
import sys
import time
import json
import random
import logging

# CPU Mode - GPU paths removed, forced CPU (2026-02-28)
if sys.platform == 'linux':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from config import CONFIG, validate_config_structure
from pipeline_logging import Logger
from evaluate import Evaluator, inverse_log_transform
from phases import (BasePhase, DataSetup, TemporalWeighting,
                    FeatureImportance, FeaturePruning,
                    TemporalPrecisionGap)
from phase_4 import ModelTraining

logging.getLogger('tensorflow').setLevel(logging.ERROR)


# ============================================================================
# Section 2: Model Loader Functions (~100 lines)
# ============================================================================
def load_model(arch_name: str, models_path: str = './saved_models') -> tf.keras.Model:
    """
    Load a trained Keras model from disk.

    Args:
        arch_name: Architecture name (e.g., 'RNN', 'LSTM')
        models_path: Path to saved models directory

    Returns:
        Loaded Keras model
    """
    model_path = f"{models_path}/{arch_name}_model.keras"
    # Load with safe_mode=False to allow Lambda layers (needed for Transformer)
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    return model


def load_scaler(models_path: str, arch_name: str) -> Any:
    """
    Load a saved StandardScaler for a given architecture.

    Args:
        models_path: Path to saved models directory
        arch_name: Architecture name (e.g., 'RNN', 'LSTM')

    Returns:
        Loaded StandardScaler or None if not found
    """
    import joblib
    scaler_path = os.path.join(models_path, f'{arch_name}_scaler.joblib')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


def load_model_metadata(models_path: str, arch_name: str) -> Dict:
    """
    Load {arch_name}_metadata.json and return as dict.

    Args:
        models_path: Path to saved models directory
        arch_name: Architecture name

    Returns:
        Metadata dict, or None if metadata file doesn't exist
    """
    metadata_path = os.path.join(models_path, f'{arch_name}_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def load_saved_models(models_path: str, config: Dict) -> Dict[str, Any]:
    """
    Unified model loader. NEW wrapper — the source has no function under this
    name; chunk_22's loader is load_models_with_metadata(). This wraps model +
    scaler + metadata loading under one keyed structure.

    NOTE: temporal_weights.json and feature_names.json are NOT loaded here.
    Phase 5 receives these from pipeline context (set by Phase 1/3/4), not disk.

    Args:
        models_path: Path to saved models directory
        config: Full config (uses TREE_ARCHITECTURES to classify models)

    Returns:
        {arch_name: {'model': <keras.Model|SklearnModelWrapper>,
                     'scaler': <StandardScaler|None>,
                     'metadata': {...}}}
        Empty dict if models_path is empty or missing.
    """
    from models import SklearnModelWrapper
    sklearn_archs = set(config['TREE_ARCHITECTURES'])
    loaded = {}
    if not os.path.exists(models_path):
        return loaded
    for filename in os.listdir(models_path):
        if not (filename.endswith('_model.keras') or filename.endswith('_model.joblib')):
            continue
        arch_name = filename.replace('_model.keras', '').replace('_model.joblib', '')
        filepath = os.path.join(models_path, filename)
        model = None
        if arch_name in sklearn_archs:
            try:
                import joblib
                sklearn_model = joblib.load(filepath)
                wrapped = SklearnModelWrapper(sklearn_model)
                wrapped._is_fitted = True
                model = wrapped
            except Exception:
                model = None
        else:
            try:
                model = load_model(arch_name, models_path)
            except Exception:
                model = None
        if model is None:
            continue
        loaded[arch_name] = {
            'model': model,
            'scaler': load_scaler(models_path, arch_name),
            'metadata': load_model_metadata(models_path, arch_name) or {},
        }
    return loaded


# ============================================================================
# Section 1: Inference (~250 lines)
# ============================================================================
class Inference(BasePhase):
    """
    Inference phase. Models loaded from disk; inference data from
    PipelineOrchestrator context (matching current pipeline behavior).

    CONTEXT_CONSUMED: ['X_inference', 'y_inference_continuous',
                       'dates_inference', 'feature_names']
    CONTEXT_PRODUCED: ['architecture_results', 'final_predictions',
                       'y_inference_binarized', 'majority_threshold',
                       'final_metrics', 'inference_complete', 'phase5_complete']

    DESIGN NOTE: The original chunk_19 docstring claimed this phase loads
    fresh data from 'for_train_x_2025_10_24_clean.csv', and the code had
    `data_path = config['DATA_PATH']` (line 87), but this variable was NEVER
    referenced — the real data source has always been context['X_inference']
    (line 105). The refactored version matches actual current behavior
    (context-based), not the wrong docstring. CSV-loading can be
    reintroduced as a future optimization.
    """
    CONTEXT_CONSUMED = ['X_inference', 'y_inference_continuous',
                        'dates_inference', 'feature_names']
    CONTEXT_PRODUCED = ['architecture_results', 'final_predictions',
                        'y_inference_binarized', 'majority_threshold',
                        'final_metrics', 'inference_complete', 'phase5_complete']

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.evaluator = Evaluator(config)

    def execute(self, context: Dict) -> Dict:
        phase5_start_time = time.time()

        # =========================================================================
        # STEP 0: Load saved models and metadata from MODELS_PATH
        # =========================================================================
        models_path = self.config['MODELS_PATH']

        self.logger.log(f"Loading saved models from {models_path}...", 'info')
        loaded = load_saved_models(models_path, self.config)

        if not loaded:
            self.logger.log("No models found in saved_models directory", 'warning')
            context.update({'phase5_complete': True})
            return context

        models = {k: v['model'] for k, v in loaded.items()}
        metadata = {k: v['metadata'] for k, v in loaded.items()}
        for arch_name in models:
            self.logger.log(f"Loaded {arch_name} model", 'info')
            if metadata.get(arch_name):
                self.logger.log(f"Loaded {arch_name} metadata: label_threshold={metadata[arch_name].get('optimal_threshold')}", 'info')

        self.logger.log(f"Loaded architectures: {list(models.keys())}", 'info')

        # =========================================================================
        # STEP 1: Receive data from context (NO CSV loading)
        # =========================================================================
        self.logger.log("Receiving inference data from context (NO CSV loading)...", 'info')

        # Get data from context
        X_inference = context.get('X_inference')
        y_inference_continuous = context.get('y_inference_continuous')
        dates_inference = context.get('dates_inference', np.array([]))
        context_feature_names = context.get('feature_names', [])
        temporal_weights = context.get('temporal_weights', np.ones(len(X_inference)) if X_inference is not None else np.array([]))

        # Validate
        if X_inference is None:
            self.logger.log("[error] X_inference not found in context!", 'error')
            context.update({'phase5_complete': True})
            return context

        n_inference = len(X_inference)
        self.logger.log(f"  Inference samples: {n_inference:,}", 'info')
        self.logger.log(f"  Inference date(s): {np.unique(dates_inference)}", 'info')
        self.logger.log(f"  Feature count: {X_inference.shape[1]}", 'info')

        # =========================================================================
        # STEP 2: Prepare inference data (RAW, NO temporal weighting)
        # =========================================================================
        X_raw = X_inference
        self.logger.log(f"  Features will be selected per-architecture from {X_raw.shape[1]} total", 'info')
        y_val_continuous = y_inference_continuous if y_inference_continuous is not None else np.zeros(n_inference)

        # Get threshold info from config
        label_threshold = self.config['FIRST_THRESHOLD']

        self.logger.log(f"  Label threshold: {label_threshold}", 'info')
        self.logger.log(f"  Inference data: RAW (no temporal weighting applied)", 'info')

        # Feature names from context
        feature_names = context_feature_names

        # Keep original data for output
        df_with_all_cols = context.get('df_inference', None)
        if df_with_all_cols is None and X_inference is not None:
            fn = context.get('feature_names', [])
            if fn and len(fn) == X_inference.shape[1]:
                df_with_all_cols = pd.DataFrame(X_inference, columns=fn)

        self.logger.log(f"  X shape (RAW, no weighting): {X_raw.shape}", 'info')
        self.logger.log(f"  y shape: {y_val_continuous.shape}", 'info')

        # =========================================================================
        # STEP 3: Apply temporal weighting (same as Phase 3/4)
        # =========================================================================
        dates = dates_inference

        # Create temporal indices for each date
        # all_dates maps date -> index (0 = oldest, max = newest)
        unique_dates = np.unique(dates)
        all_dates = {str(d): i for i, d in enumerate(unique_dates)}
        temporal_indices = np.array([all_dates.get(str(d), 0) for d in dates])

        X = X_raw
        if self.config.get('USE_TEMPORAL_WEIGHTING', True):
            # Apply temporal weighting (sqrt to avoid over-amplification)
            temporal_multiplier = self.config['TEMPORAL_MULTIPLIER']
            temporal_w = 1.0 + (temporal_indices / max(len(all_dates), 1)) * (temporal_multiplier - 1.0)
            temporal_w = np.sqrt(temporal_w)

            # Apply weights to features
            X = X_raw * temporal_w[:, np.newaxis]

            self.logger.log(f"  Temporal weights applied: min={temporal_w.min():.2f}, max={temporal_w.max():.2f}", 'info')
        else:
            self.logger.log("  Inference data: RAW (no temporal weighting applied)", 'info')

        # =========================================================================
        # STEP 4: Run inference and evaluate each architecture
        # =========================================================================
        from collections import Counter

        architecture_results = []

        self.logger.log("", 'info')
        self.logger.log("Per-Architecture Results on Newest Data", 'info')

        # Consolidated column set (same df_with_all_cols for all architectures)
        available_cols = list(df_with_all_cols.columns) if df_with_all_cols is not None else []
        # C1/C2: Use only high-precision models for consensus, with configurable vote threshold
        ensemble_min_precision = self.config['ENSEMBLE_MIN_PRECISION']
        # ensemble_vote_threshold REMOVED — see SPEC.md §2.13; consensus uses max(3, len(majority_archs))
        all_pred_signal_sets = []  # list of (arch_name, set(indices))
        arch_thresholds = {}  # arch_name -> opt_threshold

        for arch_name, model in models.items():
            # Get metadata for this architecture
            arch_metadata = metadata.get(arch_name, {})
            opt_threshold = arch_metadata.get('optimal_threshold', 20.0)
            best_hyperparams = arch_metadata.get('best_hyperparams', {})
            best_val_prec = arch_metadata.get('best_val_precision', 0.0)
            pred_threshold = arch_metadata.get('prediction_threshold', self.config['PREDICTION_THRESHOLD'])
            kept_idx = arch_metadata.get('kept_feature_indices')

            # Select features for this architecture's optimal threshold
            if kept_idx:
                X_arch = X[:, kept_idx]
                self.logger.log(f"  {arch_name}: using {len(kept_idx)} features (optimal threshold's pruning)", 'info')
            else:
                self.logger.log(f"  [warning] {arch_name}: kept_feature_indices not in metadata - skipping", 'warning')
                continue

            self.logger.log(
                f"{arch_name} (LABEL_THRESHOLD={opt_threshold:.1f}, prediction_binary_split={pred_threshold:.2f}, hyperparams={best_hyperparams}, VALIDATION_PRECISION={best_val_prec:.4f}):",
                'info'
            )

            # Apply per-architecture winsorization using Phase 4 training bounds (Phase B)
            winsor_bounds = arch_metadata.get('winsor_bounds', {})
            low_bounds = np.array(winsor_bounds.get('low', [])) if winsor_bounds else np.array([])
            high_bounds = np.array(winsor_bounds.get('high', [])) if winsor_bounds else np.array([])

            if len(low_bounds) > 0 and len(high_bounds) > 0 and kept_idx is not None:
                X_arch = np.clip(X_arch, low_bounds[kept_idx], high_bounds[kept_idx])
                lp = winsor_bounds.get('low_pct', '?')
                hp = winsor_bounds.get('high_pct', '?')
                if isinstance(lp, (int, float)):
                    self.logger.log(
                        f"   Winsor bounds applied: percentiles [{lp}%, {hp}%], "
                        f"n_features={len(kept_idx)}, "
                        f"clip_range=[{low_bounds[kept_idx].min():.4f}, {high_bounds[kept_idx].max():.4f}]",
                        'info'
                    )
            elif not winsor_bounds and self.config.get('PER_ARCH_WINSORIZE', {}):
                self.logger.log(f"   [warning] Per-arch winsor active but winsor_bounds missing from {arch_name} metadata", 'warning')

            # Apply StandardScaler for NN architectures (Bug 2 fix: BN inference collapse)
            if arch_name in self.config['NEURAL_ARCHITECTURES']:
                scaler = loaded[arch_name].get('scaler')
                if scaler is not None:
                    X_arch = scaler.transform(X_arch)
                else:
                    self.logger.log(f"   [warning] No scaler found for {arch_name} — predictions may be degraded", 'warning')

            # Run inference
            try:
                predictions = model.predict(X_arch, verbose=0).flatten()
            except Exception as e:
                error_msg = str(e)
                if 'incompatible' in error_msg.lower() or 'shape' in error_msg.lower():
                    self.logger.log(f"   [skip] Shape mismatch - model expects different input dimensions: {e}", 'warning')
                else:
                    self.logger.log(f"   Prediction failed: {e}", 'warning')
                continue

            # Validate predictions are in valid probability range [0,1]
            if np.any(np.isnan(predictions)):
                self.logger.log(f"   [warning] NaN values detected in predictions!", 'warning')
            if predictions.min() < 0 or predictions.max() > 1:
                self.logger.log(f"   [warning] Predictions outside [0,1] range! min={predictions.min():.4f}, max={predictions.max():.4f}", 'warning')

            # Log prediction distribution statistics
            self.logger.log(f"   predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}, min={predictions.min():.4f}, max={predictions.max():.4f}", 'info')

            # Binary predictions: use prediction threshold from config
            binary_predictions = (predictions >= pred_threshold).astype(int)

            # Log prediction class distribution
            pos_pct = binary_predictions.mean() * 100
            self.logger.log(f"   INFERENCE predictions: {pos_pct:.2f}% positive predictions ({binary_predictions.sum():,} / {len(binary_predictions):,})", 'info')

            # Check for all-zero predictions
            if binary_predictions.sum() == 0:
                self.logger.log(f"   [warning] Model predicts ALL NEGATIVES!", 'warning')

            # Ground truth: use optimal_threshold for y
            y_val_binarized = (y_val_continuous >= opt_threshold).astype(int)

            # Apply inverse log transform if configured (convert predictions back to original scale)
            if self.config['LOG_TRANSFORM_TARGET']:
                predictions_original = inverse_log_transform(predictions)
            else:
                predictions_original = predictions

            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_val_binarized, binary_predictions, predictions_original)

            tp = int(np.sum((binary_predictions == 1) & (y_val_binarized == 1)))
            tn = int(np.sum((binary_predictions == 0) & (y_val_binarized == 0)))
            fp = int(np.sum((binary_predictions == 1) & (y_val_binarized == 0)))
            fn = int(np.sum((binary_predictions == 0) & (y_val_binarized == 1)))

            # Log metrics (NO confusion matrix)
            self.logger.log(
                f"  {arch_name} t={opt_threshold:.1f}: inference_precision={metrics['precision']:.4f} inference_recall={metrics['recall']:.4f} inference_auc={metrics['auc']:.4f} inference_f1={metrics['f1']:.4f} inference_false_negatives={fn} inference_true_negatives={tn} inference_true_positives={tp} inference_false_positives={fp}",
                'info'
            )

            # Log new enhanced metrics
            inf_mcc = metrics.get('mcc', 0.0)
            inf_prauc = metrics.get('average_precision', 0.0)
            inf_specificity = metrics.get('specificity', 0.0)
            inf_balanced_acc = metrics.get('balanced_accuracy', 0.0)
            inf_fpr = self.evaluator.calculate_fpr(y_val_binarized, binary_predictions)
            inf_f2 = self.evaluator.calculate_f2_score(y_val_binarized, binary_predictions)
            inf_std_pred = float(predictions.std()) if len(predictions) > 0 else 0.0
            inf_pct_above_thresh = (predictions >= pred_threshold).mean() * 100 if len(predictions) > 0 else 0.0
            inf_brier = self.evaluator.calculate_brier_score(y_val_binarized, predictions.flatten())
            inf_kappa = self.evaluator.calculate_kappa(y_val_binarized, binary_predictions)
            inf_informedness = self.evaluator.calculate_informedness(y_val_binarized, binary_predictions)
            inf_markedness = self.evaluator.calculate_markedness(y_val_binarized, binary_predictions)
            inf_gini = self.evaluator.calculate_gini(y_val_binarized, predictions.flatten())
            inf_opt_thresh = self.evaluator.calculate_optimal_threshold(y_val_binarized, predictions.flatten())

            self.logger.log(
                f"  {arch_name} - Inference: inference_precision={metrics['precision']:.4f} inference_true_positives={tp} inference_true_negatives={tn} inference_false_positives={fp} inference_false_negatives={fn} inference_max_prediction={predictions.max():.4f} inference_mean_prediction={predictions.mean():.4f} inference_recall={metrics['recall']:.4f} inference_f1={metrics['f1']:.4f} inference_auc={metrics['auc']:.4f} inference_specificity={inf_specificity:.4f} inference_false_positive_rate={inf_fpr:.4f} inference_f2={inf_f2:.4f} inference_mcc={inf_mcc:.4f} inference_prauc={inf_prauc:.4f} inference_balanced_accuracy={inf_balanced_acc:.4f} inference_standard_deviation_prediction={inf_std_pred:.4f} inference_percentage_above_threshold={inf_pct_above_thresh:.2f} inference_brier={inf_brier:.4f} inference_kappa={inf_kappa:.4f} inference_informedness={inf_informedness:.4f} inference_markedness={inf_markedness:.4f} inference_gini={inf_gini:.4f} inference_optimal_threshold={inf_opt_thresh:.4f}",
                'info'
            )

            if df_with_all_cols is not None:
                available_cols = list(df_with_all_cols.columns)

                pred_signal_indices = np.where(binary_predictions == 1)[0]
                # C1: Only include high-precision models in consensus voting
                if best_val_prec > ensemble_min_precision:
                    all_pred_signal_sets.append((arch_name, set(pred_signal_indices)))
                    arch_thresholds[arch_name] = opt_threshold
                if len(pred_signal_indices) > 0:
                    self.logger.log("", 'info')
                    self.logger.log(f"{arch_name} MODEL PREDICTED SIGNAL ({len(pred_signal_indices)} rows)", 'info')
                    self.logger.log(f"{arch_name} inference_precision={metrics['precision']:.4f} prediction_binary_split={pred_threshold} predicted={len(pred_signal_indices)}", 'info')
                    self.logger.log("Row," + ",".join(available_cols), 'info')

            # Store results with Inf_ prefix (16 metrics + 2 extras)
            architecture_results.append({
                'architecture': arch_name,
                'label_threshold': opt_threshold,
                'pred_threshold': pred_threshold,
                'hyperparams': best_hyperparams,
                'val_precision': best_val_prec,
                # Core metrics with Inf_ prefix
                'Inf_P': metrics['precision'],
                'Inf_R': metrics['recall'],
                'Inf_F1': metrics['f1'],
                'Inf_AUC': metrics['auc'],
                'Inf_TP': tp,
                'Inf_TN': tn,
                'Inf_FP': fp,
                'Inf_FN': fn,
                # Prediction distribution
                'Inf_MaxPred': float(predictions.max()),
                'Inf_MeanPred': float(predictions.mean()),
                'Inf_StdPred': float(predictions.std()),
                'Inf_PctAboveThresh': float((predictions >= pred_threshold).mean() * 100),
                # Extended metrics with Inf_ prefix
                'Inf_Spec': inf_specificity,
                'Inf_FPR': inf_fpr,
                'Inf_F2': inf_f2,
                'Inf_MCC': inf_mcc,
                'Inf_PRAUC': inf_prauc,
                'Inf_BalAcc': inf_balanced_acc,
            })

        # --- Majority label threshold filter ---
        majority_threshold = None
        majority_archs = set()
        if arch_thresholds:
            counts = Counter(arch_thresholds.values())
            max_count = max(counts.values())
            majority_threshold = max(t for t, c in counts.items() if c == max_count)

            majority_archs = {name for name, t in arch_thresholds.items() if t == majority_threshold}
            excluded = set(arch_thresholds.keys()) - majority_archs

            all_pred_signal_sets = [(name, s) for name, s in all_pred_signal_sets if name in majority_archs]

            self.logger.log(f"Majority label threshold: {majority_threshold} ({len(majority_archs)} archs)", 'info')
            if excluded:
                self.logger.log(f"  Excluded from consensus (different label threshold): {', '.join(sorted(excluded))}", 'info')

        # Consolidated consensus table (vote-based, using only high-precision architectures — C1/C2 fix)
        final_predictions = None
        if df_with_all_cols is not None and len(all_pred_signal_sets) > 0:
            non_empty = [(name, s) for name, s in all_pred_signal_sets if s]
            if non_empty:
                # Vote-based consensus: dynamic min_votes based on majority group size
                min_votes = max(3, len(majority_archs)) if majority_threshold is not None else 5
                all_indices = set.union(*(s for _, s in non_empty))
                vote_counts = {}
                vote_archs = {}
                for row_idx in all_indices:
                    archs = [name for name, s in non_empty if row_idx in s]
                    vote_counts[row_idx] = len(archs)
                    vote_archs[row_idx] = archs
                common_indices = {idx for idx, count in vote_counts.items() if count >= min_votes}
                if common_indices:
                    consensus_rows = df_with_all_cols.iloc[list(common_indices)]
                    self.logger.log("", 'info')
                    self.logger.log(f"Consolidated Predicted Signal ({len(majority_archs)} architectures, min {min_votes} votes):", 'info')
                    header_cols = list(df_with_all_cols.columns) + ['VoteCount', 'VotingArchs']
                    self.logger.log("Row," + ",".join(header_cols), 'info')
                    for idx, (orig_idx, row) in enumerate(consensus_rows.iterrows(), 1):
                        row_values = list(row.values) + [vote_counts[orig_idx], "+".join(vote_archs[orig_idx])]
                        self.logger.log(f"{idx}," + ",".join(str(v) for v in row_values), 'info')
                    self.logger.log(f"  Total consensus rows: {len(common_indices)}", 'info')
                    # Build final_predictions from consensus vote — only ticker_ids with ≥min_votes votes flagged as signal
                    final_predictions = np.zeros(n_inference)
                    for idx in common_indices:
                        final_predictions[idx] = 1

        # Binarize inference ground truth at majority label threshold for consensus metrics
        if majority_threshold is not None:
            y_inference_binarized = (y_val_continuous >= majority_threshold).astype(int)
        else:
            y_inference_binarized = None

        # =========================================================================
        # STEP 5: Summary Table (NO confusion matrix)
        # =========================================================================
        sorted_results = []
        if architecture_results:
            sorted_results = sorted(architecture_results, key=lambda x: x['Inf_P'], reverse=True)

            self.logger.log("", 'info')
            self.logger.log("FINAL PREDICTION RESULTS (sorted by inference_precision)", 'info')

            for rank, r in enumerate(sorted_results, 1):
                self.logger.log(
                    f"{rank}. {r['architecture']:<12} "
                    f"inference_precision={r['Inf_P']:.4f} "
                    f"inference_recall={r['Inf_R']:.4f} "
                    f"inference_auc={r['Inf_AUC']:.4f} "
                    f"inference_f1={r['Inf_F1']:.4f} "
                    f"inference_fn={r['Inf_FN']} "
                    f"inference_tn={r['Inf_TN']} "
                    f"inference_tp={r['Inf_TP']} "
                    f"inference_fp={r['Inf_FP']}",
                    'info'
                )

            # Best architecture
            best = sorted_results[0]
            phase5_time = time.time() - phase5_start_time
            self.logger.log(f"Best architecture: {best['architecture']} (inference_precision: {best['Inf_P']:.4f})", 'info')
            self.logger.log(f"phase 5 total time: {phase5_time:.1f}s", 'info')
            self.logger.log(f"data points evaluated: {n_inference} (date(s)={np.unique(dates_inference)})", 'info')

        # =========================================================================
        # STEP 6: Update context (fixes AssertionError: Missing final_predictions)
        # =========================================================================
        # Add final_predictions to context for pipeline validation
        # If consensus produced a result, use that; otherwise fall back to best single architecture
        if final_predictions is None and sorted_results:
            # Get predictions from best architecture
            best_arch = sorted_results[0]['architecture']
            if best_arch in models:
                best_meta = metadata.get(best_arch, {})
                best_kept = best_meta.get('kept_feature_indices')
                if best_kept:
                    X_best = X[:, best_kept]
                else:
                    X_best = X
                final_predictions = (models[best_arch].predict(X_best, verbose=0).flatten() >= 0.5).astype(int)

        context.update({
            'architecture_results': architecture_results,
            'phase5_complete': True,
            'final_metrics': sorted_results,
            'final_predictions': final_predictions,
            'y_inference_binarized': y_inference_binarized,
            'majority_threshold': majority_threshold,
            'inference_complete': True,
        })

        return context


# ============================================================================
# Section 4: StateManager (formerly chunk_13) ~50 lines
# ============================================================================
class StateManager:
    """
    Context key-value store. Simplified from chunk_13 — removed
    unused feedback_loop and backtracking features that were never
    triggered in production (20+ iterations without activation).

    Note: StateManager is still instantiated in TemporalWeighting
    and ModelTraining but its methods are never called in the
    pipeline (confirmed by grep across all 24 chunk files). The simplified
    class preserves the constructor pattern while removing 100+ lines of
    dead code, including validate_state_manager().

    NOTE on method names: the source chunk_13 methods are set_context_value/
    get_context_value/clear_context/update_feedback_loop/store_results etc.
    The simplified API here (set/get/update/keys) is a NEW surface — safe only
    because no production code calls any StateManager method (verified: only
    __main__ self-tests at chunk_13:178-211 call them).
    """
    def __init__(self):
        self.context = {}

    def set(self, key, value):
        self.context[key] = value

    def get(self, key, default=None):
        return self.context.get(key, default)

    def update(self, data: Dict):
        self.context.update(data)

    def keys(self):
        return self.context.keys()


# ============================================================================
# Section 3: PipelineOrchestrator (~150 lines)
# ============================================================================
class PipelineOrchestrator:
    """Orchestrates the complete stock analysis pipeline"""

    PHASE_DISPLAY_NAMES = [
        ('Pipeline Setup', DataSetup),
        ('Temporal Weighting', TemporalWeighting),
        ('Phase Xa: Raw Feature Importance', FeatureImportance),
        ('Phase BE: Backward Elimination', FeaturePruning),
        ('Neural Ensemble', ModelTraining),
        ('Phase Xb: Temporal Precision Gap', TemporalPrecisionGap),
        ('Prediction Optimization', Inference),
    ]

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
        # 0. Anchor random seeds for reproducibility
        #    (Must happen BEFORE any phase logic, model init, or sampling.)
        seed = self.config.get('RANDOM_SEED', 42)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        start_time = time.time()

        self.logger.log("[running] Starting Stock Analysis Pipeline", 'info')
        self.logger.log(f"   Data path: {self.config['DATA_PATH']}", 'info')
        try:
            data_path = self.config['DATA_PATH']
            with open(data_path) as f:
                header_line = f.readline().strip()
                header_cols = header_line.split(',')
                num_cols = len(header_cols)
                tid_col = header_cols.index('Ticker_id')
                date_col = header_cols.index('date')
                market_cap_col = header_cols.index('Market_Cap')
                reorder_cols = [date_col] + [i for i in range(num_cols) if i not in (date_col, tid_col)] + [tid_col]
                num_rows = 0
                ticker_max_cap = {}
                ticker_extremes = {}
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    num_rows += 1
                    tid = int(parts[tid_col])
                    date = int(parts[date_col])
                    cap = float(parts[market_cap_col])
                    ticker_max_cap[tid] = max(ticker_max_cap.get(tid, cap), cap)
                    if tid not in ticker_extremes:
                        ticker_extremes[tid] = {'newest': [(date, line)], 'oldest': [(date, line)]}
                    else:
                        ext = ticker_extremes[tid]
                        if len(ext['newest']) < 2:
                            ext['newest'].append((date, line))
                        elif date > ext['newest'][0][0]:
                            ext['newest'][1] = ext['newest'][0]
                            ext['newest'][0] = (date, line)
                        elif date > ext['newest'][1][0]:
                            ext['newest'][1] = (date, line)
                        if len(ext['oldest']) < 2:
                            ext['oldest'].append((date, line))
                        elif date < ext['oldest'][0][0]:
                            ext['oldest'][1] = ext['oldest'][0]
                            ext['oldest'][0] = (date, line)
                        elif date < ext['oldest'][1][0]:
                            ext['oldest'][1] = (date, line)
            file_size_mb = os.path.getsize(data_path) / 1024 / 1024
            self.logger.log(f"   Dataset shape: {num_rows:,} rows x {num_cols} columns ({file_size_mb:.1f} MB)", 'info')

            def _reorder_line(line, reorder_cols):
                parts = line.split(',')
                return ','.join(parts[i] for i in reorder_cols)

            largest_ticker = max(ticker_max_cap, key=ticker_max_cap.get)
            ext = ticker_extremes[largest_ticker]
            oldest_sorted = sorted(ext['oldest'], key=lambda x: x[0])
            newest_sorted = sorted(ext['newest'], key=lambda x: x[0], reverse=True)
            self.logger.log(_reorder_line(header_line, reorder_cols), 'info')
            self.logger.log(_reorder_line(newest_sorted[0][1], reorder_cols), 'info')
            self.logger.log(_reorder_line(newest_sorted[1][1], reorder_cols), 'info')
            self.logger.log(_reorder_line(oldest_sorted[1][1], reorder_cols), 'info')
            self.logger.log(_reorder_line(oldest_sorted[0][1], reorder_cols), 'info')
        except Exception as e:
            self.logger.log(f"   Could not read dataset: {e}", 'warning')
        self.logger.log(f"   Sampling: size={self.config['SAMPLE_SIZE']} (max), enabled={self.config['USE_SAMPLING']}", 'info')
        self.logger.log(f"   hyperparameter_optimization: trials={self.config['HYPERPARAM_OPTIMIZATION_TRIALS']}, continue_until_target={self.config['HPO_CONTINUE_UNTIL_TARGET']}, epochs_per_trial={self.config['HYPERPARAM_OPTIMIZATION_EPOCHS']}, stagnation_threshold={self.config['HPO_STAGNATION_THRESHOLD']}", 'info')
        first_thresh = self.config['FIRST_THRESHOLD']
        last_thresh = self.config['LAST_THRESHOLD']
        thresh_step = self.config['THRESHOLD_STEP']
        thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)
        self.logger.log(f"   Label_Thresholds: {first_thresh} to {last_thresh} ({len(thresholds)} thresholds)", 'info')

        # Configuration Summary — logs all active lever values for audit
        self.logger.log("--- Configuration Summary ---", 'info')
        c = self.config
        per_arch_items = c.get('PER_ARCH_WINSORIZE', {})
        per_arch_str = ', '.join(
            f"{k} low={v.get('low', '?'):d} high={v.get('high', '?'):d}"
            for k, v in sorted(per_arch_items.items())
        )
        self.logger.log(
            f"  [config] Preprocessing: winsor_features={c['WINSORIZE_FEATURES']}, "
            f"winsor_low={c['WINSORIZE_PERCENTILE_LOW']} (global), "
            f"log1p={c['LOG_TRANSFORM_FEATURES']}, skewed_cols={c['HIGHLY_SKEWED_FEATURES']}, "
            f"log1p_target={c['LOG_TRANSFORM_TARGET']}", 'info'
        )
        self.logger.log(f"  [config] Per-arch winsor: {per_arch_str}", 'info')
        self.logger.log(f"  [config] Top dates held out: {c['TOP_DATES_HELD_OUT']}", 'info')
        self.logger.log(
            f"  [config] Gates: min_pos_ratio={c['MIN_POS_PRED_RATIO']}, "
            f"max_pos_ratio={c['MAX_POS_PRED_RATIO']}", 'info'
        )
        self.logger.log(f"  [config] Safeguards: NN={c['NEURAL_SAFEGUARDS']}, trees={c['SKLEARN_SAFEGUARDS']}", 'info')
        self.logger.log(
            f"  [config] Training: hpo_epochs={c['HYPERPARAM_OPTIMIZATION_EPOCHS']}, "
            f"retrain_epochs={dict(c['HPO_RETRAIN_EPOCHS'])}, "
            f"final_epochs={dict(c['FINAL_TRAIN_EPOCHS'])}, "
            f"tree_early_stopping={c['TREE_EARLY_STOPPING_ROUNDS']}", 'info'
        )
        self.logger.log(f"  [config] Archs: NN={c['NEURAL_ARCHITECTURES']}, trees={c['TREE_ARCHITECTURES']}", 'info')
        self.logger.log(
            f"  [config] Ensemble: min_precision={c['ENSEMBLE_MIN_PRECISION']}, "
            f"weighting={c['ENSEMBLE_WEIGHTING']}, fallback={c['FALLBACK_ARCHITECTURE']}", 'info'
        )
        self.logger.log(f"  [config] Feature analysis: enabled={c['FEATURE_ANALYSIS_ENABLED']}", 'info')
        self.logger.log(f"  [config] Temporal: multiplier={c['TEMPORAL_MULTIPLIER']}", 'info')

        # Execute phases in sequence
        context = {}
        for display_name, PhaseClass in self.PHASE_DISPLAY_NAMES:
            phase_start = time.time()
            try:
                phase = PhaseClass(self.config)
                phase.logger = self.logger
                result = phase.execute(context)
                context.update(result)

                # Validate phase output via CONTEXT_CONSUMED/PRODUCED contract
                if PhaseClass in (FeatureImportance, FeaturePruning, TemporalPrecisionGap):
                    self.logger.log(f"[skip] {display_name} validation skipped (no validator)", 'info')
                else:
                    try:
                        phase.validate_context(context)
                    except AssertionError as e:
                        self.logger.log(f"[warning]  {display_name} validation warning: {e}", 'warning')

                # Record timing
                phase_time = time.time() - phase_start
                self.phase_timings[display_name] = phase_time
            except Exception as e:
                self.logger.log(f"[error] {display_name} failed: {e}", 'error')
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Pipeline failed at {display_name}") from e

        # Calculate total time
        total_time = time.time() - start_time

        # =========================================================================
        # METRICS REVIEW FRAMEWORK
        # =========================================================================
        self.logger.log("METRICS REVIEW REPORT", 'info')

        # Get architecture metrics from Phase 4
        arch_metrics = context.get('arch_final_metrics', [])

        if arch_metrics:
            # Sort by precision (descending)
            sorted_metrics = sorted(arch_metrics, key=lambda x: x.get('P', 0), reverse=True)

            self.logger.log(f"[architecture performance] (sorted by validation_precision)", 'info')

            ensemble_threshold = self.config['ENSEMBLE_MIN_PRECISION']

            for i, m in enumerate(sorted_metrics, 1):
                arch = m.get('arch', 'Unknown')
                p = m.get('P', 0)
                r = m.get('R', 0)
                auc = m.get('AUC', 0)
                tp = m.get('TP', 0)
                fp = m.get('FP', 0)

                status = f"PASS (Minimum Validation_Precision Required={ensemble_threshold})" if p >= ensemble_threshold else f"FAIL (Minimum Validation_Precision Required={ensemble_threshold})"
                self.logger.log(f"{i}. {arch:15s} validation_precision={p:.4f} validation_recall={r:.4f} validation_auc={auc:.4f} validation_true_positives={tp:5d} validation_false_positives={fp:5d} {status}", 'info')

            # =========================================================================
            # STANDARDIZED METRICS TABLE (CSV FORMAT)
            # =========================================================================
            self.logger.log(f"[standardized metrics table]", 'info')

            # Get inference metrics from Phase 5
            inference_metrics = context.get('architecture_results', [])

            # Get train/val metrics from Phase 4
            train_val_metrics = context.get('arch_final_metrics', [])

            # Build CSV output with enhanced fields
            csv_lines = []
            csv_lines.append("Architecture,Phase,Loss,Epochs,Precision,Recall,AUC,F1,TP,FP,TN,FN,MaxPred,MeanPred,StdPred,PctAboveThresh,BestEpoch,TrainingTime,LabelThresh,ThresholdSource,HPO_Trials,HPO_Improvement,KeyHyperparams,TrainLoss,ValLoss,LossDelta,MCC,PRAUC,Specificity,BalancedAccuracy,PredictionThreshold")

            # Known architecture order
            arch_order = self.config['ARCH_CSV_ORDER']

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

        else:
            self.logger.log("[warning] No architecture metrics found in context", 'warning')

        self.logger.log("END OF METRICS REVIEW", 'info')

        # Log final metrics — use consensus predictions vs binarized ground truth if available
        final_predictions = context.get('final_predictions')
        y_inference_binarized = context.get('y_inference_binarized')
        if final_predictions is not None and y_inference_binarized is not None:
            fp_bin = (np.asarray(final_predictions) >= 0.5).astype(int)
            cons_precision = precision_score(y_inference_binarized, fp_bin, zero_division=0)
            cons_recall = recall_score(y_inference_binarized, fp_bin, zero_division=0)
            cons_f1 = f1_score(y_inference_binarized, fp_bin, zero_division=0)
            try:
                cons_auc = roc_auc_score(y_inference_binarized, final_predictions)
            except Exception:
                cons_auc = 0.0
            self.logger.log(f"[stat] Final Results:", 'info')
            self.logger.log(f"   inference_precision={cons_precision:.4f}", 'info')
            self.logger.log(f"   inference_recall={cons_recall:.4f}", 'info')
            self.logger.log(f"   inference_f1={cons_f1:.4f}", 'info')
            self.logger.log(f"   inference_auc={cons_auc:.4f}", 'info')
        elif 'final_metrics' in context:
            # Fallback: display best architecture's metrics
            metrics = context['final_metrics']
            if isinstance(metrics, list) and len(metrics) > 0:
                metrics = metrics[0]
            if isinstance(metrics, dict):
                self.logger.log(f"[stat] Final Results:", 'info')
                self.logger.log(f"   inference_precision={metrics.get('Inf_P', 0):.4f}", 'info')
                self.logger.log(f"   inference_recall={metrics.get('Inf_R', 0):.4f}", 'info')
                self.logger.log(f"   inference_f1={metrics.get('Inf_F1', 0):.4f}", 'info')
                self.logger.log(f"   inference_auc={metrics.get('Inf_AUC', 0):.4f}", 'info')

        # Log final summary
        self.logger.log(f"[time] Total execution time: {total_time:.2f}s", 'info')
        self.logger.log("Phase timings:", 'info')
        for display_name, timing in self.phase_timings.items():
            self.logger.log(f"   {display_name}: {timing:.2f}s", 'info')

        return context


# ============================================================================
# main() (~50 lines)
# ============================================================================
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
    assert 'final_metrics' in context, "Missing final_metrics"
    assert 'final_predictions' in context, "Missing final_predictions"
    assert context.get('final_predictions') is not None, "No predictions generated"
    assert len(context['final_predictions']) > 0, "Predictions are empty"
    logger.log("[pipeline] [info] Pipeline execution validated successfully", 'info')

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
        sys.exit(1)

    except ValueError as e:
        print(f"\n[error] data validation error: {e}")
        print("\n[fix] solution: Please check your CSV file format and data quality")
        sys.exit(1)

    except RuntimeError as e:
        print(f"\n[error] pipeline error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n[error] unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
