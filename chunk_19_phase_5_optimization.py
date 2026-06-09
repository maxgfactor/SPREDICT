"""
Chunk 19: Phase 5 - Prediction Optimization
Final prediction and evaluation phase

## Purpose
Phase 5 generates final predictions and evaluates each architecture using
the optimal thresholds found in Phase 4.

## Key Responsibilities
- Load trained models from Phase 4 context
- Generate predictions on full dataset
- Calculate per-architecture metrics (P, R, F1, AUC, confusion matrix)
- Report final ranking by precision

## Dependencies
- Input: models, optimal_thresholds, best_hyperparams from Phase 4
- Output: architecture_results, final_metrics for pipeline validation

## IMPORTANT: Threshold Explanation
- y (context['y']): Raw continuous ChangeY values (e.g., 0, 5.2, 22.1, 100, 32500)
- opt_threshold: The threshold found in Phase 4 (e.g., 22.1 for RNN, 21.3 for LSTM)
-
- WHY opt_threshold for y, NOT 0.5?
- - Phase 4 trained models using labels created with opt_threshold
- - y_binary = (y >= opt_threshold) ensures consistent evaluation
- - Using y >= 0.5 would treat ANY change >= $0.50 as signal (wrong!)
-
- Model outputs: probabilities (0-1) from model.predict()
- Binary predictions: (predictions >= 0.5).astype(int)
"""

import numpy as np
import time
from typing import Dict
from collections import Counter
from chunk_15_phase_base import BasePhase
from chunk_02_utils_logging import Logger
from chunk_12_evaluation_evaluator import Evaluator
from chunk_14_models_trainer import ModelTrainer
from chunk_05_data_manager import DataManager
from chunk_11_models_sklearn import FocalLoss
from chunk_04_utils_metrics import inverse_log_transform


class Phase5_PredictionOptimization(BasePhase):
    """Phase 5: Final prediction optimization and evaluation"""
    
    def __init__(self, config: Dict):
        """
        Initialize Phase 5
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = Logger(config)
        self.evaluator = Evaluator(config)
        self.data_manager = DataManager(config)
    
    def execute(self, context: Dict) -> Dict:
        """
        Execute Phase 5: Inference on newest data using saved models
        
        This phase:
        1. Loads saved models from ./saved_models/
        2. Loads preprocessing params (temporal_weights, feature_names)
        3. Loads fresh data from for_train_x_2025_10_24_clean.csv
        4. Filters for single highest/newest date (20251023)
        5. Applies preprocessing (temporal features + weights)
        6. Runs inference with each model
        7. Evaluates and reports standard metrics
        
        Args:
            context: Pipeline context (not used for data, models loaded from disk)
            
        Returns:
            Updated context with per-architecture metrics and final_predictions
        """
        phase5_start_time = time.time()
        
        # =========================================================================
        # STEP 1: Load saved models and metadata from ./saved_models/
        # =========================================================================
        from chunk_22_model_loader import load_models_with_metadata, load_preprocessing_params, load_scaler
        
        models_path = self.config['MODELS_PATH']
        data_path = self.config['DATA_PATH']
        
        self.logger.log(f"Loading saved models from {models_path}...", 'info')
        models, metadata = load_models_with_metadata(models_path)
        
        if not models:
            self.logger.log("No models found in saved_models directory", 'warning')
            context.update({'phase5_complete': True})
            return context
        
        self.logger.log(f"Loaded architectures: {list(models.keys())}", 'info')
        
        # =========================================================================
        # STEP 2: Receive data from context (NO CSV loading)
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
        # STEP 3: Prepare inference data (RAW, NO temporal weighting)
        # =========================================================================
        # Use X_inference directly - NO temporal weighting applied
        X_raw = X_inference

        # Per-architecture feature selection happens in STEP 6 using metadata['kept_feature_indices']
        self.logger.log(f"  Features will be selected per-architecture from {X_raw.shape[1]} total", 'info')
        y_val_continuous = y_inference_continuous if y_inference_continuous is not None else np.zeros(n_inference)
        
        # Get threshold info from config
        label_threshold = self.config['FIRST_THRESHOLD']
        
        self.logger.log(f"  Label threshold: {label_threshold}", 'info')
        self.logger.log(f"  Inference data: RAW (no temporal weighting applied)", 'info')
        
        # Feature names from context
        feature_names = context_feature_names
        
        # Keep original data for output
        import pandas as pd
        df_with_all_cols = context.get('df_inference', None)
        if df_with_all_cols is None and X_inference is not None:
            fn = context.get('feature_names', [])
            if fn and len(fn) == X_inference.shape[1]:
                df_with_all_cols = pd.DataFrame(X_inference, columns=fn)
        
        self.logger.log(f"  X shape (RAW, no weighting): {X_raw.shape}", 'info')
        self.logger.log(f"  y shape: {y_val_continuous.shape}", 'info')
        
        # =========================================================================
        # STEP 5: Apply temporal weighting (same as Phase 3/4)
        # =========================================================================
        # Use dates_inference from context (available since line 110)
        dates = dates_inference
        
        # Create temporal indices for each date
        # all_dates maps date -> index (0 = oldest, max = newest)
        unique_dates = np.unique(dates)
        all_dates = {str(d): i for i, d in enumerate(unique_dates)}
        temporal_indices = np.array([all_dates.get(str(d), 0) for d in dates])
        
        # Apply temporal weighting (sqrt to avoid over-amplification)
        temporal_multiplier = self.config['TEMPORAL_MULTIPLIER']
        temporal_w = 1.0 + (temporal_indices / max(len(all_dates), 1)) * (temporal_multiplier - 1.0)
        temporal_w = np.sqrt(temporal_w)
        
        # Apply weights to features
        X = X_raw * temporal_w[:, np.newaxis]
        
        self.logger.log(f"  Temporal weights applied: min={temporal_w.min():.2f}, max={temporal_w.max():.2f}", 'info')
        
        # =========================================================================
        # STEP 6: Run inference and evaluate each architecture
        # =========================================================================
        architecture_results = []
        
        self.logger.log("", 'info')
        self.logger.log("Per-Architecture Results on Newest Data", 'info')
        
        # Consolidated column set (same df_with_all_cols for all architectures)
        available_cols = list(df_with_all_cols.columns) if df_with_all_cols is not None else []
        # C1/C2: Use only high-precision models for consensus, with configurable vote threshold
        ensemble_min_precision = self.config.get('ENSEMBLE_MIN_PRECISION', 0.40)
        ensemble_vote_threshold = self.config.get('ENSEMBLE_VOTE_THRESHOLD', 0.5)
        all_pred_signal_sets = []  # list of (arch_name, set(indices))
        arch_thresholds = {}  # arch_name -> opt_threshold
        
        for arch_name, model in models.items():
            # Get metadata for this architecture
            arch_metadata = metadata.get(arch_name, {})
            opt_threshold = arch_metadata.get('optimal_threshold', 20.0)
            best_hyperparams = arch_metadata.get('best_hyperparams', {})
            best_val_prec = arch_metadata.get('best_val_precision', 0.0)
            pred_threshold = self.config['PREDICTION_THRESHOLD']
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
            
            # Apply StandardScaler for NN architectures (Bug 2 fix: BN inference collapse)
            nn_archs = ['CNN', 'RNN', 'LSTM', 'Dense', 'VAE', 'Transformer']
            if arch_name in nn_archs:
                scaler = load_scaler(arch_name, models_path)
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
            if self.config.get('LOG_TRANSFORM_TARGET', False):
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
                    self.logger.log(f"architecture: {arch_name} | inference_precision: {metrics['precision']:.4f} | prediction_binary_split: {pred_threshold} | predicted: {len(pred_signal_indices)}", 'info')
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
        # STEP 7: Summary Table (NO confusion matrix)
        # =========================================================================
        sorted_results = []
        if architecture_results:
            sorted_results = sorted(architecture_results, key=lambda x: x['Inf_P'], reverse=True)
            
            self.logger.log("", 'info')
            self.logger.log("FINAL PREDICTION RESULTS (sorted by inference_precision)", 'info')
            self.logger.log(
                f"{'Rank':>4} | {'Architecture':<8} | {'Label_Threshold':>15} | {'Prediction_Binary_Split':>22} | "
                f"{'inference_precision':>19} | {'inference_recall':>16} | {'inference_auc':>13} | {'inference_f1':>12} | "
                f"{'inference_false_negatives':>25} | {'inference_true_negatives':>24} | {'inference_true_positives':>24} | {'inference_false_positives':>25}",
                'info'
            )
            
            for rank, r in enumerate(sorted_results, 1):
                self.logger.log(
                    f"{rank:>4} | {r['architecture']:<8} | {r['label_threshold']:>15.1f} | {r['pred_threshold']:>22.2f} | "
                    f"{r['Inf_P']:>19.4f} | {r['Inf_R']:>16.4f} | {r['Inf_AUC']:>13.4f} | "
                    f"{r['Inf_F1']:>12.4f} | {r['Inf_FN']:>25} | {r['Inf_TN']:>24} | "
                    f"{r['Inf_TP']:>24} | {r['Inf_FP']:>25}",
                    'info'
                )
            
            # Best architecture
            best = sorted_results[0]
            phase5_time = time.time() - phase5_start_time
            self.logger.log(f"Best architecture: {best['architecture']} (inference_precision: {best['Inf_P']:.4f})", 'info')
            self.logger.log(f"phase 5 total time: {phase5_time:.1f}s", 'info')
            self.logger.log(f"data points evaluated: {n_inference} (date={dates_inference[0]})", 'info')
        
        # =========================================================================
        # STEP 8: Update context (fixes AssertionError: Missing final_predictions)
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
            'majority_threshold': majority_threshold
        })
        
        return context
    
    def _validate_input(self, context: Dict):
        """
        Validate Phase 5 input requirements
        
        Args:
            context: Input context
            
        Raises:
            ValueError: If validation fails
        """
        if not context.get('phase4_complete'):
            raise ValueError("Phase 4 must complete before Phase 5")
        
        required = ['X', 'y', 'models', 'arch_names', 'optimal_thresholds']
        for key in required:
            if key not in context:
                raise ValueError(f"Phase 5 missing required input: {key}")


def validate_phase5_input(context: Dict) -> bool:
    """
    Ensure Phase 5 has required inputs
    
    Args:
        context: Input context
        
    Returns:
        True if valid
    """
    assert context.get('phase4_complete') == True, "Phase 4 must be complete"
    required = ['X', 'y', 'models', 'arch_names', 'optimal_thresholds']
    for key in required:
        assert key in context, f"Phase 5 missing required input: {key}"
    return True


def validate_phase5_output(context: Dict) -> bool:
    """
    Validate Phase 5 output
    
    Args:
        context: Output context from Phase 5
        
    Returns:
        True if valid
    """
    required = ['architecture_results', 'phase5_complete']
    for key in required:
        assert key in context, f"Phase 5 missing output: {key}"
    
    # Validate architecture results
    results = context['architecture_results']
    assert isinstance(results, list), "architecture_results must be a list"
    
    for result in results:
        assert 'architecture' in result, "Missing architecture name"
        assert 'Inf_P' in result, "Missing precision"
        assert 'Inf_R' in result, "Missing recall"
        assert 'Inf_F1' in result, "Missing f1"
        assert 'Inf_AUC' in result, "Missing auc"
    
    assert context['phase5_complete'] == True
    
    return True


if __name__ == "__main__":
    # Self-test
    print("Testing Phase5_PredictionOptimization...")
    
    # Create test config
    config = {
        'LOG_VERBOSITY': 0
    }
    
    # Create mock context from Phase 4
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create mock models
    class MockModel:
        def predict(self, X, verbose=0):
            return np.random.rand(len(X))
    
    models = [MockModel(), MockModel()]
    arch_names = ['RNN', 'LSTM']
    final_thresholds = [23.3, 23.7]
    best_hyperparams_list = [{'lstm_units': 32}, {'lstm_units': 16}]
    best_val_precision_list = [0.7017, 0.7562]
    
    context = {
        'X': np.random.randn(n_samples, n_features).astype(np.float32),
        'y': np.random.randint(0, 2, n_samples).astype(np.float32),
        'dates': np.random.randint(20220101, 20230101, n_samples),
        'temporal_weights': np.ones(n_samples),
        'temporal_features': {'weights': np.ones(n_samples)},
        'models': models,
        'arch_names': arch_names,
        'optimal_thresholds': final_thresholds,
        'best_hyperparams_list': best_hyperparams_list,
        'best_val_precision_list': best_val_precision_list,
        'phase1_complete': True,
        'phase3_complete': True,
        'phase4_complete': True
    }
    
    # Run Phase 5
    phase5 = Phase5_PredictionOptimization(config)
    result = phase5.execute(context.copy())
    
    print(f"[pass] Phase 5 executed successfully")
    print(f"   Architecture results: {len(result['architecture_results'])} architectures")
    
    # Validate
    validate_phase5_output(result)
    
    print("\n[pass] Phase5_PredictionOptimization tests passed")