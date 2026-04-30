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
- - Using y >= 0.5 would treat ANY change >= $0.50 as fraud (wrong!)
-
- Model outputs: probabilities (0-1) from model.predict()
- Binary predictions: (predictions >= 0.5).astype(int)
"""

import numpy as np
import time
from typing import Dict

from chunk_15_phase_base import BasePhase
from chunk_02_utils_logging import Logger
from chunk_12_evaluation_evaluator import Evaluator
from chunk_14_models_trainer import ModelTrainer
from chunk_05_data_manager import DataManager
from chunk_11_models_sklearn import FocalLoss


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
        self.logger.log("Starting Phase 5: Inference on Newest Data", 'info')
        
        phase5_start_time = time.time()
        
        # =========================================================================
        # STEP 1: Load saved models and metadata from ./saved_models/
        # =========================================================================
        from chunk_22_model_loader import load_models_with_metadata, load_preprocessing_params
        
        models_path = self.config.get('MODELS_PATH', './saved_models')
        data_path = self.config.get('DATA_PATH', 'for_train_x_2025_10_24_clean.csv')
        
        self.logger.log(f"Loading saved models from {models_path}...", 'info')
        models, metadata = load_models_with_metadata(models_path)
        
        if not models:
            self.logger.log("No models found in saved_models directory", 'warning')
            context.update({'phase5_complete': True})
            return context
        
        self.logger.log(f"Loaded architectures: {list(models.keys())}", 'info')
        
        # =========================================================================
        # STEP 2: Load preprocessing parameters
        # =========================================================================
        self.logger.log("Loading preprocessing parameters...", 'info')
        preprocess_params = load_preprocessing_params(models_path)
        
        temporal_weights = preprocess_params.get('temporal_weights', {})
        feature_names = preprocess_params.get('feature_names', [])
        all_dates = preprocess_params.get('all_dates', {})
        
        self.logger.log(f"  temporal_weights: {len(temporal_weights)} entries", 'info')
        self.logger.log(f"  Feature names ({len(feature_names)}): {feature_names}", 'info')
        self.logger.log(f"  all_dates: {len(all_dates)} dates", 'info')
        
        # =========================================================================
        # STEP 3: Load fresh data and find highest/newest date
        # =========================================================================
        self.logger.log(f"Loading data from {data_path}...", 'info')
        
        import pandas as pd
        
        # Load data
        df = pd.read_csv(data_path)
        self.logger.log(f"  Loaded {len(df)} rows", 'info')
        
        # Find highest/newest dates in the data
        unique_dates = sorted(df['date'].unique())
        inference_date = unique_dates[-2]  # Second highest/newest date (newest is held out)
        newest_held_out = unique_dates[-1]  # Newest date is held out
        self.logger.log(f"  Second newest date for inference: {inference_date}", 'info')
        self.logger.log(f"  Newest date held out: {newest_held_out}", 'info')
        
        # Filter for only the inference date (second newest)
        df_filtered = df[df['date'] == inference_date].copy()
        self.logger.log(f"  Filtered to {len(df_filtered)} rows for date {inference_date}", 'info')
        
        if len(df_filtered) == 0:
            self.logger.log("ERROR: No data found for inference date!", 'error')
            context.update({'phase5_complete': True})
            return context
        
        # =========================================================================
        # STEP 4: Prepare features (X) and labels (y) for inference
        # =========================================================================
        target_col = self.config.get('TARGET_COLUMN', 'ChangeY')
        
        # Extract features and labels
        X_raw = df_filtered[feature_names].values if feature_names else df_filtered.iloc[:, 1:].values
        y_raw = df_filtered[target_col].values
        
        # Keep DataFrame with all columns for fraud row output
        df_with_all_cols = df_filtered[feature_names + [target_col]].copy()
        
        self.logger.log(f"  X shape: {X_raw.shape}", 'info')
        self.logger.log(f"  y shape: {y_raw.shape}", 'info')
        
        # =========================================================================
        # STEP 5: Apply temporal weighting (same as Phase 3/4)
        # =========================================================================
        # Apply temporal features
        dates = df_filtered['date'].values
        
        # Create temporal indices for each date
        temporal_indices = np.array([all_dates.get(str(d), 0) for d in dates])
        
        # Apply temporal weighting (sqrt to avoid over-amplification)
        temporal_multiplier = self.config.get('TEMPORAL_MULTIPLIER', 9.0)
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
        self.logger.log("Phase 5: Per-Architecture Results on Newest Data", 'info')
        self.logger.log("=" * 80, 'info')
        
        for arch_name, model in models.items():
            # Get metadata for this architecture
            arch_metadata = metadata.get(arch_name, {})
            opt_threshold = arch_metadata.get('optimal_threshold', 20.0)
            best_hyperparams = arch_metadata.get('best_hyperparams', {})
            best_val_prec = arch_metadata.get('best_val_precision', 0.0)
            pred_threshold = self.config.get('PREDICTION_THRESHOLD', 0.5)
            
            self.logger.log(
                f"{arch_name} (Label_Threshold={opt_threshold:.1f}, Prediction_Binary_Split={pred_threshold:.2f}, hyperparams={best_hyperparams}, Val Precision={best_val_prec:.4f}):",
                'info'
            )
            
            # Run inference
            try:
                predictions = model.predict(X, verbose=0).flatten()
            except Exception as e:
                self.logger.log(f"   Prediction failed: {e}", 'warning')
                continue
            
            # Validate predictions are in valid probability range [0,1]
            if np.any(np.isnan(predictions)):
                self.logger.log(f"   [WARNING] NaN values detected in predictions!", 'warning')
            if predictions.min() < 0 or predictions.max() > 1:
                self.logger.log(f"   [WARNING] Predictions outside [0,1] range! min={predictions.min():.4f}, max={predictions.max():.4f}", 'warning')
            
            # Log prediction distribution statistics
            self.logger.log(f"   Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}, min={predictions.min():.4f}, max={predictions.max():.4f}", 'info')
            
            # Binary predictions: use prediction threshold from config
            binary_predictions = (predictions >= pred_threshold).astype(int)
            
            # Log prediction class distribution
            pos_pct = binary_predictions.mean() * 100
            self.logger.log(f"   INFERENCE Predictions: {pos_pct:.2f}% positive predictions ({binary_predictions.sum():,} / {len(binary_predictions):,})", 'info')
            
            # Check for all-zero predictions
            if binary_predictions.sum() == 0:
                self.logger.log(f"   [WARNING] Model predicts ALL NEGATIVES!", 'warning')
            
            # Ground truth: use optimal_threshold for y
            y_binary = (y_raw >= opt_threshold).astype(int)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_binary, binary_predictions, predictions)
            
            # Calculate confusion matrix components
            tp = int(np.sum((binary_predictions == 1) & (y_binary == 1)))
            tn = int(np.sum((binary_predictions == 0) & (y_binary == 0)))
            fp = int(np.sum((binary_predictions == 1) & (y_binary == 0)))
            fn = int(np.sum((binary_predictions == 0) & (y_binary == 1)))
            
            # Log metrics (NO confusion matrix)
            self.logger.log(
                f"  {arch_name} t={opt_threshold:.1f}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} FN={fn} TN={tn} TP={tp} FP={fp}",
                'info'
            )
            
            # Log new enhanced metrics
            inf_mcc = metrics.get('mcc', 0.0)
            inf_prauc = metrics.get('average_precision', 0.0)
            inf_specificity = metrics.get('specificity', 0.0)
            inf_balanced_acc = metrics.get('balanced_accuracy', 0.0)
            self.logger.log(
                f"  {arch_name} - Inference: MCC={inf_mcc:.4f} PR-AUC={inf_prauc:.4f} Spec={inf_specificity:.4f} BalAcc={inf_balanced_acc:.4f}",
                'info'
            )
            
            # Output fraud-predicted rows (both MODEL PREDICTED and ACTUAL)
            fraud_output_cols = ['Market_Cap', '52W_Low', '52W_High', 'Change', 'ChangeY', 'Ticker_id']
            available_cols = [c for c in fraud_output_cols if c in df_with_all_cols.columns]
            
            # 1. Output MODEL PREDICTED fraud rows (probability >= 0.5)
            pred_fraud_indices = np.where(binary_predictions == 1)[0]
            if len(pred_fraud_indices) > 0:
                pred_fraud_rows = df_with_all_cols.iloc[pred_fraud_indices]
                fraud_rows_filtered = pred_fraud_rows[available_cols]
                
                self.logger.log("", 'info')
                self.logger.log(f"==== {arch_name} MODEL PREDICTED FRAUD ({len(pred_fraud_indices)} rows) ====", 'info')
                self.logger.log(f"Architecture: {arch_name} | Precision: {metrics['precision']:.4f} | Prediction_Binary_Split: {pred_threshold} | Predicted: {len(pred_fraud_indices)}", 'info')
                self.logger.log("=" * 80, 'info')
                self.logger.log("Row," + ",".join(available_cols), 'info')
                
                for idx, (_, row) in enumerate(fraud_rows_filtered.iterrows(), 1):
                    row_str = f"{idx}," + ",".join(str(v) for v in row.values)
                    self.logger.log(row_str, 'info')
                self.logger.log("=" * 80, 'info')


            
            # Store results
            architecture_results.append({
                'architecture': arch_name,
                'label_threshold': opt_threshold,
                'pred_threshold': pred_threshold,
                'hyperparams': best_hyperparams,
                'val_precision': best_val_prec,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                # NEW: Prediction distribution
                'mean_pred': float(predictions.mean()),
                'std_pred': float(predictions.std()),
                'max_pred': float(predictions.max()),
                'pct_above_thresh': float((predictions >= pred_threshold).mean() * 100),
                # NEW: Enhanced metrics
                'inf_mcc': inf_mcc,
                'inf_prauc': inf_prauc,
                'inf_specificity': inf_specificity,
                'inf_balanced_acc': inf_balanced_acc,
            })
        
        # =========================================================================
        # STEP 7: Summary Table (NO confusion matrix)
        # =========================================================================
        if architecture_results:
            # Sort by precision descending
            sorted_results = sorted(architecture_results, key=lambda x: x['precision'], reverse=True)
            
            self.logger.log("", 'info')
            self.logger.log("=" * 80, 'info')
            self.logger.log("FINAL PREDICTION RESULTS (sorted by Precision)", 'info')
            self.logger.log("=" * 80, 'info')
            self.logger.log(
                f"{'Rank':>4} | {'Architecture':<8} | {'Label_Threshold':>15} | {'Prediction_Binary_Split':>22} | "
                f"{'Precision':>10} | {'Recall':>7} | {'AUC':>7} | {'F1':>6} | "
                f"{'FN':>4} | {'TN':>5} | {'TP':>4} | {'FP':>4}",
                'info'
            )
            self.logger.log("-" * 80, 'info')
            
            for rank, r in enumerate(sorted_results, 1):
                self.logger.log(
                    f"{rank:>4} | {r['architecture']:<8} | {r['label_threshold']:>15.1f} | {r['pred_threshold']:>22.2f} | "
                    f"{r['precision']:>10.4f} | {r['recall']:>7.4f} | {r['auc']:>7.4f} | "
                    f"{r['f1']:>6.4f} | {r['fn']:>4} | {r['tn']:>5} | "
                    f"{r['tp']:>4} | {r['fp']:>4}",
                    'info'
                )
            
            self.logger.log("=" * 80, 'info')
            
            # Best architecture
            best = sorted_results[0]
            phase5_time = time.time() - phase5_start_time
            self.logger.log(f"Best Architecture: {best['architecture']} (Precision: {best['precision']:.4f})", 'info')
            self.logger.log(f"Phase 5 Total Time: {phase5_time:.1f}s", 'info')
            self.logger.log(f"Data Points Evaluated: {len(df_filtered)} (date={inference_date})", 'info')
            self.logger.log("=" * 80, 'info')
        
        self.logger.log("", 'info')
        self.logger.log("Phase 5 completed successfully", 'info')
        
        # =========================================================================
        # STEP 8: Update context (fixes AssertionError: Missing final_predictions)
        # =========================================================================
        # Add final_predictions to context for pipeline validation
        final_predictions = None
        if sorted_results:
            # Get predictions from best architecture
            best_arch = sorted_results[0]['architecture']
            if best_arch in models:
                final_predictions = models[best_arch].predict(X, verbose=0).flatten()
        
        context.update({
            'architecture_results': architecture_results,
            'phase5_complete': True,
            'final_metrics': sorted_results,  # Use sorted results (highest precision first)
            'final_predictions': final_predictions
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
        assert 'precision' in result, "Missing precision"
        assert 'recall' in result, "Missing recall"
        assert 'f1' in result, "Missing f1"
        assert 'auc' in result, "Missing auc"
    
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
    optimal_thresholds = [23.3, 23.7]
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
        'optimal_thresholds': optimal_thresholds,
        'best_hyperparams_list': best_hyperparams_list,
        'best_val_precision_list': best_val_precision_list,
        'phase1_complete': True,
        'phase3_complete': True,
        'phase4_complete': True
    }
    
    # Run Phase 5
    phase5 = Phase5_PredictionOptimization(config)
    result = phase5.execute(context.copy())
    
    print(f"[PASS] Phase 5 executed successfully")
    print(f"   Architecture results: {len(result['architecture_results'])} architectures")
    
    # Validate
    validate_phase5_output(result)
    print("[PASS] Phase 5 output validation passed")
    
    print("\n[PASS] Phase5_PredictionOptimization tests passed")