"""
phases.py — Pipeline Phases
Refactored from chunk_15, chunk_16, chunk_17, chunk_XX_phase_feature_analysis_a,
chunk_XX_feature_importance, chunk_XX_phase_backward_elimination,
chunk_XX_phase_feature_analysis_b (2026-08-07).
Section 1: BasePhase ABC
Section 2: DataSetup (Phase 1)
Section 3: TemporalWeighting (Phase 3)
Section 4: FeatureImportance (Phase Xa, 6-method analysis + pruning)
Section 5: FeaturePruning (Phase BE, per-arch backward elimination)
Section 6: TemporalPrecisionGap (Phase Xb)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_score, roc_auc_score
from scipy.stats import spearmanr, pointbiserialr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import PREDICTION_THRESHOLD_DEFAULT
from pipeline_logging import Logger
from data_loader import DataManager


# ============================================================================
# Section 1: BasePhase ABC
# ============================================================================

class BasePhase(ABC):
    """Abstract base class for all pipeline phases."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = None   # Set by subclass' own __init__

    @abstractmethod
    def execute(self, context: Dict) -> Dict:
        """Execute phase logic. Returns updated context."""
        pass

    def validate_context(self, context: Dict) -> None:
        """
        Verify CONTEXT_CONSUMED keys exist before execute() and
        CONTEXT_PRODUCED keys were written after execute().
        Called by PipelineOrchestrator after each phase.
        """
        for key in self.CONTEXT_CONSUMED:
            assert key in context, (
                f"[{self.__class__.__name__}] missing consumed context key: {key}")
        for key in self.CONTEXT_PRODUCED:
            assert key in context, (
                f"[{self.__class__.__name__}] missing produced context key: {key}")

    CONTEXT_CONSUMED = []   # Documented context keys this phase reads
    CONTEXT_PRODUCED = []   # Documented context keys this phase writes


# ============================================================================
# Section 2: DataSetup (Phase 1)
# ============================================================================

class DataSetup(BasePhase):
    """
    Phase 1: Load CSV → filter NaNs/zeros → sample → feature engineer →
    class distribution → train/val/inference split.
    """
    CONTEXT_CONSUMED = []  # Phase 1 starts with empty context
    CONTEXT_PRODUCED = ['X', 'y', 'dates', 'X_train', 'y_train_continuous',
                        'dates_train', 'X_val', 'y_val_continuous', 'dates_val',
                        'X_inference', 'y_inference_continuous', 'dates_inference',
                        'raw_target_values', 'raw_target_column', 'feature_names',
                        'data_stats', 'phase1_complete']

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.data_manager = DataManager(config)

    def execute(self, context: Dict) -> Dict:
        # Load data (skip global winsorization when per-arch mode is active)
        per_arch_active = bool(self.config.get('PER_ARCH_WINSORIZE', {}))
        try:
            X, y, dates = self.data_manager.load_data(winsorize=not per_arch_active)
            self.logger.log(f"Data loaded: {len(X)} samples, {X.shape[1]} features", 'info')
            if per_arch_active:
                self.logger.log("Per-arch winsorization active — global winsorization skipped in Phase 1", 'info')
            self.logger.log_temporal_coverage(dates)
        except FileNotFoundError as e:
            self.logger.log(f"critical: Data file not found - {e}", 'error')
            raise RuntimeError("Cannot proceed without valid stock data file") from e
        except ValueError as e:
            self.logger.log(f"data validation error: {e}", 'error')
            raise RuntimeError("Data validation failed") from e
        except Exception as e:
            self.logger.log(f"Unexpected data loading error: {e}", 'error')
            raise RuntimeError("Data loading failed") from e

        self.logger.log(f"data count: {len(X)} samples (max configured: {self.config['SAMPLE_SIZE']})", 'info')

        if self.config['TARGET_TYPE'] == 'continuous':
            raw_target = self.data_manager._raw_target_values
            if raw_target is not None:
                self.logger.log(f"Target Distribution (ChangeY):", 'info')
                self.logger.log(f"   Min: {np.nanmin(raw_target):.2f} | Max: {np.nanmax(raw_target):.2f} | Mean: {np.nanmean(raw_target):.2f}", 'info')
                self.logger.log(f"   Median: {np.nanmedian(raw_target):.2f} | Std: {np.nanstd(raw_target):.2f}", 'info')
                if np.nanmax(raw_target) > 100:
                    self.logger.log(f"sanity check: Extreme target values detected (max={np.nanmax(raw_target):.2f})", 'warning')
                first_thresh = self.config['FIRST_THRESHOLD']
                last_thresh = self.config['LAST_THRESHOLD']
                thresh_step = self.config['THRESHOLD_STEP']
                thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)

                for thresh in thresholds:
                    y_binary = (raw_target >= thresh).astype(int)
                    signal_count = int(np.sum(y_binary))
                    normal_count = len(y_binary) - signal_count
                    total = len(y_binary)
                    signal_rate = signal_count / total
                    imbalance_ratio = max(signal_count, normal_count) / min(signal_count, normal_count) if min(signal_count, normal_count) > 0 else float('inf')

                    self.logger.log(f"[class] Class Distribution (LABEL_THRESHOLD={thresh:>4.1f}):", 'info')
                    self.logger.log(f"  Total samples: {total:,}", 'info')
                    self.logger.log(f"  Signal cases: {signal_count:,} ({signal_rate:.1%})", 'info')
                    self.logger.log(f"  Normal cases: {normal_count:,} ({normal_count/total:.1%})", 'info')
                    self.logger.log(f"  Imbalance ratio: {imbalance_ratio:.1f}:1", 'info')

        stats = {
            'signal_rate': float(y.mean()),
            'missing_values': 0,  # Already handled in load_data
            'n_samples': len(X),
            'n_features': X.shape[1]
        }

        first_thresh = self.config['FIRST_THRESHOLD']
        last_thresh = self.config['LAST_THRESHOLD']
        thresh_step = self.config['THRESHOLD_STEP']
        thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)

        for thresh in thresholds:
            y_binary = (y >= thresh).astype(int)
            signal_count = int(np.sum(y_binary))
            normal_count = len(y_binary) - signal_count
            total = len(y_binary)
            signal_rate = signal_count / total
            imbalance_ratio = max(signal_count, normal_count) / min(signal_count, normal_count) if min(signal_count, normal_count) > 0 else float('inf')

            self.logger.log(f"[class] Class Distribution (LABEL_THRESHOLD={thresh:>4.1f}):", 'info')
            self.logger.log(f"  Total samples: {total:,}", 'info')
            self.logger.log(f"  Signal cases: {signal_count:,} ({signal_rate:.1%})", 'info')
            self.logger.log(f"  Normal cases: {normal_count:,} ({normal_count/total:.1%})", 'info')
            self.logger.log(f"  Imbalance ratio: {imbalance_ratio:.1f}:1", 'info')

        self.logger.log_feature_quality_metrics(X)

        original_signal_rate = y.mean()
        X, y, dates = self.data_manager.augment_signal_cases(X, y, dates)
        augmented_signal_rate = y.mean()
        if augmented_signal_rate > original_signal_rate:
            self.logger.log(f"Signal augmentation: {original_signal_rate:.4f} → {augmented_signal_rate:.4f}", 'info')

        X, y, dates = self.data_manager.concentrate_signal_cases(X, y, dates)
        self.logger.log(f"Data concentration: {len(X)} samples retained", 'info')

        X = self.data_manager.prepare_data(X)
        self.logger.log(f"Data preprocessing complete: {X.shape[1]} features", 'info')

        raw_target_values = self.data_manager._raw_target_values
        raw_target_column = self.data_manager._raw_target_column

        if raw_target_values is not None and len(raw_target_values) != len(X):
            if hasattr(self.data_manager, '_sampled_indices'):
                raw_target_values = raw_target_values[self.data_manager._sampled_indices]
            elif len(raw_target_values) > len(X):
                raw_target_values = raw_target_values[:len(X)]

        feature_names = (self.data_manager._feature_columns[:X.shape[1]]
                          if hasattr(self.data_manager, '_feature_columns')
                          else [f'feature_{i}' for i in range(X.shape[1])])

        self.logger.log(f"  Feature names ({len(feature_names)}): {feature_names}", 'info')

        # DATA SPLIT: Extract inference FIRST, then split remaining into train/val
        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)

        n_held_out = self.config['TOP_DATES_HELD_OUT']
        inference_dates = unique_dates[-n_held_out:] if len(unique_dates) >= n_held_out else unique_dates
        inference_mask = np.isin(dates, inference_dates)

        remaining_mask = ~inference_mask
        remaining_dates = dates[remaining_mask]
        remaining_unique_dates = np.unique(remaining_dates)

        val_pct = self.config['VAL_SPLIT_PERCENTAGE']
        n_remaining = len(remaining_unique_dates)
        n_train_dates = int(n_remaining * (1 - val_pct))

        train_dates_threshold = remaining_unique_dates[n_train_dates] if n_train_dates > 0 else remaining_unique_dates[0]

        train_mask = remaining_mask & (dates < train_dates_threshold)
        val_mask = remaining_mask & (dates >= train_dates_threshold)

        X_train = X[train_mask]
        y_train_continuous = y[train_mask]
        dates_train = dates[train_mask]

        X_val = X[val_mask]
        y_val_continuous = y[val_mask]
        dates_val = dates[val_mask]

        X_inference = X[inference_mask]
        y_inference_continuous = y[inference_mask]
        dates_inference = dates[inference_mask]

        self.logger.log("[data split] Summary:", 'info')
        self.logger.log(f"  Total: {len(X):,} samples, {n_dates} dates", 'info')
        self.logger.log(f"  train: {len(X_train):,} samples ({train_mask.sum() / len(X):.1%}), dates < {train_dates_threshold}", 'info')
        self.logger.log(f"  validation: {len(X_val):,} samples ({val_mask.sum() / len(X):.1%}), dates >= {train_dates_threshold}", 'info')
        self.logger.log(f"  Inference: {len(X_inference):,} samples ({inference_mask.sum() / len(X):.1%}), date(s) = {inference_dates}", 'info')

        context.update({
            'X': X,
            'y': y,
            'dates': dates,
            'features': feature_names,
            'feature_names': feature_names,
            'data_stats': stats,
            'raw_target_values': raw_target_values if raw_target_values is not None else y,
            'raw_target_column': raw_target_column if raw_target_column is not None else -1,
            'phase1_complete': True,
            'X_train': X_train,
            'y_train_continuous': y_train_continuous,
            'dates_train': dates_train,
            'X_val': X_val,
            'y_val_continuous': y_val_continuous,
            'dates_val': dates_val,
            'X_inference': X_inference,
            'y_inference_continuous': y_inference_continuous,
            'dates_inference': dates_inference,
        })

        return context


# ============================================================================
# Section 3: TemporalWeighting (Phase 3)
# ============================================================================

class TemporalWeighting(BasePhase):
    """
    Phase 3: Generate temporal weights and extract temporal features.
    Checks USE_TEMPORAL_WEIGHTING toggle.
    """
    CONTEXT_CONSUMED = ['X', 'y', 'dates', 'phase1_complete']
    CONTEXT_PRODUCED = ['temporal_weights', 'temporal_features', 'phase3_complete']

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.data_manager = DataManager(config)

    def execute(self, context: Dict) -> Dict:
        if not self.config.get('USE_TEMPORAL_WEIGHTING', True):
            self.logger.log("Temporal weighting disabled (USE_TEMPORAL_WEIGHTING=False)", 'info')
            context['phase3_complete'] = True
            return context

        self._validate_input(context)

        X = context['X']
        y = context['y']
        dates = context['dates']

        self.logger.log(f"Processing {len(X)} samples with temporal weighting", 'info')

        temporal_features = self.data_manager.extract_temporal_features(dates)
        self.logger.log(f"Extracted {len(temporal_features)} temporal features", 'info')

        strategy_config = {
            'type': 'linear',
            'multiplier': self.config['TEMPORAL_MULTIPLIER']
        }

        temporal_weights = self.data_manager.apply_temporal_weighting_strategy(dates, strategy_config)

        self.logger.log(
            f"Temporal weights: min={temporal_weights.min():.3f}, "
            f"max={temporal_weights.max():.3f}, mean={temporal_weights.mean():.3f}",
            'info'
        )

        temporal_features['weights'] = temporal_weights

        self.data_manager.validate_temporal_features(dates, temporal_features)

        context.update({
            'temporal_weights': temporal_weights,
            'temporal_features': temporal_features,
            'phase3_complete': True
        })

        return context

    def _validate_input(self, context: Dict):
        if not context.get('phase1_complete'):
            raise ValueError("Phase 1 must complete before Phase 3")
        required = ['X', 'y', 'dates']
        for key in required:
            if key not in context:
                raise ValueError(f"Phase 3 missing required input: {key}")


# ============================================================================
# Section 4: FeatureImportance (Phase Xa)
# 6-method feature importance analysis with per-threshold pruning.
# The FeatureImportanceAnalyzer class is NOT preserved; its 6 analysis methods
# + _build_quick_dense + _compute_consolidated_ranking are converted to
# standalone helpers below. Report generation (generate_report/save_report)
# is dropped.
# ============================================================================

def _log_msg(logger, msg: str, level: str = 'info'):
    if logger is not None and hasattr(logger, 'log'):
        logger.log(msg, level)


def _build_quick_dense(config: Dict, input_dim: int, arch_key: str = 'Dense') -> Any:
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001))
    arch_config = config.get('FOCAL_LOSS_CONFIG', {}).get(arch_key, {})
    if arch_config.get('enabled', False):
        from models import FocalLoss
        clf_loss = FocalLoss(
            alpha=arch_config.get('alpha', 0.75),
            gamma=arch_config.get('gamma', 2.0)
        )
        model.compile(optimizer=opt, loss=clf_loss, metrics=['AUC'])
    else:
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['AUC'])
    return model


def _analyze_correlation(config: Dict, X: np.ndarray, y_raw: np.ndarray,
                         feature_names: List[str], logger=None) -> pd.DataFrame:
    results = []
    thresholds = config['CORRELATION_THRESHOLDS']

    for i, name in enumerate(feature_names):
        feature = X[:, i]
        row = {'feature': name, 'index': i}

        try:
            row['pearson_r'] = np.corrcoef(feature, y_raw)[0, 1]
        except Exception as e:
            _log_msg(logger, f"pearson_r failed for feature {name}: {e}", 'warning')
            row['pearson_r'] = 0.0

        try:
            row['spearman_r'], _ = spearmanr(feature, y_raw)
        except Exception as e:
            _log_msg(logger, f"spearman_r failed for feature {name}: {e}", 'warning')
            row['spearman_r'] = 0.0

        for t in thresholds:
            y_bin = (y_raw >= t).astype(int)
            if len(np.unique(y_bin)) < 2:
                row[f'pb_r_t{t}'] = 0.0
                row[f'auc_t{t}'] = 0.5
            else:
                try:
                    row[f'pb_r_t{t}'], _ = pointbiserialr(y_bin, feature)
                except Exception as e:
                    _log_msg(logger, f"pointbiserialr failed for feature {name}: {e}", 'warning')
                    row[f'pb_r_t{t}'] = 0.0
                try:
                    row[f'auc_t{t}'] = roc_auc_score(y_bin, feature)
                except Exception as e:
                    _log_msg(logger, f"roc_auc_score failed for feature {name}: {e}", 'warning')
                    row[f'auc_t{t}'] = 0.5

        results.append(row)

    df = pd.DataFrame(results)
    last_t = config['CORRELATION_THRESHOLDS'][-1]
    auc_col = f'auc_t{last_t}_abs'
    df['spearman_abs'] = df['spearman_r'].abs()
    df['pearson_abs'] = df['pearson_r'].abs()
    df[auc_col] = (df[f'auc_t{last_t}'] - 0.5).abs()
    df['rank'] = (
        df['spearman_abs'].rank(ascending=False) +
        df['pearson_abs'].rank(ascending=False) +
        df[auc_col].rank(ascending=False)
    )
    return df.sort_values('rank')


def _analyze_tree(config: Dict, X: np.ndarray, y_binary: np.ndarray,
                  feature_names: List[str]) -> pd.DataFrame:
    n_trees = config['TREE_ESTIMATORS']

    rf = RandomForestClassifier(
        n_estimators=n_trees, max_depth=10,
        min_samples_leaf=50, n_jobs=-1, random_state=42
    )
    rf.fit(X, y_binary)
    rf_imp = rf.feature_importances_

    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42
    )
    gbm.fit(X, y_binary)
    gbm_imp = gbm.feature_importances_

    combined_imp = (rf_imp + gbm_imp) / 2
    df = pd.DataFrame({
        'feature': feature_names,
        'index': range(len(feature_names)),
        'rf_importance': rf_imp,
        'gbm_importance': gbm_imp,
        'combined_importance': combined_imp,
    })
    df['rank'] = df['combined_importance'].rank(ascending=False)
    return df.sort_values('rank')


def _analyze_permutation(config: Dict, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         feature_names: List[str]) -> Tuple[pd.DataFrame, float]:
    import logging
    tf_logger = logging.getLogger('tensorflow')
    old_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)

    model = _build_quick_dense(config, X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)

    base_auc = roc_auc_score(y_val, model.predict(X_val, verbose=0).flatten())

    importance_scores = []
    n_repeats = config['PERMUTATION_REPEATS']

    for i, name in enumerate(feature_names):
        aucs = []
        for rep in range(n_repeats):
            X_rep = X_val.copy()
            rng = np.random.RandomState(42 + i * 100 + rep)
            rng.shuffle(X_rep[:, i])
            pred_perm = model.predict(X_rep, verbose=0).flatten()
            aucs.append(roc_auc_score(y_val, pred_perm))

        importance_scores.append({
            'feature': name,
            'index': i,
            'auc_drop': max(0, base_auc - np.mean(aucs)),
            'mean_permuted_auc': np.mean(aucs),
        })

    tf_logger.setLevel(old_level)
    df = pd.DataFrame(importance_scores)
    df['rank'] = df['auc_drop'].rank(ascending=False)
    del model
    return df.sort_values('rank'), base_auc


def _analyze_neural(config: Dict, X_train: np.ndarray, y_train: np.ndarray,
                    feature_names: List[str], trained_model: Any = None,
                    logger=None) -> pd.DataFrame:
    import logging
    import tensorflow as tf
    tf_logger = logging.getLogger('tensorflow')
    old_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)

    if trained_model is not None:
        model = trained_model
    else:
        model = _build_quick_dense(config, X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)

    weights = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.kernel is not None:
            if layer.kernel.shape[0] == len(feature_names):
                weights = np.abs(layer.kernel.numpy())
                break

    tf_logger.setLevel(old_level)

    if weights is None:
        _log_msg(logger, f"  warning: Could not extract input weights, using random", 'warning')
        return pd.DataFrame({
            'feature': feature_names,
            'index': range(len(feature_names)),
            'mean_abs_weight': np.random.rand(len(feature_names)),
            'rank': range(1, len(feature_names) + 1),
        })

    mean_abs = np.mean(weights, axis=1)
    df = pd.DataFrame({
        'feature': feature_names,
        'index': range(len(feature_names)),
        'mean_abs_weight': mean_abs,
    })
    df['rank'] = df['mean_abs_weight'].rank(ascending=False)
    return df.sort_values('rank')


def _analyze_shap(config: Dict, X_train: np.ndarray, y_train: np.ndarray,
                  feature_names: List[str], trained_model: Any = None,
                  logger=None) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        _log_msg(logger, f"  SHAP not available, using fallback", 'warning')
        return pd.DataFrame({
            'feature': feature_names,
            'index': range(len(feature_names)),
            'mean_abs_shap': np.random.rand(len(feature_names)),
            'rank': range(1, len(feature_names) + 1),
        })

    import logging
    tf_logger = logging.getLogger('tensorflow')
    old_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)

    if trained_model is not None:
        model = trained_model
    else:
        model = _build_quick_dense(config, X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)

    n_shap = config['SHAP_SAMPLE_SIZE']
    X_sample = X_train[:min(n_shap, len(X_train))]

    try:
        background = shap.sample(X_sample, min(1000, len(X_sample)), random_state=42)
        explainer = shap.GradientExplainer(model, background)

        X_shap = X_sample[:min(2000, len(X_sample))]
        shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    except Exception as e:
        _log_msg(logger, f"  SHAP failed: {e}, using fallback", 'warning')
        mean_abs_shap = np.random.rand(len(feature_names))

    tf_logger.setLevel(old_level)
    df = pd.DataFrame({
        'feature': feature_names,
        'index': range(len(feature_names)),
        'mean_abs_shap': mean_abs_shap,
    })
    df['rank'] = df['mean_abs_shap'].rank(ascending=False)
    return df.sort_values('rank')


def _analyze_ablation(config: Dict, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      feature_names: List[str], logger=None) -> pd.DataFrame:
    import logging
    tf_logger = logging.getLogger('tensorflow')
    old_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)

    results = []
    for i, name in enumerate(feature_names):
        X_tr = X_train[:, i:i+1].copy()
        X_va = X_val[:, i:i+1].copy()

        model = _build_quick_dense(config, 1)
        model.fit(X_tr, y_train, epochs=10, batch_size=256, verbose=0)

        pred = model.predict(X_va, verbose=0).flatten()
        try:
            auc = roc_auc_score(y_val, pred)
        except Exception as e:
            _log_msg(logger, f"ablation auc failed for feature {name}: {e}", 'warning')
            auc = 0.5

        results.append({'feature': name, 'index': i, 'auc': auc, 'pred_mean': np.mean(pred)})
        del model

    tf_logger.setLevel(old_level)
    df = pd.DataFrame(results)
    df['rank'] = df['auc'].rank(ascending=False)
    return df.sort_values('rank')


def _compute_consolidated_ranking(results: Dict, feature_names: List[str],
                                  active_methods: Optional[List[str]] = None) -> pd.DataFrame:
    if active_methods is None:
        active_methods = ['correlation', 'tree', 'permutation', 'neural', 'shap', 'ablation']

    all_ranks = pd.DataFrame({'feature': feature_names, 'index': range(len(feature_names))})

    for method in active_methods:
        df = results.get(method)
        if df is not None and 'rank' in df.columns:
            all_ranks[method] = all_ranks['index'].map(
                dict(zip(df['index'], df['rank']))
            )

    rank_cols_present = [m for m in active_methods if m in all_ranks.columns]
    all_ranks['mean_rank'] = all_ranks[rank_cols_present].mean(axis=1)
    all_ranks['consolidated_rank'] = all_ranks['mean_rank'].rank()

    return all_ranks.sort_values('consolidated_rank')


def run_feature_importance_analysis(config: Dict, X: np.ndarray, y_raw: np.ndarray,
                                    feature_names: List[str], logger=None) -> Dict[str, Any]:
    """Full 6-method per-threshold feature importance analysis (from run_full_analysis)."""
    start_time = __import__('time').time()
    results_by_threshold = {}
    n_features = X.shape[1]

    first_thresh = config['FIRST_THRESHOLD']
    last_thresh = config['LAST_THRESHOLD']
    thresh_step = config['THRESHOLD_STEP']
    thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)

    active_methods = [m for m, enabled in config.get('FEATURE_IMPORTANCE_METHODS', {}).items() if enabled]

    _log_msg(logger, f"Starting feature importance analysis")
    _log_msg(logger, f"Features: {n_features}, Samples: {X.shape[0]}, Active methods: {active_methods}")

    if not active_methods:
        _log_msg(logger, f"  No active methods — keeping all features, skipping analysis", 'warning')
        return {
            'results': {},
            'results_by_threshold': {},
            'kept_indices': list(range(n_features)),
            'dropped_indices': [],
            'kept_names': feature_names,
            'dropped_names': [],
            'corr_matrix': np.corrcoef(X.T) if X.shape[1] > 1 else np.array([[1.0]]),
            'timings': {},
            'n_features_original': n_features,
            'n_features_pruned': n_features,
            'thresholds': thresholds.tolist(),
        }

    for thresh in thresholds:
        y_binary = (y_raw >= thresh).astype(int)
        X_train, X_val, y_train_bin, y_val_bin = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )

        results = {}

        if 'correlation' in active_methods:
            t0 = __import__('time').time()
            results['correlation'] = _analyze_correlation(config, X, y_raw, feature_names, logger)
            results['correlation_timing'] = __import__('time').time() - t0
        else:
            results['correlation'] = None
            results['correlation_timing'] = 0

        if 'tree' in active_methods:
            t0 = __import__('time').time()
            results['tree'] = _analyze_tree(config, X, y_binary, feature_names)
            results['tree_timing'] = __import__('time').time() - t0
        else:
            results['tree'] = None
            results['tree_timing'] = 0

        if 'permutation' in active_methods:
            t0 = __import__('time').time()
            results['permutation'], results['permutation_baseline_auc'] = _analyze_permutation(
                config, X_train, y_train_bin, X_val, y_val_bin, feature_names
            )
            results['permutation_timing'] = __import__('time').time() - t0
        else:
            results['permutation'] = None
            results['permutation_timing'] = 0

        if 'neural' in active_methods:
            t0 = __import__('time').time()
            results['neural'] = _analyze_neural(config, X_train, y_train_bin, feature_names, None, logger)
            results['neural_timing'] = __import__('time').time() - t0
        else:
            results['neural'] = None
            results['neural_timing'] = 0

        if 'shap' in active_methods:
            t0 = __import__('time').time()
            results['shap'] = _analyze_shap(config, X_train, y_train_bin, feature_names, None, logger)
            results['shap_timing'] = __import__('time').time() - t0
        else:
            results['shap'] = None
            results['shap_timing'] = 0

        if 'ablation' in active_methods:
            t0 = __import__('time').time()
            results['ablation'] = _analyze_ablation(config, X_train, y_train_bin, X_val, y_val_bin, feature_names, logger)
            results['ablation_timing'] = __import__('time').time() - t0
        else:
            results['ablation'] = None
            results['ablation_timing'] = 0

        consolidated = _compute_consolidated_ranking(results, feature_names, active_methods)
        results['consolidated'] = consolidated

        prune_pct = config['FEATURE_PRUNE_PERCENTILE']
        n_keep = max(1, int(n_features * (100 - prune_pct) / 100))
        kept_indices = consolidated.head(n_keep)['index'].tolist()
        dropped_indices = [i for i in range(n_features) if i not in kept_indices]
        results['dropped_indices'] = dropped_indices
        results['kept_indices'] = kept_indices

        pos_rate = y_binary.mean()
        log_checks = [
            (1, "Statistical_Correlation", 'correlation', 'spearman_abs'),
            (2, "Tree_Based_Importance", 'tree', 'combined_importance'),
            (3, "Permutation_Importance", 'permutation', 'auc_drop'),
            (4, "Neural_Weight_Analysis", 'neural', 'mean_abs_weight'),
            (5, "SHAP_Values", 'shap', 'mean_abs_shap'),
            (6, "Ablation_Study", 'ablation', 'auc'),
        ]
        for num, name, key, score_col in log_checks:
            if key in active_methods:
                extra = f" baseline_auc={results['permutation_baseline_auc']:.4f}" if key == 'permutation' else ""
                _log_msg(logger, _format_all_features(num, name, results[key], score_col, dropped_indices, results[f'{key}_timing'], thresh, pos_rate, extra=extra))

        kept_df = consolidated[~consolidated['index'].isin(dropped_indices)]
        parts = [f"rank={int(r['consolidated_rank'])} {r['feature']}={r['mean_rank']:.2f}" for _, r in kept_df.iterrows()]
        _log_msg(logger, f"  consolidated_ranking label_threshold={thresh:.1f} signal_rate={pos_rate:.6f} kept={len(kept_df)} kept_total={len(feature_names)} {' '.join(parts)}")

        results_by_threshold[thresh] = results
        pruned_df = consolidated[consolidated['index'].isin(dropped_indices)]
        parts = [f"rank={int(r['consolidated_rank'])} {r['feature']}={r['mean_rank']:.2f}" for _, r in pruned_df.iterrows()]
        _log_msg(logger, f"  consolidated_pruning label_threshold={thresh:.1f} signal_rate={pos_rate:.6f} pruned={len(dropped_indices)} pruned_total={n_features} {' '.join(parts)}")

    n_thresh = len(thresholds)
    drop_counts = {name: sum(1 for t, r in results_by_threshold.items()
                             if i in r['dropped_indices'])
                   for i, name in enumerate(feature_names)}
    always_pruned = sorted([f for f, c in drop_counts.items() if c == n_thresh])
    never_pruned = sorted([f for f, c in drop_counts.items() if c == 0])
    borderline = {f: c for f, c in sorted(drop_counts.items()) if 0 < c < n_thresh}
    _log_msg(logger, f"cross_threshold always_pruned={' ,'.join(always_pruned)}")
    _log_msg(logger, f"cross_threshold never_pruned={' ,'.join(never_pruned)}")
    _log_msg(logger, f"cross_threshold borderline={' ,'.join(f'{k}:{v}' for k,v in sorted(borderline.items()))}")

    first_thresh = thresholds[0]
    results = results_by_threshold[first_thresh]

    _log_msg(logger, f"Computing feature correlation matrix")
    corr_matrix = np.corrcoef(X.T) if X.shape[1] > 1 else np.array([[1.0]])

    total_time = __import__('time').time() - start_time
    timings = {'total': total_time}
    for key in ['correlation', 'tree', 'permutation', 'neural', 'shap', 'ablation']:
        timings[key] = results.get(f'{key}_timing', 0)

    _log_msg(logger, f"total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    _log_msg(logger, f"features: {n_features} total -> {len(results['kept_indices'])} kept, {len(results['dropped_indices'])} pruned")
    _log_msg(logger, f"dropped: {[feature_names[i] for i in results['dropped_indices']]}")

    return {
        'results': results,
        'results_by_threshold': results_by_threshold,
        'kept_indices': results['kept_indices'],
        'dropped_indices': results['dropped_indices'],
        'kept_names': [feature_names[i] for i in results['kept_indices']],
        'dropped_names': [feature_names[i] for i in results['dropped_indices']],
        'corr_matrix': corr_matrix,
        'timings': timings,
        'n_features_original': n_features,
        'n_features_pruned': n_keep,
        'thresholds': thresholds.tolist(),
    }


def _format_all_features(method_num: int, method_name: str, df: pd.DataFrame, score_col: str,
                         dropped_indices: List[int], timing: float, label_threshold: float,
                         pos_rate: float, extra: str = "") -> str:
    """Format the per-method feature log line (from FeatureImportanceAnalyzer._log_all_features)."""
    lines = []
    total_kept = int(len(df)) - len(dropped_indices)
    for _, row in df.iterrows():
        status = 'dropped' if int(row['index']) in dropped_indices else 'kept'
        lines.append(
            f"  method_{method_num}_{method_name} LABEL_THRESHOLD={label_threshold:.1f} "
            f"signal_rate={pos_rate:.6f} feature={row['feature']} score={row[score_col]:.6f} "
            f"rank={int(row['rank'])} status={status}"
        )
    return f"method_{method_num}_{method_name} LABEL_THRESHOLD={label_threshold:.1f} signal_rate={pos_rate:.6f} timing={timing:.1f}s kept={total_kept} dropped={len(dropped_indices)} total={len(df)}{extra}\n" + "\n".join(lines)


class FeatureImportance(BasePhase):
    """
    Phase Xa: 6-method feature importance analysis with per-threshold pruning.
    Checks FEATURE_ANALYSIS_ENABLED toggle.
    """
    CONTEXT_CONSUMED = ['X', 'raw_target_values', 'y', 'feature_names']
    CONTEXT_PRODUCED = ['threshold_kept_indices', 'threshold_dropped_indices',
                        'pruned_feature_indices', 'feature_importance_results',
                        'phaseXa_complete']

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.name = "Phase Xa: Raw Feature Importance Analysis"

    def execute(self, context: Dict) -> Dict:
        if not self.config.get('FEATURE_ANALYSIS_ENABLED', True):
            self.logger.log(f"{self.name} - Skipped (FEATURE_ANALYSIS_ENABLED=False)", 'info')
            return context

        method_config = self.config.get('FEATURE_IMPORTANCE_METHODS', {})
        active_methods = [m for m, enabled in method_config.items() if enabled]

        self.logger.log(f"{self.name} - Starting with methods: {active_methods} ({len(active_methods)}/{len(method_config)} active)", 'info')

        X = context.get('X')
        y_raw = context.get('raw_target_values')
        if y_raw is None:
            y_raw = context.get('y')
        feature_names = context.get('feature_names')

        if X is None or y_raw is None:
            self.logger.log("error: X or y not found in context", 'error')
            return context

        n_features = X.shape[1]
        n_samples = X.shape[0]
        sample_size = self.config['FEATURE_ANALYSIS_SAMPLE_SIZE']

        if n_samples > sample_size:
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

        self.logger.log(f"Analyzing {n_features} raw features on {X_sub.shape[0]} samples", 'info')

        analysis_results = run_feature_importance_analysis(
            self.config, X_sub, y_sub, feature_names, self.logger
        )

        # Report file output is dropped (per design); pruning results stored in context
        results_by_threshold = analysis_results.get('results_by_threshold', {})
        threshold_kept = {round(float(t), 1): r['kept_indices'] for t, r in results_by_threshold.items()}
        threshold_dropped = {round(float(t), 1): r['dropped_indices'] for t, r in results_by_threshold.items()}

        context['X'] = X
        context['feature_names'] = feature_names
        context['feature_importance_results'] = analysis_results
        context['threshold_kept_indices'] = threshold_kept
        context['threshold_dropped_indices'] = threshold_dropped
        pruned_indices = analysis_results['kept_indices']
        context['pruned_feature_indices'] = pruned_indices
        context['dropped_feature_indices'] = analysis_results['dropped_indices']
        context['dropped_feature_names'] = analysis_results['dropped_names']

        context['phaseXa_complete'] = True

        n_pruned = n_features - len(pruned_indices)
        self.logger.log(f"complete: {n_features} raw features (per-threshold pruning stored)", 'info')
        if threshold_kept:
            first_thr = next(iter(threshold_kept.keys()))
            self.logger.log(f"  First threshold ({first_thr}): {len(pruned_indices)} kept", 'info')

        return context


# ============================================================================
# Section 5: FeaturePruning (Phase BE)
# Per-architecture backward feature elimination.
# ============================================================================

PROXY_REGISTRY: Dict[str, Dict[str, Callable]] = {}


def register_xgboost_proxy():
    from xgboost import XGBClassifier

    def train_proxy(X, y):
        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=1,
        )
        model.fit(X, y)
        return model

    PROXY_REGISTRY['XGBoost'] = {
        'train': train_proxy,
        'importance': lambda m: m.feature_importances_,
        'predict': lambda m, X: m.predict_proba(X)[:, 1],
    }


def register_lightgbm_proxy():
    import lightgbm as lgb

    def train_proxy(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=100, num_leaves=31, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1, n_jobs=1,
        )
        model.fit(X, y)
        return model

    PROXY_REGISTRY['LightGBM'] = {
        'train': train_proxy,
        'importance': lambda m: m.feature_importances_,
        'predict': lambda m, X: m.predict_proba(X)[:, 1],
    }


def register_catboost_proxy():
    from catboost import CatBoostClassifier

    def train_proxy(X, y):
        model = CatBoostClassifier(
            iterations=100, depth=5, learning_rate=0.1,
            auto_class_weights='Balanced',
            random_seed=42, verbose=0,
        )
        model.fit(X, y)
        return model

    PROXY_REGISTRY['CatBoost'] = {
        'train': train_proxy,
        'importance': lambda m: m.get_feature_importance(),
        'predict': lambda m, X: m.predict_proba(X)[:, 1],
    }


def register_generic_nn_proxy():
    from sklearn.ensemble import RandomForestClassifier

    def train_proxy(X, y):
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=1,
        )
        model.fit(X, y)
        return model

    generic_spec = {
        'train': train_proxy,
        'importance': lambda m: m.feature_importances_,
        'predict': lambda m, X: m.predict_proba(X)[:, 1],
    }
    for arch_name in ['Dense', 'CNN', 'RNN', 'LSTM', 'VAE', 'Transformer']:
        PROXY_REGISTRY[arch_name] = generic_spec


register_xgboost_proxy()
register_lightgbm_proxy()
register_catboost_proxy()
register_generic_nn_proxy()


def resolve_feature_indices(threshold_kept_indices, arch_name, threshold_key, fallback_range):
    """Resolve kept feature indices for a given arch+threshold, with flat/legacy fallback."""
    if not threshold_kept_indices:
        return list(fallback_range)
    first_val = list(threshold_kept_indices.values())[0]
    if isinstance(first_val, dict):
        arch_dict = threshold_kept_indices.get(arch_name, {})
        kept = arch_dict.get(threshold_key)
        if kept is not None:
            return list(kept)
        return list(fallback_range)
    kept = threshold_kept_indices.get(threshold_key)
    if kept is not None:
        return list(kept)
    return list(fallback_range)


class FeaturePruning(BasePhase):
    """
    Phase BE: Per-architecture backward feature elimination.
    Checks BACKWARD_ELIMINATION_ENABLED toggle.
    """
    CONTEXT_CONSUMED = ['X', 'raw_target_values', 'y',
                        'threshold_kept_indices', 'feature_names']
    CONTEXT_PRODUCED = ['threshold_kept_indices', 'phaseBE_complete']

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.name = "Phase BE: Backward Elimination"

    def execute(self, context: Dict) -> Dict:
        if not self.config.get('BACKWARD_ELIMINATION_ENABLED', False):
            self.logger.log(f"{self.name} — Skipped (BACKWARD_ELIMINATION_ENABLED=False)", 'info')
            return context

        X = context.get('X')
        if X is None:
            self.logger.log(f"{self.name} — error: X not found in context", 'error')
            return context

        y_raw = context.get('raw_target_values')
        if y_raw is None:
            y_raw = context.get('y')
        if y_raw is None:
            self.logger.log(f"{self.name} — error: y not found in context", 'error')
            return context

        architectures = self.config.get('ACTIVE_ARCHITECTURES', []) or (
            self.config.get('NEURAL_ARCHITECTURES', []) + self.config.get('TREE_ARCHITECTURES', [])
        )
        first_threshold = self.config['FIRST_THRESHOLD']
        last_threshold = self.config['LAST_THRESHOLD']
        threshold_step = self.config['THRESHOLD_STEP']
        thresholds = np.arange(first_threshold, last_threshold + threshold_step, threshold_step)
        n_features = X.shape[1]

        train_epochs = self.config.get('BE_PROXY_TRAIN_EPOCHS', 10)
        stratify_ratio = self.config.get('BE_STRATIFY_SPLIT_RATIO', 0.20)
        elimination_steps = self.config.get('BE_ELIMINATION_STEPS', 0.50)
        min_features = self.config.get('BE_MIN_FEATURES', 10)
        tolerance = self.config.get('BE_TOLERANCE', 0.01)

        legacy_flat = context.get('threshold_kept_indices', {})
        per_arch_result: Dict[str, Dict[float, List[int]]] = {}

        self.logger.log(f"{self.name} — Starting per-architecture elimination for {len(architectures)} archs, "
                        f"{n_features} features, {len(thresholds)} thresholds", 'info')

        for arch_name in architectures:
            proxy_spec = PROXY_REGISTRY.get(arch_name)
            if proxy_spec is None:
                self.logger.log(f"{self.name} — {arch_name}: no proxy registered, skipping", 'info')
                continue

            self.logger.log(f"{self.name} — {arch_name}: starting elimination", 'info')

            arch_threshold_results: Dict[float, List[int]] = {}

            for thresh in thresholds:
                y_binary = (y_raw >= thresh).astype(int)

                pos_count = int(y_binary.sum())
                if pos_count < 2:
                    self.logger.log(f"{self.name} — {arch_name} @ LT={thresh}: only {pos_count} positives, using all features", 'info')
                    arch_threshold_results[thresh] = list(range(n_features))
                    continue

                sss = StratifiedShuffleSplit(n_splits=1, test_size=stratify_ratio, random_state=42)
                train_idx, val_idx = next(sss.split(X, y_binary))
                X_proxy_train, X_proxy_val = X[train_idx], X[val_idx]
                y_proxy_train, y_proxy_val = y_binary[train_idx], y_binary[val_idx]

                current_indices = list(range(n_features))
                best_val_prec = 0.0

                elimination_round = 0
                while len(current_indices) > min_features:
                    elimination_round += 1
                    X_sub_train = X_proxy_train[:, current_indices]
                    X_sub_val = X_proxy_val[:, current_indices]

                    model = proxy_spec['train'](X_sub_train, y_proxy_train)
                    importances = proxy_spec['importance'](model)

                    sorted_idx = np.argsort(importances)[::-1]
                    n_drop = max(1, int(len(sorted_idx) * elimination_steps))
                    keep_idx = sorted_idx[:-n_drop] if len(sorted_idx) - n_drop >= min_features else sorted_idx[:min_features]
                    candidate_indices = [current_indices[i] for i in keep_idx]

                    y_pred_proba = proxy_spec['predict'](model, X_sub_val)
                    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

                    val_prec = precision_score(y_proxy_val, y_pred_binary, zero_division=0)

                    if val_prec >= best_val_prec * (1 - tolerance):
                        best_val_prec = val_prec
                        current_indices = candidate_indices
                        self.logger.log(f"{self.name} — {arch_name} @ LT={thresh} round {elimination_round}: "
                                        f"{len(current_indices)} features kept, val_prec={val_prec:.4f}", 'info')
                    else:
                        self.logger.log(f"{self.name} — {arch_name} @ LT={thresh} round {elimination_round}: "
                                        f"stopped (val_prec {val_prec:.4f} < best {best_val_prec:.4f} × {1-tolerance:.4f})", 'info')
                        break

                arch_threshold_results[thresh] = sorted(current_indices)

            per_arch_result[arch_name] = arch_threshold_results
            total_kept = sum(len(v) for v in arch_threshold_results.values())
            self.logger.log(f"{self.name} — {arch_name}: completed ({total_kept} total feature-kept entries across thresholds)", 'info')

        context['threshold_kept_indices'] = per_arch_result
        context['phaseBE_complete'] = True

        self.logger.log(f"{self.name} — complete: {len(per_arch_result)} architectures processed", 'info')

        return context


# ============================================================================
# Section 6: TemporalPrecisionGap (Phase Xb)
# ============================================================================

class TemporalPrecisionGap(BasePhase):
    """
    Phase Xb: Compare precision on recent vs older validation dates.
    """
    # temporal_weights removed from CONTEXT_CONSUMED — fetched but never used.
    CONTEXT_CONSUMED = ['val_predictions', 'val_dates', 'val_y_raw',
                        'arch_names', 'optimal_thresholds',
                        'phase4_complete']
    CONTEXT_PRODUCED = ['temporal_precision_gap', 'best_recency_architecture',
                        'worst_recency_architecture', 'phaseXb_complete']

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = Logger(config)
        self.name = "Phase Xb: Temporal Precision Gap Analysis"

    def execute(self, context: Dict) -> Dict:
        self.logger.log(f"{self.name} - Starting", 'info')

        val_predictions = context.get('val_predictions')
        val_dates = context.get('val_dates')
        val_y_raw = context.get('val_y_raw')
        arch_names = context.get('arch_names', [])

        if val_predictions is None or len(val_predictions) == 0:
            self.logger.log("No validation predictions found in context", 'warning')
            self.logger.log("Phase 4 must store val_predictions for this analysis", 'warning')
            self.logger.log(f"{self.name} - SKIPPED", 'warning')
            context['phaseXb_complete'] = True
            return context

        if val_dates is None or val_y_raw is None:
            self.logger.log("Validation dates or target values not found", 'warning')
            self.logger.log(f"{self.name} - SKIPPED", 'warning')
            context['phaseXb_complete'] = True
            return context

        if len(arch_names) == 0:
            self.logger.log("No architecture names found", 'warning')
            self.logger.log(f"{self.name} - SKIPPED", 'warning')
            context['phaseXb_complete'] = True
            return context

        opt_thresholds = context.get('optimal_thresholds', [self.config['FIRST_THRESHOLD']])
        label_threshold = float(opt_thresholds[0]) if opt_thresholds else self.config['FIRST_THRESHOLD']
        pred_threshold = self.config['PREDICTION_THRESHOLD']

        # PRIORITY 1 FIX: Validate and Re-derive Dimensions
        if val_dates is not None and val_y_raw is not None:
            if len(val_dates) != len(val_y_raw):
                self.logger.log(f"warning: Dimension mismatch detected! val_dates: {len(val_dates)}, val_y_raw: {len(val_y_raw)}", 'warning')
                min_len = min(len(val_dates), len(val_y_raw))
                val_dates = val_dates[:min_len]
                val_y_raw = val_y_raw[:min_len]
                val_predictions = [p[:min_len] for p in val_predictions]
                self.logger.log(f"info: Aligned to {min_len} elements", 'info')

        self.logger.log(f"Analyzing temporal precision gap for {len(arch_names)} architectures", 'info')
        self.logger.log(f"Label threshold: {label_threshold}, Prediction threshold: {pred_threshold}", 'info')

        unique_dates = np.unique(val_dates)
        n_dates = len(unique_dates)

        if n_dates < 3:
            self.logger.log(f"warning: Only {n_dates} unique dates in validation - splitting may be unreliable", 'warning')

        tail_n_days = self.config['TEMPORAL_GAP_N_DAYS']
        tail_frac = self.config['TEMPORAL_GAP_TAIL_FRACTION']

        if tail_n_days > 0:
            n_tail = min(tail_n_days, n_dates // 2)
            is_recent = np.isin(val_dates, unique_dates[-n_tail:])
            is_older = np.isin(val_dates, unique_dates[:n_tail])
        else:
            n_tail = max(1, int(n_dates * tail_frac))
            recent_cutoff = unique_dates[int(n_dates * (1 - tail_frac))]
            older_cutoff = unique_dates[int(n_dates * tail_frac)]
            is_recent = np.isin(val_dates, unique_dates[unique_dates >= recent_cutoff])
            is_older = np.isin(val_dates, unique_dates[unique_dates <= older_cutoff])

        signal_mask = (val_y_raw >= label_threshold).astype(int)

        n_recent_signal_total = int(np.sum(signal_mask[is_recent]))
        n_older_signal_total = int(np.sum(signal_mask[is_older]))

        self.logger.log(f"Validation date split: {n_dates} unique dates, tail={n_tail} days each (N_DAYS={tail_n_days}, FRACTION={tail_frac:.0%})", 'info')
        self.logger.log(f"Validation Recent dates (newest {n_tail} days, {unique_dates[-n_tail]}..{unique_dates[-1]}): {np.sum(is_recent):,} samples, {n_recent_signal_total:,} signal cases", 'info')
        self.logger.log(f"Validation Older dates (oldest {n_tail} days, {unique_dates[0]}..{unique_dates[n_tail-1]}): {np.sum(is_older):,} samples, {n_older_signal_total:,} signal cases", 'info')

        results = []
        for i, (arch_name, preds) in enumerate(zip(arch_names, val_predictions)):
            preds = np.asarray(preds).flatten()

            recent_signal = signal_mask[is_recent] == 1
            older_signal = signal_mask[is_older] == 1

            recent_preds = (preds[is_recent] >= pred_threshold).astype(int)
            older_preds = (preds[is_older] >= pred_threshold).astype(int)

            n_rs = int(np.sum(recent_signal))
            n_os = int(np.sum(older_signal))

            recent_tp = int(np.sum((recent_preds == 1) & (recent_signal == 1)))
            recent_fp = int(np.sum((recent_preds == 1) & (recent_signal == 0)))
            recent_tn = int(np.sum((recent_preds == 0) & (recent_signal == 0)))
            recent_fn = int(np.sum((recent_preds == 0) & (recent_signal == 1)))

            recent_precision = recent_tp / (recent_tp + recent_fp) if (recent_tp + recent_fp) > 0 else 0.0
            recent_recall = recent_tp / (recent_tp + recent_fn) if (recent_tp + recent_fn) > 0 else 0.0
            recent_f1 = 2 * recent_precision * recent_recall / (recent_precision + recent_recall) if (recent_precision + recent_recall) > 0 else 0.0

            older_tp = int(np.sum((older_preds == 1) & (older_signal == 1)))
            older_fp = int(np.sum((older_preds == 1) & (older_signal == 0)))
            older_tn = int(np.sum((older_preds == 0) & (older_signal == 0)))
            older_fn = int(np.sum((older_preds == 0) & (older_signal == 1)))

            older_precision = older_tp / (older_tp + older_fp) if (older_tp + older_fp) > 0 else 0.0
            older_recall = older_tp / (older_tp + older_fn) if (older_tp + older_fn) > 0 else 0.0
            older_f1 = 2 * older_precision * older_recall / (older_precision + older_recall) if (older_precision + older_recall) > 0 else 0.0

            try:
                from sklearn.metrics import roc_auc_score
                recent_auc = roc_auc_score(signal_mask[is_recent], preds[is_recent]) if len(np.unique(signal_mask[is_recent])) > 1 else 0.5
                older_auc = roc_auc_score(signal_mask[is_older], preds[is_older]) if len(np.unique(signal_mask[is_older])) > 1 else 0.5
            except Exception:
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
                'n_recent_signal': n_rs,
                'n_older_signal': n_os,
                'interpretation': interpretation,
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('gap', ascending=False)

        middle_n = n_dates - 2 * n_tail
        status_lines = []
        status_lines.append(f"Temporal Precision Gap (validation set): Recent = newest {n_tail} days ({unique_dates[-n_tail]}..{unique_dates[-1]}), Older = oldest {n_tail} days ({unique_dates[0]}..{unique_dates[n_tail-1]}), middle {middle_n} days excluded as buffer")
        status_lines.append(f"{self.name} Report - FULL METRICS")
        status_lines.append(f"{'Architecture':<12} {'validation_recent_precision':>8} {'validation_recent_recall':>8} {'validation_recent_auc':>10} {'validation_recent_f1':>8} {'validation_older_precision':>8} {'validation_older_recall':>8} {'validation_older_auc':>10} {'validation_older_f1':>8} {'Gap':>6}")

        for _, row in results_df.iterrows():
            gap_str = f"{row['gap']:+.2f}"
            status_lines.append(f"{row['architecture']:<12} validation_recent_precision={row['recent_precision']:>8.4f} validation_recent_recall={row['recent_recall']:>8.4f} validation_recent_auc={row['recent_auc']:>10.4f} validation_recent_f1={row['recent_f1']:>8.4f} validation_older_precision={row['older_precision']:>8.4f} validation_older_recall={row['older_recall']:>8.4f} validation_older_auc={row['older_auc']:>10.4f} validation_older_f1={row['older_f1']:>8.4f} Gap={gap_str:>6}")

        for line in status_lines:
            self.logger.log(line, 'info')

        best_arch = results_df.iloc[0]
        worst_arch = results_df.iloc[-1]

        self.logger.log(f"BEST for recent signals: {best_arch['architecture']} (Gap: {best_arch['gap']:+.4f})", 'info')
        self.logger.log(f"WORST for recent signals: {worst_arch['architecture']} (Gap: {worst_arch['gap']:+.4f})", 'info')

        positive_gap_count = int(np.sum(results_df['gap'] > 0.05))
        self.logger.log(f"{positive_gap_count}/{len(results_df)} architectures show positive temporal precision gap", 'info')

        if positive_gap_count == 0:
            self.logger.log(f"warning: No architecture shows meaningful improvement on recent signals", 'warning')
            self.logger.log(f"Consider: stronger temporal weighting or different architectures", 'warning')

        context['temporal_precision_gap'] = results_df.to_dict('records')
        context['best_recency_architecture'] = best_arch['architecture']
        context['worst_recency_architecture'] = worst_arch['architecture']
        context['phaseXb_complete'] = True

        self.logger.log(f"{self.name} - COMPLETE", 'info')

        return context
