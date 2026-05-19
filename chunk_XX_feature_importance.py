"""
Chunk XX: Feature Importance Analysis
Multi-method feature importance analysis with auto-pruning
Uses 6 methods: Correlation, Tree, Permutation, Neural, SHAP, Ablation
"""

import os
import sys
import time
import warnings
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import spearmanr, pointbiserialr

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

CONFIG_FEATURE_ANALYSIS = {
    'FEATURE_ANALYSIS_SAMPLE_SIZE': 100000,
    'FEATURE_PRUNE_PERCENTILE': 20,
    'ABLATON_THRESHOLD': 2.0,
    'CORRELATION_THRESHOLDS': [0.0, 0.5, 1.0, 2.0],
    'TREE_ESTIMATORS': 200,
    'PERMUTATION_REPEATS': 5,
    'SHAP_SAMPLE_SIZE': 5000,
}


class FeatureImportanceAnalyzer:
    def __init__(self, config: Optional[Dict] = None, logger=None):
        self.config = {**CONFIG_FEATURE_ANALYSIS, **(config or {})}
        self.results = {}
        self.timings = {}
        self.logger = logger

    def _log(self, msg: str, level: str = 'info'):
        if self.logger:
            self.logger.log(msg, level)
        else:
            print(msg)

    def _log_all_features(self, method_prefix: str, df: pd.DataFrame, score_col: str, dropped_indices: List[int], timing: float, label_threshold: float, pos_rate: float, extra: str = ""):
        kept_df = df[~df['index'].isin(dropped_indices)].sort_values(score_col, ascending=False)
        parts = [f"{row['feature']}={row[score_col]:.6f}" for _, row in kept_df.iterrows()]
        self._log(f"  {method_prefix} (Label_Threshold={label_threshold:.1f}, +{pos_rate:.3%}{extra}, kept {len(kept_df)}/{len(df)}) Done in {timing:.1f}s: {' | '.join(parts)}")
    
    def run_full_analysis(
        self,
        X: np.ndarray,
        y_raw: np.ndarray,
        feature_names: List[str],
        dates: Optional[np.ndarray] = None,
        temporal_weights: Optional[np.ndarray] = None,
        trained_dense_model: Any = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        results_by_threshold = {}  # Dict indexed by threshold
        n_features = X.shape[1]
        
        # Get thresholds from config (synchronized with Phase 4)
        first_thresh = self.config.get('FIRST_THRESHOLD', 20.0)
        last_thresh = self.config.get('LAST_THRESHOLD', 0.0)
        thresh_step = self.config.get('THRESHOLD_STEP', -2.0)
        thresholds = np.arange(first_thresh, last_thresh + thresh_step, thresh_step)
        
        self._log(f"Starting 6-method feature importance analysis")
        self._log(f"Features: {n_features}, Samples: {X.shape[0]}")
        self._log(f"Running at {len(thresholds)} Label_Thresholds: {first_thresh} to {last_thresh}")
        
        for thresh in thresholds:
            y_binary = (y_raw >= thresh).astype(int)
            X_train, X_val, y_train_bin, y_val_bin = train_test_split(
                X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
            )
            
            results = {}  # Results for this threshold
            
            # Method 1: Statistical Correlation
            t0 = time.time()
            results['correlation'] = self._analyze_correlation(X, y_raw, feature_names)
            results['correlation_timing'] = time.time() - t0
            
            # Method 2: Tree-Based Importance
            t0 = time.time()
            results['tree'] = self._analyze_tree(X, y_binary, feature_names)
            results['tree_timing'] = time.time() - t0
            
            # Method 3: Permutation Importance
            t0 = time.time()
            results['permutation'], results['permutation_baseline_auc'] = self._analyze_permutation(
                X_train, y_train_bin, X_val, y_val_bin, feature_names
            )
            results['permutation_timing'] = time.time() - t0
            
            # Method 4: Neural Weight Analysis
            t0 = time.time()
            results['neural'] = self._analyze_neural(X_train, y_train_bin, feature_names, trained_dense_model)
            results['neural_timing'] = time.time() - t0
            
            # Method 5: SHAP Values
            t0 = time.time()
            results['shap'] = self._analyze_shap(X_train, y_train_bin, feature_names, trained_dense_model)
            results['shap_timing'] = time.time() - t0
            
            # Method 6: Ablation Study
            t0 = time.time()
            results['ablation'] = self._analyze_ablation(X_train, y_train_bin, X_val, y_val_bin, feature_names)
            results['ablation_timing'] = time.time() - t0
            
            # Compute consolidated ranking
            consolidated = self._compute_consolidated_ranking(results, feature_names)
            results['consolidated'] = consolidated
            
            # Determine pruned features
            prune_pct = self.config['FEATURE_PRUNE_PERCENTILE']
            n_keep = max(1, int(n_features * (100 - prune_pct) / 100))
            kept_indices = consolidated.head(n_keep)['index'].tolist()
            dropped_indices = [i for i in range(n_features) if i not in kept_indices]
            results['dropped_indices'] = dropped_indices
            results['kept_indices'] = kept_indices
            
            # Per-method all-feature logs (excluding pruned)
            pos_rate = y_binary.mean()
            self._log_all_features("Method 1/6: Statistical Correlation", results['correlation'], 'spearman_abs', dropped_indices, results['correlation_timing'], thresh, pos_rate)
            self._log_all_features("Method 2/6: Tree-Based Importance", results['tree'], 'combined_importance', dropped_indices, results['tree_timing'], thresh, pos_rate)
            self._log_all_features("Method 3/6: Permutation Importance", results['permutation'], 'auc_drop', dropped_indices, results['permutation_timing'], thresh, pos_rate, extra=f", baseline AUC={results['permutation_baseline_auc']:.4f}")
            self._log_all_features("Method 4/6: Neural Weight Analysis", results['neural'], 'mean_abs_weight', dropped_indices, results['neural_timing'], thresh, pos_rate)
            self._log_all_features("Method 5/6: SHAP Values", results['shap'], 'mean_abs_shap', dropped_indices, results['shap_timing'], thresh, pos_rate)
            self._log_all_features("Method 6/6: Ablation Study", results['ablation'], 'auc', dropped_indices, results['ablation_timing'], thresh, pos_rate)
            
            # Consolidated ranking of all kept features
            kept_df = consolidated[~consolidated['index'].isin(dropped_indices)]
            parts = [f"#{int(r['consolidated_rank'])} {r['feature']}={r['mean_rank']:.2f}" for _, r in kept_df.iterrows()]
            self._log(f"  Consolidated ranking (Label_Threshold={thresh:.1f}, +{pos_rate:.3%}, kept {len(kept_df)}/{len(feature_names)}): {' | '.join(parts)}")
            
            results_by_threshold[thresh] = results
            pruned_df = consolidated[consolidated['index'].isin(dropped_indices)]
            parts = [f"#{int(r['consolidated_rank'])} {r['feature']}={r['mean_rank']:.2f}" for _, r in pruned_df.iterrows()]
            self._log(f"  Consolidated pruning (Label_Threshold={thresh:.1f}, +{pos_rate:.3%}, pruned {len(dropped_indices)}/{n_features}): {' | '.join(parts)}")
        
        # Cross-threshold stability summary
        n_thresh = len(thresholds)
        drop_counts = {name: sum(1 for t, r in results_by_threshold.items()
                                 if i in r['dropped_indices'])
                       for i, name in enumerate(feature_names)}
        always_pruned = sorted([f for f, c in drop_counts.items() if c == n_thresh])
        never_pruned = sorted([f for f, c in drop_counts.items() if c == 0])
        borderline = {f: c for f, c in sorted(drop_counts.items()) if 0 < c < n_thresh}
        self._log(f"[CROSS-THRESHOLD] Always pruned ({n_thresh}/{n_thresh}): {always_pruned}")
        self._log(f"[CROSS-THRESHOLD] Never pruned (0/{n_thresh}): {never_pruned}")
        self._log(f"[CROSS-THRESHOLD] Borderline: {borderline}")
        
        # Use results from first threshold for return (unless specified otherwise)
        # This maintains backward compatibility while storing all thresholds
        first_thresh = thresholds[0]
        results = results_by_threshold[first_thresh]
        
        # Compute correlation matrix (using full data)
        self._log(f"Computing feature correlation matrix...")
        corr_matrix = np.corrcoef(X.T) if X.shape[1] > 1 else np.array([[1.0]])
        
        total_time = time.time() - start_time
        self.timings['total'] = total_time
        
        self._log(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
        self._log(f"FEATURES: {n_features} total -> {len(results['kept_indices'])} kept, {len(results['dropped_indices'])} pruned")
        self._log(f"DROPPED: {[feature_names[i] for i in results['dropped_indices']]}")
        
        return {
            'results': results,
            'results_by_threshold': results_by_threshold,
            'kept_indices': results['kept_indices'],
            'dropped_indices': results['dropped_indices'],
            'kept_names': [feature_names[i] for i in results['kept_indices']],
            'dropped_names': [feature_names[i] for i in results['dropped_indices']],
            'corr_matrix': corr_matrix,
            'timings': self.timings,
            'n_features_original': n_features,
            'n_features_pruned': n_keep,
            'thresholds': thresholds.tolist(),
        }
    
    def _analyze_correlation(
        self,
        X: np.ndarray,
        y_raw: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        results = []
        thresholds = self.config['CORRELATION_THRESHOLDS']
        
        for i, name in enumerate(feature_names):
            feature = X[:, i]
            row = {'feature': name, 'index': i}
            
            try:
                row['pearson_r'] = np.corrcoef(feature, y_raw)[0, 1]
            except:
                row['pearson_r'] = 0.0
            
            try:
                row['spearman_r'], _ = spearmanr(feature, y_raw)
            except:
                row['spearman_r'] = 0.0
            
            for t in thresholds:
                y_bin = (y_raw >= t).astype(int)
                if len(np.unique(y_bin)) < 2:
                    row[f'pb_r_t{t}'] = 0.0
                    row[f'auc_t{t}'] = 0.5
                else:
                    try:
                        row[f'pb_r_t{t}'], _ = pointbiserialr(y_bin, feature)
                    except:
                        row[f'pb_r_t{t}'] = 0.0
                    try:
                        row[f'auc_t{t}'] = roc_auc_score(y_bin, feature)
                    except:
                        row[f'auc_t{t}'] = 0.5
            
            results.append(row)
        
        df = pd.DataFrame(results)
        df['spearman_abs'] = df['spearman_r'].abs()
        df['pearson_abs'] = df['pearson_r'].abs()
        df['auc_t2_abs'] = (df['auc_t2.0'] - 0.5).abs()
        df['rank'] = (
            df['spearman_abs'].rank(ascending=False) +
            df['pearson_abs'].rank(ascending=False) +
            df['auc_t2_abs'].rank(ascending=False)
        )
        return df.sort_values('rank')
    
    def _analyze_tree(
        self,
        X: np.ndarray,
        y_binary: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        n_trees = self.config['TREE_ESTIMATORS']
        
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
    
    def _analyze_permutation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, float]:
        import logging
        tf_logger = logging.getLogger('tensorflow')
        old_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)
        
        model = self._build_quick_dense(X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)
        
        base_auc = roc_auc_score(y_val, model.predict(X_val, verbose=0).flatten())
        
        importance_scores = []
        n_repeats = self.config['PERMUTATION_REPEATS']
        
        for i, name in enumerate(feature_names):
            X_permuted = X_val.copy()
            rng = np.random.RandomState(42 + i)
            rng.shuffle(X_permuted[:, i])
            
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
        
        df = pd.DataFrame(importance_scores)
        df['rank'] = df['auc_drop'].rank(ascending=False)
        del model
        return df.sort_values('rank'), base_auc
    
    def _analyze_neural(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        trained_model: Any = None
    ) -> pd.DataFrame:
        import logging
        import tensorflow as tf
        tf_logger = logging.getLogger('tensorflow')
        old_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)
        
        if trained_model is not None:
            model = trained_model
        else:
            model = self._build_quick_dense(X_train.shape[1])
            model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)
        
        weights = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.kernel is not None:
                if layer.kernel.shape[0] == len(feature_names):
                    weights = np.abs(layer.kernel.numpy())
                    break
        
        tf_logger.setLevel(old_level)
        
        if weights is None:
            self._log(f"  WARNING: Could not extract input weights, using random", 'warning')
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
    
    def _analyze_shap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        trained_model: Any = None
    ) -> pd.DataFrame:
        try:
            import shap
        except ImportError:
            self._log(f"  SHAP not available, using fallback", 'warning')
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
            model = self._build_quick_dense(X_train.shape[1])
            model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)
        
        n_shap = self.config['SHAP_SAMPLE_SIZE']
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
            self._log(f"  SHAP failed: {e}, using fallback", 'warning')
            mean_abs_shap = np.random.rand(len(feature_names))
        
        df = pd.DataFrame({
            'feature': feature_names,
            'index': range(len(feature_names)),
            'mean_abs_shap': mean_abs_shap,
        })
        df['rank'] = df['mean_abs_shap'].rank(ascending=False)
        return df.sort_values('rank')
    
    def _analyze_ablation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        import logging
        tf_logger = logging.getLogger('tensorflow')
        old_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)
        
        results = []
        for i, name in enumerate(feature_names):
            X_tr = X_train[:, i:i+1].copy()
            X_va = X_val[:, i:i+1].copy()
            
            model = self._build_quick_dense(1)
            model.fit(X_tr, y_train, epochs=10, batch_size=256, verbose=0)
            
            pred = model.predict(X_va, verbose=0).flatten()
            try:
                auc = roc_auc_score(y_val, pred)
            except:
                auc = 0.5
            
            results.append({'feature': name, 'index': i, 'auc': auc, 'pred_mean': np.mean(pred)})
            del model
        
        df = pd.DataFrame(results)
        df['rank'] = df['auc'].rank(ascending=False)
        return df.sort_values('rank')
    
    def _build_quick_dense(self, input_dim: int) -> Any:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        return model
    
    def _compute_consolidated_ranking(
        self,
        results: Dict,
        feature_names: List[str]
    ) -> pd.DataFrame:
        methods = ['correlation', 'tree', 'permutation', 'neural', 'shap', 'ablation']
        rank_cols = {
            'correlation': 'rank',
            'tree': 'rank',
            'permutation': 'rank',
            'neural': 'rank',
            'shap': 'rank',
            'ablation': 'rank',
        }
        
        all_ranks = pd.DataFrame({'feature': feature_names, 'index': range(len(feature_names))})
        
        for method in methods:
            df = results.get(method)
            if df is not None and 'rank' in df.columns:
                all_ranks[method] = all_ranks['index'].map(
                    dict(zip(df['index'], df['rank']))
                )
        
        rank_cols_present = [m for m in methods if m in all_ranks.columns]
        all_ranks['mean_rank'] = all_ranks[rank_cols_present].mean(axis=1)
        all_ranks['consolidated_rank'] = all_ranks['mean_rank'].rank()
        
        return all_ranks.sort_values('consolidated_rank')
    
    def generate_report(self, analysis_results: Dict) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        lines.append("=" * 80)
        
        n_orig = analysis_results['n_features_original']
        n_pruned = analysis_results['n_features_pruned']
        lines.append(f"\nOriginal features: {n_orig} | After pruning: {n_pruned} | Dropped: {n_orig - n_pruned}")
        
        timings = analysis_results['timings']
        lines.append(f"Runtime: {timings.get('total', 0)/60:.1f} min")
        lines.append(f"  - Correlation: {timings.get('correlation', 0):.1f}s")
        lines.append(f"  - Tree: {timings.get('tree', 0):.1f}s")
        lines.append(f"  - Permutation: {timings.get('permutation', 0):.1f}s")
        lines.append(f"  - Neural: {timings.get('neural', 0):.1f}s")
        lines.append(f"  - SHAP: {timings.get('shap', 0):.1f}s")
        lines.append(f"  - Ablation: {timings.get('ablation', 0):.1f}s")
        
        lines.append(f"\nDropped features: {analysis_results['dropped_names']}")
        lines.append(f"Kept features: {analysis_results['kept_names']}")
        
        results = analysis_results['results']
        lines.append("\n" + "-" * 80)
        lines.append("CONSOLIDATED RANKING (1 = most important)")
        lines.append("-" * 80)
        
        consolidated = results['consolidated']
        if 'consolidated_rank' in consolidated.columns:
            display_cols = ['feature', 'mean_rank', 'consolidated_rank']
        else:
            display_cols = ['feature', 'mean_rank']
        
        available = [c for c in display_cols if c in consolidated.columns]
        if 'index' not in available:
            available = ['feature', 'mean_rank', 'consolidated_rank']
        
        lines.append(consolidated[['feature', 'mean_rank', 'consolidated_rank']].to_string(index=False))
        
        lines.append("\n" + "-" * 80)
        lines.append("TOP FEATURES PER METHOD")
        lines.append("-" * 80)
        
        method_labels = {
            'correlation': 'Spearman Correlation',
            'tree': 'Tree Importance (RF+GBM)',
            'permutation': 'Permutation Importance',
            'neural': 'Neural Weight Magnitude',
            'shap': 'SHAP Values',
            'ablation': 'Ablation Study (AUC)',
        }
        
        for method, label in method_labels.items():
            df = results.get(method)
            if df is not None:
                lines.append(f"\n{label}:")
                rank_col = 'rank'
                if rank_col in df.columns:
                    top5 = df.nsmallest(5, rank_col)[['feature', 'rank']]
                    if 'combined_importance' in df.columns:
                        top5 = df.nsmallest(5, rank_col)[['feature', 'rank', 'combined_importance']]
                    elif 'mean_abs_shap' in df.columns:
                        top5 = df.nsmallest(5, rank_col)[['feature', 'rank', 'mean_abs_shap']]
                    elif 'auc' in df.columns:
                        top5 = df.nsmallest(5, rank_col)[['feature', 'rank', 'auc']]
                    elif 'spearman_abs' in df.columns:
                        top5 = df.nsmallest(5, rank_col)[['feature', 'rank', 'spearman_abs']]
                    lines.append(f"  {top5.to_string(index=False)}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    def save_report(self, analysis_results: Dict, output_path: str = './feature_importance_report.txt') -> None:
        report = self.generate_report(analysis_results)
        self._log(f"{report}")
        
        with open(output_path, 'w') as f:
            f.write(report)
        self._log(f"Report saved to {output_path}")
        
        results = analysis_results['results']
        consolidated = results.get('consolidated')
        if consolidated is not None:
            csv_path = output_path.replace('.txt', '.csv')
            consolidated.to_csv(csv_path, index=False)
            self._log(f"Consolidated ranking saved to {csv_path}")
        
        meta_path = output_path.replace('.txt', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'n_features_original': analysis_results['n_features_original'],
                'n_features_pruned': analysis_results['n_features_pruned'],
                'dropped_names': analysis_results['dropped_names'],
                'kept_names': analysis_results['kept_names'],
                'timings': analysis_results['timings'],
            }, f, indent=2)
        self._log(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    print("Testing FeatureImportanceAnalyzer...")
    
    np.random.seed(42)
    n_samples = 10000
    n_features = 21
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X = np.random.randn(n_samples, n_features)
    y_raw = X[:, 0] * 2.0 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    y_raw = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) * 10 - 5
    
    y_binary_t2 = (y_raw >= 2.0).astype(int)
    while y_binary_t2.sum() < 100 or (1 - y_binary_t2).sum() < 100:
        y_raw = X[:, 0] * 2.0 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.3
        y_raw = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) * 10 - 5
        y_binary_t2 = (y_raw >= 2.0).astype(int)
    
    print(f"Test data: {n_samples} samples, positive rate (t=2.0): {y_binary_t2.mean():.1%}")
    
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.run_full_analysis(X, y_raw, feature_names)
    analyzer.save_report(results, './feature_importance_report_test.txt')
    
    print(f"\n[PASS] FeatureImportanceAnalyzer test passed")
    print(f"[PASS] Kept {results['n_features_pruned']}/{results['n_features_original']} features")
    print(f"[PASS] Dropped: {results['dropped_names']}")
