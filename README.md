# Machine Learning Ensemble Pipeline

**Python 3.12** · **TensorFlow 2.18.1** · **MIT License**  

**Last updated**: 2026-07-12  
**References**: [SPEC.md](SPEC.md) (technical specification) · [GIS.md](GIS.md) (iteration strategy & results)

## Table of Contents
1. [Overview](#overview)
2. [Dataset Quick Facts](#dataset-quick-facts)
3. [System Context](#system-context)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Directory Structure](#directory-structure)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

**Vision**: A general-purpose ML framework for imbalanced classification — deploy multiple architectures across any domain, iteratively narrow in on optimal hyperparameter spaces via systematic cost-tiered tuning, and discover which models capture signal from rare events.

**Mission**: An automated ML pipeline that loads market data, engineers temporal features to account for recency bias, optimizes classification thresholds, tunes hyperparameters via Bayesian optimization (Optuna), and ensembles 9 diverse architectures to produce more reliable strength signals. These architectures span gradient-boosted trees (CatBoost, LightGBM, XGBoost) and neural networks (Dense, VAE, CNN, RNN, LSTM, Transformer) — a diversity required by the dataset's extreme class imbalance, where no single model can reliably detect the minority class. Trained and evaluated on historical market data, this pipeline demonstrates the methodology on technical indicators — serving as a reference implementation for the ensemble classification approach.

*Disclaimer: This software is for educational and research purposes only. It does not constitute financial advice, investment recommendation, or solicitation to buy or sell securities. Past performance or model outputs do not guarantee future results. Use at your own risk.*

### Why 9 Architectures

Each architecture reveals different dataset characteristics:

| Order | Architecture | Data Insight |
|-------|-------------|--------------|
| 1 | CatBoost | Feature importance, splits |
| 2 | LightGBM | Leaf-wise splits, efficient feature learning |
| 3 | XGBoost | Regularized splitting, sparsity-aware learning |
| 4 | Dense | Global feature interactions |
| 5 | VAE | Latent distribution shape |
| 6 | CNN | Local pattern scales |
| 7 | RNN | Temporal/sequential patterns |
| 8 | LSTM | Temporal/sequential patterns |
| 9 | Transformer | Feature attention maps |

**Principle**: Designed for extreme class imbalance (259:1). Methodically discovers which architectures capture signal by tuning configurables in ascending training-cost impact: zero (gates, thresholds, percentiles) → low (preprocessing) → moderate (HPO spaces) → high (architectural changes).

## Dataset Quick Facts

| Attribute | Value |
|-----------|-------|
| Dataset size | ~6.7 million records |
| Class imbalance | 259:1 ratio (0.4% signal) |
| Features | 34 (pre-elimination, all features kept) |
| Date range | 2022-03-01 to 2025-10-23 |
| Input format | CSV with headers, date format YYYYMMDD |

## System Context

| Component | Description |
|-----------|-------------|
| **Input** | CSV file with financial features + date + target |
| **Processing** | Data loading → Temporal weighting → Model training → Ensemble → Prediction |
| **Output** | Strength signal predictions (binary + probability), model files, evaluation metrics |
| **Execution** | CPU-only, training & inference modes |

```
CSV Input → Phase 1 (Data Loading) → Phase 3 (Temporal Weighting) → 
Phase 4 (Training/Ensemble) → Phase 5 (Prediction/Output)
```

## Prerequisites

- **Python** ≥ 3.12
- **Memory** ≤ 16 GB RAM
- **Storage** ~500 MB model files, ~10 GB during execution
- **Runtime** CPU-only (no GPU required)

## Installation

```bash
git clone <repo-url>
cd cicd
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline Architecture

```
           Phase 1  -->  Phase 3  -->  Phase Xa -->  Phase BE  -->  Phase 4  -->  Phase 5
           Setup         Temporal       Feature       Backward       Training/    Inference &
                         Segmentation   Importance    Elimination    Validation    Evaluation
                         & Weighting    & Pruning     (per-arch)     & Ensemble
                                        |
                                        +-- 4a. Threshold Search
                                        +-- 4b. HyperParameter Optimization (HPO)
                                        +-- 4c. Post-HPO Threshold Search/Re-Evaluation
                                        +-- 4d. Ensemble Assembly
                                        +-- 4e. Model Persistence
                                                        |
                                                        +-- Load saved models
                                                        +-- Run inference on newest date
                                                        +-- Per-arch metrics
                                                        +-- Best architecture selection

           Input: CSV  -->  Output: X, y, dates
                              Output: Predictions, metrics, saved models
```

**Architecture groups**: Gradient boosting models (CatBoost, LightGBM, XGBoost) handle imbalance natively and produce well-calibrated probabilities. Neural networks (CNN, RNN, LSTM, Dense, VAE, Transformer) use focal loss and require feature normalization.

### Phase Details

| Phase | Description | Key Outputs | Source File |
|-------|-------------|-------------|-------------|
| Phase 1 | Data loading, preprocessing | X, y (continuous), dates | chunk_16_phase_1_setup.py |
| Phase 3 | Temporal segmentation + weighting | temporal_weights, date segments | chunk_17_phase_3_temporal.py |
| Phase Xa | Feature importance + pruning | threshold_kept_indices, all 34 kept | chunk_XX_phase_feature_analysis_a.py |
| Phase BE | Per-architecture backward elimination | `{arch: {threshold: kept_indices}}` | chunk_XX_phase_backward_elimination.py |
| Phase 4a | Threshold Search | optimal_threshold per architecture | chunk_18_phase_4_ensemble.py |
| Phase 4b | HyperParameter Optimization | best_hyperparams per architecture | chunk_21_hyperparam_optimizer.py |
| Phase 4c | Post-HPO Threshold Search/Re-Evaluation *(config-disabled by default)* | Refined threshold per architecture *(requires `ENABLE_POST_HPO_THRESHOLD_SEARCH: True`)* | chunk_18_phase_4_ensemble.py |
| Phase 4d | Ensemble Assembly | Precision-weighted predictions | chunk_18_phase_4_ensemble.py |
| Phase 4e | Model Persistence | ./saved_models/ | chunk_18_phase_4_ensemble.py |
| Phase 5 | Inference on newest date | Predictions, per-arch metrics, rankings | chunk_19_phase_5_optimization.py |

> **Two thresholds**: The pipeline distinguishes *label threshold* (binarizes the continuous target for training, swept 20→10→0) from *prediction binary split* (converts model probability → binary prediction, fixed at 0.5). These are independent — changing the label threshold selects which label definition the model learns from; the binary split never changes.

### Temporal Segmentation & Recency Bias

The dataset spans multiple market regimes — bull/bear cycles, high/low volatility periods, sector rotations — each exhibiting different dynamics. A single model trained uniformly across all regimes would dilute regime-specific signals and miss the patterns most relevant to the current market.

Phase 3 addresses this by:

- **Breaking the dataset into temporal sections** based on differences in market dynamics, allowing feature engineering and weighting to account for each regime's unique characteristics
- **Applying time-weighted sampling** that prioritizes recent dates over older ones — the most predictive patterns for tomorrow are the ones that emerged most recently

This recency bias is intentional: market micro-structures evolve, and yesterday's regime tells you more about today than a regime from six months ago.

See [SPEC.md §2.4 Pipeline Phases](SPEC.md#24-pipeline-phases) for the detailed Phase-to-Phase Model Propagation Logic (5-section evaluation flow, branch gates, ensemble assembly).

## Directory Structure

```
cicd/
├── chunk_01_config.py                  # Configuration and constants
├── chunk_02_utils_logging.py           # Logger class and formatting
├── chunk_04_utils_metrics.py           # Metric calculation utilities
├── chunk_05_data_manager.py            # Data loading and management
├── chunk_07_data_temporal.py           # Temporal feature extraction
├── chunk_08_models_base.py             # Base neural architectures (VAE, CNN, RNN)
├── chunk_09_models_advanced.py         # Transformer, attention, and hybrid architectures
├── chunk_10_models_ensemble.py         # Ensemble builders and aggregators
├── chunk_11_models_sklearn.py          # Model wrappers (sklearn, LightGBM, XGBoost, CatBoost)
├── chunk_12_evaluation_evaluator.py    # Model evaluation utilities
├── chunk_13_state_manager.py           # Pipeline state management
├── chunk_14_models_trainer.py          # Model training orchestration
├── chunk_15_phase_base.py              # Abstract base phase class
├── chunk_16_phase_1_setup.py           # Phase 1: Pipeline setup
├── chunk_17_phase_3_temporal.py        # Phase 3: Temporal weighting
├── chunk_18_phase_4_ensemble.py        # Phase 4: Neural ensemble
├── chunk_19_phase_5_optimization.py    # Phase 5: Prediction optimization
├── chunk_20_pipeline_main.py           # Main orchestrator
├── chunk_21_hyperparam_optimizer.py    # Hyperparameter optimization (Optuna)
├── chunk_22_model_loader.py            # Model loading for predictions
├── chunk_XX_feature_importance.py      # 6-method feature importance engine
├── chunk_XX_phase_feature_analysis_a.py # Phase Xa: Raw feature analysis (prunes raw features only; temporal features not model inputs)
├── chunk_XX_phase_backward_elimination.py # Phase BE: Per-arch feature elimination
├── chunk_XX_phase_feature_analysis_b.py # Phase Xb: Temporal precision gap
├── archived_models/                    # Frozen model snapshots (see GIS.md §10)
└── requirements.txt                    # Python dependencies
```

## Usage

### Testing Individual Chunks

All 23 chunk files include `if __name__ == "__main__"` self-test blocks and can be run independently:

```bash
# Configuration & utilities
python chunk_01_config.py             # Config validation
python chunk_02_utils_logging.py      # Logger
python chunk_04_utils_metrics.py      # Metric calculations

# Data pipeline
python chunk_05_data_manager.py       # Data loading
python chunk_07_data_temporal.py      # Temporal features

# Models (each builds + validates its architecture)
python chunk_08_models_base.py        # VAE, CNN, RNN, LSTM, Dense
python chunk_09_models_advanced.py    # Transformer, attention, hybrids
python chunk_10_models_ensemble.py    # Ensemble builders
python chunk_11_models_sklearn.py     # sklearn, LightGBM, XGBoost, CatBoost

# Training & evaluation
python chunk_12_evaluation_evaluator.py  # Evaluator
python chunk_13_state_manager.py         # State manager
python chunk_14_models_trainer.py        # Trainer

# Pipeline phases (each runs independently)
python chunk_15_phase_base.py         # Base phase interface
python chunk_16_phase_1_setup.py      # Phase 1: Setup
python chunk_17_phase_3_temporal.py   # Phase 3: Temporal weighting
python chunk_18_phase_4_ensemble.py   # Phase 4: Neural ensemble
python chunk_19_phase_5_optimization.py # Phase 5: Prediction optimization

# Analysis & orchestration
python chunk_20_pipeline_main.py      # Full pipeline
python chunk_21_hyperparam_optimizer.py # HPO (Optuna)
python chunk_22_model_loader.py       # Model loading
python chunk_XX_feature_importance.py      # 6-method feature importance
python chunk_XX_phase_feature_analysis_a.py # Feature analysis A
python chunk_XX_phase_feature_analysis_b.py # Feature analysis B
```

### Running Complete Pipeline

```bash
# Activate virtual environment (recommended)
./tf_venv/bin/python chunk_20_pipeline_main.py

# Or run directly
python chunk_20_pipeline_main.py
```

### Virtual Environment

A TensorFlow virtual environment is provided in `tf_venv/`:
- TensorFlow 2.18.1
- Includes all required dependencies (pandas, scikit-learn, scipy, optuna)

### Using in Custom Code

```python
from chunk_20_pipeline_main import main, PipelineOrchestrator
from chunk_01_config import CONFIG, update_config

# Run with default config
results = main()

# Run with custom config
custom_config = update_config({'SAMPLE_SIZE': 50000})
results = main(custom_config)

# Use orchestrator directly
orchestrator = PipelineOrchestrator(CONFIG)
context = orchestrator.run()
```


## Contributing

## License

MIT
