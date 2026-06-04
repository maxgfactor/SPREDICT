# Fraud Detection Ensemble Pipeline

**Vision**: Build the most precise fraud detection system that minimizes false positives while maximizing true positives, enabling confident fraud detection without disrupting legitimate transactions.

**Mission**: Orchestrate an automated ML pipeline that loads financial transaction data, engineers temporal features, optimizes classification thresholds, tunes hyperparameters via Bayesian optimization (Optuna), ensembles 9 architectures (CatBoost, LightGBM, XGBoost + 6 neural networks), and evaluates final predictions with comprehensive metrics.

This directory contains the decomposed pipeline broken into 21 individually testable chunks.

## Directory Structure

```
cicd/
├── chunk_01_config.py                  # Configuration and constants
├── chunk_02_utils_logging.py           # Logger class and formatting
├── chunk_04_utils_metrics.py           # Metric calculation utilities
├── chunk_05_data_manager.py            # Data loading and management
├── chunk_07_data_temporal.py           # Temporal feature extraction
├── chunk_08_models_base.py             # Base neural architectures (VAE, CNN, RNN)
├── chunk_09_models_advanced.py         # Advanced architectures (Transformer, GNN)
├── chunk_10_models_ensemble.py         # Ensemble builders and aggregators
├── chunk_11_models_sklearn.py          # Scikit-learn model wrappers
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
├── chunk_XX_phase_feature_analysis_a.py # Phase Xa: Raw feature analysis
├── chunk_XX_phase_feature_analysis_b.py # Phase Xb: Temporal precision gap
├── legacy files/                       # Moved non-functional/legacy code and logs
├── for_train_x_2025_10_24_clean.csv    # Input data (~938 MB)
└── ... (output artifacts, config backups, documentation)
```

## Dependency Layers

### Phase 1: Foundation (00-04)
Independent chunks with no dependencies.
- `00_validation_framework.py` - Validation utilities
- `01_config.py` - Configuration constants
- `02_utils_logging.py` - Logging utilities
- `03_utils_memory.py` - Memory utilities
- `04_utils_metrics.py` - Metric utilities

### Phase 2: Data Layer (05-07)
Depends on: Phase 1
- `05_data_manager.py` - Data loading and validation
- `06_data_augmentation.py` - Data augmentation
- `07_data_temporal.py` - Temporal features

### Phase 3: Model Architecture (08-11)
Depends on: Phase 1
- `08_models_base.py` - Basic neural models
- `09_models_advanced.py` - Advanced neural models
- `10_models_ensemble.py` - Ensemble models
- `11_models_sklearn.py` - Sklearn models

### Phase 4: Training & Evaluation (12-14)
Depends on: Phase 1-3
- `12_evaluation_evaluator.py` - Evaluation utilities
- `13_state_manager.py` - State management
- `14_models_trainer.py` - Training orchestration

### Phase 5: Pipeline Phases (15-19)
Depends on: Phase 1-4
- `15_phase_base.py` - Base phase class
- `16_phase_1_setup.py` - Phase 1 implementation
- `17_phase_3_temporal.py` - Phase 3 implementation (Phase 2 removed)
- `18_phase_4_ensemble.py` - Phase 4 implementation:
  - 4a. Threshold Optimization (find best threshold for each architecture)
  - 4b. Hyperparameter Optimization (Optuna Bayesian search)
  - 4c. Ensemble Creation (combine model predictions)
  - 4d. Model Persistence (save to ./saved_models/)
- `19_phase_5_optimization.py` - Phase 5 implementation (final predictions)

**Note**: Phase 2 was intentionally removed - threshold optimization was moved into Phase 4

### Phase 6: Orchestration (20-21)
Depends on: All phases
- `chunk_20_pipeline_main.py` - Main orchestrator
- `chunk_21_hyperparam_optimizer.py` - Hyperparameter optimization (Optuna)

## Usage

### Testing Individual Chunks

Each chunk can be tested independently:

```bash
python 01_config.py          # Test configuration
python 08_models_base.py     # Test base models
python 16_phase_1_setup.py   # Test Phase 1
```

### Running Complete Pipeline

```bash
# Activate virtual environment (recommended)
./tf_venv/bin/python chunk_20_pipeline_main.py

# Or run directly
python chunk_20_pipeline_main.py  # Requires pip install of dependencies
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

## Input/Output Contracts

Each chunk has defined validation functions:

- `validate_phase1_output(context)` - Validates Phase 1 output
- `validate_phase3_output(context)` - Validates Phase 3 output
- `validate_phase4_output(context)` - Validates Phase 4 output
- `validate_phase5_output(context)` - Validates Phase 5 output
- `validate_pipeline_execution(context)` - Validates complete pipeline

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Phase 1     │    │  Phase 3     │    │  Phase 4     │              │
│  │  Setup       │───▶│  Temporal    │───▶│  Training    │              │
│  │              │    │  Weighting   │    │              │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                  │                       │
│  Input: CSV                              ┌──────▼───────┐              │
│  Output: X, y, dates                    │ 4a. Threshold │              │
│                                         │     Search    │              │
│                                         └──────┬───────┘              │
│                                                │                       │
│                                         ┌──────▼───────┐              │
│                                         │ 4b. HPO       │              │
│                                         │ (Optuna)      │              │
│                                         └──────┬───────┘              │
│                                                │                       │
│                                         ┌──────▼───────┐              │
│                                         │ 4c. Ensemble │              │
│                                         │    Creation  │              │
│                                         └──────┬───────┘              │
│                                                │                       │
│                                         ┌──────▼───────┐              │
│                                         │ 4d. Model    │              │
│                                         │    Save      │              │
│                                         └──────┬───────┘              │
│                                                │                       │
│  ┌─────────────────────────────────────────────▼──────────┐            │
│  │                   Phase 5                           │            │
│  │           Final Predictions & Evaluation             │            │
│  │  - Per-architecture metrics (P, R, F1, AUC)         │            │
│  │  - Confusion matrix                                 │            │
│  │  - Best architecture selection                      │            │
│  └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  Output: Predictions, metrics, saved models                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase Details

| Phase | Description | Key Outputs |
|-------|-------------|-------------|
| Phase 1 | Data loading, preprocessing | X, y (continuous), dates |
| Phase 3 | Temporal feature engineering | temporal_weights |
| Phase 4a | Threshold optimization | optimal_threshold per architecture |
| Phase 4b | Hyperparameter optimization | best_hyperparams per architecture |
| Phase 4c | Ensemble creation | Combined predictions |
| Phase 4d | Model persistence | ./saved_models/ |
| Phase 5 | Final evaluation | Metrics, rankings |

### Phase-to-Phase Model Propagation Logic

Each architecture passes through 5 evaluation sections. At every gate the
**better result propagates forward** — if a later phase doesn't improve, the
prior phase's model+threshold is carried over unchanged.

```
Section 1 (threshold search across multiple label thresholds)
  │  best threshold → threshold_opt_model + optimal_threshold
  ▼
Section 2 (HPO — Optuna Bayesian optimization)
  │  best trial → hpo_best_model + hpo_val_precision
  ▼
Section 3 (election gate)
  ├─ HPO improved → use HPO model           [HYPERPARAMETER_OPTIMIZATION]
  └─ HPO did not improve → carry over S1    [PRE-HYPERPARAMETER_OPTIMIZATION]
  │
  ▼
Section 4 (post-HPO threshold search)
  ├─ post-HPO precision > S3 precision → adopt new threshold   [section4]
  └─ post-HPO did not improve → carry over S3 threshold+model  [section3]
  │
  ▼
Section 5 FINAL (uses Section 4's elected model+threshold)
```

#### Section 1 — Threshold Search
- Train model with **default hyperparameters** at each label threshold (20→10→0)
- Evaluate at prediction threshold 0.5 at each label threshold
- **Output**: `optimal_threshold` (label threshold with best val precision),
  `threshold_opt_model` (model trained at that threshold), all per-threshold
  metrics stored in `all_results`

#### Section 2 — Hyperparameter Optimization (HPO)
- Run Optuna Bayesian optimization (5–30 trials) using the `optimal_threshold`
  from Section 1 with the same 0.5 prediction threshold
- **Output**: `hpo_best_model` + `hpo_val_precision` + `best_hyperparams`
- HPO trials that fail MaxPred, TP, or min-precision gates are rejected
  silently; the surviving best trial is the "HPO best"

#### Section 3 — HPO Election Gate
Compare HPO precision vs pre-HPO precision at prediction threshold 0.5:

- **Branch 1** (HPO did NOT improve — 7 archs: CatBoost through Transformer):
  The pre-HPO model (`threshold_opt_model`) is the best. Model and threshold
  are identical to Section 1. **No re-evaluation** — copy all metrics from
  Section 1's `section2_TP/FP/TN/FN/AUC/F1/R/pred` directly.
  Tag: `[PRE-HYPERPARAMETER_OPTIMIZATION]`

- **Branch 2** (HPO improved — 2 archs: LSTM, VAE):
  The HPO model (`hpo_best_model`) is better. Re-evaluate on validation data
  and compute precision from own TP/FP.
  Tag: `[HYPERPARAMETER_OPTIMIZATION]`

- **Branch 3** (all HPO trials rejected):
  No HPO model exists. Use Section 1 baseline metrics; compute precision from
  `baseline_cm` TP/FP.
  Tag: `[PRE-HYPERPARAMETER_OPTIMIZATION]`

- **Safety net**: `section3_precision = section3_TP / (section3_TP + section3_FP)`
  recalculated after all branches to guarantee self-consistency.

#### Section 4 — Post-HPO Threshold Search
- `model_for_post_hpo` = the model elected in Section 3
  (`threshold_opt_model` if HPO didn't improve, `hpo_best_model` if it did)
- Run a second threshold search (same label thresholds) using `retrain_model=False`
  (inference-only, no retraining per threshold)
- **Decision**: if `post_hpo_prec > section3_precision`:
    adopt `final_threshold = post_hpo_thresh`, `threshold_source = 'section4'`
  Else:
    keep `final_threshold = optimal_threshold` (Section 1) and the elected model
- If post-HPO was not elected for S5, Section 4 overrides its own logged
  metrics to match whatever model S5 will actually use (prevents
  cross-contamination in the log)

#### Section 5 — Final Evaluation
- Use Section 4's elected model + `final_threshold`
- Evaluate on validation data for the final log line
- Evaluate on inference data (newest held-out date) for production predictions
- The same model that produced Section 4's metrics must be used here —
  silent model swaps between S4 and S5 are a functional failure

#### Ensemble Assembly (after Section 5)
- All architectures with `VAL_PRECISION ≥ ENSEMBLE_MIN_PRECISION` (0.40) are
  eligible for the ensemble
- Precision-weighted voting: each architecture's vote weight =
  `precision_i / sum(precisions of all eligible archs)`
- Fallback: if no architecture meets 0.40, use the highest-precision arch alone

## CI/CD Integration

### Sequential Testing Order

```
Phase 1 (Chunks 00-04) → Phase 2 (Chunks 05-07) → Phase 3 (Chunks 08-11) →
Phase 4 (Chunks 12-14) → Phase 5 (Chunks 15-19) → Phase 6 (Chunks 20-21)

Pipeline Execution Order: Phase 1 → Phase 3 → Phase 4 → Phase 5
(Note: Phase 2 threshold optimization removed - merged into Phase 5)
```

### Example CI Pipeline

```yaml
# .github/workflows/test.yml
steps:
  - name: Test Foundation Layer
    run: |
      python 00_validation_framework.py
      python 01_config.py
      python 02_utils_logging.py
      python 03_utils_memory.py
      python 04_utils_metrics.py
  
  - name: Test Data Layer
    run: |
      python 05_data_manager.py
      python 06_data_augmentation.py
      python 07_data_temporal.py
  
  - name: Test Model Architectures
    run: |
      python 08_models_base.py
      python 09_models_advanced.py
      python 10_models_ensemble.py
      python 11_models_sklearn.py
  
  - name: Test Training Layer
    run: |
      python 12_evaluation_evaluator.py
      python 13_state_manager.py
      python 14_models_trainer.py
  
  - name: Test Pipeline Phases
    run: |
      python 15_phase_base.py
      python 16_phase_1_setup.py
      python 17_phase_3_temporal.py
      python 18_phase_4_ensemble.py
      python 19_phase_5_optimization.py
  
  - name: Test Full Pipeline
    run: python 20_pipeline_main.py
```

## Future Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **Unit Testing** | pytest, tests/ directory, conftest.py, coverage targets | 📋 Planned |
| **CI/CD** | GitHub Actions (ci.yml, train.yml), Ruff, Black, MyPy | 📋 Planned |
| **Documentation Site** | MkDocs, GitHub Pages, architecture docs, model card | 📋 Planned |
| **Containerization** | Dockerfile, docker-compose for reproducible builds | 💡 Future |
| **MLOps** | MLflow tracking, model registry, Feast, Evidently, Airflow | 💡 Future |

## Key Features

1. **Modular Design**: Each chunk is self-contained and testable
2. **Clear Contracts**: Input/output validation at each phase
3. **Incremental Testing**: Test early phases before later ones
4. **Error Isolation**: Failures are contained to specific chunks
5. **Documentation**: Each chunk includes docstrings and examples

## Migration from Monolithic

The original `study9011_enhanced_final.py` (6,402 lines) has been decomposed into:
- 22 individual Python files
- Clear separation of concerns
- Testable units

## Notes

- All chunks use relative imports from the cicd/ directory
- Each chunk includes a `if __name__ == "__main__":` self-test block
- Validation functions are provided for each phase boundary
- The original file is preserved for reference

## Documentation Map

| File | Purpose | When to Read |
|------|---------|-------------|
| **README.md** | Technical overview, directory structure, usage | First visit, new contributors |
| **SPEC.md** | Full specification: results, configs, history, failures | Before making code changes |
| **shortmemory.txt** | Current project state, key nuances, bug fixes | Every session start |
| **longmemory.txt** | Principles, SOPs, bug patterns, best practices | When designing new features |
| **legacy files/cloudnativetransformationplan.txt** | *Superseded — content migrated to README and longmemory* | Archived |
| **legacy files/** | Non-functional/legacy code, logs, and unused resources | Reference only |

## CWDA: Current Working Directory Analysis

The working directory is organized into functional groups determined by tracing the actual import graph from `chunk_20_pipeline_main.py`:

| Group | Count | Description |
|-------|-------|-------------|
| **Functional Pipeline Code** | 24 files | Direct or transitive imports of the orchestrator — pipeline cannot run without these |
| **Non-Functional Code** | 7 files → `legacy files/` | Zero imports by pipeline code (orphaned chunks, standalone scripts, shell wrappers) |
| **Functional Logs** | 1 file | `pipeline_cpu.log` — latest active run |
| **Non-Functional Logs** | 4 files → `legacy files/` | Dated and legacy run logs |
| **Functional Documentation** | 4 files + 1 dir | README, SPEC, shortmemory, longmemory |
| **Generated Artifacts** | ~30 files | saved models, feature reports, metrics CSV, config snapshots |
| **Infrastructure** | tf_venv/, .git/, .gitignore | Environment and VCS |
| **Unused Resource** | → `legacy files/` | 2.8 GB CUDA WSL installer (pipeline runs CPU-only) |

## Project Continuity

### Rules for Maintaining Context Across Sessions

1. **Before Closing**
   - Update shortmemory.txt with any significant decisions or in-progress tasks
   - Note any pending next steps or open questions

2. **When Reopening**
   - Review shortmemory.txt for conversation history and key findings
   - Review README.md for current project structure and state

3. **For Maximum Continuity**
   - Keep shortmemory.txt updated with decisions, not just conversations
   - Note file state changes if any occurred since last session
   - Document any in-progress tasks or planned work