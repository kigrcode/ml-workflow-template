# ML Workflow Template

## Overview
An end-to-end classical machine learning template designed for reuse across 
projects. Built with a clean separation between orchestration notebooks and 
reusable src modules, with all dataset-specific settings centralised in 
config.yaml.

## Workflow
Run notebooks in order:
1. `1. EDA` — exploratory data analysis
2. `2. Data Preprocessing` — feature type detection, imputation, encoding, scaling
3. `3. Baseline Models` — cross-validated comparison of all models
4. `4. Feature Engineering` — automated and domain-specific feature generation
5. `5. Hyperparameter Tuning` — Optuna Bayesian optimisation or random search
6. `6. Threshold Tuning` — optimal classification threshold selection
7. `7. Final Evaluation` — single test set evaluation, reports and artifacts

## Project structure

PROJECT NAME TEMPLATE/
├── data/
│   ├── raw/              # original immutable data — never modified
│   ├── interim/          # engineered features pre-preprocessing
│   └── processed/        # final preprocessed data for modelling
├── models/
│   ├── baseline/         # all baseline models
│   ├── tuned/            # hyperparameter tuned models
│   └── final/            # definitive model for deployment
├── notebooks/            # one notebook per workflow stage
├── reports/              # evaluation reports and figures
├── src/
│   ├── config/           # paths, settings, config loader
│   ├── data/             # data loading utilities
│   ├── features/         # pipeline, type detection, feature engineering
│   ├── models/
│   │   ├── training/     # trainer, registry, metrics, tuning, threshold
│   │   └── prediction/   # prediction utilities
│   ├── utils/            # reporting utilities
│   └── visualization/    # EDA plotting functions
└── tests/                # unit tests

## Setup

### 1. Create and activate the environment
```bash
conda env create -f environment.yml
conda activate ml-template
```

### 2. Install the project as a package
```bash
pip install -e .
```
This makes src modules importable from anywhere in the project.

### 3. Add your data
Place your raw data file(s) in `data/raw/` before running any notebooks.

### 4. Configure your project
Edit `config.yaml` — this is the only file you need to change when 
starting a new project:

```yaml
target: "your_target_column"
drop_columns:
  - "id_column"
```

## Design decisions

### Preprocessing inside CV folds
The preprocessor is fitted inside each cross validation fold on training 
data only, then applied to the validation fold to prevent data leakage.

### Test set held out until final evaluation
The test set is split off at the start of the preprocessing notebook and 
never touched until notebook 7. All model selection, feature engineering 
and tuning decisions are made using CV scores only.

### Config-driven design
All dataset-specific settings live in config.yaml — target column, columns 
to drop, split parameters, pipeline settings, model selection and tuning 
parameters. Notebooks and src modules contain no hardcoded dataset-specific 
values. Starting a new project means editing config.yaml only.

### Feature engineering after baseline modelling
Feature engineering is deliberately placed after baseline modelling. SHAP 
values from the baseline model inform which features are worth engineering.

### Two-stage feature selection
Feature selection uses mutual information as a fast pre-filter to eliminate 
obvious noise, followed by SHAP-based refinement for precise, model-aware 
selection.

### Automated feature engineering is opt-in
All automated feature engineering (interactions, polynomial, binning, 
aggregations) is disabled by default in config.yaml.

### Optional dependencies
XGBoost, LightGBM and CatBoost are imported conditionally. If any are not 
installed the model registry still loads and the workflow runs with whatever 
is available.

## Usage notes

### Threshold tuning metric
Choose the threshold metric in config.yaml based on the business problem:
- `f1` — balanced precision and recall, good default
- `recall` — minimise false negatives, use for medical/fraud detection
- `precision` — minimise false positives, use for spam/ad targeting
- `accuracy` — use only when classes are balanced

### Feature engineering flags
Enable selectively based on model type and dataset size:
- `interactions` and `polynomial` — most useful for linear models
- `binning` — useful for non-monotonic numeric relationships
- `aggregations` — useful when group statistics carry signal


## Development notes
AI-assisted development workflow was used throughout, with architectural 
decisions, methodological choices and code review handled by the developer. 

