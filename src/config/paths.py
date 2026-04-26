from pathlib import Path

# Project root is always the parent of the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Config
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
TUNED_MODELS_DIR = MODELS_DIR / "tuned"
FINAL_MODELS_DIR = MODELS_DIR / "final"

# Report directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
