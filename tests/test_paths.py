"""
Tests for src/config/paths.py

Verifies that all paths resolve correctly relative to the
project root regardless of where the test is run from.
"""

import pytest
from pathlib import Path

from src.config.paths import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    BASELINE_MODELS_DIR,
    TUNED_MODELS_DIR,
    FINAL_MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    ARTIFACTS_DIR,
    CONFIG_PATH,
)


# =========================================================
# TESTS — project root
# =========================================================

def test_project_root_is_path():
    assert isinstance(PROJECT_ROOT, Path)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_project_root_contains_pyproject_toml():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_project_root_contains_config_yaml():
    assert (PROJECT_ROOT / "config.yaml").exists()


def test_project_root_contains_src():
    assert (PROJECT_ROOT / "src").exists()


# =========================================================
# TESTS — path resolution
# =========================================================

def test_all_paths_are_path_objects():
    paths = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, BASELINE_MODELS_DIR, TUNED_MODELS_DIR,
        FINAL_MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
        ARTIFACTS_DIR, CONFIG_PATH,
    ]
    for path in paths:
        assert isinstance(path, Path), f"{path} is not a Path object"


def test_all_paths_are_absolute():
    paths = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, BASELINE_MODELS_DIR, TUNED_MODELS_DIR,
        FINAL_MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
        ARTIFACTS_DIR,
    ]
    for path in paths:
        assert path.is_absolute(), f"{path} is not absolute"


def test_all_paths_under_project_root():
    paths = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, BASELINE_MODELS_DIR, TUNED_MODELS_DIR,
        FINAL_MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
        ARTIFACTS_DIR,
    ]
    for path in paths:
        assert str(PROJECT_ROOT) in str(path), \
            f"{path} is not under project root"


def test_data_dirs_under_data():
    for path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR]:
        assert "data" in path.parts


def test_model_dirs_under_models():
    for path in [BASELINE_MODELS_DIR, TUNED_MODELS_DIR, FINAL_MODELS_DIR]:
        assert "models" in path.parts


def test_config_path_points_to_yaml():
    assert CONFIG_PATH.suffix == ".yaml"