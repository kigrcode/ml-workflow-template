import yaml
from src.config.paths import PROJECT_ROOT


def load_config() -> dict:
    """
    Load the central project config from config.yaml in the project root.
    This is the single source of truth for all dataset-specific settings.
    """
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Please create one in the project root before running notebooks."
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)