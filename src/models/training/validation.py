import inspect


def validate_param_overrides(model_class, overrides: dict) -> None:
    """
    Strict, minimal validation:
    - every override key must exist in model_class.__init__ signature
    - raise ValueError immediately on any invalid key
    """
    if not overrides:
        return

    valid_params = inspect.signature(model_class.__init__).parameters

    for key in overrides:
        if key not in valid_params:
            raise ValueError(
                f"Invalid parameter '{key}' for {model_class.__name__}. "
                f"Allowed parameters include: {list(valid_params.keys())}"
            )