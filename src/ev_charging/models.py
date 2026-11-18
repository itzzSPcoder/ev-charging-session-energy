from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.ev_charging.features import build_preprocessor


def get_model_pipeline(model_type: str = "ridge", **kwargs) -> Pipeline:
    """Create a full sklearn Pipeline combining preprocessing and regressor."""
    preprocessor = build_preprocessor()

    if model_type == "ridge":
        reg = Ridge(
            alpha=kwargs.get("alpha", 1.0),
            random_state=kwargs.get("random_state", 42)
        )
    elif model_type == "rf":
        reg = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            random_state=kwargs.get("random_state", 42)
        )
    elif model_type == "gb":
        reg = GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 4),
            random_state=kwargs.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", reg)
    ])
    return pipeline

