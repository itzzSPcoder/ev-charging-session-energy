import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from src.ev_charging.data import clean_vehicle_name
from src.ev_charging.features import ALL_FEATURES


class EVChargingPredictor:
    """Predictor service class loading the champion model."""

    def __init__(self, model_path: str = "models/model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict_one(self, data: Dict[str, Any]) -> float:
        """Predict energy_kwh for a single input record."""
        df = pd.DataFrame([data])
        if "vehicle" in df.columns:
            df["vehicle"] = df["vehicle"].apply(clean_vehicle_name)
        X = df[ALL_FEATURES]
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Predict energy_kwh for a batch of records."""
        df_copy = df.copy()
        if "vehicle" in df_copy.columns:
            df_copy["vehicle"] = df_copy["vehicle"].apply(clean_vehicle_name)
        X = df_copy[ALL_FEATURES]
        preds = self.model.predict(X)
        return [float(p) for p in preds]

