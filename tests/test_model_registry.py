import os
import sqlite3
import joblib
import pandas as pd
import pytest
from src.ev_charging.features import ALL_FEATURES


def test_mlflow_sqlite_integrity():
    """Verify that mlflow.db exists and contains the required tracking runs."""
    assert os.path.exists("mlflow.db"), "mlflow.db must exist in root"

    conn = sqlite3.connect("mlflow.db")
    c = conn.cursor()

    # Check runs count >= 3
    runs = c.execute("SELECT run_uuid FROM runs").fetchall()
    assert len(runs) >= 3, f"Expected at least 3 MLflow runs, found {len(runs)}"

    # Check data_sha256 logged in params
    params = c.execute("SELECT key, value FROM params WHERE key = 'data_sha256'").fetchall()
    assert len(params) >= 3, "Expected data_sha256 to be logged across runs"
    assert params[0][1] == "225915736a4f872e9c15a7ed1ef315e78fb8f69180106347c9d1ddf98122c443"

    # Check mae metric logged
    metrics = c.execute("SELECT DISTINCT key FROM metrics WHERE key LIKE '%mae%'").fetchall()
    assert len(metrics) > 0, "Expected metrics containing 'mae' to be logged"

    # Check registered model exists
    registered_models = c.execute("SELECT name FROM registered_models").fetchall()
    assert ("ev_charging_model",) in registered_models, "ev_charging_model must be in registry"

    conn.close()


def test_champion_model_loading_and_inference():
    """Verify champion model loads from models/model.joblib and predicts correctly."""
    assert os.path.exists("models/model.joblib"), "models/model.joblib must exist"
    model = joblib.load("models/model.joblib")

    # Sample input
    sample_df = pd.DataFrame([{
        "battery_start_pct": 25.0,
        "session_minutes": 90.0,
        "charger_kw": 30.0,
        "temp_c": 28.0,
        "vehicle_age_yr": 3.0,
        "queue_before": 2,
        "vehicle": "car",
        "station_type": "Highway",
    }])

    pred = model.predict(sample_df[ALL_FEATURES])
    assert len(pred) == 1
    assert float(pred[0]) > 0.0
