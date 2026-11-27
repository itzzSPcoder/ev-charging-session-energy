import numpy as np
import pandas as pd
from src.ev_charging.metrics import compute_mae, compute_psi, compute_slice_mae
from src.ev_charging.features import ALL_FEATURES, TARGET, compute_psi as features_compute_psi
from src.ev_charging.models import get_model_pipeline


def test_compute_mae():
    """Test MAE computation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])
    mae = compute_mae(y_true, y_pred)
    assert round(mae, 4) == round((2.0 + 2.0 + 1.0) / 3.0, 4)


def test_compute_psi_stability():
    """Test PSI on identical vs shifted distributions."""
    np.random.seed(42)
    ref = np.random.normal(25.0, 5.0, 1000)
    same = np.random.normal(25.0, 5.0, 1000)
    shifted = np.random.normal(45.0, 5.0, 1000)

    psi_same = compute_psi(ref, same, num_bins=10)
    psi_shifted = compute_psi(ref, shifted, num_bins=10)

    assert psi_same < 0.1  # very stable
    assert psi_shifted > 0.5  # significant drift

    # Verify features export of compute_psi
    assert features_compute_psi(ref, same, num_bins=10) == psi_same



def test_model_pipeline_synthetic_fit_and_predict():
    """Test sklearn pipeline creation and fitting on synthetic data without CSV dependency."""
    df_synthetic = pd.DataFrame({
        "battery_start_pct": [10.0, 20.0, 50.0, 80.0, 30.0, 40.0],
        "session_minutes": [30.0, 60.0, 90.0, 120.0, 150.0, 180.0],
        "charger_kw": [7.0, 22.0, 50.0, 50.0, 22.0, 7.0],
        "temp_c": [20.0, 25.0, 30.0, 35.0, 22.0, 28.0],
        "vehicle_age_yr": [1.0, 2.0, 3.0, 4.0, 5.0, 2.5],
        "queue_before": [0, 1, 2, 3, 1, 2],
        "vehicle": ["car", "two_wheeler", "three_wheeler", "car", "two_wheeler", "three_wheeler"],
        "station_type": ["Mall", "Highway", "Residential", "Highway", "Mall", "Residential"],
        "energy_kwh": [5.0, 15.0, 25.0, 40.0, 20.0, 22.0],
    })

    pipe = get_model_pipeline(model_type="ridge", alpha=1.0)
    pipe.fit(df_synthetic[ALL_FEATURES], df_synthetic[TARGET])
    preds = pipe.predict(df_synthetic[ALL_FEATURES])

    assert len(preds) == len(df_synthetic)
    assert all(isinstance(p, float) for p in preds)

