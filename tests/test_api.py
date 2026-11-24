import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    """Test GET /health returns status: ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_valid(client):
    """Test POST /predict with a valid payload returns 200 and a numeric prediction."""
    payload = {
        "battery_start_pct": 35.0,
        "session_minutes": 120.0,
        "charger_kw": 25.0,
        "temp_c": 28.5,
        "vehicle_age_yr": 3.0,
        "queue_before": 2,
        "vehicle": "two_wheeler",
        "station_type": "Mall"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0


def test_predict_rule_temp_c_above_75_rejected(client):
    """Client rule 1: temp_c above 75 is rejected with 422 Unprocessable Entity."""
    payload = {
        "battery_start_pct": 35.0,
        "session_minutes": 60.0,
        "charger_kw": 25.0,
        "temp_c": 75.1,  # > 75 -> invalid!
        "vehicle_age_yr": 3.0,
        "queue_before": 2,
        "vehicle": "car",
        "station_type": "Highway"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rule_session_minutes_below_5_rejected(client):
    """Client rule 2: session_minutes below 5 is rejected with 422 Unprocessable Entity."""
    payload = {
        "battery_start_pct": 35.0,
        "session_minutes": 4.0,  # < 5 -> invalid!
        "charger_kw": 25.0,
        "temp_c": 25.0,
        "vehicle_age_yr": 3.0,
        "queue_before": 2,
        "vehicle": "car",
        "station_type": "Highway"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field_rejected(client):
    """Reject request missing a mandatory schema field."""
    payload = {
        "session_minutes": 60.0,
        "charger_kw": 25.0,
        "temp_c": 25.0,
        "vehicle_age_yr": 3.0,
        "queue_before": 2,
        "vehicle": "car",
        "station_type": "Highway"
        # missing battery_start_pct
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_messy_vehicle_normalized(client):
    """Verify input mapping inside /predict uses clean_vehicle_name to map raw spellings."""
    for messy_name in ["TWO_WHEELER", " car ", "Three_Wheeler", "thr."]:
        payload = {
            "battery_start_pct": 35.0,
            "session_minutes": 60.0,
            "charger_kw": 25.0,
            "temp_c": 25.0,
            "vehicle_age_yr": 3.0,
            "queue_before": 2,
            "vehicle": messy_name,
            "station_type": "Mall"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "prediction" in response.json()

