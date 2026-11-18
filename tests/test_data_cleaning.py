import pandas as pd
from src.ev_charging.data import clean_vehicle_name, clean_dataframe, detect_data_issues


def test_clean_vehicle_name_variations():
    """Test that all messy vehicle spelling variants map to the 3 real categories."""
    assert clean_vehicle_name("car") == "car"
    assert clean_vehicle_name("Car") == "car"
    assert clean_vehicle_name("CAR") == "car"
    assert clean_vehicle_name(" car ") == "car"
    assert clean_vehicle_name("car.") == "car"

    assert clean_vehicle_name("two_wheeler") == "two_wheeler"
    assert clean_vehicle_name("Two_Wheeler") == "two_wheeler"
    assert clean_vehicle_name("TWO_WHEELER") == "two_wheeler"
    assert clean_vehicle_name("two wheeler") == "two_wheeler"
    assert clean_vehicle_name("two.") == "two_wheeler"

    assert clean_vehicle_name("three_wheeler") == "three_wheeler"
    assert clean_vehicle_name("Three_Wheeler") == "three_wheeler"
    assert clean_vehicle_name("three wheeler") == "three_wheeler"
    assert clean_vehicle_name("thr.") == "three_wheeler"


def test_clean_dataframe_drops_duplicates():
    """Test that duplicate rows are removed."""
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "battery_start_pct": [30, 30, 40],
        "vehicle": [" car", " car", "TWO_WHEELER"],
    })
    cleaned = clean_dataframe(df, drop_stuck=False)
    assert len(cleaned) == 2
    assert cleaned["vehicle"].tolist() == ["car", "two_wheeler"]


def test_detect_data_issues_synthetic():
    """Test detection logic on synthetic in-memory data."""
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2", "R3"],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"],
        "battery_start_pct": [20, 20, 50, 70],
        "session_minutes": [60, 60, 120, 180],
        "charger_kw": [30.0, 30.0, 50.0, 22.0],
        "temp_c": [25.0, 25.0, 30.0, 28.0],
        "vehicle_age_yr": [2.0, 2.0, 4.0, 1.0],
        "queue_before": [1, 1, 3, 2],
        "vehicle": ["car", "car", "Car", "CAR"],
        "station_type": ["Mall", "Mall", "Highway", "Residential"],
        "energy_kwh": [15.0, 15.0, 35.0, 45.0],
    })
    issues = detect_data_issues(df)
    assert issues["duplicate_rows"]["present"] is True
    assert issues["duplicate_rows"]["count"] == 1
    assert issues["spelling_variants"]["present"] is True
    assert issues["spelling_variants"]["true_levels"] == 1

