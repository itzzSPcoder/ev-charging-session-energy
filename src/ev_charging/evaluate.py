import pandas as pd
import numpy as np
from src.ev_charging.features import NUMERIC_FEATURES, ALL_FEATURES, TARGET
from src.ev_charging.metrics import compute_mae, compute_psi, compute_slice_mae


def compute_all_psi(reference_df: pd.DataFrame, production_df: pd.DataFrame, num_bins: int = 10) -> dict:
    """Compute PSI for every numeric feature between reference and production."""
    psi_dict = {}
    for col in NUMERIC_FEATURES:
        if col in reference_df.columns and col in production_df.columns:
            psi_val = compute_psi(reference_df[col].values, production_df[col].values, num_bins=num_bins)
            psi_dict[col] = round(psi_val, 4)
    return psi_dict


def evaluate_drift_and_slices(v1_model, cleaned_train_df: pd.DataFrame, later_df: pd.DataFrame) -> dict:
    """
    Perform monitoring evaluations required by Section 7 of PROBLEM.md:
    1. PSI for every numeric column.
    2. Most drifted feature.
    3. v1 MAE on later.csv separately for each station_type.
    4. Worst served group.
    """
    # 1. PSI for each numeric feature
    psi_results = compute_all_psi(cleaned_train_df, later_df, num_bins=10)
    most_drifted = max(psi_results.items(), key=lambda x: x[1])[0]

    # 2. Slice MAE by station_type
    later_eval = later_df.copy()
    later_eval["v1_pred"] = v1_model.predict(later_eval[ALL_FEATURES])
    mae_by_group = compute_slice_mae(later_eval, TARGET, "v1_pred", "station_type")

    # Worst served group has highest MAE
    worst_served = max(mae_by_group.items(), key=lambda x: x[1])[0]

    return {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
    }


def compare_models_for_promotion(
    v1_model,
    v2_model,
    later_check_df: pd.DataFrame,
    threshold: float = 0.15,
) -> dict:
    """
    Compare v1 and v2 on the check split (last 30% of sorted later.csv):
    Promote v2 only if its MAE is at least 15% lower than v1's.
    """
    y_check = later_check_df[TARGET].values
    X_check = later_check_df[ALL_FEATURES]

    v1_pred = v1_model.predict(X_check)
    v2_pred = v2_model.predict(X_check)

    mae_v1 = round(compute_mae(y_check, v1_pred), 4)
    mae_v2 = round(compute_mae(y_check, v2_pred), 4)

    improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improvement >= threshold else "keep"

    return {
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "improvement": round(float(improvement), 4),
        "decision": decision,
    }

