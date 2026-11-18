import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def compute_mae(y_true, y_pred) -> float:
    """Compute Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_psi(reference: np.ndarray, production: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between reference and production
    distributions using quantile bins derived from the reference data.
    """
    ref_arr = np.asarray(reference, dtype=float)
    prod_arr = np.asarray(production, dtype=float)

    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref_arr, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    ref_counts, _ = np.histogram(ref_arr, bins=bins)
    prod_counts, _ = np.histogram(prod_arr, bins=bins)

    ref_pct = ref_counts / len(ref_arr)
    prod_pct = prod_counts / len(prod_arr)

    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    prod_pct = np.where(prod_pct == 0, eps, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


def compute_slice_mae(df: pd.DataFrame, target_col: str, pred_col: str, slice_col: str) -> dict:
    """Compute MAE for each subset defined by slice_col."""
    results = {}
    for group_val, group_df in df.groupby(slice_col):
        mae = compute_mae(group_df[target_col], group_df[pred_col])
        results[str(group_val)] = round(float(mae), 4)
    return results

