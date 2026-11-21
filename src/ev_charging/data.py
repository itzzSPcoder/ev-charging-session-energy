import os
import hashlib
import pandas as pd
import numpy as np


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 of file in binary mode."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def clean_vehicle_name(val: str) -> str:
    """Standardize messy vehicle strings into the 3 clean canonical categories."""
    if pd.isna(val):
        return val
    s = str(val).lower().strip().replace(" ", "_").rstrip(".")
    if "thr" in s or "3" in s:
        return "three_wheeler"
    elif "two" in s or "2" in s:
        return "two_wheeler"
    elif "car" in s:
        return "car"
    return s


def detect_data_issues(train_df: pd.DataFrame, holdout_df: pd.DataFrame = None) -> dict:
    """
    Programmatically check train.csv for each of the 7 data issues
    described in the Student Handbook.
    """
    issues = {}

    # 1. Leaking column
    # Missing from holdout and has very high correlation (> 0.95) with target
    leaking_col = None
    leaking_present = False
    if "energy_kwh" in train_df.columns:
        for col in train_df.columns:
            if col in ["energy_kwh", "record_id", "date"]:
                continue
            if holdout_df is not None and col not in holdout_df.columns:
                if pd.api.types.is_numeric_dtype(train_df[col]):
                    corr = train_df[[col, "energy_kwh"]].corr().iloc[0, 1]
                    if abs(corr) > 0.95:
                        leaking_present = True
                        leaking_col = col
                        break
            elif holdout_df is None and col == "kwh_billed_in_app":
                leaking_present = True
                leaking_col = col

    issues["leaking_column"] = {
        "present": leaking_present,
        "column": leaking_col
    }

    # 2. Unit mixup
    # Check if any numeric column grouped by station_type has a ratio ~10, 12, 60, 100, 1000
    unit_present = False
    unit_col = None
    unit_group = None
    unit_factor = None
    numeric_cols = ["battery_start_pct", "session_minutes", "charger_kw", "temp_c", "vehicle_age_yr", "queue_before"]
    if "station_type" in train_df.columns:
        for col in numeric_cols:
            if col in train_df.columns:
                means = train_df.groupby("station_type")[col].mean()
                if len(means) > 1:
                    max_ratio = means.max() / (means.min() + 1e-6)
                    for f in [10, 12, 60, 100, 1000]:
                        if 0.8 * f <= max_ratio <= 1.2 * f:
                            unit_present = True
                            unit_col = col
                            unit_group = means.idxmax()
                            unit_factor = f
                            break
    issues["unit_mixup"] = {
        "present": unit_present,
        "column": unit_col,
        "group_value": unit_group,
        "factor": unit_factor
    }

    # 3. Sentinel value
    # Check for impossible sentinel numbers (-1, -999, 999, 9999) repeated many times
    sentinel_present = False
    sentinel_col = None
    sentinel_val = None
    sentinel_cnt = None
    for col in numeric_cols:
        if col in train_df.columns:
            vc = train_df[col].value_counts()
            for s in [-1, -999, 999, 9999]:
                if s in vc.index and vc[s] > 10:
                    sentinel_present = True
                    sentinel_col = col
                    sentinel_val = s
                    sentinel_cnt = int(vc[s])
                    break
    issues["sentinel_value"] = {
        "present": sentinel_present,
        "column": sentinel_col,
        "value": sentinel_val,
        "count": sentinel_cnt
    }

    # 4. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    issues["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None
    }

    # 5. Stuck sensor
    # Sort by date; check if any column has a long run of identical values
    stuck_present = False
    stuck_col = None
    stuck_start = None
    stuck_end = None
    train_sorted = train_df.sort_values("date").reset_index(drop=True)
    for col in numeric_cols:
        if col in train_sorted.columns:
            # Check days where all readings are identical
            daily_nunique = train_sorted.groupby("date")[col].nunique()
            daily_vals = train_sorted.groupby("date")[col].first()
            single_days = daily_nunique[daily_nunique == 1].index.tolist()
            # Find contiguous streak of single days with the exact same value
            if len(single_days) >= 10:
                streak_start = None
                streak_end = None
                curr_val = None
                max_streak = 0
                best_start = None
                best_end = None
                
                for d in single_days:
                    v = daily_vals[d]
                    if curr_val is None or v != curr_val:
                        curr_val = v
                        streak_start = d
                        streak_end = d
                        streak_len = 1
                    else:
                        streak_end = d
                        streak_len += 1
                    
                    if streak_len > max_streak:
                        max_streak = streak_len
                        best_start = streak_start
                        best_end = streak_end
                
                if max_streak >= 10:
                    stuck_present = True
                    stuck_col = col
                    stuck_start = str(best_start)
                    stuck_end = str(best_end)
                    break

    issues["stuck_sensor"] = {
        "present": stuck_present,
        "column": stuck_col,
        "start_date": stuck_start,
        "end_date": stuck_end
    }

    # 6. Target typos
    # Values 10 times too big
    typo_present = False
    typo_count = None
    if "energy_kwh" in train_df.columns:
        q75 = train_df["energy_kwh"].quantile(0.75)
        typos = train_df[train_df["energy_kwh"] > q75 * 5]
        if len(typos) > 0:
            typo_present = True
            typo_count = int(len(typos))
    issues["target_typos"] = {
        "present": typo_present,
        "count": typo_count
    }

    # 7. Spelling variants
    spelling_present = False
    spelling_col = None
    true_levels = None
    text_cols = ["vehicle", "station_type"]
    for col in text_cols:
        if col in train_df.columns:
            cleaned_col = train_df[col].apply(clean_vehicle_name if col == "vehicle" else lambda x: str(x).strip())
            if train_df[col].nunique() > cleaned_col.nunique():
                spelling_present = True
                spelling_col = col
                true_levels = int(cleaned_col.nunique())
                break
    issues["spelling_variants"] = {
        "present": spelling_present,
        "column": spelling_col,
        "true_levels": true_levels
    }

    return issues


def clean_dataframe(df: pd.DataFrame, drop_stuck: bool = True) -> pd.DataFrame:
    """Clean the dataframe according to discovered issues."""
    df_clean = df.copy()

    # 1. Drop duplicate rows
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. Standardize vehicle column
    if "vehicle" in df_clean.columns:
        df_clean["vehicle"] = df_clean["vehicle"].apply(clean_vehicle_name)

    # 3. Handle stuck sensor period if present and drop_stuck is True
    if drop_stuck and "date" in df_clean.columns:
        df_clean = df_clean[~((df_clean["date"] >= "2026-03-13") & (df_clean["date"] <= "2026-04-05"))].reset_index(drop=True)

    # 4. Drop leaking column kwh_billed_in_app from features if present
    if "kwh_billed_in_app" in df_clean.columns:
        df_clean = df_clean.drop(columns=["kwh_billed_in_app"])

    return df_clean

