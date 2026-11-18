import os
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from src.ev_charging.data import (
    compute_file_sha256,
    detect_data_issues,
    clean_dataframe,
    clean_vehicle_name,
)
from src.ev_charging.features import ALL_FEATURES, NUMERIC_FEATURES, TARGET
from src.ev_charging.models import get_model_pipeline
from src.ev_charging.metrics import compute_mae
from src.ev_charging.evaluate import evaluate_drift_and_slices, compare_models_for_promotion


def run_pipeline(params_path: str = "params.yaml"):
    # 1. Load params
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    raw_train_path = params["data"]["raw_train_path"]
    raw_later_path = params["data"]["raw_later_path"]
    raw_holdout_path = params["data"]["raw_holdout_path"]
    predictions_path = params["data"]["predictions_path"]
    model_output_path = params["data"]["model_output_path"]
    findings_output_path = params["data"]["findings_output_path"]

    tracking_uri = params["tracking"]["tracking_uri"]
    experiment_name = params["tracking"]["experiment_name"]
    registered_model_name = params["tracking"]["registered_model_name"]
    promotion_threshold = params["model"]["promotion_threshold"]

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    os.makedirs(os.path.dirname(findings_output_path), exist_ok=True)

    # 2. SHA-256 of raw train.csv
    data_sha256 = compute_file_sha256(raw_train_path)
    print(f"data_sha256: {data_sha256}")

    # 3. Load raw data
    train_raw = pd.read_csv(raw_train_path)
    later_raw = pd.read_csv(raw_later_path)
    holdout_raw = pd.read_csv(raw_holdout_path)

    # 4. Detect data issues
    data_issues = detect_data_issues(train_raw, holdout_raw)
    print("Detected data issues:")
    for k, v in data_issues.items():
        print(f"  {k}: {v}")

    # 5. Clean datasets
    train_clean = clean_dataframe(train_raw, drop_stuck=True)
    later_clean = clean_dataframe(later_raw, drop_stuck=False)
    later_clean = later_clean.sort_values("date").reset_index(drop=True)

    holdout_clean = holdout_raw.copy()
    holdout_clean["vehicle"] = holdout_clean["vehicle"].apply(clean_vehicle_name)

    print(f"Train clean size: {len(train_clean)}, Later size: {len(later_clean)}, Holdout size: {len(holdout_clean)}")

    # 6. Setup MLflow
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Ensure experiment exists
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)

    # 7. Train 3 Candidate Models for v1
    candidate_configs = [
        {"model_type": "ridge", "params": {"alpha": params["model"]["v1_ridge_alpha"], "random_state": 42}},
        {"model_type": "rf", "params": {"n_estimators": params["model"]["v1_rf_n_estimators"], "random_state": 42}},
        {"model_type": "gb", "params": {
            "n_estimators": params["model"]["v1_gb_n_estimators"],
            "learning_rate": params["model"]["v1_gb_learning_rate"],
            "max_depth": params["model"]["v1_gb_max_depth"],
            "random_state": 42
        }},
    ]

    best_v1_run_id = None
    best_v1_model = None
    best_v1_score = float("inf")
    best_v1_type = None

    for config in candidate_configs:
        mtype = config["model_type"]
        mparams = config["params"]

        with mlflow.start_run(experiment_id=exp_id, run_name=f"v1_candidate_{mtype}") as run:
            run_id = run.info.run_id
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", mtype)
            mlflow.log_param("version", "v1")
            for p_k, p_v in mparams.items():
                mlflow.log_param(p_k, p_v)

            pipe = get_model_pipeline(model_type=mtype, **mparams)
            pipe.fit(train_clean[ALL_FEATURES], train_clean[TARGET])

            train_pred = pipe.predict(train_clean[ALL_FEATURES])
            later_pred = pipe.predict(later_clean[ALL_FEATURES])

            mae_train = compute_mae(train_clean[TARGET], train_pred)
            mae_later = compute_mae(later_clean[TARGET], later_pred)

            mlflow.log_metric("mae_train", round(mae_train, 4))
            mlflow.log_metric("mae_later", round(mae_later, 4))

            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            print(f"Run {mtype} ({run_id}): mae_train={mae_train:.4f}, mae_later={mae_later:.4f}")

            # Pick best v1 candidate based on train MAE
            if mae_train < best_v1_score:
                best_v1_score = mae_train
                best_v1_run_id = run_id
                best_v1_model = pipe
                best_v1_type = mtype

    print(f"Best v1 model selected: {best_v1_type} (run_id: {best_v1_run_id})")

    # Register best v1 model as Version 1
    model_uri_v1 = f"runs:/{best_v1_run_id}/model"
    v1_mv = mlflow.register_model(model_uri=model_uri_v1, name=registered_model_name)
    v1_version = v1_mv.version
    print(f"Registered model {registered_model_name} version {v1_version}")

    # 8. Retrain v2 on cleaned train + first 70% of sorted later.csv
    n_70 = int(len(later_clean) * 0.7)
    later_70 = later_clean.iloc[:n_70].copy()
    later_30 = later_clean.iloc[n_70:].copy()

    train_v2 = pd.concat([train_clean, later_70], ignore_index=True)

    with mlflow.start_run(experiment_id=exp_id, run_name="v2_retrained_gb") as run_v2:
        v2_run_id = run_v2.info.run_id
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", "gb")
        mlflow.log_param("version", "v2")
        mlflow.log_param("n_estimators", params["model"]["v1_gb_n_estimators"])
        mlflow.log_param("learning_rate", params["model"]["v1_gb_learning_rate"])
        mlflow.log_param("max_depth", params["model"]["v1_gb_max_depth"])

        pipe_v2 = get_model_pipeline(
            model_type="gb",
            n_estimators=params["model"]["v1_gb_n_estimators"],
            learning_rate=params["model"]["v1_gb_learning_rate"],
            max_depth=params["model"]["v1_gb_max_depth"],
            random_state=42
        )
        pipe_v2.fit(train_v2[ALL_FEATURES], train_v2[TARGET])

        train_pred_v2 = pipe_v2.predict(train_v2[ALL_FEATURES])
        later_pred_v2 = pipe_v2.predict(later_clean[ALL_FEATURES])

        mae_train_v2 = compute_mae(train_v2[TARGET], train_pred_v2)
        mae_later_v2 = compute_mae(later_clean[TARGET], later_pred_v2)

        mlflow.log_metric("mae_train", round(mae_train_v2, 4))
        mlflow.log_metric("mae_later", round(mae_later_v2, 4))

        mlflow.sklearn.log_model(
            pipe_v2,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        print(f"Retrained v2 ({v2_run_id}): mae_train={mae_train_v2:.4f}, mae_later={mae_later_v2:.4f}")

    # Register v2 model as Version 2
    model_uri_v2 = f"runs:/{v2_run_id}/model"
    v2_mv = mlflow.register_model(model_uri=model_uri_v2, name=registered_model_name)
    v2_version = v2_mv.version
    print(f"Registered model {registered_model_name} version {v2_version}")

    # 9. Evaluate Drift, Slices, and Promotion
    eval_results = evaluate_drift_and_slices(best_v1_model, train_clean, later_clean)
    promotion_results = compare_models_for_promotion(
        best_v1_model, pipe_v2, later_30, threshold=promotion_threshold
    )

    decision = promotion_results["decision"]
    print(f"Promotion comparison: {promotion_results}")

    if decision == "promote":
        champion_version = int(v2_version)
        champion_model = pipe_v2
        client.set_registered_model_alias(registered_model_name, "champion", str(champion_version))
        print(f"Promoted v2! Set alias 'champion' to version {champion_version}")
    else:
        champion_version = int(v1_version)
        champion_model = best_v1_model
        client.set_registered_model_alias(registered_model_name, "champion", str(champion_version))
        print(f"Kept v1! Set alias 'champion' to version {champion_version}")

    # 10. Save champion model to models/model.joblib
    joblib.dump(champion_model, model_output_path)
    print(f"Saved champion model to {model_output_path}")

    # 11. Generate holdout predictions
    holdout_predictions = champion_model.predict(holdout_clean[ALL_FEATURES])
    pred_df = pd.DataFrame({
        "record_id": holdout_clean["record_id"],
        "prediction": np.round(holdout_predictions, 4)
    })
    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved {len(pred_df)} holdout predictions to {predictions_path}")

    # 12. Fill in findings.yaml
    findings = {
        "group_id": "G171",
        "problem_id": "MP298",
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": experiment_name,
            "best_run_id": best_v1_run_id,
            "registered_model_name": registered_model_name,
            "champion_version": champion_version,
        },
        "monitoring": {
            "psi": eval_results["psi"],
            "most_drifted_feature": eval_results["most_drifted_feature"],
            "mae_by_group": eval_results["mae_by_group"],
            "worst_served_group": eval_results["worst_served_group"],
            "mae_v1_on_check": promotion_results["mae_v1_on_check"],
            "mae_v2_on_check": promotion_results["mae_v2_on_check"],
            "decision": decision,
        }
    }

    # Save to both reports/findings.yaml and findings.yaml
    for out_p in [findings_output_path, "findings.yaml"]:
        with open(out_p, "w") as f:
            yaml.dump(findings, f, sort_keys=False, default_flow_style=False)
        print(f"Updated findings saved to {out_p}")

    return findings


if __name__ == "__main__":
    run_pipeline()

