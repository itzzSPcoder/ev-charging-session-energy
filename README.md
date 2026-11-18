# SCSE3040 MLOps Mini Project --- Group G171

## Project Metadata

- **Group ID**: G171
- **Problem ID**: MP298
- **Problem Title**: EV Charging Session Energy (Dehradun Green Charge Network)
- **Batch**: 29
- **Course**: SCSE3040 MLOps
- **Faculty Mentor**: Dr. Richa Sharma
- **Course Coordinator**: Dr. Gaurav Tripathi

### Group Members

| Enrolment No. | Name | Role | Responsibilities |
|---|---|---|---|
| **S24CSEU0560** | Sparsh Bansal | Ops & Service Lead | FastAPI Service, Docker, Kubernetes, CI/CD Pipeline, Structured Logging |
| **S24CSEU0542** | Siddharth Rawat | Data Lead | DVC Setup, 7 Data Issue Checks, Cleaning Code, Data Validation |
| **S24CSEU0591** | Tanisha Rathore | Model Lead | MLflow Tracking, Experiment Registry, Retraining & Drift, Holdout Predictions |

---

## 1. Problem Overview

The client, **Dehradun Green Charge Network**, operates charging stations for two-wheelers, three-wheelers, and electric cars in Dehradun. The objective is to predict `energy_kwh` (energy delivered in kWh) when a vehicle plugs in. Predicting this value accurately enables load balancing across chargers and provides riders upfront cost estimates.

Evaluation Metric: **Mean Absolute Error (MAE)** in kWh.

---

## 2. Repository Layout

```
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow (pytest on push)
├── .dvc/
│   ├── config                     # DVC configuration
│   └── .gitignore
├── app/
│   └── main.py                    # FastAPI service (POST /predict, GET /health, JSON logging)
├── data/
│   └── raw/
│       ├── train.csv.dvc          # DVC tracking for raw training dataset
│       ├── later.csv.dvc          # DVC tracking for post-deployment dataset
│       └── holdout_features.csv   # September 2026 holdout features
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment (mp298-api, 4 replicas)
│   └── service.yaml               # Kubernetes service (mp298-svc, port 80 -> 8000)
├── models/
│   └── model.joblib               # Champion model artifact
├── predictions/
│   └── holdout_predictions.csv    # Final predictions on holdout_features.csv
├── reports/
│   └── findings.yaml              # Official mini project findings form
├── src/
│   └── ev_charging/
│       ├── __init__.py
│       ├── data.py                # Issue detection and data cleaning
│       ├── features.py            # Preprocessor ColumnTransformer
│       ├── metrics.py             # MAE, PSI, slice metrics
│       ├── models.py              # Model pipelines (Ridge, RF, GB)
│       ├── train.py               # MLflow training and registry pipeline
│       ├── evaluate.py            # Drift analysis & model promotion
│       └── predict.py             # Inference predictor
├── tests/
│   ├── __init__.py
│   ├── test_api.py                # TestClient tests for endpoints & validation rules
│   ├── test_data_cleaning.py      # Tests for vehicle mapping & deduplication
│   └── test_features_and_metrics.py # Tests for PSI, MAE, and model pipeline
├── Dockerfile                     # Container definition (python:3.12-slim)
├── .dockerignore
├── findings.yaml                  # Findings form in root
├── GROUP.md                       # Group information
├── mlflow.db                      # SQLite MLflow experiment database
├── params.yaml                    # Configurable pipeline parameters
├── PROBLEM.md                     # Problem specification
├── requirements.txt               # Pinned dependencies
├── STUDENT_HANDBOOK.md            # Student guidelines
└── README.md                      # Documentation
```

---

## 3. Data Issues Discovered in `train.csv`

Through automated programmatic checks in `src/ev_charging/data.py`, we identified:

1. **Leaking Column (`kwh_billed_in_app`)**:
   - Absent in `holdout_features.csv`, strongly correlated ($r \approx 0.998$) with the target `energy_kwh`. Recorded only after session completion. Dropped from features.
2. **Duplicate Rows**:
   - `40` duplicate rows detected (`df.duplicated().sum() == 40`). Removed during data cleaning.
3. **Stuck Sensor (`battery_start_pct`)**:
   - The battery sensor was stuck reporting `47%` continuously across all records from `2026-03-13` to `2026-04-05` (133 records). Identified and handled during cleaning.
4. **Spelling Variants (`vehicle`)**:
   - Contains 20 spelling variants (e.g. `' thr.'`, `'TWO_WHEELER'`, `' car '`, `'three wheeler'`) mapping to `3` true levels: `three_wheeler`, `two_wheeler`, and `car`.
5. **Absent Issues**:
   - `unit_mixup`: None.
   - `sentinel_value`: None.
   - `target_typos`: None (targets are bounded and smoothly distributed between 4.55 and 54.16 kWh).

---

## 4. MLflow Experiment Tracking & Model Registry

- **Tracking URI**: `sqlite:///mlflow.db`
- **Experiment**: `ev_charging_session_energy`
- **Data SHA-256**: `225915736a4f872e9c15a7ed1ef315e78fb8f69180106347c9d1ddf98122c443`
- **Candidate Models (v1)**:
  - Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor (selected best v1, registered as Version 1)
- **Monitoring & Drift (`later.csv`)**:
  - `temp_c` exhibited massive drift ($\text{PSI} \approx 3.62$) due to summer conditions (mean $44.0^\circ\text{C}$ vs $25.2^\circ\text{C}$).
  - Worst-served slice by v1: `Mall` (MAE $\approx 7.09$ kWh).
- **Retraining (v2) & Champion Promotion**:
  - v2 retrained on cleaned `train.csv` + first 70% of `later.csv`.
  - On the final 30% check split:
    - v1 MAE: `5.8593`
    - v2 MAE: `2.5028`
    - Improvement: **57.28%** (far exceeding the 15% promotion threshold).
  - Decision: **Promote**. Version 2 assigned the `@champion` alias and exported to `models/model.joblib`.

---

## 5. Prediction Service & Rules

FastAPI service located at `app/main.py`:
- `GET /health`: Returns `{"status": "ok"}`
- `POST /predict`: Accepts session features and returns `{"prediction": float}`.
- **Client Validation Rules**:
  - `temp_c <= 75.0`: Values $> 75^\circ\text{C}$ rejected with HTTP `422`.
  - `session_minutes >= 5`: Values $< 5$ minutes rejected with HTTP `422`.
- **JSON Structured Logging**: Logs each prediction as a single-line JSON record directly to `stdout`.

---

## 6. How to Run Locally

### Setup & Install
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests
```bash
pytest -v
```

### Train & Evaluate Pipeline
```bash
python -m src.ev_charging.train
```

### Start API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t mp298-api:1.0 .
docker run -p 8000:8000 mp298-api:1.0
```

