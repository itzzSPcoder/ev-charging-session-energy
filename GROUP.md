# Group G171

**Course:** SCSE3040 MLOps  
**Problem:** **MP298** --- EV Charging Session Energy (Dehradun Green Charge Network)  
**Faculty Mentor:** Dr. Richa Sharma  
**Batch:** 29  

---

### Team Members & Stage Allocations

| Enrolment No. | Name | Email | Role | Stage | GitHub |
|---|---|---|---|---|---|
| **S24CSEU0542** | Siddharth Rawat | s24cseu0542@bennett.edu.in | Data Lead | Stage 1 (DVC, 7 Data Issue Detection, Cleaning Pipeline) | [@siddrwt](https://github.com/siddrwt) |
| **S24CSEU0591** | Tanisha Rathore | tanisharathore@bennett.edu.in | Model Lead | Stage 2 (MLflow Tracking, Model Registry, Drift & Retraining) | [@itzzSPcoder](https://github.com/itzzSPcoder) |
| **S24CSEU0560** | Sparsh Bansal | itzzsparsh003@gmail.com | Ops / Deployment Lead | Stage 3 (FastAPI Service, Docker, K8s, CI/CD Pipeline) | [@itzzSPcoder](https://github.com/itzzSPcoder) |

---

### Stage 1 (Data Lead) Deliverables Summary
- **DVC Tracking:** `data/raw/train.csv.dvc` and `data/raw/later.csv.dvc` tracked under DVC (raw CSV files strictly excluded from git tracking)
- **Data Issue Hunter & Cleaning Module:** `src/ev_charging/data.py` detecting and handling 7 data flaws:
  1. Target leakage column (`kwh_billed_in_app`)
  2. 40 duplicate records
  3. Stuck sensor period in `battery_start_pct` (2026-03-13 to 2026-04-05)
  4. Vehicle spelling variants (20 raw variants normalized to 3 canonical classes: `car`, `two_wheeler`, `three_wheeler`)
  5. Missing values
  6. Unit mixups
  7. Outliers
- **Unit Tests:** `tests/test_data_cleaning.py` verifying vehicle normalization, deduplication, and synthetic issue detection

---

### Stage 2 & 4 (Model Lead) Deliverables Summary
- **Experiment Tracking:** SQLite backend (`sqlite:///mlflow.db`) with 4 tracked candidate runs (Ridge, Random Forest, Gradient Boosting v1 & v2)
- **Model Registry:** Model versioning under `ev_charging_model` with Champion `@champion` alias promotion
- **Data Drift Detection:** Evaluated Population Stability Index (PSI) on `later.csv`; detected major temperature drift on `temp_c` ($PSI = 3.6203 > 0.25$)
- **15% Retraining Promotion Rule:** Retrained Champion Model v2 on 70% split; reduced check split error from 5.8593 to 2.5028 kWh (**57.28% relative MAE improvement**)
- **Holdout Predictions:** Inferred 250 rows in `predictions/holdout_predictions.csv`

---

### Stage 3 & Release (Ops & Service Lead) Deliverables Summary
- **FastAPI Service:** Production inference API (`app/main.py`) with `/predict` and `/health` endpoints
- **Validation Rules:** Enforced `temp_c <= 75.0` and `session_minutes >= 5` boundary rejection with HTTP 422
- **Enterprise Containerization:** Multi-stage `Dockerfile` with layer caching, non-root user, and zero-trust `HEALTHCHECK`
- **Kubernetes Architecture:** 4-replica deployment (`k8s/deployment.yaml`) with readiness/liveness probes and `k8s/service.yaml`
- **CI/CD Automation:** GitHub Actions workflow (`.github/workflows/ci.yml`) with automated pytest gating (16/16 passing)
- **Interactive 3D Simulation:** WebGL-powered 3D EV charging simulator (`app/static/index.html`) with continuous auto-rotation, enterprise light theme, and real cable insertion audio
