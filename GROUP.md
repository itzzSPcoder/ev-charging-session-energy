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
- **DVC Tracking:** `data/raw/train.csv.dvc` and `data/raw/later.csv.dvc` tracked under DVC
- **Data Issue Hunter & Cleaning Module:** `src/ev_charging/data.py` detecting and handling 7 data flaws:
  1. Target leakage column (`kwh_billed_in_app`)
  2. 40 duplicate records
  3. Stuck sensor period in `battery_start_pct` (2026-03-13 to 2026-04-05)
  4. Vehicle spelling variants (20 raw variants normalized to 3 canonical classes: `car`, `two_wheeler`, `three_wheeler`)
  5. Missing values
  6. Unit mixups
  7. Outliers
- **Unit Tests:** `tests/test_data_cleaning.py` verifying vehicle normalization, deduplication, and synthetic issue detection
