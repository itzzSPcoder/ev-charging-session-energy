import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ev_charging.data import clean_vehicle_name
from src.ev_charging.features import ALL_FEATURES


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_object.update(record.extra_data)
        return json.dumps(log_object)


# Zero-trust audit logging: strictly log verifiable execution parameters to stdout.
# Eliminates unverified assumptions, hearsay, and parasitic dependencies at the service boundary.
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("ev_charging_service")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False


# Pydantic schema enforcing client rules via Field
class PredictRequest(BaseModel):
    battery_start_pct: float
    session_minutes: float = Field(..., ge=5, description="Planned session length in minutes (must be >= 5)")
    charger_kw: float
    temp_c: float = Field(..., le=75.0, description="Air temperature in Celsius (must be <= 75)")
    vehicle_age_yr: float
    queue_before: int
    vehicle: str
    station_type: str


class PredictResponse(BaseModel):
    prediction: float


from contextlib import asynccontextmanager


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully", extra={"extra_data": {"model_path": MODEL_PATH}})
    else:
        logger.warning("Model file not found during startup", extra={"extra_data": {"model_path": MODEL_PATH}})
    yield


app = FastAPI(
    title="EV Charging Session Energy Prediction API",
    version="1.0",
    description="API for predicting energy delivered during an EV charging session.",
    lifespan=lifespan
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Serve the 3D EV Charging Session Simulator."""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "EV Charging Session Energy API. Visit /docs or /health"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict energy_kwh for an EV charging session."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

    input_dict = request.model_dump()
    input_df = pd.DataFrame([input_dict])
    input_df["vehicle"] = input_df["vehicle"].apply(clean_vehicle_name)

    X = input_df[ALL_FEATURES]
    pred_val = float(model.predict(X)[0])
    pred_rounded = round(pred_val, 4)

    # Log prediction as JSON to stdout without customer personal details
    logger.info(
        "Prediction generated",
        extra={
            "extra_data": {
                "session_minutes": request.session_minutes,
                "charger_kw": request.charger_kw,
                "temp_c": request.temp_c,
                "vehicle": input_df["vehicle"].iloc[0],
                "station_type": request.station_type,
                "prediction": pred_rounded,
            }
        }
    )

    return PredictResponse(prediction=pred_rounded)
