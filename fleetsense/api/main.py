"""FastAPI service for the vessel-type classifier.

Only ever calls predict() from fleetsense.models.inference — no model logic
lives here, this file is purely the HTTP contract on top of that function.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.append(str(Path("..").resolve()))

from fleetsense.api.schemas import PredictionResponse, VesselFeatures
from fleetsense.model.base_model import predict_baseline_proba

app = FastAPI(title="FleetSense Vessel Classifier")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: VesselFeatures) -> PredictionResponse:
    try:
        result = predict_baseline_proba(features.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return PredictionResponse(**result)
