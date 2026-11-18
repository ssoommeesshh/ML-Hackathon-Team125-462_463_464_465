"""FastAPI entry point for the Laptop 1 PathSnap inference service."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from .contracts import PredictionRequest, PredictionResponse, ServiceStatus, TripListResponse, TripSummary
from .service import Settings, discover_trip_ids, load_predictor, run_prediction, trip_directory

logging.basicConfig(level=os.getenv("PATHSNAP_LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("pathsnap.api")

settings = Settings.from_environment()
allowed_origins = [origin.strip() for origin in os.getenv("PATHSNAP_ALLOWED_ORIGINS", "").split(",") if origin.strip()]

app = FastAPI(title="PathSnap Inference API", version="0.1.0")
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@lru_cache(maxsize=1)
def predictor():
    return load_predictor(settings)


def predictor_ready() -> bool:
    if not settings.predictor_factory:
        return False
    try:
        predictor()
    except Exception:
        logger.exception("Predictor configuration could not be loaded")
        return False
    return True


@app.get("/health", response_model=ServiceStatus, tags=["operations"])
def health() -> ServiceStatus:
    trips = discover_trip_ids(settings)
    ready = predictor_ready()
    detail = "predictor configured" if ready else "predictor unavailable; prediction requests fail closed"
    return ServiceStatus(
        predictor_ready=ready,
        evaluation_partition=settings.evaluation_partition,
        trip_count=len(trips),
        detail=detail,
    )


@app.get("/v1/trips", response_model=TripListResponse, tags=["inference"])
def list_trips() -> TripListResponse:
    return TripListResponse(items=[TripSummary(trip_id=trip_id) for trip_id in discover_trip_ids(settings)])


@app.post("/v1/predictions", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest, raw_request: Request) -> PredictionResponse:
    if not predictor_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No selected-model predictor is configured on this server.",
        )
    try:
        folder = trip_directory(settings, request.trip_id)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    try:
        result = run_prediction(predictor(), request.trip_id, folder)
    except Exception as error:
        logger.exception("Prediction failed for trip %s from %s", request.trip_id, raw_request.client.host if raw_request.client else "unknown")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server-side inference failed; inspect Laptop 1 logs.") from error
    logger.info("Prediction completed for trip %s from %s", request.trip_id, raw_request.client.host if raw_request.client else "unknown")
    return PredictionResponse(trip_id=request.trip_id, **result.model_dump())
