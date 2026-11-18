"""Stable request and response contracts shared by the API and UI adapter."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TripSummary(StrictModel):
    trip_id: str = Field(pattern=r"^\d{8}$")


class TripListResponse(StrictModel):
    items: list[TripSummary]


class PredictionRequest(StrictModel):
    trip_id: str = Field(pattern=r"^\d{8}$")


class ModelInfo(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class PredictionMetrics(StrictModel):
    """Safe, aggregate-only evaluation values exposed to the evaluator UI.

    A plugin may omit a field when the required server-side reference is not
    available.  Per-observation target arcs must never be placed here.
    """

    top1_accuracy: float | None = Field(default=None, ge=0, le=1)
    route_continuity: float | None = Field(default=None, ge=0, le=1)
    inference_ms: float | None = Field(default=None, ge=0)


class InferencePayload(StrictModel):
    """The only shape a future selected-model adapter may return.

    GeoJSON is intentionally represented as JSON objects so the project does
    not need a second geospatial API dependency.  The service validates the
    expected top-level geometry types before exposing it to the UI.
    """

    model: ModelInfo
    observation_count: int = Field(ge=0)
    predicted_arc_count: int = Field(ge=0)
    observations: dict[str, Any]
    predicted_route: dict[str, Any]
    metrics: PredictionMetrics

    @field_validator("observations")
    @classmethod
    def observations_must_be_feature_collection(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "FeatureCollection":
            raise ValueError("observations must be a GeoJSON FeatureCollection")
        return value

    @field_validator("predicted_route")
    @classmethod
    def route_must_be_linestring_feature(cls, value: dict[str, Any]) -> dict[str, Any]:
        geometry = value.get("geometry", {}) if value.get("type") == "Feature" else {}
        if geometry.get("type") != "LineString":
            raise ValueError("predicted_route must be a GeoJSON LineString Feature")
        return value


class PredictionResponse(InferencePayload):
    trip_id: str = Field(pattern=r"^\d{8}$")


class ServiceStatus(StrictModel):
    api_status: Literal["ok"] = "ok"
    predictor_ready: bool
    evaluation_partition: str
    trip_count: int = Field(ge=0)
    detail: str
