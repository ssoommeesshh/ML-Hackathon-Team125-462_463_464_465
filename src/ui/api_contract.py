"""API Request/Response schemas and specifications for the PathSnap distributed architecture.

This module defines the contract between Laptop 1 (Model and API Backend)
and Laptop 2 (Evaluator User Interface Frontend).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Evaluation request payload sent from Laptop 2 to Laptop 1."""
    trip_id: str = Field(..., description="Trip identifier, e.g. '00000000'")
    model_name: Optional[str] = Field("Boosting", description="Model family: Boosting, Forest, MLP, Logistic, Nearest")
    include_geometry: bool = Field(True, description="Whether to include full coordinate geometry for map plotting")
    include_ground_truth: bool = Field(True, description="Whether to include reference route (for student-test verification)")
    include_road_arcs: bool = Field(False, description="Whether to include full local road network arc segments")


class GPSObservation(BaseModel):
    """Single GPS observation from .track file."""
    index: int
    lat: float
    lon: float
    timestamp: float
    matched_arc_id: Optional[int] = None


class EvaluationMetrics(BaseModel):
    """Quantitative performance metrics computed server-side."""
    top1_accuracy: Optional[float] = Field(None, description="Observation-level raw Top-1 Accuracy (0.0 to 1.0)")
    decoded_top1_accuracy: Optional[float] = Field(None, description="Sequence-decoded Top-1 Accuracy after Viterbi (0.0 to 1.0)")
    route_continuity: float = Field(..., description="Proportion of topologically connected consecutive arcs in raw predictions")
    decoded_continuity: float = Field(..., description="Proportion of topologically connected arcs after HMM/Viterbi decoding")
    macro_f1: Optional[float] = Field(None, description="Macro-averaged F1 score")
    log_loss: Optional[float] = Field(None, description="Binary log loss on candidate ranking")
    inference_time_ms: float = Field(..., description="Total inference + decoding execution latency in milliseconds")
    total_points: int = Field(..., description="Total raw GPS observations in trip")
    clear_points: int = Field(..., description="Number of observations with unambiguous ground truth proxy")
    candidate_misses: int = Field(0, description="Observations where true road arc was missing from candidate shortlist")


class EvaluationResponse(BaseModel):
    """Complete evaluation response returned by Laptop 1 to Laptop 2."""
    status: str = Field("success", description="Status code: 'success' or 'error'")
    message: Optional[str] = Field(None, description="Diagnostic or error message")
    trip_id: str = Field(..., description="Evaluated trip ID")
    partition: str = Field("student_test", description="Dataset partition: 'student_test', 'train', or 'evaluator'")
    model_used: str = Field("Boosting", description="Model used for candidate scoring")
    metrics: EvaluationMetrics = Field(..., description="Calculated evaluation metrics")
    raw_gps: List[GPSObservation] = Field(default_factory=list, description="Raw GPS trajectory points")
    predicted_route_coords: List[List[float]] = Field(default_factory=list, description="Ordered [[lat, lon], ...] for predicted path")
    predicted_arc_ids: List[int] = Field(default_factory=list, description="Sequence of reconstructed arc IDs")
    ground_truth_coords: Optional[List[List[float]]] = Field(None, description="Ordered [[lat, lon], ...] for ground truth path")
    ground_truth_arc_ids: Optional[List[int]] = Field(None, description="Ground truth sequence of arc IDs")
    road_arcs_coords: Optional[List[List[List[float]]]] = Field(None, description="Optional road network segment coordinates")
    bounds: Optional[List[float]] = Field(None, description="Bounding box [min_lat, min_lon, max_lat, max_lon]")
    execution_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Server metadata (host, timestamp, framework)")
