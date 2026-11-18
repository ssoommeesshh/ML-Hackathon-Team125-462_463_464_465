"""Runtime discovery and plug-in boundary for PathSnap inference.

No fallback model is implemented here.  Returning invented routes while a
model is undecided would violate the evaluation rules and make the UI appear
to work when it is not running the real pipeline.
"""
from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contracts import InferencePayload, PredictionMetrics

TRIP_ID = re.compile(r"^\d{8}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_root: Path
    evaluation_partition: str
    predictor_factory: str | None

    @classmethod
    def from_environment(cls) -> "Settings":
        root = Path(os.getenv("PATHSNAP_DATA_ROOT", str(PROJECT_ROOT.parent / "Student folder")))
        return cls(
            data_root=root.expanduser().resolve(),
            evaluation_partition=os.getenv("PATHSNAP_EVALUATION_PARTITION", "evaluator_test"),
            predictor_factory=os.getenv("PATHSNAP_PREDICTOR_FACTORY") or None,
        )


class Predictor(Protocol):
    """Interface implemented after model selection and artifact export."""

    name: str
    version: str

    def predict(self, trip_id: str, trip_directory: Path) -> InferencePayload | dict:
        """Run the complete selected-model and decoder pipeline for one trip."""


class UnavailablePredictor:
    """Fail closed until a genuine selected-model adapter is configured."""

    name = "unconfigured"
    version = "unavailable"

    def predict(self, trip_id: str, trip_directory: Path) -> InferencePayload:
        raise RuntimeError(
            "No predictor is configured. Select and export a model, then set "
            "PATHSNAP_PREDICTOR_FACTORY to module:function on Laptop 1."
        )


def discover_trip_ids(settings: Settings) -> list[str]:
    """Discover evaluation-trip folders without reading routes or labels."""
    partition = settings.data_root / settings.evaluation_partition
    if not partition.is_dir():
        return []
    return sorted(
        entry.name for entry in partition.iterdir()
        if entry.is_dir() and TRIP_ID.fullmatch(entry.name)
    )


def trip_directory(settings: Settings, trip_id: str) -> Path:
    if not TRIP_ID.fullmatch(trip_id):
        raise ValueError("trip_id must contain exactly eight digits")
    candidate = (settings.data_root / settings.evaluation_partition / trip_id).resolve()
    partition = (settings.data_root / settings.evaluation_partition).resolve()
    if candidate.parent != partition or not candidate.is_dir():
        raise FileNotFoundError(f"evaluation trip {trip_id} was not found")
    return candidate


def load_predictor(settings: Settings) -> Predictor:
    """Load a team-owned factory named ``package.module:function``.

    The factory receives ``Settings`` and must return an object with ``name``,
    ``version`` and ``predict(trip_id, trip_directory)``.  Keeping this adapter
    separate prevents the frontend contract from changing when the final model,
    scaler, decoder parameters, or artifact path are chosen.
    """
    if not settings.predictor_factory:
        return UnavailablePredictor()
    module_name, separator, factory_name = settings.predictor_factory.partition(":")
    if not separator or not module_name or not factory_name:
        raise RuntimeError("PATHSNAP_PREDICTOR_FACTORY must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), factory_name)
    predictor = factory(settings)
    if not all(hasattr(predictor, name) for name in ("name", "version", "predict")):
        raise RuntimeError("predictor factory returned an object without the required interface")
    return predictor


def run_prediction(predictor: Predictor, trip_id: str, folder: Path) -> InferencePayload:
    payload = predictor.predict(trip_id, folder)
    if isinstance(payload, InferencePayload):
        return payload
    return InferencePayload.model_validate(payload)


def unavailable_metrics() -> PredictionMetrics:
    """Convenience value for model adapters that have no server-side labels."""
    return PredictionMetrics()
