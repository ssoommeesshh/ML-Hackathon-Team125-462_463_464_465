"""HistGradientBoosting predictor adapter for the PathSnap inference API.

This module implements the predictor-factory contract defined in
``src/api/service.py``.  It trains the HistGradientBoosting classifier on the
full 80-trip training partition at startup, then runs the complete pipeline for
any evaluation trip:  GPS cleaning → candidate extraction → model scoring →
HMM/Viterbi decoding → GeoJSON output.

Connect it by setting the environment variable before starting Laptop 1:

    $env:PATHSNAP_PREDICTOR_FACTORY = "src.api.predictor_boosting:create_predictor"
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.api.contracts import InferencePayload, ModelInfo, PredictionMetrics
from src.data.prepare_candidates import FEATURES, extract

logger = logging.getLogger("pathsnap.predictor")

SEED = 125
INPUTS = FEATURES + ["distance_above_best_m"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = PROJECT_ROOT / "results" / "ordered_pretraining"


# ---------------------------------------------------------------------------
# GPS cleaning (inline, simplified from clean_tracks for single-trip usage)
# ---------------------------------------------------------------------------

GAP_THRESHOLD_S = 10.0
SPEED_FLAG_KMH = 180.0


def _haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def clean_track_from_file(track_path: Path) -> pd.DataFrame:
    """Conservative GPS cleaning for a single trip's .track file."""
    rows = []
    seg_id = -1
    force_new = True
    prev = None

    for idx, line in enumerate(track_path.read_text(errors="replace").splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            force_new = True
            prev = None
            continue
        try:
            lon, lat, ts = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            force_new = True
            prev = None
            continue
        if not (math.isfinite(lon) and math.isfinite(lat) and math.isfinite(ts)):
            force_new = True
            prev = None
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            force_new = True
            prev = None
            continue

        # Exact duplicate check
        if prev is not None:
            if lon == prev["longitude"] and lat == prev["latitude"] and ts == prev["timestamp_seconds"]:
                continue

        new_seg = force_new
        dt = None
        speed = None
        if prev is not None and not force_new:
            dt = ts - prev["timestamp_seconds"]
            if dt <= 0:
                new_seg = True
            elif dt > GAP_THRESHOLD_S:
                new_seg = True
            else:
                d_km = _haversine_km(prev["longitude"], prev["latitude"], lon, lat)
                speed = d_km / dt * 3600

        if new_seg:
            seg_id += 1
            dt = None
            speed = None

        row = {
            "original_row_index": idx,
            "segment_id": seg_id,
            "longitude": lon,
            "latitude": lat,
            "timestamp_seconds": ts,
            "delta_t_seconds": dt if dt is not None else 0.0,
            "is_segment_start": new_seg,
            "implied_speed_kmh": speed if speed is not None else 0.0,
            "speed_flag": bool(speed is not None and speed > SPEED_FLAG_KMH),
        }
        rows.append(row)
        prev = row
        force_new = False

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Viterbi HMM decoder (from evaluate_families.py)
# ---------------------------------------------------------------------------

def viterbi_decode(frame: pd.DataFrame, arcs: np.ndarray, prob: np.ndarray):
    """Run Viterbi sequence decoding on candidate probabilities."""
    group = frame.groupby("original_row_index", sort=False)
    slot = group.cumcount().to_numpy()
    obs = group.ngroup().to_numpy()
    shape = (obs[-1] + 1, slot.max() + 1)

    ids = np.full(shape, -1, dtype=int)
    ids[obs, slot] = frame.arc_id.to_numpy()
    emission = np.full(shape, -np.inf)
    emission[obs, slot] = np.log(np.clip(prob, 1e-8, 1))

    raw = ids[np.arange(len(ids)), emission.argmax(axis=1)]

    rows = frame.iloc[np.flatnonzero(slot == 0)]
    pieces = np.split(
        np.arange(len(ids)),
        np.flatnonzero(np.diff(rows.segment_id.to_numpy())) + 1,
    )

    path = raw.copy()
    for ix in pieces:
        scores = emission[ix[0]].copy()
        back = []
        for i in ix[1:]:
            prev_ids = ids[i - 1, :, None]
            curr_ids = ids[i, None, :]
            connected = (prev_ids == curr_ids) | (arcs[prev_ids, 1] == arcs[curr_ids, 0])
            options = scores[:, None] - 2.0 * (~connected)
            parents = options.argmax(axis=0)
            scores = options.max(axis=0) + emission[i]
            back.append(parents)

        state = scores.argmax()
        for offset in range(len(ix) - 1, -1, -1):
            path[ix[offset]] = ids[ix[offset], state]
            if offset:
                state = back[offset - 1][state]

    return rows, raw, path


# ---------------------------------------------------------------------------
# GeoJSON construction helpers
# ---------------------------------------------------------------------------

def _build_observations_geojson(gps_df: pd.DataFrame) -> dict:
    """Build GeoJSON FeatureCollection from GPS observations."""
    features = []
    for _, row in gps_df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": {
                "index": int(row["original_row_index"]),
                "timestamp": float(row["timestamp_seconds"]),
                "segment_id": int(row["segment_id"]),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _build_route_geojson(arc_ids: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> dict:
    """Build GeoJSON LineString Feature from a sequence of arc IDs."""
    coords = []
    for aid in arc_ids:
        if aid < 0 or aid >= len(arcs):
            continue
        u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
        from_pt = [float(nodes[u, 0]), float(nodes[u, 1])]  # [lon, lat]
        to_pt = [float(nodes[v, 0]), float(nodes[v, 1])]
        if not coords or coords[-1] != from_pt:
            coords.append(from_pt)
        coords.append(to_pt)
    if len(coords) < 2:
        coords = [[0.0, 0.0], [0.0, 0.0]]
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"arc_count": int(len(arc_ids))},
    }


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class BoostingPredictor:
    """HistGradientBoosting predictor implementing the Predictor protocol."""

    name = "HistGradientBoosting"
    version = "bayesian-tuned-v1-seed125"

    def __init__(self, model: HistGradientBoostingClassifier, data_root: Path):
        self._model = model
        self._data_root = data_root

    def predict(self, trip_id: str, trip_directory: Path) -> InferencePayload:
        """Run the full pipeline for one trip and return an InferencePayload."""
        t0 = perf_counter()

        # 1. Load raw trip files
        track_path = trip_directory / f"{trip_id}.track"
        nodes_path = trip_directory / f"{trip_id}.nodes"
        arcs_path = trip_directory / f"{trip_id}.arcs"

        if not track_path.exists():
            raise FileNotFoundError(f"Track file not found: {track_path}")

        nodes = np.loadtxt(str(nodes_path), ndmin=2)
        arcs = np.loadtxt(str(arcs_path), dtype=int, ndmin=2)

        # 2. Clean GPS track
        gps_df = clean_track_from_file(track_path)
        if gps_df.empty:
            raise ValueError(f"No valid GPS observations in trip {trip_id}")

        logger.info("Trip %s: %d GPS points after cleaning", trip_id, len(gps_df))

        # 3. Extract candidate features
        features, xy, starts, vectors = extract(gps_df, nodes, arcs)
        features["distance_above_best_m"] = (
            features.distance_m
            - features.groupby("original_row_index").distance_m.transform("min")
        )

        logger.info(
            "Trip %s: %d candidate rows, %d observations",
            trip_id, len(features), features.original_row_index.nunique(),
        )

        # 4. Score candidates with HistGradientBoosting
        prob = self._model.predict_proba(features[INPUTS])[:, 1]

        # 5. Viterbi HMM decode
        rows, raw_arc_ids, decoded_arc_ids = viterbi_decode(features, arcs, prob)

        inference_ms = (perf_counter() - t0) * 1000
        logger.info("Trip %s: inference completed in %.1f ms", trip_id, inference_ms)

        # 6. Compute route continuity metric
        linked = np.diff(rows.segment_id.to_numpy()) == 0
        if linked.any():
            decoded_cont = float(
                (
                    (decoded_arc_ids[:-1] == decoded_arc_ids[1:])
                    | (arcs[decoded_arc_ids[:-1], 1] == arcs[decoded_arc_ids[1:], 0])
                )[linked].mean()
            )
        else:
            decoded_cont = 1.0

        # 7. Build GeoJSON outputs
        observations_geojson = _build_observations_geojson(gps_df)
        predicted_route_geojson = _build_route_geojson(decoded_arc_ids, nodes, arcs)

        # 8. Build metrics — no reference route available for evaluator trips,
        #    so top1_accuracy is omitted per the API contract
        metrics = PredictionMetrics(
            route_continuity=round(decoded_cont, 4),
            inference_ms=round(inference_ms, 2),
        )

        return InferencePayload(
            model=ModelInfo(name=self.name, version=self.version),
            observation_count=len(gps_df),
            predicted_arc_count=len(np.unique(decoded_arc_ids)),
            observations=observations_geojson,
            predicted_route=predicted_route_geojson,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Factory function (entry point for PATHSNAP_PREDICTOR_FACTORY)
# ---------------------------------------------------------------------------

def create_predictor(settings) -> BoostingPredictor:
    """Train HistGradientBoosting on all training data and return a Predictor.

    This is the factory function referenced by:
        PATHSNAP_PREDICTOR_FACTORY=src.api.predictor_boosting:create_predictor
    """
    logger.info("Training HistGradientBoosting on full training partition...")
    t0 = perf_counter()

    manifest = pd.read_csv(STORE / "manifest.csv", dtype={"trip_id": str})
    train_manifest = manifest[manifest.partition == "train"]

    # Collect training data from all 80 training trips
    all_frames = []
    for _, row in train_manifest.iterrows():
        trip = row.trip_id
        f = pd.read_pickle(STORE / "features" / "train" / (trip + ".pkl"))
        f["distance_above_best_m"] = (
            f.distance_m - f.groupby("original_row_index").distance_m.transform("min")
        )
        labels = pd.read_pickle(STORE / "targets" / "train" / (trip + ".pkl"))
        merged = f.merge(labels, on="original_row_index", validate="many_to_one")

        # Sample eligible training rows (clear & in candidates)
        eligible = merged.loc[~merged.ambiguous & merged.in_candidates]
        ids = eligible.original_row_index.unique()
        rng = np.random.default_rng(SEED + int(trip))
        chosen = rng.choice(ids, min(128, len(ids)), replace=False)
        all_frames.append(eligible[eligible.original_row_index.isin(chosen)])

    train_data = pd.concat(all_frames, ignore_index=True)
    X = train_data[INPUTS]
    y = (train_data.arc_id == train_data.proxy_arc_id)

    # Bayesian-tuned hyperparameters (results/bayesian_tuning/best_hyperparameters.json)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06872300152781644,
        max_iter=140,
        max_leaf_nodes=19,
        min_samples_leaf=37,
        l2_regularization=0.007177871455795662,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(X, y)

    elapsed = perf_counter() - t0
    logger.info(
        "HistGradientBoosting trained on %d rows from %d trips in %.1f s",
        len(train_data), len(train_manifest), elapsed,
    )

    return BoostingPredictor(model=model, data_root=settings.data_root)
