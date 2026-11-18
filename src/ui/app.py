"""PathSnap Evaluator Web Interface Application (Laptop 2).

Hosts the evaluator UI dashboard, manages the connection to Laptop 1 (Model & API),
and provides high-speed mock/offline simulation when Laptop 1 is disconnected.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ui.api_contract import EvaluationMetrics, EvaluationRequest, EvaluationResponse
from src.ui.mock_engine import LocalEvaluationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pathsnap.ui")

app = FastAPI(
    title="PathSnap Evaluator Interface",
    description="Evaluator-facing web interface for distributed map-matching (Laptop 2)",
    version="2.0.0",
)

# Enable CORS for cross-laptop requests and flexible local hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = LocalEvaluationEngine(repo_root=REPO_ROOT)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/api/health")
async def health_check():
    """Health check endpoint for Laptop 2 UI server."""
    return {
        "status": "healthy",
        "role": "Laptop 2 (Evaluator UI)",
        "timestamp": time.time(),
        "available_trips": len(engine.list_evaluation_trips()),
    }


@app.get("/api/trips")
async def list_trips():
    """Return the list of ten reserved evaluation trips."""
    return {"trips": engine.list_evaluation_trips()}


@app.post("/api/ping_backend")
async def ping_backend(request: Request):
    """Test HTTP connection to Laptop 1 (Model API)."""
    body = await request.json()
    backend_url = body.get("backend_url", "http://localhost:8000").rstrip("/")

    start = time.perf_counter()
    candidates = [
        f"{backend_url}/health",
    ]

    last_error = ""
    for url in candidates:
        try:
            resp = requests.get(url, timeout=3)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            if resp.status_code == 200:
                health_data = resp.json()
                predictor_ok = health_data.get("predictor_ready", False)
                trip_count = health_data.get("trip_count", 0)
                detail = health_data.get("detail", "")
                return {
                    "connected": True,
                    "target_url": backend_url,
                    "endpoint_reached": url,
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "predictor_ready": predictor_ok,
                    "trip_count": trip_count,
                    "message": f"Connected to Laptop 1 ({latency_ms} ms) — {'predictor ready' if predictor_ok else 'predictor NOT ready'}, {trip_count} trips — {detail}",
                }
        except Exception as e:
            last_error = str(e)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "connected": False,
        "target_url": backend_url,
        "latency_ms": latency_ms,
        "message": f"Could not reach Laptop 1: {last_error}",
    }


@app.post("/api/evaluate")
async def evaluate_trip(
    req: EvaluationRequest,
    mode: str = Query("auto", description="'live', 'mock', or 'auto'"),
    backend_url: Optional[str] = Query(None, description="Laptop 1 API Base URL"),
):
    """Execute trip evaluation either via live Laptop 1 API or local fallback engine."""
    logger.info(f"Evaluation requested: Trip={req.trip_id}, Model={req.model_name}, Mode={mode}")

    # Determine whether to attempt live API call
    should_try_live = mode in ["live", "auto"] and backend_url and backend_url.lower() != "mock"

    if should_try_live:
        target_base = backend_url.rstrip("/")
        prediction_url = f"{target_base}/v1/predictions"
        prediction_payload = {"trip_id": req.trip_id}

        try:
            start_req = time.perf_counter()
            resp = requests.post(prediction_url, json=prediction_payload, timeout=30)
            network_ms = round((time.perf_counter() - start_req) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Laptop 1 API succeeded via {prediction_url} in {network_ms} ms")
                return _adapt_backend_response(data, req, network_ms)

            elif resp.status_code == 503:
                logger.warning(f"Laptop 1 predictor not ready (503): {resp.text}")
                if mode == "live":
                    raise HTTPException(status_code=502, detail="Laptop 1 predictor is not configured yet.")
            elif resp.status_code == 404:
                logger.warning(f"Trip not found on Laptop 1 (404): {resp.text}")
                if mode == "live":
                    raise HTTPException(status_code=404, detail=f"Trip {req.trip_id} not found on Laptop 1.")
            else:
                logger.warning(f"Laptop 1 returned HTTP {resp.status_code}: {resp.text}")
                if mode == "live":
                    raise HTTPException(status_code=502, detail=f"Laptop 1 returned HTTP {resp.status_code}")

        except requests.exceptions.ConnectionError as conn_err:
            logger.info(f"Laptop 1 not running at {target_base}: {conn_err}")
            if mode == "live":
                raise HTTPException(
                    status_code=502,
                    detail=f"Laptop 1 API is not reachable at {target_base}. Ensure server is running.",
                )
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Connection to {prediction_url} failed: {exc}")
            if mode == "live":
                raise HTTPException(
                    status_code=502,
                    detail=f"Laptop 1 API connection failed: {exc}",
                )

    # Fallback to local evaluation engine (Mock / Standalone Mode)
    logger.info(f"Using local evaluation engine for trip {req.trip_id}")
    local_resp = engine.evaluate_trip_locally(
        trip_id=req.trip_id,
        model_name=req.model_name or "Boosting",
        include_geometry=req.include_geometry,
        include_ground_truth=req.include_ground_truth,
        include_road_arcs=req.include_road_arcs,
    )
    return local_resp.model_dump()


def _adapt_backend_response(raw_data: Dict[str, Any], req: EvaluationRequest, latency_ms: float) -> JSONResponse:
    """Transform backend PredictionResponse (GeoJSON) into the UI's EvaluationResponse schema."""
    metrics_data = raw_data.get("metrics", {})
    model_info = raw_data.get("model", {})
    observation_count = int(raw_data.get("observation_count", 0))
    predicted_arc_count = int(raw_data.get("predicted_arc_count", 0))

    # Backend returns route_continuity and inference_ms; top1_accuracy may be null.
    # For live API, the decoded values ARE the values (backend runs Viterbi internally).
    backend_continuity = metrics_data.get("route_continuity")
    backend_top1 = metrics_data.get("top1_accuracy")
    backend_inference_ms = metrics_data.get("inference_ms", latency_ms)

    metrics = EvaluationMetrics(
        top1_accuracy=float(backend_top1) if backend_top1 is not None else None,
        decoded_top1_accuracy=float(backend_top1) if backend_top1 is not None else None,
        route_continuity=float(backend_continuity) if backend_continuity is not None else 0.0,
        decoded_continuity=float(backend_continuity) if backend_continuity is not None else 0.0,
        macro_f1=None,
        log_loss=None,
        inference_time_ms=float(backend_inference_ms),
        total_points=observation_count,
        clear_points=observation_count,
        candidate_misses=0,
    )

    # Parse GeoJSON observations → raw_gps list of {index, lat, lon, timestamp}
    raw_gps = []
    observations = raw_data.get("observations", {})
    if observations.get("type") == "FeatureCollection":
        for feat in observations.get("features", []):
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [0, 0])  # [lon, lat]
                raw_gps.append({
                    "index": props.get("index", 0),
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "timestamp": float(props.get("timestamp", 0)),
                })

    # Parse GeoJSON predicted_route LineString → [[lat, lon], ...] for Leaflet
    predicted_route_coords = []
    predicted_arc_ids = []
    pred_route = raw_data.get("predicted_route", {})
    if pred_route.get("type") == "Feature":
        route_geom = pred_route.get("geometry", {})
        route_props = pred_route.get("properties", {})
        if route_geom.get("type") == "LineString":
            for coord in route_geom.get("coordinates", []):
                # GeoJSON [lon, lat] → Leaflet [lat, lon]
                predicted_route_coords.append([float(coord[1]), float(coord[0])])
        predicted_arc_ids = list(range(route_props.get("arc_count", predicted_arc_count)))

    # Compute bounding box from GPS observations
    bounds = None
    if raw_gps:
        lats = [p["lat"] for p in raw_gps]
        lons = [p["lon"] for p in raw_gps]
        bounds = [min(lats), min(lons), max(lats), max(lons)]

    resp = EvaluationResponse(
        status="success",
        message=f"Live prediction from {model_info.get('name', 'Unknown')} v{model_info.get('version', '?')} via Laptop 1.",
        trip_id=raw_data.get("trip_id", req.trip_id),
        partition="student_test",
        model_used=model_info.get("name", req.model_name or "Custom"),
        metrics=metrics,
        raw_gps=raw_gps,
        predicted_route_coords=predicted_route_coords,
        predicted_arc_ids=predicted_arc_ids,
        ground_truth_coords=None,  # Backend never exposes ground truth per eval rules
        ground_truth_arc_ids=None,
        bounds=bounds,
        execution_info={
            "mode": "live_distributed_api",
            "host": "Laptop 1",
            "latency_ms": latency_ms,
            "model_name": model_info.get("name"),
            "model_version": model_info.get("version"),
            "observation_count": observation_count,
            "predicted_arc_count": predicted_arc_count,
        },
    )
    return JSONResponse(content=resp.model_dump())


# Mount static assets directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/style.css")
async def get_css():
    """Direct route for stylesheet."""
    return FileResponse(str(STATIC_DIR / "style.css"), media_type="text/css")


@app.get("/app.js")
async def get_js():
    """Direct route for client JavaScript."""
    return FileResponse(str(STATIC_DIR / "app.js"), media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>PathSnap UI static/index.html not found.</h3>", status_code=404)
    return FileResponse(str(index_file))


def main():
    """CLI launcher for Laptop 2 UI server."""
    parser = argparse.ArgumentParser(description="PathSnap Evaluator User Interface Server (Laptop 2)")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Port to run UI on (default: 8501)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Target Laptop 1 API URL")
    parser.add_argument("--browser", action="store_true", help="Automatically open web browser on startup")
    args = parser.parse_args()

    local_url = f"http://localhost:{args.port}"
    print("=" * 70)
    print(" 🚀 PathSnap: Distributed Map-Matching & Route Reconstruction")
    print("    Laptop 2 (Evaluator User Interface)")
    print("=" * 70)
    print(f" * UI Dashboard:    {local_url}")
    print(f" * Target API Host: {args.api_url}")
    print(f" * Network Binding: http://{args.host}:{args.port}")
    print("=" * 70)

    if args.browser:
        webbrowser.open(local_url)

    uvicorn.run("src.ui.app:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
