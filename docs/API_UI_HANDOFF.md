# API and evaluator UI handoff

This scaffold implements the required two-laptop boundary without pretending a
final model has been selected. Laptop 1 owns model artifacts, preprocessing,
candidate construction, decoder configuration, raw evaluation data, and
reference labels. Laptop 2 owns only the static evaluator UI. The UI receives
GPS observations, predicted route geometry, and safe aggregate metrics; it
never receives reference route IDs or point-level labels.

## What is implemented now

- FastAPI health, trip-discovery, and prediction endpoints in `src/api/`.
- Runtime discovery of eight-digit folders in the configured evaluation
  partition; no trip list is hardcoded.
- A fail-closed unconfigured predictor. `POST /v1/predictions` returns HTTP
  503 until a genuine selected-model adapter is configured.
- A dependency-free static UI in `frontend/` that lists server-discovered trips,
  requests an inference result, shows safe metrics, and plots GPS observations
  with the predicted route. It does not render a reference route.

## Laptop 1: API

Set `PATHSNAP_DATA_ROOT` to the folder containing the evaluation partition and
set `PATHSNAP_EVALUATION_PARTITION` if its name differs from `evaluator_test`.
Run from the repository root:

```powershell
$env:PATHSNAP_DATA_ROOT = "D:\path\to\Student folder"
$env:PATHSNAP_EVALUATION_PARTITION = "evaluator_test"
$env:PATHSNAP_ALLOWED_ORIGINS = "http://LAPTOP-2-IP:8080"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` reports whether a predictor is ready and how many trips are
available. `GET /v1/trips` returns only discoverable trip IDs. `POST
/v1/predictions` accepts `{"trip_id":"00000000"}` and writes completion or
failure details to Laptop 1's server log.

## Connect the final selected model

After the team selects and exports a model, create a team-owned module with a
factory and set `PATHSNAP_PREDICTOR_FACTORY` to `package.module:function`.
The factory receives `Settings` and returns an object with `name`, `version`,
and `predict(trip_id, trip_directory)`.

`predict` must run the genuine preprocessing, candidate generation, selected
model/scaler, and sequence decoder. It returns an `InferencePayload` (or an
equivalent dictionary) containing:

```python
{
  "model": {"name": "team-selected-model", "version": "artifact-id"},
  "observation_count": 0,
  "predicted_arc_count": 0,
  "observations": {"type": "FeatureCollection", "features": []},
  "predicted_route": {
    "type": "Feature",
    "geometry": {"type": "LineString", "coordinates": [[longitude, latitude]]},
    "properties": {},
  },
  "metrics": {"top1_accuracy": 0.0, "route_continuity": 0.0, "inference_ms": 0.0},
}
```

`observations` and `predicted_route` are GeoJSON. `metrics` are optional
aggregate values; omit a value when its server-side reference is unavailable.
Never return route labels, target arcs, point-level correctness, model weights,
or raw evaluator-test files to the UI.

## Laptop 2: UI

Edit `frontend/config.js` to set Laptop 1's reachable address, then serve the
folder from Laptop 2:

```powershell
cd frontend
python -m http.server 8080
```

Open `http://localhost:8080`. The API must allow that exact UI origin through
`PATHSNAP_ALLOWED_ORIGINS`. For a same-machine demo, use
`http://127.0.0.1:8080` in both settings.

## Required integration checks

1. Laptop 1 `/health` lists exactly the evaluation trips made available at run
   time; the current repository intentionally contains no evaluator partition.
2. Select every available trip from Laptop 2 and confirm a server log entry is
   written for each request.
3. Confirm the selected adapter runs the exported artifact and decoder rather
   than a hardcoded route.
4. Confirm the UI receives no reference route, proxy target, or per-point
   label. Screenshot the UI and Laptop 1 API log for the final report.
