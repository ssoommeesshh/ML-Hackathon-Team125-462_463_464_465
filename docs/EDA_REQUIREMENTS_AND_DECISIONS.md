# EDA Requirements, Evidence, and Downstream Decisions

This document closes the exploratory-data-analysis checklist in `Dataset Exploration.pdf`. It is a record of what was observed from the available 80 training and 10 student-test trips, not an assertion about the unavailable evaluator-test partition.

## Requirement traceability

| Reference requirement | Evidence in this repository | Decision for downstream work |
| --- | --- | --- |
| Trip counts by partition | `results/tables/trip_inventory.csv` | Use 80 training trips for fitting/CV and 10 student-test trips only for development evaluation. Evaluator-test data and labels remain outside development. |
| GPS observations per trip | `results/tables/m2_per_trip_audit.csv`, `results/tables/per_trip_statistics.csv` | Treat a trip as the atomic unit: never split rows from one trip across training and validation folds. |
| File availability, sizes, schema, and types | `results/tables/file_availability_and_sizes.csv`, `results/tables/file_schema.csv` | Parse headerless whitespace-separated files with the verified schemas. `.track` is `(longitude, latitude, timestamp_seconds)`; `.nodes` is `(longitude, latitude)`; `.arcs` is `(from_node_id, to_node_id)`; `.route` is an ordered `arc_id` sequence. |
| Missing/invalid GPS observations | `results/tables/m2_per_trip_audit.csv`, `results/tables/cleaning_summary.csv` | The audited 227,577 points are valid; keep them unchanged. Future malformed/non-finite/out-of-range values are dropped and logged, never mean-imputed. |
| Timestamp ordering and irregular sampling | `results/tables/m2_per_trip_audit.csv`, `results/figures/m2_sampling_gap_histogram.png` | Preserve file order. Do not sort or interpolate observations. Begin a new segment after an invalid row, non-increasing timestamp, or a gap greater than 10 seconds. |
| Geographic extent and coordinate plausibility | `results/tables/m4_per_trip_summary.csv`, `src/data/m4_geographic_analysis.py` | Coordinates are WGS84 `(longitude, latitude)`. The dataset spans longitudes -100.418538 to 137.237029 and latitudes -34.640291 to 61.699590, so no single UTM zone is valid for the whole dataset. |
| Road-network scale, topology, islands, and isolated nodes | `results/tables/m3_road_network_per_trip_summary.csv`, `results/tables/m3_road_network_notes.txt` | Preserve raw road graphs. During candidate generation, retain candidates in the trip's giant component; do not globally delete graph islands. Parallel arcs require directional/heading-aware disambiguation. |
| GPS gaps and other anomalous trips | `results/tables/m2_flagged_trips.csv` | A 10-second gap is a trajectory break, not missing data to fill. Keep both endpoint observations but prohibit a model transition across the resulting segment boundary unless a later model explicitly supports it. |
| Reference route availability and structure | `results/tables/m4_per_trip_summary.csv`, `results/tables/m3_road_network_notes.txt` | A `.route` row is an arc ID, not a one-to-one GPS label. Use route data only as an ordered path-level reference for permitted training/evaluation; do not align GPS row number to route row number. |
| Route-label limitations | `results/tables/m3_road_network_notes.txt` | Do not repair or hand-edit routes. Eleven trips have minor continuity imperfections; record these as label limitations when evaluating rather than treating every label as infallible truth. |
| Coordinate/trajectory visualisation | `results/figures/m3_trip_overlay_and_ground_truth_00000000.png`, `results/figures/m3_folium_road_network_trip_00000000.html`, `results/figures/m3_folium_road_network_trip_00000076.html` | Use the maps for qualitative inspection only. They must not be used to manually alter predictions or labels. |

## Cleaning decisions

The cleaning output is reproducible through `src/data/clean_tracks.py` and `notebooks/04_track_cleaning.ipynb`.

- Do not overwrite raw files.
- Do not use mean, median, forward-fill, backward-fill, or interpolation for longitude, latitude, or timestamps.
- Drop only unusable rows: malformed field counts, nonnumeric/non-finite values, out-of-range coordinates, invalid timestamps, and exact duplicate observations. Log every dropped row and reason.
- Retain valid observations in original file order with `original_row_index`.
- Add `segment_id`; segment at invalid-row boundaries, repeated/backward timestamps, and timestamp gaps over 10 seconds.
- Compute unusually high implied speed only as a review flag. Do not automatically drop it before map matching, because GPS noise and sparse sampling can produce an apparently implausible straight-line speed.

The committed run retained all 227,577 observations, dropped zero rows, and created 707 segments. The non-destructive segmentation is therefore the material cleaning result.

## Geospatial contract for feature extraction

1. Store all source and cleaned coordinates in WGS84 longitude/latitude order.
2. Use Haversine/geodesic distance in metres for GPS-to-GPS distance, GPS-to-node distance, and GPS-to-arc distance calculations that must work across all trips. This avoids using Euclidean distance in degrees and avoids an invalid global CRS.
3. If a feature needs planar coordinates (for example, local perpendicular projection onto a road arc), select a projection **per trip** from that trip's centroid. A UTM zone may be selected from the centroid longitude and hemisphere; never impose one dataset-wide UTM zone.
4. Keep calculations scoped to the trip's own road network. Node indices and arc indices are local to that trip.
5. Preserve `segment_id` in every point-level feature table. Candidate transition features must not bridge two segments by default.
6. Parallel arcs are legitimate alternatives, not duplicates to delete. Include arc direction and GPS heading/bearing where available.

## Leakage and evaluation contract

- Split by `trip_id`, never by GPS point or segment. Segments from one trip must remain in the same fold.
- Fit any learned transformation (scaler, encoder, feature-selection threshold, calibration, or hyperparameter) on training trips only, then apply it unchanged to student-test and evaluator-test trips.
- Student-test routes may be used to measure development performance, but not to fit a model intended for that evaluation.
- Evaluator-test data is excluded from training, feature selection, threshold selection, hyperparameter selection, and manual inspection of known routes.
- Do not create features from the reference route that would not be available at prediction time. Reference routes are labels/evaluation material, not inference inputs.
- Report metrics both overall and by trip/segment so a few long trips cannot dominate results.

## Work following the exploratory analysis

The [pretraining handoff](PRETRAINING_HANDOFF.md) documents segment geometry, spatial shortlisting, weak supervision and fold assignment. The [family comparison](MODEL_FAMILY_DECISIONS.md) documents the shared relative-distance feature, train-fold scaling, five scorers and decoder. The measured comparison and ordered-route diagnostics are in [the evaluation report](MODEL_COMPARISON_RESULTS.md).
