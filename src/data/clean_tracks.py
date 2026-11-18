"""Conservative GPS track cleaning for the ML-Hackathon dataset.

Raw .track files are never changed.  A point is dropped only when it cannot be
used reliably (malformed, non-finite/out-of-range coordinate, or invalid
timestamp).  Mean/median imputation and interpolation are deliberately not
used: a synthetic location or time could create a false road transition.

Valid points are retained in original file order.  A new segment begins after
an invalid row, a conflicting/reversed timestamp, or a gap greater than ten
seconds.  This makes discontinuities explicit for downstream map matching.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_ROOT = Path("Student folder")
PARTITIONS = {
    "train": DATA_ROOT / "train",
    "student_test": DATA_ROOT / "student_test",
}
OUTPUT_DIR = Path("cleaning_output")
GAP_THRESHOLD_SECONDS = 10.0
SPEED_FLAG_KMH = 180.0

CLEAN_COLUMNS = [
    "partition",
    "trip_id",
    "original_row_index",
    "segment_id",
    "longitude",
    "latitude",
    "timestamp_seconds",
    "delta_t_seconds",
    "is_segment_start",
    "implied_speed_kmh",
    "speed_flag",
    "cleaning_action",
    "cleaning_reason",
]


def parse_finite(token: str) -> float | None:
    """Return a finite float or None for a nonnumeric, NaN, or infinite token."""
    try:
        value = float(token)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres between WGS84 coordinates."""
    radius_km = 6371.0088
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, (lon1, lat1, lon2, lat2))
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def validation_reason(fields: list[str]) -> tuple[str | None, tuple[float, float, float] | None]:
    """Validate one whitespace-separated raw row without imputing any values."""
    if len(fields) != 3:
        return ("wrong_field_count" if len(fields) > 3 else "missing_field"), None

    longitude, latitude, timestamp = (parse_finite(field) for field in fields)
    if longitude is None:
        return "invalid_longitude", None
    if latitude is None:
        return "invalid_latitude", None
    if timestamp is None:
        return "invalid_timestamp", None
    if not -180 <= longitude <= 180:
        return "longitude_out_of_range", None
    if not -90 <= latitude <= 90:
        return "latitude_out_of_range", None
    return None, (longitude, latitude, timestamp)


def clean_track(track_path: Path, partition: str) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Clean one track while retaining valid observations and recording decisions."""
    trip_id = track_path.stem
    retained: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    segment_id = -1
    force_new_segment = True
    previous: dict[str, object] | None = None
    input_nonblank_rows = 0
    exact_duplicates_removed = 0
    gap_breaks = 0
    timestamp_breaks = 0

    for original_row_index, raw_line in enumerate(track_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not raw_line.strip():
            continue
        input_nonblank_rows += 1
        fields = raw_line.split()
        reason, values = validation_reason(fields)
        if reason is not None:
            rejected.append(
                {
                    "partition": partition,
                    "trip_id": trip_id,
                    "original_row_index": original_row_index,
                    "raw_row": raw_line,
                    "cleaning_action": "dropped",
                    "cleaning_reason": reason,
                }
            )
            force_new_segment = True
            previous = None
            continue

        longitude, latitude, timestamp = values  # type: ignore[misc]
        if previous is not None:
            exact_duplicate = (
                longitude == previous["longitude"]
                and latitude == previous["latitude"]
                and timestamp == previous["timestamp_seconds"]
            )
            if exact_duplicate:
                exact_duplicates_removed += 1
                rejected.append(
                    {
                        "partition": partition,
                        "trip_id": trip_id,
                        "original_row_index": original_row_index,
                        "raw_row": raw_line,
                        "cleaning_action": "dropped",
                        "cleaning_reason": "exact_duplicate_observation",
                    }
                )
                continue

        segment_reason = "start_of_trip" if force_new_segment else ""
        delta_t: float | None = None
        speed_kmh: float | None = None
        if previous is not None and not force_new_segment:
            delta_t = timestamp - float(previous["timestamp_seconds"])
            if delta_t <= 0:
                segment_reason = "repeated_timestamp" if delta_t == 0 else "backward_timestamp"
                timestamp_breaks += 1
            elif delta_t > GAP_THRESHOLD_SECONDS:
                segment_reason = "gap_over_10_seconds"
                gap_breaks += 1
            else:
                distance_km = haversine_km(
                    float(previous["longitude"]), float(previous["latitude"]), longitude, latitude
                )
                speed_kmh = distance_km / delta_t * 3600

        is_segment_start = bool(segment_reason)
        if is_segment_start:
            segment_id += 1
            # A new segment has no meaningful link to the prior segment.
            delta_t = None
            speed_kmh = None

        row = {
            "partition": partition,
            "trip_id": trip_id,
            "original_row_index": original_row_index,
            "segment_id": segment_id,
            "longitude": longitude,
            "latitude": latitude,
            "timestamp_seconds": timestamp,
            "delta_t_seconds": delta_t,
            "is_segment_start": is_segment_start,
            "implied_speed_kmh": speed_kmh,
            "speed_flag": bool(speed_kmh is not None and speed_kmh > SPEED_FLAG_KMH),
            "cleaning_action": "kept",
            "cleaning_reason": segment_reason,
        }
        retained.append(row)
        previous = row
        force_new_segment = False

    summary = {
        "partition": partition,
        "trip_id": trip_id,
        "input_nonblank_rows": input_nonblank_rows,
        "retained_rows": len(retained),
        "dropped_rows": len(rejected),
        "exact_duplicates_removed": exact_duplicates_removed,
        "segment_count": segment_id + 1 if retained else 0,
        "gap_over_10_seconds_breaks": gap_breaks,
        "timestamp_order_breaks": timestamp_breaks,
        "speed_flag_rows": sum(bool(row["speed_flag"]) for row in retained),
    }
    return retained, rejected, summary


def run_cleaning(
    data_root: Path = DATA_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean every available training and student-test track and write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for partition in PARTITIONS:
        partition_dir = data_root / partition
        for track_path in sorted(partition_dir.glob("*/*.track")):
            retained, rejected, summary = clean_track(track_path, partition)
            cleaned_rows.extend(retained)
            rejected_rows.extend(rejected)
            summaries.append(summary)

    cleaned = pd.DataFrame(cleaned_rows, columns=CLEAN_COLUMNS)
    rejected = pd.DataFrame(
        rejected_rows,
        columns=["partition", "trip_id", "original_row_index", "raw_row", "cleaning_action", "cleaning_reason"],
    )
    summary = pd.DataFrame(summaries).sort_values(["partition", "trip_id"]).reset_index(drop=True)

    cleaned.to_csv(output_dir / "cleaned_gps_points.csv", index=False)
    rejected.to_csv(output_dir / "dropped_gps_rows.csv", index=False)
    summary.to_csv(output_dir / "cleaning_summary.csv", index=False)

    segment_lengths = cleaned.groupby(["partition", "trip_id", "segment_id"], as_index=False).size().rename(
        columns={"size": "points_in_segment"}
    )
    segment_lengths.to_csv(output_dir / "segment_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(segment_lengths["points_in_segment"], bins=50, edgecolor="white")
    plt.xlabel("GPS points in retained segment")
    plt.ylabel("Number of segments")
    plt.title("Segment lengths after conservative GPS cleaning")
    plt.tight_layout()
    plt.savefig(output_dir / "segment_length_distribution.png", dpi=180)
    plt.close()

    notes = [
        "Cleaning policy: raw files are unchanged; no mean/median/interpolation imputation is used.",
        "Rows are dropped only for malformed fields, invalid numeric values, invalid coordinates, invalid timestamps, or exact duplicate observations.",
        f"A new segment begins after an invalid row, repeated/backward timestamp, or a timestamp gap over {GAP_THRESHOLD_SECONDS:g} seconds.",
        f"Speeds above {SPEED_FLAG_KMH:g} km/h are flagged for review only; they are not automatically removed.",
        f"Trips processed: {len(summary)}; retained points: {len(cleaned)}; dropped rows: {len(rejected)}; segments: {len(segment_lengths)}.",
    ]
    (output_dir / "cleaning_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return cleaned, summary, rejected


if __name__ == "__main__":
    cleaned_points, trip_summary, dropped_rows = run_cleaning()
    print(
        f"Cleaned {len(cleaned_points)} points across {len(trip_summary)} trips; "
        f"dropped {len(dropped_rows)} rows. Outputs: {OUTPUT_DIR.resolve()}"
    )
