"""Build trip-local candidate matrices, separate weak targets and grouped folds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["distance_m", "arc_length_m", "fraction", "heading_cos",
            "heading_known", "speed_mps", "dt_s", "segment_start", "speed_flag"]


def geometry(points, starts, vectors):
    squared = (vectors * vectors).sum(axis=-1)
    offsets = points[:, None] - starts
    fraction = np.clip((offsets * vectors).sum(axis=-1) / np.maximum(squared, 1e-12), 0, 1)
    distance = np.linalg.norm(offsets - fraction[..., None] * vectors, axis=-1)
    return distance, fraction


def extract(frame, nodes, arcs):
    gps = frame[["longitude", "latitude"]].to_numpy()
    centre = gps.mean(axis=0)
    scale = np.radians(1) * 6371008.8 * np.array([np.cos(np.radians(centre[1])), 1])
    xy = (gps - centre) * scale
    nodes = (nodes - centre) * scale
    starts = nodes[arcs[:, 0]]
    vectors = nodes[arcs[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    graph = coo_matrix((np.ones(len(arcs)), arcs.T), shape=(len(nodes), len(nodes)))
    _, components = connected_components(graph, directed=False)
    usable = (components[arcs[:, 0]] == np.bincount(components).argmax()) & (lengths > 0)
    ids = np.flatnonzero(usable)
    if not len(ids):
        raise ValueError("No usable road geometry")
    # Sample long arcs along their length so a nearby interior can enter the shortlist.
    counts = np.maximum(1, np.ceil(lengths[ids] / 30).astype(int))
    owners = np.repeat(ids, counts)
    offsets = np.arange(len(owners)) - np.repeat(np.cumsum(counts) - counts, counts)
    samples = starts[owners] + vectors[owners] * ((offsets + .5) / np.repeat(counts, counts))[:, None]
    tree = cKDTree(samples)
    _, indices = tree.query(xy, k=list(range(1, min(48, len(samples)) + 1)))
    pool = np.sort(owners[indices], axis=1)
    dist, frac = geometry(xy, starts[pool], vectors[pool])
    repeated = np.diff(pool, axis=1, prepend=-1) == 0
    dist[repeated] = np.inf
    order = np.argsort(dist, axis=1, kind="stable")[:, :8]
    row, slot = np.where(np.isfinite(np.take_along_axis(dist, order, 1)))
    col = order[row, slot]
    arc = pool[row, col]
    motion = np.diff(xy, axis=0, prepend=xy[:1])
    step = np.linalg.norm(motion, axis=1)
    known = (~frame.is_segment_start.to_numpy()) & (step >= 2)
    alignment = (motion[row] * vectors[arc]).sum(axis=1) / (np.maximum(step[row], 1e-8) * lengths[arc])
    features = pd.DataFrame(dict(original_row_index=frame.original_row_index.to_numpy()[row],
        segment_id=frame.segment_id.to_numpy()[row], arc_id=arc, distance_m=dist[row,col],
        arc_length_m=lengths[arc], fraction=frac[row,col], heading_cos=np.where(known[row], alignment, 0),
        heading_known=known[row], speed_mps=frame.implied_speed_kmh.fillna(0).to_numpy()[row]/3.6,
        dt_s=frame.delta_t_seconds.fillna(0).to_numpy()[row], segment_start=frame.is_segment_start.to_numpy()[row],
        speed_flag=frame.speed_flag.to_numpy()[row]))
    return features, xy, starts, vectors


def weak_targets(xy, starts, vectors, route):
    # Score all reference arcs independently of the generated candidate shortlist.
    ids = np.unique(route)
    target = []
    for batch in np.array_split(xy, max(1, int(np.ceil(len(xy)*len(ids)/200000)))):
        distances, _ = geometry(batch, starts[ids], vectors[ids])
        best = distances.argmin(axis=1)
        minimum = distances[np.arange(len(batch)), best]
        ambiguous = ((distances - minimum[:, None]) <= 1).sum(axis=1) > 1
        target.extend(zip(ids[best], minimum, ambiguous))
    return pd.DataFrame(target, columns=["proxy_arc_id", "reference_distance_m", "ambiguous"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"ordered_pretraining")
    args = parser.parse_args()
    points = pd.read_csv(ROOT/"results/tables/cleaned_gps_points.csv", dtype={"trip_id": str})
    records = []
    for (partition, trip), frame in points.groupby(["partition","trip_id"]):
        source = args.data_root/partition/trip/trip
        nodes = np.loadtxt(source.with_suffix(".nodes"), ndmin=2)
        arcs = np.loadtxt(source.with_suffix(".arcs"), dtype=int, ndmin=2)
        features, xy, starts, vectors = extract(frame, nodes, arcs)
        key = "original_row_index"
        assert features[key].nunique() == len(frame) and not features.duplicated([key,"arc_id"]).any()
        assert np.isfinite(features[FEATURES].to_numpy(dtype=float)).all()
        if __package__:
            from .ordered_alignment import align
        else:
            from ordered_alignment import align
        route = np.loadtxt(source.with_suffix(".route"), dtype=int, ndmin=1)
        labels = align(xy, starts, vectors, route, frame.segment_id.to_numpy(), geometry)
        nearest = weak_targets(xy, starts, vectors, route)
        labels["changed_from_nearest"] = labels.proxy_arc_id != nearest.proxy_arc_id
        labels[key] = frame[key].to_numpy()
        pairs = pd.MultiIndex.from_frame(features[[key,"arc_id"]])
        labels["in_candidates"] = pd.MultiIndex.from_frame(labels[[key,"proxy_arc_id"]]).isin(pairs)
        for folder, table in [("features", features), ("targets", labels)]:
            dest = args.output/folder/partition
            dest.mkdir(parents=True, exist_ok=True)
            table.to_pickle(dest/(trip+".pkl"))
        clear = ~labels.ambiguous
        records.append(dict(partition=partition, trip_id=trip, points=len(frame), candidates=len(features),
            ambiguous=int((~clear).sum()), clear_misses=int((clear & ~labels.in_candidates).sum()),
            changed=int(labels.changed_from_nearest.sum()), far=int((labels.reference_distance_m>50).sum()),
            median_distance_m=labels.reference_distance_m.median()))
    manifest = pd.DataFrame(records)
    assert manifest.trip_id.is_unique
    loads = np.zeros(5, dtype=int)
    manifest["validation_fold"] = -1
    for i in manifest.query("partition == 'train'").sort_values("candidates", ascending=False, kind="stable").index:
        fold = int(loads.argmin())
        manifest.loc[i,"validation_fold"] = fold
        loads[fold] += manifest.loc[i,"candidates"]
    manifest.to_csv(args.output/"manifest.csv", index=False)
    print(manifest.groupby("partition")[["points","candidates","ambiguous","clear_misses"]].sum())


if __name__ == "__main__":
    main()
