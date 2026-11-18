import argparse
import csv
import sys
from pathlib import Path


def read_split_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.split() for line in handle if line.strip()]


def evenly_sample(items, limit):
    if limit is None or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]

    indices = [
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [items[index] for index in indices]


def resolve_route(route_ids, arcs, nodes):
    route_points = []
    invalid_arc_ids = 0

    for arc_id in route_ids:
        if arc_id < 0 or arc_id >= len(arcs):
            invalid_arc_ids += 1
            continue

        from_node_id, to_node_id = arcs[arc_id]
        if (
            from_node_id < 0
            or from_node_id >= len(nodes)
            or to_node_id < 0
            or to_node_id >= len(nodes)
        ):
            invalid_arc_ids += 1
            continue

        from_point = (nodes[from_node_id][1], nodes[from_node_id][0])
        to_point = (nodes[to_node_id][1], nodes[to_node_id][0])

        if not route_points or route_points[-1] != from_point:
            route_points.append(from_point)
        if route_points[-1] != to_point:
            route_points.append(to_point)

    return route_points, invalid_arc_ids


def parse_trip(data_root, partition, trip_dir):
    trip_id = trip_dir.name
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    missing_files = [
        path.name
        for path in (track_path, nodes_path, arcs_path, route_path)
        if not path.exists()
    ]

    track_rows = read_split_lines(track_path) if track_path.exists() else []
    gps_points = []
    malformed_gps_rows = 0
    for row in track_rows:
        try:
            if len(row) < 2:
                raise ValueError
            lon = float(row[0])
            lat = float(row[1])
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError
            gps_points.append((lat, lon))
        except (TypeError, ValueError, IndexError):
            malformed_gps_rows += 1

    node_rows = read_split_lines(nodes_path) if nodes_path.exists() else []
    nodes = []
    malformed_node_rows = 0
    for row in node_rows:
        try:
            if len(row) < 2:
                raise ValueError
            nodes.append((float(row[0]), float(row[1])))
        except (TypeError, ValueError, IndexError):
            malformed_node_rows += 1

    arc_rows = read_split_lines(arcs_path) if arcs_path.exists() else []
    arcs = []
    malformed_arc_rows = 0
    for row in arc_rows:
        try:
            if len(row) < 2:
                raise ValueError
            arcs.append((int(row[0]), int(row[1])))
        except (TypeError, ValueError, IndexError):
            malformed_arc_rows += 1

    route_rows = read_split_lines(route_path) if route_path.exists() else []
    route_ids = []
    malformed_route_rows = 0
    for row in route_rows:
        try:
            if len(row) != 1:
                raise ValueError
            route_ids.append(int(row[0]))
        except (TypeError, ValueError, IndexError):
            malformed_route_rows += 1

    valid_gps_points = gps_points
    if valid_gps_points:
        longitudes = [point[1] for point in valid_gps_points]
        latitudes = [point[0] for point in valid_gps_points]
        min_lon = min(longitudes)
        min_lat = min(latitudes)
        max_lon = max(longitudes)
        max_lat = max(latitudes)
    else:
        min_lon = min_lat = max_lon = max_lat = ""

    route_points, invalid_arc_ids = resolve_route(route_ids, arcs, nodes)
    invalid_node_references = sum(
        from_node_id < 0
        or from_node_id >= len(nodes)
        or to_node_id < 0
        or to_node_id >= len(nodes)
        for from_node_id, to_node_id in arcs
    )

    unique_route_arcs = len(set(route_ids))
    repeated_route_arc_entries = sum(route_ids.count(arc_id) - 1 for arc_id in set(route_ids))

    summary = {
        "partition": partition,
        "trip_id": trip_id,
        "gps_points": len(valid_gps_points),
        "malformed_gps_rows": malformed_gps_rows,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "road_nodes": len(nodes),
        "malformed_node_rows": malformed_node_rows,
        "road_arcs": len(arcs),
        "malformed_arc_rows": malformed_arc_rows,
        "route_entries": len(route_ids),
        "unique_route_arcs": unique_route_arcs,
        "repeated_route_arc_entries": repeated_route_arc_entries,
        "invalid_route_arc_ids": invalid_arc_ids + malformed_route_rows,
        "invalid_arc_node_references": invalid_node_references,
        "malformed_route_rows": malformed_route_rows,
        "missing_files": "|".join(missing_files),
    }

    return summary, gps_points, nodes, arcs, route_ids, route_points


def create_folium_map(
    folium,
    summary,
    gps_points,
    nodes,
    arcs,
    route_points,
    output_path,
    gps_limit,
    road_limit,
):
    if gps_points:
        center = (
            (summary["min_lat"] + summary["max_lat"]) / 2,
            (summary["min_lon"] + summary["max_lon"]) / 2,
        )
    elif nodes:
        center = (nodes[0][1], nodes[0][0])
    else:
        center = (0, 0)

    trip_map = folium.Map(
        location=center,
        zoom_start=13,
        control_scale=True,
        tiles=None,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Esri, HERE, Garmin, FAO, NOAA, USGS, "
            "EPA, NPS"
        ),
        name="Esri World Street Map",
        overlay=False,
        control=True,
    ).add_to(trip_map)

    gps_layer = folium.FeatureGroup(name="GPS points")
    for point in evenly_sample(gps_points, gps_limit):
        folium.CircleMarker(
            location=point,
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.55,
            weight=1,
        ).add_to(gps_layer)
    gps_layer.add_to(trip_map)

    road_layer = folium.FeatureGroup(name="Road network")
    sampled_arcs = evenly_sample(arcs, road_limit)
    for from_node_id, to_node_id in sampled_arcs:
        if (
            from_node_id < 0
            or from_node_id >= len(nodes)
            or to_node_id < 0
            or to_node_id >= len(nodes)
        ):
            continue
        start = (nodes[from_node_id][1], nodes[from_node_id][0])
        end = (nodes[to_node_id][1], nodes[to_node_id][0])
        folium.PolyLine(
            locations=[start, end],
            color="gray",
            weight=1,
            opacity=0.35,
        ).add_to(road_layer)
    road_layer.add_to(trip_map)

    route_layer = folium.FeatureGroup(name="Reference route")
    if route_points:
        folium.PolyLine(
            locations=route_points,
            color="red",
            weight=4,
            opacity=0.9,
        ).add_to(route_layer)
    route_layer.add_to(trip_map)

    folium.LayerControl().add_to(trip_map)

    bounds = []
    if gps_points:
        bounds.append((summary["min_lat"], summary["min_lon"]))
        bounds.append((summary["max_lat"], summary["max_lon"]))
    if route_points:
        bounds.extend(route_points)
    if bounds:
        trip_map.fit_bounds(bounds)

    title = (
        f"M4: {summary['partition']} / {summary['trip_id']}<br>"
        f"GPS: {summary['gps_points']} | "
        f"Route entries: {summary['route_entries']} | "
        f"Road nodes: {summary['road_nodes']} | "
        f"Road arcs: {summary['road_arcs']}<br>"
        f"Map sample limits: {gps_limit or 'all'} GPS, "
        f"{road_limit or 'all'} road arcs"
    )
    folium.map.Marker(
        location=center,
        icon=folium.DivIcon(
            icon_size=(500, 60),
            icon_anchor=(0, 0),
            html=f'<div style="font-size:14px; background:white; padding:6px; '
                 f'border:1px solid #777"><b>{title}</b></div>',
        ),
    ).add_to(trip_map)

    trip_map.save(str(output_path))


def add_trip_to_overview(
    folium,
    map_layers,
    summary,
    nodes,
    arcs,
    route_points,
    road_limit,
):
    partition = summary["partition"]
    trip_id = summary["trip_id"]
    road_layer = map_layers[partition]["road"]
    route_layer = map_layers[partition]["route"]

    for from_node_id, to_node_id in evenly_sample(arcs, road_limit):
        if (
            from_node_id < 0
            or from_node_id >= len(nodes)
            or to_node_id < 0
            or to_node_id >= len(nodes)
        ):
            continue

        start = (nodes[from_node_id][1], nodes[from_node_id][0])
        end = (nodes[to_node_id][1], nodes[to_node_id][0])
        folium.PolyLine(
            locations=[start, end],
            color="gray",
            weight=1,
            opacity=0.25,
            tooltip=f"{partition} road network",
        ).add_to(road_layer)

    if route_points:
        route_color = "red" if partition == "train" else "darkblue"
        folium.PolyLine(
            locations=route_points,
            color=route_color,
            weight=3,
            opacity=0.8,
            tooltip=f"Reference route: {partition}/{trip_id}",
            popup=f"Reference route: {partition}/{trip_id}",
        ).add_to(route_layer)


def create_overview_map(
    folium,
    output_path,
    gps_points_by_partition,
    map_layers,
    bounds,
):
    from folium.plugins import FastMarkerCluster

    overview = folium.Map(
        location=(0, 0),
        zoom_start=2,
        control_scale=True,
        tiles=None,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Esri, HERE, Garmin, FAO, NOAA, USGS, "
            "EPA, NPS"
        ),
        name="Esri World Street Map",
        overlay=False,
        control=True,
    ).add_to(overview)

    for partition in ("train", "student_test"):
        gps_layer = folium.FeatureGroup(
            name=f"GPS observations - {partition} (clustered)"
        )
        points = gps_points_by_partition.get(partition, [])
        if points:
            cluster = FastMarkerCluster(
                data=points,
                name=f"GPS observations - {partition}",
            )
            cluster.add_to(gps_layer)
        gps_layer.add_to(overview)
        map_layers[partition]["road"].add_to(overview)
        map_layers[partition]["route"].add_to(overview)

    if bounds:
        overview.fit_bounds(bounds, padding=(20, 20))

    folium.LayerControl(collapsed=True).add_to(overview)
    overview.save(str(output_path))


def write_csv(output_path, fieldnames, rows):
    try:
        handle = output_path.open("w", encoding="utf-8", newline="")
    except PermissionError:
        output_path = output_path.with_name(
            f"{output_path.stem}_rerun{output_path.suffix}"
        )
        print(
            f"WARNING: Original CSV is locked; writing to {output_path}",
            file=sys.stderr,
        )
        handle = output_path.open("w", encoding="utf-8", newline="")

    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "data" and script_dir.parent.name.lower() == "src":
        project_root = script_dir.parent.parent
    else:
        project_root = script_dir

    data_candidates = [
        project_root / "Student folder" / "Student folder",
        project_root / "Student folder",
        project_root / "data",
    ]
    data_root_default = next(
        (
            candidate
            for candidate in data_candidates
            if (candidate / "train").is_dir()
            or (candidate / "student_test").is_dir()
        ),
        data_candidates[0],
    )

    parser = argparse.ArgumentParser(
        description="Read-only M4 geographic extent and reference-route audit."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_root_default,
        help="Dataset directory containing train and student_test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "M4_output",
        help="Directory for CSV summaries and Folium maps.",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["train", "student_test"],
        choices=["train", "student_test"],
        help="Partitions to analyse.",
    )
    parser.add_argument(
        "--trip-id",
        action="append",
        default=None,
        help="Restrict analysis to a trip ID. Repeat to select multiple IDs.",
    )
    parser.add_argument(
        "--gps-limit",
        type=int,
        default=1000,
        help="Maximum GPS points displayed per map; use 0 for all.",
    )
    parser.add_argument(
        "--road-limit",
        type=int,
        default=1000,
        help="Maximum road arcs displayed per map; use 0 for all.",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Write CSV outputs only.",
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Create one all-trips Folium map without rewriting CSVs or trip maps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.no_maps and args.overview_only:
        print(
            "ERROR: --no-maps and --overview-only cannot be used together.",
            file=sys.stderr,
        )
        return 2

    if not data_root.exists():
        print(f"ERROR: Data root does not exist: {data_root}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "maps"
    if not args.no_maps and not args.overview_only:
        map_dir.mkdir(parents=True, exist_ok=True)

    try:
        import folium
    except ImportError:
        folium = None
        if not args.no_maps or args.overview_only:
            print(
                "ERROR: Folium is not installed. Install it with "
                "'py -m pip install folium'.",
                file=sys.stderr,
            )
            return 3

    gps_limit = None if args.gps_limit == 0 else args.gps_limit
    road_limit = None if args.road_limit == 0 else args.road_limit

    trip_rows = []
    overview_gps = {partition: [] for partition in args.partitions}
    overview_bounds = []
    overview_layers = {}
    if args.overview_only:
        for partition in args.partitions:
            overview_layers[partition] = {
                "road": folium.FeatureGroup(
                    name=f"Road network - {partition}",
                    show=False,
                ),
                "route": folium.FeatureGroup(
                    name=f"Reference routes - {partition}",
                    show=True,
                ),
            }

    for partition in args.partitions:
        partition_dir = data_root / partition
        if not partition_dir.exists():
            print(
                f"WARNING: Partition directory does not exist: {partition_dir}",
                file=sys.stderr,
            )
            continue

        trip_dirs = sorted(
            (path for path in partition_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        if args.trip_id:
            requested = set(args.trip_id)
            trip_dirs = [path for path in trip_dirs if path.name in requested]

        for trip_dir in trip_dirs:
            summary, gps_points, nodes, arcs, route_ids, route_points = parse_trip(
                data_root, partition, trip_dir
            )
            trip_rows.append(summary)
            print(
                f"Processed {partition}/{trip_dir.name}",
                flush=True,
            )

            if args.overview_only:
                overview_gps[partition].extend(gps_points)
                if gps_points:
                    overview_bounds.extend(
                        [
                            (summary["min_lat"], summary["min_lon"]),
                            (summary["max_lat"], summary["max_lon"]),
                        ]
                    )
                overview_bounds.extend(route_points)
                add_trip_to_overview(
                    folium,
                    overview_layers,
                    summary,
                    nodes,
                    arcs,
                    route_points,
                    road_limit,
                )
            elif not args.no_maps:
                map_path = map_dir / f"{partition}_{trip_dir.name}.html"
                create_folium_map(
                    folium,
                    summary,
                    gps_points,
                    nodes,
                    arcs,
                    route_points,
                    map_path,
                    gps_limit,
                    road_limit,
                )

    if args.overview_only:
        overview_path = output_dir / "all_trips_overview.html"
        create_overview_map(
            folium,
            overview_path,
            overview_gps,
            overview_layers,
            overview_bounds,
        )
        print(f"Created {overview_path}")
        print(
            "Included GPS observations: "
            + ", ".join(
                f"{partition}={len(points)}"
                for partition, points in overview_gps.items()
            )
        )
        print(
            "Road arcs are sampled to "
            f"{road_limit or 'all'} per trip; reference routes use all valid "
            "route entries."
        )
        return 0

    trip_fieldnames = [
        "partition",
        "trip_id",
        "gps_points",
        "malformed_gps_rows",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "road_nodes",
        "malformed_node_rows",
        "road_arcs",
        "malformed_arc_rows",
        "route_entries",
        "unique_route_arcs",
        "repeated_route_arc_entries",
        "invalid_route_arc_ids",
        "invalid_arc_node_references",
        "malformed_route_rows",
        "missing_files",
    ]
    trip_csv = output_dir / "m4_per_trip.csv"
    trip_csv = write_csv(trip_csv, trip_fieldnames, trip_rows)

    partition_rows = []
    for partition in args.partitions:
        rows = [row for row in trip_rows if row["partition"] == partition]
        valid_gps = [row for row in rows if row["gps_points"] > 0]
        route_rows = [row for row in rows if row["route_entries"] > 0]
        partition_rows.append(
            {
                "partition": partition,
                "trip_count": len(rows),
                "gps_backed_trips": len(valid_gps),
                "route_backed_trips": len(route_rows),
                "invalid_route_arc_ids": sum(
                    row["invalid_route_arc_ids"] for row in rows
                ),
                "invalid_arc_node_references": sum(
                    row["invalid_arc_node_references"] for row in rows
                ),
                "missing_file_trips": sum(
                    bool(row["missing_files"]) for row in rows
                ),
            }
        )

    partition_fieldnames = [
        "partition",
        "trip_count",
        "gps_backed_trips",
        "route_backed_trips",
        "invalid_route_arc_ids",
        "invalid_arc_node_references",
        "missing_file_trips",
    ]
    partition_csv = output_dir / "m4_partition_summary.csv"
    partition_csv = write_csv(partition_csv, partition_fieldnames, partition_rows)

    print(f"Created {trip_csv}")
    print(f"Created {partition_csv}")
    if not args.no_maps:
        print(f"Created Folium maps in {map_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
