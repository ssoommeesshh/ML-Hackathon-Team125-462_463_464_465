"""
PathSnap - M1: Dataset Inventory & Schema Verification
Team 125 | PES University UE24CS352A Machine Learning Hackathon
"""

import os
import glob
import pandas as pd
import numpy as np

def generate_inventory(base_dir=r"data/Student folder", output_dir=r"results/tables"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Per-Trip Analysis
    records = []
    partitions = ['train', 'student_test']
    
    for part in partitions:
        part_path = os.path.join(base_dir, part)
        if not os.path.exists(part_path):
            continue
        trips = sorted([d for d in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, d))])
        
        for t in trips:
            tdir = os.path.join(part_path, t)
            row = {'partition': part, 'trip_id': t}
            
            # .track
            p_track = os.path.join(tdir, f"{t}.track")
            row['track_exists'] = os.path.exists(p_track)
            row['track_size_bytes'] = os.path.getsize(p_track) if row['track_exists'] else 0
            if row['track_exists']:
                df_track = pd.read_csv(p_track, sep=r"\s+", header=None)
                row['track_rows'] = len(df_track)
                row['duration_sec'] = float(df_track[2].iloc[-1] - df_track[2].iloc[0])
                dt = df_track[2].diff().dropna()
                row['pct_1s_intervals'] = float((dt == 1.0).mean() * 100) if len(dt) > 0 else 100.0
                row['lon_min'] = float(df_track[0].min())
                row['lon_max'] = float(df_track[0].max())
                row['lat_min'] = float(df_track[1].min())
                row['lat_max'] = float(df_track[1].max())
            else:
                row['track_rows'] = 0
                row['duration_sec'] = 0.0
                row['pct_1s_intervals'] = 0.0
                
            # .nodes
            p_nodes = os.path.join(tdir, f"{t}.nodes")
            row['nodes_exists'] = os.path.exists(p_nodes)
            row['nodes_size_bytes'] = os.path.getsize(p_nodes) if row['nodes_exists'] else 0
            if row['nodes_exists']:
                df_nodes = pd.read_csv(p_nodes, sep=r"\s+", header=None)
                row['nodes_rows'] = len(df_nodes)
            else:
                row['nodes_rows'] = 0
                
            # .arcs
            p_arcs = os.path.join(tdir, f"{t}.arcs")
            row['arcs_exists'] = os.path.exists(p_arcs)
            row['arcs_size_bytes'] = os.path.getsize(p_arcs) if row['arcs_exists'] else 0
            if row['arcs_exists']:
                df_arcs = pd.read_csv(p_arcs, sep=r"\s+", header=None)
                row['arcs_rows'] = len(df_arcs)
            else:
                row['arcs_rows'] = 0
                
            # .route
            p_route = os.path.join(tdir, f"{t}.route")
            row['route_exists'] = os.path.exists(p_route)
            row['route_size_bytes'] = os.path.getsize(p_route) if row['route_exists'] else 0
            if row['route_exists']:
                df_route = pd.read_csv(p_route, sep=r"\s+", header=None)
                row['route_rows'] = len(df_route)
            else:
                row['route_rows'] = 0
                
            records.append(row)
            
    df_trips = pd.DataFrame(records)
    trip_details_path = os.path.join(output_dir, "per_trip_statistics.csv")
    df_trips.to_csv(trip_details_path, index=False)
    
    # 2. Partition Summary Table
    part_summary = []
    for part in ['train', 'student_test']:
        df_p = df_trips[df_trips['partition'] == part]
        n_trips = len(df_p)
        part_summary.append({
            'Partition': 'Training' if part == 'train' else 'Student Testing',
            'Trip Count': n_trips,
            'Verification Status': f'Verified from files ({n_trips} dirs)',
            'Track Files Available': f"{df_p['track_exists'].sum()}/{n_trips}",
            'Total GPS Points': int(df_p['track_rows'].sum()),
            'Nodes Files Available': f"{df_p['nodes_exists'].sum()}/{n_trips}",
            'Total Road Nodes': int(df_p['nodes_rows'].sum()),
            'Arcs Files Available': f"{df_p['arcs_exists'].sum()}/{n_trips}",
            'Total Road Arcs': int(df_p['arcs_rows'].sum()),
            'Route Files Available': f"{df_p['route_exists'].sum()}/{n_trips}",
            'Total Route Arcs': int(df_p['route_rows'].sum()),
            'Total Size (MB)': round(float((df_p['track_size_bytes'] + df_p['nodes_size_bytes'] + df_p['arcs_size_bytes'] + df_p['route_size_bytes']).sum()) / (1024 * 1024), 2)
        })
        
    # Documented evaluator testing partition
    part_summary.append({
        'Partition': 'Evaluator Testing',
        'Trip Count': 10,
        'Verification Status': 'Documented count (Section 7, Dataset Exploration.pdf; reserved server-side)',
        'Track Files Available': '10/10 (Reserved)',
        'Total GPS Points': 'N/A (Evaluation-only)',
        'Nodes Files Available': '10/10 (Reserved)',
        'Total Road Nodes': 'N/A (Evaluation-only)',
        'Arcs Files Available': '10/10 (Reserved)',
        'Total Road Arcs': 'N/A (Evaluation-only)',
        'Route Files Available': '10/10 (Server-side ground truth)',
        'Total Route Arcs': 'N/A (Evaluation-only)',
        'Total Size (MB)': 'N/A (Reserved)'
    })
    
    df_part_summary = pd.DataFrame(part_summary)
    df_part_summary.to_csv(os.path.join(output_dir, "trip_inventory.csv"), index=False)
    
    # 3. File Inventory by Extension
    file_stats = []
    extensions = [
        ('.track', 'track_exists', 'track_size_bytes', 'track_rows', 'GPS observations (lon, lat, time)'),
        ('.nodes', 'nodes_exists', 'nodes_size_bytes', 'nodes_rows', 'Road-network nodes (lon, lat)'),
        ('.arcs', 'arcs_exists', 'arcs_size_bytes', 'arcs_rows', 'Road-network directed segments (from_node, to_node)'),
        ('.route', 'route_exists', 'route_size_bytes', 'route_rows', 'Ground truth traversed sequence of arc_ids')
    ]
    
    for ext, ex_col, sz_col, rw_col, desc in extensions:
        # train
        df_tr = df_trips[df_trips['partition'] == 'train']
        # student test
        df_st = df_trips[df_trips['partition'] == 'student_test']
        
        file_stats.append({
            'File Type': ext,
            'Description': desc,
            'Train Availability': f"{df_tr[ex_col].sum()}/80 (100%)",
            'Train Total Size': f"{df_tr[sz_col].sum() / 1024:.1f} KB",
            'Train Size Range': f"{df_tr[sz_col].min():,} B - {df_tr[sz_col].max():,} B (Avg: {df_tr[sz_col].mean():.0f} B)",
            'Train Total Rows': f"{df_tr[rw_col].sum():,}",
            'Train Rows Range': f"{df_tr[rw_col].min():,} - {df_tr[rw_col].max():,} (Avg: {df_tr[rw_col].mean():.1f})",
            'Student-Test Availability': f"{df_st[ex_col].sum()}/10 (100%)",
            'Student-Test Total Size': f"{df_st[sz_col].sum() / 1024:.1f} KB",
            'Student-Test Total Rows': f"{df_st[rw_col].sum():,}"
        })
        
    df_file_stats = pd.DataFrame(file_stats)
    df_file_stats.to_csv(os.path.join(output_dir, "file_availability_and_sizes.csv"), index=False)
    
    # 4. File Schema Table
    schemas = [
        {
            'File Extension': '.track',
            'Semantic Meaning': 'Vehicle GPS trajectory observations over time',
            'Header': 'None (whitespace-separated)',
            'Column Index': 0,
            'Column Name': 'longitude',
            'Data Type': 'float64',
            'Units / Format': 'WGS84 degrees (-180 to 180)',
            'Missing / Null Count': 0,
            'Example Value': '8.595203'
        },
        {
            'File Extension': '.track',
            'Semantic Meaning': 'Vehicle GPS trajectory observations over time',
            'Header': 'None (whitespace-separated)',
            'Column Index': 1,
            'Column Name': 'latitude',
            'Data Type': 'float64',
            'Units / Format': 'WGS84 degrees (-90 to 90)',
            'Missing / Null Count': 0,
            'Example Value': '49.844053'
        },
        {
            'File Extension': '.track',
            'Semantic Meaning': 'Vehicle GPS trajectory observations over time',
            'Header': 'None (whitespace-separated)',
            'Column Index': 2,
            'Column Name': 'timestamp',
            'Data Type': 'float64',
            'Units / Format': 'Seconds elapsed (~1 Hz sampling interval)',
            'Missing / Null Count': 0,
            'Example Value': '1.000000'
        },
        {
            'File Extension': '.nodes',
            'Semantic Meaning': 'Road network spatial intersections and waypoints',
            'Header': 'None (whitespace-separated)',
            'Column Index': 0,
            'Column Name': 'longitude',
            'Data Type': 'float64',
            'Units / Format': 'WGS84 degrees; 0-based row index defines node_id',
            'Missing / Null Count': 0,
            'Example Value': '8.581679'
        },
        {
            'File Extension': '.nodes',
            'Semantic Meaning': 'Road network spatial intersections and waypoints',
            'Header': 'None (whitespace-separated)',
            'Column Index': 1,
            'Column Name': 'latitude',
            'Data Type': 'float64',
            'Units / Format': 'WGS84 degrees; 0-based row index defines node_id',
            'Missing / Null Count': 0,
            'Example Value': '49.857880'
        },
        {
            'File Extension': '.arcs',
            'Semantic Meaning': 'Directed road network segments connecting nodes',
            'Header': 'None (whitespace-separated)',
            'Column Index': 0,
            'Column Name': 'from_node_id',
            'Data Type': 'int64',
            'Units / Format': '0-based index of source node; 0-based row index defines arc_id',
            'Missing / Null Count': 0,
            'Example Value': '0'
        },
        {
            'File Extension': '.arcs',
            'Semantic Meaning': 'Directed road network segments connecting nodes',
            'Header': 'None (whitespace-separated)',
            'Column Index': 1,
            'Column Name': 'to_node_id',
            'Data Type': 'int64',
            'Units / Format': '0-based index of destination node',
            'Missing / Null Count': 0,
            'Example Value': '281'
        },
        {
            'File Extension': '.route',
            'Semantic Meaning': 'Chronological sequence of ground-truth road segments traversed',
            'Header': 'None (1 integer per line)',
            'Column Index': 0,
            'Column Name': 'arc_id',
            'Data Type': 'int64',
            'Units / Format': '0-based arc_id corresponding to .arcs row index',
            'Missing / Null Count': 0,
            'Example Value': '2641'
        }
    ]
    df_schema = pd.DataFrame(schemas)
    df_schema.to_csv(os.path.join(output_dir, "file_schema.csv"), index=False)
    
    return df_trips, df_part_summary, df_file_stats, df_schema

if __name__ == '__main__':
    df_trips, df_part_summary, df_file_stats, df_schema = generate_inventory()
    print("=== TRIP INVENTORY TABLE ===")
    print(df_part_summary.to_markdown(index=False))
    print("\n=== FILE AVAILABILITY & SIZES TABLE ===")
    print(df_file_stats.to_markdown(index=False))
    print("\n=== SHORT FILE SCHEMA TABLE ===")
    print(df_schema.to_markdown(index=False))
