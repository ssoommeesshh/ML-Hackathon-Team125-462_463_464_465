# PathSnap: Distributed Map-Matching & Route Reconstruction

The [model comparison results](docs/MODEL_COMPARISON_RESULTS.md) cover the five team-owned families and the nearest-segment baseline. The [experiment decisions and reproduction commands](docs/MODEL_FAMILY_DECISIONS.md) describe the grouped folds, feature analysis, sequence decoder and reasons for deferring other families.

**Team 125** | PES University — Department of Computer Science and Engineering  
**Course:** UE24CS352A Machine Learning (2026)

---

## 📌 Project Overview
PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle routes on complex road networks by matching noisy, sparse GPS observations. GPS devices in urban environments frequently suffer from multipath distortions, signal loss, and sampling jitter, causing observed trajectories to drift away from physical roads. PathSnap formulates this as a map-matching problem solved through candidate generation, machine learning scoring, Hidden Markov Model (HMM) sequence decoding with Viterbi, and two-laptop distributed deployment (FastAPI backend + interactive UI).

---

## 📂 Repository Structure
```
├── docs/                                # Hackathon problem statement & guidelines
│   ├── Main Guidelines.pdf
│   ├── Dataset Exploration.pdf
│   ├── Deliverables.pdf
│   ├── Models and Training Guidelines.pdf
│   └── Assessment and Plagiarism Policy.pdf
├── notebooks/                           # Step-by-step exploratory and experimentation notebooks
│   ├── 01_m1_dataset_inventory.ipynb   # Executed M1 notebook with cell-wise outputs & plots
│   ├── 02_m2_gps_observation_audit.ipynb # Executed M2 GPS track audit notebook
│   └── 03_m3_road_network_analysis.ipynb # Executed M3 Road-network topological analysis notebook
├── src/                                 # Production source code
│   └── data/
│       ├── dataset_inventory.py         # Automated dataset verification & schema generator
│       └── road_network_analysis.py     # Production road-network graph analyzer & visualizer
├── results/                             # Generated metrics, tables, and visualization artifacts
│   ├── figures/
│   │   ├── m1_inventory_distributions.png
│   │   ├── m2_sampling_gap_histogram.png
│   │   ├── m3_road_network_topology_00000000.png
│   │   ├── m3_trip_overlay_and_ground_truth_00000000.png
│   │   ├── m3_dataset_road_network_distributions.png
│   │   ├── m3_folium_road_network_trip_00000000.html
│   │   └── m3_folium_road_network_trip_00000076.html
│   └── tables/
│       ├── trip_inventory.csv           # Partition-level inventory table
│       ├── file_availability_and_sizes.csv # Detailed file availability & size metrics
│       ├── file_schema.csv              # Exact schema specification
│       ├── per_trip_statistics.csv      # Complete metrics for all 90 inspected trips
│       ├── m2_per_trip_audit.csv        # M2 GPS audit per trip
│       ├── m2_flagged_trips.csv         # M2 flagged anomalous trips
│       ├── m3_road_network_per_trip_summary.csv # M3 complete 90-trip graph metrics
│       ├── m3_student_test_road_network_summary.csv # M3 student-testing 10-trip summary
│       ├── m3_partition_comparison_table.csv # M3 Train vs Student-Test comparison
│       └── m3_road_network_notes.txt    # M3 topological analysis notes
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Milestone M1: Dataset Inventory & File Schema

### 1. Partition Verification & Trip Inventory Table
The dataset consists of **100 total trips** across three partitions. The on-disk training and student testing counts were strictly verified against the raw files, while evaluator testing count is documented per guidelines:

| Partition | Trip Count | Verification Status | .track Available | .nodes Available | .arcs Available | .route Available | Total GPS Points | Total Road Nodes | Total Road Arcs | Total Route Arcs | Partition Size (MB) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | **80** | Verified from disk (80 folders) | 80/80 (100%) | 80/80 (100%) | 80/80 (100%) | 80/80 (100%) | 208,417 | 1,933,485 | 3,876,904 | 31,336 | **85.58 MB** |
| **Student Testing** | **10** | Verified from disk (10 folders) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 19,160 | 90,689 | 185,041 | 3,477 | **4.17 MB** |
| **Evaluator Testing** | **10** | Documented count (Dataset Exploration.pdf §7) | 10/10 (Reserved) | 10/10 (Reserved) | 10/10 (Reserved) | 10/10 (Server-side) | *Reserved* | *Reserved* | *Reserved* | *Reserved* | *Reserved* |
| **Total** | **100** | Full Benchmark Suite | **90/90** | **90/90** | **90/90** | **90/90** | **227,577** | **2,024,174** | **4,061,945** | **34,813** | **89.75 MB** |

---

### 2. File Availability & File Size Breakdown
Each trip directory contains exactly four whitespace-separated, headerless files:

| File Type | Semantic Meaning | Train Avail | Train Size (Total) | Train Size Range | Train Rows Range (Avg) | Student Test Avail | Student Test Size | Student Test Rows |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`.track`** | Raw vehicle GPS observations | 80/80 (100%) | 6.55 MB (6,711 KB) | 11.5 KB – 626.4 KB | 361 – 19,313 (Avg: 2,605) | 10/10 (100%) | 611.2 KB | 19,160 |
| **`.nodes`** | Road network intersection vertices | 80/80 (100%) | 36.65 MB (37,531 KB) | 5.6 KB – 2.95 MB | 299 – 155,003 (Avg: 24,169) | 10/10 (100%) | 1.74 MB (1,780 KB) | 90,689 |
| **`.arcs`** | Road network directed edges | 80/80 (100%) | 42.22 MB (43,230 KB) | 4.6 KB – 3.86 MB | 649 – 307,392 (Avg: 48,461) | 10/10 (100%) | 1.81 MB (1,858 KB) | 185,041 |
| **`.route`** | Chronological ground-truth arc IDs | 80/80 (100%) | 161.8 KB | 253 B – 12.1 KB | 59 – 2,266 (Avg: 392) | 10/10 (100%) | 17.9 KB | 3,477 |

---

### 3. Short File Schema Table

| File Type | Format / Delimiter | Col Index | Column Name | Data Type | Semantics & Range | Null Count | Example Value |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| **`.track`** | Whitespace-separated | 0 | `longitude` | `float64` | WGS84 angular coordinate in degrees `[-180.0, 180.0]` | 0 | `8.595203` |
| **`.track`** | Whitespace-separated | 1 | `latitude` | `float64` | WGS84 angular coordinate in degrees `[-90.0, 90.0]` | 0 | `49.844053` |
| **`.track`** | Whitespace-separated | 2 | `timestamp` | `float64` | Relative elapsed time in seconds; nominally 1 Hz (~94.3% exactly 1s) | 0 | `1.000000` |
| **`.nodes`** | Whitespace-separated | 0 | `longitude` | `float64` | Road network vertex longitude; 0-based row index implicitly defines `node_id` | 0 | `8.581679` |
| **`.nodes`** | Whitespace-separated | 1 | `latitude` | `float64` | Road network vertex latitude; 0-based row index implicitly defines `node_id` | 0 | `49.857880` |
| **`.arcs`** | Whitespace-separated | 0 | `from_node_id` | `int64` | 0-based index of source node; 0-based row index implicitly defines `arc_id` | 0 | `0` |
| **`.arcs`** | Whitespace-separated | 1 | `to_node_id` | `int64` | 0-based index of destination node | 0 | `281` |
| **`.route`** | Single value per line | 0 | `arc_id` | `int64` | Ground-truth traversed road segment index matching `.arcs` row index | 0 | `2641` |

---

### 4. Key Quality & Sanity Findings
- **Missing Values**: Exactly **0** null, NaN, or corrupted values across all 227,577 GPS rows, 2,024,174 nodes, and 4,061,945 arcs.
- **Sampling Frequency**: **94.32%** of successive GPS timestamps have an exact $\Delta t = 1.0\text{ s}$, confirming a nominal 1 Hz sampling rate. Trips with $\Delta t > 1\text{ s}$ represent GPS dropouts that will be handled via transition probabilities in the HMM.
- **Topological Integrity**: 100% of `from_node_id` and `to_node_id` values in `.arcs` point to valid 0-based row indices in `.nodes`. 100% of `arc_id` values in `.route` point to valid rows in `.arcs`.
- **Geographical Extent**: Trips span multiple regions (longitude -100.418538 to 137.237029; latitude -34.640291 to 61.699590). Geographic calculations use Haversine/geodesic distance; a planar projection, when required, must be selected per trip rather than using one dataset-wide UTM zone.

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the inventory script:
   ```bash
   python src/data/dataset_inventory.py
   ```
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/01_m1_dataset_inventory.ipynb
   ```

---

## 🛰️ Milestone M2: GPS Observation Audit

`notebooks/02_m2_gps_observation_audit.ipynb` audits all 80 training and 10 student-test `.track` files without modifying the raw observations. It checks malformed and invalid GPS rows, timestamp ordering, and sampling gaps, then writes the handoff artifacts below:

- `results/tables/m2_per_trip_audit.csv` - one row per `(partition, trip_id)` with observation counts, validation counts, timestamp statistics, and flags.
- `results/tables/m2_flagged_trips.csv` - only trips needing review, with the reason in `flags`.
- `results/figures/m2_sampling_gap_histogram.png` - positive timestamp-gap distribution and the 10-second threshold.
- `results/tables/m2_audit_notes.txt` - threshold rationale and aggregate findings.

Run the notebook from the repository root or from `notebooks/`; it resolves the project root automatically. Raw data must be available in `Student folder/` alongside the repository before running it.

---

## 🧹 GPS Track Cleaning

`src/data/clean_tracks.py` and `notebooks/04_track_cleaning.ipynb` implement conservative GPS cleaning across the training and student-test partitions. Raw `.track` files are never modified and no mean, median, forward-fill, or interpolation is used for geographic coordinates or timestamps.

- Invalid rows and exact duplicate observations are removed only when encountered and are recorded in `results/tables/dropped_gps_rows.csv`.
- Valid observations retain their original file order and receive a `segment_id`; a new segment begins after an invalid row, repeated/backward timestamp, or time gap over 10 seconds.
- Implied speeds above 180 km/h are flagged for review, not automatically removed.

The committed run retained all 227,577 valid GPS points, removed zero rows, and produced 707 usable segments. The complete cleaned-point table, trip summary, segment summary, policy note, and segment-length distribution are in `results/`.


---

## 🛣️ Milestone M3: Road-Network Topological & Connectivity Analysis


otebooks/03_m3_road_network_analysis.ipynb and src/data/road_network_analysis.py perform comprehensive graph-theoretical characterization across all **90 road networks** (80 Training trips + 10 Student-Testing trips). The analysis models each trip as a directed multigraph $\mathcal{G}_{\text{dir}} = (\mathcal{V}, \mathcal{E})$ and undirected projection $\mathcal{G}_{\text{undir}}$ to evaluate network scale, degree distributions, connectivity, component decomposition, and ground-truth route feasibility.

### 1. Student-Testing Road-Network Summary Table (All 10 Trips)
The 10 reserved student-test trips serve as the development evaluation set:

| Trip ID | Nodes ($|V|$) | Arcs ($|E|$) | Ratio ($|E|/|V|$) | Mean Deg | Max Deg | Isolated Nodes | Components | Giant Comp (%) | Self Loops | Parallel Edges | GPS Points | Route Arcs | Route Continuity (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00000000** | 8,542 | 18,196 | 2.130 | 2.153 | 6 | 0 | 100 | 87.40% | 0 | 6 | 2,503 | 179 | 100.0% |
| **00000007** | 2,061 | 4,369 | 2.120 | 2.129 | 4 | 0 | 6 | 96.99% | 0 | 0 | 1,354 | 123 | 100.0% |
| **00000009** | 6,745 | 13,106 | 1.943 | 2.125 | 5 | 0 | 8 | 96.93% | 0 | 62 | 1,049 | 429 | 100.0% |
| **00000040** | 12,945 | 27,026 | 2.088 | 2.154 | 6 | 0 | 15 | 98.96% | 0 | 0 | 2,995 | 309 | 100.0% |
| **00000052** | 1,770 | 3,577 | 2.021 | 2.312 | 5 | 0 | 7 | 94.97% | 0 | 2 | 262 | 95 | 100.0% |
| **00000056** | 14,759 | 27,631 | 1.872 | 2.106 | 5 | 0 | 24 | 97.75% | 0 | 22 | 3,958 | 1,323 | 100.0% |
| **00000065** | 6,448 | 13,184 | 2.045 | 2.295 | 5 | 0 | 5 | 99.07% | 0 | 4 | 768 | 145 | 100.0% |
| **00000076** | 26,635 | 56,027 | 2.104 | 2.179 | 6 | 0 | 40 | 96.22% | 0 | 29 | 1,486 | 374 | 99.73% |
| **00000077** | 4,424 | 9,622 | 2.175 | 2.220 | 5 | 0 | 11 | 97.13% | 0 | 0 | 3,911 | 268 | 100.0% |
| **00000091** | 6,360 | 12,303 | 1.934 | 2.194 | 6 | 0 | 11 | 98.49% | 2 | 3 | 874 | 232 | 100.0% |

---

### 2. Aggregate Comparison: Training vs. Student-Testing Partitions

| Metric | Train Partition (80 Trips) Mean $\pm$ Std | Train Median [Min, Max] | Student-Test Partition (10 Trips) Mean $\pm$ Std | Student-Test Median [Min, Max] |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Nodes ($|V|$)** | ,168.56 \pm 32,272.30$ | 9,608.0 [299, 155,003] | ,068.90 \pm 7,452.18$ | 6,596.5 [1,770, 26,635] |
| **Number of Arcs ($|E|$)** | ,461.30 \pm 64,738.84$ | 19,526.5 [649, 307,392] | ,504.10 \pm 15,496.93$ | 13,145.0 [3,577, 56,027] |
| **Edge-to-Node Ratio** | .02 \pm 0.13$ | 2.03 [1.63, 2.37] | .04 \pm 0.10$ | 2.07 [1.87, 2.18] |
| **Mean Degree ($ar{k}$)** | .17 \pm 0.15$ | 2.16 [1.85, 2.76] | .19 \pm 0.07$ | 2.17 [2.11, 2.31] |
| **Max Degree ({\max}$)** | .45 \pm 1.03$ | 5.0 [4, 8] | .30 \pm 0.67$ | 5.0 [4, 6] |
| **Connected Components** | .01 \pm 60.68$ | 20.0 [1, 421] | .70 \pm 29.18$ | 11.0 [5, 100] |
| **Giant Comp Coverage (%)** | .90\% \pm 3.88\%$ | 98.11% [69.41%, 100.0%] | .39\% \pm 3.40\%$ | 97.06% [87.40%, 99.07%] |
| **Self Loops ( = v$)** | .28 \pm 1.04$ | 0.0 [0, 8] | .20 \pm 0.63$ | 0.0 [0, 2] |
| **Parallel Edges** | .19 \pm 54.71$ | 2.0 [0, 322] | .80 \pm 19.99$ | 3.5 [0, 62] |
| **Route Continuity Rate** | .94\% \pm 0.24\%$ | 100.0% [98.13%, 100.0%] | .97\% \pm 0.08\%$ | 100.0% [99.73%, 100.0%] |

---

### 3. Key Topological Insights & Downstream HMM/Viterbi Impact
1. **The Disconnected Sub-Island Problem**: Road networks are not single connected graphs (mean of 23–39 components per trip). However, a single dominant **Giant Component covers ~96.6% of nodes**, and **100% of ground-truth routes lie entirely within this Giant Component**. During candidate projection, candidates snapped to disconnected sub-islands must be strictly pruned; otherwise, Viterbi transition probabilities will collapse to zero ((s_t \mid s_{t-1}) = 0$).
2. **Topological Edge Cases**:
   - **Self-Loops**: Present in 9 trips. Specifically, in **Trip 00000035** (	rain), Node 18927 has 2 self-loops (18927 -> 18927) and no edges to other nodes, forming an isolated size-1 island.
   - **Multi-Edges**: 60 out of 90 trips contain parallel edges (dual carriageways/lanes). Vehicle heading alignment features must be used to disambiguate the correct lane/direction.
3. **Route Continuity**: 79/90 trips exhibit 100% strict sequential continuity ( == u_{t+1}$). 11 trips have minor reference path discontinuities (98.13% to 99.73%), demonstrating that the HMM decoder must support transition smoothing over small spatial gaps.

---

### 4. Generated M3 Artifacts
- **Notebook:** 
otebooks/03_m3_road_network_analysis.ipynb (Executed with full outputs and visual displays)
- **Production Script:** src/data/road_network_analysis.py
- **Tables:**
  - 
esults/tables/m3_road_network_per_trip_summary.csv (Complete 30-column metrics for all 90 trips)
  - 
esults/tables/m3_student_test_road_network_summary.csv (10 student-testing trips)
  - 
esults/tables/m3_partition_comparison_table.csv (Train vs. Test aggregates)
  - 
esults/tables/m3_road_network_notes.txt (Comprehensive analytical findings)
- **Figures & Visualizations:**
  - 
esults/figures/m3_road_network_topology_00000000.png (4-panel graph structural decomposition)
  - 
esults/figures/m3_trip_overlay_and_ground_truth_00000000.png (Trajectory overlay with GPS noise & ground truth)
  - 
esults/figures/m3_dataset_road_network_distributions.png (Dataset-wide distributions)
  - 
esults/figures/m3_folium_road_network_trip_00000000.html (Interactive Folium GIS map with Esri base tiles)
  - 
esults/figures/m3_folium_road_network_trip_00000076.html (Interactive Folium GIS map for Trip 76)
