# PathSnap: Distributed Map-Matching & Route Reconstruction

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
│   └── 01_m1_dataset_inventory.ipynb   # Executed M1 notebook with cell-wise outputs & plots
├── src/                                 # Production source code
│   └── data/
│       └── dataset_inventory.py         # Automated dataset verification & schema generator
├── results/                             # Generated metrics, tables, and visualization artifacts
│   ├── figures/
│   │   └── m1_inventory_distributions.png
│   └── tables/
│       ├── trip_inventory.csv           # Partition-level inventory table
│       ├── file_availability_and_sizes.csv # Detailed file availability & size metrics
│       ├── file_schema.csv              # Exact schema specification
│       └── per_trip_statistics.csv      # Complete metrics for all 90 inspected trips
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
- **Geographical Extent**: Bounding box coordinates center around Darmstadt/Hesse, Germany (Longitude $\approx 8.58^\circ\text{--}8.62^\circ$, Latitude $\approx 49.84^\circ\text{--}49.88^\circ$). Calculations will use Haversine geodesic distance or UTM Zone 32N projection.

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
