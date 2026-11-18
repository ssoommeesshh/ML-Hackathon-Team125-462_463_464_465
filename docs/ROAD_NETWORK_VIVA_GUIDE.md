# PathSnap — Milestone M3: Road-Network Analysis
## Viva Defense & Presentation Guide for Review 1

**Team 125** | PES University — UE24CS352A Machine Learning  
**Author Role:** Road-Network Topological & Connectivity Analysis  
**Evaluator Target:** Viva Presentation & Faculty Cross-Question Defense  

---

## 🎯 1. Structured Presentation Flow (What to Say & Show)

### Step 1: Opening Statement (30–45 Seconds)
> *"Good morning / afternoon Ma'am. For Phase 1, I was responsible for **Milestone M3: Road-Network Analysis**. Rather than treating the road network simply as a set of background coordinates, I performed a rigorous graph-theoretical and topological audit across all **90 trips** (80 training + 10 student-test trips).*
> 
> *I constructed both **directed multigraphs** and **undirected projections** for each trip to quantify network scale, degree connectivity, component fragmentation, and ground-truth route feasibility. Our key discovery is that **no road network in the dataset is a single connected graph**—each trip contains an average of **23 to 39 disconnected sub-islands**, with a single **Giant Component covering ~96.6% of nodes**. Crucially, we verified that **100% of ground-truth routes lie entirely within this Giant Component**.*
> 
> *Let me walk you through our notebook, tables, and visualizations."*

---

### Step 2: What Screen / Artifacts to Show
1. **Show the Executed Notebook:** Open `notebooks/03_m3_road_network_analysis.ipynb` (demonstrates live code, clean cell execution, and pre-computed tables).
2. **Show the Summary Table:** Scroll to the **10 Student-Test Trips Summary Table** (Section 3).
3. **Show the 4-Panel Topology Figure:** Point to `m3_road_network_topology_00000000.png` (Section 5).
4. **Show the Interactive GIS Map:** Switch to browser tab displaying `m3_folium_road_network_trip_00000000.html` and toggle the layers using the top-right LayerControl.

---

## ⚙️ 2. Backend Architecture & Mechanics (How It Works)

### Graph Data Pipeline
```
Raw Files (.nodes, .arcs)
       │
       ▼
1. Node Parser: (x, y) = (lon, lat)
   ↳ 0-based row index implicitly defines node_id (0 to |V|-1)
       │
       ▼
2. Arc Parser: (u, v) = (from_node_id, to_node_id)
   ↳ 0-based row index implicitly defines arc_id (0 to |E|-1)
       │
       ▼
3. NetworkX Dual-Graph Construction:
   ├── DiGraph G_dir: Preserves one-way / two-way street directions
   └── Graph G_undir: Undirected projection for topological connectivity
       │
       ▼
4. Algorithmic Metric Extraction:
   ├── Connected Components: BFS/DFS traversal over G_undir -> O(V + E)
   ├── Strongly Connected Components: Tarjan's SCC algorithm over G_dir -> O(V + E)
   ├── In/Out/Total Degree Spectrum: In-degree and Out-degree distributions
   └── Route Alignment & Continuity: Sequential matching of route arc_ids
```

### Tile Server & GIS Backend Fix
- **Initial Issue:** Default Leaflet/Folium tiles (`tile.openstreetmap.org`) return `HTTP 403 Forbidden` when loaded from local `file:///` paths because OpenStreetMap's volunteer tile policy blocks requests lacking standard web `Referer` headers.
- **Backend Fix:** Integrated **Esri World Street Map** (`server.arcgisonline.com`) as the primary tile provider. It allows local file rendering, requires no API key, and provides clear, high-contrast street geometry with zero rate-limit blocks.

---

## 📊 3. The 5 Crucial Numbers to Know by Heart

| Metric | Train Partition (80 Trips) | Student-Test Partition (10 Trips) | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **Branching Ratio ($\|E\|/\|V\|$)** | **$2.02 \pm 0.13$** | **$2.04 \pm 0.10$** | **Linear Invariant ($\|E\| \approx 2.01\|V\|$):** Proves planar street topology where bidirectional roads have opposite directed arcs ($(u, v)$ and $(v, u)$). |
| **Average Node Degree ($\bar{k}$)** | **$2.17$** (Max: 8) | **$2.19$** (Max: 6) | Real-world junctions: dead ends ($k=1$), straight segments ($k=2$), T-junctions ($k=3$), and 4-way crossings ($k=4$). |
| **Disconnected Components** | **$39.01$** avg (range: 1 to 421) | **$22.70$** avg (range: 5 to 100) | Road extracts contain isolated subgraphs (parking lots, residential enclaves, footpaths). |
| **Giant Component Coverage** | **$96.90\%$** of all nodes | **$96.39\%$** of all nodes | The primary interconnected street core accounts for >96% of the navigable space. |
| **Route in Giant Component** | **100.0%** (80/80 trips) | **100.0%** (10/10 trips) | The vehicle path never touches a disconnected island. |

---

## ⚔️ 4. Anticipated Faculty Cross-Questions & Bulletproof Answers

### Q1: *"Why did you model both a directed and an undirected graph?"*
> **Answer:** *"Real-world road networks are fundamentally directed because streets have legal traffic flow constraints (one-way streets vs. two-way avenues). We use the **Directed Graph ($\mathcal{G}_{\text{dir}}$)** to evaluate legal vehicle transitions and parameterize the HMM transition matrix. However, to evaluate physical **spatial reachability and network fragmentation**, graph theory uses the **Undirected Projection ($\mathcal{G}_{\text{undir}}$)** to identify connected components. Using both gave us full structural visibility."*

---

### Q2: *"Why are there disconnected components in the road network? Is the data broken?"*
> **Answer:** *"No, Ma'am, the data is not broken. When bounding-box bounding queries extract road networks around a GPS track from OpenStreetMap, smaller localized road snippets—such as gated communities, dead-end service alleys, pedestrian paths, or multi-level parking ramps—are included in the geographic bounding box but lack mapped road connections to the main arterial grid. Across all 90 trips, these disconnected components are very small (typically 2 to 8 nodes each) and account for only ~3.5% of nodes."*

---

### Q3: *"What is an isolated node vs. a disconnected component? Did you find any isolated nodes?"*
> **Answer:** *"An **isolated node** has degree 0—meaning it has no incident arcs at all. A **disconnected component** is a maximal connected subgraph that has no edges connecting it to the rest of the network.*
> 
> *In our audit, **zero isolated nodes ($deg=0$) exist in 89 of 90 trips**. However, we uncovered a fascinating topological edge case in **Trip 00000035** (`train`): **Node 18927** has two self-loops to itself (`18927 -> 18927`) and **zero connections to any other node**. It forms a unique size-1 disconnected component. Our code explicitly detects and flags this edge case so graph traversal algorithms don't enter infinite loops."*

---

### Q4: *"How does your road-network analysis directly impact candidate generation and the HMM?"*
> **Answer:** *"This is the direct bridge to Phase 2:
> 1. **Candidate Pruning:** Since we proved that 100% of ground-truth routes lie in the Giant Component, our candidate generator can immediately discard any candidate road segment on a disconnected sub-island. If a noisy GPS fix were projected onto an island, the HMM Viterbi transition probability to the main network would be zero ($-\infty$ in log-space), breaking the sequence decoder.
> 2. **Multi-Edge Resolution:** 60 of 90 trips contain parallel edges (multiple arcs between the same endpoints, representing multi-lane or dual carriageways). Euclidean distance alone cannot distinguish between parallel arcs, so we must feed **vehicle heading and bearing alignment** into our classifier."*

---

### Q5: *"Why is Route Continuity less than 100% in Trip 00000076?"*
> **Answer:** *"In 79 of 90 trips, route continuity is exactly 100.0%, meaning every successive arc satisfies $to\_node(e_t) = from\_node(e_{t+1})$. However, in 11 trips—including Student-Test Trip 76 (99.73% continuity)—there are minor physical gap jumps where consecutive ground-truth arcs do not share a junction vertex. This aligns with Section 6 of `Dataset Exploration.pdf`, which notes that reference routes contain physical artifacts and GPS dropouts. For our HMM, this means transition probabilities cannot be hard zero for non-adjacent arcs; we must allow a small transition floor probability $\epsilon$ or compute shortest-path network distances."*

---

### Q6: *"What is the time complexity of your road analysis?"*
> **Answer:** *"Building the graph from adjacency pairs is $O(V + E)$. Finding connected components using Breadth-First Search (BFS) and strongly connected components using Tarjan's algorithm both run in linear time $\mathcal{O}(V + E)$. Degree counting is $\mathcal{O}(V)$. Across the entire dataset of 2 million nodes and 4 million arcs, our pipeline completes the entire 90-trip audit in approximately 60 seconds."*

---

## 📁 5. Repository Verification Checklist
- [x] **Jupyter Notebook:** `notebooks/03_m3_road_network_analysis.ipynb` (Executed with live outputs)
- [x] **Production Script:** `src/data/road_network_analysis.py`
- [x] **Summary Tables:** `results/tables/m3_road_network_per_trip_summary.csv` & `m3_student_test_road_network_summary.csv`
- [x] **Comparative Table:** `results/tables/m3_partition_comparison_table.csv`
- [x] **Publication Figures:** `results/figures/m3_road_network_topology_00000000.png`, `m3_trip_overlay_and_ground_truth_00000000.png`, `m3_dataset_road_network_distributions.png`
- [x] **Interactive GIS Maps:** `results/figures/m3_folium_road_network_trip_00000000.html` & `00000076.html`
- [x] **GitHub Status:** Committed (`27539d4`) and pushed to `main` branch.
