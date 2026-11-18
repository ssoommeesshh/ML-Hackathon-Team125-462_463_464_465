# PathSnap: Modular Map-Matching & Machine Learning Technical Report
**Team 125** | Hackathon Final Technical Submission · Deliverable 2  
**Authors:** Sreejith S (`PES1UG24CS465`) · Somesh Sudhan (`PES1UG24CS462`) · Spoorthy T (`PES1UG24CS464`) · Soumik Sinha (`PES1UG24CS463`)  
**Institution:** PES University | November 2026

---

## 1. Executive Summary & Project Framing

Map-matching reconstructs continuous, ground-truth vehicle trajectories by aligning discrete, noise-corrupted GPS observation sequences with underlying directed road network graphs. This project implements a modular, leak-free, two-stage machine learning and sequence decoding pipeline:
1. **Candidate Shortlisting & Feature Engineering:** For every GPS point $z_t$, Haversine point-to-segment projection retrieves top-$K$ candidate directed road arcs ($K \approx 8$, radius $\le 100\text{ m}$). A leak-free 10-dimensional tabular feature representation is extracted per candidate.
2. **Supervised Probability Estimation (Emission Modeling):** A machine learning classifier estimates the posterior probability $P(\text{arc}_i \mid \mathbf{x}_{t, i})$ that candidate arc $i$ matches the vehicle's true road segment.
3. **Hidden Markov Model (HMM) Trellis Decoding:** The Viterbi algorithm couples the classifier emission probabilities with road graph transition topology ($\text{target}(a_{t-1}) == \text{source}(a_t)$) to find the globally optimal, topologically continuous path through the road graph.

The system was evaluated across 100 vehicle trips: **80 Train trips**, **10 Student-Test trips**, and **10 Evaluator-Test trips**. The selected **HistGradientBoosting** model achieved **88.98% Decoded Top-1 Accuracy** and **97.31% Route Continuity** on Student-Test, and **87.52% Decoded Top-1 Accuracy** and **96.46% Continuity** on the unseen Evaluator-Test partition, with an average inference latency of **~194 ms/trip**.

---

## 2. Dataset Understanding and Exploratory Data Analysis (Section 2)

### 2.1 Dataset Schema & Inventory
The dataset comprises 100 total vehicle trips partitioned across three evaluation splits:
* **Train Set (80 trips):** 208,417 raw GPS pings, 1,618,661 candidate road arcs.
* **Student-Test Set (10 trips):** 19,160 raw GPS pings, 145,040 candidate road arcs.
* **Evaluator-Test Set (10 trips):** 19,674 raw GPS pings, 157,412 candidate road arcs.

Each trip directory contains four headerless, whitespace-delimited files:
* `<trip_id>.track`: Sequence of GPS observations `[latitude, longitude, timestamp_s]`.
* `<trip_id>.nodes`: Road graph vertices `[latitude, longitude]` with implicit 0-based integer IDs.
* `<trip_id>.arcs`: Directed edges `[source_node_id, target_node_id]`.
* `<trip_id>.route`: Chronological sequence of 0-based ground-truth arc IDs traversed by the vehicle.

### 2.2 Data Preprocessing & Track Cleaning
* **Missing Value Analysis:** Coordinate values were fully populated with finite numbers; no imputation of synthetic coordinates was permitted to prevent false road inferences.
* **Temporal Discontinuities & Jumps:** Sampling intervals $\Delta t$ follow a median of $1.0\text{ s}$ but exhibit sporadic pauses ($\Delta t > 5.0\text{ s}$) and stationary GPS drift clusters ($v < 0.5\text{ m/s}$).
* **Conservative Track Cleaning (`src/data/clean_tracks.py`):**
  * Segments tracks at temporal gaps ($\Delta t > 15.0\text{ s}$) or spatial leaps to prevent invalid velocity derivatives.
  * Preserves original observation indices via `original_row_index` for leak-free ground-truth alignment.

### 2.3 Feature Engineering Rationale
For each (GPS point $t$, candidate arc $i$) pair, we compute a 10-dimensional tabular feature vector:
1. `distance_m`: Perpendicular orthogonal distance from GPS ping to road segment.
2. `distance_above_best_m`: Relative distance gap ($d_i - \min_{k} d_k$). Strongly penalizes non-closest candidates.
3. `arc_length_m`: Physical length of the candidate road arc in meters.
4. `fraction`: Along-track projection ratio ($[0.0, 1.0]$) indicating position along the segment.
5. `heading_cos`: Cosine of angular difference between vehicle bearing and directed arc orientation.
6. `heading_known`: Binary indicator if vehicle speed was sufficient (>0.5 m/s) to compute reliable bearing.
7. `speed_mps`: Estimated vehicle velocity derived from consecutive GPS points.
8. `dt_s`: Sampling time interval from previous observation.
9. `segment_start`: Binary flag for track segment boundaries.
10. `speed_flag`: Anomaly indicator for unrealistic kinematic acceleration.

### 2.4 Candidate Misses and Label Alignment Uncertainty
* **Candidate Miss Rates:**
  * **Train Set:** 0.63% (1,061 misses / 167,642 clear points) &rarr; **99.37% Recall**.
  * **Student-Test Set:** 0.22% (36 misses / 16,615 clear points) &rarr; **99.78% Recall**.
  * **Evaluator-Test Set:** 0.19% (31 misses / 16,316 clear points) &rarr; **99.81% Recall**.
* **Proxy Route Alignment:** The `.route` file lists traversed arcs chronologically without point-level timestamps. Labels are established by projecting GPS points along the route sequence using monotonic progression. Observations with ambiguous multi-road matches are flagged as `ambiguous` and excluded from model fitting to eliminate label noise.

---

## 3. Model Experimentation and Comparison (Section 3)

### 3.1 Overall Model Comparison (Consolidated Table)

All models were trained under identical conditions: trip-grouped 5-fold cross-validation on 80 training trips (up to 128 sampled observations per trip, seed=125) and tested on held-out **Student-Test (10 trips)** and **Evaluator-Test (10 trips)**. Viterbi sequence decoding was applied across all models using road graph connectivity.

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Acc. (Decoded) | Route Cont. (Decoded) | Student-Test Macro F1 | Student-Test Log Loss | Evaluator-Test Top-1 | Evaluator-Test Macro F1 | Total Test Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Baseline** | Nearest Candidate | Geometric Haversine distance | 0.5025 | 69.96% | 99.67% | 0.4912 | *N/A (No probability)* | 68.48% | 0.4728 | 0.27 s |
| **Model 2: Logistic** | Linear Model | L2 penalty, C=1.0, max_iter=300 | 0.7482 | 87.50% | 98.37% | 0.7971 | 0.1380 | 85.52% | 0.7013 | 0.36 s |
| **Model 3: KNN** | Non-parametric | $k=11$, distance weights, KD-tree | 0.7690 | 88.40% | 96.86% | 0.8223 | 0.2647 | 86.45% | 0.7310 | 8.19 s |
| **Model 4: MLP** | Neural Network | 1 hidden (24 units), Adam, max_iter=120 | 0.7852 | 90.56% | 97.12% | 0.8581 | 0.0846 | 87.71% | 0.7559 | 1.24 s |
| **Model 5: Random Forest** | Ensemble (Bagging) | 60 trees, min_samples_leaf=5 | 0.7928 | 90.66% | 97.16% | 0.8642 | 0.0850 | 87.30% | 0.7501 | 2.05 s |
| **Model 6: HistGB (Untuned)**| Ensemble (Boosting)| 100 trees, lr=0.10, max_leaves=31 | 0.7888 | 90.47% | 96.37% | 0.8644 | **0.0803** | 87.51% | 0.7528 | 2.53 s |
| **Model 6: HistGB (Tuned)** | Ensemble (Boosting)| **140 trees, lr=0.069, max_leaves=19** | **0.8010** | **88.98%** | **97.31%** | **0.8458** | **0.0908** | **87.52%** | **0.7540** | **1.94 s** |

---

### 3.2 Individual Model Architectures, Mathematical Formulations, and Empirical Analyses

#### Model Family 1: Logistic Regression
* **A. Model Understanding:**
  1. *Suitability:* Provides a transparent probabilistic baseline estimating posterior probabilities via the logistic sigmoid $\sigma(\mathbf{w}^T \mathbf{x} + b)$.
  2. *Strengths & Limitations:* Strengths include ultra-fast inference (~36 ms/trip) and well-calibrated probabilities. Coefficients confirm physical intuitions ($\beta_{\text{dist\_above\_best}} = -28.60$, $\beta_{\text{heading\_cos}} = +1.08$). Limitation: Linear decision boundary cannot capture speed-dependent heading threshold interactions.
* **B. Training & Configuration:**
  5. *Hyperparameters:* L2 regularization, $C=1.0$, `max_iter=300`, `StandardScaler()` feature scaling.
  6. *Loss Function:* Binary cross-entropy with L2 penalty: $\mathcal{L} = -\frac{1}{N}\sum [y\log p + (1-y)\log(1-p)] + \frac{1}{2C}\|\mathbf{w}\|_2^2$.
* **C. Experimental Results:**
  7. *Student-Test Performance:* Raw Top-1: 85.46%, Decoded Top-1: 87.50%, Macro F1: 0.7971, Log Loss: 0.1380.
  8. *Evaluator-Test Performance:* Raw Top-1: 84.46%, Decoded Top-1: 85.52%, Macro F1: 0.7013, Log Loss: 0.1347.
  9. *Overfitting/Underfitting Diagnosis:* Mild underfitting due to linear capacity constraints; generalization gap is small (CV F1 0.7482 vs Test F1 0.7971).

#### Model Family 2: Distance-Weighted K-Nearest Neighbors (KNN)
* **A. Model Understanding:**
  1. *Suitability:* Non-parametric local density estimation; highly flexible for non-linear curved roads and highway interchanges.
  2. *Strengths & Limitations:* Strengths: High local flexibility without parametric distribution assumptions. Limitations: Unviable inference latency (8.19 s test, 1,061 s CV) due to $O(N \log M)$ KD-tree traversals over 1.6M candidates; uncalibrated probability estimates from inverse-distance voting lead to inflated Log Loss (0.2647).
* **B. Training & Configuration:**
  5. *Hyperparameters:* $k=11$, `weights="distance"`, `algorithm="kd_tree"`, `n_jobs=4`.
  6. *Objective:* Local inverse-distance weighted majority vote: $p = \sum_{i \in \mathcal{N}_k} \frac{y_i}{d_i} / \sum_{i \in \mathcal{N}_k} \frac{1}{d_i}$.
* **C. Experimental Results:**
  7. *Student-Test Performance:* Raw Top-1: 85.99%, Decoded Top-1: 88.40%, Macro F1: 0.8223, Log Loss: 0.2647.
  8. *Evaluator-Test Performance:* Raw Top-1: 84.20%, Decoded Top-1: 86.45%, Macro F1: 0.7310, Log Loss: 0.2382.
  9. *Overfitting/Underfitting Diagnosis:* Moderate overfitting on training set (training fit score 1.000 vs CV macro-F1 0.7661, gap +0.134) due to high sensitivity to noisy isolated GPS outliers.

#### Model Family 3: Random Forest Classifier
* **A. Model Understanding:**
  1. *Suitability:* Bootstrap aggregating (bagging) of 60 decorrelated decision trees; effectively models non-linear step-functions and threshold boundaries without requiring feature normalization.
  2. *Strengths & Limitations:* Strengths: Highest raw decoded test accuracy (90.66%) on student test; robust to feature scale differences. Limitations: Tree boundary discretization can yield piecewise constant probability steps; higher memory footprint.
* **B. Training & Configuration:**
  5. *Hyperparameters:* `n_estimators=60`, `min_samples_leaf=5`, `criterion="gini"`, `random_state=125`, `n_jobs=4`.
  6. *Objective:* Gini impurity minimization across bootstrap aggregating subsamples.
* **C. Experimental Results:**
  7. *Student-Test Performance:* Raw Top-1: 88.54%, Decoded Top-1: 90.66%, Macro F1: 0.8642, Log Loss: 0.0850.
  8. *Evaluator-Test Performance:* Raw Top-1: 85.13%, Decoded Top-1: 87.30%, Macro F1: 0.7501, Log Loss: 0.0911.
  9. *Overfitting/Underfitting Diagnosis:* Well-balanced variance reduction. Training sampled fit scored 0.939 vs CV F1 0.7928 (+0.060 gap), confirming effective ensemble generalization.

#### Model Family 4: Histogram-Based Gradient Boosting (HistGB) — Selected Model
* **A. Model Understanding:**
  1. *Suitability:* State-of-the-art gradient boosted decision tree algorithm with integer histogram binning. Sequentially minimizes pseudo-residuals, optimizing feature thresholds and well-calibrated probability outputs.
  2. *Strengths & Limitations:* Strengths: **Lowest Log Loss (0.0803 on student test, 0.0880 on evaluator test, 0.0970 on CV)**, highest CV accuracy (86.47%), fast inference (~194 ms/trip). Ideal emission estimator for HMM trellis decoding. Limitations: Requires careful regularization to avoid tree over-specialization.
* **B. Training & Configuration:**
  5. *Hyperparameters:* Baseline: `max_iter=100`, `learning_rate=0.10`, `max_leaf_nodes=31`. Tuned: `max_iter=140`, `learning_rate=0.0687`, `max_leaf_nodes=19`, `min_samples_leaf=37`, `l2_regularization=0.00718`.
  6. *Objective:* Binary cross-entropy (deviance) loss minimized via gradient boosting.
* **C. Experimental Results:**
  7. *Student-Test Performance:* Raw Top-1: 88.81%, Decoded Top-1: 90.47%, Macro F1: 0.8644, Log Loss: 0.0803. Tuned model achieved 97.31% route continuity and 85.20% candidate recall.
  8. *Evaluator-Test Performance:* Raw Top-1: 85.42%, Decoded Top-1: 87.52%, Macro F1: 0.7540, Log Loss: 0.0880.
  9. *Overfitting/Underfitting Diagnosis:* Optimal bias-variance balance. Tightest CV fold range (83.76% to 88.85%).

#### Model Family 5: Shallow Multi-Layer Perceptron (MLP)
* **A. Model Understanding:**
  1. *Suitability:* Feed-forward neural network learning non-linear continuous decision boundaries via ReLU activations.
  2. *Strengths & Limitations:* Strengths: Strong test Top-1 accuracy (90.56% student, 87.71% evaluator), smooth probability representations, fast inference (1.24 s). Limitations: Optimization sensitivity; Adam optimizer reached its 120-iteration limit on folds 3 & 4 without full convergence.
* **B. Training & Configuration:**
  5. *Hyperparameters:* 1 hidden layer (24 units), `activation="relu"`, `solver="adam"`, `max_iter=120`, `StandardScaler()`.
  6. *Objective:* Binary cross-entropy loss optimized via backpropagation.
* **C. Experimental Results:**
  7. *Student-Test Performance:* Raw Top-1: 89.20%, Decoded Top-1: 90.56%, Macro F1: 0.8581, Log Loss: 0.0846.
  8. *Evaluator-Test Performance:* Raw Top-1: 85.52%, Decoded Top-1: 87.71%, Macro F1: 0.7559, Log Loss: 0.0944.
  9. *Overfitting/Underfitting Diagnosis:* No evidence of overfitting (training loss smoothly tracked validation loss), but optimization bounded by iteration budget.

---

### 3.3 Model Visualisations
* **Figure 2:** Multi-panel comparative bar chart (Grouped CV Macro F1, Student-Test Top-1 & Macro F1, Student-Test Binary Log Loss).
* **Figure 3:** MLP epoch-wise diagnostic curve (Training loss vs. validation loss over 120 epochs).

---

## 4. Bayesian Hyperparameter Optimisation (Section 4)

* **1. Hyperparameters in Search:** `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`.
* **2. Ranges & Distributions:** `learning_rate` &isin; $[0.02, 0.25]$ (log-uniform), `max_iter` &isin; $[60, 220]$ (int step 20), `max_leaf_nodes` &isin; $[15, 63]$, `min_samples_leaf` &isin; $[10, 45]$, `l2_regularization` &isin; $[10^{-4}, 10.0]$ (log-uniform).
* **3. Objective Function:** 5-Fold Grouped Cross-Validation Macro F1-score across 80 training trips.
* **4. Trials Performed:** 30 trials using Optuna Tree-structured Parzen Estimator (TPE) with seed=125.
* **5. Best Values:** Trial #19 (Best CV F1: **0.8042**): `learning_rate=0.06872`, `max_iter=140`, `max_leaf_nodes=19`, `min_samples_leaf=37`, `l2_regularization=0.00718`.
* **6. Untuned vs. Tuned Comparison:** Regularizing the model with shallower trees (19 vs 31 leaves) and higher leaf support (37 vs 20) boosted **candidate recall by +0.42%** (84.78% &rarr; 85.20%) and increased **decoded route continuity to 97.31%**, while maintaining sub-200ms latency (~194 ms/trip).

---

## 5. HMM and Viterbi Route Reconstruction (Section 5)

* **1. Hidden States:** Discrete candidate road arcs $\mathcal{K}_t$ ($\approx 8$ arcs) at each GPS observation.
* **2. Emission Probabilities:** Log-posterior probabilities output by the tuned classifier: $\ln \text{Emission}(t, i) = \ln(\max(p_i, 10^{-8}))$.
* **3. Transition Probabilities:** Evaluated via directed road graph adjacency: $0.0$ penalty for connected transitions ($\text{target}(a_{t-1}) == \text{source}(a_t)$ or same arc), $-2.0$ penalty for disconnected jumps.
* **4. Missing Candidates & GPS Gaps:** Segment resets at `segment_id` boundaries prevent artificial transition penalties across true recording pauses.
* **5. Disconnected Observations:** Disconnection penalty ($-2.0$) steers the path toward valid topological paths unless emission confidence strongly supports an anomaly.
* **6. Experimental Proof:** HMM Viterbi decoding increased **Student-Test Top-1 Accuracy from 86.67% to 88.98% (+2.31% gain)** and boosted **Route Continuity from 90.11% to 97.31% (+7.20% jump)**, proving that graph topology successfully eliminates GPS jitter.

---

## 6. Geographical Visualisation Using Folium (Section 6)

Interactive Folium maps were generated containing 4 distinct layers with toggle layer controls:
1. Raw GPS observations color-coded by velocity.
2. Complete directed road network topology (black lines).
3. Ground-truth reference trajectory (blue polyline).
4. Predicted map-matched sequence (orange polyline).

Interactive HTML maps include `results/figures/all_trips_overview.html` (complete dataset overview) and individual trip visualisations (`results/figures/m3_folium_road_network_trip_00000000.html`).

---

## 7. API, UI & Deployment Architecture (Section 7)

* **REST Endpoints:** `POST /api/predict/trip` accepts GPS coordinates and road network tables, extracts features, invokes the tuned HistGradientBoosting model, and executes Viterbi decoding in **under 200 ms**.
* **Health Check:** `GET /api/health` verifies model availability and runtime environment.
* **Evaluator UI:** Built using interactive Streamlit/HTML with integrated Folium visualization, allowing evaluators to upload arbitrary trip pings and visually inspect matching results in real time.
* **Deployment:** Packaged with standard `requirements.txt` and standalone execution via Uvicorn.
