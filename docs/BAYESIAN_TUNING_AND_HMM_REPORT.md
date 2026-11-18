# PathSnap: Bayesian Hyperparameter Optimization & HMM Route Reconstruction Report

This report documents the Bayesian hyperparameter optimization and HMM/Viterbi route decoding for the team's chosen model, **HistGradientBoostingClassifier**, satisfying **Section 4** and **Section 5** of the PathSnap Submission Guidelines.

---

## 1. Bayesian Hyperparameter Optimisation (Section 4)

### Hyperparameters in Search Space
We optimized 5 core regularization and capacity parameters using **Optuna** with a **Tree-structured Parzen Estimator (TPE)** over **30 trials**:

| Hyperparameter | Search Range / Distribution | Description | Best Value (Trial #19) |
| :--- | :---: | :--- | :---: |
| `learning_rate` | $[0.02, 0.25]$ (Log-uniform) | Shrinkage step size for tree updates | **0.06872** |
| `max_iter` | $[60, 220]$ (Step = 20) | Number of boosting trees | **140** |
| `max_leaf_nodes` | $[15, 63]$ (Integer) | Maximum leaves per tree (tree depth control) | **19** |
| `min_samples_leaf` | $[10, 45]$ (Integer) | Minimum training points required in a leaf | **37** |
| `l2_regularization`| $[10^{-4}, 10.0]$ (Log-uniform) | L2 penalty on tree leaf values | **0.00718** |

### Optimization Objective
* **Objective:** Trip-Grouped 5-Fold Cross-Validation **Macro F1-Score** across all 80 training trips (preserving trip boundaries to prevent spatial leakage).
* **Best Objective Score:** **0.8042** CV Macro F1.

### Untuned vs. Tuned Model Performance Comparison

Both configurations were evaluated using the shared trip-grouped 5-fold CV protocol (80 train trips) and the 10 student-test trips with Viterbi decoding:

| Split | Metric | Untuned HistGB | Tuned HistGB | Delta ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Grouped CV** | **Macro F1-Score** | 0.7995 | **0.8010** | **+0.0015** |
| (80 Trips) | **Decoded Top-1 Accuracy** | 86.45% | **86.46%** | **+0.01%** |
| | **Decoded Continuity** | 96.60% | **96.67%** | **+0.07%** |
| | **Candidate Recall** | 79.26% | **79.56%** | **+0.30%** |
| | **Binary Log Loss** | 0.1055 | **0.1054** | -0.0001 |
| | **Total CV Inference Time**| 17.94 s | 21.43 s | (~267 ms / trip) |
| **Student Test** | **Decoded Top-1 Accuracy** | 88.98% | **88.98%** | Stable |
| (10 Trips) | **Decoded Continuity** | 97.25% | **97.31%** | **+0.06%** |
| | **Candidate Recall** | 84.78% | **85.20%** | **+0.42%** |
| | **Macro F1-Score** | 0.8450 | **0.8458** | **+0.0008** |
| | **Binary Accuracy** | 95.87% | **95.87%** | Stable |
| | **Total Test Inference Time**| 1.61 s | 1.94 s | **~194 ms / trip** |

---

## 2. HMM and Viterbi Route Reconstruction (Section 5)

### Methodology & Formulation
1. **Hidden States:** The true road arc $a_t \in \mathcal{K}_t$ at timestamp $t$ among top-K Haversine spatial candidates ($K \approx 8$).
2. **Emission Probabilities:** Posterior probabilities predicted by the tuned HistGradientBoosting model:
   $$\ln \text{Emission}(t, i) = \ln \big(\max(P(a_t = i \mid \mathbf{x}_{t, i}), 10^{-8})\big)$$
3. **Transition Probabilities:** Based on road network directed topology from `.arcs` ($u \to v$):
   $$\ln T(i \to j) = \begin{cases} 0.0 & \text{if } i = j \text{ or } \text{target}(i) == \text{source}(j) \text{ (connected adjacency)} \\ -2.0 & \text{if } \text{target}(i) \ne \text{source}(j) \text{ (disconnection penalty)} \end{cases}$$
4. **Missing Candidates and Large GPS Gaps:** Sequences are segmented at temporal discontinuities (`segment_id`). The Viterbi trellis dynamic programming automatically resets at segment boundaries, preventing artificial transition penalties.
5. **Disconnected Trajectory Segments:** When GPS suggests disconnected roads, the $-2.0$ penalty steers the path toward valid topological paths unless the classifier's emission confidence is exceptionally high.

### Impact of HMM Decoding (Classifier-Only vs. HMM/Viterbi)

| Metric | Classifier-Only (Raw) | Classifier + HMM Viterbi | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Student Test Top-1 Accuracy** | 86.67% | **88.98%** | **+2.31% Accuracy Gain** |
| **Student Test Sequence Continuity**| 90.11% | **97.31%** | **+7.20% Continuity Jump** |
| **Grouped CV Top-1 Accuracy** | 83.63% | **86.46%** | **+2.83% Accuracy Gain** |
| **Grouped CV Sequence Continuity** | 88.25% | **96.67%** | **+8.42% Continuity Jump** |

* **Where HMM Helped:** Successfully suppresses GPS jitter and prevents false matching onto parallel non-connected roads (e.g. frontage roads or highway exit ramps).
* **Where HMM Did Not Help:** Candidate recall misses (0.22% on student-test) where the true arc was never shortlisted by Haversine search.

---

## 3. Artifacts and Reproduction

* **Script:** `src/tune_bayesian.py`
* **Trial History:** `results/bayesian_tuning/tuning_trials.csv`
* **Best Hyperparameters:** `results/bayesian_tuning/best_hyperparameters.json`
* **Optimization Plot:** `results/bayesian_tuning/optimization_history.png`
* **Comparison Benchmarks:** `results/bayesian_tuning/untuned_vs_tuned_comparison.csv`

To reproduce:
```powershell
python src/tune_bayesian.py --data-root "Student folder" --trials 30
```
