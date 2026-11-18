# Shallow MLP: Ordered-Route Proxy Evaluation

## Scope

This report records the shallow MLP run made with
`src/evaluate_families.py --model MLP`. The evaluator also runs the nearest
candidate baseline. It uses the ordered-route proxy and grouped trip folds.
These results are distinct from the earlier independent nearest-route
comparison under `results/model_comparison/`.

The route file does not label every GPS timestamp. The target arc for a GPS
point is inferred from its geometry relative to the ordered reference route.
Consequently, these metrics measure agreement with that proxy, not verified
ground-truth labels at each timestamp.

## Model and protocol

- One hidden layer with 24 units, implemented by scikit-learn
  `MLPClassifier`.
- Standard scaling is applied before fitting.
- The estimator uses its default Adam optimizer and a maximum of 120
  iterations; the random seed is 125.
- Candidate arcs are scored as a binary classification problem: whether a
  candidate arc matches the inferred proxy arc.
- Training and validation are grouped by trip using five folds, so points
  from a trip are not split between its training and validation sets.
- Up to 128 eligible observations are sampled per training trip for each fit.
  Ambiguous labels are excluded from candidate fitting and binary confusion
  counts.
- Fold `-1` is the ten-trip `student_test` partition. The results are not from
  the separate evaluator test set.
- The shared transition-penalized sequence decoder is applied after candidate
  scoring. Candidate misses are counted separately and cannot be recovered
  by the classifier.

## Results

| Split | Raw top-1 proxy agreement | Decoded top-1 proxy agreement | Decoded continuity | Binary accuracy | Binary precision | Binary recall | Binary F1 | Candidate miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grouped cross-validation | 83.78% | 86.28% | 97.04% | 94.78% | 80.50% | 78.51% | 0.795 | 0.63% |
| Student-test | 86.92% | 89.11% | 97.84% | 95.60% | 81.31% | 86.91% | 0.840 | 0.22% |

### Student-test binary candidate confusion counts

Rows are actual proxy-match classes; columns are predicted classes. These
counts are over candidate rows, not GPS observations or complete routes.

| Actual \ Predicted | Not proxy match (0) | Proxy match (1) |
|---|---:|---:|
| Not proxy match (0) | 104,821 | 3,312 |
| Proxy match (1) | 2,171 | 14,408 |

The grouped cross-validation counts are 1,095,573 true negatives, 31,690
false positives, 35,796 false negatives and 130,785 true positives.

The confusion-matrix figure also includes the row-normalized percentages:
`results/ordered_comparison/mlp_confusion_matrices.png`.

## Convergence note

The run completed all five grouped folds and the student-test fold. Scikit-learn
reported that the MLP reached its 120-iteration limit without convergence on
grouped folds 3 and 4. Interpret those scores as results from the configured
iteration budget, not as evidence that the optimizer converged.

## Result files

- `results/ordered_comparison/comparison.csv` — aggregate rows for the MLP and
  the model results currently collected by the team.
- `results/ordered_comparison/trip_metrics.csv` — per-trip rows for the MLP and
  the model results currently collected by the team.
- `results/ordered_comparison/feature_medians_by_trip.csv` — candidate feature
  summaries used for diagnostics.
- `results/ordered_comparison/mlp_confusion_matrices.png` — grouped-CV and
  student-test MLP confusion matrices.
- `src/evaluate_families.py` — evaluation and model implementation.
- `docs/RUN_MODEL_COMPARISON.md` and `docs/ORDERED_ALIGNMENT.md` — setup,
  commands, shared preprocessing, proxy-label definition and limitations.

The evaluator fits models in memory for each fold; this run did not export a
final trained MLP or model weights. Bayesian hyperparameter optimization and
deployment artifacts are also outside this fixed-configuration experiment.
The assignment also requires evaluation across at least five models, Bayesian
hyperparameter optimization, a final model artifact, and API/UI deployment.
This MLP report documents one model run and does not by itself satisfy those
requirements.

## Reproduce

From the repository root in PowerShell, with the supplied dataset available at
`../Student folder`:

```powershell
python src/data/prepare_candidates.py --data-root "../Student folder"
python src/evaluate_families.py --data-root "../Student folder" --model MLP
```

Candidate preparation reads the committed cleaned GPS table and regenerates
trip-local feature and target pickle files under
`results/ordered_pretraining/`. Those generated intermediates are intentionally
not included in version control.
