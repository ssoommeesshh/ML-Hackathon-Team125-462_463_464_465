# GPS labels for the next model comparison

Use `results/ordered_pretraining` for all five scorers. The input features, candidate rows and validation folds match the earlier experiment exactly. Labels are inferred offline from the full supplied route independently of the candidate shortlist. A candidate miss is therefore observable.

The optimisation assigns each point a zero-based occurrence in the ordered route. It minimises the sum of point-to-arc distances capped at 50 m, plus 0.02 times forward travel measured between arc midpoints in metres. At a cleaned segment boundary the travel penalty resets while route order still holds. Repeated occurrences of an arc remain separate states. Points may share an occurrence and may skip occurrences. Open endpoints allow partial GPS coverage. Order is enforced between arc occurrences only. Motion within a single arc and instantaneous speed are unconstrained.

The alternative margin measures how much the best whole-trip cost increases if that observation must take a different arc ID. A margin of one cost unit or less, or an assigned distance over 50 m, marks a point uncertain. The distance cap limits the influence of severe offsets. These thresholds are engineering choices and the margin is not a calibrated probability. Exact timestamps for route arcs are unavailable.

| Partition | Points | Uncertain | Accepted | Accepted arc absent | Accepted arc in candidates |
|---|---:|---:|---:|---:|---:|
| Train | 208417 | 40775 | 167642 | 1061 | 166581 |
| Student-test | 19160 | 2545 | 16615 | 36 | 16579 |

Compared with independent nearest-route labels, 12492 training and 1662 student-test assignments changed, including uncertain points. Filtering distant points changes the coverage denominator. Report uncertainty over all points and candidate misses over accepted points. Fit on accepted hits. Count accepted misses as failures in end-to-end top-1.

The audit verified all 90 route-index lookups, forward progression, original GPS keys, candidate feature frames and fold assignments. Trip 00000001 shows steady progression with median distance 4.3 m. Trip 00000076, which has a reference discontinuity, progresses from occurrence 0 to 373 and has 299 uncertain points, all from the margin rule. Training trip 00000054 reaches its last occurrence early and then shows a long plateau. Its median residual is 142 m and 17240 points exceed the distance cutoff. A monotone solution alone would conceal this failure. Closely spaced service roads, opposite directions and erroneous reference geometry can still yield incorrect accepted labels.

From the repository root run `python src/data/prepare_candidates.py --data-root "../Student folder"` then `python src/data/audit_ordered.py --data-root "../Student folder"`. See [per-trip audit](../results/ordered_pretraining/alignment_audit.csv) and [maps and progression plots](../results/ordered_pretraining/alignment_examples.png). Previous scores in model_comparison used independent nearest-route targets.
