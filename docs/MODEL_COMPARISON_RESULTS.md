# Measured model comparison

## Scope and label-quality limitation

The figures below are **legacy nearest-route proxy scores**, not accuracy against independently observed per-point ground truth. They were produced before the ordered-label audit and must not be interpreted as definitive map-matching accuracy. The ordered labels are also inferred from the supplied route, so they remain a useful training/evaluation proxy rather than a direct measurement of the vehicle's true position.

The ordered alignment audit marks labels uncertain when the best alternative route arc changes the whole-trip cost by at most one unit, or when the assigned route arc is more than 50 m away. It found 40,775 / 208,417 training points (19.56%) and 2,545 / 19,160 student-test points (13.28%) uncertain. Of the accepted labels, 1,061 / 167,642 training points (0.63%) and 36 / 16,615 student-test points (0.22%) have no matching arc in the generated candidate set; no scorer can recover those points under the fixed shortlist. The full aggregate is in [the alignment-limitation summary](../results/ordered_pretraining/alignment_limitations_summary.csv) and per-trip evidence is in [the alignment audit](../results/ordered_pretraining/alignment_audit.csv).

Weak GPS-to-route agreement is concentrated rather than uniform. Training trips `00000054` and `00000033` have median residuals of 142.1 m and 122.4 m respectively, with 17,240 and 3,440 points beyond the 50 m cutoff; their inferred point labels should be excluded from fitting and from any claim about model quality. Student-test trip `00000056` contains 27 such distant points, and trips `00000009`, `00000076`, and `00000077` have 18.7%–20.2% uncertain labels; trip `00000076` also has a reference-route discontinuity. Candidate misses must be counted as end-to-end failures, while uncertainty must be reported over all observations rather than dropped from the denominator.

Scores are macro-averages over held-out trips. Top-1 uses the spatial proxy. Raw and decoded scores use the same candidates. Until this comparison is rerun on accepted ordered-label hits, its rankings are exploratory only.

| Model | CV decoded | CV fold range | Student raw | Student decoded | Student decoded continuity | Student F1 (pooled) |
|---|---:|---:|---:|---:|---:|---:|
| Boosting | 86.47% | 83.76%–88.85% | 88.81% | 90.47% | 96.37% | 0.864 |
| Forest | 86.37% | 83.95%–88.61% | 88.54% | 90.66% | 97.16% | 0.864 |
| KNN | 84.61% | 82.19%–87.10% | 87.70% | 89.48% | 96.25% | 0.838 |
| Logistic | 84.05% | 81.56%–85.72% | 87.73% | 88.88% | 98.66% | 0.820 |
| MLP | 86.28% | 83.97%–88.71% | 89.20% | 90.56% | 97.12% | 0.858 |
| Nearest | 71.80% | 68.88%–74.37% | 61.12% | 71.42% | 99.67% | 0.508 |

## Training feature distributions

Each entry is the median of the per-trip class medians. Units are metres for distances. Heading cosine includes the neutral placeholder for unknown headings.

| Proxy match | Distance | Distance above best | Heading cosine |
|---|---:|---:|---:|
| False | 25.82 | 20.41 | -0.114 |
| True | 4.38 | 0.00 | 0.933 |

[Per-trip timings and raw metrics](../results/model_comparison/trip_metrics.csv) · [Aggregate metrics](../results/model_comparison/comparison.csv) · [Confusion matrices](../results/model_comparison/confusion_matrices.png)

## Logistic

L2-regularised binary log loss, C=1, up to 300 iterations.

Mean CV macro-F1 was 0.847. The final fit scored 0.869 on its own sampled candidate rows. Their difference of +0.022 is a generalisation diagnostic. Training sampling excludes candidate misses, so the gap alone cannot isolate overfitting.

Student-test macro-F1 was 0.896, raw top-1 87.73%, and mean trip log loss 0.133. Ordered-route F1 changed from 0.598 to 0.683 after decoding. Final fitting took 0.37 s and scoring plus decoding all ten trips took 1.21 s.

## KNN

Inverse-distance class vote over 11 scaled neighbours. Fitting builds the neighbour index.

Mean CV macro-F1 was 0.866. The final fit scored 1.000 on its own sampled candidate rows. Their difference of +0.134 is a generalisation diagnostic. Training sampling excludes candidate misses, so the gap alone cannot isolate overfitting.

Student-test macro-F1 was 0.906, raw top-1 87.70%, and mean trip log loss 0.231. Ordered-route F1 changed from 0.587 to 0.642 after decoding. Final fitting took 0.45 s and scoring plus decoding all ten trips took 138.92 s.

## Forest

Mean positive class probability across 60 bootstrap trees using Gini impurity and a minimum leaf size of five.

Mean CV macro-F1 was 0.879. The final fit scored 0.939 on its own sampled candidate rows. Their difference of +0.060 is a generalisation diagnostic. Training sampling excludes candidate misses, so the gap alone cannot isolate overfitting.

Student-test macro-F1 was 0.922, raw top-1 88.54%, and mean trip log loss 0.085. Ordered-route F1 changed from 0.598 to 0.665 after decoding. Final fitting took 4.11 s and scoring plus decoding all ten trips took 2.05 s.

## Boosting

Binary log loss, 100 histogram boosting iterations, learning rate 0.1 and up to 31 leaves per tree.

Mean CV macro-F1 was 0.876. The final fit scored 0.922 on its own sampled candidate rows. Their difference of +0.046 is a generalisation diagnostic. Training sampling excludes candidate misses, so the gap alone cannot isolate overfitting.

Student-test macro-F1 was 0.922, raw top-1 88.81%, and mean trip log loss 0.080. Ordered-route F1 changed from 0.596 to 0.649 after decoding. Final fitting took 2.47 s and scoring plus decoding all ten trips took 2.53 s.

## MLP

Binary cross-entropy with Adam, one 24-unit ReLU layer, initial learning rate 0.001 and a 120-epoch limit.

Mean CV macro-F1 was 0.875. The final fit scored 0.904 on its own sampled candidate rows. Their difference of +0.029 is a generalisation diagnostic. Training sampling excludes candidate misses, so the gap alone cannot isolate overfitting.

Student-test macro-F1 was 0.918, raw top-1 89.20%, and mean trip log loss 0.085. Ordered-route F1 changed from 0.609 to 0.661 after decoding. Final fitting took 46.30 s and scoring plus decoding all ten trips took 1.24 s.

Ordered-route F1 is twice the length of the longest common arc subsequence divided by the sum of the two sequence lengths, after collapsing consecutive repetitions. It preserves traversal order and uses the supplied trip route directly. Scores average trips equally. Macro-F1 averages the two candidate classes. CV averages fold-level macro-F1 equally.

[Score comparisons](../results/model_comparison/score_comparison.png) · [MLP learning curve](../results/model_comparison/mlp_learning_curve.png) · [Consolidated metrics](../results/model_comparison/development_comparison.csv)

## How to read the differences

Logistic led student-test ordered-route F1 at 0.683. Forest led decoded proxy top-1 at 90.66%. The first metric compares the traversed arc sequence with the supplied route, while the second gives each labelled observation a vote within its trip. Long stationary stretches and frequent observations on one arc can make these metrics rank models differently.

The KNN training macro-F1 was 1.000. Distance-weighted voting can exactly reproduce many training examples because their distance to themselves is zero. Use its held-out fold scores to assess generalisation.

In the separate 120-epoch MLP diagnostic, training log loss moved from 0.223 to 0.100, and validation log loss from 0.254 to 0.195. The lowest observed validation loss occurred at epoch 120. Timings reflect this machine and concurrent diagnostic work.
