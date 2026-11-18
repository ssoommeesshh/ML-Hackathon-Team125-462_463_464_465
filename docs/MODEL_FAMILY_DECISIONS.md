# Choosing the comparison set

The experiment asks which scoring rule puts the best directed road arc first for a GPS observation on a new trip. The team selected five assignment families. The nearest-segment rule provides a sixth, geometric reference. Each scorer sees the same candidate arcs and passes its scores to the same sequence decoder.

| Assignment family | Owner and representative | Benefit in this dataset | Limitation or reason for deferral |
|---|---|---|---|
| Linear and probabilistic | M1 · Logistic Regression | Scaled distance and heading effects give a transparent probability baseline. | A linear logit gives heading one coefficient even when its usefulness changes with speed. Ridge classifiers need a probability conversion, while Gaussian naive Bayes makes a poor independence assumption for absolute and relative distance. |
| Instance-based | M1 · distance-weighted KNN | Local examples can capture combinations of heading, speed and projection position. Scaling gives heading a comparable influence to metres. | Neighbour search adds prediction cost and repeated road configurations can produce very confident votes. Eleven neighbours with inverse-distance weighting smooth the effect of individual observations. A radius rule could leave sparse queries without neighbours. |
| Tree-based | Deferred · single Decision Tree | Inspectable thresholds could explain errors on individual roads. | A single tree's thresholds can vary across trips. The forest already tests tree splits with variance reduction. Extra Trees would give another closely related comparison. |
| Ensemble | M2 · Random Forest | Bagging captures interactions such as heading mattering most when two roads have similar distances. | Memory and scoring time grow with tree count. Sixty trees and a minimum leaf size of five bound the initial experiment. An additional bagging variant would duplicate much of this test. |
| Boosting | M3 · HistGradientBoosting | Sequential corrections can capture subtle distance and heading interactions. Histogram bins keep training practical. | Binning reduces resolution and repeated fitting can emphasize weak-label errors. A fixed 100 iterations keeps the initial comparison reproducible. Further boosting libraries add dependencies before the family comparison is settled. |
| Kernel-based | Deferred · linear or RBF SVM | Nonlinear margins could separate confusing candidate geometries. | RBF fitting scales poorly with many candidate rows and probability calibration adds a grouped validation step. Logistic regression covers the linear baseline. |
| Neural networks | M4 · shallow MLP | A 24-unit hidden layer learns smooth interactions from the same scaled inputs. | Scaling, iteration budget and seed affect convergence. A deeper network adds parameters without additional input information. |
| Unsupervised and clustering | Deferred · K-Means, DBSCAN or agglomerative clustering | Could identify operating regimes such as stops and poor GPS coverage. | Cluster IDs describe feature similarity. Converting them to probabilities of a particular directed arc requires a separate supervised mapping and evaluation. |

## Shared protocol

Within the linear family, SGD could reduce fit cost on a larger training set but adds learning-rate sensitivity. Radius Neighbours could expose low-density queries, whereas fixed K guarantees a vote. A standalone Extra Trees ensemble would broaden the randomisation experiment once bagging versus boosting has been measured. XGBoost, LightGBM and CatBoost mainly add implementation and tuning choices here because all ten inputs are numeric. Polynomial SVM adds degree sensitivity, and linear SVM would overlap with the linear baseline. Mini-Batch K-Means offers cheaper clustering, DBSCAN identifies noise under a density threshold, and agglomerative clustering exposes nested structure. All three still require an additional mapping from clusters to road-candidate relevance.

Use the pretraining manifest and its five trip folds. Each fit includes up to 128 randomly selected eligible observations per training trip, retaining all candidates for each. The seed is 125 plus numeric trip ID. This budget gives short trips representation and makes KNN practical. It discards some training information. Full-trip evaluation includes all unambiguous observations, including candidate misses. The development comparison fits each model again using samples from all 80 training trips. Fit scalers inside each fit for Logistic, KNN and MLP. Axis-aligned trees use features in their original units.

Supervision comes from the spatial proxy described in [the handoff](PRETRAINING_HANDOFF.md). A candidate matches when its directed arc equals that proxy. Distinct route arcs within one metre of the best distance make an observation ambiguous. This geometry-based construction favours distance-sensitive scores. Direction on a parallel road and repeated traversals remain sources of uncertainty. Inputs exclude IDs and reference-route information.

## Feature decisions

| Input | Reason for inclusion and interpretation |
|---|---|
| Candidate distance and distance above best candidate | Absolute distance captures poor GPS-to-map agreement. Subtracting the minimum for that observation measures the cost of selecting an alternative. A nearest-distance tie gives both directions zero relative distance. Computation uses every candidate and requires no labels. |
| Arc length and projection fraction | Describe how far the segment extends and whether the GPS projection lies at a junction or in its interior. Fractions use directed endpoint order. |
| Heading cosine and heading-known indicator | Opposite arcs can have identical distance but opposite direction. A segment start or motion below two metres makes heading unreliable. The indicator distinguishes an unavailable cosine from a perpendicular heading. |
| Speed, elapsed time, speed flag and segment start | Carry observation reliability into the candidate score. These quantities are shared across one observation's candidates, so their ranking value comes mainly through nonlinear interactions. Segment starts reset motion and decoding. |

The per-trip feature medians file gives descriptive positive and negative candidate distributions among eligible observations. Comparisons across trips avoid weighting long routes heavily. Latitude/longitude, road IDs and trip IDs could identify particular locations. The chosen inputs transfer across networks through relative geometry and metre-based distances.

### What the training observations show

Across the 208,417 training points, 160,881 had two candidate arcs whose distances differed by at most one metre. For the 155,353 such observations with a usable direction of motion, nearest-segment top-1 was 41.1% among unambiguous proxies. With a distance separation above one metre and heading available, it was 84.2%. The ambiguous proxy fractions in these groups were 19.6% and 2.8% respectively. This supports testing features that distinguish arc direction and shows where spatial supervision is weakest. The association alone cannot quantify the causal gain from heading. Stationary and segment-start observations retain their own heading-availability indicator.

Run `python src/feature_diagnostics.py` to reproduce [the strata counts](../results/model_comparison/training_feature_strata.csv) and [the training EDA figure](../results/model_comparison/training_feature_strata.png).

## Reading the evaluation

The primary comparison is mean per-trip decoded top-1 against unambiguous proxy labels on grouped CV. Candidate-level confusion counts, precision, recall, F1 and accuracy pool rows at threshold 0.5. Log loss and pointwise top-1 average trip scores. The proxy candidate miss rate pools unambiguous observations, including misses even when a classifier cannot recover them. The nearest baseline ranks by distance, using exp(−distance/20 m) as a similarity for decoding. Its binary threshold metrics are descriptive and its score has no fitted probability calibration.

The shared Viterbi decoder treats candidate arcs as states. It sums log candidate scores, subtracts two log units for a change between non-adjacent arcs, permits staying on the same arc and resets at each cleaned segment. Dynamic programming retains the best predecessor per state and backtracks. The finite penalty permits movement over intermediate arcs absent from the candidate sets. This fixed approximation can reward staying on an incorrect road. Strict continuity counts within-segment transitions that stay on an arc or follow its directed endpoint. Report continuity alongside accuracy. Candidate miss rate and proxy ambiguity bound the interpretation of both.

Run from the repository root with `python src/evaluate_families.py --data-root "../Student folder"`. The script records per-trip metrics, fold identifiers, fit row counts, fit time and combined scoring/decoding time. Fit time repeats across that fold's trip records. The comparison is the initial fixed-configuration experiment. Bayesian optimisation, final model choice and deployment are subsequent assignment stages.

Run `python src/sequence_diagnostics.py --data-root "../Student folder"` for the full-training replay, ordered-route scores and a separate MLP epoch diagnostic. The epoch diagnostic uses the same deterministic observation sample and fold 0 as its held-out validation set. Both curves use unregularised binary log loss evaluated after each partial-fit epoch. This diagnostic illustrates the learning trajectory on that fixed subset. Run `python src/report_families.py` to regenerate the comparison notes and figures from the measured outputs. This run used scikit-learn 1.9.1.
