# Preparing the first training run

Run from the repository root using its Python requirements and the existing cleaned GPS table.

~~~powershell
python src/data/prepare_candidates.py --data-root "../Student folder"
~~~

The output directory defaults to `results/pretraining`. Its manifest identifies every partition, eight-digit trip ID, candidate count, ambiguous reference count, clear candidate misses and validation fold. Each trip has separate pandas pickle files under `features/<partition>/` and `targets/<partition>/`. Load only locally generated pickles. Per-trip storage bounds memory and preserves the identity of graph-local arc IDs.

## Decisions and limitations

| Decision | Rationale and tradeoff |
|---|---|
| Use cleaned points in file order | The M2 audit found 617 gaps over 10 seconds. Reset motion at each segment start to avoid inventing a journey across those breaks. Retain the existing 180 km/h indicator to expose questionable speeds. |
| Use trip-local metric coordinates | Longitude scale depends on latitude. A tangent-plane approximation centred at the trip mean supports orthogonal projections onto short road segments. Distances are approximate metres. This version assumes the complete trip is available when processing it. |
| Search the largest weakly connected component | M3 found the supplied routes inside this component. This narrows the search but can miss a real journey on a smaller component. Keep directed parallel arcs as separate candidates and omit zero-length geometry. |
| Spatial shortlist | Sample the centre of each successive arc subdivision of at most 30 m. Retrieve 48 nearest samples, deduplicate by arc, compute clamped point-to-segment distance, and keep up to eight arcs. This includes long arc interiors and bounds candidate rows. It is approximate in dense networks. Equal distances resolve by arc ID. With no radius cutoff even remote observations receive candidates, so inspect distance and coverage. These are initial engineering settings, not tuned parameters. |
| Retain distance, position, direction and motion | See the exact input list below. A two-metre motion minimum suppresses unstable heading at near-stationary points. Direction cosine distinguishes travel along and against a directed arc. The validity indicators distinguish neutral placeholders from observed motion. |
| Construct weak observation targets offline | The supplied route lists traversed arcs without point timestamps. Find the nearest segment among all unique route arcs independently of candidate generation. Mark observations ambiguous when two distinct route arcs fall within 1 m of the best distance. Repeated traversal of the same arc shares its ID. This spatial proxy loses traversal order and can be wrong near junctions, revisits, parallel roads or GPS drift. Inspect reference distance and report proxy-based metrics separately from ordered route evaluation. |
| Keep ambiguous cases and misses in the audit | For initial supervised fitting, use observations with a unique proxy present in the shortlist. For evaluation, compute candidate miss rate over all unambiguous observations and count misses as failures for end-to-end top-1. Report the ambiguous fraction separately. Keeping only successful candidate sets in the evaluation denominator would inflate performance. |
| Five deterministic trip folds | Assign training trips largest first to the fold with the fewest candidate rows. Each trip and all its segments stay together. Fold -1 identifies student-test. Fit scalers, imputers and learned transformations inside each training fold once a model requires them. This keeps the same raw numerical inputs available for comparing models. |

## Loading inputs and supervision

From the repository root, this example prepares one training trip. Apply the same operation separately to the training and validation trip lists from the manifest.

~~~python
from pathlib import Path
import pandas as pd
from src.data.prepare_candidates import FEATURES

root = Path("results/pretraining")
manifest = pd.read_csv(root / "manifest.csv", dtype={"trip_id": str})
train = manifest.query("partition == 'train' and validation_fold != 0")
validation = manifest.query("partition == 'train' and validation_fold == 0")
trip = train.iloc[0].trip_id
candidates = pd.read_pickle(root / "features" / "train" / f"{trip}.pkl")
labels = pd.read_pickle(root / "targets" / "train" / f"{trip}.pkl")
rows = candidates.merge(labels.query("~ambiguous and in_candidates"),
                        on="original_row_index", validate="many_to_one")
X = rows[FEATURES]
y = rows.arc_id.eq(rows.proxy_arc_id)
~~~

The numeric inputs are perpendicular or endpoint distance in metres (`distance_m`), directed arc length in metres, fractional position from source to destination (0–1), cosine of heading difference (−1 to 1), heading availability, incoming speed in m/s, elapsed seconds, segment-start indicator and speed-review flag. Arc IDs, segment IDs and row indices identify records. Preserve them for scoring and route reconstruction. Prediction uses only the feature files. Ground-truth routes, proxy IDs, ambiguity, coverage and reference distance belong exclusively to supervision and evaluation.
