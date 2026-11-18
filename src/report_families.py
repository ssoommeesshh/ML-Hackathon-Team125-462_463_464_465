"""Render the measured comparison and descriptive feature statistics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "results/model_comparison"
scores = pd.read_csv(out/"comparison.csv")
trips = pd.read_csv(out/"trip_metrics.csv")
manifest = pd.read_csv(root/"results/pretraining/manifest.csv")
covered = trips.merge(manifest[["trip_id","validation_fold","points"]],left_on="trip",
                      right_on="trip_id",suffixes=("","_expected"),validate="many_to_one")
assert len(covered)==len(manifest)*6 and not covered.duplicated(["model","trip"]).any()
assert (covered.points==covered.points_expected).all() and (covered.fold==covered.validation_fold).all()
cv = trips.query("fold >= 0").groupby(["model","fold"]).decoded_top1.mean()
cv.to_csv(out/"fold_top1.csv")
dev = scores.query("split == 'student_test'").set_index("model")
def macro_f1(frame):
    errors=frame.fp+frame.fn
    return (2*frame.tp/(2*frame.tp+errors)+2*frame.tn/(2*frame.tn+errors))/2

fold_counts=trips.query("fold>=0").groupby(["model","fold"])[["tp","tn","fp","fn"]].sum()
cv_f1=macro_f1(fold_counts).groupby("model").mean()
dev["macro_f1"]=macro_f1(dev)
dev["cv_macro_f1"]=cv_f1
dev["fit_s"]=trips.query("fold==-1").groupby("model").fit_s.first()
dev=dev.join(pd.read_csv(out/"ordered_route_scores.csv").groupby("model")[["raw_route_f1","decoded_route_f1"]].mean())
dev=dev.join(pd.read_csv(out/"fit_diagnostics.csv").set_index("model"))
dev.to_csv(out/"development_comparison.csv")
fig,axes=plt.subplots(1,3,figsize=(14,5),layout="constrained")
dev.cv_macro_f1.plot.bar(ax=axes[0],title="Grouped CV · macro-F1")
dev[["top1","macro_f1"]].plot.bar(ax=axes[1],title="Student-test · candidate scoring")
dev.loss.dropna().plot.bar(ax=axes[2],title="Student-test · log loss")
fig.savefig(out/"score_comparison.png",dpi=150)
plt.close(fig)
history=pd.read_csv(out/"mlp_epoch_diagnostic.csv")
ax=history.plot(x="epoch",y=["training_loss","validation_loss"],ylabel="Binary log loss",title="MLP epoch diagnostic · fold 0")
ax.figure.tight_layout()
ax.figure.savefig(out/"mlp_learning_curve.png",dpi=150)
plt.close(ax.figure)
fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
for ax, (name,row) in zip(axes.flat, dev.iterrows()):
    matrix = row[["tn","fp","fn","tp"]].to_numpy(dtype=int).reshape(2,2)
    ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{matrix[i,j]:,}",ha="center",va="center",
                    color="white" if matrix[i,j] > matrix[i].sum()/2 else "black")
    ax.set(title=name,xlabel="Predicted class",ylabel="Proxy class",xticks=[0,1],yticks=[0,1])
fig.suptitle("Student-test candidate confusion counts")
fig.savefig(out/"confusion_matrices.png",dpi=150)
plt.close(fig)

def pct(x):
    return f"{100*x:.2f}%"

alignment = pd.read_csv(root / "results/ordered_pretraining/alignment_audit.csv")
alignment_summary = alignment.groupby("partition")[["points", "ambiguous", "clear", "miss_clear", "over_50m"]].sum()
train_alignment = alignment_summary.loc["train"]
student_alignment = alignment_summary.loc["student_test"]
worst = alignment.query("partition == 'train'").sort_values("over_50m", ascending=False).head(2)
worst_text = "; ".join(
    f"{row.trip_id} ({int(row.over_50m):,} points >50 m; median residual {row.median_m:.1f} m)"
    for row in worst.itertuples()
)

lines=["# Measured model comparison", "", "## Scope and label-quality limitation", "",
       "The figures below are **legacy nearest-route proxy scores**, not accuracy against independently observed per-point ground truth. "
       "They were produced before the ordered-label audit and must not be interpreted as definitive map-matching accuracy. "
       "The ordered labels are also inferred from the supplied route, so they remain a useful training/evaluation proxy rather than a direct measurement of the vehicle's true position.", "",
       "The ordered alignment audit marks labels uncertain when the best alternative route arc changes the whole-trip cost by at most one unit, or when the assigned route arc is more than 50 m away. "
       f"It found {int(train_alignment.ambiguous):,} / {int(train_alignment.points):,} training points ({pct(train_alignment.ambiguous/train_alignment.points)}) and "
       f"{int(student_alignment.ambiguous):,} / {int(student_alignment.points):,} student-test points ({pct(student_alignment.ambiguous/student_alignment.points)}) uncertain. "
       f"Of accepted labels, {int(train_alignment.miss_clear):,} / {int(train_alignment.clear):,} training points ({pct(train_alignment.miss_clear/train_alignment.clear)}) and "
       f"{int(student_alignment.miss_clear):,} / {int(student_alignment.clear):,} student-test points ({pct(student_alignment.miss_clear/student_alignment.clear)}) have no matching arc in the generated candidate set; no scorer can recover those points under the fixed shortlist. "
       "The full aggregate is in [the alignment-limitation summary](../results/ordered_pretraining/alignment_limitations_summary.csv) and per-trip evidence is in [the alignment audit](../results/ordered_pretraining/alignment_audit.csv).", "",
       f"Weak GPS-to-route agreement is concentrated rather than uniform: {worst_text}. Their inferred point labels should be excluded from fitting and from any claim about model quality. "
       "Candidate misses must be counted as end-to-end failures, while uncertainty must be reported over all observations rather than dropped from the denominator.", "",
       "Scores are macro-averages over held-out trips. Top-1 uses the spatial proxy. Raw and decoded scores use the same candidates. "
       "Until this comparison is rerun on accepted ordered-label hits, its rankings are exploratory only.", "",
       "| Model | CV decoded | CV fold range | Student raw | Student decoded | Student decoded continuity | Student F1 (pooled) |",
       "|---|---:|---:|---:|---:|---:|---:|"]
for name,row in dev.iterrows():
    mean = scores.query("split == 'grouped_cv' and model == @name").decoded_top1.iloc[0]
    lines.append(f"| {name} | {pct(mean)} | {pct(cv[name].min())}–{pct(cv[name].max())} | {pct(row.top1)} | {pct(row.decoded_top1)} | {pct(row.decoded_continuity)} | {row.f1:.3f} |")
eda = pd.read_csv(out/"feature_medians_by_trip.csv").query("partition == 'train'")
stats = eda.groupby("positive")[["distance_m","distance_above_best_m","heading_cos"]].median()
lines += ["","## Training feature distributions","",
          "Each entry is the median of the per-trip class medians. Units are metres for distances. Heading cosine includes the neutral placeholder for unknown headings.","",
          "| Proxy match | Distance | Distance above best | Heading cosine |","|---|---:|---:|---:|"]
for matched, row in stats.iterrows():
    lines.append(f"| {matched} | {row.distance_m:.2f} | {row.distance_above_best_m:.2f} | {row.heading_cos:.3f} |")
lines += ["","[Per-trip timings and raw metrics](../results/model_comparison/trip_metrics.csv) · [Aggregate metrics](../results/model_comparison/comparison.csv) · [Confusion matrices](../results/model_comparison/confusion_matrices.png)"]
objectives={"Logistic":"L2-regularised binary log loss, C=1, up to 300 iterations.",
            "KNN":"Inverse-distance class vote over 11 scaled neighbours. Fitting builds the neighbour index.",
            "Forest":"Mean positive class probability across 60 bootstrap trees using Gini impurity and a minimum leaf size of five.",
            "Boosting":"Binary log loss, 100 histogram boosting iterations, learning rate 0.1 and up to 31 leaves per tree.",
            "MLP":"Binary cross-entropy with Adam, one 24-unit ReLU layer, initial learning rate 0.001 and a 120-epoch limit."}
for name, objective in objectives.items():
    row=dev.loc[name]
    lines += ["",f"## {name}","",objective,"",
        f"Mean CV macro-F1 was {row.cv_macro_f1:.3f}. The final fit scored {row.training_macro_f1:.3f} on its own sampled candidate rows. Their difference of {row.training_macro_f1-row.cv_macro_f1:+.3f} is a generalisation diagnostic. Training sampling excludes candidate misses, so the gap alone cannot isolate overfitting.",
        "",f"Student-test macro-F1 was {row.macro_f1:.3f}, raw top-1 {pct(row.top1)}, and mean trip log loss {row.loss:.3f}. Ordered-route F1 changed from {row.raw_route_f1:.3f} to {row.decoded_route_f1:.3f} after decoding. Final fitting took {row.fit_s:.2f} s and scoring plus decoding all ten trips took {row.inference_s:.2f} s."]
lines += ["","Ordered-route F1 is twice the length of the longest common arc subsequence divided by the sum of the two sequence lengths, after collapsing consecutive repetitions. It preserves traversal order and uses the supplied trip route directly. Scores average trips equally. Macro-F1 averages the two candidate classes. CV averages fold-level macro-F1 equally.",
          "","[Score comparisons](../results/model_comparison/score_comparison.png) · [MLP learning curve](../results/model_comparison/mlp_learning_curve.png) · [Consolidated metrics](../results/model_comparison/development_comparison.csv)"]
winner=dev.decoded_route_f1.idxmax()
point_winner=dev.decoded_top1.idxmax()
lines += ["","## How to read the differences","",
          f"{winner} led student-test ordered-route F1 at {dev.loc[winner,'decoded_route_f1']:.3f}. {point_winner} led decoded proxy top-1 at {pct(dev.loc[point_winner,'decoded_top1'])}. The first metric compares the traversed arc sequence with the supplied route, while the second gives each labelled observation a vote within its trip. Long stationary stretches and frequent observations on one arc can make these metrics rank models differently.",
          "",f"The KNN training macro-F1 was {dev.loc['KNN','training_macro_f1']:.3f}. Distance-weighted voting can exactly reproduce many training examples because their distance to themselves is zero. Use its held-out fold scores to assess generalisation.",
          "",f"In the separate 120-epoch MLP diagnostic, training log loss moved from {history.training_loss.iloc[0]:.3f} to {history.training_loss.iloc[-1]:.3f}, and validation log loss from {history.validation_loss.iloc[0]:.3f} to {history.validation_loss.iloc[-1]:.3f}. The lowest observed validation loss occurred at epoch {history.loc[history.validation_loss.idxmin(),'epoch']:.0f}. Timings reflect this machine and concurrent diagnostic work."]
(root/"docs/MODEL_COMPARISON_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
