"""Measure ambiguity and direction information using training trips only."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
store = root/"results/pretraining"
out = root/"results/model_comparison"
out.mkdir(exist_ok=True)
trips = []
for path in sorted((store/"features/train").glob("*.pkl")):
    f = pd.read_pickle(path).sort_values(["original_row_index","distance_m","arc_id"])
    g = f.groupby("original_row_index",sort=False)
    first = g.head(1).set_index("original_row_index")
    second = f.loc[g.cumcount()==1].set_index("original_row_index")
    first["near_tie"] = second.distance_m.sub(first.distance_m).le(1)
    first["candidate_count"] = g.size()
    first["trip"] = path.stem
    t = pd.read_pickle(store/"targets/train"/path.name).set_index("original_row_index")
    trips.append(first.join(t))

frame = pd.concat(trips)
records = []
for (tied, known), chunk in frame.groupby(["near_tie","heading_known"]):
    clear = chunk[~chunk.ambiguous]
    records.append(dict(distance_tie_within_1m=tied,heading_known=known,
        observations=len(chunk),trips=chunk.trip.nunique(),median_candidate_count=chunk.candidate_count.median(),
        ambiguous_proxy_fraction=chunk.ambiguous.mean(),miss_fraction_among_clear=(~clear.in_candidates).mean(),
        nearest_top1_among_clear=np.mean(clear.arc_id==clear.proxy_arc_id)))
report=pd.DataFrame(records)
report.to_csv(out/"training_feature_strata.csv",index=False)
fig,ax=plt.subplots(figsize=(8,4), layout="constrained")
report.index=[f"{'Close tie' if tie else 'Clearer distance'}\n{'heading available' if known else 'heading unavailable'}"
              for tie, known in zip(report.distance_tie_within_1m, report.heading_known)]
report[["nearest_top1_among_clear","ambiguous_proxy_fraction"]].rename(columns={
    "nearest_top1_among_clear":"Nearest top-1 on clear proxies",
    "ambiguous_proxy_fraction":"Ambiguous proxy fraction"}).plot.bar(ax=ax,rot=0)
ax.set(ylabel="Fraction",ylim=(0,1),title="Training GPS observations grouped by road-distance tie and motion")
fig.savefig(out/"training_feature_strata.png",dpi=150)
print(report.to_string(index=False))
