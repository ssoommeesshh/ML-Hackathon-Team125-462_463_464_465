"""Check the regenerated labels and compare the frozen inputs trip by trip."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_candidates import ROOT


def main(data_root):
    base = ROOT / "results"
    old, new = base/"pretraining", base/"ordered_pretraining"
    def manifest(store):
        return pd.read_csv(store/"manifest.csv", dtype={"trip_id": str}).set_index(["partition","trip_id"])
    pd.testing.assert_frame_equal(manifest(old)[["validation_fold","points","candidates"]],
                                  manifest(new)[["validation_fold","points","candidates"]])
    rows = []
    examples = ["00000001","00000076","00000054"]
    fig, axes = plt.subplots(2,3,figsize=(13,6),layout="constrained")
    for part,trip in manifest(new).index:
        name = Path(part)/(trip+".pkl")
        f = pd.read_pickle(new/"features"/name)
        pd.testing.assert_frame_equal(f,pd.read_pickle(old/"features"/name))
        labels = pd.read_pickle(new/"targets"/name)
        source = data_root/part/trip/trip
        route = np.loadtxt(source.with_suffix(".route"),dtype=int,ndmin=1)
        index = labels.route_index.to_numpy()
        assert np.array_equal(route[index],labels.proxy_arc_id)
        assert np.all(np.diff(index)>=0)
        assert np.array_equal(f.original_row_index.unique(),labels.original_row_index)
        clear = ~labels.ambiguous
        d = labels.reference_distance_m
        rows.append(dict(partition=part,trip_id=trip,points=len(labels),clear=int(clear.sum()),
            ambiguous=int(labels.ambiguous.sum()),margin_le_one=int((labels.alternative_margin<=1).sum()),
            over_50m=int((d>50).sum()),miss_clear=int((clear & ~labels.in_candidates).sum()),
            changed=int(labels.changed_from_nearest.sum()),median_m=d.median(),p95_m=d.quantile(.95),
            first_position=index[0],last_position=index[-1],route_entries=len(route),
            skipped_occurrences=len(route)-len(np.unique(index)),backward_steps=int((np.diff(index)<0).sum())))
        if trip in examples:
            j = examples.index(trip)
            gps = np.loadtxt(source.with_suffix(".track"))
            nodes = np.loadtxt(source.with_suffix(".nodes"))
            arcs = np.loadtxt(source.with_suffix(".arcs"),dtype=int)
            from matplotlib.collections import LineCollection
            ax=axes[0,j]
            ax.add_collection(LineCollection(nodes[arcs[route]],colors="black",linewidths=1))
            ax.scatter(*gps[:,:2].T,s=2,c=np.where(clear,"steelblue","darkorange"))
            ax.autoscale()
            ax.set_aspect(1/np.cos(np.radians(gps[:,1].mean())))
            ax.set_title(f"{part} / {trip}")
            axes[1,j].scatter(labels.original_row_index,index,s=2,c=np.where(clear,"steelblue","darkorange"))
            axes[1,j].set(xlabel="GPS row",ylabel="Reference route position")
    fig.suptitle("Reference arcs (black), accepted GPS labels (blue), uncertain labels (orange)")
    fig.savefig(new/"alignment_examples.png",dpi=120)
    report=pd.DataFrame(rows)
    report.to_csv(new/"alignment_audit.csv",index=False)
    print(report.groupby("partition")[["points","clear","ambiguous","miss_clear","changed","over_50m","backward_steps"]].sum().to_string())
    print(report[report.trip_id.isin(examples)].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True)
    main(parser.parse_args().data_root)
