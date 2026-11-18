"""Shared trip-grouped comparison. Run from any directory with --data-root."""
import argparse
import os
from pathlib import Path
from time import perf_counter

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data.prepare_candidates import FEATURES, ROOT

STORE = ROOT / "results/ordered_pretraining"
OUT = ROOT / "results/ordered_comparison"
INPUTS = FEATURES + ["distance_above_best_m"]
SEED = 125


def models():
    return {
        "Nearest": None,
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(11, weights="distance", algorithm="kd_tree", n_jobs=4)),
        "Forest": RandomForestClassifier(n_estimators=60, min_samples_leaf=5, n_jobs=4, random_state=SEED),
        "Boosting": HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=SEED),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(24,), max_iter=120, random_state=SEED)),
    }


def read_trip(part, trip):
    f = pd.read_pickle(STORE/"features"/part/(trip+".pkl"))
    f["distance_above_best_m"] = f.distance_m - f.groupby("original_row_index").distance_m.transform("min")
    labels = pd.read_pickle(STORE/"targets"/part/(trip+".pkl"))
    return f.merge(labels, on="original_row_index", validate="many_to_one")


def decode(frame, arcs, p):
    # Pad observation groups to a rectangular lattice. Each column is a local candidate.
    group = frame.groupby("original_row_index", sort=False)
    slot = group.cumcount().to_numpy()
    obs = group.ngroup().to_numpy()
    shape = (obs[-1]+1, slot.max()+1)
    ids = np.full(shape, -1)
    ids[obs, slot] = frame.arc_id
    emission = np.full(shape, -np.inf)
    emission[obs, slot] = np.log(np.clip(p, 1e-8, 1))
    raw = ids[np.arange(len(ids)), emission.argmax(axis=1)]
    rows = frame.iloc[np.flatnonzero(slot == 0)]
    pieces = np.split(np.arange(len(ids)), np.flatnonzero(np.diff(rows.segment_id))+1)
    path = raw.copy()
    for ix in pieces:
        scores = emission[ix[0]].copy()
        back = []
        for i in ix[1:]:
            prev, curr = ids[i-1,:,None], ids[i,None,:]
            connected = (prev == curr) | (arcs[prev,1] == arcs[curr,0])
            options = scores[:,None] - 2.0 * (~connected)
            parents = options.argmax(axis=0)
            scores = options.max(axis=0) + emission[i]
            back.append(parents)
        state = scores.argmax()
        for offset in range(len(ix)-1,-1,-1):
            path[ix[offset]] = ids[ix[offset],state]
            if offset:
                state = back[offset-1][state]
    return rows, raw, path


def evaluate(frame, arcs, model):
    start = perf_counter()
    if model is None:
        prob = np.exp(-frame.distance_m.to_numpy()/20)
    else:
        prob = model.predict_proba(frame[INPUTS])[:,1]
    rows, raw, path = decode(frame, arcs, prob)
    seconds = perf_counter()-start
    clear = ~rows.ambiguous.to_numpy()
    target = rows.proxy_arc_id.to_numpy()
    keep = ~frame.ambiguous.to_numpy()
    y = (frame.arc_id == frame.proxy_arc_id).to_numpy()[keep]
    tn, fp, fn, tp = confusion_matrix(y, prob[keep]>=.5, labels=[False,True]).ravel()
    linked = np.diff(rows.segment_id) == 0
    def continuous(sequence):
        return ((sequence[:-1]==sequence[1:]) | (arcs[sequence[:-1],1]==arcs[sequence[1:],0]))[linked].mean()
    return dict(points=len(rows), clear=int(clear.sum()), miss=int((clear & ~rows.in_candidates).sum()),
        top1=(raw[clear]==target[clear]).mean(), decoded_top1=(path[clear]==target[clear]).mean(),
        continuity=continuous(raw), decoded_continuity=continuous(path), tn=tn, fp=fp, fn=fn, tp=tp,
        loss=log_loss(y,prob[keep],labels=[False,True]) if model is not None else np.nan,
        inference_s=seconds)


def main(data_root, selected_models=None):
    OUT.mkdir(exist_ok=True)
    manifest = pd.read_csv(STORE/"manifest.csv",dtype={"trip_id":str})
    fits, scores, eda = {}, [], []
    for row in manifest.itertuples():
        f = read_trip(row.partition,row.trip_id)
        eligible = f.loc[~f.ambiguous & f.in_candidates]
        if row.partition=="train":
            ids = eligible.original_row_index.unique()
            chosen = np.random.default_rng(SEED+int(row.trip_id)).choice(ids,min(128,len(ids)),replace=False)
            fits[row.trip_id] = eligible[eligible.original_row_index.isin(chosen)]
        for positive, subset in eligible.groupby(eligible.arc_id==eligible.proxy_arc_id):
            eda.append(dict(partition=row.partition,trip_id=row.trip_id,positive=positive,n=len(subset),
                **subset[INPUTS].median().to_dict()))
    pd.DataFrame(eda).to_csv(OUT/"feature_medians_by_trip.csv",index=False)
    for fold in [0,1,2,3,4,-1]:
        fit = pd.concat([f for trip,f in fits.items() if trip not in set(manifest.loc[manifest.validation_fold==fold,"trip_id"])])
        fitted = models()
        if selected_models is not None:
            fitted = {name: model for name, model in fitted.items()
                      if name in selected_models or name == "Nearest"}
        timing = {}
        for name, model in fitted.items():
            start = perf_counter()
            if model is not None:
                model.fit(fit[INPUTS],fit.arc_id==fit.proxy_arc_id)
            timing[name] = perf_counter()-start
        for row in manifest[manifest.validation_fold==fold].itertuples():
            f = read_trip(row.partition,row.trip_id)
            arcs = np.loadtxt(data_root/row.partition/row.trip_id/(row.trip_id+".arcs"),dtype=int)
            for name, model in fitted.items():
                scores.append(dict(model=name,fold=fold,trip=row.trip_id,fit_rows=len(fit),fit_s=timing[name],
                    **evaluate(f,arcs,model)))
        pd.DataFrame(scores).to_csv(OUT/"trip_metrics.csv",index=False)
        print("Completed fold",fold,flush=True)
    df=pd.DataFrame(scores)
    df["split"]=np.where(df.fold==-1,"student_test","grouped_cv")
    summary=df.groupby(["split","model"])[["top1","decoded_top1","continuity","decoded_continuity","loss"]].mean()
    totals=df.groupby(["split","model"])[["tn","fp","fn","tp","clear","miss","inference_s"]].sum()
    summary=summary.join(totals)
    summary["accuracy"]=(summary.tp+summary.tn)/(summary.tp+summary.tn+summary.fp+summary.fn)
    summary["precision"]=summary.tp/(summary.tp+summary.fp)
    summary["recall"]=summary.tp/(summary.tp+summary.fn)
    summary["f1"]=2*summary.tp/(2*summary.tp+summary.fp+summary.fn)
    summary["candidate_miss_rate"]=summary.miss/summary.clear
    summary.to_csv(OUT/"comparison.csv")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--model",choices=["Logistic","KNN","Forest","Boosting","MLP"],
                        help="Run one assigned model plus the geometric baseline.")
    args=parser.parse_args()
    main(args.data_root, {args.model} if args.model else None)
