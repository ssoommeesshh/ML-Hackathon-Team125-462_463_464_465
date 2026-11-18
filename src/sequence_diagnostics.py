"""Replay the frozen development configurations for ordered-route diagnostics."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from evaluate_families import INPUTS, OUT, STORE, SEED, models, read_trip, decode
from pathlib import Path


def route_f1(pred, ref):
    # Bit-parallel longest common subsequence on collapsed adjacent arc runs.
    pred = pred[np.r_[True, pred[1:] != pred[:-1]]]
    ref = ref[np.r_[True, ref[1:] != ref[:-1]]]
    masks = {}
    for i, arc in enumerate(ref):
        masks[arc] = masks.get(arc, 0) | (1 << i)
    state = 0
    for arc in pred:
        union = state | masks.get(arc, 0)
        state = union & ~(union - ((state << 1) | 1))
    return 2 * state.bit_count() / (len(pred)+len(ref))


def main(data):
    manifest = pd.read_csv(STORE/"manifest.csv",dtype={"trip_id":str})
    train = []
    for row in manifest.query("partition=='train'").itertuples():
        frame = read_trip("train",row.trip_id)
        eligible = frame[~frame.ambiguous & frame.in_candidates]
        ids = eligible.original_row_index.unique()
        chosen = np.random.default_rng(SEED+int(row.trip_id)).choice(ids,min(128,len(ids)),replace=False)
        train.append(eligible[eligible.original_row_index.isin(chosen)].assign(fold=row.validation_fold))
    train = pd.concat(train)
    fitted = models()
    quality = []
    y = train.arc_id==train.proxy_arc_id
    for name,model in fitted.items():
        if model is not None:
            model.fit(train[INPUTS],y)
            quality.append(dict(model=name,training_macro_f1=f1_score(y,model.predict(train[INPUTS]),average="macro")))
    pd.DataFrame(quality).to_csv(OUT/"fit_diagnostics.csv",index=False)
    results = []
    for trip in manifest.query("partition=='student_test'").trip_id:
        frame = read_trip("student_test",trip)
        source = data/"student_test"/trip/trip
        arcs = np.loadtxt(source.with_suffix(".arcs"),dtype=int)
        ref = np.loadtxt(source.with_suffix(".route"),dtype=int)
        for name,model in fitted.items():
            p = model.predict_proba(frame[INPUTS])[:,1] if model is not None else np.exp(-frame.distance_m/20).to_numpy()
            _,raw,path = decode(frame,arcs,p)
            results.append(dict(model=name,trip_id=trip,raw_route_f1=route_f1(raw,ref),decoded_route_f1=route_f1(path,ref)))
    pd.DataFrame(results).to_csv(OUT/"ordered_route_scores.csv",index=False)
    # Separate epoch diagnostic with a fixed training-trip fold for validation.
    tr = train[train.fold!=0]
    va = train[train.fold==0]
    scaler, mlp = models()["MLP"].steps
    xt = scaler[1].fit_transform(tr[INPUTS])
    xv = scaler[1].transform(va[INPUTS])
    yt, yv = tr.arc_id==tr.proxy_arc_id, va.arc_id==va.proxy_arc_id
    history=[]
    for epoch in range(1,121):
        mlp[1].partial_fit(xt,yt,classes=[False,True])
        history.append((epoch,log_loss(yt,mlp[1].predict_proba(xt)),log_loss(yv,mlp[1].predict_proba(xv))))
    pd.DataFrame(history,columns=["epoch","training_loss","validation_loss"]).to_csv(OUT/"mlp_epoch_diagnostic.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True)
    main(parser.parse_args().data_root)
