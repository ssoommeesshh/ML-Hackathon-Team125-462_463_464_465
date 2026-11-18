"""
Bayesian Hyperparameter Optimization and HMM Route Reconstruction for HistGradientBoosting.
Complies with PathSnap Hackathon Guidelines (Section 4: Bayesian Tuning & Section 5: HMM/Viterbi).
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, log_loss

from data.prepare_candidates import FEATURES, ROOT

STORE = ROOT / "results/ordered_pretraining"
OUT_TUNE = ROOT / "results/bayesian_tuning"
INPUTS = FEATURES + ["distance_above_best_m"]
SEED = 125

optuna.logging.set_verbosity(optuna.logging.WARNING)


def read_trip(part, trip):
    f = pd.read_pickle(STORE / "features" / part / (trip + ".pkl"))
    f["distance_above_best_m"] = f.distance_m - f.groupby("original_row_index").distance_m.transform("min")
    labels = pd.read_pickle(STORE / "targets" / part / (trip + ".pkl"))
    return f.merge(labels, on="original_row_index", validate="many_to_one")


def decode_viterbi(frame, arcs, p):
    """
    Viterbi Trellis Route Reconstruction:
    - Hidden States: Candidate road arcs at each observation.
    - Emissions: log posterior probabilities ln(p) from classifier.
    - Transitions: Connected graph adjacency (0.0 penalty) vs disconnected (-2.0 penalty).
    - Disconnection handling: Segment resets and backtracking.
    """
    group = frame.groupby("original_row_index", sort=False)
    slot = group.cumcount().to_numpy()
    obs = group.ngroup().to_numpy()
    shape = (obs[-1] + 1, slot.max() + 1)
    ids = np.full(shape, -1)
    ids[obs, slot] = frame.arc_id
    emission = np.full(shape, -np.inf)
    emission[obs, slot] = np.log(np.clip(p, 1e-8, 1))
    raw = ids[np.arange(len(ids)), emission.argmax(axis=1)]
    rows = frame.iloc[np.flatnonzero(slot == 0)]
    pieces = np.split(np.arange(len(ids)), np.flatnonzero(np.diff(rows.segment_id)) + 1)
    path = raw.copy()
    for ix in pieces:
        scores = emission[ix[0]].copy()
        back = []
        for i in ix[1:]:
            prev, curr = ids[i - 1, :, None], ids[i, None, :]
            connected = (prev == curr) | (arcs[prev, 1] == arcs[curr, 0])
            options = scores[:, None] - 2.0 * (~connected)
            parents = options.argmax(axis=0)
            scores = options.max(axis=0) + emission[i]
            back.append(parents)
        state = scores.argmax()
        for offset in range(len(ix) - 1, -1, -1):
            path[ix[offset]] = ids[ix[offset], state]
            if offset:
                state = back[offset - 1][state]
    return rows, raw, path


def evaluate_trip(frame, arcs, model):
    start = perf_counter()
    prob = model.predict_proba(frame[INPUTS])[:, 1]
    rows, raw, path = decode_viterbi(frame, arcs, prob)
    seconds = perf_counter() - start
    clear = ~rows.ambiguous.to_numpy()
    target = rows.proxy_arc_id.to_numpy()
    keep = ~frame.ambiguous.to_numpy()
    y = (frame.arc_id == frame.proxy_arc_id).to_numpy()[keep]
    tn, fp, fn, tp = confusion_matrix(y, prob[keep] >= 0.5, labels=[False, True]).ravel()
    linked = np.diff(rows.segment_id) == 0

    def continuous(sequence):
        return ((sequence[:-1] == sequence[1:]) | (arcs[sequence[:-1], 1] == arcs[sequence[1:], 0]))[linked].mean()

    return dict(
        points=len(rows),
        clear=int(clear.sum()),
        miss=int((clear & ~rows.in_candidates).sum()),
        top1=(raw[clear] == target[clear]).mean(),
        decoded_top1=(path[clear] == target[clear]).mean(),
        continuity=continuous(raw),
        decoded_continuity=continuous(path),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        loss=log_loss(y, prob[keep], labels=[False, True]),
        inference_s=seconds,
    )


def run_bayesian_optimization(data_root, n_trials=30):
    OUT_TUNE.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(STORE / "manifest.csv", dtype={"trip_id": str})

    # Prepare training samples (128 observations per train trip, deterministic)
    fits = {}
    train_trips_data = {}
    train_manifest = manifest[manifest.partition == "train"]

    print("Loading pretraining candidate dataset...", flush=True)
    for row in train_manifest.itertuples():
        f = read_trip(row.partition, row.trip_id)
        train_trips_data[row.trip_id] = f
        eligible = f.loc[~f.ambiguous & f.in_candidates]
        ids = eligible.original_row_index.unique()
        chosen = np.random.default_rng(SEED + int(row.trip_id)).choice(ids, min(128, len(ids)), replace=False)
        fits[row.trip_id] = eligible[eligible.original_row_index.isin(chosen)]

    # Pre-cache fold training sets
    fold_train_fits = {}
    for fold in range(5):
        val_trips = set(manifest.loc[manifest.validation_fold == fold, "trip_id"])
        fold_train_fits[fold] = pd.concat([f for trip, f in fits.items() if trip not in val_trips])

    # Objective: Maximize mean CV Macro F1 across 5 folds
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "max_iter": trial.suggest_int("max_iter", 60, 220, step=20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 45),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "random_state": SEED,
            "early_stopping": False,
        }

        f1_scores = []
        for fold in range(5):
            fit = fold_train_fits[fold]
            clf = HistGradientBoostingClassifier(**params)
            clf.fit(fit[INPUTS], fit.arc_id == fit.proxy_arc_id)

            # Evaluate on validation fold trips
            val_rows = manifest[manifest.validation_fold == fold]
            tp_sum, fp_sum, fn_sum = 0, 0, 0
            for row in val_rows.itertuples():
                f = train_trips_data[row.trip_id]
                keep = ~f.ambiguous.to_numpy()
                y = (f.arc_id == f.proxy_arc_id).to_numpy()[keep]
                prob = clf.predict_proba(f[INPUTS])[:, 1]
                pred = prob[keep] >= 0.5
                tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

            fold_f1 = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum) if (2 * tp_sum + fp_sum + fn_sum) > 0 else 0
            f1_scores.append(fold_f1)

        return np.mean(f1_scores)

    print(f"Starting Bayesian Optimization with {n_trials} trials (Tree-structured Parzen Estimator)...", flush=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"\nOptimization Finished! Best Trial: #{study.best_trial.number}", flush=True)
    print(f"Best CV Macro F1: {study.best_value:.4f}", flush=True)
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_TUNE / "tuning_trials.csv", index=False)

    best_params = {**study.best_params, "random_state": SEED, "early_stopping": False}
    with open(OUT_TUNE / "best_hyperparameters.json", "w") as fp:
        json.dump(best_params, fp, indent=2)

    # Now evaluate Untuned vs Tuned on both CV and Student-Test
    untuned_params = {"max_iter": 100, "early_stopping": False, "random_state": SEED}
    models_to_compare = {
        "Untuned_HistGB": HistGradientBoostingClassifier(**untuned_params),
        "Tuned_HistGB": HistGradientBoostingClassifier(**best_params),
    }

    comparison_records = []
    print("\nRunning full evaluation comparing Untuned vs Tuned models across 5 folds and Student Test...", flush=True)

    for fold in [0, 1, 2, 3, 4, -1]:
        val_trips = set(manifest.loc[manifest.validation_fold == fold, "trip_id"])
        fit = pd.concat([f for trip, f in fits.items() if trip not in val_trips])

        fold_models = {
            "Untuned_HistGB": HistGradientBoostingClassifier(**untuned_params),
            "Tuned_HistGB": HistGradientBoostingClassifier(**best_params),
        }
        for name, clf in fold_models.items():
            clf.fit(fit[INPUTS], fit.arc_id == fit.proxy_arc_id)

        target_manifest = manifest[manifest.validation_fold == fold]
        for row in target_manifest.itertuples():
            f = read_trip(row.partition, row.trip_id)
            arcs = np.loadtxt(data_root / row.partition / row.trip_id / (row.trip_id + ".arcs"), dtype=int)
            for name, clf in fold_models.items():
                res = evaluate_trip(f, arcs, clf)
                comparison_records.append(dict(model=name, fold=fold, trip=row.trip_id, **res))

    df_comp = pd.DataFrame(comparison_records)
    df_comp["split"] = np.where(df_comp.fold == -1, "student_test", "grouped_cv")

    # Grouped summary
    summary = df_comp.groupby(["split", "model"])[
        ["top1", "decoded_top1", "continuity", "decoded_continuity", "loss"]
    ].mean()
    totals = df_comp.groupby(["split", "model"])[["tn", "fp", "fn", "tp", "clear", "miss", "inference_s"]].sum()
    summary = summary.join(totals)
    summary["accuracy"] = (summary.tp + summary.tn) / (summary.tp + summary.tn + summary.fp + summary.fn)
    summary["precision"] = summary.tp / (summary.tp + summary.fp)
    summary["recall"] = summary.tp / (summary.tp + summary.fn)
    summary["f1"] = 2 * summary.tp / (2 * summary.tp + summary.fp + summary.fn)
    summary["candidate_miss_rate"] = summary.miss / summary.clear

    summary.to_csv(OUT_TUNE / "untuned_vs_tuned_comparison.csv")
    df_comp.to_csv(OUT_TUNE / "untuned_vs_tuned_trip_metrics.csv", index=False)

    print("\n=== Untuned vs Tuned Model Summary ===")
    print(summary[["top1", "decoded_top1", "decoded_continuity", "loss", "accuracy", "precision", "recall", "f1", "inference_s"]].to_string())

    return study.best_params, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "Student folder")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    run_bayesian_optimization(args.data_root, args.trials)
