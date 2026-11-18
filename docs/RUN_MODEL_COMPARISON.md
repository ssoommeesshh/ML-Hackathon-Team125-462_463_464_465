# Setup and run the shared model experiments

This guide prepares the chronological GPS-to-route proxy and lets each model owner run one scorer on the shared candidates, labels, features and five trip folds. Run commands from the repository root. Keep the supplied raw dataset available in the adjacent `Student folder`.

## Environment

~~~powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

If PowerShell blocks activation, call `.venv\Scripts\python.exe` directly for the commands below.

## Generate the shared data once

~~~powershell
python src/data/prepare_candidates.py --data-root "../Student folder"
python src/data/audit_ordered.py --data-root "../Student folder"
~~~

Candidate features are generated from the cleaned GPS table. Labels use monotonic positions in each trip's ordered route. Review `results/ordered_pretraining/alignment_audit.csv` and `docs/ORDERED_ALIGNMENT.md` before interpreting scores. The output uses local per-trip pickle files and a manifest.

## Run the model assigned to you

~~~powershell
python src/evaluate_families.py --data-root "../Student folder" --model Logistic
python src/evaluate_families.py --data-root "../Student folder" --model KNN
python src/evaluate_families.py --data-root "../Student folder" --model Forest
python src/evaluate_families.py --data-root "../Student folder" --model Boosting
python src/evaluate_families.py --data-root "../Student folder" --model MLP
~~~

Each command also runs the nearest-segment reference baseline. It writes trip metrics incrementally to `results/ordered_comparison/trip_metrics.csv` and aggregates to `comparison.csv`. Run a single model command at a time because all owners write to the same result files. The default command without `--model` runs all five for an integrated comparison.

## Shared protocol

The five folds keep all points from each training trip together. Each fit uses up to 128 eligible observations sampled per training trip with a deterministic seed, keeping all candidate arcs for each sampled observation. Fold -1 is the ten student-test trips. Ambiguous labels are not used for candidate fitting. Accepted labels missing from the candidate shortlist count as misses in end-to-end coverage. Model outputs use one shared sequence decoder, which resets at cleaned segment boundaries.

The route file does not identify the arc at each GPS timestamp. Labels are inferred from GPS-to-route geometry and route order. Scores therefore measure agreement with that proxy. Report alignment uncertainty, candidate misses, and trip-level variation alongside model metrics.

## Outputs and reproduction

- `results/ordered_pretraining/features` and `targets` contain the shared inputs and inferred labels.
- `results/ordered_pretraining/manifest.csv` records folds and per-trip coverage.
- `results/ordered_pretraining/alignment_audit.csv` records distances, uncertainty, route progression, and misses.
- `results/ordered_comparison/trip_metrics.csv` records each model and trip as it completes.
- `results/ordered_comparison/comparison.csv` contains aggregate CV and student-test scores after all folds complete.

The package intentionally stops at the fixed-configuration comparison. Bayesian tuning and final model selection should use the training-trip folds and be documented as a later experiment.
