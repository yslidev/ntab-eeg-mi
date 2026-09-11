# Left fist vs right fist from scalp EEG

A motor-imagery decoder for the PhysioNet EEG Motor Movement/Imagery Database
(EEGMMIDB), built for the NT@B software-division recruitment project.

The interesting part of this repository is not the classifier. It is the gap
between the first accuracy figure you can get out of this dataset and the one
that survives contact with a held-out person. Most of the code exists to
measure that gap and to rule out the boring explanations for it.

**Short version of the result.** See `RESULTS.md` for every number, and the
summary table in [Results](#results) below.

---

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Data is fetched from PhysioNet. The S3 mirror is roughly eighty times faster
than the main site, so `scripts/fetch_one.sh` pulls from there into the same
cache directory MNE would use (`~/mne_data/MNE-eegbci-data/...`).

```bash
bash -c 'xargs -n1 -P 12 ./scripts/fetch_one.sh < scripts/filelist.txt'     # task runs
bash -c 'xargs -n1 -P 12 ./scripts/fetch_one.sh < scripts/filelist_r2.txt'  # eyes-closed baseline
```

Roughly 2 GB, a couple of minutes.

## Predicting

```bash
python predict.py ~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S042/S042R04.edf
```

It takes any number of raw EDF paths, prints per-file accuracy when the file
carries `T1`/`T2` annotations, and writes per-trial predictions with `--out`.
Label 0 is the left fist, label 1 is the right fist.

To rebuild the model from scratch:

```bash
python scripts/02_build_cache.py      # epoch cache, ~2 min
python scripts/08_train_final.py      # writes model/mi_lr_model.joblib
```

## Reproducing the analysis

Scripts are numbered in dependency order.

| script | what it does |
|---|---|
| `01_audit.py` | structural audit of all 763 recordings |
| `02_build_cache.py` | wide broadband epoch cache |
| `03_lateralisation.py` | establishes which annotation is which hand |
| `04_model_selection.py` | feature/classifier/band/window sweep, DEV subjects only |
| `05_main_regimes.py` | the headline table, EVAL subjects |
| `06_controls.py` | placebo windows, electrode lesions, subject identity, transfer |
| `07_deep.py` | EEGNet, including its overfitting behaviour |
| `08_train_final.py` | trains the shipped model |
| `09_figures.py` | figures from the result CSVs |
| `10_who_is_decodable.py` | resting-state predictors of per-subject accuracy |
| `11_holdout_check.py` | runs `predict.py` on the 12 untouched subjects |
| `12_interpretation.py` | time-resolved decoding and scalp weight maps |
| `13_make_results_md.py` | regenerates `RESULTS.md` |
