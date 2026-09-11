#!/usr/bin/env python3
"""Predict left-fist vs right-fist for every motor-imagery trial in an EDF file.

Usage
-----
    python predict.py path/to/S042R04.edf
    python predict.py data/*.edf --out preds.csv

The model was trained on 93 other subjects. It is applied here with the same
unsupervised per-recording alignment used at training time: the mean spatial
covariance of the file's own trials is used to whiten it. That uses no labels,
so it is legal on unlabelled data, but it does mean a file's predictions depend
mildly on the other trials in the same file. scripts/06_controls.py measures
what that is worth.
"""
from __future__ import annotations
import argparse, pathlib, sys, warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODEL = ROOT / "model" / "mi_lr_model.joblib"


def load_model(path=DEFAULT_MODEL):
    import joblib
    if not pathlib.Path(path).is_file():
        raise SystemExit(
            f"No model at {path}. Build it with:\n"
            f"  python scripts/02_build_cache.py && python scripts/08_train_final.py")
    return joblib.load(path)


def epochs_from_edf(edf_path, meta):
    """Reproduce the training preprocessing exactly, on one raw EDF."""
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    if abs(raw.info["sfreq"] - meta["sfreq"]) > 1e-6:
        raw.resample(meta["sfreq"], verbose="ERROR")
    lo, hi = meta["band"]
    raw.filter(lo, hi, method="iir", iir_params=dict(order=4, ftype="butter"),
               verbose="ERROR")

    missing = [c for c in meta["ch_names"] if c not in raw.ch_names]
    if missing:
        raise SystemExit(f"{edf_path.name}: missing channels {missing[:5]}"
                         f"{'...' if len(missing) > 5 else ''}")
    raw.pick(meta["ch_names"])                       # also fixes channel ORDER

    tmin, tmax = meta["window"]
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    want = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    if len(want) == 2:
        ep = mne.Epochs(raw, events, want, tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose="ERROR", on_missing="ignore")
        code2lab = {want["T1"]: 0, want["T2"]: 1}
        y_true = np.array([code2lab[c] for c in ep.events[:, 2]])
    elif len(want) == 1:                              # only one class annotated
        ep = mne.Epochs(raw, events, want, tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose="ERROR", on_missing="ignore")
        only = list(want)[0]
        y_true = np.full(len(ep), 0 if only == "T1" else 1)
    else:                                             # no cue annotations at all
        ep = mne.make_fixed_length_epochs(raw, duration=tmax - tmin, overlap=0.0,
                                          preload=True, verbose="ERROR")
        y_true = None
    if len(ep) == 0:
        raise SystemExit(f"{edf_path.name}: no usable trials found")
    X = ep.get_data(copy=True)
    onsets = ep.events[:, 0] / raw.info["sfreq"]
    return X, y_true, onsets


def predict_file(edf_path, model=None):
    """Return (labels, probabilities, onsets, y_true_or_None)."""
    from models import AlignShrunk, precompute_cov
    m = model or load_model()
    edf_path = pathlib.Path(edf_path)
    X, y_true, onsets = epochs_from_edf(edf_path, m)
    alpha = m.get("align_alpha", 1.0)
    if alpha:
        X = AlignShrunk(alpha=alpha).transform(X.astype(np.float64))
    C = precompute_cov(X)
    pipe = m["pipeline"]
    pred = pipe.predict(C)
    proba = pipe.predict_proba(C)[:, 1] if hasattr(pipe, "predict_proba") else None
    return pred, proba, onsets, y_true


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edf", nargs="+", help="one or more raw .edf files")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=None, help="write a CSV of per-trial predictions")
    args = ap.parse_args()

    m = load_model(args.model)
    names = {0: "left_fist", 1: "right_fist"}
    recs, accs = [], []
    for f in args.edf:
        f = pathlib.Path(f)
        pred, proba, onsets, y_true = predict_file(f, m)
        line = f"{f.name}: {len(pred)} trials"
        if y_true is not None:
            acc = float((pred == y_true).mean())
            accs.append((acc, len(pred)))
            line += f"  accuracy {acc:.3f}  ({int((pred==y_true).sum())}/{len(pred)})"
        else:
            line += "  (no T1/T2 annotations; fixed-length windows, no ground truth)"
        print(line)
        for i, p in enumerate(pred):
            recs.append(dict(file=f.name, trial=i, onset_s=round(float(onsets[i]), 3),
                             pred=int(p), pred_label=names[int(p)],
                             p_right=None if proba is None else round(float(proba[i]), 4),
                             true=None if y_true is None else int(y_true[i])))
    if accs:
        tot = sum(n for _, n in accs)
        print(f"\npooled accuracy over {len(accs)} file(s), {tot} trials: "
              f"{sum(a*n for a, n in accs)/tot:.4f}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader(); w.writerows(recs)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
