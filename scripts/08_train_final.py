"""Train and serialise the model that predict.py ships with.

Trained on every usable subject (105) so that a new subject is genuinely
out-of-sample for whoever runs predict.py.
"""
import sys, pathlib, json, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, joblib
import cached, models, chosen, config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--paradigm", default="imagined", choices=["imagined", "executed", "both"])
ap.add_argument("--out", default="model/mi_lr_model.joblib")
args = ap.parse_args()

pars = ["imagined", "executed"] if args.paradigm == "both" else [args.paradigm]
Xs, ys, subs = [], [], []
for p in pars:
    es = cached.load_pool(p)
    X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
    Xa = chosen.featurize(X, es.subject)        # whiten per subject, then covariances
    Xs.append(Xa); ys.append(es.y); subs.append(es.subject)
X = np.concatenate(Xs); y = np.concatenate(ys); sub = np.concatenate(subs)
print(f"training on {len(y)} epochs from {len(np.unique(sub))} subjects, "
      f"paradigm={args.paradigm}")

pipe = chosen.make().fit(X, y)

out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(dict(
    pipeline=pipe, ch_names=chs, band=CFG.BAND, window=CFG.WINDOW,
    sfreq=160.0, align_alpha=chosen.ALIGN_ALPHA, model=chosen.NAME,
    n_components=chosen.N_COMPONENTS,
    shrinkage=chosen.SHRINKAGE,
    labels={0: "left_fist", 1: "right_fist"},
    paradigm=args.paradigm, n_train_subjects=int(len(np.unique(sub))),
    n_train_epochs=int(len(y)),
), out, compress=3)
print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")
