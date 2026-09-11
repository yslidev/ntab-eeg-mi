"""Train and serialise the model that predict.py ships with.

Trained on every usable subject (105) so that a new subject is genuinely
out-of-sample for whoever runs predict.py.
"""
import sys, pathlib, json, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, joblib
import cached, models, config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--paradigm", default="imagined", choices=["imagined", "executed", "both"])
ap.add_argument("--out", default="model/mi_lr_model.joblib")
args = ap.parse_args()

pars = ["imagined", "executed"] if args.paradigm == "both" else [args.paradigm]
Xs, ys, subs = [], [], []
for p in pars:
    es = cached.load_pool(p)
    X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
    ea = models.EuclideanAlign()
    for u in np.unique(es.subject):                 # align each subject separately
        m = es.subject == u
        Xs.append(ea.transform(X[m])); ys.append(es.y[m]); subs.append(es.subject[m])
X = np.concatenate(Xs); y = np.concatenate(ys); sub = np.concatenate(subs)
print(f"training on {len(y)} epochs from {len(np.unique(sub))} subjects, "
      f"paradigm={args.paradigm}")

Cov = models.precompute_cov(X)
pipe = models.ts_lr_le().fit(Cov, y)

out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(dict(
    pipeline=pipe, ch_names=chs, band=CFG.BAND, window=CFG.WINDOW,
    sfreq=160.0, cov_estimator="oas", euclidean_align=True,
    labels={0: "left_fist", 1: "right_fist"},
    paradigm=args.paradigm, n_train_subjects=int(len(np.unique(sub))),
    n_train_epochs=int(len(y)),
), out, compress=3)
print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")
