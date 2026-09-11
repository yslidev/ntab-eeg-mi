"""The numbers you get if you split the data the easy way.

Two shortcuts are common enough in published EEG work to be worth measuring
rather than merely warning about:

  1. Pool every subject's trials into one pile and take a random k-fold. The
     test set then contains trials from people whose other trials are in the
     training set, and subject identity is ~98% decodable from these features,
     so the model can memorise people.

  2. The same, but after "augmenting" each trial into several overlapping
     crops. Now the test set contains crops that overlap in time with training
     crops from the same trial.

Neither tells you anything about a new person. Both are reported here next to
the honest leave-one-subject-out number so the size of the gap is visible.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
import cached, models, chosen, evaluation as E, config as CFG

OUT = pathlib.Path("results/naive_splits.csv")
rows = []


def rec(**kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{kw['tag']:<52} acc={kw['acc']:.4f} [{kw['lo']:.3f},{kw['hi']:.3f}]  "
          f"n={kw['n']}", flush=True)


def pooled_kfold(C, y, tag, n_splits=5, seed=0):
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(n_splits, shuffle=True,
                                  random_state=seed).split(C, y):
        pred[te] = clone(chosen.make()).fit(C[tr], y[tr]).predict(C[te])
    k = int((pred == y).sum())
    lo, hi = E.binom_ci(k, len(y))
    rec(tag=tag, acc=k / len(y), lo=lo, hi=hi, n=len(y))


es = cached.load_pool(CFG.PARADIGM)
m = np.isin(es.subject, np.concatenate([CFG.EVAL_SUBJECTS, CFG.DEV_SUBJECTS]))

# --- 1. pooled random k-fold over whole trials ----------------------------
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
Cov = chosen.featurize(X, es.subject)
pooled_kfold(Cov, es.y, "NAIVE: pooled random 5-fold over trials, subjects mixed")

# without the alignment, which is what removes subject identity
Cov_raw = models.precompute_cov(X)
pooled_kfold(Cov_raw, es.y,
             "NAIVE: same, but with no per-subject alignment")
del Cov_raw

# --- 2. pooled random k-fold over overlapping crops -----------------------
# four 2 s crops per trial, hopping 0.5 s: adjacent crops share 75% of samples
t = cached.times(es)
crops, ys, subs = [], [], []
for start in [0.5, 1.0, 1.5, 2.0]:
    sel = (t >= start) & (t <= start + 2.0)
    n_t = int(sel.sum())
    crops.append(X[..., np.where(sel)[0][:n_t]])
    ys.append(es.y); subs.append(es.subject)
Xc = np.concatenate(crops); yc = np.concatenate(ys); sc = np.concatenate(subs)
del crops
Cc = chosen.featurize(Xc, sc)
pooled_kfold(Cc, yc, "NAIVE: pooled random 5-fold over overlapping 2 s crops")

# same crops, but split by subject: the augmentation alone is not the problem
r = E.loso(Cc, yc, sc, np.tile(es.run, 4), chosen.make,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=6)
s = E.summarize(r)
rec(tag="HONEST: same crops, leave-one-subject-out",
    acc=s["pooled_acc"], lo=s["pooled_lo"], hi=s["pooled_hi"], n=s["n_trials"])
del Cc, Xc

# --- 3. the honest reference ----------------------------------------------
r = E.loso(Cov, es.y, es.subject, es.run, chosen.make,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=6)
s = E.summarize(r)
rec(tag="HONEST: whole trials, leave-one-subject-out",
    acc=s["pooled_acc"], lo=s["pooled_lo"], hi=s["pooled_hi"], n=s["n_trials"])

print("\nwrote", OUT)
