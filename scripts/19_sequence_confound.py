"""The cue sequence is not random, and that explains two alarming controls.

Consecutive trials in a run carry *different* labels 76.8% of the time
(lag-1 correlation -0.54, p ~ 1e-245). BCI2000's target sequence strongly
alternates. Two consequences, both measured here:

  1. Predicting the PREVIOUS trial's label from the CURRENT trial's window
     looks like evidence of carry-over between trials. It is not. A decoder
     that is right about the current label a fraction `a` of the time scores
     a*q + (1-a)*(1-q) on the previous label for free, where q = 0.768.

  2. The 4.2 s of rest before a cue carries the lateralised aftermath of the
     PREVIOUS trial. Because the next label is usually the opposite, that rest
     period predicts the upcoming label above chance without containing any
     information about it.

The decisive test is to split trials by whether the label repeated. If the
pre-cue window works only through the previous trial, it should be right on
repeats and wrong on alternations, in mirror image.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from joblib import Parallel, delayed
import cached, chosen, evaluation as E, config as CFG

OUT = pathlib.Path("results/sequence_confound.csv")
rows = []


def rec(**kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{kw['tag']:<54} acc={kw['acc']:.4f} [{kw['lo']:.3f},{kw['hi']:.3f}] "
          f"n={kw['n']}", flush=True)


es = cached.load_pool(CFG.PARADIGM)
y, sub, run, trial = es.y, es.subject, es.run, es.trial

# --- sequence statistics ---------------------------------------------------
prev = np.full(len(y), -1)
for s in np.unique(sub):
    for r in np.unique(run[sub == s]):
        idx = np.where((sub == s) & (run == r))[0]
        idx = idx[np.argsort(trial[idx])]
        prev[idx[1:]] = y[idx[:-1]]
has_prev = prev >= 0
changed = (y != prev) & has_prev
q = float(changed.sum() / has_prev.sum())
bt = stats.binomtest(int(changed.sum()), int(has_prev.sum()), 0.5)
print(f"consecutive-trial pairs: {int(has_prev.sum())}")
print(f"P(label changes) = {q:.4f}  95% CI "
      f"{tuple(round(v, 4) for v in bt.proportion_ci())}  p vs 0.5 = {bt.pvalue:.2e}")
print(f"lag-1 correlation of the label sequence = {1 - 2 * q:+.4f}\n")


def loso_scores(Cov, labels, mask=None, tag=""):
    """Leave-one-subject-out, returning per-trial correctness."""
    keep = np.ones(len(labels), bool) if mask is None else mask

    def one(s):
        tr = (sub != s) & keep
        te = (sub == s) & keep
        if te.sum() == 0 or len(np.unique(labels[tr])) < 2:
            return None
        mdl = clone(chosen.make()).fit(Cov[tr], labels[tr])
        p = mdl.predict(Cov[te])
        return np.where(te)[0], p

    res = [r for r in Parallel(n_jobs=5)(delayed(one)(s) for s in CFG.EVAL_SUBJECTS)
           if r is not None]
    idx = np.concatenate([r[0] for r in res])
    pred = np.concatenate([r[1] for r in res])
    return idx, pred


def report(tag, idx, pred, truth):
    k = int((pred == truth[idx]).sum()); n = len(idx)
    lo, hi = E.binom_ci(k, n)
    rec(tag=tag, acc=k / n, lo=lo, hi=hi, n=n)
    return pred


# --- the task window -------------------------------------------------------
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
Cov_task = chosen.featurize(X, sub)
del X

i, p = loso_scores(Cov_task, y)
report("task window -> CURRENT label", i, p, y)

i, p = loso_scores(Cov_task, prev, mask=has_prev)
obs_prev = report("task window -> PREVIOUS label (looks like carry-over)", i, p, prev)

acc_cur = rows[0]["acc"]
expected = acc_cur * q + (1 - acc_cur) * (1 - q)
print(f"\n  expected from sequence structure alone, given {acc_cur*100:.1f}% on the "
      f"current label: {expected*100:.1f}%")
print(f"  observed: {rows[1]['acc']*100:.1f}%  -> "
      f"{'no evidence of genuine carry-over' if abs(rows[1]['acc']-expected) < 0.03 else 'MORE than the structure explains'}\n")

# --- the pre-cue window ----------------------------------------------------
Xp, _, _ = cached.prepare(es, band=CFG.BAND, window=(-2.4, -0.4))
Cov_pre = chosen.featurize(Xp, sub)
del Xp

i, p = loso_scores(Cov_pre, y)
report("PRE-CUE rest -> CURRENT label", i, p, y)

i, p = loso_scores(Cov_pre, prev, mask=has_prev)
report("PRE-CUE rest -> PREVIOUS label", i, p, prev)

# the decisive split: does the pre-cue window help on repeats and hurt on
# alternations, which is what a pure previous-trial trace would do?
i, p = loso_scores(Cov_pre, y)
rep = has_prev[i] & (y[i] == prev[i])
alt = has_prev[i] & (y[i] != prev[i])
for name, m in [("PRE-CUE -> current, trials where label REPEATED", rep),
                ("PRE-CUE -> current, trials where label ALTERNATED", alt)]:
    k = int((p[m] == y[i][m]).sum()); n = int(m.sum())
    lo, hi = E.binom_ci(k, n)
    rec(tag=name, acc=k / n, lo=lo, hi=hi, n=n)

print("\nIf the pre-cue window only carries the previous trial, accuracy on the")
print("current label should sit ABOVE chance on repeats and BELOW chance on")
print("alternations. If it carried genuine advance information it would be")
print("above chance on both.")
print("\nwrote", OUT)
