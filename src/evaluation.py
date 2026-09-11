"""Evaluation regimes.

Each function answers a *different* question, and the difference between them
is the substance of this project:

  within_subject(split="random") -- can a model trained on some of a person's
      trials label their other trials, when the two sets are interleaved in time?
  within_subject(split="run")    -- ... when the test trials come from a
      different recording run (different block, minutes apart)?
  loso()                         -- can a model trained on 104 people label a
      person it has never seen?
  loso(calib_n=k)                -- ... given k labelled trials from them?

Per-recording alignment is applied *before* these functions are called (see
src/chosen.py), because it is unsupervised and fold-independent, so it does not
need to be inside the fold loop.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed



def binom_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = stats.beta.ppf(alpha / 2, k, n - k + 1), stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (0.0 if np.isnan(lo) else lo, 1.0 if np.isnan(hi) else hi)


def binom_sig_threshold(n, alpha=0.05, p0=0.5):
    """Smallest accuracy that beats chance at one-sided alpha for n trials."""
    return stats.binom.ppf(1 - alpha, n, p0) / n if n > 0 else np.nan


def _shuffle_within(y, groups, rng):
    """Permute labels inside each group, preserving per-group class counts."""
    y = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y[m] = rng.permutation(y[m])
    return y


def within_subject(X, y, subject, run, make_model, split="random", n_splits=5,
                   shuffle_labels=False, seed=0, n_jobs=6):
    """Train and test inside each subject. Returns one row per subject."""
    def one(s):
        rng = np.random.default_rng(seed + int(s))
        m = subject == s
        Xs, ys, rs = X[m], y[m], run[m]
        if shuffle_labels:
            ys = _shuffle_within(ys, rs if split == "run" else np.zeros_like(rs), rng)
        if split == "random":
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                         random_state=seed).split(Xs, ys))
        elif split == "run":
            folds = [(np.where(rs != r)[0], np.where(rs == r)[0])
                     for r in np.unique(rs)]
        else:
            raise ValueError(split)
        preds = np.full(len(ys), -1)
        for tr, te in folds:
            if len(np.unique(ys[tr])) < 2:
                continue
            mdl = clone(make_model())
            mdl.fit(Xs[tr], ys[tr])
            preds[te] = mdl.predict(Xs[te])
        ok = preds >= 0
        acc = (preds[ok] == ys[ok]).mean()
        lo, hi = binom_ci((preds[ok] == ys[ok]).sum(), ok.sum())
        return dict(subject=int(s), n=int(ok.sum()), acc=float(acc),
                    ci_lo=lo, ci_hi=hi,
                    sig_thresh=binom_sig_threshold(int(ok.sum())),
                    majority=float(max(np.mean(ys == 0), np.mean(ys == 1))))

    subs = np.unique(subject)
    return Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in subs)


def loso(X, y, subject, run, make_model, calib_n=0,
         shuffle_labels=False, seed=0, n_jobs=5, subjects=None):
    """Leave-one-subject-out, optionally giving the held-out subject k labelled
    calibration trials taken in recording order, so calibration data always
    precedes test data in time."""
    subs = np.unique(subject) if subjects is None else np.asarray(subjects)

    def one(s):
        rng = np.random.default_rng(seed + int(s))
        te_m, tr_m = subject == s, subject != s
        if te_m.sum() == 0 or len(np.unique(y[tr_m])) < 2:
            return None                      # e.g. after aggressive epoch dropping
        Xtr, ytr, str_ = X[tr_m], y[tr_m].copy(), subject[tr_m]
        Xte, yte, rte = X[te_m], y[te_m].copy(), run[te_m]
        if shuffle_labels:
            ytr = _shuffle_within(ytr, str_, rng)
            yte = _shuffle_within(yte, rte, rng)

        keep = np.ones(len(yte), bool)
        if calib_n > 0:
            # Take calibration trials in recording order -- earliest runs first,
            # class-balanced -- so calibration data always precedes test data.
            take = []
            need = {0: calib_n // 2, 1: calib_n - calib_n // 2}
            for r_ in np.unique(rte):
                pool = np.where(rte == r_)[0]
                for lab in (0, 1):
                    cand = pool[yte[pool] == lab]
                    n_take = min(need[lab], len(cand))
                    take += list(cand[:n_take])
                    need[lab] -= n_take
                if need[0] <= 0 and need[1] <= 0:
                    break
            take = np.array(take, int)
            if len(take):
                Xtr = np.concatenate([Xtr, Xte[take]])
                ytr = np.concatenate([ytr, yte[take]])
                keep[take] = False

        if keep.sum() == 0 or len(np.unique(ytr)) < 2:
            return None
        mdl = clone(make_model())
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte[keep])
        acc = (pred == yte[keep]).mean()
        lo, hi = binom_ci((pred == yte[keep]).sum(), keep.sum())
        tr_pred = mdl.predict(Xtr[: min(len(ytr), 1500)])
        return dict(subject=int(s), n=int(keep.sum()), acc=float(acc),
                    ci_lo=lo, ci_hi=hi, sig_thresh=binom_sig_threshold(int(keep.sum())),
                    train_acc=float((tr_pred == ytr[: len(tr_pred)]).mean()))

    out = Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in subs)
    return [r for r in out if r is not None]


def summarize(rows, name=""):
    import pandas as pd
    df = pd.DataFrame(rows)
    n_tot = df.n.sum()
    k_tot = (df.acc * df.n).sum()
    pooled = k_tot / n_tot
    lo, hi = binom_ci(round(k_tot), int(n_tot))
    return dict(name=name, pooled_acc=pooled, pooled_lo=lo, pooled_hi=hi,
                mean_sub_acc=df.acc.mean(), sd_sub_acc=df.acc.std(),
                median=df.acc.median(), q25=df.acc.quantile(.25),
                q75=df.acc.quantile(.75), min=df.acc.min(), max=df.acc.max(),
                n_subjects=len(df), n_trials=int(n_tot),
                frac_sig=float((df.acc > df.sig_thresh).mean()))
