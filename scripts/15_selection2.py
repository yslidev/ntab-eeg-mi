"""Second selection pass, DEV subjects only.

Pass 1 (scripts/04_model_selection.py) compared feature families at one
arbitrary regularisation strength and found that Euclidean alignment helps CSP
(61.6 -> 68.3) and hurts the tangent-space models. That comparison was unfair:
the tangent-space model turned out to be very sensitive to its regularisation.
This pass works entirely in the covariance domain (see models.CovCSP), which is
about twenty times faster, and tunes each family's own knob before comparing.
"""
import sys, pathlib, time, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, evaluation as E, config as CFG

OUT = pathlib.Path("results/selection2.csv")
rows = []


def record(**kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{kw['family']:<14}{kw['tag']:<26} LOSO {kw['pooled_acc']:.4f} "
          f"[{kw['pooled_lo']:.3f},{kw['pooled_hi']:.3f}]  "
          f"per-subj {kw['mean_sub_acc']:.3f}+-{kw['sd_sub_acc']:.3f}  "
          f"{kw['secs']:.0f}s", flush=True)


def run(family, tag, data, make_model, n_jobs=6):
    t0 = time.time()
    r = E.loso(data, es.y, es.subject, es.run, make_model,
               subjects=CFG.DEV_SUBJECTS, n_jobs=n_jobs)
    record(family=family, tag=tag, secs=time.time() - t0, **E.summarize(r, tag))


def aligned_cov(X, subject, alpha=1.0):
    tr = models.AlignShrunk(alpha=alpha)
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    Xa = np.concatenate([tr.transform(X[subject == u]) for u in subs])[np.argsort(order)]
    C = models.precompute_cov(Xa)
    del Xa; gc.collect()
    return C


es = cached.load_pool(CFG.PARADIGM)
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)

# --- 1. how strong should the alignment be? -------------------------------
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    C = aligned_cov(X, es.subject, alpha=a)
    run("alignment", f"CovCSP6, alpha={a}", C, lambda: models.covcsp_lda(6))
    run("alignment", f"tangent-space, alpha={a}", C, lambda: models.ts_lr_le(C=1e-3))
    del C; gc.collect()

Cov = aligned_cov(X, es.subject, alpha=1.0)

# --- 2. CSP components and head -------------------------------------------
for k in [2, 4, 6, 8, 12, 16, 24, 32]:
    run("CSP", f"k={k}, shrinkage LDA", Cov, lambda k=k: models.covcsp_lda(k))
for k in [6, 12]:
    for c in [0.03, 0.3, 3.0]:
        run("CSP", f"k={k}, LR C={c:g}", Cov, lambda k=k, c=c: models.covcsp_lr(k, C=c))
for sh in [0.0, 0.02, 0.05, 0.2, 0.5]:
    run("CSP", f"k=8, shrinkage={sh}", Cov, lambda sh=sh: models.covcsp_lda(8, sh))

# --- 3. tangent space, properly regularised -------------------------------
for c in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    run("tangent-space", f"logeuclid C={c:g}", Cov, lambda c=c: models.ts_lr_le(C=c))
for c in [3e-4, 1e-3, 3e-3]:
    run("tangent-space", f"riemann C={c:g}", Cov, lambda c=c: models.ts_lr(C=c), 5)

# --- 4. log-variance, properly regularised --------------------------------
Xa = models.AlignShrunk(1.0)
subs = np.unique(es.subject)
order = np.concatenate([np.where(es.subject == u)[0] for u in subs])
XA = np.concatenate([Xa.transform(X[es.subject == u]) for u in subs])[np.argsort(order)]
F = models.LogVar().transform(XA)
for c in [3e-3, 1e-2, 3e-2, 1e-1, 1.0]:
    run("log-variance", f"C={c:g}", F, lambda c=c: models.lr_only(C=c))
del F, XA; gc.collect()
del Cov, X; gc.collect()

# --- 5. band --------------------------------------------------------------
BEST_K = 6
for band in [(8, 13), (13, 30), (8, 30), (7, 35), (4, 38), (4, 8)]:
    Xb, _, _ = cached.prepare(es, band=band, window=CFG.WINDOW)
    C = aligned_cov(Xb, es.subject); del Xb; gc.collect()
    run("band", f"{band[0]}-{band[1]} Hz", C, lambda: models.covcsp_lda(BEST_K))
    del C; gc.collect()

# --- 6. window ------------------------------------------------------------
for win in [(0.0, 2.0), (0.5, 2.5), (0.5, 3.5), (0.0, 4.0), (1.0, 4.0), (0.0, 4.1)]:
    Xw, _, _ = cached.prepare(es, band=CFG.BAND, window=win)
    C = aligned_cov(Xw, es.subject); del Xw; gc.collect()
    run("window", f"{win[0]}-{win[1]} s", C, lambda: models.covcsp_lda(BEST_K))
    del C; gc.collect()

# --- 7. filter bank -------------------------------------------------------
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30), (30, 38)]
covs = []
for band in BANDS:
    Xb, _, _ = cached.prepare(es, band=band, window=CFG.WINDOW)
    covs.append(aligned_cov(Xb, es.subject)); del Xb; gc.collect()


class FBCovCSP(models.BaseEstimator, models.TransformerMixin):
    """CSP per sub-band on precomputed covariances; features concatenated."""
    def __init__(self, k=4, shrinkage=0.1):
        self.k, self.shrinkage = k, shrinkage

    def fit(self, C, y):
        n = C.shape[0] // len(BANDS)
        self.c_ = [models.CovCSP(self.k, self.shrinkage).fit(C[i * n:(i + 1) * n], y)
                   for i in range(len(BANDS))]
        return self

    def transform(self, C):
        n = C.shape[0] // len(BANDS)
        return np.concatenate([c.transform(C[i * n:(i + 1) * n])
                               for i, c in enumerate(self.c_)], axis=1)


# stack bands along the epoch axis; the estimator knows how to unstack
STACK = np.concatenate(covs, axis=0)
del covs; gc.collect()


def fb_pipe(k=4, C=0.3):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("fb", FBCovCSP(k)), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


n = len(es.y)
idx_all = np.arange(n)


def fb_run(tag, k, c):
    t0 = time.time()
    # index the stacked array through a small wrapper: rebuild per fold
    def one(s):
        from sklearn.base import clone
        tr, te = es.subject != s, es.subject == s
        sel = lambda m: np.concatenate([STACK[i * n:(i + 1) * n][m]
                                        for i in range(len(BANDS))], axis=0)
        mdl = clone(fb_pipe(k, c)).fit(sel(tr), es.y[tr])
        p = mdl.predict(sel(te))
        k_ = int((p == es.y[te]).sum()); nn = int(te.sum())
        lo, hi = E.binom_ci(k_, nn)
        return dict(subject=int(s), n=nn, acc=k_ / nn, ci_lo=lo, ci_hi=hi,
                    sig_thresh=E.binom_sig_threshold(nn))
    from joblib import Parallel, delayed
    r = Parallel(n_jobs=4)(delayed(one)(s) for s in CFG.DEV_SUBJECTS)
    record(family="filter-bank", tag=tag, secs=time.time() - t0, **E.summarize(r, tag))


for k in [2, 4]:
    for c in [0.1, 1.0]:
        fb_run(f"FBCSP k={k} C={c:g}", k, c)

print("\nwrote", OUT)
