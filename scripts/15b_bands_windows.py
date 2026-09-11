"""Band, window, log-variance and filter-bank sweeps -- DEV subjects only.

Continues scripts/15_selection2.py. The affine-invariant tangent-space configs
were dropped after the log-Euclidean variant of the same family had been
tuned across three orders of magnitude of regularisation and still lost to CSP
by four points; each remaining one cost six minutes.
"""
import sys, pathlib, time, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, evaluation as E, config as CFG

OUT = pathlib.Path("results/selection2.csv")
rows = pd.read_csv(OUT).to_dict("records") if OUT.is_file() else []
K = 12                                     # one-standard-error choice from pass 2


def record(**kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{kw['family']:<14}{kw['tag']:<26} LOSO {kw['pooled_acc']:.4f} "
          f"[{kw['pooled_lo']:.3f},{kw['pooled_hi']:.3f}]  {kw['secs']:.0f}s", flush=True)


def run(family, tag, data, make_model, n_jobs=6):
    t0 = time.time()
    r = E.loso(data, es.y, es.subject, es.run, make_model,
               subjects=CFG.DEV_SUBJECTS, n_jobs=n_jobs)
    record(family=family, tag=tag, secs=time.time() - t0, **E.summarize(r, tag))


def aligned(X, subject):
    tr = models.AlignShrunk(alpha=1.0)
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    return np.concatenate([tr.transform(X[subject == u]) for u in subs])[np.argsort(order)]


es = cached.load_pool(CFG.PARADIGM)

# --- log-variance, properly regularised -----------------------------------
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
XA = aligned(X, es.subject)
F = models.LogVar().transform(XA)
for c in [3e-3, 1e-2, 3e-2, 1e-1, 1.0]:
    run("log-variance", f"C={c:g}", F, lambda c=c: models.lr_only(C=c))
del F, XA, X; gc.collect()

# --- band ------------------------------------------------------------------
for band in [(4, 8), (8, 13), (13, 30), (8, 30), (7, 35), (4, 38)]:
    Xb, _, _ = cached.prepare(es, band=band, window=CFG.WINDOW)
    C = models.precompute_cov(aligned(Xb, es.subject)); del Xb; gc.collect()
    run("band", f"{band[0]}-{band[1]} Hz", C, lambda: models.covcsp_lda(K))
    del C; gc.collect()

# --- window ----------------------------------------------------------------
for win in [(0.0, 2.0), (0.5, 2.5), (0.5, 3.5), (0.0, 4.0), (1.0, 4.0), (0.0, 4.1)]:
    Xw, _, _ = cached.prepare(es, band=CFG.BAND, window=win)
    C = models.precompute_cov(aligned(Xw, es.subject)); del Xw; gc.collect()
    run("window", f"{win[0]}-{win[1]} s", C, lambda: models.covcsp_lda(K))
    del C; gc.collect()

# --- filter bank -----------------------------------------------------------
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30), (30, 38)]
covs = []
for band in BANDS:
    Xb, _, _ = cached.prepare(es, band=band, window=CFG.WINDOW)
    covs.append(models.precompute_cov(aligned(Xb, es.subject)))
    del Xb; gc.collect()
STACK = np.stack(covs, axis=1)          # (n_epochs, n_bands, ch, ch)
del covs; gc.collect()
print("filter bank stack", STACK.shape, f"{STACK.nbytes/1e9:.2f} GB", flush=True)


class FBCovCSP(models.BaseEstimator, models.TransformerMixin):
    """One CSP per sub-band on precomputed covariances, features concatenated."""
    def __init__(self, k=4, shrinkage=0.1):
        self.k, self.shrinkage = k, shrinkage

    def fit(self, C, y):
        self.c_ = [models.CovCSP(self.k, self.shrinkage).fit(C[:, b], y)
                   for b in range(C.shape[1])]
        return self

    def transform(self, C):
        return np.concatenate([c.transform(C[:, b]) for b, c in enumerate(self.c_)],
                              axis=1)


def fb_pipe(k=4, c=0.3):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("fb", FBCovCSP(k)), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=c, max_iter=3000))])


for k in [2, 4]:
    for c in [0.1, 1.0]:
        run("filter-bank", f"FBCSP k={k} C={c:g}", STACK,
            lambda k=k, c=c: fb_pipe(k, c), 4)

print("\nwrote", OUT)
