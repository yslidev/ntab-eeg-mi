"""Model, band and window selection -- on DEV subjects only.

Everything chosen here is chosen against leave-one-subject-out folds drawn
from the 40 DEV subjects. The 65 EVAL subjects are not touched.

Note on compute: features that do not depend on the training fold (log-variance,
filter-bank power, covariance matrices) are computed once for all epochs. Only
CSP, which is a *supervised* spatial filter, has to be refitted per fold.
"""
import sys, pathlib, time, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, evaluation as E, config as CFG

OUT = pathlib.Path("results/model_selection.csv")
rows = []


def record(**kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{kw['tag']:<30} LOSO {kw['pooled_acc']:.4f} "
          f"[{kw['pooled_lo']:.3f},{kw['pooled_hi']:.3f}]  "
          f"within/run {kw.get('within_run', float('nan')):.4f}  "
          f"{kw['secs']:.0f}s", flush=True)


def run(tag, data, make_model, do_within=True, n_jobs=6):
    t0 = time.time()
    r = E.loso(data, es.y, es.subject, es.run, make_model,
               subjects=CFG.DEV_SUBJECTS, n_jobs=n_jobs)
    s = E.summarize(r, tag)
    wr = float("nan")
    if do_within:
        m = np.isin(es.subject, CFG.DEV_SUBJECTS)
        rw = E.within_subject(data[m], es.y[m], es.subject[m], es.run[m],
                              make_model, split="run", n_jobs=n_jobs)
        wr = E.summarize(rw)["pooled_acc"]
    record(tag=tag, within_run=wr, secs=time.time() - t0, **s)


def aligned(X, subject):
    ea = models.AlignShrunk(alpha=1.0)
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    return np.concatenate([ea.transform(X[subject == u]) for u in subs])[np.argsort(order)]


es = cached.load_pool(CFG.PARADIGM)
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)

# --- 1. fold-independent feature families ---------------------------------
run("logvar_lr", models.LogVar().transform(X), models.lr_only)
run("fbcsp_lite", models.FilterBank(sfreq=es.sfreq).transform(X), models.lr_only)
Cov = models.precompute_cov(X)
run("ts_riemann", Cov, models.ts_lr)
run("ts_logeuclid", Cov, models.ts_lr_le)

# --- 2. CSP: supervised, must be refitted per fold ------------------------
run("csp_lda_6", X, lambda: models.csp_lda(6), n_jobs=3)
run("csp_lda_12", X, lambda: models.csp_lda(12), n_jobs=3)

# --- 3. unsupervised per-subject alignment --------------------------------
Xa = aligned(X, es.subject)
del X; gc.collect()
Cova = models.precompute_cov(Xa)
run("ts_logeuclid+EA", Cova, models.ts_lr_le)
run("ts_riemann+EA", Cova, models.ts_lr)
run("csp_lda_6+EA", Xa, lambda: models.csp_lda(6), n_jobs=3)
run("logvar_lr+EA", models.LogVar().transform(Xa), models.lr_only)
del Xa; gc.collect()

# --- 4. regularisation strength -------------------------------------------
for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    run(f"ts_le+EA C={c}", Cova, lambda c=c: models.ts_lr_le(C=c), do_within=False)
del Cova; gc.collect()

# --- 5. frequency band ----------------------------------------------------
for band in [(4, 8), (8, 13), (13, 30), (8, 30), (4, 38), (1, 45)]:
    Xb, _, _ = cached.prepare(es, band=band, window=CFG.WINDOW)
    Cb = models.precompute_cov(aligned(Xb, es.subject))
    del Xb; gc.collect()
    run(f"band {band[0]}-{band[1]}Hz", Cb, models.ts_lr_le, do_within=False)
    del Cb; gc.collect()

# --- 6. analysis window ---------------------------------------------------
for win in [(0.0, 2.0), (0.5, 2.5), (0.5, 3.5), (0.0, 4.0), (1.0, 3.0), (1.0, 4.1)]:
    Xw, _, _ = cached.prepare(es, band=CFG.BAND, window=win)
    Cw = models.precompute_cov(aligned(Xw, es.subject))
    del Xw; gc.collect()
    run(f"window {win[0]}-{win[1]}s", Cw, models.ts_lr_le, do_within=False)
    del Cw; gc.collect()

print("\nwrote", OUT)
