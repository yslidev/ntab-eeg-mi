"""The two control blocks that scripts/06_controls.py crashed before reaching.

06 died in the artefact block: dropping the noisiest 20% of epochs left one
subject with no usable test trials, and shrinkage LDA refuses an empty array.
evaluation.loso now returns None for such a fold instead of raising. These are
the same two blocks, appended to results/controls.csv.
"""
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, chosen, data as D, evaluation as E, config as CFG

OUT = pathlib.Path("results/controls.csv")
rows = pd.read_csv(OUT).to_dict("records") if OUT.is_file() else []
JOBS = 6


def rec(block, tag, **kw):
    rows.append(dict(block=block, tag=tag, **kw))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[{block}] {tag:<46} acc={kw.get('pooled_acc', np.nan):.4f} "
          f"[{kw.get('pooled_lo', np.nan):.3f},{kw.get('pooled_hi', np.nan):.3f}]",
          flush=True)


es = cached.load_pool(CFG.PARADIGM)
y, sub, run = es.y, es.subject, es.run
rows = [r for r in rows if r.get("block") not in ("artefact", "early-window")]

# --- artefact rejection ----------------------------------------------------
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
p2p = (X.max(-1) - X.min(-1)).max(1) * 1e6
print(f"peak-to-peak within 8-30 Hz per epoch (uV): median {np.median(p2p):.0f}, "
      f"p90 {np.percentile(p2p, 90):.0f}, p99 {np.percentile(p2p, 99):.0f}, "
      f"max {p2p.max():.0f}", flush=True)
Xa = chosen.featurize(X, sub)
del X; gc.collect()
for tag, thr in [("keep everything", np.inf),
                 ("drop worst 1% of epochs", np.percentile(p2p, 99)),
                 ("drop worst 5% of epochs", np.percentile(p2p, 95)),
                 ("drop worst 20% of epochs", np.percentile(p2p, 80))]:
    m = p2p <= thr
    r = E.loso(Xa[m], y[m], sub[m], run[m], chosen.make,
               subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
    rec("artefact", f"{tag} [{int(m.sum())} epochs]", **E.summarize(r))
del Xa; gc.collect()

# --- is the early window visual? -------------------------------------------
for win in [(0.0, 2.0), (0.5, 3.5), (2.0, 4.0)]:
    for tag, picks in [("all 64", None), ("sensorimotor", D.MOTOR_CH),
                       ("parieto-occipital", D.OCCIPITAL_CH)]:
        Xp, cp, _ = cached.prepare(es, band=CFG.BAND, window=win, picks=picks)
        n_ch = Xp.shape[1]
        k = min(chosen.N_COMPONENTS, max(2, (n_ch // 2) * 2 - 2))
        Xw = chosen.featurize(Xp, sub); del Xp; gc.collect()
        r = E.loso(Xw, y, sub, run,
                   lambda k=k: models.covcsp_lda(k, chosen.SHRINKAGE),
                   subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
        rec("early-window", f"{win[0]}-{win[1]} s, {tag} [{n_ch} ch]", **E.summarize(r))
        del Xw; gc.collect()

print("\nwrote", OUT)
