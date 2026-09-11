"""The headline table: the same model under five different train/test splits,
each with a label-shuffled twin, for both paradigms.

Test folds are the 65 EVAL subjects, which were not used for any model choice.
"""
import sys, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, evaluation as E, config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ts_logeuclid")
ap.add_argument("--out", default="results/main_regimes.csv")
args = ap.parse_args()

MAKE = {"ts_logeuclid": models.ts_lr_le, "ts_riemann": models.ts_lr}[args.model]
rows = []


def record(**kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{kw['paradigm']:<9}{kw['regime']:<26}{'SHUFFLED' if kw['shuffled'] else '':<9}"
          f"acc={kw['pooled_acc']:.4f} [{kw['pooled_lo']:.3f},{kw['pooled_hi']:.3f}]  "
          f"per-subj {kw['mean_sub_acc']:.3f}±{kw['sd_sub_acc']:.3f}  "
          f"sig {kw['frac_sig']*100:.0f}%  {kw['secs']:.0f}s", flush=True)


for paradigm in ["imagined", "executed"]:
    es = cached.load(paradigm)
    X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
    ea = models.EuclideanAlign()
    subs_all = np.unique(es.subject)
    order = np.concatenate([np.where(es.subject == u)[0] for u in subs_all])
    inv = np.argsort(order)
    Xa = np.concatenate([ea.transform(X[es.subject == u]) for u in subs_all])[inv]
    Cov, Cova = models.precompute_cov(X), models.precompute_cov(Xa)

    for shuf in [False, True]:
        # within-subject, interleaved random folds -- the optimistic number
        t0 = time.time()
        m = np.isin(es.subject, CFG.EVAL_SUBJECTS)
        r = E.within_subject(Cov[m], es.y[m], es.subject[m], es.run[m], MAKE,
                             split="random", shuffle_labels=shuf, n_jobs=6)
        record(paradigm=paradigm, regime="A within-subj, random CV", shuffled=shuf,
               secs=time.time() - t0, **E.summarize(r))

        # within-subject, whole runs held out -- same person, different block
        t0 = time.time()
        r = E.within_subject(Cov[m], es.y[m], es.subject[m], es.run[m], MAKE,
                             split="run", shuffle_labels=shuf, n_jobs=6)
        record(paradigm=paradigm, regime="B within-subj, leave-run-out", shuffled=shuf,
               secs=time.time() - t0, **E.summarize(r))

        # cross-subject
        for tag, data, kw in [
                ("C cross-subj (LOSO)", Cov, {}),
                ("D LOSO + EA align", Cova, {}),
                ("E LOSO + EA + 16 calib", Cova, dict(calib_n=16))]:
            t0 = time.time()
            r = E.loso(data, es.y, es.subject, es.run, MAKE,
                       subjects=CFG.EVAL_SUBJECTS, shuffle_labels=shuf,
                       n_jobs=5, **kw)
            record(paradigm=paradigm, regime=tag, shuffled=shuf,
                   secs=time.time() - t0, **E.summarize(r))

# calibration curve
es = cached.load("imagined")
X, _, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
ea = models.EuclideanAlign(); subs_all = np.unique(es.subject)
order = np.concatenate([np.where(es.subject == u)[0] for u in subs_all])
Xa = np.concatenate([ea.transform(X[es.subject == u]) for u in subs_all])[np.argsort(order)]
Cova = models.precompute_cov(Xa)
for k in [0, 4, 8, 16, 24, 32]:
    t0 = time.time()
    r = E.loso(Cova, es.y, es.subject, es.run, MAKE, subjects=CFG.EVAL_SUBJECTS,
               calib_n=k, n_jobs=5)
    record(paradigm="imagined", regime=f"F LOSO + EA + {k:2d} calib", shuffled=False,
           secs=time.time() - t0, **E.summarize(r))

print("\nwrote", args.out)
