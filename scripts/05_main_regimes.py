"""The headline table: the same model under five definitions of 'held out',
each with a label-shuffled twin, for both paradigms.

Test folds are the EVAL subjects, which were not used for any model choice.
"""
import sys, pathlib, time, argparse, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, chosen, evaluation as E, config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/main_regimes.csv")
ap.add_argument("--jobs", type=int, default=6)
args = ap.parse_args()

MAKE = chosen.make
rows, per_sub = [], []


def record(per=None, **kw):
    rows.append(kw)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    if per is not None:
        for d in per:
            per_sub.append(dict(paradigm=kw["paradigm"], regime=kw["regime"],
                                shuffled=kw["shuffled"], **d))
        pd.DataFrame(per_sub).to_csv("results/per_subject_rows.csv", index=False)
    print(f"{kw['paradigm']:<9}{kw['regime']:<26}{'SHUFFLED' if kw['shuffled'] else '':<9}"
          f"acc={kw['pooled_acc']:.4f} [{kw['pooled_lo']:.3f},{kw['pooled_hi']:.3f}]  "
          f"per-subj {kw['mean_sub_acc']:.3f}+-{kw['sd_sub_acc']:.3f}  "
          f"sig {kw['frac_sig']*100:.0f}%  {kw['secs']:.0f}s", flush=True)


for paradigm in ["imagined", "executed"]:
    es = cached.load_pool(paradigm)
    X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
    Xa = chosen.featurize(X, es.subject)
    del X; gc.collect()
    m = np.isin(es.subject, CFG.EVAL_SUBJECTS)

    for shuf in [False, True]:
        t0 = time.time()
        r = E.within_subject(Xa[m], es.y[m], es.subject[m], es.run[m], MAKE,
                             split="random", shuffle_labels=shuf, n_jobs=args.jobs)
        record(paradigm=paradigm, regime="A within-subj, random CV", shuffled=shuf,
               secs=time.time() - t0, per=r, **E.summarize(r))

        t0 = time.time()
        r = E.within_subject(Xa[m], es.y[m], es.subject[m], es.run[m], MAKE,
                             split="run", shuffle_labels=shuf, n_jobs=args.jobs)
        record(paradigm=paradigm, regime="B within-subj, leave-run-out", shuffled=shuf,
               secs=time.time() - t0, per=r, **E.summarize(r))

        t0 = time.time()
        r = E.loso(Xa, es.y, es.subject, es.run, MAKE,
                   subjects=CFG.EVAL_SUBJECTS, shuffle_labels=shuf, n_jobs=args.jobs)
        record(paradigm=paradigm, regime="C cross-subj, LOSO + align", shuffled=shuf,
               secs=time.time() - t0, per=r, **E.summarize(r))

        t0 = time.time()
        r = E.loso(Xa, es.y, es.subject, es.run, MAKE, subjects=CFG.EVAL_SUBJECTS,
                   calib_n=16, shuffle_labels=shuf, n_jobs=args.jobs)
        record(paradigm=paradigm, regime="D LOSO + 16 calib trials", shuffled=shuf,
               secs=time.time() - t0, per=r, **E.summarize(r))
    del Xa; gc.collect()

# --- calibration curve ----------------------------------------------------
es = cached.load_pool("imagined")
X, _, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
Xa = chosen.featurize(X, es.subject); del X; gc.collect()
for k in [0, 4, 8, 16, 24, 32]:
    t0 = time.time()
    r = E.loso(Xa, es.y, es.subject, es.run, MAKE, subjects=CFG.EVAL_SUBJECTS,
               calib_n=k, n_jobs=args.jobs)
    record(paradigm="imagined", regime=f"F LOSO + {k:2d} calib", shuffled=False,
           secs=time.time() - t0, **E.summarize(r))

ps = pd.DataFrame(per_sub)
ps = ps[~ps.shuffled]
wide = (ps[ps.paradigm == "imagined"]
        .pivot_table(index="subject", columns="regime", values="acc"))
for c, want in [("C cross-subj, LOSO + align", "acc_loso"),
                ("B within-subj, leave-run-out", "acc_within_run"),
                ("A within-subj, random CV", "acc_within_random"),
                ("D LOSO + 16 calib trials", "acc_loso_calib")]:
    if c in wide.columns:
        wide[want] = wide[c]
wide.to_csv("results/per_subject_accuracy.csv")
print("\nwrote", args.out, "and results/per_subject_accuracy.csv", wide.shape)
