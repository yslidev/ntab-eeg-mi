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
rows, per_sub = [], []


def record(per=None, **kw):
    rows.append(kw)
    if per is not None:
        for d in per:
            per_sub.append(dict(paradigm=kw["paradigm"], regime=kw["regime"],
                                shuffled=kw["shuffled"], **d))
        pd.DataFrame(per_sub).to_csv("results/per_subject_rows.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{kw['paradigm']:<9}{kw['regime']:<26}{'SHUFFLED' if kw['shuffled'] else '':<9}"
          f"acc={kw['pooled_acc']:.4f} [{kw['pooled_lo']:.3f},{kw['pooled_hi']:.3f}]  "
          f"per-subj {kw['mean_sub_acc']:.3f}±{kw['sd_sub_acc']:.3f}  "
          f"sig {kw['frac_sig']*100:.0f}%  {kw['secs']:.0f}s", flush=True)


for paradigm in ["imagined", "executed"]:
    es = cached.load_pool(paradigm)
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
               secs=time.time() - t0, per=r, **E.summarize(r))

        # within-subject, whole runs held out -- same person, different block
        t0 = time.time()
        r = E.within_subject(Cov[m], es.y[m], es.subject[m], es.run[m], MAKE,
                             split="run", shuffle_labels=shuf, n_jobs=6)
        record(paradigm=paradigm, regime="B within-subj, leave-run-out", shuffled=shuf,
               secs=time.time() - t0, per=r, **E.summarize(r))

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
                   secs=time.time() - t0, per=r, **E.summarize(r))

# calibration curve
es = cached.load_pool("imagined")
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
           secs=time.time() - t0, per=r, **E.summarize(r))

print("\nwrote", args.out)


# --- tidy per-subject table for downstream analyses ----------------------
ps = pd.DataFrame(per_sub)
ps = ps[~ps.shuffled]
wide = (ps[ps.paradigm == "imagined"]
        .pivot_table(index="subject", columns="regime", values="acc"))
ren = {c: "acc_" + c.split(" ", 1)[1].replace(" ", "_").replace(",", "")
       for c in wide.columns}
wide = wide.rename(columns=ren)
for c, want in [("acc_LOSO_+_EA_align", "acc_loso"),
                ("acc_within-subj_leave-run-out", "acc_within_run"),
                ("acc_within-subj_random_CV", "acc_within_random"),
                ("acc_cross-subj_(LOSO)", "acc_loso_noalign")]:
    if c in wide.columns:
        wide[want] = wide[c]
wide.to_csv("results/per_subject_accuracy.csv")
print("wrote results/per_subject_accuracy.csv", wide.shape)
