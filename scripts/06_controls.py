"""Controls: what else could explain the number?

Each block here is a way for the headline accuracy to be real-looking but
wrong. We run them all on the same pipeline as the headline model.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace
import cached, models, data as D, evaluation as E, config as CFG

OUT = pathlib.Path("results/controls.csv")
rows = []
MAKE = models.ts_lr_le


def rec(block, tag, **kw):
    rows.append(dict(block=block, tag=tag, **kw))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    acc = kw.get("pooled_acc", np.nan)
    print(f"[{block}] {tag:<40} acc={acc:.4f} "
          f"[{kw.get('pooled_lo', np.nan):.3f},{kw.get('pooled_hi', np.nan):.3f}]", flush=True)


def align(X, subject):
    ea = models.EuclideanAlign()
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    return np.concatenate([ea.transform(X[subject == u]) for u in subs])[np.argsort(order)]


es = cached.load_pool(CFG.PARADIGM)
y, sub, run = es.y, es.subject, es.run

# =====================================================================
# 1. WINDOW PLACEBO. The 4.2 s before each cue is rest. If the pipeline
#    decodes the upcoming label from it, the "decoding" is not the task.
# =====================================================================
for tag, win in [("task window 0.5-3.5s", (0.5, 3.5)),
                 ("PRE-CUE -2.0 to -0.2s", (-2.0, -0.2)),
                 ("PRE-CUE -2.4 to -0.4s", (-2.4, -0.4)),
                 ("late window 2.5-4.5s", (2.5, 4.5))]:
    X, _, _ = cached.prepare(es, band=CFG.BAND, window=win)
    Cov = models.precompute_cov(align(X, sub))
    r = E.loso(Cov, y, sub, run, MAKE, subjects=CFG.EVAL_SUBJECTS, n_jobs=5)
    rec("window", f"LOSO  {tag}", **E.summarize(r))
    m = np.isin(sub, CFG.EVAL_SUBJECTS)
    r = E.within_subject(models.precompute_cov(X)[m], y[m], sub[m], run[m],
                         MAKE, split="random", n_jobs=6)
    rec("window", f"within/randomCV  {tag}", **E.summarize(r))

# =====================================================================
# 2. CHANNEL LESION. If a classifier restricted to occipital or frontal
#    electrodes does as well as one on the sensorimotor strip, we are not
#    decoding sensorimotor rhythm.
# =====================================================================
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
for tag, picks in [("all 64 channels", None),
                   ("sensorimotor strip (15)", D.MOTOR_CH),
                   ("C3/Cz/C4 only (3)", ["C3", "Cz", "C4"]),
                   ("frontal (17)", D.FRONTAL_CH),
                   ("parieto-occipital (17)", D.OCCIPITAL_CH)]:
    Xp, cp, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW, picks=picks)
    Cov = models.precompute_cov(align(Xp, sub))
    r = E.loso(Cov, y, sub, run, MAKE, subjects=CFG.EVAL_SUBJECTS, n_jobs=5)
    rec("lesion", f"{tag} [{Xp.shape[1]} ch]", **E.summarize(r))

# =====================================================================
# 3. LABEL CARRY-OVER. Give each trial the label of the PREVIOUS trial in
#    the same run. Above chance would mean trials are not independent.
# =====================================================================
Cov = models.precompute_cov(align(X, sub))
y_prev = y.copy(); keep = np.ones(len(y), bool)
for s in np.unique(sub):
    for r_ in np.unique(run[sub == s]):
        m = np.where((sub == s) & (run == r_))[0]
        m = m[np.argsort(es.trial[m])]
        y_prev[m[1:]] = y[m[:-1]]
        keep[m[0]] = False
r = E.loso(Cov[keep], y_prev[keep], sub[keep], run[keep], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=5)
rec("carryover", "LOSO, label = PREVIOUS trial's class", **E.summarize(r))

# =====================================================================
# 4. WHO, NOT WHAT. How much subject identity is in these same features?
#    Multi-class subject ID, trained on two runs and tested on the third,
#    so it cannot rely on within-run continuity.
# =====================================================================
def subject_id_decode(Cov, sub, run, label):
    ts = Pipeline([("ts", TangentSpace(metric="logeuclid")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=1.0, max_iter=1500))])
    runs = np.unique(run)
    accs = []
    for held in runs:
        tr, te = run != held, run == held
        mdl = ts.fit(Cov[tr], sub[tr])
        accs.append((mdl.predict(Cov[te]) == sub[te]).mean())
    acc = float(np.mean(accs))
    n_cls = len(np.unique(sub))
    rec("identity", label, pooled_acc=acc, pooled_lo=acc, pooled_hi=acc,
        chance=1.0 / n_cls, n_classes=n_cls)


subject_id_decode(Cov, sub, run, f"subject ID from MI trials ({len(np.unique(sub))}-way)")

# same features, but from the eyes-open baseline run where nobody is doing anything
base = D.load_epochs  # epoch run 1 into fixed 4 s windows
import mne
Xb, sb, rb = [], [], []
for s in np.unique(sub):
    try:
        raw = D._read_run(int(s), 1)
        if abs(raw.info["sfreq"] - 160) > 1e-6:
            raw.resample(160.0, verbose="ERROR")
        raw.filter(CFG.BAND[0], CFG.BAND[1], method="iir",
                   iir_params=dict(order=4, ftype="butter"), verbose="ERROR")
        ep = mne.make_fixed_length_epochs(raw, duration=3.0, overlap=0.0,
                                          preload=True, verbose="ERROR")
        d = ep.get_data(copy=True)[:, :64, :480]
        Xb.append(d); sb.append(np.full(len(d), s))
        rb.append(np.arange(len(d)) % 3)          # pseudo-runs: split the rest run in 3
    except Exception:
        pass
Xb = np.concatenate(Xb); sb = np.concatenate(sb); rb = np.concatenate(rb)
subject_id_decode(models.precompute_cov(Xb), sb, rb,
                  f"subject ID from EYES-OPEN REST ({len(np.unique(sb))}-way)")

# =====================================================================
# 5. CROSS-PARADIGM. Does a model trained on executed movement transfer
#    to imagined movement?
# =====================================================================
ei = cached.load_pool("imagined"); ee = cached.load_pool("executed")
Xi, _, _ = cached.prepare(ei, band=CFG.BAND, window=CFG.WINDOW)
Xe, _, _ = cached.prepare(ee, band=CFG.BAND, window=CFG.WINDOW)
Ci = models.precompute_cov(align(Xi, ei.subject))
Ce = models.precompute_cov(align(Xe, ee.subject))

for name, (Ctr, etr), (Cte, ete) in [
        ("train EXECUTED -> test IMAGINED", (Ce, ee), (Ci, ei)),
        ("train IMAGINED -> test EXECUTED", (Ci, ei), (Ce, ee))]:
    accs, ns = [], []
    for s in CFG.EVAL_SUBJECTS:                  # still subject-disjoint
        trm = etr.subject != s
        tem = ete.subject == s
        if tem.sum() == 0:
            continue
        mdl = MAKE().fit(Ctr[trm], etr.y[trm])
        p = mdl.predict(Cte[tem])
        accs.append((p == ete.y[tem]).mean()); ns.append(tem.sum())
    acc = float(np.average(accs, weights=ns))
    lo, hi = E.binom_ci(round(acc * sum(ns)), sum(ns))
    rec("cross-paradigm", name, pooled_acc=acc, pooled_lo=lo, pooled_hi=hi,
        mean_sub_acc=float(np.mean(accs)), sd_sub_acc=float(np.std(accs)),
        n_subjects=len(accs), n_trials=int(sum(ns)))

# within-subject cross-paradigm: same person, train on their executed trials
accs, ns = [], []
for s in CFG.EVAL_SUBJECTS:
    trm, tem = ee.subject == s, ei.subject == s
    if trm.sum() < 20 or tem.sum() < 20:
        continue
    mdl = MAKE().fit(Ce[trm], ee.y[trm])
    p = mdl.predict(Ci[tem])
    accs.append((p == ei.y[tem]).mean()); ns.append(tem.sum())
acc = float(np.average(accs, weights=ns))
lo, hi = E.binom_ci(round(acc * sum(ns)), sum(ns))
rec("cross-paradigm", "SAME subject: executed -> imagined", pooled_acc=acc,
    pooled_lo=lo, pooled_hi=hi, mean_sub_acc=float(np.mean(accs)),
    sd_sub_acc=float(np.std(accs)), n_subjects=len(accs), n_trials=int(sum(ns)))


# =====================================================================
# 6. HOW MUCH DOES THE ALIGNMENT KNOW? The alignment above is estimated
#    from all of a test subject's trials, including the ones being
#    predicted. Realistic deployment would estimate it from a short
#    calibration block. Redo it using only the subject's FIRST run.
# =====================================================================
def align_first_run_only(X, subject, run):
    out = np.empty_like(X)
    ea = models.EuclideanAlign()
    for u in np.unique(subject):
        m = subject == u
        first = np.unique(run[m])[0]
        cal = X[m & (run == first)]
        C = np.einsum("nct,ndt->cd", cal.astype(np.float64),
                      cal.astype(np.float64)) / (len(cal) * cal.shape[-1])
        C += 1e-10 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        w, V = np.linalg.eigh(C)
        R = (V @ np.diag(w ** -0.5) @ V.T).astype(X.dtype)
        out[m] = np.einsum("cd,ndt->nct", R, X[m])
    return out


Xf = align_first_run_only(X, sub, run)
Covf = models.precompute_cov(Xf)
later = run != np.array([np.unique(run[sub == s_])[0] for s_ in sub])
r = E.loso(Covf[later], y[later], sub[later], run[later], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=5)
rec("align-scope", "EA from 1st run only, tested on runs 2-3", **E.summarize(r))
r = E.loso(Cov[later], y[later], sub[later], run[later], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=5)
rec("align-scope", "EA from all 3 runs, tested on runs 2-3", **E.summarize(r))
del Xf, Covf

# =====================================================================
# 7. IS THERE RUN-SPECIFIC NUISANCE AT ALL? If a classifier can tell which
#    of a subject's three runs a trial came from, then random CV within a
#    subject is sharing run-specific structure between train and test.
# =====================================================================
accs = []
for s_ in CFG.EVAL_SUBJECTS:
    m = sub == s_
    if m.sum() < 30:
        continue
    from sklearn.model_selection import StratifiedKFold
    Cs, rs = Cov[m], run[m]
    pred = np.zeros(len(rs))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(Cs, rs):
        mdl = Pipeline([("ts", TangentSpace(metric="logeuclid")),
                        ("sc", StandardScaler()),
                        ("clf", LogisticRegression(C=0.1, max_iter=2000))])
        mdl.fit(Cs[tr_i], rs[tr_i])
        pred[te_i] = mdl.predict(Cs[te_i])
    accs.append((pred == rs).mean())
rec("run-identity", "which of this subject's 3 runs is it? (3-way)",
    pooled_acc=float(np.mean(accs)), pooled_lo=float(np.mean(accs)),
    pooled_hi=float(np.mean(accs)), chance=1 / 3, n_subjects=len(accs))

print("\nwrote", OUT)
