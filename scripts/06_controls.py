"""Controls: what else could explain the number?

Each block is a way for the headline accuracy to look real and be wrong.
All of them use the selected pipeline (src/chosen.py) unless stated.
"""
import sys, pathlib, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace
import cached, models, chosen, data as D, evaluation as E, config as CFG

OUT = pathlib.Path("results/controls.csv")
rows = []
MAKE = chosen.make
JOBS = 6


def rec(block, tag, **kw):
    rows.append(dict(block=block, tag=tag, **kw))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[{block}] {tag:<46} acc={kw.get('pooled_acc', np.nan):.4f} "
          f"[{kw.get('pooled_lo', np.nan):.3f},{kw.get('pooled_hi', np.nan):.3f}]",
          flush=True)


es = cached.load_pool(CFG.PARADIGM)
y, sub, run = es.y, es.subject, es.run

# =====================================================================
# 1. WINDOW PLACEBO. The 4.2 s before each cue is rest. If the pipeline
#    decodes the upcoming label from it, whatever it is decoding is not
#    the task. The -2.4..-0.4 s variant leaves a wider guard band, because
#    the band-pass is zero-phase and can smear energy backwards in time.
# =====================================================================
for tag, win in [("task window 0.5-3.5 s", (0.5, 3.5)),
                 ("PRE-CUE -2.0 to -0.2 s", (-2.0, -0.2)),
                 ("PRE-CUE -2.4 to -0.4 s", (-2.4, -0.4)),
                 ("late window 2.5-4.5 s", (2.5, 4.5))]:
    X, _, _ = cached.prepare(es, band=CFG.BAND, window=win)
    Xa = chosen.featurize(X, sub); del X; gc.collect()
    r = E.loso(Xa, y, sub, run, MAKE, subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
    rec("window", f"LOSO  {tag}", **E.summarize(r))
    m = np.isin(sub, CFG.EVAL_SUBJECTS)
    r = E.within_subject(Xa[m], y[m], sub[m], run[m], MAKE, split="random", n_jobs=JOBS)
    rec("window", f"within/randomCV  {tag}", **E.summarize(r))
    del Xa; gc.collect()

# =====================================================================
# 2. CHANNEL LESION. If a classifier restricted to occipital or frontal
#    electrodes does as well as one on the sensorimotor strip, we are not
#    decoding sensorimotor rhythm.
# =====================================================================
for tag, picks in [("all 64 channels", None),
                   ("sensorimotor strip", D.MOTOR_CH),
                   ("C3/Cz/C4 only", ["C3", "Cz", "C4"]),
                   ("frontal", D.FRONTAL_CH),
                   ("parieto-occipital", D.OCCIPITAL_CH)]:
    Xp, cp, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW, picks=picks)
    n_ch = Xp.shape[1]
    k = min(chosen.N_COMPONENTS, max(2, (n_ch // 2) * 2 - 2))
    Xa = chosen.featurize(Xp, sub); del Xp; gc.collect()
    r = E.loso(Xa, y, sub, run, lambda k=k: models.covcsp_lda(k, chosen.SHRINKAGE),
               subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
    rec("lesion", f"{tag} [{n_ch} ch]", **E.summarize(r))
    del Xa; gc.collect()

# =====================================================================
# 3. LABEL CARRY-OVER. Give each trial the label of the PREVIOUS trial in
#    the same run. Above chance means trials are not independent.
# =====================================================================
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
Xa = chosen.featurize(X, sub)
y_prev = y.copy(); keep = np.ones(len(y), bool)
for s in np.unique(sub):
    for r_ in np.unique(run[sub == s]):
        idx = np.where((sub == s) & (run == r_))[0]
        idx = idx[np.argsort(es.trial[idx])]
        y_prev[idx[1:]] = y[idx[:-1]]
        keep[idx[0]] = False
r = E.loso(Xa[keep], y_prev[keep], sub[keep], run[keep], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
rec("carryover", "LOSO, label = PREVIOUS trial's class", **E.summarize(r))

# =====================================================================
# 4. WHO, NOT WHAT. How much subject identity is in these same signals?
#    Probed with a tangent-space multinomial classifier rather than CSP,
#    because this is a question about information content, not the model.
#    Trained on two runs and tested on the third, so it cannot lean on
#    within-run continuity.
# =====================================================================
def subject_id_decode(Cov, s_, r_, label):
    ts = Pipeline([("ts", TangentSpace(metric="logeuclid")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=1.0, max_iter=1500))])
    accs = []
    for held in np.unique(r_):
        tr, te = r_ != held, r_ == held
        accs.append((ts.fit(Cov[tr], s_[tr]).predict(Cov[te]) == s_[te]).mean())
    acc = float(np.mean(accs))
    rec("identity", label, pooled_acc=acc, pooled_lo=acc, pooled_hi=acc,
        chance=1.0 / len(np.unique(s_)), n_classes=len(np.unique(s_)))


Cov_raw = models.precompute_cov(X)
subject_id_decode(Cov_raw, sub, run,
                  f"subject ID from MI trials, no alignment ({len(np.unique(sub))}-way)")
del Cov_raw; gc.collect()
Cov_al = Xa
subject_id_decode(Cov_al, sub, run,
                  f"subject ID from MI trials, after alignment ({len(np.unique(sub))}-way)")

# same probe on the eyes-open baseline run, where nobody is doing anything
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
        d = ep.get_data(copy=True)[:, :64, :480].astype(np.float32)
        Xb.append(d); sb.append(np.full(len(d), s)); rb.append(np.arange(len(d)) % 3)
    except Exception:
        pass
Xb = np.concatenate(Xb); sb = np.concatenate(sb); rb = np.concatenate(rb)
subject_id_decode(models.precompute_cov(Xb), sb, rb,
                  f"subject ID from EYES-OPEN REST ({len(np.unique(sb))}-way)")
del Xb; gc.collect()

# =====================================================================
# 5. IS THERE RUN-SPECIFIC NUISANCE AT ALL? If a classifier can tell which
#    of a subject's three runs a trial came from, then random CV within a
#    subject shares run-specific structure between train and test.
# =====================================================================
from sklearn.model_selection import StratifiedKFold
accs = []
for s in CFG.EVAL_SUBJECTS:
    m = sub == s
    if m.sum() < 30 or len(np.unique(run[m])) < 3:
        continue
    Cs, rs = Cov_al[m], run[m]
    pred = np.zeros(len(rs))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(Cs, rs):
        mdl = Pipeline([("ts", TangentSpace(metric="logeuclid")),
                        ("sc", StandardScaler()),
                        ("clf", LogisticRegression(C=0.1, max_iter=2000))])
        pred[te_i] = mdl.fit(Cs[tr_i], rs[tr_i]).predict(Cs[te_i])
    accs.append((pred == rs).mean())
rec("run-identity", "which of this subject's 3 runs is it? (3-way)",
    pooled_acc=float(np.mean(accs)), pooled_lo=float(np.mean(accs)),
    pooled_hi=float(np.mean(accs)), chance=1 / 3, n_subjects=len(accs))
del Cov_al; gc.collect()

# =====================================================================
# 6. HOW MUCH DOES THE ALIGNMENT KNOW? The alignment is estimated from all
#    of a test subject's trials, including the ones being predicted.
#    Realistic deployment would estimate it from a calibration block.
# =====================================================================
def align_first_run_only(X, subject, run):
    out = np.empty_like(X)
    for u in np.unique(subject):
        m = subject == u
        cal = X[m & (run == np.unique(run[m])[0])].astype(np.float64)
        C = np.einsum("nct,ndt->cd", cal, cal) / (len(cal) * cal.shape[-1])
        C /= np.trace(C) / C.shape[0]
        w, V = np.linalg.eigh(C)
        R = (V @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ V.T).astype(X.dtype)
        out[m] = np.einsum("cd,ndt->nct", R, X[m])
    return out


first_of = np.array([np.unique(run[sub == s])[0] for s in np.unique(sub)])
first_map = dict(zip(np.unique(sub), first_of))
later = np.array([run[i] != first_map[sub[i]] for i in range(len(run))])

Xf = models.precompute_cov(align_first_run_only(X, sub, run))
r = E.loso(Xf[later], y[later], sub[later], run[later], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
rec("align-scope", "align from 1st run only, test on runs 2-3", **E.summarize(r))
del Xf; gc.collect()
r = E.loso(Xa[later], y[later], sub[later], run[later], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
rec("align-scope", "align from all 3 runs, test on runs 2-3", **E.summarize(r))
Xn = models.precompute_cov(X)
r = E.loso(Xn[later], y[later], sub[later], run[later], MAKE,
           subjects=CFG.EVAL_SUBJECTS, n_jobs=JOBS)
rec("align-scope", "no alignment at all, test on runs 2-3", **E.summarize(r))
del X, Xa, Xn; gc.collect()

# =====================================================================
# 7. CROSS-PARADIGM. Does a model trained on executed movement transfer
#    to imagined movement?
# =====================================================================
ei, ee = cached.load_pool("imagined"), cached.load_pool("executed")
Xi, _, _ = cached.prepare(ei, band=CFG.BAND, window=CFG.WINDOW)
Xe, _, _ = cached.prepare(ee, band=CFG.BAND, window=CFG.WINDOW)
Xi = chosen.featurize(Xi, ei.subject); Xe = chosen.featurize(Xe, ee.subject)

for name, (Atr, etr), (Ate, ete) in [
        ("train EXECUTED -> test IMAGINED", (Xe, ee), (Xi, ei)),
        ("train IMAGINED -> test EXECUTED", (Xi, ei), (Xe, ee))]:
    accs, ns = [], []
    for s in CFG.EVAL_SUBJECTS:
        trm, tem = etr.subject != s, ete.subject == s
        if tem.sum() == 0:
            continue
        p = MAKE().fit(Atr[trm], etr.y[trm]).predict(Ate[tem])
        accs.append((p == ete.y[tem]).mean()); ns.append(int(tem.sum()))
    acc = float(np.average(accs, weights=ns))
    lo, hi = E.binom_ci(round(acc * sum(ns)), sum(ns))
    rec("cross-paradigm", name, pooled_acc=acc, pooled_lo=lo, pooled_hi=hi,
        mean_sub_acc=float(np.mean(accs)), sd_sub_acc=float(np.std(accs)),
        n_subjects=len(accs), n_trials=int(sum(ns)))

accs, ns = [], []
for s in CFG.EVAL_SUBJECTS:
    trm, tem = ee.subject == s, ei.subject == s
    if trm.sum() < 20 or tem.sum() < 20:
        continue
    p = MAKE().fit(Xe[trm], ee.y[trm]).predict(Xi[tem])
    accs.append((p == ei.y[tem]).mean()); ns.append(int(tem.sum()))
acc = float(np.average(accs, weights=ns))
lo, hi = E.binom_ci(round(acc * sum(ns)), sum(ns))
rec("cross-paradigm", "SAME subject: executed -> imagined", pooled_acc=acc,
    pooled_lo=lo, pooled_hi=hi, mean_sub_acc=float(np.mean(accs)),
    sd_sub_acc=float(np.std(accs)), n_subjects=len(accs), n_trials=int(sum(ns)))

print("\nwrote", OUT)
