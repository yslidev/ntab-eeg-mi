"""The two EEGNet configurations that scripts/07_deep.py crashed before reaching:
how accuracy scales with the number of training subjects, and what happens
within a single subject with 43 trials. Appends to results/deep.csv.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import cached, deep, chosen, evaluation as E, config as CFG

OUT = pathlib.Path("results/deep.csv")
rows = pd.read_csv(OUT).to_dict("records") if OUT.is_file() else []
EPOCHS = 90


def rec(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{kw['tag']:<44} test={kw['pooled_acc']:.4f} "
          f"train={kw.get('train_acc', float('nan')):.3f} {kw['secs']:.0f}s", flush=True)


es = cached.load_pool(CFG.PARADIGM)
X, chs, _ = cached.prepare(es, band=(4.0, 38.0), window=CFG.WINDOW)
X = np.ascontiguousarray(chosen.align(X, es.subject)[..., ::2], dtype=np.float32)
y, sub = es.y, es.subject
rng = np.random.default_rng(CFG.SEED)
GROUPS = np.array_split(CFG.EVAL_SUBJECTS, 3)

# --- accuracy against number of training subjects --------------------------
for n_tr in [8, 24, 64]:
    t0 = time.time()
    pool = rng.choice(np.setdiff1d(CFG.ALL_SUBJECTS, CFG.EVAL_SUBJECTS[:20]),
                      size=n_tr, replace=False)
    va_s = pool[: max(2, n_tr // 6)]
    tr_s = pool[max(2, n_tr // 6):]
    te = np.isin(sub, CFG.EVAL_SUBJECTS[:20])
    tr, va = np.isin(sub, tr_s), np.isin(sub, va_s)
    pred, info = deep.train_eval(X[tr], y[tr], X[te], y[te], X[va], y[va],
                                 epochs=EPOCHS)
    per = []
    for s in CFG.EVAL_SUBJECTS[:20]:
        m = sub[te] == s
        if m.sum() == 0:
            continue
        k, n = int((pred[m] == y[te][m]).sum()), int(m.sum())
        lo, hi = E.binom_ci(k, n)
        per.append(dict(subject=int(s), n=n, acc=k / n, ci_lo=lo, ci_hi=hi,
                        sig_thresh=E.binom_sig_threshold(n)))
    rec(tag=f"EEGNet, {n_tr} training subjects", train_acc=info["train_acc"],
        secs=time.time() - t0, n_train_subjects=int(len(tr_s)),
        **E.summarize(per))

# --- within one subject: 43 trials, where memorisation is unmissable -------
t0 = time.time()
per, tr_accs = [], []
for s in CFG.EVAL_SUBJECTS[:12]:
    m = sub == s
    Xs, ys_ = X[m], y[m]
    pred = np.zeros(len(ys_), int)
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys_):
        p, info = deep.train_eval(Xs[tr_i], ys_[tr_i], Xs[te_i], ys_[te_i], epochs=60)
        pred[te_i] = p; tr_accs.append(info["train_acc"])
    k, n = int((pred == ys_).sum()), len(ys_)
    lo, hi = E.binom_ci(k, n)
    per.append(dict(subject=int(s), n=n, acc=k / n, ci_lo=lo, ci_hi=hi,
                    sig_thresh=E.binom_sig_threshold(n)))
rec(tag="EEGNet within-subject (random CV, 12 subj)",
    train_acc=float(np.mean(tr_accs)), secs=time.time() - t0,
    n_train_subjects=1, **E.summarize(per))

print("\nwrote", OUT)
