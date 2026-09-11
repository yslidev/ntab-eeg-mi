"""EEGNet: does a convolutional network do better, and where does it overfit?

Cross-subject folds here are groups of 5 EVAL subjects rather than one at a
time, purely for compute. Training sets still exclude every test subject.
"""
import sys, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import cached, models, deep, chosen, evaluation as E, config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=70)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--out", default="results/deep.csv")
args = ap.parse_args()

rows = []
def rec(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{kw['tag']:<42} test={kw['pooled_acc']:.4f} "
          f"train={kw.get('train_acc', float('nan')):.3f} {kw['secs']:.0f}s", flush=True)

es = cached.load_pool(CFG.PARADIGM)
# 80 Hz is ample for an 8-30 Hz phenomenon and halves the dominant cost, which
# is the temporal convolution running at full 64-channel spatial resolution.
X, chs, _ = cached.prepare(es, band=(4.0, 38.0), window=CFG.WINDOW)
# Same per-recording whitening the classical pipeline uses, so the comparison
# is about the model and not about the preprocessing.
X = chosen.align(X, es.subject)
X = np.ascontiguousarray(X[..., ::2], dtype=np.float32)
print(f"EEGNet input: {X.shape} at 80 Hz, Euclidean-aligned")
y, sub = es.y, es.subject
rng = np.random.default_rng(CFG.SEED)

GROUPS = np.array_split(CFG.EVAL_SUBJECTS, args.folds)


def cross_subject(train_pool, tag, epochs, seed=0, n_groups=None):
    """train_pool: subject ids allowed in training (validation carved from it)."""
    t0 = time.time()
    per_sub, tr_accs, hists = [], [], []
    for g in (GROUPS if n_groups is None else GROUPS[:n_groups]):
        te = np.isin(sub, g)
        pool = np.array([s for s in train_pool if s not in g])
        if len(pool) < 6:          # need enough to carve a validation set
            continue
        r = np.random.default_rng(seed + int(g[0]))
        va_s = r.choice(pool, size=max(2, min(len(pool) // 5, len(pool) - 4)),
                        replace=False)
        tr = np.isin(sub, [s for s in pool if s not in va_s])
        va = np.isin(sub, va_s)
        pred, info = deep.train_eval(X[tr], y[tr], X[te], y[te], X[va], y[va],
                                     epochs=epochs, seed=seed)
        tr_accs.append(info["train_acc"]); hists.append(info["hist"])
        for s in g:
            m = sub[te] == s
            if m.sum() == 0:
                continue
            k = int((pred[m] == y[te][m]).sum()); n = int(m.sum())
            lo, hi = E.binom_ci(k, n)
            per_sub.append(dict(subject=int(s), n=n, acc=k / n, ci_lo=lo, ci_hi=hi,
                                sig_thresh=E.binom_sig_threshold(n)))
    s_ = E.summarize(per_sub)
    rec(tag=tag, train_acc=float(np.mean(tr_accs)), secs=time.time() - t0,
        n_train_subjects=len(train_pool), **s_)
    return per_sub, hists


# 1. the headline cross-subject number, all available training subjects
ps, hists = cross_subject(CFG.ALL_SUBJECTS, "EEGNet cross-subject (all train subj)",
                          args.epochs)
pd.DataFrame(ps).to_csv("results/deep_per_subject.csv", index=False)
pd.DataFrame([h for hh in hists for h in hh]).to_csv("results/deep_curves.csv", index=False)

# 2. shuffled labels, same machinery -- must land on chance
ys = y.copy()
for s in np.unique(sub):
    m = sub == s
    ys[m] = rng.permutation(ys[m])
y_real, y = y, ys
cross_subject(CFG.ALL_SUBJECTS, "EEGNet cross-subject, SHUFFLED labels", args.epochs, n_groups=3)
y = y_real

# 3. how much does it need? accuracy vs number of training subjects
for n_tr in [8, 24, 64]:
    pool = CFG.DEV_SUBJECTS if n_tr <= len(CFG.DEV_SUBJECTS) else CFG.ALL_SUBJECTS
    pool = rng.choice(CFG.ALL_SUBJECTS, size=min(n_tr, len(CFG.ALL_SUBJECTS)),
                      replace=False)
    cross_subject(pool, f"EEGNet, {n_tr} training subjects", args.epochs, n_groups=3)

# 4. within-subject: ~43 trials per person. This is where it memorises.
t0 = time.time()
from sklearn.model_selection import StratifiedKFold
per_sub, tr_accs = [], []
for s in CFG.EVAL_SUBJECTS[:12]:
    m = sub == s
    Xs, ys_ = X[m], y[m]
    pred = np.zeros(len(ys_), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys_):
        p, info = deep.train_eval(Xs[tr], ys_[tr], Xs[te], ys_[te], epochs=50)
        pred[te] = p; tr_accs.append(info["train_acc"])
    k = int((pred == ys_).sum()); n = len(ys_)
    lo, hi = E.binom_ci(k, n)
    per_sub.append(dict(subject=int(s), n=n, acc=k / n, ci_lo=lo, ci_hi=hi,
                        sig_thresh=E.binom_sig_threshold(n)))
rec(tag="EEGNet within-subject (random CV, 12 subj)",
    train_acc=float(np.mean(tr_accs)), secs=time.time() - t0,
    n_train_subjects=1, **E.summarize(per_sub))

print("\nwrote", args.out)
