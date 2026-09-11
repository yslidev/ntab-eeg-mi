"""Is the model's error a per-trial mistake or a per-person offset?

If a cross-subject classifier were reading the task, its predictions for any
one subject would be roughly balanced between the two classes, because the
cues are balanced. If instead it is partly reading who the person is, its
decision function will sit off-centre for that person, and their predictions
will lean one way regardless of the cue.

That is measurable, and the fix is measurable too: re-centre each subject's
decision values on their own median. Class counts are near-balanced by design,
so a median split uses no labels -- it is legal on unlabelled data.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cached, models, chosen, evaluation as E, config as CFG

es = cached.load_pool(CFG.PARADIGM)
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
Cov = chosen.featurize(X, es.subject)
del X
y, sub = es.y, es.subject


def one(s):
    tr, te = sub != s, sub == s
    mdl = clone(chosen.make()).fit(Cov[tr], y[tr])
    d = mdl.decision_function(Cov[te])
    return dict(subject=int(s), scores=d, y=y[te],
                run=es.run[te], trial=es.trial[te])


res = Parallel(n_jobs=5)(delayed(one)(s) for s in CFG.EVAL_SUBJECTS)

rows = []
for r in res:
    d, yy = r["scores"], r["y"]
    raw_pred = (d > 0).astype(int)
    cen_pred = (d > np.median(d)).astype(int)
    rows.append(dict(
        subject=r["subject"], n=len(yy),
        frac_pred_right=float(raw_pred.mean()),
        frac_true_right=float(yy.mean()),
        mean_score=float(d.mean()), sd_score=float(d.std()),
        acc_raw=float((raw_pred == yy).mean()),
        acc_centred=float((cen_pred == yy).mean()),
        auc=float(stats.rankdata(d)[yy == 1].mean() / len(d) if len(set(yy)) > 1 else np.nan)))
df = pd.DataFrame(rows)
df.to_csv("results/subject_bias.csv", index=False)

# per-trial record, for the non-stationarity analysis below
trial_rows = []
for r in res:
    for i in range(len(r["y"])):
        trial_rows.append(dict(subject=r["subject"], run=int(r["run"][i]),
                               trial=int(r["trial"][i]), y=int(r["y"][i]),
                               score=float(r["scores"][i]),
                               correct=int((r["scores"][i] > 0) == r["y"][i])))
tr_df = pd.DataFrame(trial_rows)
tr_df.to_csv("results/per_trial_predictions.csv", index=False)

n = df.n.sum()
acc_raw = float((df.acc_raw * df.n).sum() / n)
acc_cen = float((df.acc_centred * df.n).sum() / n)
lo1, hi1 = E.binom_ci(round(acc_raw * n), int(n))
lo2, hi2 = E.binom_ci(round(acc_cen * n), int(n))

print("=== per-subject prediction bias, cross-subject model ===")
print(f"subjects: {len(df)}, trials: {n}")
print(f"\ntrue fraction of 'right fist' cues per subject: "
      f"{df.frac_true_right.mean():.3f} ± {df.frac_true_right.std():.3f} "
      f"(range {df.frac_true_right.min():.2f}-{df.frac_true_right.max():.2f})")
print(f"model's predicted fraction of 'right'  per subject: "
      f"{df.frac_pred_right.mean():.3f} ± {df.frac_pred_right.std():.3f} "
      f"(range {df.frac_pred_right.min():.2f}-{df.frac_pred_right.max():.2f})")

# how much wider is the predicted spread than chance sampling would give?
exp_sd = float(np.sqrt(np.mean(0.25 / df.n)))
print(f"\nspread expected from sampling alone: sd {exp_sd:.3f}")
print(f"spread observed in predictions:      sd {df.frac_pred_right.std():.3f}"
      f"   ({df.frac_pred_right.std()/exp_sd:.1f}x)")
chi = ((df.frac_pred_right - 0.5) ** 2 * df.n / 0.25).sum()
print(f"chi-square for 'every subject is unbiased': {chi:.1f} on {len(df)} df, "
      f"p = {stats.chi2.sf(chi, len(df)):.2e}")

print(f"\naccuracy, decision threshold at 0:          {acc_raw*100:.2f}% "
      f"[{lo1*100:.1f}, {hi1*100:.1f}]")
print(f"accuracy, threshold at each subject's median: {acc_cen*100:.2f}% "
      f"[{lo2*100:.1f}, {hi2*100:.1f}]")
print(f"gain from removing the per-subject offset:   "
      f"{(acc_cen-acc_raw)*100:+.2f} points")
t, p = stats.wilcoxon(df.acc_centred, df.acc_raw)
print(f"paired Wilcoxon over subjects: W={t:.0f}, p={p:.2e}")
print(f"\nmean per-subject AUC: {df.auc.mean():.4f} "
      f"(AUC is threshold-free, so it already ignores the offset)")

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
ax = axes[0]
ax.hist(df.frac_pred_right, bins=16, color="#2b6cb0", alpha=.85,
        label="model's predictions")
ax.hist(df.frac_true_right, bins=16, color="#c53030", alpha=.55,
        label="actual cues")
ax.axvline(.5, color="black", ls=":")
ax.set_xlabel("fraction labelled 'right fist'"); ax.set_ylabel("subjects")
ax.legend(fontsize=8); ax.set_title("the model leans, per person", fontsize=10)

ax = axes[1]
ax.scatter(df.mean_score, df.acc_raw, s=26, color="#2b6cb0")
ax.axvline(0, color="black", ls=":"); ax.axhline(.5, color="#a0aec0", ls=":")
rho, pv = stats.spearmanr(df.mean_score.abs(), df.acc_raw)
ax.set_xlabel("mean decision value for that subject")
ax.set_ylabel("accuracy")
ax.set_title(f"|offset| vs accuracy: rho={rho:+.2f}, p={pv:.3f}", fontsize=9)

ax = axes[2]
ax.scatter(df.acc_raw, df.acc_centred, s=26, color="#2b6cb0")
lim = [min(df.acc_raw.min(), df.acc_centred.min()) - .03,
       max(df.acc_raw.max(), df.acc_centred.max()) + .03]
ax.plot(lim, lim, color="black", lw=1)
ax.set_xlabel("threshold at 0"); ax.set_ylabel("threshold at subject's median")
ax.set_title(f"pooled {acc_raw*100:.1f}% -> {acc_cen*100:.1f}%", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig10_subject_bias.png", dpi=150)
print("\nwrote figures/fig10_subject_bias.png")

# =====================================================================
# Does the signal hold up across the session? The three imagined runs are
# separated by other runs, so run 12 happens many minutes after run 4.
# =====================================================================
print("\n=== non-stationarity within a session ===")
g = tr_df.groupby("run").agg(n=("correct", "size"), k=("correct", "sum"))
g["acc"] = g.k / g.n
for r_, row in g.iterrows():
    lo, hi = E.binom_ci(int(row.k), int(row.n))
    print(f"run {int(r_):2d}: {row.acc*100:5.2f}%  [{lo*100:.1f}, {hi*100:.1f}]  "
          f"n={int(row.n)}")
from scipy.stats import chi2_contingency
tab = np.array([[row.k, row.n - row.k] for _, row in g.iterrows()])
chi2, pv, _, _ = chi2_contingency(tab)
print(f"difference across runs: chi2={chi2:.2f}, p={pv:.3f}")

print("\naccuracy by trial position within a run:")
tp = tr_df.groupby("trial").agg(n=("correct", "size"), k=("correct", "sum"))
tp["acc"] = tp.k / tp.n
print("  " + "  ".join(f"{int(i)}:{v*100:.0f}" for i, v in tp.acc.items()))
rho_, pv_ = stats.spearmanr(tr_df.trial, tr_df.correct)
print(f"trial position vs correctness: rho={rho_:+.4f}, p={pv_:.3f}, "
      f"n={len(tr_df)}")
