"""All remaining figures, built from the result CSVs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path("results"); F = pathlib.Path("figures"); F.mkdir(exist_ok=True)
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#a0aec0"

# ---------------- fig2: the regime ladder ---------------------------------
mr = pd.read_csv(R / "main_regimes.csv")
main = mr[mr.regime.str.startswith(("A", "B", "C", "D", "E"))]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
for ax, par in zip(axes, ["imagined", "executed"]):
    g = main[main.paradigm == par]
    real = g[~g.shuffled].sort_values("regime")
    shuf = g[g.shuffled].sort_values("regime")
    ypos = np.arange(len(real))
    ax.barh(ypos + .19, real.pooled_acc - .5, left=.5, height=.36, color=BLUE,
            label="real labels")
    ax.errorbar(real.pooled_acc, ypos + .19,
                xerr=[real.pooled_acc - real.pooled_lo, real.pooled_hi - real.pooled_acc],
                fmt="none", ecolor="black", capsize=2.5, lw=1)
    ax.barh(ypos - .19, shuf.pooled_acc - .5, left=.5, height=.36, color=GREY,
            label="labels shuffled")
    ax.errorbar(shuf.pooled_acc, ypos - .19,
                xerr=[shuf.pooled_acc - shuf.pooled_lo, shuf.pooled_hi - shuf.pooled_acc],
                fmt="none", ecolor="black", capsize=2.5, lw=1)
    ax.axvline(.5, color="black", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[2:] for r in real.regime], fontsize=9)
    ax.set_xlim(.44, .80); ax.set_xlabel("pooled accuracy")
    ax.set_title(par, fontsize=11)
    ax.invert_yaxis()
axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.16),
               ncol=2, frameon=False)
fig.suptitle("One model, four definitions of 'held out'.  "
             "Grey = the same run with labels permuted inside each block.",
             fontsize=11)
fig.tight_layout(); fig.savefig(F / "fig2_regime_ladder.png", dpi=150)

# ---------------- fig3: per-subject spread --------------------------------
ps = pd.read_csv(R / "per_subject_rows.csv")
ps = ps[(~ps.shuffled) & (ps.paradigm == "imagined")]
order = sorted(ps.regime.unique())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                         gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
data = [ps[ps.regime == r].acc.values for r in order]
parts = ax.violinplot(data, showmeans=True, widths=.85)
for b in parts["bodies"]:
    b.set_facecolor(BLUE); b.set_alpha(.55)
for i, d in enumerate(data):
    ax.scatter(np.random.normal(i + 1, .055, len(d)), d, s=7, color="#1a365d", alpha=.55)
ax.axhline(.5, color=RED, ls=":")
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels([r[:1] for r in order])
ax.set_ylabel("per-subject accuracy")
ax.set_title("Every subject, every regime (imagined)\n"
             + "   ".join(f"{r[:1]}={r[2:]}" for r in order), fontsize=7.5)

ax = axes[1]
w = ps.pivot_table(index="subject", columns="regime", values="acc")
a = [r for r in order if r.startswith("A")][0]
b = [r for r in order if r.startswith("C")][0]
ax.scatter(w[a], w[b], s=26, color=BLUE, alpha=.8)
lim = [min(w[a].min(), w[b].min()) - .02, max(w[a].max(), w[b].max()) + .02]
ax.plot(lim, lim, color="black", lw=1); ax.axhline(.5, color=GREY, ls=":")
ax.axvline(.5, color=GREY, ls=":")
ax.set_xlabel(a[2:]); ax.set_ylabel(b[2:])
rho = w[[a, b]].corr(method="spearman").iloc[0, 1]
ax.set_title(f"same subjects, two regimes (Spearman rho = {rho:.2f})", fontsize=10)
fig.tight_layout(); fig.savefig(F / "fig3_per_subject.png", dpi=150)

# ---------------- fig7: calibration curve ---------------------------------
cal = mr[mr.regime.str.startswith("F")].copy()
if len(cal):
    cal["k"] = cal.regime.str.extract(r"(\d+) calib").astype(int)
    cal = cal.sort_values("k")
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.fill_between(cal.k, cal.pooled_lo, cal.pooled_hi, alpha=.25, color=BLUE)
    ax.plot(cal.k, cal.pooled_acc, "o-", color=BLUE)
    wr = mr[(mr.regime.str.startswith("B")) & (~mr.shuffled)
            & (mr.paradigm == "imagined")].pooled_acc.head(1)
    if len(wr):
        ax.axhline(float(wr.iloc[0]), color=RED, ls="--", lw=1.2,
                   label="within-subject, leave-run-out")
    ax.axhline(.5, color=GREY, ls=":")
    ax.set_xlabel("labelled calibration trials from the new subject")
    ax.set_ylabel("accuracy on their remaining trials")
    ax.set_title("How many labels does a new person cost?", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(F / "fig7_calibration.png", dpi=150)

# ---------------- fig8: deep net -----------------------------------------
dp = R / "deep.csv"
if dp.is_file():
    d = pd.read_csv(dp)
    cur = R / "deep_curves.csv"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    scal = d[d.tag.str.contains("training subjects")].copy()
    if len(scal):
        scal["n"] = scal.tag.str.extract(r"(\d+) training").astype(int)
        scal = scal.sort_values("n")
        ax.errorbar(scal.n, scal.pooled_acc,
                    yerr=[scal.pooled_acc - scal.pooled_lo,
                          scal.pooled_hi - scal.pooled_acc],
                    fmt="o-", color=BLUE, capsize=3)
        ax.axhline(.5, color=GREY, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("training subjects")
        ax.set_ylabel("cross-subject accuracy")
        ax.set_title("EEGNet: does more people help?", fontsize=10)
    ax = axes[1]
    if cur.is_file():
        c = pd.read_csv(cur).groupby("epoch")[["train_acc", "val_acc"]].mean()
        ax.plot(c.index, c.train_acc, color=RED, label="train")
        ax.plot(c.index, c.val_acc, color=BLUE, label="held-out subjects")
        ax.axhline(.5, color=GREY, ls=":")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(fontsize=9)
        ax.set_title("where it overfits", fontsize=10)
    ax = axes[2]
    sel = d[~d.tag.str.contains("training subjects")]
    ax.barh(np.arange(len(sel)), sel.pooled_acc - .5, left=.5, color=BLUE)
    ax.errorbar(sel.pooled_acc, np.arange(len(sel)),
                xerr=[sel.pooled_acc - sel.pooled_lo, sel.pooled_hi - sel.pooled_acc],
                fmt="none", ecolor="black", capsize=2.5)
    ax.scatter(sel.train_acc, np.arange(len(sel)), marker="|", s=260, color=RED,
               label="train accuracy")
    ax.axvline(.5, color="black", lw=1)
    ax.set_yticks(np.arange(len(sel)))
    ax.set_yticklabels([t.replace("EEGNet ", "") for t in sel.tag], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("accuracy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig8_deep.png", dpi=150)

# ---------------- fig9: controls ------------------------------------------
cp = R / "controls.csv"
if cp.is_file():
    c = pd.read_csv(cp)
    blocks = [("window", "analysis window"), ("lesion", "electrode subset"),
              ("early-window", "where the early advantage lives")]
    blocks = [b for b in blocks if (c.block == b[0]).any()]
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.0 * len(blocks), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (blk, ttl) in zip(axes, blocks):
        g = c[c.block == blk]
        ax.barh(np.arange(len(g)), g.pooled_acc - .5, left=.5,
                color=[RED if "PRE-CUE" in t else BLUE for t in g.tag])
        ax.errorbar(g.pooled_acc, np.arange(len(g)),
                    xerr=[g.pooled_acc - g.pooled_lo, g.pooled_hi - g.pooled_acc],
                    fmt="none", ecolor="black", capsize=2.5)
        ax.axvline(.5, color="black", lw=1)
        ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.tag, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("accuracy"); ax.set_title(ttl, fontsize=11)
    fig.tight_layout(); fig.savefig(F / "fig9_controls.png", dpi=150)

# ---------------- fig4: time-resolved (rebuilt from the CSV) --------------
tp = R / "time_resolved.csv"
if tp.is_file():
    t = pd.read_csv(tp)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.axvspan(t.centre_s.min() - .2, 0, color="#9aa5b1", alpha=.12)
    ax.fill_between(t.centre_s, t.pooled_lo, t.pooled_hi, alpha=.25, color=BLUE)
    ax.plot(t.centre_s, t.pooled_acc, color=BLUE, lw=2)
    ax.axhline(.5, color="#666", ls=":", lw=1)
    ax.axvline(0, color=RED, lw=1.2)
    pk = t.loc[t.pooled_acc.idxmax()]
    ax.plot([pk.centre_s], [pk.pooled_acc], "o", color=RED, ms=5, zorder=5)
    ax.annotate(f"{pk.pooled_acc*100:.1f}% at {pk.centre_s:+.2f} s",
                (pk.centre_s, pk.pooled_acc), textcoords="offset points",
                xytext=(10, 6), fontsize=9, color=RED)
    ax.text(t.centre_s.min(), ax.get_ylim()[1], " rest, before the cue", ha="left",
            va="top", fontsize=9, color="#555")
    ax.text(0.07, .502, "cue", color=RED, fontsize=9)
    ax.set_xlabel("centre of 0.75 s window, seconds from cue")
    ax.set_ylabel("cross-subject accuracy")
    ax.set_title("Where in the trial the information is\n"
                 "leave-one-subject-out at every window position, 58 evaluation subjects",
                 fontsize=10)
    ax.set_xlim(t.centre_s.min() - .15, t.centre_s.max() + .15)
    fig.tight_layout(); fig.savefig(F / "fig4_time_resolved.png", dpi=150)

print("figures written to", F)
