"""What is the model using, and when?

(a) Time-resolved decoding: slide a 1 s window across the whole epoch,
    including the rest period before the cue. Information should appear after
    the cue and not before. This is the pre-cue control drawn as a curve.
(b) Spatial origin: project the tangent-space classifier's weights back onto
    the scalp, and show the group CSP patterns next to them.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, mne
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyriemann.utils.mean import mean_covariance
import cached, models, evaluation as E, config as CFG

mne.set_log_level("ERROR")
es = cached.load_pool(CFG.PARADIGM)
SUBSET = CFG.EVAL_SUBJECTS[::2]          # every other EVAL subject, for compute


def aligned(X, subject):
    ea = models.EuclideanAlign()
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    return np.concatenate([ea.transform(X[subject == u]) for u in subs])[np.argsort(order)]


# ---------------- (a) time-resolved ---------------------------------------
centres = np.arange(-1.8, 4.1, 0.25)
rows = []
for c in centres:
    t0 = time.time()
    X, chs, _ = cached.prepare(es, band=CFG.BAND, window=(c - 0.5, c + 0.5))
    Cov = models.precompute_cov(aligned(X, es.subject))
    del X
    r = E.loso(Cov, es.y, es.subject, es.run, models.ts_lr_le,
               subjects=SUBSET, n_jobs=6)
    s = E.summarize(r)
    rows.append(dict(centre_s=float(c), **s))
    pd.DataFrame(rows).to_csv("results/time_resolved.csv", index=False)
    print(f"t={c:+.2f}s  acc={s['pooled_acc']:.4f} "
          f"[{s['pooled_lo']:.3f},{s['pooled_hi']:.3f}]  {time.time()-t0:.0f}s", flush=True)
    del Cov

tr = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.fill_between(tr.centre_s, tr.pooled_lo, tr.pooled_hi, alpha=.25, color="#2b6cb0")
ax.plot(tr.centre_s, tr.pooled_acc, color="#2b6cb0", lw=2)
ax.axhline(0.5, color="grey", ls=":")
ax.axvspan(-2.5, 0, color="#999", alpha=.13)
ax.axvline(0, color="#c53030", lw=1.2)
ax.text(-1.25, ax.get_ylim()[1] * 0.995, "rest (pre-cue)", ha="center",
        va="top", fontsize=9, color="#555")
ax.text(0.06, 0.5015, "cue", color="#c53030", fontsize=9)
ax.set_xlabel("centre of 1 s window, seconds from cue")
ax.set_ylabel("cross-subject accuracy")
ax.set_title("Time-resolved decoding (LOSO + alignment, every 2nd EVAL subject)",
             fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig4_time_resolved.png", dpi=150)
print("wrote figures/fig4_time_resolved.png")

# ---------------- (b) where on the scalp ----------------------------------
X, chs, _ = cached.prepare(es, band=CFG.BAND, window=CFG.WINDOW)
Xa = aligned(X, es.subject)
Cov = models.precompute_cov(Xa)
pipe = models.ts_lr_le().fit(Cov, es.y)

n = len(chs)
w = pipe.named_steps["clf"].coef_.ravel() / pipe.named_steps["sc"].scale_
W = np.zeros((n, n))
iu = np.triu_indices(n)
W[iu] = w
W = W + W.T - np.diag(np.diag(W))
chan_weight = np.abs(W).sum(0)                  # total weight touching each channel
diag_weight = np.diag(W)                        # sign of each channel's own variance term

info = mne.create_info(chs, es.sfreq, "eeg")
info.set_montage("standard_1005", on_missing="ignore")

csp = models.csp_lda(6)
csp.fit(X, es.y)
patterns = csp.named_steps["csp"].patterns_

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, v, ttl, cm in [
        (axes[0], diag_weight, "tangent-space weight\n(own-variance term)", "RdBu_r"),
        (axes[1], chan_weight, "tangent-space weight\n(total |weight| per channel)", "Reds"),
        (axes[2], patterns[0], "CSP pattern 1", "RdBu_r"),
        (axes[3], patterns[-1], "CSP pattern 6", "RdBu_r")]:
    lim = np.abs(v).max()
    vlim = (0, lim) if cm == "Reds" else (-lim, lim)
    im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cm,
                                 vlim=vlim, contours=4)
    ax.set_title(ttl, fontsize=9)
fig.suptitle("Where the cross-subject classifier gets its evidence", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig5_spatial.png", dpi=150)
print("wrote figures/fig5_spatial.png")

pd.DataFrame(dict(channel=chs, diag_weight=diag_weight,
                  total_abs_weight=chan_weight)).to_csv(
    "results/channel_weights.csv", index=False)
