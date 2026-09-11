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
import cached, models, chosen, evaluation as E, config as CFG

mne.set_log_level("ERROR")
es = cached.load_pool(CFG.PARADIGM)
SUBSET = CFG.EVAL_SUBJECTS               # all EVAL subjects: the covariance-domain
                                         # CSP is cheap enough now


def aligned(X, subject):
    ea = models.EuclideanAlign()
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    return np.concatenate([ea.transform(X[subject == u]) for u in subs])[np.argsort(order)]


# ---------------- (a) time-resolved ---------------------------------------
centres = np.arange(-1.75, 4.1, 0.25)
rows = []
for c in centres:
    t0 = time.time()
    X, chs, _ = cached.prepare(es, band=CFG.BAND, window=(c - 0.5, c + 0.5))
    Xa = chosen.featurize(X, es.subject)
    del X
    r = E.loso(Xa, es.y, es.subject, es.run, chosen.make, subjects=SUBSET, n_jobs=6)
    s = E.summarize(r)
    rows.append(dict(centre_s=float(c), **s))
    pd.DataFrame(rows).to_csv("results/time_resolved.csv", index=False)
    print(f"t={c:+.2f}s  acc={s['pooled_acc']:.4f} "
          f"[{s['pooled_lo']:.3f},{s['pooled_hi']:.3f}]  {time.time()-t0:.0f}s", flush=True)
    del Xa

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
Cov = chosen.featurize(X, es.subject)
del X
pipe = chosen.make().fit(Cov, es.y)
csp = pipe.named_steps["csp"]
lda = pipe.named_steps["clf"]

info = mne.create_info(chs, es.sfreq, "eeg")
info.set_montage("standard_1005", on_missing="ignore")

pat = csp.patterns_                       # (n_components, n_chan)
coef = lda.coef_.ravel()
k = pat.shape[0]
fig, axes = plt.subplots(1, k, figsize=(2.05 * k, 2.9))
for i, ax in enumerate(np.atleast_1d(axes)):
    v = pat[i]
    lim = np.abs(v).max()
    im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-lim, lim), contours=4)
    ax.set_title(f"CSP {i+1}\nLDA weight {coef[i]:+.2f}", fontsize=9)
fig.suptitle("Spatial patterns of the cross-subject classifier "
             "(fitted on all 93 pool subjects)", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig5_spatial.png", dpi=150)
print("wrote figures/fig5_spatial.png")

# Where does a component draw its power from? Summarise each pattern by how
# much of its energy sits over the left vs right sensorimotor strip.
LEFT = [c for c in ["FC5", "FC3", "C5", "C3", "CP5", "CP3"] if c in chs]
RIGHT = [c for c in ["FC6", "FC4", "C6", "C4", "CP6", "CP4"] if c in chs]
li = [float((pat[i][[chs.index(c) for c in LEFT]] ** 2).sum()
            - (pat[i][[chs.index(c) for c in RIGHT]] ** 2).sum()) for i in range(k)]
tot = np.abs(pat).sum(1)
pd.DataFrame(dict(component=np.arange(1, k + 1), lda_weight=coef,
                  left_minus_right_energy=li,
                  frac_energy_sensorimotor=[
                      float(np.abs(pat[i][[chs.index(c) for c in LEFT + RIGHT]]).sum() / tot[i])
                      for i in range(k)])).to_csv("results/csp_components.csv", index=False)
print(pd.read_csv("results/csp_components.csv").to_string(index=False))
