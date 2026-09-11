"""Is T1 the left fist or the right fist? Decide it from the physiology.

Motor imagery/execution desynchronises the sensorimotor rhythm over the
hemisphere *contralateral* to the limb. We use a baseline-free laterality
index, because the only available baseline (the rest period before the cue)
is contaminated by the post-movement beta rebound of the *previous* trial --
the runs alternate task/rest with no gap.

  LI = mean_trials log10( power(left-hemisphere cluster) / power(right cluster) )

If T1 is the LEFT fist, T1 trials desynchronise the RIGHT hemisphere more, so
right-cluster power drops, so LI(T1) > LI(T2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import cached

mne.set_log_level("ERROR")
LEFT_C = ["FC3", "FC5", "C3", "C5", "CP3", "CP5"]
RIGHT_C = ["FC4", "FC6", "C4", "C6", "CP4", "CP6"]
BANDS = {"mu (8-13 Hz)": (8.0, 13.0), "beta (13-30 Hz)": (13.0, 30.0),
         "mu+beta (8-30 Hz)": (8.0, 30.0)}
WIN = (0.5, 3.5)

recs, topo = [], {}
for par in ["executed", "imagined"]:
    es = cached.load(par)
    for bname, band in BANDS.items():
        X, chs, _ = cached.prepare(es, band=band, window=WIN)
        pw = (X ** 2).mean(-1)                                   # (n, ch)
        li_l = np.log10(pw[:, [chs.index(c) for c in LEFT_C]]).mean(1)
        li_r = np.log10(pw[:, [chs.index(c) for c in RIGHT_C]]).mean(1)
        li = li_l - li_r
        for s in np.unique(es.subject):
            m = es.subject == s
            recs.append(dict(paradigm=par, band=bname, subject=int(s),
                             li_T1=li[m & (es.y == 0)].mean(),
                             li_T2=li[m & (es.y == 1)].mean()))
        if band == (8.0, 30.0):
            # topography of the between-class log-power contrast (T1 - T2)
            lp = np.log10(pw)
            topo[par] = lp[es.y == 0].mean(0) - lp[es.y == 1].mean(0)
            topo["info"] = mne.create_info(chs, es.sfreq, "eeg")
            topo["info"].set_montage("standard_1005", on_missing="ignore")

df = pd.DataFrame(recs)
df["d"] = df.li_T1 - df.li_T2
df.to_csv("results/lateralisation.csv", index=False)

print(f"{'paradigm':<10}{'band':<18}{'mean d':>9}{'t':>8}{'p':>11}"
      f"{'% subj d>0':>12}   verdict")
for par in ["executed", "imagined"]:
    for bname in BANDS:
        d = df[(df.paradigm == par) & (df.band == bname)].d.dropna()
        t, p = stats.ttest_1samp(d, 0)
        frac = (d > 0).mean() * 100
        verdict = "T1 = LEFT fist" if t > 0 else "T1 = RIGHT fist"
        if p > 0.05:
            verdict += "  (n.s. — no evidence)"
        print(f"{par:<10}{bname:<18}{d.mean():>+9.4f}{t:>8.2f}{p:>11.2e}"
              f"{frac:>11.0f}%   {verdict}")

print("\nd = LI(T1) - LI(T2); LI = log10 power(left cluster / right cluster).")
print("d > 0 means T1 suppresses the RIGHT hemisphere more, i.e. T1 is the LEFT fist.")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
vmax = max(np.abs(topo["executed"]).max(), np.abs(topo["imagined"]).max())
for ax, par in zip(axes, ["executed", "imagined"]):
    im, _ = mne.viz.plot_topomap(topo[par], topo["info"], axes=ax, show=False,
                                 cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
    ax.set_title(par, fontsize=11)
fig.suptitle("8–30 Hz log-power contrast, T1 minus T2 (group mean, 105 subjects)\n"
             "red = more power on T1 trials", fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.8, label="Δ log10 power")
fig.savefig("figures/fig1_lateralisation.png", dpi=150, bbox_inches="tight")
print("wrote figures/fig1_lateralisation.png")
