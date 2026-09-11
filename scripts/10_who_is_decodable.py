"""Who is decodable, and can we tell in advance?

The spread of per-subject accuracy is the largest effect in this whole project
-- larger than the difference between any two classifiers. This script asks
whether that spread is predictable from a recording in which the subject is
doing nothing at all: the one-minute eyes-open and eyes-closed baseline runs.

If it is, then a chunk of what looks like "some people are better at motor
imagery" is really "some people have a measurable sensorimotor rhythm", and
you could screen for it in 60 seconds without ever running the task.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, mne
from scipy import stats
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D, config as CFG

mne.set_log_level("ERROR")
POST = ["O1", "Oz", "O2", "PO3", "POz", "PO4", "P3", "Pz", "P4"]
SENSORI = ["C3", "C4", "C1", "C2", "CP3", "CP4", "FC3", "FC4"]


def aperiodic_slope(f, p):
    """log-log linear fit over 2-40 Hz, masking the alpha/beta band and 60 Hz."""
    m = (f >= 2) & (f <= 40) & ~((f > 6) & (f < 16)) & ~((f > 55) & (f < 65))
    sl, ic = np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)
    return sl, ic


def rest_features(subject):
    out = {"subject": int(subject)}
    psds = {}
    for run, tag in [(1, "eo"), (2, "ec")]:
        try:
            raw = D._read_run(int(subject), run)
        except Exception:
            return None
        if abs(raw.info["sfreq"] - 160) > 1e-6:
            raw.resample(160.0, verbose="ERROR")
        X = raw.get_data() * 1e6
        f, P = welch(X, fs=160.0, nperseg=512, noverlap=256, axis=-1)
        psds[tag] = (f, P, raw.ch_names)

        idx = lambda names: [raw.ch_names.index(c) for c in names if c in raw.ch_names]
        band = lambda lo, hi, ii: float(P[ii][:, (f >= lo) & (f < hi)].mean())
        out[f"{tag}_alpha_post"] = np.log10(band(8, 13, idx(POST)))
        out[f"{tag}_mu_sensori"] = np.log10(band(8, 13, idx(SENSORI)))
        out[f"{tag}_beta_sensori"] = np.log10(band(13, 30, idx(SENSORI)))
        out[f"{tag}_broadband"] = np.log10(band(2, 45, list(range(len(raw.ch_names)))))
        out[f"{tag}_line60"] = np.log10(band(58, 62, list(range(len(raw.ch_names)))) /
                                        band(45, 55, list(range(len(raw.ch_names)))))
        # peak alpha frequency over posterior sensors
        pm = (f >= 6) & (f <= 14)
        out[f"{tag}_alpha_peak_hz"] = float(f[pm][P[idx(POST)].mean(0)[pm].argmax()])
        # how far the mu bump rises above this subject's own 1/f background
        sp = P[idx(SENSORI)].mean(0)
        sl, ic = aperiodic_slope(f, sp)
        fit = 10 ** (ic + sl * np.log10(np.clip(f, 1e-9, None)))
        mm = (f >= 8) & (f < 13)
        out[f"{tag}_mu_prominence_db"] = float(10 * np.log10((sp[mm] / fit[mm]).max()))
        out[f"{tag}_aperiodic_slope"] = float(sl)
    # alpha reactivity: eyes-closed minus eyes-open posterior alpha
    out["alpha_reactivity"] = out["ec_alpha_post"] - out["eo_alpha_post"]
    out["mu_reactivity"] = out["ec_mu_sensori"] - out["eo_mu_sensori"]
    return out


rows = [r for r in (rest_features(s) for s in CFG.ALL_SUBJECTS) if r]
rest = pd.DataFrame(rows).set_index("subject")
rest.to_csv("results/rest_features.csv")
print(f"resting features for {len(rest)} subjects")

# --- join to per-subject decoding accuracy -------------------------------
acc = pd.read_csv("results/per_subject_accuracy.csv").set_index("subject")
df = rest.join(acc, how="inner").dropna(subset=["acc_loso"])
print(f"joined with decoding accuracy for {len(df)} subjects\n")

targets = [c for c in ["acc_loso", "acc_within_run"] if c in df.columns]
feats = [c for c in rest.columns]
res = []
for tgt in targets:
    for c in feats:
        r, p = stats.spearmanr(df[c], df[tgt])
        res.append(dict(target=tgt, feature=c, rho=r, p=p, n=len(df)))
res = pd.DataFrame(res).sort_values("p")
res.to_csv("results/rest_vs_accuracy.csv", index=False)
print(res.head(14).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

n_tests = len(res)
print(f"\nBonferroni threshold for {n_tests} tests: p < {0.05/n_tests:.5f}")
print("surviving:", res[res.p < 0.05 / n_tests].feature.tolist() or "none")

# --- a picture of the strongest association ------------------------------
best = res[res.target == "acc_loso"].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].scatter(df[best.feature], df.acc_loso, s=22, alpha=.75, color="#2b6cb0")
m, b = np.polyfit(df[best.feature], df.acc_loso, 1)
xs = np.linspace(df[best.feature].min(), df[best.feature].max(), 20)
ax[0].plot(xs, m * xs + b, color="#c53030", lw=1.5)
ax[0].axhline(0.5, ls=":", color="grey")
ax[0].set_xlabel(best.feature); ax[0].set_ylabel("cross-subject accuracy")
ax[0].set_title(f"rho={best.rho:+.2f}, p={best.p:.1e}, n={len(df)}", fontsize=10)

ax[1].hist(df.acc_loso, bins=18, color="#2b6cb0", alpha=.85)
ax[1].axvline(0.5, color="grey", ls=":")
ax[1].axvline(df.acc_loso.mean(), color="#c53030", lw=1.5,
              label=f"mean {df.acc_loso.mean():.3f}")
ax[1].set_xlabel("cross-subject accuracy"); ax[1].set_ylabel("subjects")
ax[1].legend(fontsize=8); ax[1].set_title("per-subject spread", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig6_who_is_decodable.png", dpi=150)
print("\nwrote figures/fig6_who_is_decodable.png")

# =====================================================================
# How much of the per-subject spread is a real individual difference, and
# how much is just 43 coin flips per person?
# =====================================================================
print("\n=== variance decomposition of per-subject accuracy ===")
rows_ps = pd.read_csv("results/per_subject_rows.csv")
rows_ps = rows_ps[~rows_ps.shuffled]
for (par, reg), g in rows_ps.groupby(["paradigm", "regime"]):
    if len(g) < 10:
        continue
    p_hat, n = g.acc.values, g.n.values
    v_obs = p_hat.var(ddof=1)
    v_binom = float(np.mean(p_hat * (1 - p_hat) / n))
    v_true = max(v_obs - v_binom, 0.0)
    print(f"{par:<9}{reg:<28} observed sd {np.sqrt(v_obs):.4f}  "
          f"expected-from-coin-flips sd {np.sqrt(v_binom):.4f}  "
          f"implied true between-subject sd {np.sqrt(v_true):.4f}  "
          f"({v_true/v_obs*100:.0f}% of variance)")

# Is decodability a stable property of a person? Compare each subject's
# accuracy on imagined trials with their accuracy on executed trials.
w = rows_ps.pivot_table(index="subject", columns=["paradigm", "regime"], values="acc")
for reg in sorted({c[1] for c in w.columns}):
    if ("imagined", reg) in w and ("executed", reg) in w:
        a_, b_ = w[("imagined", reg)], w[("executed", reg)]
        m = a_.notna() & b_.notna()
        if m.sum() < 10:
            continue
        r_, p_ = stats.spearmanr(a_[m], b_[m])
        print(f"imagined vs executed accuracy, {reg}: rho={r_:+.3f} p={p_:.3f} n={m.sum()}")
