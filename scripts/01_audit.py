"""Structural audit of every downloaded recording.

Nothing here is modelling. The point is to find out what is actually in the
files before trusting any of it: sampling rates, run lengths, event counts,
annotation durations, amplitude outliers.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
import data as D

RUNS = D.RUNS_EXEC_LR + D.RUNS_IMAG_LR + [1]
rows = []
for s in range(1, 110):
    for r in sorted(RUNS):
        f = D.CACHE / f"S{s:03d}" / f"S{s:03d}R{r:02d}.edf"
        if not f.is_file():
            continue
        try:
            rows.append(D.audit_run(s, r))
        except Exception as e:
            rows.append(dict(subject=s, run=r, error=f"{type(e).__name__}: {e}"))

df = pd.DataFrame(rows)
out = pathlib.Path("results"); out.mkdir(exist_ok=True)
df.to_csv(out / "audit_runs.csv", index=False)

print(f"audited {len(df)} runs from {df.subject.nunique()} subjects\n")
print("--- sampling rate ---");   print(df.sfreq.value_counts().to_string())
print("\n--- run duration (s) ---"); print(df.duration_s.value_counts().head(8).to_string())
print("\n--- channel count ---");  print(df.n_chan.value_counts().to_string())
print("\n--- task-trial counts per run (T1+T2) ---")
df["n_task"] = df.n_T1 + df.n_T2
print(df.n_task.value_counts().to_string())
print("\n--- T1 annotation duration ---"); print(df.dur_T1.round(3).value_counts().head().to_string())

task = df[df.run != 1]           # run 1 is the eyes-open baseline: no trials by design
bad = task[(task.sfreq != 160) | (task.n_chan != 64) | (task.n_task != 15)
           | (task.duration_s < 120) | (task.duration_s > 126)]
print(f"\n--- non-conforming runs: {len(bad)} across subjects "
      f"{sorted(bad.subject.unique().tolist())} ---")
print(bad[["subject", "run", "sfreq", "n_chan", "duration_s", "n_T0", "n_T1",
           "n_T2", "dur_T1"]].to_string(index=False))

print("\n--- amplitude: subjects with the largest peak |uV| ---")
amp = df.groupby("subject").max_abs_uv.max().sort_values(ascending=False)
print(amp.head(12).round(0).to_string())
print(f"\nmedian peak |uV| across runs: {df.max_abs_uv.median():.0f}")

print("\n--- class balance within subject (imagined runs) ---")
im = df[df.run.isin(D.RUNS_IMAG_LR)].groupby("subject")[["n_T1", "n_T2"]].sum()
im["imbalance"] = im.n_T1 - im.n_T2
print(im.imbalance.value_counts().sort_index().to_string())
print("\nsubjects with |imbalance| > 3:", sorted(im.index[im.imbalance.abs() > 3].tolist()))
