"""Cache wide, broadband, cue-locked epochs once so every later analysis is cheap.

Design choice: filtering is done on the *continuous* run (before epoching) at
1-45 Hz, and we keep a deliberately wide window (-2.5 to 4.5 s around cue).
Narrow-band filtering and window selection then happen in memory. This keeps
filter edge artefacts at run boundaries and lets us re-cut the analysis window
without re-reading 763 EDF files.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from joblib import Parallel, delayed
import data as D

TMIN, TMAX = -2.5, 4.5
OUT = pathlib.Path("cache"); OUT.mkdir(exist_ok=True)
SUBJECTS = [s for s in range(1, 110) if s not in D.KNOWN_BAD]


def one(subject, runs):
    try:
        return D.load_epochs(subject, runs, tmin=TMIN, tmax=TMAX,
                             l_freq=1.0, h_freq=45.0)
    except Exception as e:
        print(f"  S{subject:03d} failed: {e}", flush=True)
        return None


for name, runs in [("imagined", D.RUNS_IMAG_LR), ("executed", D.RUNS_EXEC_LR)]:
    t0 = time.time()
    sets = Parallel(n_jobs=8, verbose=0)(delayed(one)(s, runs) for s in SUBJECTS)
    es = D.concat(sets)
    np.savez_compressed(
        OUT / f"{name}.npz", X=es.X.astype(np.float32), y=es.y,
        subject=es.subject, run=es.run, trial=es.trial, onset=es.onset,
        ch_names=np.array(es.ch_names), sfreq=es.sfreq, tmin=TMIN, tmax=TMAX)
    print(f"{name}: {len(es)} epochs, {len(np.unique(es.subject))} subjects, "
          f"X={es.X.shape}, {time.time()-t0:.0f}s", flush=True)
