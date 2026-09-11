"""In-memory band-pass + window selection on top of the cached wide epochs."""
from __future__ import annotations
import pathlib
import numpy as np
from scipy.signal import butter, sosfiltfilt
import data as D

CACHE = pathlib.Path(__file__).resolve().parents[1] / "cache"
_MEM = {}


def load(paradigm: str) -> D.EpochSet:
    if paradigm not in _MEM:
        z = np.load(CACHE / f"{paradigm}.npz", allow_pickle=False)
        es = D.EpochSet(z["X"], z["y"], z["subject"], z["run"], z["trial"],
                        z["onset"], list(z["ch_names"]), float(z["sfreq"]))
        es.tmin, es.tmax = float(z["tmin"]), float(z["tmax"])
        _MEM[paradigm] = es
    return _MEM[paradigm]


def times(es) -> np.ndarray:
    return es.tmin + np.arange(es.X.shape[-1]) / es.sfreq


def prepare(es, band=(8.0, 30.0), window=(0.5, 3.5), picks=None, order=4,
            dtype=np.float32, block=256):
    """Band-pass (zero-phase) the wide epoch then crop to `window`.

    Filtering the wide (-2.5..4.5 s) epoch and cropping the interior keeps the
    filter's edge transient outside the analysis window. Done in blocks: the
    full cache is ~1.2 GB in float32 and doubling it is what makes this machine
    swap.
    """
    if picks is not None:
        idx = [es.ch_names.index(c) for c in picks if c in es.ch_names]
    else:
        idx = list(range(es.X.shape[1]))
    chs = [es.ch_names[i] for i in idx]
    t = times(es)
    m = (t >= window[0]) & (t <= window[1])
    sos = (butter(order, [band[0], band[1]], btype="bandpass", fs=es.sfreq,
                  output="sos") if band is not None else None)

    out = np.empty((es.X.shape[0], len(idx), int(m.sum())), dtype=dtype)
    for i in range(0, es.X.shape[0], block):
        chunk = es.X[i:i + block][:, idx].astype(np.float64)
        if sos is not None:
            chunk = sosfiltfilt(sos, chunk, axis=-1)
        out[i:i + block] = chunk[..., m].astype(dtype)
    return out, chs, t[m]
