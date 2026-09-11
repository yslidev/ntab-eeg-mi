"""Loading, auditing and epoching of the PhysioNet EEG Motor Movement/Imagery DB.

One module-level rule: every epoch we ever produce carries provenance
(subject, run, trial index within run, wall-clock onset). Every evaluation
split in this project is defined on that provenance, never on row order.
"""
from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass

import numpy as np
import mne
from mne.datasets import eegbci

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# Run groups. Documented by PhysioNet; the T1/T2 semantics are verified
# empirically in scripts/03_lateralisation.py rather than taken on faith.
RUNS_EXEC_LR = [3, 7, 11]     # left vs right fist, actually performed
RUNS_IMAG_LR = [4, 8, 12]     # left vs right fist, imagined
RUNS_BASELINE = [1, 2]        # eyes open / eyes closed

SFREQ = 160.0

# Subjects whose recordings deviate from the nominal format. Populated by the
# audit (scripts/01_audit.py); kept here so every downstream script agrees.
KNOWN_BAD = (88, 89, 92, 100)

# Sensorimotor strip; used for channel-lesion controls.
MOTOR_CH = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC1", "FCz", "FC2", "FC4",
            "CP3", "CP1", "CPz", "CP2", "CP4"]
FRONTAL_CH = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
              "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"]
OCCIPITAL_CH = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8",
                "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]


@dataclass
class EpochSet:
    """X: (n, ch, t) float64 volts. y: 0=left fist, 1=right fist."""
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    run: np.ndarray
    trial: np.ndarray       # index of the trial within its run
    onset: np.ndarray       # seconds from the start of that run
    ch_names: list
    sfreq: float

    def __len__(self):
        return len(self.y)

    def subset(self, mask):
        m = np.asarray(mask)
        return EpochSet(self.X[m], self.y[m], self.subject[m], self.run[m],
                        self.trial[m], self.onset[m], self.ch_names, self.sfreq)


CACHE = pathlib.Path.home() / "mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"


def _read_run(subject: int, run: int) -> mne.io.BaseRaw:
    local = CACHE / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"
    path = str(local) if local.is_file() else \
        eegbci.load_data(subject, [run], update_path=True, verbose="ERROR")[0]
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)                       # 'Fc5.' -> 'FC5'
    raw.set_montage("standard_1005", on_missing="warn", verbose="ERROR")
    return raw


def audit_run(subject: int, run: int) -> dict:
    """Cheap structural facts about one recording, for the data audit."""
    raw = _read_run(subject, run)
    ann = raw.annotations
    desc = list(ann.description)
    durs = np.asarray(ann.duration)
    return dict(
        subject=subject, run=run,
        sfreq=float(raw.info["sfreq"]),
        n_chan=len(raw.ch_names),
        duration_s=float(raw.n_times / raw.info["sfreq"]),
        n_ann=len(desc),
        n_T0=desc.count("T0"), n_T1=desc.count("T1"), n_T2=desc.count("T2"),
        dur_T1=float(np.median(durs[np.array(desc) == "T1"])) if "T1" in desc else np.nan,
        dur_T0=float(np.median(durs[np.array(desc) == "T0"])) if "T0" in desc else np.nan,
        first_desc=desc[0] if desc else "",
        max_abs_uv=float(np.abs(raw.get_data()).max() * 1e6),
    )


def load_epochs(subject: int, runs, tmin=0.5, tmax=3.5, l_freq=8.0, h_freq=30.0,
                event_labels=("T1", "T2"), baseline=None, picks="eeg",
                notch=False) -> EpochSet | None:
    """Band-pass filter each run, then cut cue-locked epochs.

    Filtering happens per continuous run *before* epoching so that filter
    edge effects land at run boundaries, not at every trial boundary.
    """
    Xs, ys, subs, rns, trs, ons = [], [], [], [], [], []
    ch_names, sfreq = None, None
    for run in runs:
        try:
            raw = _read_run(subject, run)
        except Exception:
            continue
        if abs(raw.info["sfreq"] - SFREQ) > 1e-6:
            # Non-standard sampling rate: resample so the time axis is comparable.
            raw.resample(SFREQ, verbose="ERROR")
        if notch:
            raw.notch_filter([60.0], verbose="ERROR")
        if l_freq is not None or h_freq is not None:
            raw.filter(l_freq, h_freq, method="iir",
                       iir_params=dict(order=4, ftype="butter"), verbose="ERROR")

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        want = {k: v for k, v in event_id.items() if k in event_labels}
        if len(want) < len(event_labels):
            continue
        ep = mne.Epochs(raw, events, want, tmin=tmin, tmax=tmax, baseline=baseline,
                        picks=picks, preload=True, proj=False, verbose="ERROR",
                        on_missing="ignore")
        if len(ep) == 0:
            continue
        code2lab = {want[k]: i for i, k in enumerate(event_labels)}
        y = np.array([code2lab[c] for c in ep.events[:, 2]])
        Xs.append(ep.get_data(copy=True))
        ys.append(y)
        subs.append(np.full(len(y), subject))
        rns.append(np.full(len(y), run))
        trs.append(np.arange(len(y)))
        ons.append(ep.events[:, 0] / raw.info["sfreq"])
        ch_names, sfreq = ep.ch_names, ep.info["sfreq"]

    if not Xs:
        return None
    n_t = min(x.shape[-1] for x in Xs)
    Xs = [x[..., :n_t] for x in Xs]
    return EpochSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(subs),
                    np.concatenate(rns), np.concatenate(trs), np.concatenate(ons),
                    ch_names, sfreq)


def concat(sets) -> EpochSet:
    sets = [s for s in sets if s is not None and len(s)]
    n_t = min(s.X.shape[-1] for s in sets)
    return EpochSet(
        np.concatenate([s.X[..., :n_t] for s in sets]),
        np.concatenate([s.y for s in sets]),
        np.concatenate([s.subject for s in sets]),
        np.concatenate([s.run for s in sets]),
        np.concatenate([s.trial for s in sets]),
        np.concatenate([s.onset for s in sets]),
        sets[0].ch_names, sets[0].sfreq)
