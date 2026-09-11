"""Classifier factories. Everything is an sklearn Pipeline over (n, ch, t) arrays."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import mne
mne.set_log_level("ERROR")
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt


class LogVar(BaseEstimator, TransformerMixin):
    """log band-power per channel: the simplest sensible EEG feature."""
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log(np.var(X, axis=-1) + 1e-20)


class FilterBank(BaseEstimator, TransformerMixin):
    """Split into sub-bands and stack log-variance features (FBCSP-lite)."""
    def __init__(self, bands=((4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
                              (24, 28), (28, 32), (32, 40)), sfreq=160.0):
        self.bands, self.sfreq = bands, sfreq

    def fit(self, X, y=None): return self

    def transform(self, X):
        out = []
        for lo, hi in self.bands:
            sos = butter(4, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos")
            out.append(np.log(np.var(sosfiltfilt(sos, X, axis=-1), axis=-1) + 1e-20))
        return np.concatenate(out, axis=1)


class EuclideanAlign(BaseEstimator, TransformerMixin):
    """Per-recording whitening by the mean spatial covariance (He & Wu, 2020).

    Unsupervised: uses the *unlabelled* trials of whichever recording it is
    handed. It removes a large part of the between-subject covariance shift.
    """
    def fit(self, X, y=None): return self

    def transform(self, X):
        Xd = X.astype(np.float64, copy=False)
        C = np.einsum("nct,ndt->cd", Xd, Xd) / (X.shape[0] * X.shape[-1])
        C += 1e-10 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        w, V = np.linalg.eigh(C)
        R = V @ np.diag(w ** -0.5) @ V.T
        return np.einsum("cd,ndt->nct", R.astype(X.dtype), X)


def csp_lda(n_components=6, reg=0.1):
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg=reg, log=True,
                    norm_trace=False, rank="full")),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def riemann_ts(C=0.1):
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="liblinear"))])


def logvar_lr(C=0.1):
    return Pipeline([("f", LogVar()), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])


def fbcsp_lite(C=0.05, sfreq=160.0):
    return Pipeline([("f", FilterBank(sfreq=sfreq)), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def riemann_svm(C=1.0):
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler()),
        ("clf", SVC(C=C, kernel="rbf", probability=True))])


REGISTRY = {
    "logvar_lr": logvar_lr,
    "csp_lda": csp_lda,
    "fbcsp_lite": fbcsp_lite,
    "riemann_ts": riemann_ts,
    "riemann_svm": riemann_svm,
}


# ---------------------------------------------------------------------------
# Covariance fast path: in leave-one-subject-out the covariance of every epoch
# is recomputed 105 times for no reason. Precompute once, then the per-fold
# pipeline only has to fit the tangent-space projection and the classifier.
# ---------------------------------------------------------------------------

def precompute_cov(X, estimator="oas", block=512):
    """Covariances in blocks so a 4,500-epoch array never doubles in memory."""
    est = Covariances(estimator=estimator)
    out = np.empty((X.shape[0], X.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(0, len(X), block):
        out[i:i + block] = est.fit_transform(X[i:i + block].astype(np.float64))
    return out


def ts_lr(C=0.1, metric="riemann"):
    return Pipeline([("ts", TangentSpace(metric=metric)), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000, solver="liblinear"))])


def ts_lr_le(C=0.1):
    return ts_lr(C=C, metric="logeuclid")


COV_REGISTRY = {"riemann_ts": ts_lr, "riemann_ts_le": ts_lr_le}


def lr_only(C=0.1):
    """For features that do not depend on the fold and can be precomputed."""
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])
