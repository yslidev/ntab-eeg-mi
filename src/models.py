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


class AlignShrunk(BaseEstimator, TransformerMixin):
    """Euclidean alignment with a dial.

    alpha=0 is no alignment, alpha=1 is full whitening by the recording's own
    mean covariance. In between it whitens by ((1-a)I + a*C), which lets us ask
    how much of the between-subject covariance shift is worth removing.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y=None): return self

    def transform(self, X):
        if self.alpha == 0:
            return X
        Xd = X.astype(np.float64, copy=False)
        C = np.einsum("nct,ndt->cd", Xd, Xd) / (X.shape[0] * X.shape[-1])
        C /= np.trace(C) / C.shape[0]
        C = (1 - self.alpha) * np.eye(C.shape[0]) + self.alpha * C
        w, V = np.linalg.eigh(C)
        R = V @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ V.T
        return np.einsum("cd,ndt->nct", R.astype(X.dtype), X)


class FBCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP: a separate CSP per sub-band, log-variance features
    concatenated. Supervised, so it is refitted inside every fold."""
    def __init__(self, bands=((8, 12), (12, 16), (16, 20), (20, 24), (24, 30)),
                 n_components=4, sfreq=160.0, reg=0.1):
        self.bands, self.n_components = bands, n_components
        self.sfreq, self.reg = sfreq, reg

    def _filt(self, X, lo, hi):
        sos = butter(4, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos")
        return np.ascontiguousarray(sosfiltfilt(sos, X.astype(np.float64), axis=-1))

    def fit(self, X, y):
        self.csps_ = []
        for lo, hi in self.bands:
            c = CSP(n_components=self.n_components, reg=self.reg, log=True,
                    norm_trace=False, rank="full")
            c.fit(self._filt(X, lo, hi), y)
            self.csps_.append(c)
        return self

    def transform(self, X):
        return np.concatenate(
            [c.transform(self._filt(X, lo, hi))
             for c, (lo, hi) in zip(self.csps_, self.bands)], axis=1)


def fbcsp(n_components=4, C=0.1, sfreq=160.0,
          bands=((8, 12), (12, 16), (16, 20), (20, 24), (24, 30))):
    return Pipeline([("fb", FBCSP(bands=bands, n_components=n_components, sfreq=sfreq)),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


class CovCSP(BaseEstimator, TransformerMixin):
    """CSP that consumes precomputed per-epoch covariance matrices.

    Mathematically this is CSP with per-epoch covariance estimation and trace
    normalisation. The practical point is that the covariance of an epoch does
    not depend on the training fold, so it can be computed once for the whole
    dataset instead of being recomputed inside all 58 leave-one-subject-out
    folds. MNE's CSP estimates its class covariances from the concatenated raw
    trials, which means a 64 x 1.9M float64 matrix per class per fold; that is
    what made the sweep need 4 GB per worker and 150 s per configuration.
    """
    def __init__(self, n_components=6, shrinkage=0.1, log=True, normalise=True):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.log = log
        self.normalise = normalise

    def _prep(self, C):
        C = np.asarray(C, dtype=np.float64)
        if self.normalise:
            C = C / np.trace(C, axis1=-2, axis2=-1)[:, None, None]
        return C

    @staticmethod
    def _shrink(C, r):
        n = C.shape[0]
        return (1 - r) * C + r * np.trace(C) / n * np.eye(n)

    def fit(self, C, y):
        from scipy.linalg import eigh
        C = self._prep(C)
        classes = np.unique(y)
        Ca = self._shrink(C[y == classes[0]].mean(0), self.shrinkage)
        Cb = self._shrink(C[y == classes[1]].mean(0), self.shrinkage)
        w, V = eigh(Ca, Ca + Cb)
        k = self.n_components // 2
        idx = np.concatenate([np.arange(k), np.arange(len(w) - k, len(w))])
        self.filters_ = V[:, idx]                       # (n_chan, n_components)
        # patterns, for interpretation: columns of the inverse of the full basis
        self.patterns_ = np.linalg.pinv(V).T[:, idx].T
        self.mean_var_ = None
        F = self._power(C)
        self.mean_var_ = F.mean(0)
        return self

    def _power(self, C):
        W = self.filters_
        p = np.einsum("ck,ncd,dk->nk", W, C, W)
        return np.clip(p, 1e-20, None)

    def transform(self, C):
        p = self._power(self._prep(C))
        p = p / p.sum(1, keepdims=True)                 # relative power
        return np.log(p) if self.log else p


def covcsp_lda(n_components=6, shrinkage=0.1):
    return Pipeline([("csp", CovCSP(n_components, shrinkage)),
                     ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                        shrinkage="auto"))])


def covcsp_lr(n_components=6, shrinkage=0.1, C=1.0):
    return Pipeline([("csp", CovCSP(n_components, shrinkage)),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])
