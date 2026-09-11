"""The single definition of the selected pipeline.

Fixed after model selection on DEV subjects (scripts/04_model_selection.py and
scripts/15_selection2.py) and imported by the headline table, the controls, the
interpretation figures and the shipped artefact, so those cannot drift apart.

Pipeline, end to end:
    raw EDF
      -> standardise channel names, 10-05 montage, resample to 160 Hz
      -> 4th-order Butterworth band-pass, zero-phase, on the continuous run
      -> cue-locked epochs
      -> Euclidean alignment: whiten each recording by its own mean spatial
         covariance (unsupervised, no labels)
      -> per-epoch spatial covariance, OAS estimator
      -> CSP in the covariance domain, log relative power of k components
      -> shrinkage LDA
"""
import numpy as np
import models

NAME = "align + CovCSP + shrinkage LDA"
BAND = (8.0, 30.0)
WINDOW = (0.5, 3.5)
N_COMPONENTS = 6
SHRINKAGE = 0.1
ALIGN_ALPHA = 1.0


def make():
    return models.covcsp_lda(N_COMPONENTS, SHRINKAGE)


def align(X, subject, alpha=None):
    """Whiten each subject's trials by that subject's own mean covariance.

    Unsupervised: no labels are used. It is transductive with respect to the
    test subject, which is measured explicitly in scripts/06_controls.py.
    """
    a = ALIGN_ALPHA if alpha is None else alpha
    if a == 0:
        return X
    tr = models.AlignShrunk(alpha=a)
    subs = np.unique(subject)
    order = np.concatenate([np.where(subject == u)[0] for u in subs])
    return np.concatenate([tr.transform(X[subject == u]) for u in subs])[np.argsort(order)]


def featurize(X, subject, alpha=None):
    """Trials -> the covariance matrices the classifier actually consumes."""
    return models.precompute_cov(align(X, subject, alpha))
