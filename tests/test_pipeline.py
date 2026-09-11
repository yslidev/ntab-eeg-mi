"""Small checks that the pieces do what they claim. Run with: python -m pytest tests"""
import pathlib, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import models, evaluation as E


def test_alignment_makes_mean_covariance_a_multiple_of_identity():
    """AlignShrunk trace-normalises before whitening, so the aligned mean
    covariance is proportional to the identity rather than equal to it.
    Downstream, CovCSP trace-normalises again, so the scale is irrelevant."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    X = np.einsum("cd,ndt->nct", A, rng.normal(size=(40, 8, 200)))
    Xa = models.AlignShrunk(alpha=1.0).transform(X)
    C = np.einsum("nct,ndt->cd", Xa, Xa) / (len(Xa) * Xa.shape[-1])
    C = C / np.trace(C) * 8
    assert np.allclose(C, np.eye(8), atol=1e-8)


def test_full_alignment_removes_between_recording_covariance_differences():
    rng = np.random.default_rng(4)
    shapes = []
    for _ in range(5):                       # five "subjects", different mixing
        A = rng.normal(size=(6, 6))
        X = np.einsum("cd,ndt->nct", A, rng.normal(size=(30, 6, 300)))
        Xa = models.AlignShrunk(alpha=1.0).transform(X)
        C = np.einsum("nct,ndt->cd", Xa, Xa) / (len(Xa) * Xa.shape[-1])
        shapes.append(C / np.trace(C))
    shapes = np.array(shapes)
    assert shapes.std(0).max() < 1e-9


def test_alignment_alpha_zero_is_identity():
    X = np.random.default_rng(1).normal(size=(10, 4, 50))
    assert np.array_equal(models.AlignShrunk(alpha=0.0).transform(X), X)


def test_covcsp_separates_a_planted_difference():
    """Class 1 has extra variance in channel 0, class 0 in channel 7."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 8, 400))
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    X[y == 1, 0] *= 3.0
    X[y == 0, 7] *= 3.0
    C = models.precompute_cov(X)
    mdl = models.covcsp_lda(4).fit(C[::2], y[::2])
    assert (mdl.predict(C[1::2]) == y[1::2]).mean() > 0.9


def test_binomial_significance_threshold():
    assert E.binom_sig_threshold(43) > 0.5
    assert E.binom_sig_threshold(43) < 0.68
    assert E.binom_sig_threshold(10000) < 0.52


def test_shuffle_preserves_class_counts_per_group():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ys = E._shuffle_within(y, g, np.random.default_rng(3))
    for grp in (0, 1):
        assert sorted(ys[g == grp]) == sorted(y[g == grp])


@pytest.mark.skipif(not (ROOT / "model" / "mi_lr_model.joblib").is_file(),
                    reason="run scripts/08_train_final.py first")
def test_predict_roundtrip_on_one_edf():
    import data as D
    from predict import load_model, predict_file
    m = load_model()
    f = D.CACHE / "S001" / "S001R04.edf"
    if not f.is_file():
        pytest.skip("EDF not downloaded")
    pred, proba, onsets, y = predict_file(f, m)
    assert len(pred) == len(onsets) == len(y)
    assert set(np.unique(pred)) <= {0, 1}
    assert proba is None or ((proba >= 0).all() and (proba <= 1).all())
