"""Frozen experimental configuration.

The subject split below is the backbone of the whole evaluation.

  HOLDOUT (12 subjects) -- touched by nothing. Not in model selection, not in
      any reported cross-validation, not in the training set of the shipped
      model. Used once, at the very end, to run predict.py end-to-end on
      recordings that are unseen in every sense.
  DEV (35 subjects)  -- the held-out folds used to choose the feature set,
      classifier, frequency band, analysis window and regularisation.
  EVAL (58 subjects) -- the held-out folds used for every reported number.
      Never consulted while making a choice.

Training sets always contain every *other* non-holdout subject, so DEV and EVAL
regimes see the same amount of training data; only the test folds differ.
"""
import numpy as np
import data as D

_USABLE = np.array([s for s in range(1, 110) if s not in D.KNOWN_BAD])

_rng = np.random.default_rng(20260911)
HOLDOUT_SUBJECTS = np.sort(_rng.choice(_USABLE, size=12, replace=False))
_POOL = np.array([s for s in _USABLE if s not in HOLDOUT_SUBJECTS])
_perm = _rng.permutation(len(_POOL))
DEV_SUBJECTS = np.sort(_POOL[_perm[:35]])
EVAL_SUBJECTS = np.sort(_POOL[_perm[35:]])
ALL_SUBJECTS = np.sort(_POOL)          # everything the models may train on

# Frozen defaults, chosen in scripts/04_model_selection.py on DEV only.
BAND = (8.0, 30.0)
WINDOW = (0.5, 3.5)
PARADIGM = "imagined"
SEED = 0
