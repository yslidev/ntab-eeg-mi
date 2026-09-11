"""Frozen experimental configuration.

The subject split below is the backbone of the whole evaluation. Every model
choice, band, window and hyper-parameter in this project is selected using
DEV subjects as the held-out folds. Headline numbers are then reported using
EVAL subjects as the held-out folds, which were never looked at during
selection. Training sets always contain every *other* subject, so the two
regimes are comparable in training size; only the test folds differ.
"""
import numpy as np
import data as D

ALL_SUBJECTS = np.array([s for s in range(1, 110) if s not in D.KNOWN_BAD])
DEV_SUBJECTS = ALL_SUBJECTS[:40]
EVAL_SUBJECTS = ALL_SUBJECTS[40:]

# Frozen defaults, chosen in scripts/04_model_selection.py on DEV only.
BAND = (8.0, 30.0)
WINDOW = (0.5, 3.5)
PARADIGM = "imagined"
SEED = 0
