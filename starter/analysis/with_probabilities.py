"""Supplied helper. You don't need to read or change this file."""

from sklearn.calibration import CalibratedClassifierCV


def with_probabilities(model):
    """Wraps a classifier that outputs raw scores (such as LinearSVC) so that it also outputs probabilities between 0 and 1, learned from the training rows only."""
    return CalibratedClassifierCV(model, method="sigmoid", cv=5)
