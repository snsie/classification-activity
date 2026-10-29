"""Fit and score models on the supplied folds. You don't need to change this file.

Probabilities are rounded to 4 decimals before any metric is computed, so every number
your app recomputes with m11 matches the exported one exactly.
"""

from __future__ import annotations

import warnings

import numpy as np

import m11

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*X has feature names.*")

N_FOLDS = 5


def rounded(p) -> list[float]:
    return [float(v) for v in np.round(np.asarray(p, dtype=float), 4)]


def proba(model, X) -> list[float]:
    """The model's estimated probability of the positive class, rounded to 4 decimals."""
    return rounded(model.predict_proba(X)[:, 1])


def cross_validate(build, X, y: list[int], folds: list[int]) -> dict:
    """Fit a fresh model on 4 folds and predict the 5th, five times, using only the supplied folds.

    Returns the out-of-fold (OOF) probability of every development row, and each fold's
    training AUROC (on the rows it learned from) and CV AUROC (on the fold it never saw)."""
    folds_arr = np.asarray(folds)
    y_arr = np.asarray(y)
    oof = [None] * len(y)
    rows = []
    for k in range(1, N_FOLDS + 1):
        tr, va = np.flatnonzero(folds_arr != k), np.flatnonzero(folds_arr == k)
        model = build().fit(X.iloc[tr], y_arr[tr])
        p_tr, p_va = proba(model, X.iloc[tr]), proba(model, X.iloc[va])
        for i, v in zip(va, p_va):
            oof[i] = v
        rows.append({
            "fold": k, "n_train": int(len(tr)), "n_valid": int(len(va)),
            "train_auroc": m11.auroc(p_tr, y_arr[tr].tolist()),
            "cv_auroc": m11.auroc(p_va, y_arr[va].tolist()),
        })
    return {"oof": oof, "folds": rows}


def fit_and_test(build, X_dev, y_dev: list[int], X_test, y_test: list[int]) -> dict:
    """Refit on every development row, then predict the test rows once."""
    model = build().fit(X_dev, np.asarray(y_dev))
    p = proba(model, X_test)
    return {"p": p, "auroc": m11.auroc(p, y_test)}
