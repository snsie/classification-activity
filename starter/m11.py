"""Shared metric helpers for the M11 "Judge the Model" activity (Python version of app/m11.js).

Every number in your app and in the analysis comes from these functions, so the Python and
browser versions always agree. They are ported from the M11 lecture app (web/src/data.ts)
and tested against each other on shared fixtures.

    import m11
    dev = m11.load("diabetes_dev")             # results/diabetes_dev.json, or None if missing
    p, y = m11.oof(dev, "logreg")              # development out-of-fold probabilities and labels
    c = m11.confusion(p, y, 0.5)               # {"tp", "fn", "fp", "tn"}; flag when p >= threshold
    m11.metrics(c)                             # sensitivity, specificity, ... (None = undefined)
    m11.roc(p, y)                              # ROC points and AUROC
    m11.calibration(p, y)                      # 10 equal-size bins
    m11.threshold_for_rule(p, y)               # the diabetes rule (sensitivity >= 0.90)
    m11.fold_summary(dev)                      # mean ± SD per model, and the one-SD flag

Undefined values (for example precision when no one is flagged, 0 ÷ 0) are None here and
null in JSON. Show them as "undefined", never as 0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DATASETS = ["readmission", "diabetes", "heart"]
MODELS = ["baseline", "logreg", "knn", "svm", "rf", "lgbm"]
FITTED = ["logreg", "knn", "svm", "rf", "lgbm"]
# Simplest first: the §16 rule prefers the earliest model that is within one SD of the best.
SIMPLICITY = ["logreg", "knn", "svm", "rf", "lgbm"]
MODEL_LABEL = {
    "baseline": "Prevalence baseline",
    "logreg": "Logistic regression",
    "knn": "k-nearest neighbours",
    "svm": "Support vector machine",
    "rf": "Random forest",
    "lgbm": "Gradient boosting",
}
DATASET_LABEL = {"readmission": "30-day readmission", "diabetes": "Diabetes screening", "heart": "Heart disease"}
# The diabetes job's rule, fixed by the activity.
RULE_SENSITIVITY = 0.90
RULE_TEXT = ("Among development out-of-fold thresholds where sensitivity is at least 0.90, "
             "the one with the highest specificity (ties: the higher sensitivity).")

# Thresholds can come from sliders (0.1 + 0.2 ≠ 0.3), so p within 1e-9 below t still counts as flagged.
EPS = 1e-9


# ---------------------------------------------------------------- loading
def load(name: str, folder: Path | str | None = None) -> dict | None:
    """results/<name>.json, e.g. "heart_dev", "heart_test", "decisions" or "test_record". None if it doesn't exist yet."""
    path = Path(folder or RESULTS) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def oof(dev: dict, model: str) -> tuple[list[float], list[int]]:
    """Development out-of-fold probabilities and labels for one model."""
    return dev["models"][model]["oof"], dev["y"]


def test_preds(test: dict, model: str) -> tuple[list[float], list[int]]:
    """Test-set probabilities and labels for one model."""
    return test["models"][model]["p"], test["y"]


# ---------------------------------------------------------------- the confusion matrix and its metrics
def confusion(p, y, threshold: float) -> dict:
    """Counts at one threshold. A patient is flagged (predicted positive) when p >= threshold."""
    tp = fn = fp = tn = 0
    for pi, yi in zip(p, y):
        flag = pi >= threshold - EPS
        if yi == 1:
            if flag:
                tp += 1
            else:
                fn += 1
        elif flag:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _ratio(a: float, b: float):
    return None if b == 0 else a / b


def metrics(c: dict) -> dict:
    """Every metric from the four counts. Denominators, in words:
    sensitivity (TPR) = TP ÷ (TP + FN): of the patients who had the outcome, the share flagged
    specificity      = TN ÷ (TN + FP): of the patients who did not, the share not flagged
    precision        = TP ÷ (TP + FP): of the patients flagged, the share who had the outcome
    FPR              = FP ÷ (FP + TN): of the patients who did not, the share flagged (1 − specificity)
    F1               = 2TP ÷ (2TP + FP + FN): the harmonic mean of precision and sensitivity
    """
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    n = tp + fn + fp + tn
    flagged = tp + fp
    return {
        "n": n,
        "positives": tp + fn,
        "negatives": fp + tn,
        "flagged": flagged,
        "flagged_per_100": _ratio(100 * flagged, n),
        "prevalence": _ratio(tp + fn, n),
        "accuracy": _ratio(tp + tn, n),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "precision": _ratio(tp, tp + fp),
        "fpr": _ratio(fp, fp + tn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def majority_accuracy(y) -> float | None:
    """Accuracy of "the most common class for everyone"."""
    n = len(y)
    if n == 0:
        return None
    pos = sum(1 for v in y if v == 1)
    return max(pos, n - pos) / n


# ---------------------------------------------------------------- ROC and AUROC
def _desc_order(p) -> list[int]:
    """Indices by descending probability; ties by index, so every run gives the same order."""
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


def auroc(p, y) -> float | None:
    """AUROC: the chance that a random positive gets a higher probability than a random negative (ties count ½).
    It describes ranking, not accuracy. None if only one class is present."""
    n = len(p)
    idx = sorted(range(n), key=lambda i: (p[i], i))
    n_pos = 0
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and p[idx[j + 1]] == p[idx[i]]:
            j += 1
        mid = (i + j) / 2 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            if y[idx[k]] == 1:
                n_pos += 1
                rank_sum += mid
        i = j + 1
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc(p, y) -> dict:
    """The ROC curve: one (FPR, TPR) point per distinct threshold, from flagging no one (0, 0) to everyone (1, 1).
    thr[k] is the threshold for point k (flag p >= thr[k]); thr[0] is infinity."""
    order = _desc_order(p)
    n_pos = sum(1 for v in y if v == 1)
    n_neg = len(y) - n_pos
    fpr, tpr, thr = [0.0], [0.0], [math.inf]
    tp = fp = 0
    k = 0
    while k < len(order):
        v = p[order[k]]
        while k < len(order) and p[order[k]] == v:
            if y[order[k]] == 1:
                tp += 1
            else:
                fp += 1
            k += 1
        fpr.append(fp / n_neg if n_neg else 0.0)
        tpr.append(tp / n_pos if n_pos else 0.0)
        thr.append(v)
    return {"fpr": fpr, "tpr": tpr, "thr": thr, "auroc": auroc(p, y)}


# ---------------------------------------------------------------- calibration
def calibration(p, y, bins: int = 10) -> list[dict]:
    """Calibration plot data: patients sorted by probability and cut into `bins` equal-size groups.
    For each: the mean predicted probability and the observed share with the outcome.
    On a well-calibrated model the two are close (the points sit near the diagonal)."""
    n = len(p)
    idx = sorted(range(n), key=lambda i: (p[i], i))
    groups: list[list[int]] = [[] for _ in range(bins)]
    for r, i in enumerate(idx):
        groups[min(bins - 1, (r * bins) // n)].append(i)
    out = []
    for g in groups:
        if not g:
            continue
        sp = sy = 0.0
        lo, hi = math.inf, -math.inf
        for i in g:
            sp += p[i]
            sy += y[i]
            lo = min(lo, p[i])
            hi = max(hi, p[i])
        out.append({"lo": lo, "hi": hi, "mean_pred": sp / len(g), "observed": sy / len(g), "n": len(g)})
    return out


# ---------------------------------------------------------------- the diabetes threshold rule
def threshold_for_rule(p, y, min_sensitivity: float = RULE_SENSITIVITY) -> dict | None:
    """The diabetes job's rule: among the distinct probabilities used as thresholds (flag p >= t), keep those
    with sensitivity >= min_sensitivity and choose the one with the highest specificity. If two tie on
    specificity, choose the higher sensitivity. Use it on development out-of-fold probabilities only."""
    order = _desc_order(p)
    n_pos = sum(1 for v in y if v == 1)
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return None
    best = None
    tp = fp = 0
    k = 0
    while k < len(order):  # lower the threshold one distinct probability at a time
        t = p[order[k]]
        while k < len(order) and p[order[k]] == t:
            if y[order[k]] == 1:
                tp += 1
            else:
                fp += 1
            k += 1
        sens = tp / n_pos
        spec = (n_neg - fp) / n_neg if n_neg else 0.0
        if sens >= min_sensitivity - EPS and (best is None or spec > best[0] or (spec == best[0] and sens > best[1])):
            best = (spec, sens, t)
    if best is None:
        return None
    t = best[2]
    c = confusion(p, y, t)
    return {"threshold": t, "counts": c, "metrics": metrics(c), "min_sensitivity": min_sensitivity}


# ---------------------------------------------------------------- comparing models across folds
def mean(a) -> float | None:
    return sum(a) / len(a) if a else None


def sd(a) -> float | None:
    """Sample standard deviation (n − 1), as in the lecture."""
    if len(a) < 2:
        return None
    m = mean(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / (len(a) - 1))


def fold_summary(dev: dict) -> dict:
    """Mean ± SD of training and cross-validated (CV) AUROC across the 5 folds, for every model in a
    development results file, plus the §16 comparison with logistic regression:
      lead_over_logreg  = model's mean CV AUROC − logreg's mean CV AUROC
      within_one_sd     = |lead| <= logreg's fold SD (so the difference is within fold-to-fold noise)
      best              = fitted model with the highest mean CV AUROC
      favoured          = the simplest model (logreg < knn < svm < rf < lgbm) whose mean CV AUROC is within
                          one logreg SD of the best: the model the lecture's rule prefers
    """
    out: dict = {"models": {}}
    for m, entry in dev["models"].items():
        cv = [f["cv_auroc"] for f in entry["folds"]]
        tr = [f["train_auroc"] for f in entry["folds"]]
        out["models"][m] = {
            "cv_folds": cv, "train_folds": tr,
            "mean_cv": mean([v for v in cv if v is not None]), "sd_cv": sd([v for v in cv if v is not None]),
            "mean_train": mean([v for v in tr if v is not None]), "sd_train": sd([v for v in tr if v is not None]),
        }
    ref = out["models"].get("logreg")
    for s in out["models"].values():
        if ref is None or s["mean_cv"] is None or ref["sd_cv"] is None:
            s["lead_over_logreg"] = s["within_one_sd"] = None
        else:
            s["lead_over_logreg"] = s["mean_cv"] - ref["mean_cv"]
            s["within_one_sd"] = abs(s["lead_over_logreg"]) <= ref["sd_cv"] + EPS
    fitted = [m for m in SIMPLICITY if m in out["models"] and out["models"][m]["mean_cv"] is not None]
    out["best"] = max(fitted, key=lambda m: (out["models"][m]["mean_cv"], -SIMPLICITY.index(m))) if fitted else None
    out["favoured"] = None
    if out["best"] and ref is not None and ref["sd_cv"] is not None:
        bar = out["models"][out["best"]]["mean_cv"] - ref["sd_cv"]
        out["favoured"] = next(m for m in fitted if out["models"][m]["mean_cv"] >= bar - EPS)
    return out


# ---------------------------------------------------------------- formatting
def fmt_pct(v, d: int = 1) -> str:
    """0.1234 → "12.3%"; None → "undefined"."""
    return "undefined" if v is None else f"{v * 100:.{d}f}%"


def fmt_num(v, d: int = 3) -> str:
    return "undefined" if v is None else f"{v:.{d}f}"
