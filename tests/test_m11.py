"""m11.py on fixtures: undefined metrics, ties, a hand-checked example, scikit-learn agreement.

    .venv/bin/python -m pytest tests/test_m11.py

Fixtures cover undefined metrics (no one flagged, one class only, empty), tied probabilities
and tied thresholds, the real results, and a hand-checked example.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
SOLUTION = ROOT / "instructor" / "solution" / "judge-the-model"
sys.path.insert(0, str(STARTER))
import m11  # noqa: E402

THRESHOLDS = [0.0, 0.1, 0.25, 0.3, 0.5, 0.5000000001, 0.7, 0.99, 1.0, 1.01]


def cases() -> list[dict]:
    rng = random.Random(0)
    out = [
        {"name": "hand-checked", "p": [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05], "y": [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]},
        {"name": "no one flagged at 0.5", "p": [0.1, 0.2, 0.3, 0.4], "y": [1, 0, 1, 0]},
        {"name": "one class only (all negative)", "p": [0.2, 0.6, 0.4], "y": [0, 0, 0]},
        {"name": "one class only (all positive)", "p": [0.2, 0.6, 0.4], "y": [1, 1, 1]},
        {"name": "all tied", "p": [0.3] * 8, "y": [1, 0, 0, 1, 0, 0, 0, 1]},
        {"name": "ties across classes", "p": [0.5, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75, 0.1], "y": [1, 0, 1, 0, 1, 1, 0, 0]},
        {"name": "fewer rows than bins", "p": [0.2, 0.7, 0.4], "y": [0, 1, 1]},
    ]
    for n, prev in [(37, 0.3), (250, 0.1), (1000, 0.45)]:
        y = [1 if rng.random() < prev else 0 for _ in range(n)]
        p = [round(min(1, max(0, rng.gauss(0.3 + 0.3 * v, 0.2))), 2) for v in y]  # 2 decimals: many ties
        out.append({"name": f"random n={n}", "p": p, "y": y})
    for key in m11.DATASETS if SOLUTION.exists() else []:
        d = json.loads((SOLUTION / "results" / f"{key}.json").read_text())
        for m in ("logreg", "knn"):
            out.append({"name": f"{key}/{m}", "p": d["models"][m]["test_p"], "y": d["test"]["y"]})
    return out


def by_name() -> dict:
    return {c["name"]: c for c in cases()}


def test_undefined_values():
    c = by_name()["no one flagged at 0.5"]
    m = m11.metrics(m11.confusion(c["p"], c["y"], 0.5))
    assert m["precision"] is None and m["flagged"] == 0 and m["sensitivity"] == 0
    neg = by_name()["one class only (all negative)"]
    assert m11.auroc(neg["p"], neg["y"]) is None
    assert m11.metrics(m11.confusion(neg["p"], neg["y"], 0.0))["sensitivity"] is None
    pos = by_name()["one class only (all positive)"]
    assert m11.metrics(m11.confusion(pos["p"], pos["y"], 0.0))["specificity"] is None
    assert m11.majority_accuracy([]) is None


def test_hand_checked():
    hc = by_name()["hand-checked"]
    c = m11.confusion(hc["p"], hc["y"], 0.5)
    assert c == {"tp": 3, "fn": 1, "fp": 2, "tn": 4}
    m = m11.metrics(c)
    assert m["sensitivity"] == 3 / 4 and m["specificity"] == 4 / 6 and m["precision"] == 3 / 5 and m["f1"] == 6 / 9
    # AUROC by counting pairs: positives 0.9, 0.8, 0.6, 0.3 beat 6, 6, 5 and 3 of the 6 negatives: 20 of 24 pairs
    assert m11.auroc(hc["p"], hc["y"]) == pytest.approx(20 / 24)


def test_ties():
    t = by_name()["ties across classes"]
    # Exactly-equal probability counts as flagged; 0.5000000001 is within the 1e-9 slider tolerance.
    assert m11.confusion(t["p"], t["y"], 0.5)["tp"] == 3
    assert m11.confusion(t["p"], t["y"], 0.5000000001)["tp"] == 3
    a = by_name()["all tied"]
    assert m11.auroc(a["p"], a["y"]) == 0.5


def test_counts_add_up():
    for c in cases():
        for t in THRESHOLDS:
            cc = m11.confusion(c["p"], c["y"], t)
            assert sum(cc.values()) == len(c["y"]), (c["name"], t)


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    for c in cases():
        if 0 < sum(c["y"]) < len(c["y"]):
            assert m11.auroc(c["p"], c["y"]) == pytest.approx(roc_auc_score(c["y"], c["p"]), abs=1e-12), c["name"]
            assert m11.roc(c["p"], c["y"])["tpr"][-1] == 1 and m11.roc(c["p"], c["y"])["fpr"][-1] == 1


def test_calibration_groups_cover_every_row():
    for c in cases():
        if c["p"]:
            bins = m11.calibration(c["p"], c["y"])
            assert sum(b["n"] for b in bins) == len(c["p"]) and len(bins) <= 10


def test_auroc_summary():
    base = SOLUTION if SOLUTION.exists() else STARTER
    for key in m11.DATASETS:
        res = m11.load(key, base / "results")
        s = m11.auroc_summary(res)
        for m, v in s.items():
            assert v["train"] == res["models"][m]["train_auroc"] and v["test"] == res["models"][m]["test_auroc"]
            assert v["gap"] == pytest.approx(v["train"] - v["test"])
        assert s["baseline"]["test"] == 0.5
