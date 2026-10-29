"""Parity: m11.py and m11.js give identical results on shared fixtures.

    .venv/bin/python -m pytest tests/test_parity.py

Fixtures cover undefined metrics (no one flagged, one class only, empty), tied probabilities
and tied thresholds, the real development results, and a hand-checked example. Counts must be
equal; floats must agree to 1e-12 (both helpers do the same arithmetic in the same order).
"""

from __future__ import annotations

import json
import math
import random
import subprocess
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
        {"name": "rule tie on specificity", "p": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "y": [1, 1, 1, 1, 0, 0]},
        {"name": "fewer rows than bins", "p": [0.2, 0.7, 0.4], "y": [0, 1, 1]},
    ]
    for n, prev in [(37, 0.3), (250, 0.1), (1000, 0.45)]:
        y = [1 if rng.random() < prev else 0 for _ in range(n)]
        p = [round(min(1, max(0, rng.gauss(0.3 + 0.3 * v, 0.2))), 2) for v in y]  # 2 decimals: many ties
        out.append({"name": f"random n={n}", "p": p, "y": y})
    for path in sorted((SOLUTION / "results").glob("*_dev.json")) if SOLUTION.exists() else []:
        d = json.loads(path.read_text())
        for m in ("logreg", "knn"):
            out.append({"name": f"{path.stem}/{m}", "p": d["models"][m]["oof"], "y": d["y"]})
    for c in out:
        c["thresholds"] = THRESHOLDS
    return out


def dev_files() -> list[dict]:
    base = SOLUTION if SOLUTION.exists() else STARTER
    files = [json.loads(p.read_text()) for p in sorted((base / "results").glob("*_dev.json"))]
    # A file without logreg: the one-SD flags and the favoured model are undefined.
    files.append({"models": {"baseline": files[0]["models"]["baseline"]}})
    return files


def clean(v):
    return json.loads(json.dumps(v, default=str).replace("Infinity", '"inf"'))


def python_side(cs, devs) -> dict:
    out = {"cases": [], "dev": []}
    for c in cs:
        out["cases"].append(clean({
            "name": c["name"],
            "confusion": [m11.confusion(c["p"], c["y"], t) for t in c["thresholds"]],
            "metrics": [m11.metrics(m11.confusion(c["p"], c["y"], t)) for t in c["thresholds"]],
            "auroc": m11.auroc(c["p"], c["y"]),
            "roc": m11.roc(c["p"], c["y"]),
            "calibration": m11.calibration(c["p"], c["y"]),
            "rule": m11.threshold_for_rule(c["p"], c["y"]),
            "rule_80": m11.threshold_for_rule(c["p"], c["y"], 0.8),
            "majority": m11.majority_accuracy(c["y"]),
        }))
    out["dev"] = [clean(m11.fold_summary(d)) for d in devs]
    return out


def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a)} vs {sorted(b)}"]
        return [x for k in a for x in diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
    return [] if math.isclose(a, b, rel_tol=0, abs_tol=1e-12) else [f"{path}: {a!r} vs {b!r}"]


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    cs, devs = cases(), dev_files()
    f = tmp_path_factory.mktemp("parity") / "fixtures.json"
    f.write_text(json.dumps({"cases": cs, "dev_files": devs}))
    js = json.loads(subprocess.run(["node", str(ROOT / "tests" / "parity.mjs"), str(f), str(STARTER / "app" / "m11.js")],
                                   cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    return python_side(cs, devs), js, cs


def test_cases_identical(both):
    py, js, cs = both
    for a, b, c in zip(py["cases"], js["cases"], cs):
        problems = diff(a, b, c["name"])
        assert not problems, problems[:5]


def test_fold_summary_identical(both):
    py, js, _ = both
    problems = diff(py["dev"], js["dev"])
    assert not problems, problems[:5]


def test_undefined_values(both):
    py, _, _ = both
    by = {c["name"]: c for c in py["cases"]}
    m = by["no one flagged at 0.5"]["metrics"][THRESHOLDS.index(0.5)]
    assert m["precision"] is None and m["flagged"] == 0 and m["sensitivity"] == 0
    neg = by["one class only (all negative)"]
    assert neg["auroc"] is None and neg["rule"] is None and neg["metrics"][0]["sensitivity"] is None
    assert by["one class only (all positive)"]["metrics"][0]["specificity"] is None


def test_hand_checked(both):
    py, _, _ = both
    hc = py["cases"][0]
    c = hc["confusion"][THRESHOLDS.index(0.5)]
    assert c == {"tp": 3, "fn": 1, "fp": 2, "tn": 4}
    m = hc["metrics"][THRESHOLDS.index(0.5)]
    assert m["sensitivity"] == 3 / 4 and m["specificity"] == 4 / 6 and m["precision"] == 3 / 5 and m["f1"] == 6 / 9
    # AUROC by counting pairs: positives 0.9, 0.8, 0.6, 0.3 beat 6, 6, 5 and 3 of the 6 negatives: 20 of 24 pairs
    assert hc["auroc"] == pytest.approx(20 / 24)


def test_tie_breaking(both):
    py, _, _ = both
    by = {c["name"]: c for c in py["cases"]}
    # Thresholds 0.7 and 0.6 both have specificity 1; the rule picks the higher sensitivity (0.6 catches 4 of 4).
    r = by["rule tie on specificity"]["rule"]
    assert r["threshold"] == 0.6 and r["metrics"]["sensitivity"] == 1 and r["metrics"]["specificity"] == 1
    # Exactly-equal probability counts as flagged; 0.5000000001 does not flag 0.5.
    t = by["ties across classes"]
    assert t["confusion"][THRESHOLDS.index(0.5)]["tp"] == 3
    assert t["confusion"][THRESHOLDS.index(0.5000000001)]["tp"] == 3  # within the 1e-9 slider tolerance
    assert by["all tied"]["auroc"] == 0.5


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    for c in cases():
        if 0 < sum(c["y"]) < len(c["y"]):
            assert m11.auroc(c["p"], c["y"]) == pytest.approx(roc_auc_score(c["y"], c["p"]), abs=1e-12), c["name"]


def test_rule_is_best_feasible():
    """Brute force: no threshold with sensitivity >= 0.90 has higher specificity than the rule's."""
    for c in cases():
        r = m11.threshold_for_rule(c["p"], c["y"])
        if r is None or r["metrics"]["specificity"] is None:  # one class only
            continue
        for t in set(c["p"]):
            m = m11.metrics(m11.confusion(c["p"], c["y"], t))
            if m["sensitivity"] >= 0.9:
                assert m["specificity"] <= r["metrics"]["specificity"] + 1e-15, (c["name"], t)
