"""The SVM demo's stored boundary is what scikit-learn fits to the same made-up points."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent


def test_svm_demo_matches_sklearn():
    code = 'import("./site/assets/demo-math.js").then(m => process.stdout.write(JSON.stringify(m.SVM_DEMO)))'
    demo = json.loads(subprocess.run(["node", "-e", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    X = np.array([[p["x"], p["z"]] for p in demo["points"]])
    y = np.array([p["y"] for p in demo["points"]])
    m = SVC(kernel="linear", C=1e6).fit(X, y)
    assert np.allclose(m.coef_[0], demo["w"], atol=1e-5)
    assert abs(m.intercept_[0] - demo["b"]) < 1e-5
    assert sorted(m.support_.tolist()) == sorted(demo["supportVectors"])


def test_demos_use_no_activity_data():
    """The demos are illustrative: they must not read any data or results file."""
    for f in ["demos.js", "demo-math.js"]:
        text = (ROOT / "site" / "assets" / f).read_text()
        assert not re.search(r"fetch\(|results/|\.csv|starter/", text), f
