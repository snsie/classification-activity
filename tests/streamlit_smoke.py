"""Route B smoke test for one app folder; run as a subprocess by tests/test_apps.py.

    python tests/streamlit_smoke.py <app folder>

Runs every section × dataset × model with Streamlit's AppTest and prints a JSON summary.
A separate process per folder keeps each folder's m11.py and results/ apart.
"""

import json
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

folder = Path(sys.argv[1]).resolve()
os.chdir(folder)
sys.path.insert(0, str(folder))
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Ranking vs. Being Right", "4. Training vs. Test"]
out = {"runs": 0, "exceptions": [], "unlabelled": [], "slider": None}


def texts(at) -> str:
    parts = [e.value for kind in ("caption", "markdown", "info", "warning", "header", "subheader", "title")
             for e in getattr(at, kind)]
    for t in at.table:
        parts.append(t.value.to_string())
    return "\n".join(str(p) for p in parts)


for section in SECTIONS:
    for dataset in ["readmission", "diabetes"]:
        for model in ["baseline", "logreg", "knn", "rf", "lgbm"]:
            at = AppTest.from_file(str(folder / "streamlit_app.py"), default_timeout=120).run()
            at.sidebar.radio[0].set_value(section).run()
            at.sidebar.selectbox[0].set_value(dataset).run()
            at.sidebar.selectbox[1].set_value(model).run()
            out["runs"] += 1
            if at.exception:
                out["exceptions"].append([section, dataset, model, str(at.exception[0].value)[:300]])
                continue
            body = texts(at)
            trained = "has no results yet" not in body
            if trained and "Test ·" not in body and "Training and Test ·" not in body:
                out["unlabelled"].append([section, dataset, model])
            if section.startswith("2.") and dataset == "diabetes" and model == "logreg" and at.slider:
                at.slider[0].set_value(0.2).run()
                out["slider"] = not at.exception and "0.20" in texts(at)
print(json.dumps(out))
