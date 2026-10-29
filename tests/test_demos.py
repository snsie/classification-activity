"""The Models page demos are illustrative: they read no data or results file."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_demos_use_no_activity_data():
    for f in ["demos.js", "demo-math.js"]:
        text = (ROOT / "site" / "assets" / f).read_text()
        assert not re.search(r"fetch\(|results/|\.csv|starter/", text), f
