"""The Streamlit app runs from the untouched starter and from the private full solution.

    .venv/bin/python -m pytest tests/test_apps.py

tests/streamlit_smoke.py runs every section × dataset × model with Streamlit's AppTest,
in a separate process per folder so each folder's m11.py and results/ stay apart.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = {"starter": ROOT / "starter", "solution": ROOT / "instructor" / "solution" / "judge-the-model"}


@pytest.mark.parametrize("which", ["starter", "solution"])
def test_streamlit_app(which, tmp_path):
    folder = FOLDERS[which]
    if not folder.exists():
        pytest.skip("run instructor/make_solution.py first")
    copy = tmp_path / which
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "streamlit_smoke.py"), str(copy)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["runs"] == 32
    assert not out["exceptions"], out["exceptions"][:3]
    assert not out["unlabelled"], out["unlabelled"][:5]
    if which == "solution":
        assert out["slider"] is True
