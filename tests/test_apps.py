"""Both apps (Streamlit and Node) run from the untouched starter and from the private full solution.

    .venv/bin/python -m pytest tests/test_apps.py

tests/streamlit_smoke.py runs every section × dataset × model with Streamlit's AppTest, and
tests/web_smoke.py does the same for the Node app in headless Chromium. Each runs in a separate
process per folder so each folder's m11 and results/ stay apart. test_launcher runs the expected
agent-written Bash launcher for each route the way a student does: bash run_app.sh, then Ctrl+C.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = {"starter": ROOT / "starter", "solution": ROOT / "instructor" / "solution" / "judge-the-model"}


@pytest.mark.parametrize("app", ["streamlit", "web"])
@pytest.mark.parametrize("which", ["starter", "solution"])
def test_app(which, app, tmp_path):
    folder = FOLDERS[which]
    if not folder.exists():
        pytest.skip("run instructor/make_solution.py first")
    if app == "web" and not shutil.which("node"):
        pytest.skip("node not installed")
    copy = tmp_path / which
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    r = subprocess.run([sys.executable, str(ROOT / "tests" / f"{app}_smoke.py"), str(copy)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["runs"] == 4 * 2 * 5  # sections × datasets × models
    assert not out["exceptions"], out["exceptions"][:3]
    assert not out["unlabelled"], out["unlabelled"][:5]
    if which == "solution":
        assert out["slider"] is True


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("route,probe", [("streamlit", "_stcore/health"), ("node", "results/diabetes.json")])
def test_launcher(route, probe, tmp_path):
    """bash run_app.sh starts the chosen app from any folder, serves it on localhost, and stops on Ctrl+C."""
    folder = FOLDERS["solution"]
    if not folder.exists():
        pytest.skip("run instructor/make_solution.py first")
    if route == "node" and not shutil.which("node"):
        pytest.skip("node not installed")
    copy = tmp_path / "repo"
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copy(ROOT / "instructor" / "solution_files" / "launchers" / f"{route}.sh", copy / "run_app.sh")
    port = free_port()
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}  # an activated environment
    proc = subprocess.Popen(["bash", str(copy / "run_app.sh"), str(port)], cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://localhost:{port}/{probe}", timeout=2) as r:
                    assert r.status == 200
                    break
            except OSError:
                assert proc.poll() is None, proc.stdout.read()
                time.sleep(0.5)
        else:
            pytest.fail(f"{route} app never answered on port {port}")
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=5) as r:
            assert r.status == 200
    finally:
        os.killpg(proc.pid, signal.SIGINT)  # Ctrl+C in the student's terminal
        out = proc.communicate(timeout=30)[0]
    assert f"http://localhost:{port}/" in out, out


def close(a, b, tol: float = 1e-12) -> bool:
    """Nested dicts and lists equal, floats within tol."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, z, tol) for x, z in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= tol
    return a == b


def test_web_matches_m11():
    """web/m11.js gives the same numbers as m11.py (on the solution's full results)."""
    folder = FOLDERS["solution"]
    if not (folder / "web").exists():
        pytest.skip("run instructor/make_solution.py first")
    if not shutil.which("node"):
        pytest.skip("node not installed")
    r = subprocess.run(["node", str(ROOT / "tests" / "web_parity.mjs"), str(folder)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    js = json.loads(r.stdout)
    sys.path.insert(0, str(folder))
    try:
        import m11
    finally:
        sys.path.pop(0)
    for k in m11.DATASETS:
        res = m11.load(k, folder / "results")
        assert close(js[k]["summary"], m11.auroc_summary(res))
        for m in res["models"]:
            p, y = m11.preds(res, m)
            assert close(js[k]["models"][m]["metrics"], [m11.metrics(m11.confusion(p, y, t)) for t in (0.12, 0.3, 0.5)])
            assert close(js[k]["models"][m]["calibration"], m11.calibration(p, y))
            assert js[k]["models"][m]["roc_points"] == len(m11.roc(p, y)["fpr"])
