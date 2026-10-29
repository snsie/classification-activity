"""Both app routes run from the untouched starter and from the private full solution.

    .venv/bin/python -m pytest tests/test_apps.py
    (first time: .venv/bin/python -m playwright install chromium)

Route A (GitHub Pages): each folder is served as a static site and driven in headless Chromium
through every section × dataset × model, watching for console errors and partition labels.
Route B (Streamlit): tests/streamlit_smoke.py runs the same grid with Streamlit's AppTest.
"""

from __future__ import annotations

import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
SOLUTION = ROOT / "instructor" / "solution" / "judge-the-model"
FOLDERS = {"starter": STARTER, "solution": SOLUTION}
SECTIONS = ["beyond-accuracy", "threshold", "ranking", "fair-comparison"]
DATASETS = ["readmission", "diabetes", "heart"]
MODELS = ["baseline", "logreg", "knn", "svm", "rf", "lgbm"]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def serve(folder: Path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(folder)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


@pytest.mark.parametrize("which", ["starter", "solution"])
def test_route_a(browser, which):
    folder = FOLDERS[which]
    if not folder.exists():
        pytest.skip("run instructor/make_solution.py first")
    srv, base = serve(folder)
    try:
        page = browser.new_page()
        errors: list[str] = []
        missing: set[str] = set()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and not m.text.startswith("Failed to load resource") and errors.append(m.text))
        page.on("response", lambda r: r.status >= 400 and missing.add(r.url.rsplit("/", 1)[-1]))
        page.goto(base)  # the root index.html opens app/
        page.wait_for_url("**/app/")
        page.wait_for_selector("#section h2")
        recs = page.inner_text("#recs")
        assert ("not written yet" not in recs) if which == "solution" else ("No recommendations yet" in recs)
        unlabelled = []
        for section in SECTIONS:
            page.goto(f"{base}app/#{section}")
            for d in DATASETS:
                for m in MODELS:
                    page.select_option("#dataset", d)
                    page.select_option("#model", m)
                    page.wait_for_function("document.querySelector('#section h2') !== null")
                    page.wait_for_timeout(30)
                    text = page.inner_text("#section")
                    if "no results" not in text and "Development (OOF)".lower() not in text.lower():
                        unlabelled.append((section, d, m))
        assert not errors, errors[:3]
        assert not unlabelled, unlabelled[:5]
        # Before the test run, only the not-yet-written test files may be missing (m11.load returns null).
        expected = {f"{d}_test.json" for d in DATASETS} | {"test_record.json", "favicon.ico"}
        assert missing <= expected, missing - expected
        if which == "solution":
            assert missing <= {"favicon.ico"}, missing
        if which == "solution":
            page.goto(f"{base}app/#threshold")
            page.select_option("#dataset", "diabetes")
            page.select_option("#model", "logreg")
            page.wait_for_selector("#t")
            before = page.inner_text("#t-live")
            page.fill("#t", "0.2")
            page.dispatch_event("#t", "input")
            assert page.inner_text("#t-out") == "0.20" and page.inner_text("#t-live") != before
            assert "Saved diabetes threshold" in page.inner_text("#section")
            page.goto(f"{base}app/#fair-comparison")
            page.select_option("#dataset", "heart")
            assert "Test" in page.inner_text("#section") and "Within one SD of logreg?" in page.inner_text("#section")
        page.close()
    finally:
        srv.shutdown()


def test_route_a_streamlit_and_pages_agree():
    """The Route A starter and Route B starter show the same baseline numbers (both through m11)."""
    sys.path.insert(0, str(STARTER))
    import m11
    dev = m11.load("heart_dev")
    c = m11.confusion(*m11.oof(dev, "baseline"), 0.5)
    assert c["tp"] == 0 and c["fp"] == 0  # the baseline's probability is below 0.50 on every fold


@pytest.mark.parametrize("which", ["starter", "solution"])
def test_route_b(which, tmp_path):
    folder = FOLDERS[which]
    if not folder.exists():
        pytest.skip("run instructor/make_solution.py first")
    copy = tmp_path / which
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "streamlit_smoke.py"), str(copy)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["runs"] == 72
    assert not out["exceptions"], out["exceptions"][:3]
    assert not out["unlabelled"], out["unlabelled"][:5]
    if which == "solution":
        assert out["slider"] is True
