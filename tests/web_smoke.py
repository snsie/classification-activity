"""Node-app smoke test for one app folder; run as a subprocess by tests/test_apps.py.

    python tests/web_smoke.py <app folder>

Starts web/server.mjs, clicks through every section × dataset × model in headless Chromium,
and prints a JSON summary in the same shape as tests/streamlit_smoke.py.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

folder = Path(sys.argv[1]).resolve()
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]
LABEL = {"readmission": "30-day readmission", "diabetes": "Diabetes screening"}
out = {"runs": 0, "exceptions": [], "unlabelled": [], "slider": None}

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = subprocess.Popen(["node", "web/server.mjs", str(port)], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
try:
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(f"http://127.0.0.1:{port}/")
        page.wait_for_function("!document.getElementById('content').textContent.includes('Loading')")
        for i, section in enumerate(SECTIONS):
            for dataset in ["readmission", "diabetes"]:
                for model in ["baseline", "logreg", "knn", "rf", "lgbm"]:
                    page.check(f"#sections input[value='{i}']")
                    page.select_option("#dataset", dataset)
                    page.select_option("#model", model)
                    page.wait_for_function("([s, d]) => document.getElementById('section-title').textContent === s"
                                           " && document.getElementById('context').textContent.includes(d)", arg=[section, LABEL[dataset]])
                    page.wait_for_timeout(50)
                    out["runs"] += 1
                    if errors:
                        out["exceptions"].append([section, dataset, model, errors[0][:300]])
                        errors.clear()
                        continue
                    body = page.inner_text("#content")
                    trained = "no results" not in body
                    if trained and "Test ·" not in body and "Training and Test ·" not in body:
                        out["unlabelled"].append([section, dataset, model])
                    if section.startswith("2.") and dataset == "diabetes" and model == "logreg" and page.locator("#t").count():
                        page.fill("#t", "0.2")
                        out["slider"] = not errors and "0.20" in page.inner_text("#content")
        browser.close()
finally:
    server.terminate()
    server.wait(timeout=10)
print(json.dumps(out))
