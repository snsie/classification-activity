"""End-to-end checks for the built or deployed site.

    npm run build && node scripts/serve.mjs 4174 &     # local preview at http://localhost:4174/classification-activity/
    .venv/bin/python tests/e2e.py http://localhost:4174/classification-activity/

(First time: .venv/bin/python -m playwright install chromium)
Checks links and downloads (including the starter zip), help buttons (keyboard and touch), the
mobile menu, visible keyboard focus, dark mode, page width at several screen sizes, the demos,
numbers against the manifest, saved checklists, console errors, and an automated accessibility
scan (axe-core) of every page in light and dark mode.
"""

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import expect, sync_playwright

BASE = sys.argv[1].rstrip("/") + "/"
PAGES = ["", "models.html", "analysis.html", "beyond-accuracy.html", "threshold.html", "ranking.html",
         "train-vs-test.html", "data.html", "hosting.html", "submit.html", "glossary.html"]
AXE_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
CACHE = Path(__file__).resolve().parent / ".cache"
STARTER = Path(__file__).resolve().parent.parent / "starter"  # the downloadable zip must match it
failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        failures.append(msg)


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def axe_source() -> str:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "axe.min.js"
    if not f.exists():
        f.write_bytes(fetch(AXE_URL)[1])
    return f.read_text()


with sync_playwright() as p:
    browser = p.chromium.launch()

    # ---- every page loads without errors; every internal link and download resolves
    print("Links and downloads")
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    targets: set[str] = set()
    anchors: set[str] = set()
    for path in PAGES:
        resp = page.goto(BASE + path)
        check(resp.status == 200, f"page {path or 'index'} → {resp.status}")
        for attr in ["href", "src"]:
            for v in page.eval_on_selector_all(f"[{attr}]", f"els => els.map(e => e.getAttribute('{attr}'))"):
                if v.startswith(("http", "mailto:", "data:")):
                    continue
                if v.startswith("#"):
                    anchors.add(f"{path}{v}")
                    continue
                full = urljoin(BASE + path, v)
                targets.add(full.split("#")[0])
                if "#" in v:
                    anchors.add(full.replace(BASE, ""))
    check(not errors, f"no console errors on any page {errors[:3]}")
    for t in sorted(targets):
        status, body = fetch(t)
        check(status == 200 and urlparse(t).path.startswith(urlparse(BASE).path), f"{t.replace(BASE, '')} → {status}")
        if t.endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(body))
            names = z.namelist()
            need = ["judge-the-model/check.py", "judge-the-model/DECISIONS.md", "judge-the-model/AGENT_BRIEF.md", "judge-the-model/analysis/data.py",
                    "judge-the-model/m11.py", "judge-the-model/streamlit_app.py", "judge-the-model/web/app.js", "judge-the-model/package.json",
                    "judge-the-model/results/diabetes.json", "judge-the-model/data/manifest.json", "judge-the-model/README.md"]
            check(z.testzip() is None and all(n in names for n in need), f"starter zip valid with {len(names)} files")
            check(not any("__pycache__" in n or "solution" in n or "reproduce" in n for n in names), "starter zip holds no caches or instructor files")
            check(not any(n.endswith(("analysis/models.py", "run_app.sh")) for n in names), "starter zip leaves models.py and run_app.sh to the agent")
            source = {f"judge-the-model/{p.relative_to(STARTER).as_posix()}": p for p in STARTER.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and ".venv" not in p.parts and p.name != ".DS_Store"}
            check(set(names) == set(source) and all(z.read(n) == source[n].read_bytes() for n in names),
                  "starter zip matches starter/ file for file")
            readme = z.read("judge-the-model/README.md").decode()
            check("paste your public app link" in readme.splitlines()[0], "starter README has the app-link placeholder on line 1")
    for a in sorted(anchors):
        path, frag = a.split("#", 1)
        page.goto(BASE + path)
        check(page.locator(f"[id='{frag}']").count() == 1, f"anchor #{frag} exists on {path or 'index'}")
    status, _ = fetch(BASE + "no-such-page")
    check(status == 404, f"missing page returns 404 ({status})")

    # ---- numbers on the site come from the manifest
    print("Numbers from the built data")
    _, raw = fetch(BASE + "starter/data/manifest.json")
    man = json.loads(raw)["datasets"]
    page.goto(BASE + "data.html")
    for key, name in [("readmission", "30-day readmission"), ("diabetes", "Diabetes screening")]:
        m = man[key]
        row = page.locator("table tr", has_text=name).first.inner_text()
        check(f"{m['n_rows']:,}" in row and f"{m['n_pos']:,}" in row and f"{m['prevalence'] * 100:.1f}%" in row,
              f"{key}: data table shows {m['n_rows']:,} rows, {m['n_pos']:,} positive")
    todo = sum(fetch(BASE + p)[1].decode().count("TODO(instructor)") for p in PAGES)
    print(f"  info visible TODO(instructor) marks on the site: {todo}")

    # ---- help buttons: keyboard
    print("Help buttons")
    page.goto(BASE + "ranking.html")
    helps = page.locator("button.help")
    check(helps.count() >= 4, f"{helps.count()} help buttons on the ranking page")
    for path in PAGES:
        page.goto(BASE + path)
        labels = page.locator("button.help").evaluate_all("bs => bs.map(b => b.getAttribute('aria-label'))")
        if labels:
            check(all(lb and lb.startswith("What is") for lb in labels), f"every help button on {path or 'index'} has an accessible label")
    page.goto(BASE + "ranking.html")
    first = page.locator("button.help").first
    first.focus()
    page.keyboard.press("Enter")
    panel = page.locator("#" + first.get_attribute("aria-controls"))
    expect(panel).to_be_visible()
    check(first.get_attribute("aria-expanded") == "true", "Enter opens the explanation")
    check("ROC curve" in panel.inner_text(), "panel shows the right term")
    panel.locator(".help-more").click()
    check(panel.locator(".more").is_visible(), "'Tell me more' reveals more detail")
    page.keyboard.press("Escape")
    check(panel.is_hidden() and first.get_attribute("aria-expanded") == "false", "Escape closes it")
    check(page.evaluate("document.activeElement.classList.contains('help')"), "focus returns to the ? button")
    page.keyboard.press("Space")
    check(panel.is_visible(), "Space opens it too")
    page.locator("button.help").nth(2).click()
    check(panel.is_hidden(), "opening another explanation closes the first")

    # ---- keyboard focus is visible
    print("Keyboard focus")
    page.goto(BASE)
    page.keyboard.press("Tab")
    check(page.evaluate("document.activeElement.classList.contains('skip')"), "first Tab reaches 'Skip to content'")
    outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle + ' ' + getComputedStyle(document.activeElement).outlineWidth")
    check(outline.startswith("solid") and not outline.endswith(" 0px"), f"focused element has a visible outline ({outline})")
    page.keyboard.press("Enter")
    check(page.evaluate("location.hash") == "#main", "skip link jumps to the content")

    # ---- touch and small screen
    print("Touch and mobile menu")
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    m = ctx.new_page()
    m.goto(BASE + "threshold.html")
    nav = m.locator("#site-nav")
    check(nav.is_hidden(), "menu collapsed on a phone")
    m.locator(".menu-toggle").tap()
    check(nav.is_visible() and m.locator(".menu-toggle").get_attribute("aria-expanded") == "true", "tapping Menu opens it")
    h = m.locator("button.help").nth(1)
    h.tap()
    check(m.locator("#" + h.get_attribute("aria-controls")).is_visible(), "tapping ? opens the explanation")
    ctx.close()

    # ---- page width at several screen sizes
    print("Responsive layout")
    for w in [320, 390, 768, 1024, 1440]:
        v = browser.new_page(viewport={"width": w, "height": 900})
        wide = []
        for path in PAGES:
            v.goto(BASE + path)
            v.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            sw = v.evaluate("document.documentElement.scrollWidth")
            if sw > w:
                wide.append(f"{path or 'index'}={sw}")
        check(not wide, f"no sideways scrolling at {w}px {wide}")
        v.close()

    # ---- dark mode
    print("Dark mode")
    for scheme, dark in [("light", False), ("dark", True)]:
        c = browser.new_context(color_scheme=scheme)
        v = c.new_page()
        v.goto(BASE + "models.html")
        bg = v.evaluate("getComputedStyle(document.body).backgroundColor")
        lum = sum(int(x) for x in re.findall(r"\d+", bg)[:3]) / 3
        check((lum < 60) == dark, f"{scheme} scheme: body background {bg}")
        c.close()

    # ---- demos (illustrative numbers from demo-math.js)
    print("Demos")
    page.goto(BASE + "models.html")
    lr = page.locator("#lr-read")
    check("10 of 12" in lr.inner_text(), "logistic regression demo starts with one patient wrong on each side")
    page.locator("#lr-slope").fill("1.5")
    page.locator("#lr-int").fill("-3")
    check("4 of 12" in lr.inner_text() and "1.5 × Feature 1 − 3.0" in lr.inner_text(), "logistic regression demo: sliders move the boundary")
    page.locator("#lr-reset").click()
    check("10 of 12" in lr.inner_text(), "logistic regression demo: reset restores the line")
    read = page.locator("#knn-read")
    shaded = lambda: page.locator("#knn-svg path.region-pos").get_attribute("d")
    before_shade = shaded()
    check("of its 7 nearest neighbours" in read.inner_text(), "kNN demo starts at k = 7")
    page.locator("#knn-k").fill("0")
    check("of its 1 nearest neighbours" in read.inner_text() and "always 0 or 1" in read.inner_text(), "kNN demo: k = 1 explained")
    page.locator("#knn-k").fill("6")
    check("of its 25 nearest neighbours" in read.inner_text(), "kNN demo: k = 25")
    check(shaded() != before_shade, "kNN demo: the shaded prediction changes with k")
    page.locator("#knn-scale").uncheck()
    check("decides almost alone" in read.inner_text(), "kNN demo: unscaled distances explained")
    page.locator("#knn-scale").check()
    before = read.inner_text()
    page.locator("#knn-probe").focus()
    page.keyboard.press("Shift+ArrowRight")
    check(read.inner_text() != before and page.evaluate("document.activeElement.id") == "knn-probe",
          "kNN probe moves with the arrow keys and keeps focus")
    box = page.locator("#knn-svg").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.7)
    check("A = " in read.inner_text() and read.inner_text() != before, "kNN probe moves with a click or tap")
    # ---- checklists persist in this browser
    print("Checklists")
    page.goto(BASE + "submit.html")
    page.locator("#sub-1").check()
    page.locator("#sub-3").check()
    page.reload()
    check(page.locator("#sub-1").is_checked() and page.locator("#sub-3").is_checked(), "ticks survive a reload")
    check("2 of 6 done" in page.locator('.checklist-status[data-for="submit"]').inner_text(), "progress count shown")

    # ---- the starting prompt can be copied
    print("Starting prompt")
    c = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
    v = c.new_page()
    v.goto(BASE + "hosting.html#start-prompt")
    v.get_by_role("button", name="Copy prompt").click()
    copied = v.evaluate("navigator.clipboard.readText()")
    check(copied.startswith("Read AGENT_BRIEF.md") and "check.py --stage setup" in copied, "Copy prompt copies the starting prompt")
    c.close()

    # ---- automated accessibility scan
    print("Accessibility (axe-core, WCAG 2 A/AA)")
    axe = axe_source()
    for scheme in ["light", "dark"]:
        c = browser.new_context(color_scheme=scheme)
        v = c.new_page()
        for path in PAGES:
            v.goto(BASE + path)
            v.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            v.add_script_tag(content=axe)
            res = v.evaluate("axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']}})")
            bad = [f"{x['id']} ({len(x['nodes'])}): {x['nodes'][0]['target']}" for x in res["violations"]]
            check(not bad, f"{scheme} {path or 'index'}: no violations {bad[:3]}")
        c.close()

    browser.close()

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
