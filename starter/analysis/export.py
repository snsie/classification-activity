"""Read DECISIONS.md, check git history, and export your decisions. You don't need to change this file.

DECISIONS.md has four fixed sections. Each holds labelled prompts; your answer is the text
after each label. Keep the headings and labels exactly as they are.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import m11

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "DECISIONS.md"
RESULTS = ROOT / "results"
WORD_LIMIT = 250

# section key → (heading, [(answer key, label, template text after the label)])
SECTIONS = {
    "before_results": ("Before results", [
        ("readmission", "- readmission:", ""),
        ("diabetes", "- diabetes:", ""),
        ("heart", "- heart:", ""),
        ("prediction", "Prediction:", "on heart, which model will have the highest CV AUROC, and will its lead over logreg be within one SD?"),
        ("leak", "Leak:", 'pick diabetes HbA1c or heart `ca`. Using "When is it known?", explain why it is excluded.'),
    ]),
    "before_test": ("Before test", [
        ("model", "Diabetes model:", "<logreg|knn|rf|lgbm|svm>"),
        ("reason", "Reason (1–2 sentences):", ""),
    ]),
    "verification": ("Verification", [
        ("hand_calculation", "Hand calculation:", "one metric recomputed from a confusion matrix, with arithmetic."),
        ("check", "Check:", "one check you made of the code or outputs, and what you found."),
        ("ai_help", "AI and other help:", 'tools used, or "none".'),
    ]),
    "recommendations": ("Recommendations", [
        ("readmission", "- readmission:", '<model and threshold, or "none is adequate for this job"> because ...'),
        ("diabetes", "- diabetes:", "..."),
        ("heart", "- heart:", "..."),
    ]),
}
LABELS = {"before_results": "Before results", "before_test": "Before test",
          "verification": "Verification", "recommendations": "Recommendations"}


class DecisionsError(Exception):
    pass


# ---------------------------------------------------------------- parsing
def _sections(text: str) -> dict[str, str]:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1)
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _clean(answer: str, template: str) -> str:
    answer = answer.strip()
    if template and answer.startswith(template):
        answer = answer[len(template):].strip()
    return " ".join(answer.split())


def _filled(answer: str) -> bool:
    if not answer or answer in ("...", "…"):
        return False
    if re.search(r"<model and threshold|<logreg\|", answer):
        return False
    return bool(re.search(r"\w", answer.replace("because ...", "")))


def parse(text: str) -> dict:
    """{section key: {"answers": {key: text}, "complete": bool, "missing": [labels]}}."""
    raw = _sections(text)
    out = {}
    for key, (heading, slots) in SECTIONS.items():
        if heading not in raw:
            raise DecisionsError(f'DECISIONS.md has no "## {heading}" heading; restore it from the template.')
        lines = raw[heading].splitlines()
        starts = []
        for skey, label, template in slots:
            hit = next((i for i, ln in enumerate(lines) if ln.strip().startswith(label)), None)
            if hit is None:
                raise DecisionsError(f'"## {heading}" in DECISIONS.md is missing the label "{label}"; restore it from the template.')
            starts.append(hit)
        answers, missing = {}, []
        for n, (skey, label, template) in enumerate(slots):
            i = starts[n]
            later = [s for s in starts if s > i]
            end = min(later) if later else len(lines)
            first = lines[i].strip()[len(label):]
            ans = _clean("\n".join([first] + lines[i + 1:end]), template)
            answers[skey] = ans if _filled(ans) else ""
            if not answers[skey]:
                missing.append(label.lstrip("- ").rstrip(":"))
        out[key] = {"answers": answers, "complete": not missing, "missing": missing}
    model = out["before_test"]["answers"]["model"]
    if model:
        token = re.sub(r"[`*_.,;:!]", "", model.split()[0]).lower()
        out["before_test"]["answers"]["model"] = token
        if token not in m11.FITTED:
            out["before_test"]["complete"] = False
            out["before_test"]["missing"].append("Diabetes model (one of logreg, knn, rf, lgbm, svm)")
    return out


def read(path: Path = DECISIONS) -> dict:
    if not path.exists():
        raise DecisionsError("DECISIONS.md is missing; restore it from the starter.")
    return parse(path.read_text(encoding="utf-8"))


def word_count(parsed: dict) -> int:
    """Words you wrote, not counting the Recommendations lines."""
    return sum(len(a.split()) for k, s in parsed.items() if k != "recommendations" for a in s["answers"].values())


# ---------------------------------------------------------------- git
def git(*args: str, check: bool = True) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def is_repo() -> bool:
    return git("rev-parse", "--is-inside-work-tree") == "true" and git("rev-parse", "--verify", "HEAD") is not None


def head() -> str | None:
    return git("rev-parse", "HEAD")


def show(commit: str, path: str) -> str | None:
    return git("show", f"{commit}:{path}")


def is_ancestor(a: str, b: str) -> bool:
    try:
        return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT, capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def committed(section: str) -> tuple[bool, str]:
    """Is this DECISIONS.md section filled in, and identical in the last commit? (ok, one-sentence reason)"""
    name = LABELS[section]
    try:
        now = read()[section]
    except DecisionsError as e:
        return False, str(e)
    if not now["complete"]:
        return False, f'Fill in "{name}" in DECISIONS.md first (missing: {", ".join(now["missing"])}).'
    if not is_repo():
        return False, "This folder is not a git repository with a commit yet; follow step 1 on the website (git init, commit, push)."
    text = show("HEAD", "DECISIONS.md")
    if text is None:
        return False, f'Commit DECISIONS.md with "{name}" filled in (git add DECISIONS.md, git commit, git push).'
    try:
        then = parse(text)[section]
    except DecisionsError:
        then = None
    if then is None or then["answers"] != now["answers"]:
        return False, f'Commit your "{name}" answers first: git add DECISIONS.md, then git commit and git push.'
    return True, ""


def unpushed() -> bool:
    """True when the branch has commits that aren't on GitHub yet (or has no upstream)."""
    ahead = git("rev-list", "--count", "@{upstream}..HEAD")
    return ahead is None or ahead != "0"


# ---------------------------------------------------------------- export
def export() -> dict:
    parsed = read()
    out = {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": head() if is_repo() else None,
        "words": word_count(parsed),
        "word_limit": WORD_LIMIT,
        "sections": {k: v["answers"] for k, v in parsed.items()},
        "complete": {k: v["complete"] for k, v in parsed.items()},
        "recommendations": parsed["recommendations"]["answers"],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "decisions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
