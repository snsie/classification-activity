"""check.py and the workflow's guard, on a known-good results set and known-bad ones.

    .venv/bin/python instructor/make_solution.py      # builds the known-good repository first
    .venv/bin/python -m pytest tests/test_check.py

Each known-bad case is a clone of the full solution, changed the way a student might change it.
check.py must fail with a message that says what to fix.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
SOLUTION = ROOT / "instructor" / "solution" / "judge-the-model"
PY = sys.executable

pytestmark = pytest.mark.skipif(not SOLUTION.exists(), reason="run instructor/make_solution.py first")


def run(cwd: Path, *args: str, ok: bool | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    if ok is True:
        assert r.returncode == 0, r.stdout + r.stderr
    if ok is False:
        assert r.returncode != 0, r.stdout + r.stderr
    return r


def clone(tmp_path: Path, name: str = "repo") -> Path:
    dest = tmp_path / name
    run(tmp_path, "git", "clone", "-q", str(SOLUTION), str(dest), ok=True)
    run(dest, "git", "config", "user.name", "Test")
    run(dest, "git", "config", "user.email", "test@example.invalid")
    return dest


def commit(repo: Path, msg: str) -> None:
    run(repo, "git", "add", "-A", ok=True)
    run(repo, "git", "commit", "-q", "-m", msg, ok=True)


def check(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return run(repo, PY, "check.py", *args)


def fails(r: subprocess.CompletedProcess) -> list[str]:
    return [ln.strip()[len("FAIL"):].strip() for ln in r.stdout.splitlines() if ln.strip().startswith("FAIL")]


# ---------------------------------------------------------------- known good
def test_known_good_passes(tmp_path):
    repo = clone(tmp_path)
    for stage in ("setup", "results"):
        r = check(repo, "--stage", stage)
        assert r.returncode == 0, r.stdout
    r = check(repo)
    assert r.returncode == 0 and "All checks passed" in r.stdout, r.stdout


def test_untouched_starter_passes_setup(tmp_path):
    dest = tmp_path / "starter"
    shutil.copytree(STARTER, dest, ignore=shutil.ignore_patterns("__pycache__"))
    r = check(dest, "--stage", "setup")
    assert r.returncode == 0, r.stdout
    r = check(dest, "--stage", "results")
    assert r.returncode != 0 and any(f.startswith("analysis/models.py is missing") for f in fails(r)), r.stdout


def test_rerun_passes(tmp_path):
    """Rerunning the analysis later (same code) must not be flagged."""
    repo = clone(tmp_path)
    run(repo, PY, "-m", "analysis", ok=True)
    commit(repo, "Rerun")
    r = check(repo)
    assert r.returncode == 0, r.stdout


# ---------------------------------------------------------------- known bad
def test_preprocessing_outside_pipeline(tmp_path):
    repo = clone(tmp_path)
    p = repo / "analysis" / "models.py"
    s = p.read_text()
    s = re.sub(r"def knn\(\):.*?\n\n\n", 'def knn():\n    """kNN on data scaled beforehand."""\n    return KNeighborsClassifier(n_neighbors=25)\n\n\n', s, flags=re.S)
    p.write_text(s)
    commit(repo, "kNN without a pipeline")
    r = check(repo)
    assert r.returncode == 1
    assert any(f.startswith("knn() must build encode() → SimpleImputer") for f in fails(r)), r.stdout


def test_missing_imputer(tmp_path):
    repo = clone(tmp_path)
    p = repo / "analysis" / "models.py"
    p.write_text(p.read_text().replace('        SimpleImputer(strategy="median"),\n        RandomForestClassifier', "        RandomForestClassifier"))
    commit(repo, "Forest without imputation")
    r = check(repo)
    assert r.returncode == 1 and any(f.startswith("rf() must build encode() → SimpleImputer") for f in fails(r)), r.stdout


def test_missing_model(tmp_path):
    repo = clone(tmp_path)
    p = repo / "analysis" / "models.py"
    p.write_text(p.read_text().replace(', "lgbm": lgbm}', "}"))
    commit(repo, "Drop a model")
    r = check(repo)
    assert r.returncode == 1 and any(f.startswith("MODELS in analysis/models.py must have exactly the keys") for f in fails(r)), r.stdout
    r = run(repo, PY, "-m", "analysis", ok=False)
    assert "must have exactly these keys" in r.stdout


def test_off_spec_model_misses_reference(tmp_path):
    """A setting the structure check doesn't name still moves the test AUROC away from the reference."""
    repo = clone(tmp_path)
    p = repo / "analysis" / "models.py"
    p.write_text(p.read_text().replace("LGBMClassifier(random_state=SEED,", "LGBMClassifier(max_depth=2, random_state=SEED,"))
    run(repo, PY, "-m", "analysis", ok=True)
    commit(repo, "Shallow boosting")
    r = check(repo)
    assert r.returncode == 1 and any("lgbm test AUROC" in f and "make lgbm() match the spec" in f for f in fails(r)), r.stdout


def test_trained_on_test_rows(tmp_path):
    repo = clone(tmp_path)
    p = repo / "analysis" / "data.py"
    s = p.read_text().replace('    part = df[df.split == split]\n',
                              '    part = df if split == "train" else df[df.split == split]  # every row, test rows included\n')
    s = s.replace('    return df.loc[df.split == split, "row_id"]', '    return (df if split == "train" else df.loc[df.split == split])["row_id"]')
    p.write_text(s)
    run(repo, PY, "-m", "analysis", ok=True)
    commit(repo, "Train on everything")
    r = check(repo)
    assert r.returncode == 1
    assert any("does not match the train rows" in f and "never train on test rows" in f for f in fails(r)), r.stdout


def test_changed_model_setting(tmp_path):
    repo = clone(tmp_path)
    p = repo / "analysis" / "models.py"
    p.write_text(p.read_text().replace("KNeighborsClassifier(n_neighbors=25)", "KNeighborsClassifier(n_neighbors=5)"))
    commit(repo, "Tune k")
    r = check(repo)
    assert r.returncode == 1 and any(f.startswith("knn() must build") and "KNeighborsClassifier(n_neighbors=25)" in f for f in fails(r)), r.stdout


def test_hand_edited_auroc(tmp_path):
    repo = clone(tmp_path)
    p = repo / "results" / "diabetes.json"
    d = json.loads(p.read_text())
    d["models"]["knn"]["test_auroc"] += 0.02
    p.write_text(json.dumps(d))
    commit(repo, "Nudge a number")
    r = check(repo)
    assert r.returncode == 1
    assert any(f.startswith("diabetes: knn test AUROC") and "never edit results by hand" in f for f in fails(r)), r.stdout


def test_stale_export(tmp_path):
    repo = clone(tmp_path)
    d = repo / "DECISIONS.md"
    d.write_text(d.read_text().replace("- readmission: none is adequate", "- readmission: still none is adequate"))
    commit(repo, "Edit a recommendation without rerunning")
    r = check(repo)
    assert r.returncode == 1 and any("run `python -m analysis` again" in f for f in fails(r)), r.stdout


def test_incomplete_decisions(tmp_path):
    repo = clone(tmp_path)
    d = repo / "DECISIONS.md"
    d.write_text(re.sub(r"- diabetes: logistic regression.*", "- diabetes: ...", d.read_text()))
    run(repo, PY, "-m", "analysis", ok=True)
    commit(repo, "Drop a recommendation")
    r = check(repo)
    assert r.returncode == 1 and any('Fill in "Recommendations"' in f and "diabetes" in f for f in fails(r)), r.stdout


def test_incomplete_design_choice(tmp_path):
    repo = clone(tmp_path)
    d = repo / "DECISIONS.md"
    d.write_text(re.sub(r"- AI suggestion I turned down:.*", "- AI suggestion I turned down:", d.read_text()))
    run(repo, PY, "-m", "analysis", ok=True)
    commit(repo, "Drop the AI reflection")
    r = check(repo)
    assert r.returncode == 1 and any('Fill in "Design choice and AI use"' in f and "AI suggestion" in f for f in fails(r)), r.stdout


def test_no_launcher(tmp_path):
    repo = clone(tmp_path)
    (repo / "run_app.sh").unlink()
    commit(repo, "Drop the launcher")
    r = check(repo)
    assert r.returncode == 1 and any(f.startswith("Commit a Bash launcher") for f in fails(r)), r.stdout


def test_uncommitted_results(tmp_path):
    repo = clone(tmp_path)
    (repo / "results" / "decisions.json").write_text((repo / "results" / "decisions.json").read_text() + "\n")
    r = check(repo)
    assert r.returncode == 1 and any(f.startswith("Commit your latest results/") for f in fails(r)), r.stdout


# ---------------------------------------------------------------- the workflow's guard
def test_guard_no_models(tmp_path):
    dest = tmp_path / "s"
    shutil.copytree(STARTER, dest, ignore=shutil.ignore_patterns("__pycache__"))
    r = run(dest, PY, "-m", "analysis", ok=False)
    assert "analysis/models.py is missing; ask your agent" in r.stdout


def test_every_failure_is_one_sentence(tmp_path):
    repo = clone(tmp_path)
    (repo / "results" / "decisions.json").unlink()
    p = repo / "results" / "readmission.json"
    d = json.loads(p.read_text())
    del d["models"]["rf"]
    p.write_text(json.dumps(d))
    r = check(repo)
    msgs = fails(r)
    assert msgs
    for m in msgs:
        assert m.endswith((".", ")")) and len(re.findall(r"[.;:] [A-Z]", m)) == 0, m
