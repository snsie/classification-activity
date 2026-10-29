"""check.py and the workflow's guards, on a known-good results set and known-bad ones.

    .venv/bin/python instructor/make_solution.py      # builds the known-good repository first
    .venv/bin/python -m pytest tests/test_check.py

Each known-bad case is a clone of the full solution (so its git history is real), changed the way
a student might change it. check.py must fail with a message that says what to fix.
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


def commit_of(repo: Path, subject: str) -> str:
    return run(repo, "git", "log", "--format=%H", f"--grep=^{subject}$", ok=True).stdout.split()[0]


# ---------------------------------------------------------------- known good
def test_known_good_passes(tmp_path):
    repo = clone(tmp_path)
    for stage in ("setup", "train", "test"):
        r = check(repo, "--stage", stage)
        assert r.returncode == 0, r.stdout
    r = check(repo)
    assert r.returncode == 0 and "All checks passed" in r.stdout, r.stdout


def test_untouched_starter_passes_setup(tmp_path):
    dest = tmp_path / "starter"
    shutil.copytree(STARTER, dest, ignore=shutil.ignore_patterns("__pycache__"))
    r = check(dest, "--stage", "setup")
    assert r.returncode == 0, r.stdout
    r = check(dest, "--stage", "train")
    assert r.returncode != 0 and any("TODO" in f for f in fails(r)), r.stdout


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
    assert any(f.startswith("knn() must return one Pipeline") for f in fails(r)), r.stdout


def test_reshuffled_folds(tmp_path):
    repo = clone(tmp_path)
    p = repo / "analysis" / "data.py"
    s = p.read_text().replace(
        "    return dev[predictors(key)].reset_index(drop=True), dev[TARGET].astype(int).tolist(), dev.fold.astype(int).tolist()",
        "    import numpy as np\n"
        "    from sklearn.model_selection import StratifiedKFold\n"
        "    y = dev[TARGET].astype(int).to_numpy()\n"
        "    folds = np.zeros(len(y), int)\n"
        "    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(y, y), 1):\n"
        "        folds[va] = k\n"
        "    return dev[predictors(key)].reset_index(drop=True), y.tolist(), folds.tolist()")
    p.write_text(s)
    run(repo, PY, "-m", "analysis", "train", ok=True)
    commit(repo, "Retrain on my own folds")
    r = check(repo)
    assert r.returncode == 1
    assert any("differ from data/" in f and "never reshuffle" in f for f in fails(r)), r.stdout


def test_test_run_before_before_test_committed(tmp_path):
    repo = clone(tmp_path)
    run(repo, "git", "reset", "-q", "--hard", commit_of(repo, "Train results"), ok=True)
    d = repo / "DECISIONS.md"
    d.write_text(d.read_text().replace("Diabetes model: <logreg|knn|rf|lgbm|svm>", "Diabetes model: logreg")
                 .replace("Reason (1–2 sentences):", "Reason (1–2 sentences): simplest model within one SD."))
    # The workflow refuses...
    r = run(repo, PY, "-m", "analysis", "test", ok=False)
    assert 'Commit your "Before test" answers first' in r.stdout
    # ...so simulate a student who bypassed the guard, then committed decisions and results together.
    code = ("import sys; from analysis import export, __main__ as w; "
            "export.committed = lambda s: (True, ''); sys.argv = ['x', 'test']; w.main()")
    run(repo, PY, "-c", code, ok=True)
    commit(repo, "Before test and test results")
    r = check(repo, "--stage", "test")
    assert r.returncode == 1
    assert any('has no complete "Before test"' in f for f in fails(r)), r.stdout


def test_hand_edited_metric(tmp_path):
    repo = clone(tmp_path)
    p = repo / "results" / "heart_dev.json"
    d = json.loads(p.read_text())
    d["models"]["knn"]["folds"][2]["cv_auroc"] += 0.02
    p.write_text(json.dumps(d))
    commit(repo, "Nudge a number")
    r = check(repo)
    assert r.returncode == 1
    assert any(f.startswith("heart: knn fold 3 CV AUROC") and "never edit results by hand" in f for f in fails(r)), r.stdout


def test_hand_edited_test_counts(tmp_path):
    repo = clone(tmp_path)
    p = repo / "results" / "test_record.json"
    d = json.loads(p.read_text())
    d["diabetes_threshold"]["test"]["counts"]["tp"] += 1
    p.write_text(json.dumps(d))
    commit(repo, "Nudge test counts")
    r = check(repo)
    assert r.returncode == 1 and any("diabetes test counts" in f for f in fails(r)), r.stdout


def test_decision_changed_after_test(tmp_path):
    repo = clone(tmp_path)
    d = repo / "DECISIONS.md"
    d.write_text(d.read_text().replace("Diabetes model: logreg", "Diabetes model: rf"))
    run(repo, PY, "-m", "analysis", "export", ok=True)
    commit(repo, "Change my mind")
    r = check(repo)
    assert r.returncode == 1
    assert any("does not match \"Before test\"" in f for f in fails(r)), r.stdout


def test_before_results_changed_after_training(tmp_path):
    repo = clone(tmp_path)
    d = repo / "DECISIONS.md"
    d.write_text(d.read_text().replace("Random forest will have the highest CV AUROC", "kNN will have the highest CV AUROC"))
    run(repo, PY, "-m", "analysis", "export", ok=True)
    commit(repo, "Rewrite my prediction")
    r = check(repo)
    assert r.returncode == 1 and any('"Before results" in DECISIONS.md changed after `train` ran' in f for f in fails(r)), r.stdout


def test_stale_export(tmp_path):
    repo = clone(tmp_path)
    d = repo / "DECISIONS.md"
    d.write_text(d.read_text().replace("- heart: logistic regression", "- heart: none is adequate; logistic regression"))
    commit(repo, "Edit a recommendation without exporting")
    r = check(repo)
    assert r.returncode == 1 and any("run `python -m analysis export`" in f for f in fails(r)), r.stdout


# ---------------------------------------------------------------- the workflow's guards
def test_guards_before_results(tmp_path):
    dest = tmp_path / "s"
    shutil.copytree(STARTER, dest, ignore=shutil.ignore_patterns("__pycache__"))
    r = run(dest, PY, "-m", "analysis", "train", ok=False)
    assert "Fill in \"Before results\"" in r.stdout
    r = run(dest, PY, "-m", "analysis", "explore", ok=False)
    assert "Run `python -m analysis train` first" in r.stdout
    r = run(dest, PY, "-m", "analysis", "test", ok=False)
    assert "Run `python -m analysis train` first" in r.stdout
    # Filled in but not a repository, then filled in but not committed.
    sol = (SOLUTION / "DECISIONS.md").read_text()
    before = sol[sol.index("## Before results"):sol.index("## Before test")]
    tmpl = (dest / "DECISIONS.md").read_text()
    (dest / "DECISIONS.md").write_text(tmpl[:tmpl.index("## Before results")] + before + tmpl[tmpl.index("## Before test"):])
    r = run(dest, PY, "-m", "analysis", "train", ok=False)
    assert "not a git repository" in r.stdout
    run(dest, "git", "init", "-q", "-b", "main", ok=True)
    run(dest, "git", "config", "user.name", "Test")
    run(dest, "git", "config", "user.email", "test@example.invalid")
    (dest / "DECISIONS.md").write_text(tmpl)
    commit(dest, "Starter")
    (dest / "DECISIONS.md").write_text(tmpl[:tmpl.index("## Before results")] + before + tmpl[tmpl.index("## Before test"):])
    r = run(dest, PY, "-m", "analysis", "train", ok=False)
    assert 'Commit your "Before results" answers first' in r.stdout
    commit(dest, "Before results")
    r = run(dest, PY, "-m", "analysis", "train", ok=False)
    assert "Complete the knn() TODO" in r.stdout
    # Every refusal is one sentence.
    assert r.stdout.strip().count(". ") == 0 or r.stdout.strip().endswith(".")


def test_guards_second_test_run(tmp_path):
    repo = clone(tmp_path)
    r = run(repo, PY, "-m", "analysis", "test", ok=False)
    assert "already been opened once" in r.stdout and "--force" in r.stdout
    run(repo, PY, "-m", "analysis", "test", "--force", ok=True)
    rec = json.loads((repo / "results" / "test_record.json").read_text())
    assert [x["forced"] for x in rec["runs"]] == [False, True]
    commit(repo, "Forced rerun")
    r = check(repo)
    assert "opened 2 times" in r.stdout


def test_every_failure_is_one_sentence(tmp_path):
    repo = clone(tmp_path)
    for p in (repo / "results").glob("*_test.json"):
        p.unlink()
    (repo / "results" / "decisions.json").unlink()
    r = check(repo)
    msgs = fails(r)
    assert msgs
    for m in msgs:
        assert m.endswith((".", ")")) and len(re.findall(r"[.;:] [A-Z]", m)) == 0, m


def test_legitimate_retrain_passes(tmp_path):
    """Rerunning train later (same decisions, same code) must not be flagged."""
    repo = clone(tmp_path)
    run(repo, PY, "-m", "analysis", "train", ok=True)
    commit(repo, "Retrain")
    r = check(repo)
    assert r.returncode == 0, r.stdout
