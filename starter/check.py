"""Integrity checks for the M11 "Judge the Model" activity.

    python check.py                 everything (run this before you submit)
    python check.py --stage setup   data, starter results and folders (step 1)
    python check.py --stage train   + your two pipelines and the development results (step 3)
    python check.py --stage test    + the test run, the saved threshold and the test record (step 5)

Each stage includes the ones before it. Every failure says what to fix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import m11  # noqa: E402

# The default logistic regression's mean CV AUROC on the supplied folds (instructor's reference run).
REFERENCE_LOGREG = {"readmission": 0.6325, "diabetes": 0.8059, "heart": 0.8699}
TOLERANCE = 0.005
STAGES = ["setup", "train", "test", "all"]
AUROC_TOL = 1e-9


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")

    def fail(self, msg: str) -> None:
        print(f"  FAIL  {msg}")
        self.fails.append(msg)

    def warn(self, msg: str) -> None:
        print(f"  note  {msg}")
        self.warns.append(msg)

    def check(self, cond: bool, ok: str, fix: str) -> bool:
        (self.ok if cond else self.fail)(ok if cond else fix)
        return cond


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=AUROC_TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# ---------------------------------------------------------------- the supplied data, read directly
# check.py never relies on analysis/data.py (which you could change): it reads the CSVs itself.
@lru_cache(maxsize=None)
def dictionary() -> dict:
    return json.loads((ROOT / "data" / "datasets.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def table(key: str):
    import pandas as pd
    return pd.read_csv(ROOT / "data" / f"{key}.csv", dtype={"split": str, "fold": "Int64"}, keep_default_na=False, na_values=[""])


def supplied_predictors(key: str) -> list[str]:
    return [c["name"] for c in dictionary()[key]["columns"] if c["kind"] != "id"]


def rows(key: str, split: str) -> tuple[list[int], list[int], list[int]]:
    """row_id, target and fold of the supplied development or test rows."""
    df = table(key)
    part = df[df.split == split]
    folds = part.fold.astype(int).tolist() if split == "dev" else []
    return part.row_id.astype(int).tolist(), part.target.astype(int).tolist(), folds


# ---------------------------------------------------------------- setup
def check_setup(r: Report) -> bool:
    print("Setup")
    if sys.version_info < (3, 11):
        r.fail(f"Python {sys.version.split()[0]} is too old; install Python 3.11 or newer and recreate your environment.")
    try:
        import lightgbm, numpy, pandas, sklearn  # noqa: F401, E401
        r.ok("Python packages import")
    except ImportError as e:
        r.fail(f"Package {e.name} is missing; run `pip install -r requirements.txt` in your environment.")
        return False
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    good = True
    for key in m11.DATASETS:
        m = manifest["datasets"][key]
        path = ROOT / "data" / m["file"]
        if not r.check(path.exists() and sha256(path) == m["sha256"], f"data/{m['file']} matches its fingerprint",
                       f"data/{m['file']} differs from the supplied file; restore it from the starter zip (never edit the data)."):
            good = False
            continue
        df = table(key)
        dev, test = df[df.split == "dev"], df[df.split == "test"]
        disjoint = (len(dev) + len(test) == len(df) and df.row_id.is_unique
                    and set(dev.row_id).isdisjoint(test.row_id) and len(dev) == m["n_dev"] and len(test) == m["n_test"])
        folds_ok = dev.fold.notna().all() and set(dev.fold.astype(int)) == {1, 2, 3, 4, 5} and test.fold.isna().all()
        r.check(disjoint and folds_ok, f"{key}: development and test rows are disjoint; every development row has a fold 1–5",
                f"{key}: the split or fold columns are wrong; restore data/{m['file']} from the starter zip.")
        excluded = {c["name"] for c in dictionary()[key]["excluded"]}
        try:
            from analysis import data
            used = set(data.predictors(key))
        except Exception:  # noqa: BLE001  (a broken analysis/data.py is reported below)
            used = {"<analysis/data.py does not load>"}
        bad = sorted(used - set(supplied_predictors(key)) | (used & ({"row_id", "split", "fold", "target"} | excluded)))
        r.check(not bad, f"{key}: no ID, outcome or leaked column is a predictor",
                f"{key}: {', '.join(bad)} must not be a predictor; restore analysis/data.py from the starter.")
    for key in m11.DATASETS:
        dev = m11.load(f"{key}_dev")
        if r.check(dev is not None, f"results/{key}_dev.json exists",
                   f"results/{key}_dev.json is missing; restore it from the starter zip or run `python -m analysis train`."):
            good &= check_dev_file(r, key, dev, require_all=False)
    from analysis import export
    if export.is_repo():
        r.ok("this folder is a git repository with at least one commit")
    else:
        r.warn("This folder isn't a git repository with a commit yet; follow step 1 on the website (git init, commit, push).")
    return good


def check_dev_file(r: Report, key: str, dev: dict, require_all: bool) -> bool:
    """Rows, folds and labels match the supplied data; every recomputed fold AUROC matches the export."""
    ids, y, folds = rows(key, "dev")
    ok = r.check(dev.get("row_id") == ids and dev.get("y") == y,
                 f"{key}: development results cover exactly the development rows",
                 f"{key}: results/{key}_dev.json does not match the development rows; rerun `python -m analysis train`.")
    ok &= r.check(dev.get("fold") == folds, f"{key}: every out-of-fold prediction comes from the supplied folds",
                  f"{key}: the folds in results/{key}_dev.json differ from data/{key}.csv; use the supplied `fold` column "
                  "(never reshuffle) and rerun `python -m analysis train`.")
    bad = sorted(set(dev.get("features", [])) - set(supplied_predictors(key)))
    ok &= r.check(not bad, f"{key}: only the supplied predictors were used",
                  f"{key}: {', '.join(bad)} were used as predictors; restore analysis/data.py and rerun training.")
    missing = [m for m in m11.MODELS if m not in dev["models"]]
    if require_all:
        ok &= r.check(not missing, f"{key}: development results for all six models",
                      f"{key}: no development results for {', '.join(missing)}; run `python -m analysis train`.")
    if not ok:
        return False
    problems = []
    for name, entry in dev["models"].items():
        p = entry["oof"]
        if len(p) != len(y) or any(v is None or not 0 <= v <= 1 for v in p):
            problems.append(f"{name} does not have exactly one probability per development row")
            continue
        for f in entry["folds"]:
            idx = [i for i, k in enumerate(folds) if k == f["fold"]]
            if not close(m11.auroc([p[i] for i in idx], [y[i] for i in idx]), f["cv_auroc"]):
                problems.append(f"{name} fold {f['fold']} CV AUROC")
        if sorted(f["fold"] for f in entry["folds"]) != [1, 2, 3, 4, 5]:
            problems.append(f"{name} folds")
    return r.check(not problems, f"{key}: every exported fold AUROC matches the one recomputed with m11",
                   f"{key}: {'; '.join(problems[:3])} differ from the recomputed values; never edit results by hand, "
                   "rerun `python -m analysis train`.")


# ---------------------------------------------------------------- train
SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler")
EXPECTED = {  # final estimator class and settings that must not change
    "baseline": ("DummyClassifier", {"strategy": "prior"}),
    "logreg": ("LogisticRegression", {"max_iter": 5000, "C": 1.0}),
    "knn": ("KNeighborsClassifier", {"n_neighbors": 25}),
    "svm": ("LinearSVC", {"C": 1.0}),
    "rf": ("RandomForestClassifier", {"n_estimators": 300, "min_samples_leaf": 3, "random_state": 0}),
    "lgbm": ("LGBMClassifier", {"random_state": 0, "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31}),
}


def check_pipelines(r: Report) -> bool:
    from sklearn.pipeline import Pipeline

    from analysis.models import MODELS
    ok = True
    for name, (cls, params) in EXPECTED.items():
        try:
            model = MODELS[name]()
        except NotImplementedError:
            r.fail(f"{name}() still has its TODO; complete it in analysis/models.py, following the logreg() example.")
            ok = False
            continue
        inner = model
        if name == "svm":
            if not r.check(type(model).__name__ == "CalibratedClassifierCV", "svm is passed to with_probabilities()",
                           "svm() must return with_probabilities(<your pipeline>), because an SVM's raw score is not a probability."):
                ok = False
                continue
            inner = model.estimator
        if name in ("knn", "svm"):
            steps = [type(s).__name__ for _, s in inner.steps] if isinstance(inner, Pipeline) else []
            scaled = any(s in SCALERS for s in steps[:-1])
            if not r.check(isinstance(inner, Pipeline) and scaled and "SimpleImputer" in steps[:-1],
                           f"{name} is a Pipeline with imputation and scaling inside",
                           f"{name}() must return one Pipeline with encode(), SimpleImputer and StandardScaler inside it, "
                           "like logreg(), so scaling is learned from training rows only."):
                ok = False
                continue
        final = inner.steps[-1][1] if isinstance(inner, Pipeline) else inner
        got = final.get_params()
        same = type(final).__name__ == cls and all(got.get(k) == v for k, v in params.items())
        ok &= r.check(same, f"{name} uses {cls} with the fixed settings",
                      f"{name}() must end with {cls}({', '.join(f'{k}={v!r}' for k, v in params.items())}); "
                      "restore the settings in analysis/models.py (no tuning).")
    return ok


def check_train(r: Report, strict: bool) -> bool:
    print("Train")
    ok = check_pipelines(r)
    devs = {}
    for key in m11.DATASETS:
        dev = m11.load(f"{key}_dev")
        if dev is None:
            r.fail(f"results/{key}_dev.json is missing; run `python -m analysis train`.")
            ok = False
            continue
        ok &= check_dev_file(r, key, dev, require_all=True)
        devs[key] = dev
        if "logreg" in dev["models"]:
            got = m11.fold_summary(dev)["models"]["logreg"]["mean_cv"]
            ok &= r.check(abs(got - REFERENCE_LOGREG[key]) <= TOLERANCE,
                          f"{key}: logreg CV AUROC {got:.4f} matches the reference {REFERENCE_LOGREG[key]:.4f} (±{TOLERANCE})",
                          f"{key}: logreg CV AUROC {got:.4f} is not within ±{TOLERANCE} of the reference {REFERENCE_LOGREG[key]:.4f}; "
                          "restore logreg() and the supplied folds, then rerun `python -m analysis train`.")
    if len(devs) == len(m11.DATASETS) and all(len(d["models"]) == len(m11.MODELS) for d in devs.values()):
        ok &= check_order(r, "before_results", {d["commit"] for d in devs.values()}, "results/*_dev.json", "train", strict)
    return ok


def check_order(r: Report, section: str, commits: set, pattern: str, stage: str, strict: bool) -> bool:
    """The decisions were complete in a commit before the first results commit, and haven't changed since."""
    from analysis import export
    name = export.LABELS[section]
    if not export.is_repo():
        r.fail("This folder isn't a git repository; follow step 1 on the website (git init, commit, push).")
        return False
    if len(commits) != 1 or None in commits:
        r.fail(f"The results come from different runs; rerun `python -m analysis {stage}` once for all datasets.")
        return False
    commit = commits.pop()
    then = answers_at(commit, section)
    if not r.check(then is not None, f'"{name}" was committed before `python -m analysis {stage}` ran (commit {commit[:7]})',
                   f'Commit {commit[:7]} has no complete "{name}"; fill it in, commit and push, then rerun `python -m analysis {stage}`.'):
        return False
    now = export.read()[section]["answers"]
    ok = r.check(now == then, f'"{name}" is unchanged since then',
                 f'"{name}" in DECISIONS.md changed after `{stage}` ran; restore the committed answers (git show {commit[:7]}:DECISIONS.md).')
    first = first_results_commit(stage)
    if first is None:
        msg = f"Commit and push results/ after `python -m analysis {stage}`, so your app and the git history show them."
        (r.fail if strict else r.warn)(msg)
        return ok and not strict
    log = (export.git("log", "--reverse", "--format=%H", "--", "DECISIONS.md") or "").split()
    decided = next((c for c in log if answers_at(c, section) is not None), None)
    ok &= r.check(decided is not None and decided != first and export.is_ancestor(decided, first),
                  f'git history: "{name}" (commit {(decided or "")[:7]}) comes before the first {stage} results commit ({first[:7]})',
                  f'In git history the first {stage} results commit ({first[:7]}) does not come after a commit with "{name}" '
                  f"filled in; decisions must be committed before `python -m analysis {stage}` runs.")
    ok &= r.check(answers_at(first, section) == now, f'"{name}" is the same as at the first {stage} results commit',
                  f'"{name}" changed after the first {stage} results commit ({first[:7]}); restore those answers '
                  "and explain any surprise in Canvas instead.")
    return ok


def answers_at(commit: str, section: str) -> dict | None:
    """The section's answers in DECISIONS.md at a commit, or None if it isn't complete there."""
    from analysis import export
    text = export.show(commit, "DECISIONS.md")
    try:
        s = export.parse(text)[section] if text is not None else None
    except export.DecisionsError:
        return None
    return s["answers"] if s and s["complete"] else None


def first_results_commit(stage: str) -> str | None:
    """The oldest commit containing trained development results, or any test results."""
    from analysis import export
    log = export.git("log", "--reverse", "--format=%H", "--", "results/")
    for c in (log or "").split():
        names = (export.git("ls-tree", "--name-only", c, "results/") or "").split()
        for n in names:
            if stage == "test" and n.endswith("_test.json"):
                return c
            if stage == "train" and n.endswith("_dev.json"):
                blob = export.show(c, n)
                try:
                    if blob and len(json.loads(blob)["models"]) > 1:
                        return c
                except (ValueError, KeyError):
                    continue
    return None


# ---------------------------------------------------------------- test
def check_test(r: Report, strict: bool) -> bool:
    print("Test")
    from analysis import export
    ok = True
    tests = {}
    for key in m11.DATASETS:
        t = m11.load(f"{key}_test")
        if not r.check(t is not None, f"results/{key}_test.json exists",
                       f"results/{key}_test.json is missing; run `python -m analysis test` (once)."):
            ok = False
            continue
        test_ids, y_test, _ = rows(key, "test")
        cover = t["row_id"] == test_ids and t["y"] == y_test
        ok &= r.check(cover, f"{key}: test predictions cover the test rows only",
                      f"{key}: results/{key}_test.json does not match the test rows; rerun `python -m analysis test --force` (logged).")
        problems = [m for m in m11.MODELS if m not in t["models"]
                    or len(t["models"][m]["p"]) != len(y_test)
                    or not close(m11.auroc(t["models"][m]["p"], y_test), t["models"][m]["auroc"])]
        ok &= r.check(cover and not problems, f"{key}: every exported test AUROC matches the one recomputed with m11",
                      f"{key}: test results for {', '.join(problems) or 'the rows'} are missing or differ from the recomputed "
                      "values; never edit results by hand.")
        tests[key] = t
    record = m11.load("test_record")
    if not r.check(record is not None, "results/test_record.json exists",
                   "results/test_record.json is missing; run `python -m analysis test` (once)."):
        return False
    decisions = export.read()["before_test"]["answers"]
    ok &= r.check(record["before_test"] == decisions,
                  f"test_record.json matches \"Before test\" (model {decisions['model']})",
                  "results/test_record.json does not match \"Before test\" in DECISIONS.md; restore the answers you committed "
                  "before the test run (never change them afterwards).")
    model = record["before_test"]["model"]
    dev = m11.load("diabetes_dev")
    saved = record["diabetes_threshold"]
    if dev and model in dev.get("models", {}):
        rule = m11.threshold_for_rule(*m11.oof(dev, model))
        same = saved["model"] == model and rule["threshold"] == saved["threshold"] and \
            saved["development"] == {"counts": rule["counts"], "metrics": rule["metrics"]}
        ok &= r.check(same, f"saved diabetes threshold {saved['threshold']} matches the job's rule for {model}",
                      f"the saved diabetes threshold or its counts don't match the rule for {model}; "
                      "rerun `python -m analysis test --force` (logged) and never edit results by hand.")
    if "diabetes" in tests and model in tests["diabetes"]["models"]:
        tp, ty = m11.test_preds(tests["diabetes"], model)
        c = m11.confusion(tp, ty, saved["threshold"])
        adds = sum(saved["test"]["counts"].values()) == len(ty)
        ok &= r.check(adds and saved["test"] == {"counts": c, "metrics": m11.metrics(c)},
                      "diabetes test confusion counts add up and match the recomputed ones",
                      "the diabetes test counts in results/test_record.json differ from the recomputed ones; never edit results by hand.")
    runs = record.get("runs", [])
    if len(runs) > 1:
        r.warn(f"The test set was opened {len(runs)} times (forced runs are logged); explain why in Canvas.")
    ok &= check_order(r, "before_test", {t["commit"] for t in tests.values()} | {record["commit"]},
                      "results/*_test.json", "test", strict)
    return ok


# ---------------------------------------------------------------- everything else
def check_all(r: Report) -> bool:
    print("Submission")
    from analysis import export
    parsed = export.read()
    ok = True
    for key, s in parsed.items():
        ok &= r.check(s["complete"], f'"{export.LABELS[key]}" in DECISIONS.md is complete',
                      f'Fill in "{export.LABELS[key]}" in DECISIONS.md (missing: {", ".join(s["missing"])}).')
    words = export.word_count(parsed)
    if words > export.WORD_LIMIT:
        r.warn(f"DECISIONS.md has {words} words; the limit is {export.WORD_LIMIT}, not counting the Recommendations lines.")
    d = m11.load("decisions")
    ok &= r.check(d is not None and d.get("sections") == {k: v["answers"] for k, v in parsed.items()},
                  "results/decisions.json matches DECISIONS.md",
                  "results/decisions.json is out of date; run `python -m analysis export`, then commit and push.")
    dirty = export.git("status", "--porcelain", "--", "results", "DECISIONS.md")
    ok &= r.check(dirty == "", "results/ and DECISIONS.md are committed",
                  "Commit your latest results/ and DECISIONS.md (git add, git commit, git push).")
    if export.unpushed():
        r.warn("Some commits are not on GitHub yet; run `git push`.")
    first = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0] if (ROOT / "README.md").exists() else ""
    if "http" not in first or "paste your" in first:
        r.warn("Put your public app link on the first line of README.md.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES[:-1], help="check up to this stage only (default: everything)")
    level = STAGES.index(ap.parse_args().stage or "all")
    r = Report()
    from analysis import export
    try:
        check_setup(r)
        if level >= 1:
            check_train(r, strict=level == 3)
        if level >= 2:
            check_test(r, strict=level == 3)
        if level >= 3:
            check_all(r)
    except export.DecisionsError as e:
        r.fail(str(e))
    print()
    if r.fails:
        print(f"{len(r.fails)} check(s) failed. Fix the first one, then run python check.py again.")
        sys.exit(1)
    print("All checks passed." + (f" {len(r.warns)} note(s) above." if r.warns else ""))


if __name__ == "__main__":
    main()
