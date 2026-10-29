"""Integrity checks for the M11 "Judge the Model" activity.

    python check.py                   everything (run this before you submit)
    python check.py --stage setup     the data and the starter results (step 1)
    python check.py --stage results   + your kNN pipeline and the results of `python -m analysis` (step 3)

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

# The default logistic regression's test AUROC (instructor's reference run).
REFERENCE_LOGREG = {"readmission": 0.6396, "diabetes": 0.8266}
TOLERANCE = 0.005
STAGES = ["setup", "results", "all"]
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
    return pd.read_csv(ROOT / "data" / f"{key}.csv", dtype={"split": str}, keep_default_na=False, na_values=[""])


def supplied_predictors(key: str) -> list[str]:
    return [c["name"] for c in dictionary()[key]["columns"] if c["kind"] != "id"]


def rows(key: str, split: str) -> tuple[list[int], list[int]]:
    """row_id and target of the supplied training or test rows."""
    part = table(key)[table(key).split == split]
    return part.row_id.astype(int).tolist(), part.target.astype(int).tolist()


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
        train, test = df[df.split == "train"], df[df.split == "test"]
        r.check(len(train) + len(test) == len(df) and df.row_id.is_unique and len(train) == m["n_train"] and len(test) == m["n_test"],
                f"{key}: every row is a training or a test row, never both",
                f"{key}: the split column is wrong; restore data/{m['file']} from the starter zip.")
        excluded = {c["name"] for c in dictionary()[key]["excluded"]}
        try:
            from analysis import data
            used = set(data.predictors(key))
        except Exception:  # noqa: BLE001  (a broken analysis/data.py is reported below)
            used = {"<analysis/data.py does not load>"}
        bad = sorted(used - set(supplied_predictors(key)) | (used & ({"row_id", "split", "target"} | excluded)))
        r.check(not bad, f"{key}: no ID, outcome or leaked column is a predictor",
                f"{key}: {', '.join(bad)} must not be a predictor; restore analysis/data.py from the starter.")
    for key in m11.DATASETS:
        res = m11.load(key)
        if r.check(res is not None, f"results/{key}.json exists",
                   f"results/{key}.json is missing; restore it from the starter zip or run `python -m analysis`."):
            good &= check_results_file(r, key, res, require_all=False)
    from analysis import export
    if export.is_repo():
        r.ok("this folder is a git repository with at least one commit")
    else:
        r.warn("This folder isn't a git repository with a commit yet; follow step 1 on the website (git init, commit, push).")
    return good


def check_results_file(r: Report, key: str, res: dict, require_all: bool) -> bool:
    """Rows and labels match the supplied data; every recomputed AUROC matches the export."""
    ok = True
    for part in ("train", "test"):
        ids, y = rows(key, part)
        ok &= r.check(res.get(part, {}).get("row_id") == ids and res[part].get("y") == y,
                      f"{key}: {m11.PARTITION_LABEL[part].lower()} results cover exactly the {part} rows",
                      f"{key}: results/{key}.json does not match the {part} rows in data/{key}.csv; use the supplied `split` "
                      "column (never reshuffle, never train on test rows) and rerun `python -m analysis`.")
    bad = sorted(set(res.get("features", [])) - set(supplied_predictors(key)))
    ok &= r.check(not bad, f"{key}: only the supplied predictors were used",
                  f"{key}: {', '.join(bad)} were used as predictors; restore analysis/data.py and rerun `python -m analysis`.")
    missing = [m for m in m11.MODELS if m not in res["models"]]
    if require_all:
        ok &= r.check(not missing, f"{key}: results for all {len(m11.MODELS)} models",
                      f"{key}: no results for {', '.join(missing)}; run `python -m analysis`.")
    if not ok:
        return False
    problems = []
    for name, entry in res["models"].items():
        for part in ("train", "test"):
            p, y = m11.preds(res, name, part)
            if len(p) != len(y) or any(v is None or not 0 <= v <= 1 for v in p):
                problems.append(f"{name} does not have exactly one {part} probability per row")
            elif not close(m11.auroc(p, y), entry[f"{part}_auroc"]):
                problems.append(f"{name} {part} AUROC")
    return r.check(not problems, f"{key}: every exported AUROC matches the one recomputed with m11",
                   f"{key}: {'; '.join(problems[:3])} differ from the recomputed values; never edit results by hand, "
                   "rerun `python -m analysis`.")


# ---------------------------------------------------------------- results
SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler")
EXPECTED = {  # final estimator class and settings that must not change
    "baseline": ("DummyClassifier", {"strategy": "prior"}),
    "logreg": ("LogisticRegression", {"max_iter": 5000, "C": 1.0}),
    "knn": ("KNeighborsClassifier", {"n_neighbors": 25}),
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
        if name == "knn":
            steps = [type(s).__name__ for _, s in model.steps] if isinstance(model, Pipeline) else []
            scaled = any(s in SCALERS for s in steps[:-1])
            if not r.check(isinstance(model, Pipeline) and scaled and "SimpleImputer" in steps[:-1],
                           "knn is a Pipeline with imputation and scaling inside",
                           "knn() must return one Pipeline with encode(), SimpleImputer and StandardScaler inside it, "
                           "like logreg(), so scaling is learned from training rows only."):
                ok = False
                continue
        final = model.steps[-1][1] if isinstance(model, Pipeline) else model
        got = final.get_params()
        same = type(final).__name__ == cls and all(got.get(k) == v for k, v in params.items())
        ok &= r.check(same, f"{name} uses {cls} with the fixed settings",
                      f"{name}() must end with {cls}({', '.join(f'{k}={v!r}' for k, v in params.items())}); "
                      "restore the settings in analysis/models.py (no tuning).")
    return ok


def check_results(r: Report) -> bool:
    print("Results")
    ok = check_pipelines(r)
    for key in m11.DATASETS:
        res = m11.load(key)
        if res is None:
            r.fail(f"results/{key}.json is missing; run `python -m analysis`.")
            ok = False
            continue
        ok &= check_results_file(r, key, res, require_all=True)
        if "logreg" in res["models"]:
            got = res["models"]["logreg"]["test_auroc"]
            ok &= r.check(abs(got - REFERENCE_LOGREG[key]) <= TOLERANCE,
                          f"{key}: logreg test AUROC {got:.4f} matches the reference {REFERENCE_LOGREG[key]:.4f} (±{TOLERANCE})",
                          f"{key}: logreg test AUROC {got:.4f} is not within ±{TOLERANCE} of the reference {REFERENCE_LOGREG[key]:.4f}; "
                          "restore logreg() and the supplied data, then rerun `python -m analysis`.")
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
        r.warn(f"Before results in DECISIONS.md has {words} words; the limit is {export.WORD_LIMIT}.")
    d = m11.load("decisions")
    ok &= r.check(d is not None and d.get("sections") == {k: v["answers"] for k, v in parsed.items()},
                  "results/decisions.json matches DECISIONS.md",
                  "results/decisions.json is out of date; run `python -m analysis` again, then commit and push.")
    if not export.is_repo():
        r.fail("This folder isn't a git repository; follow step 1 on the website (git init, commit, push).")
        return False
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
            check_results(r)
        if level >= 2:
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
