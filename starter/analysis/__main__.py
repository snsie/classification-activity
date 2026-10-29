"""The staged analysis workflow. Run from the starter folder:

    python -m analysis train     fit every model on the supplied folds → results/<dataset>_dev.json
    python -m analysis explore   diabetes: each model's threshold for the job's rule (development only)
    python -m analysis test      refit on all development rows, predict the locked test set once
    python -m analysis export    copy your DECISIONS.md answers to results/decisions.json

Each stage refuses to run until its prerequisite is met, and says why.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

import m11
from analysis import data, evaluate, export
from analysis.models import MODELS

RESULTS = export.RESULTS


def stop(msg: str) -> None:
    print(f"Stopped: {msg}")
    sys.exit(1)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def versions() -> dict:
    return {"python": platform.python_version(),
            **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm"]}}


def write(name: str, obj: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")


def trained() -> bool:
    devs = [m11.load(f"{k}_dev") for k in data.DATASETS]
    return all(d and all(m in d["models"] for m in MODELS) for d in devs)


# ---------------------------------------------------------------- stages
def train(args) -> None:
    ok, why = export.committed("before_results")
    if not ok:
        stop(why)
    for name in ("knn", "svm"):
        try:
            MODELS[name]()
        except NotImplementedError:
            stop(f"Complete the {name}() TODO in analysis/models.py first, following the logreg() example.")
    commit = export.head()
    start = time.time()
    for key in data.DATASETS:
        X, y, folds = data.development(key)
        out = {
            "dataset": key, "partition": "development", "commit": commit, "created": now(), "versions": versions(),
            "n": len(y), "n_pos": sum(y), "features": data.predictors(key),
            "row_id": data.development_ids(key), "fold": folds, "y": y, "models": {},
        }
        for name, build in MODELS.items():
            t0 = time.time()
            out["models"][name] = evaluate.cross_validate(build, X, y, folds)
            cv = [f["cv_auroc"] for f in out["models"][name]["folds"]]
            print(f"  {key:<12} {name:<9} CV AUROC {m11.mean(cv):.3f} ± {m11.sd(cv):.3f}  ({time.time() - t0:.0f} s)", flush=True)
        write(f"{key}_dev", out)
    print(f"Wrote results/<dataset>_dev.json for {len(data.DATASETS)} datasets in {time.time() - start:.0f} s "
          f"(decisions from commit {commit[:7]}).")
    print("Next: python check.py --stage train, then commit and push results/.")


def explore(args) -> None:
    if not trained():
        stop("Run `python -m analysis train` first; there are no development results for every model yet.")
    dev = m11.load("diabetes_dev")
    print("Diabetes, development out-of-fold (OOF) predictions only. The job's rule:")
    print(f"  {m11.RULE_TEXT}\n")
    print(f"  {'Model':<24}{'Threshold':>10}{'Sensitivity':>13}{'Specificity':>13}{'Precision':>11}{'Flagged per 100':>17}{'CV AUROC':>10}")
    summary = m11.fold_summary(dev)
    for name in m11.FITTED:
        p, y = m11.oof(dev, name)
        r = m11.threshold_for_rule(p, y)
        mt = r["metrics"]
        print(f"  {m11.MODEL_LABEL[name]:<24}{r['threshold']:>10.4f}{m11.fmt_pct(mt['sensitivity']):>13}"
              f"{m11.fmt_pct(mt['specificity']):>13}{m11.fmt_pct(mt['precision']):>11}{mt['flagged_per_100']:>17.1f}"
              f"{summary['models'][name]['mean_cv']:>10.3f}")
    print(f"\n  {dev['n']:,} development adults, {dev['n_pos']:,} with diabetes. Test results stay locked.")
    print("Choose your model, fill in `Before test` in DECISIONS.md, then commit and push.")


def test(args) -> None:
    if not trained():
        stop("Run `python -m analysis train` first; the test set opens only after training on the development rows.")
    ok, why = export.committed("before_test")
    if not ok:
        stop(why)
    record = m11.load("test_record")
    if record and not args.force:
        stop("The test set has already been opened once; a second run needs --force, and that run is logged.")
    decisions = export.read()
    choice = decisions["before_test"]["answers"]
    commit = export.head()
    p, y = m11.oof(m11.load("diabetes_dev"), choice["model"])
    rule = m11.threshold_for_rule(p, y)
    start = time.time()
    tests = {}
    for key in data.DATASETS:
        X, y_dev, _ = data.development(key)
        X_test, y_test = data.test(key)
        out = {"dataset": key, "partition": "test", "commit": commit, "created": now(), "versions": versions(),
               "n": len(y_test), "n_pos": sum(y_test), "row_id": data.test_ids(key), "y": y_test, "models": {}}
        for name, build in MODELS.items():
            out["models"][name] = evaluate.fit_and_test(build, X, y_dev, X_test, y_test)
            print(f"  {key:<12} {name:<9} test AUROC {m11.fmt_num(out['models'][name]['auroc'])}", flush=True)
        write(f"{key}_test", out)
        tests[key] = out
    tp, ty = m11.test_preds(tests["diabetes"], choice["model"])
    runs = (record or {}).get("runs", []) + [{"time": now(), "commit": commit, "forced": bool(record)}]
    write("test_record", {
        "commit": commit,
        "before_test": {"model": choice["model"], "reason": choice["reason"]},
        "diabetes_threshold": {
            "model": choice["model"], "rule": m11.RULE_TEXT, "threshold": rule["threshold"],
            "development": {"counts": rule["counts"], "metrics": rule["metrics"]},
            "test": {"counts": m11.confusion(tp, ty, rule["threshold"]), "metrics": m11.metrics(m11.confusion(tp, ty, rule["threshold"]))},
        },
        "runs": runs,
    })
    print(f"Opened the test set once in {time.time() - start:.0f} s. Diabetes model {choice['model']}, "
          f"saved threshold {rule['threshold']:.4f} (from development data only).")
    if record:
        print("This was a forced second run; it is logged in results/test_record.json.")
    print("Don't change any decision now. Fill in Verification and Recommendations, run `python -m analysis export`, then commit and push.")


def run_export(args) -> None:
    try:
        out = export.export()
    except export.DecisionsError as e:
        stop(str(e))
    done = [export.LABELS[k] for k, v in out["complete"].items() if v]
    print(f"Wrote results/decisions.json. Complete sections: {', '.join(done) or 'none yet'}.")
    if out["words"] > export.WORD_LIMIT:
        print(f"Note: {out['words']} words, over the {export.WORD_LIMIT}-word limit (Recommendations lines not counted).")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("train", help="fit every model on the supplied folds")
    sub.add_parser("explore", help="diabetes thresholds for the job's rule (development only)")
    t = sub.add_parser("test", help="open the locked test set once")
    t.add_argument("--force", action="store_true", help="run again after the first test run (logged)")
    sub.add_parser("export", help="copy DECISIONS.md to results/decisions.json")
    args = ap.parse_args()
    try:
        {"train": train, "explore": explore, "test": test, "export": run_export}[args.stage](args)
    except export.DecisionsError as e:
        stop(str(e))


if __name__ == "__main__":
    main()
