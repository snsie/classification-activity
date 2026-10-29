"""Write the starter's baseline-only results and the logistic regression reference run.

  * starter/results/<key>.json: the prevalence baseline only, so the starter app runs from the start.
  * starter/results/decisions.json: the untouched DECISIONS.md template, exported.
  * reproduce/reference.json: the default logreg's test AUROC per dataset. check.py holds the same
    numbers (REFERENCE_LOGREG) and allows ±0.005; reproduce.py --check confirms they agree.

Both use the starter's own code (analysis/, m11.py), so they are exactly what students run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"


def _import_starter(starter: Path):
    sys.path.insert(0, str(starter))
    for mod in [m for m in sys.modules if m == "m11" or m.startswith("analysis")]:
        del sys.modules[mod]
    import m11  # noqa: F401
    from analysis import __main__ as workflow, data, evaluate, export, models
    return workflow, data, evaluate, export, models


def baseline_results(starter: Path = STARTER) -> None:
    workflow, data, _, export, models = _import_starter(starter)
    workflow.RESULTS = export.RESULTS = starter / "results"
    for key in data.DATASETS:
        workflow.write(key, workflow.train_and_test(key, {"baseline": models.baseline}))
    out = export.export()
    out["exported_at"] = None  # the template, not anyone's export
    (starter / "results" / "decisions.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  baseline-only results for {', '.join(data.DATASETS)}")


def reference_run(starter: Path = STARTER) -> dict:
    _, data, evaluate, _, models = _import_starter(starter)
    ref = {}
    for key in data.DATASETS:
        r = evaluate.fit_and_score(models.logreg, *data.rows(key, "train"), *data.rows(key, "test"))
        ref[key] = round(r["test_auroc"], 4)
        print(f"  logreg {key}: test AUROC {ref[key]}")
    return ref


if __name__ == "__main__":
    baseline_results()
    ref = reference_run()
    (ROOT / "reproduce" / "reference.json").write_text(json.dumps({"logreg_test_auroc": ref, "tolerance": 0.005}, indent=2) + "\n")
