"""Write the starter's baseline-only results and the reference run.

  * starter/results/<key>.json: the prevalence baseline only, so the starter app runs from the start.
  * starter/results/decisions.json: the untouched DECISIONS.md template, exported.
  * reproduce/reference.json: every classifier's test AUROC per dataset. check.py holds the same
    numbers (REFERENCE_TEST_AUROC) and allows ±0.005; reproduce.py --check confirms they agree.

Both use the starter's own workflow (analysis/, m11.py) with the reference models (reference_models.py),
so they are exactly what students run once their agent has written analysis/models.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"


def _import_starter(starter: Path):
    sys.path.insert(0, str(starter))
    for mod in [m for m in sys.modules if m in ("m11", "reference_models") or m.startswith("analysis")]:
        del sys.modules[mod]
    import m11  # noqa: F401
    import reference_models as models
    from analysis import __main__ as workflow, data, evaluate, export
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
        ref[key] = {}
        for name, build in models.MODELS.items():
            if name == "baseline":
                continue
            r = evaluate.fit_and_score(build, *data.rows(key, "train"), *data.rows(key, "test"))
            ref[key][name] = round(r["test_auroc"], 4)
            print(f"  {name} {key}: test AUROC {ref[key][name]}")
    return ref


if __name__ == "__main__":
    baseline_results()
    ref = reference_run()
    (ROOT / "reproduce" / "reference.json").write_text(json.dumps({"test_auroc": ref, "tolerance": 0.005}, indent=2) + "\n")
