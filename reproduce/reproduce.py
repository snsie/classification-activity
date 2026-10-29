"""Rebuild the starter's data and baseline results from the raw public files.

    pip install -r reproduce/requirements.txt       # pinned versions (Python 3.11)
    python reproduce/reproduce.py                    # rebuild starter/data, starter/results and reproduce/reference.json
    python reproduce/reproduce.py --check            # rebuild in a temporary folder and compare with the committed files

Raw files are downloaded into reproduce/raw/ (not committed) and checked against known SHA-256
fingerprints before use (sources.py).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STARTER = ROOT / "starter"
sys.path.insert(0, str(HERE))

import build_results  # noqa: E402
import make_data  # noqa: E402
import sources  # noqa: E402

GENERATED = ["data/readmission.csv", "data/diabetes.csv", "data/datasets.json", "data/README.md",
             "results/readmission.json", "results/diabetes.json", "results/decisions.json"]


def same_json(a, b, path="") -> list[str]:
    """Differences between two JSON values; numbers may differ by floating-point noise only."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [d for k in a for d in same_json(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same_json(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def compare(built: Path, committed: Path) -> list[str]:
    problems = []
    for rel in GENERATED:
        new, old = built / rel, committed / rel
        if not old.exists():
            problems.append(f"{rel}: not committed")
        elif rel.endswith(".json"):
            problems += [f"{rel}{d}" for d in same_json(json.loads(new.read_text()), json.loads(old.read_text()))[:5]]
        elif new.read_bytes() != old.read_bytes():
            problems.append(f"{rel}: contents differ")
    # The manifest's versions may differ by machine; its counts and fingerprints may not.
    a, b = (json.loads((p / "data" / "manifest.json").read_text()) for p in (built, committed))
    problems += [f"data/manifest.json{d}" for d in same_json(a["datasets"], b["datasets"], ".datasets")[:5]]
    problems += [f"data/manifest.json{d}" for d in same_json(a["raw_files"], b["raw_files"], ".raw_files")[:5]]
    return problems


def check_reference(ref: dict) -> list[str]:
    """check.py's REFERENCE_LOGREG must equal the reference run."""
    text = (STARTER / "check.py").read_text(encoding="utf-8")
    got = json.loads(re.search(r"REFERENCE_LOGREG = (\{.*?\})", text).group(1).replace("'", '"'))
    return [] if got == ref else [f"check.py REFERENCE_LOGREG {got} differs from the reference run {ref}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild in a temporary folder and compare with the committed files")
    ap.add_argument("--raw", type=Path, default=HERE / "raw", help="folder for the raw files")
    a = ap.parse_args()

    print("1. Raw files (SHA-256 checked)")
    raw_files = sources.download(a.raw)
    with tempfile.TemporaryDirectory() as tmp:
        target = STARTER
        if a.check:
            target = Path(tmp) / "starter"
            shutil.copytree(STARTER, target, ignore=shutil.ignore_patterns("__pycache__"))
        print("2. Datasets, dictionary and manifest")
        make_data.make_data(a.raw, target / "data", raw_files)
        print("3. Baseline-only results and the logreg reference run")
        build_results.baseline_results(target)
        ref = build_results.reference_run(target)
        problems = check_reference(ref)
        if not a.check:
            (HERE / "reference.json").write_text(json.dumps({"logreg_test_auroc": ref, "tolerance": 0.005}, indent=2) + "\n")
            print(f"Rebuilt {len(GENERATED) + 1} files in {STARTER}.")
            if problems:
                print("Update check.py: " + problems[0])
            return
        problems += compare(target, STARTER)
    if problems:
        print("MISMATCH with the committed files:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"Reproduced: all {len(GENERATED) + 1} generated files match, and check.py's reference matches the reference run.")


if __name__ == "__main__":
    main()
