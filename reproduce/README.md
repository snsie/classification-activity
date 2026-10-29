# Reproduce the data and baseline results (instructor only)

Rebuilds `starter/data/` and `starter/results/` from the original public files, and runs the reference run that `starter/check.py` compares against. Not linked from the website.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r reproduce/requirements.txt
python reproduce/reproduce.py --check     # rebuild in a temporary folder and compare with the committed files
python reproduce/reproduce.py             # rebuild in place (then rebuild the site: npm run build)
```

| Step | File | What it does |
|---|---|---|
| 1 | `sources.py` | Downloads 7 NHANES `.xpt` files (CDC) and the UCI readmission zip into `reproduce/raw/` (not committed). Stops if any file's SHA-256 differs from the recorded fingerprint. |
| 2 | `make_data.py` | Builds the two cohorts (as in the lecture app's `pipeline/datasets.py`), adds the fixed train/test split, and writes the CSVs, `datasets.json`, `README.md` (the dictionary) and `manifest.json`. |
| 3 | `build_results.py` | Writes the baseline-only `results/<key>.json` and the empty `results/decisions.json` with the starter's own workflow, and runs every classifier on the supplied split: `reference.json` (their test AUROCs). The models come from `reference_models.py`, built to the spec in `starter/AGENT_BRIEF.md` (the starter ships no `analysis/models.py`; each student's agent writes it). |

Notes:
- **Exact versions:** `requirements.txt` pins Python 3.11 packages. `--check` compares CSVs and the dictionary byte for byte, and JSON numbers to 1e-9.
- **The reference run** (`reference.json`) must equal `REFERENCE_TEST_AUROC` in `starter/check.py`; `--check` fails if they drift. If you change a model or the split, update both, the spec in `AGENT_BRIEF.md`, `check.py`'s `EXPECTED`, `instructor/solution_files/models.py` and rebuild the solution (`instructor/make_solution.py`).
- **On GitHub:** the **Reproduce data** workflow runs `--check` and the m11 tests.
