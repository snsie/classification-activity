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
| 1 | `sources.py` | Downloads 7 NHANES `.xpt` files (CDC), the Cleveland file and its documentation (UCI), and the UCI readmission zip into `reproduce/raw/` (not committed). Stops if any file's SHA-256 differs from the recorded fingerprint. |
| 2 | `make_data.py` | Builds the three cohorts (readmission and diabetes as in the lecture app's `pipeline/datasets.py`; heart from the UCI documentation, `ca` excluded), adds the fixed split and folds, and writes the CSVs, `datasets.json`, `README.md` (the dictionary) and `manifest.json`. |
| 3 | `build_results.py` | Writes the baseline-only `results/<key>_dev.json` and the empty `results/decisions.json` with the starter's own code, and runs the default logistic regression on the supplied folds: `reference.json`. |

Notes:
- **Exact versions:** `requirements.txt` pins Python 3.11 packages. `--check` compares CSVs and the dictionary byte for byte, and JSON numbers to 1e-9.
- **The reference run** (`reference.json`) must equal `REFERENCE_LOGREG` in `starter/check.py`; `--check` fails if they drift. If you change a model or the folds, update both and rebuild the solution (`instructor/make_solution.py`).
- **On GitHub:** the **Reproduce data** workflow runs `--check` and the m11 parity tests.
