# Judge the Model: M11 take-home activity

**Website: <https://snsie.github.io/classification-activity/>**

Students run a supplied Python workflow that trains four classifiers (plus a prevalence baseline) on the training rows of two health datasets and scores them on the test rows, write their decisions in `DECISIONS.md`, and build one app with four sections, in Streamlit or Node.js (their choice), that helps a clinical lead judge whether any model is fit for each job. They submit a public GitHub repository (app link on the README's first line) and three Canvas answers.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

Modelled on the M10 activity ([regression-activity](https://github.com/snsie/regression-activity)); concepts and wording follow the M11 lecture app ([classification-app](https://github.com/snsie/classification-app)).

## What's here
| Folder | What it is |
|---|---|
| [`starter/`](starter) | **The student starter**, also downloadable as a zip from the website: the two datasets with a fixed train/test split and their dictionary (`data/`), the workflow (`analysis/`), the shared metric helpers (`m11.py`), the app twice (Streamlit `streamlit_app.py`; Node `web/` + `package.json`, no npm dependencies, hosted as static files on GitHub Pages), baseline-only `results/`, `DECISIONS.md`, `AGENT_BRIEF.md`, `SETUP.md` and `check.py`. |
| [`reproduce/`](reproduce) | Instructor only (not linked from the site). Rebuilds `starter/data/` and `starter/results/` from the raw UCI and CDC files, with SHA-256 checks, and runs the logistic-regression reference run that `check.py` compares against. |
| [`site/`](site) | Website source: page fragments (`pages/`), `layout.html`, the glossary (`terms.json`), and styles, behaviour and demos (`assets/`). |
| [`scripts/`](scripts) | `build.mjs` builds the site into `dist/` and zips the starter; `serve.mjs` previews it under the GitHub Pages subpath. |
| [`tests/`](tests) | `m11.py` fixtures, `check.py` known-good and known-bad fixtures, both starter apps, the demos, and an end-to-end site check with an accessibility scan. |
| `instructor/` | **Git-ignored, never pushed.** The private full solution (`make_solution.py` builds it), `EXPECTED.md` with expected values, and the solution app. |

## Build and preview the site
Requires Node 20 or later; there are no npm dependencies.
```bash
npm run build      # → dist/ (fails on unknown glossary terms, missing downloads or numbers not in the manifest)
npm test           # demo math
npm run preview    # http://localhost:4173/classification-activity/
```
Pushing to `main` runs [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds and tests the site, then publishes `dist/` to the `gh-pages` branch (**Settings → Pages → Deploy from a branch → `gh-pages` / root**).

## Tests
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r reproduce/requirements.txt streamlit==1.66.1 playwright pytest
.venv/bin/python -m playwright install chromium
.venv/bin/python reproduce/reproduce.py --check       # data and baseline results reproduce; check.py's reference matches
.venv/bin/python instructor/make_solution.py           # private full solution (instructor machines only)
.venv/bin/python -m pytest tests/                      # m11, check.py fixtures, both apps (Streamlit, Node), demos
npm run build && node scripts/serve.mjs 4174 &
.venv/bin/python tests/e2e.py http://localhost:4174/classification-activity/   # links, downloads, keyboard, dark mode, axe
```
`tests/test_check.py` and the solution half of `tests/test_apps.py` skip when `instructor/` is absent (as on GitHub).

## Data
| Dataset | Source | Licence | Rows (positive) |
|---|---|---|---|
| 30-day readmission | UCI Diabetes 130-US Hospitals 1999–2008 (Strack et al. 2014) | CC BY 4.0 | 20,000 (1,796) |
| Diabetes screening | NHANES 2017–2018, CDC/NCHS | Public domain | 5,257 (997) |

Each CSV has a stratified 80/20 train/test split (`random_state=0`). `starter/data/manifest.json` records sources, fingerprints, versions and counts; `starter/data/README.md` is the dictionary, with a **When is it known?** entry for every column.
