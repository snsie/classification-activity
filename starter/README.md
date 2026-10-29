**App:** _paste your public app link here_

# Judge the Model

An app that helps a clinical lead judge whether any classifier is fit for each job: 30-day readmission and diabetes screening. It reads the results my Python analysis exported to `results/`.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

Submit the link to **this repository** on Canvas, plus your five Canvas answers.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check.py --stage setup
```

| Step | Command | Before you run it |
|---|---|---|
| 2 | — | Fill in `Before results` in `DECISIONS.md`; commit and push |
| 3 | `python -m analysis` | Complete the kNN TODO in `analysis/models.py` |
| 5 | `python -m analysis` (again) | Fill in `Recommendations` and `Design choice and AI use` in `DECISIONS.md` |
| any | `python check.py` | — |

Run the app on your computer: `streamlit run streamlit_app.py` (Streamlit) or `npm start` (Node.js). Build **one** of the two; see `SETUP.md`.

## What's here
| File | What it is |
|---|---|
| `DECISIONS.md` | Your decisions: what matters for each job, your recommendations, and one design choice |
| `AGENT_BRIEF.md` | Rules for your AI coding agent (have it read this first) |
| `SETUP.md` | How to run and publish on Streamlit Community Cloud or GitHub Pages |
| `analysis/` | The workflow; you complete one pipeline in `analysis/models.py` |
| `m11.py` | The shared metric helpers: every number comes from these |
| `streamlit_app.py`, `requirements.txt` | Your app if you choose Streamlit, hosted on Streamlit Community Cloud |
| `web/`, `package.json` | Your app if you choose Node.js (`web/app.js`, with the metric helpers in `web/m11.js`), hosted on GitHub Pages (`index.html` and `.nojekyll` at the top level are for Pages) |
| `results/` | What the analysis exports; the app reads only these files |
| `data/` | The two datasets, the dictionary (`data/README.md`) and `manifest.json` |
| `check.py` | Integrity checks: `python check.py` must pass before you submit |
| `prompts/` | Optional: your AI prompts, one file per session |
