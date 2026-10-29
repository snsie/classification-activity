**App:** _paste your public app link here_

# Judge the Model

An app that helps a clinical lead judge whether any classifier is fit for each job: 30-day readmission, diabetes screening and heart disease. It reads the results my Python analysis exported to `results/`.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

Submit the link to **this repository** on Canvas, plus your four Canvas answers.

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
| 3 | `python -m analysis train` | Complete the kNN and SVM TODOs in `analysis/models.py` |
| 4 | `python -m analysis explore` | — then fill in `Before test`; commit and push |
| 5 | `python -m analysis test` (once) | — then fill in `Verification` and `Recommendations` |
| 5 | `python -m analysis export` | — |
| any | `python check.py` | — |

Run the app on your computer:
- **Route A (GitHub Pages):** `npm run serve` (or `python -m http.server 8000`), then open <http://localhost:8000/app/>.
- **Route B (Streamlit):** `streamlit run streamlit_app.py`

## What's here
| File | What it is |
|---|---|
| `DECISIONS.md` | Your decisions, saved before the results they depend on |
| `AGENT_BRIEF.md` | Rules for your AI coding agent (have it read this first) |
| `SETUP.md` | How to run and publish (GitHub Pages or Streamlit) |
| `analysis/` | The staged workflow; you complete two pipelines in `analysis/models.py` |
| `m11.py`, `app/m11.js` | The shared metric helpers: every number comes from these |
| `index.html`, `app/` | Route A: GitHub Pages (`index.html` opens `app/`) |
| `streamlit_app.py`, `requirements.txt` | Route B: Streamlit Community Cloud |
| `results/` | What the analysis exports; the app reads only these files |
| `data/` | The three datasets, the dictionary (`data/README.md`) and `manifest.json` |
| `check.py` | Integrity checks: `python check.py` must pass before you submit |
| `prompts/` | Optional: your AI prompts, one file per session |
