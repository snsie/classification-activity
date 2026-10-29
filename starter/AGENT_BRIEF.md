# Brief for the AI coding agent

Read this before changing any code. The student makes every decision; your job is to build what they ask for **without breaking the rules below**.

## The project
A take-home activity. The student runs a supplied Python workflow that trains five classifiers (plus a prevalence baseline) on three health datasets, then builds **one app with four sections** that helps **a clinical lead** judge whether any model is fit for each job. The clinical lead knows the clinical problem but not the metrics, and must decide whether to use a model.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

| Dataset | Job | Positive class |
|---|---|---|
| `readmission` | Predict readmission within 30 days, at discharge | Readmitted within 30 days |
| `diabetes` | Flag adults for follow-up testing; finding positives matters most | Doctor diagnosis or HbA1c ≥ 6.5% |
| `heart` | Classify disease within a diagnostic-workup cohort | Angiographic disease |

| Section | After using it, the reader can… |
|---|---|
| 1. Beyond Accuracy | name the positive class, see why accuracy alone misleads, and read a confusion matrix as four groups of patients |
| 2. A Threshold for the Job | see that the threshold is a decision separate from the model, and what the job's rule gains and gives up compared with 0.50 |
| 3. Ranking vs. Being Right | read an ROC curve, say what AUROC does and doesn't mean, and tell good ranking apart from trustworthy probabilities |
| 4. A Fair Comparison | judge whether one model is really better than another, and why the test set is opened only once |

Every section has a dataset selector (all three) and a model selector (all five models plus the baseline). A **Recommendations** panel at the top shows the three lines from `results/decisions.json`. The activity website lists each section's required minimum.

## Files
- `analysis/`: the staged workflow (`python -m analysis train | explore | test | export`). `analysis/models.py` holds the six models.
- `m11.py` (Python) and `app/m11.js` (browser): the shared metric helpers. `load`, `oof`, `confusion`, `metrics`, `roc`, `calibration`, `threshold_for_rule` / `thresholdForRule`, `fold_summary` / `foldSummary`.
- `results/<dataset>_dev.json`: development out-of-fold (OOF) probabilities, and training and CV AUROC per fold. `results/<dataset>_test.json`: test probabilities (after `test`). `results/test_record.json`: the saved choices. `results/decisions.json`: the exported `DECISIONS.md`.
- `app/` (Route A, GitHub Pages) and `streamlit_app.py` (Route B): plain starter sections that already read the results.
- `data/`: the CSVs and the dictionary (`data/README.md`), with a "When is it known?" entry for every column.

## Rules for the analysis (do not break these)
1. **Use the supplied `fold` column.** Never reshuffle, never make new folds or splits, and never fit on `test` rows. Only `python -m analysis test` touches the test set, once.
2. **Keep all preprocessing inside a `Pipeline`.** Encoding, imputation and scaling are steps of the model's pipeline, so they are refit inside each training fold. Never scale or impute the whole dataset first.
3. When helping with the two TODOs in `analysis/models.py`, follow `logreg()`: `encode()` → `SimpleImputer(strategy="median")` → `StandardScaler()` → the classifier. kNN uses `KNeighborsClassifier(n_neighbors=25)`; the SVM uses `LinearSVC()`, and the whole pipeline is passed to `with_probabilities()`. Don't change any other model or setting, and don't tune anything.
4. Never add a column as a predictor, and never use `row_id`, `split`, `fold` or `target` as one.

## Rules for the numbers
5. **Compute every number through `m11`.** Never type, hard-code, estimate or invent a number. If a number isn't in `results/`, say so instead of making one up.
6. The app reads **only** the files in `results/`. It never retrains a model.
7. Undefined values (0 ÷ 0) are `null`/`None`. Show them as "undefined", never as 0.
8. Label every table and plot **Development (OOF)** or **Test**. A cross-validated (CV) result is never a test result.

## Rules for decisions
9. **Never write or edit `DECISIONS.md`, and never write Canvas answers.** These are the student's. When a decision is needed (a model, a threshold, a recommendation), ask the student.
10. Don't change `results/` by hand. Rerun the workflow instead.

## Wording (do not break these)
- AUROC describes **ranking**: how often a patient with the outcome gets a higher probability than one without. It is **not** accuracy, and it doesn't choose a threshold.
- An SVM's raw score is **not a probability**. The supplied helper converts it; describe the converted value as an estimated probability.
- CV is not test. Never label a development result as a test result.
- Describe associations, not causes. Never say a feature causes the outcome, or that changing it would change a patient's risk.
- Plain language for a clinical lead: say each metric's denominator in words ("of the patients who were readmitted…").

## How to work with the student
- After writing app code, explain it in **three sentences**: what it shows, where its numbers come from, and how to check one of them.
- Libraries from a CDN are fine for Route A; list every Python import in `requirements.txt` for Route B.
- Route A tip: when a slider moves, redraw only the results, not the slider itself, so dragging keeps working.
- Never commit passwords, tokens or API keys.

## Before you say you're done
Run `python check.py` and fix anything it reports. Then remind the student to commit and push, and to open the app in a private window.
