# Brief for the AI coding agent

Read this before changing any code. The student makes every decision; your job is to build what they ask for **without breaking the rules below**.

## The project
A take-home activity. The student runs a supplied Python workflow that trains three classifiers (plus a prevalence baseline) on two health datasets, then builds **one Streamlit app with four sections** that helps **a clinical lead** judge whether any model is fit for each job. The clinical lead knows the clinical problem but not the metrics, and must decide whether to use a model.

**Central question:** Which classifier would you trust for this job, at what threshold, and with what limits?

| Dataset | Job | Positive class |
|---|---|---|
| `readmission` | Predict readmission within 30 days, at discharge | Readmitted within 30 days |
| `diabetes` | Flag adults for follow-up testing; finding positives matters most | Doctor diagnosis or HbA1c ≥ 6.5% |

| Section | After using it, the reader can… |
|---|---|
| 1. Beyond Accuracy | name the positive class, see why accuracy alone misleads, and read a confusion matrix as four groups of patients |
| 2. A Threshold for the Job | see that the threshold is a decision separate from the model, and what moving it gains and gives up compared with 0.50 |
| 3. Ranking vs. Being Right | read an ROC curve, say what AUROC does and doesn't mean, and tell good ranking apart from trustworthy probabilities |
| 4. Training vs. Test | see why a model is judged on patients it never saw, and spot a model that memorised its training patients |

Every section has a dataset selector (both datasets) and a model selector (all three models plus the baseline). A **Recommendations** panel at the top shows the lines from `results/decisions.json`. The activity website lists each section's required minimum.

## Files
- `analysis/`: the workflow (`python -m analysis`). `analysis/models.py` holds the four models.
- `m11.py`: the shared metric helpers. `load`, `preds`, `confusion`, `metrics`, `majority_accuracy`, `auroc`, `roc`, `calibration`, `auroc_summary`.
- `results/<dataset>.json`: for each model, the training and test probabilities and AUROCs. `results/decisions.json`: the exported `DECISIONS.md`.
- `streamlit_app.py`: plain starter sections that already read the results.
- `data/`: the CSVs and the dictionary (`data/README.md`), with a "When is it known?" entry for every column.

## Rules for the analysis (do not break these)
1. **Use the supplied `split` column.** Never reshuffle, never make new splits, and never fit a model on `test` rows.
2. **Keep all preprocessing inside a `Pipeline`.** Encoding, imputation and scaling are steps of the model's pipeline, so they learn from the training rows only. Never scale or impute the whole dataset first.
3. When helping with the TODO in `analysis/models.py`, follow `logreg()`: `encode()` → `SimpleImputer(strategy="median")` → `StandardScaler()` → `KNeighborsClassifier(n_neighbors=25)`. Don't change any other model or setting, and don't tune anything.
4. Never add a column as a predictor, and never use `row_id`, `split` or `target` as one.

## Rules for the numbers
5. **Compute every number through `m11`.** Never type, hard-code, estimate or invent a number. If a number isn't in `results/`, say so instead of making one up.
6. The app reads **only** the files in `results/`. It never retrains a model.
7. Undefined values (0 ÷ 0) are `None`. Show them as "undefined", never as 0.
8. Label every table and plot **Training** or **Test**. A training result is never a test result.

## Rules for decisions
9. **Never write or edit `DECISIONS.md`, and never write Canvas answers.** These are the student's. When a decision is needed (a model, a threshold, a recommendation), ask the student.
10. Don't change `results/` by hand. Rerun the workflow instead.

## Wording (do not break these)
- AUROC describes **ranking**: how often a patient with the outcome gets a higher probability than one without. It is **not** accuracy, and it doesn't choose a threshold.
- Training results flatter every model. Never present a training result as what to expect on new patients.
- Describe associations, not causes. Never say a feature causes the outcome, or that changing it would change a patient's risk.
- Plain language for a clinical lead: say each metric's denominator in words ("of the patients who were readmitted…").

## How to work with the student
- **Offer options, not one answer.** When the student asks how to show something, give two or three different designs with a sentence on each, and let them choose. Their design choice must be their own idea, and they record one of your suggestions they turned down.
- After writing app code, explain it in **three sentences**: what it shows, where its numbers come from, and how to check one of them.
- List every Python import in `requirements.txt`, so Streamlit Community Cloud can install it.
- Never commit passwords, tokens or API keys.

## Before you say you're done
Run `python check.py` and fix anything it reports. Then remind the student to commit and push, and to open the app in a private window.
