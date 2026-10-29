# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 250 words in total, not counting the Recommendations lines. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission:
- diabetes:
- heart:

Prediction: on heart, which model will have the highest CV AUROC, and will its lead over logreg be within one SD?

Leak: pick diabetes HbA1c or heart `ca`. Using "When is it known?", explain why it is excluded.

## Before test
Diabetes model: <logreg|knn|rf|lgbm|svm>
Reason (1–2 sentences):

## Verification
Hand calculation: one metric recomputed from a confusion matrix, with arithmetic.

Check: one check you made of the code or outputs, and what you found.

AI and other help: tools used, or "none".

## Recommendations
- readmission: <model and threshold, or "none is adequate for this job"> because ...
- diabetes: ...
- heart: ...
