// Shared metric helpers for the M11 "Judge the Model" activity (browser version of m11.py).
// Every number your app shows comes from these functions, so the browser and Python always agree.
// Ported from the M11 lecture app (web/src/data.ts) and tested against m11.py on shared fixtures.
//
//   import * as m11 from "./m11.js";
//   const dev = await m11.load("diabetes_dev");   // results/diabetes_dev.json, or null if missing
//   const [p, y] = m11.oof(dev, "logreg");        // development out-of-fold probabilities and labels
//   const c = m11.confusion(p, y, 0.5);           // {tp, fn, fp, tn}; flag when p >= threshold
//   m11.metrics(c);                               // sensitivity, specificity, ... (null = undefined)
//   m11.roc(p, y);                                // ROC points and AUROC
//   m11.calibration(p, y);                        // 10 equal-size bins
//   m11.thresholdForRule(p, y);                   // the diabetes rule (sensitivity >= 0.90)
//   m11.foldSummary(dev);                         // mean ± SD per model, and the one-SD flag
//
// Undefined values (for example precision when no one is flagged, 0 ÷ 0) are null.
// Show them as "undefined", never as 0.

export const DATASETS = ["readmission", "diabetes", "heart"];
export const MODELS = ["baseline", "logreg", "knn", "svm", "rf", "lgbm"];
export const FITTED = ["logreg", "knn", "svm", "rf", "lgbm"];
// Simplest first: the §16 rule prefers the earliest model that is within one SD of the best.
export const SIMPLICITY = ["logreg", "knn", "svm", "rf", "lgbm"];
export const MODEL_LABEL = {
  baseline: "Prevalence baseline",
  logreg: "Logistic regression",
  knn: "k-nearest neighbours",
  svm: "Support vector machine",
  rf: "Random forest",
  lgbm: "Gradient boosting",
};
export const DATASET_LABEL = { readmission: "30-day readmission", diabetes: "Diabetes screening", heart: "Heart disease" };
// The diabetes job's rule, fixed by the activity.
export const RULE_SENSITIVITY = 0.9;
export const RULE_TEXT =
  "Among development out-of-fold thresholds where sensitivity is at least 0.90, the one with the highest specificity (ties: the higher sensitivity).";

// Thresholds can come from sliders (0.1 + 0.2 ≠ 0.3), so p within 1e-9 below t still counts as flagged.
const EPS = 1e-9;

// ---- loading
// Paths are resolved from this file, so results/ is found wherever the page lives (repo root or app/).
const RESULTS = new URL("../results/", import.meta.url);
const cache = {};

/** results/<name>.json, e.g. "heart_dev", "heart_test", "decisions" or "test_record". null if it doesn't exist yet. */
export function load(name) {
  if (!cache[name]) {
    cache[name] = fetch(new URL(`${name}.json`, RESULTS), { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
  }
  return cache[name];
}

/** Development out-of-fold probabilities and labels for one model. */
export const oof = (dev, model) => [dev.models[model].oof, dev.y];
/** Test-set probabilities and labels for one model. */
export const testPreds = (test, model) => [test.models[model].p, test.y];

// ---- the confusion matrix and its metrics
/** Counts at one threshold. A patient is flagged (predicted positive) when p >= threshold. */
export function confusion(p, y, threshold) {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (let i = 0; i < p.length; i++) {
    const flag = p[i] >= threshold - EPS;
    if (y[i] === 1) flag ? tp++ : fn++;
    else flag ? fp++ : tn++;
  }
  return { tp, fn, fp, tn };
}

const ratio = (a, b) => (b === 0 ? null : a / b);

/**
 * Every metric from the four counts. Denominators, in words:
 * sensitivity (TPR) = TP ÷ (TP + FN): of the patients who had the outcome, the share flagged
 * specificity      = TN ÷ (TN + FP): of the patients who did not, the share not flagged
 * precision        = TP ÷ (TP + FP): of the patients flagged, the share who had the outcome
 * FPR              = FP ÷ (FP + TN): of the patients who did not, the share flagged (1 − specificity)
 * F1               = 2TP ÷ (2TP + FP + FN): the harmonic mean of precision and sensitivity
 */
export function metrics(c) {
  const { tp, fn, fp, tn } = c;
  const n = tp + fn + fp + tn;
  const flagged = tp + fp;
  return {
    n,
    positives: tp + fn,
    negatives: fp + tn,
    flagged,
    flagged_per_100: ratio(100 * flagged, n),
    prevalence: ratio(tp + fn, n),
    accuracy: ratio(tp + tn, n),
    sensitivity: ratio(tp, tp + fn),
    specificity: ratio(tn, tn + fp),
    precision: ratio(tp, tp + fp),
    fpr: ratio(fp, fp + tn),
    f1: ratio(2 * tp, 2 * tp + fp + fn),
  };
}

/** Accuracy of "the most common class for everyone". */
export function majorityAccuracy(y) {
  const n = y.length;
  if (n === 0) return null;
  let pos = 0;
  for (const v of y) if (v === 1) pos++;
  return Math.max(pos, n - pos) / n;
}

// ---- ROC and AUROC
/** Indices by descending probability; ties by index, so every run gives the same order. */
function descOrder(p) {
  const idx = Array.from({ length: p.length }, (_, i) => i);
  idx.sort((a, b) => p[b] - p[a] || a - b);
  return idx;
}

/**
 * AUROC: the chance that a random positive gets a higher probability than a random negative (ties count ½).
 * It describes ranking, not accuracy. null if only one class is present.
 */
export function auroc(p, y) {
  const n = p.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b] || a - b);
  let nPos = 0, rankSum = 0;
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && p[idx[j + 1]] === p[idx[i]]) j++;
    const mid = (i + j) / 2 + 1; // ranks are 1-based
    for (let k = i; k <= j; k++) {
      if (y[idx[k]] === 1) {
        nPos++;
        rankSum += mid;
      }
    }
    i = j + 1;
  }
  const nNeg = n - nPos;
  if (nPos === 0 || nNeg === 0) return null;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/**
 * The ROC curve: one (FPR, TPR) point per distinct threshold, from flagging no one (0, 0) to everyone (1, 1).
 * thr[k] is the threshold for point k (flag p >= thr[k]); thr[0] is Infinity.
 */
export function roc(p, y) {
  const order = descOrder(p);
  let nPos = 0;
  for (const v of y) if (v === 1) nPos++;
  const nNeg = y.length - nPos;
  const fpr = [0], tpr = [0], thr = [Infinity];
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; ) {
    const v = p[order[k]];
    while (k < order.length && p[order[k]] === v) {
      if (y[order[k]] === 1) tp++;
      else fp++;
      k++;
    }
    fpr.push(nNeg ? fp / nNeg : 0);
    tpr.push(nPos ? tp / nPos : 0);
    thr.push(v);
  }
  return { fpr, tpr, thr, auroc: auroc(p, y) };
}

// ---- calibration
/**
 * Calibration plot data: patients sorted by probability and cut into `bins` equal-size groups.
 * For each: the mean predicted probability and the observed share with the outcome.
 * On a well-calibrated model the two are close (the points sit near the diagonal).
 */
export function calibration(p, y, bins = 10) {
  const n = p.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b] || a - b);
  const groups = Array.from({ length: bins }, () => []);
  idx.forEach((i, r) => groups[Math.min(bins - 1, Math.floor((r * bins) / n))].push(i));
  const out = [];
  for (const g of groups) {
    if (!g.length) continue;
    let sp = 0, sy = 0, lo = Infinity, hi = -Infinity;
    for (const i of g) {
      sp += p[i];
      sy += y[i];
      lo = Math.min(lo, p[i]);
      hi = Math.max(hi, p[i]);
    }
    out.push({ lo, hi, mean_pred: sp / g.length, observed: sy / g.length, n: g.length });
  }
  return out;
}

// ---- the diabetes threshold rule
/**
 * The diabetes job's rule: among the distinct probabilities used as thresholds (flag p >= t), keep those
 * with sensitivity >= minSensitivity and choose the one with the highest specificity. If two tie on
 * specificity, choose the higher sensitivity. Use it on development out-of-fold probabilities only.
 */
export function thresholdForRule(p, y, minSensitivity = RULE_SENSITIVITY) {
  const order = descOrder(p);
  let nPos = 0;
  for (const v of y) if (v === 1) nPos++;
  const nNeg = y.length - nPos;
  if (nPos === 0) return null;
  let best = null;
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; ) {
    // lower the threshold one distinct probability at a time
    const t = p[order[k]];
    while (k < order.length && p[order[k]] === t) {
      if (y[order[k]] === 1) tp++;
      else fp++;
      k++;
    }
    const sens = tp / nPos;
    const spec = nNeg ? (nNeg - fp) / nNeg : 0;
    if (sens >= minSensitivity - EPS && (best === null || spec > best[0] || (spec === best[0] && sens > best[1]))) {
      best = [spec, sens, t];
    }
  }
  if (best === null) return null;
  const t = best[2];
  const c = confusion(p, y, t);
  return { threshold: t, counts: c, metrics: metrics(c), min_sensitivity: minSensitivity };
}

// ---- comparing models across folds
export const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);

/** Sample standard deviation (n − 1), as in the lecture. */
export function sd(a) {
  if (a.length < 2) return null;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}

/**
 * Mean ± SD of training and cross-validated (CV) AUROC across the 5 folds, for every model in a
 * development results file, plus the §16 comparison with logistic regression:
 *   lead_over_logreg = model's mean CV AUROC − logreg's mean CV AUROC
 *   within_one_sd    = |lead| <= logreg's fold SD (so the difference is within fold-to-fold noise)
 *   best             = fitted model with the highest mean CV AUROC
 *   favoured         = the simplest model (logreg < knn < svm < rf < lgbm) whose mean CV AUROC is within
 *                      one logreg SD of the best: the model the lecture's rule prefers
 */
export function foldSummary(dev) {
  const out = { models: {} };
  const ok = (a) => a.filter((v) => v !== null);
  for (const [m, entry] of Object.entries(dev.models)) {
    const cv = entry.folds.map((f) => f.cv_auroc);
    const tr = entry.folds.map((f) => f.train_auroc);
    out.models[m] = {
      cv_folds: cv, train_folds: tr,
      mean_cv: mean(ok(cv)), sd_cv: sd(ok(cv)),
      mean_train: mean(ok(tr)), sd_train: sd(ok(tr)),
    };
  }
  const ref = out.models.logreg;
  for (const s of Object.values(out.models)) {
    if (!ref || s.mean_cv === null || ref.sd_cv === null) {
      s.lead_over_logreg = s.within_one_sd = null;
    } else {
      s.lead_over_logreg = s.mean_cv - ref.mean_cv;
      s.within_one_sd = Math.abs(s.lead_over_logreg) <= ref.sd_cv + EPS;
    }
  }
  const fitted = SIMPLICITY.filter((m) => out.models[m] && out.models[m].mean_cv !== null);
  out.best = null;
  for (const m of fitted) if (out.best === null || out.models[m].mean_cv > out.models[out.best].mean_cv) out.best = m;
  out.favoured = null;
  if (out.best && ref && ref.sd_cv !== null) {
    const bar = out.models[out.best].mean_cv - ref.sd_cv;
    out.favoured = fitted.find((m) => out.models[m].mean_cv >= bar - EPS);
  }
  return out;
}

// ---- formatting
/** 0.1234 → "12.3%"; null → "undefined". */
export const fmtPct = (v, d = 1) => (v === null || v === undefined ? "undefined" : `${(v * 100).toFixed(d)}%`);
export const fmtNum = (v, d = 3) => (v === null || v === undefined ? "undefined" : v.toFixed(d));
