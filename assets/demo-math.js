// Math behind the illustrative demos on the Models page. Pure functions, tested in
// tests/demo-math.test.mjs and tests/test_demos.py. All demo data is made up; none of it
// comes from the activity's datasets.

// ---- kNN: 60 made-up patients with two features on very different scales
export const KNN_DEMO = {
  features: [
    { name: "a", label: "Feature A", unit: "units", min: 0, max: 200 },
    { name: "b", label: "Feature B", unit: "units", min: 0, max: 2 },
  ],
  kValues: [1, 3, 5, 7, 9, 15, 25],
};

/** A small deterministic random number generator (LCG), so the demo looks the same for everyone. */
export function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

/** The made-up points: positives tend to have higher A and higher B. */
export function knnPoints(n = 60, seed = 11) {
  const r = lcg(seed);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = 10 + 180 * r();
    const b = 0.1 + 1.8 * r();
    const score = a / 200 + b / 2 + 0.45 * (r() - 0.5);
    pts.push({ a: Math.round(a), b: Math.round(b * 100) / 100, y: score > 1 ? 1 : 0 });
  }
  return pts;
}

/** Mean and standard deviation of each feature in the made-up training points. */
export function scaler(points) {
  const out = {};
  for (const f of ["a", "b"]) {
    const v = points.map((p) => p[f]);
    const m = v.reduce((s, x) => s + x, 0) / v.length;
    const sd = Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length);
    out[f] = { mean: m, sd };
  }
  return out;
}

/** Euclidean distance, either in raw units or after scaling each feature. */
export function distance(p, q, sc = null) {
  const da = sc ? (p.a - q.a) / sc.a.sd : p.a - q.a;
  const db = sc ? (p.b - q.b) / sc.b.sd : p.b - q.b;
  return Math.sqrt(da * da + db * db);
}

/** The k nearest training points to a probe (ties broken by index) and the share that are positive. */
export function knnPredict(points, probe, k, scaled) {
  const sc = scaled ? scaler(points) : null;
  const order = points.map((p, i) => ({ i, d: distance(p, probe, sc) })).sort((x, y) => x.d - y.d || x.i - y.i);
  const nbrs = order.slice(0, k).map((o) => o.i);
  const pos = nbrs.filter((i) => points[i].y === 1).length;
  return { neighbours: nbrs, positives: pos, probability: pos / k };
}

/**
 * kNN's prediction across a grid covering the plot, for the shaded background. Each cell's
 * neighbours are sorted once (same tie-break as knnPredict), and the running count of positives is
 * kept, so the prediction for any k is a lookup: counts[cell * (n + 1) + k] / k. Cells run row by
 * row from the bottom-left; cell centres are at the middle of each step.
 */
export function knnSurface(points, scaled, nx, ny) {
  const [fa, fb] = KNN_DEMO.features;
  const sc = scaled ? scaler(points) : null;
  const n = points.length;
  const counts = new Uint8Array(nx * ny * (n + 1));
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const probe = { a: fa.min + ((i + 0.5) / nx) * (fa.max - fa.min), b: fb.min + ((j + 0.5) / ny) * (fb.max - fb.min) };
      const order = points.map((p, idx) => ({ idx, d: distance(p, probe, sc) })).sort((x, y) => x.d - y.d || x.idx - y.idx);
      const base = (j * nx + i) * (n + 1);
      for (let r = 0; r < n; r++) counts[base + r + 1] = counts[base + r] + points[order[r].idx].y;
    }
  }
  return { nx, ny, n, counts, probability: (cell, k) => counts[cell * (n + 1) + k] / k };
}

// ---- Logistic regression: 12 made-up patients on two features, each from 0 to 10
export const LOGREG_DEMO = {
  points: [
    { f1: 7, f2: 6, y: 1 }, { f1: 8, f2: 4, y: 1 }, { f1: 6, f2: 8, y: 1 },
    { f1: 9, f2: 7, y: 1 }, { f1: 8.5, f2: 2, y: 1 }, { f1: 3.5, f2: 4, y: 1 },
    { f1: 1.5, f2: 3, y: 0 }, { f1: 3, f2: 4, y: 0 }, { f1: 1, f2: 5, y: 0 },
    { f1: 4.5, f2: 2.5, y: 0 }, { f1: 1.5, f2: 7, y: 0 }, { f1: 6.5, f2: 5, y: 0 },
  ],
  slope: { min: -3, max: 3, step: 0.1, start: -1 },
  intercept: { min: -10, max: 20, step: 0.5, start: 10 },
};

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/**
 * The demo's score: how far a patient sits above the boundary Feature 2 = slope × Feature 1 + intercept.
 * It is a weighted sum plus an intercept (weights −slope and 1), so it is a logistic-regression score.
 */
export const logregScore = (p, slope, intercept) => p.f2 - (slope * p.f1 + intercept);
