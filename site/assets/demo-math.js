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

// ---- SVM: 16 made-up patients, two features, separable. Boundary fitted with scikit-learn
// SVC(kernel="linear", C=1e6) (a hard margin); tests/test_demos.py refits and compares.
export const SVM_DEMO = {
  points: [
    { x: 1, z: 1, y: -1 }, { x: 1.5, z: 2.2, y: -1 }, { x: 2.2, z: 1.1, y: -1 }, { x: 2.4, z: 2.6, y: -1 },
    { x: 1.1, z: 3.2, y: -1 }, { x: 3.1, z: 1.4, y: -1 }, { x: 0.7, z: 2.0, y: -1 }, { x: 2.0, z: 3.6, y: -1 },
    { x: 4.1, z: 3.6, y: 1 }, { x: 4.6, z: 2.2, y: 1 }, { x: 3.7, z: 4.6, y: 1 }, { x: 5.1, z: 3.1, y: 1 },
    { x: 4.3, z: 5.1, y: 1 }, { x: 5.4, z: 4.3, y: 1 }, { x: 3.6, z: 3.3, y: 1 }, { x: 4.8, z: 1.4, y: 1 },
  ],
  w: [1.379646, 0.689567],
  b: -6.242107,
  supportVectors: [5, 7, 14],
  range: [0, 6],
};

/** The SVM's raw score: signed distance-like value w·x + b. Positive side if > 0. Not a probability. */
export const svmScore = (m, x, z) => m.w[0] * x + m.w[1] * z + m.b;

/** Distance from the boundary in feature units: |score| ÷ ‖w‖. */
export const svmDistance = (m, x, z) => Math.abs(svmScore(m, x, z)) / Math.hypot(m.w[0], m.w[1]);

/** z on the line w·(x, z) + b = level, for drawing the boundary (level 0) and the margins (±1). */
export const svmLineZ = (m, x, level = 0) => (level - m.b - m.w[0] * x) / m.w[1];
