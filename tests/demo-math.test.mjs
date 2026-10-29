// Checks the Models page demos' math (node --test tests/). All demo data is made up.
import { test } from "node:test";
import assert from "node:assert/strict";
import { KNN_DEMO, SVM_DEMO, distance, knnPoints, knnPredict, scaler, svmDistance, svmLineX, svmLineZ, svmScore, svmSide } from "../site/assets/demo-math.js";

test("kNN points are deterministic and contain both classes", () => {
  const a = knnPoints(), b = knnPoints();
  assert.deepEqual(a, b);
  const pos = a.filter((p) => p.y === 1).length;
  assert.ok(pos > 10 && pos < 50, `positives ${pos}`);
  for (const p of a) assert.ok(p.a >= 0 && p.a <= 200 && p.b >= 0 && p.b <= 2);
});

test("kNN neighbours match a brute-force search, and the probability is their positive share", () => {
  const pts = knnPoints();
  for (const scaled of [false, true]) {
    for (const k of KNN_DEMO.kValues) {
      const probe = { a: 120, b: 0.7 };
      const r = knnPredict(pts, probe, k, scaled);
      const sc = scaled ? scaler(pts) : null;
      const d = pts.map((p) => distance(p, probe, sc));
      const kth = [...d].sort((x, y) => x - y)[k - 1];
      assert.equal(r.neighbours.length, k);
      for (const i of r.neighbours) assert.ok(d[i] <= kth + 1e-12);
      assert.equal(r.probability, r.neighbours.filter((i) => pts[i].y === 1).length / k);
    }
  }
});

test("without scaling, Feature A decides almost alone", () => {
  const pts = knnPoints();
  const probe = { a: 100, b: 1.0 };
  const raw = knnPredict(pts, probe, 7, false).neighbours.map((i) => pts[i]);
  const scaled = knnPredict(pts, probe, 7, true).neighbours.map((i) => pts[i]);
  const spread = (ns, f) => Math.max(...ns.map((p) => p[f])) - Math.min(...ns.map((p) => p[f]));
  // Unscaled neighbours sit in a narrow band of A but spread across B.
  assert.ok(spread(raw, "a") < spread(scaled, "a"));
  assert.ok(spread(raw, "b") > spread(scaled, "b"));
});

test("SVM boundary separates the classes with margin 1 at the support vectors", () => {
  const m = SVM_DEMO;
  // w and b are stored to 6 decimals, so the support vectors sit within 1e-3 of the margin.
  m.points.forEach((p, i) => {
    const marginValue = p.y * svmScore(m, p.x, p.z);
    if (m.supportVectors.includes(i)) assert.ok(Math.abs(marginValue - 1) < 1e-3, `support vector ${i}: ${marginValue}`);
    else assert.ok(marginValue > 1, `point ${i} inside the margin: ${marginValue}`);
  });
  assert.ok(m.supportVectors.some((i) => m.points[i].y > 0) && m.supportVectors.some((i) => m.points[i].y < 0));
});

test("SVM helpers: the boundary line has score 0, distance is |score| ÷ ‖w‖", () => {
  const m = SVM_DEMO;
  for (const x of [0, 2.5, 6]) assert.ok(Math.abs(svmScore(m, x, svmLineZ(m, x))) < 1e-9);
  assert.ok(Math.abs(svmScore(m, 1, svmLineZ(m, 1, 1)) - 1) < 1e-9);
  assert.equal(svmDistance(m, 0, 0), Math.abs(m.b) / Math.hypot(...m.w));
  for (const z of [0, 3, 6]) assert.ok(Math.abs(svmLineZ(m, svmLineX(m, z, -1), -1) - z) < 1e-9);
});

test("SVM sides: each shaded region has the right sign and together they cover the plot", () => {
  const m = SVM_DEMO;
  const area = (poly) => Math.abs(poly.reduce((s, [x, z], i) => { const [x2, z2] = poly[(i + 1) % poly.length]; return s + x * z2 - x2 * z; }, 0)) / 2;
  const pos = svmSide(m, 1), neg = svmSide(m, -1);
  for (const [x, z] of pos) assert.ok(svmScore(m, x, z) >= -1e-9);
  for (const [x, z] of neg) assert.ok(svmScore(m, x, z) <= 1e-9);
  const [lo, hi] = m.range;
  assert.ok(Math.abs(area(pos) + area(neg) - (hi - lo) ** 2) < 1e-9);
});
