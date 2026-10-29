// Checks the Models page demos' math (node --test tests/). All demo data is made up.
import { test } from "node:test";
import assert from "node:assert/strict";
import { KNN_DEMO, distance, knnPoints, knnPredict, scaler } from "../site/assets/demo-math.js";

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
