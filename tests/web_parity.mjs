// Print web/m11.js numbers for the solution's full results as JSON so tests/test_apps.py can compare them with m11.py.
//     node tests/web_parity.mjs <solution folder>
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const folder = resolve(process.argv[2]);
const m11 = await import(pathToFileURL(join(folder, "web", "m11.js")));
const out = {};
for (const k of m11.DATASETS) {
  const res = JSON.parse(readFileSync(join(folder, "results", `${k}.json`), "utf8"));
  out[k] = { summary: m11.aurocSummary(res), models: {} };
  for (const m of Object.keys(res.models)) {
    const [p, y] = m11.preds(res, m);
    out[k].models[m] = {
      metrics: [0.12, 0.3, 0.5].map((t) => m11.metrics(m11.confusion(p, y, t))),
      calibration: m11.calibration(p, y),
      roc_points: m11.roc(p, y).fpr.length,
    };
  }
}
console.log(JSON.stringify(out));
