// Runs every m11.js helper on the shared fixtures and prints the results as JSON.
// Called by tests/test_parity.py, which runs m11.py on the same fixtures and compares.
//   node tests/parity.mjs <fixtures.json> [m11.js path]
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const [fixturesPath, m11Path = "starter/app/m11.js"] = process.argv.slice(2);
const m11 = await import(pathToFileURL(resolve(m11Path)).href);
const { cases, dev_files: devFiles } = JSON.parse(readFileSync(fixturesPath, "utf8"));

// Infinity is not JSON: write it as the string "inf", as test_parity.py does.
const clean = (v) => JSON.parse(JSON.stringify(v, (k, x) => (x === Infinity ? "inf" : x === -Infinity ? "-inf" : x)));

const out = { cases: [], dev: [] };
for (const c of cases) {
  out.cases.push(clean({
    name: c.name,
    confusion: c.thresholds.map((t) => m11.confusion(c.p, c.y, t)),
    metrics: c.thresholds.map((t) => m11.metrics(m11.confusion(c.p, c.y, t))),
    auroc: m11.auroc(c.p, c.y),
    roc: m11.roc(c.p, c.y),
    calibration: m11.calibration(c.p, c.y),
    rule: m11.thresholdForRule(c.p, c.y),
    rule_80: m11.thresholdForRule(c.p, c.y, 0.8),
    majority: m11.majorityAccuracy(c.y),
  }));
}
for (const d of devFiles) out.dev.push(clean(m11.foldSummary(d)));
process.stdout.write(JSON.stringify(out));
