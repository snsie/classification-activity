// Starter app for GitHub Pages: four plain sections that read results/ and show correct numbers.
// Build each section up to its required minimum (see the activity website). Keep every number
// coming from m11.js, and label every table and plot "Development (OOF)" or "Test".
import * as m11 from "./m11.js";

const SECTIONS = {
  "beyond-accuracy": { title: "1. Beyond Accuracy", render: renderBeyondAccuracy },
  threshold: { title: "2. A Threshold for the Job", render: renderThreshold },
  ranking: { title: "3. Ranking vs. Being Right", render: renderRanking },
  "fair-comparison": { title: "4. A Fair Comparison", render: renderFairComparison },
};
const state = { dataset: "readmission", model: "baseline" };
const $ = (id) => document.getElementById(id);
const pct = (v) => m11.fmtPct(v, 1);
const num = (v) => m11.fmtNum(v, 3);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const DEV = '<span class="partition">Development (OOF)</span>';
const TEST = '<span class="partition">Test</span>';

function section() {
  const s = location.hash.slice(1);
  return SECTIONS[s] ? s : "beyond-accuracy";
}

async function setup() {
  $("dataset").innerHTML = m11.DATASETS.map((k) => `<option value="${k}">${m11.DATASET_LABEL[k]}</option>`).join("");
  $("dataset").onchange = (e) => { state.dataset = e.target.value; render(); };
  $("model").onchange = (e) => { state.model = e.target.value; render(); };
  window.onhashchange = render;
  await renderRecommendations();
  render();
}

async function renderRecommendations() {
  const d = await m11.load("decisions");
  const recs = d ? d.recommendations : {};
  const filled = m11.DATASETS.filter((k) => recs[k]);
  $("recs").innerHTML = filled.length
    ? `<ul>${m11.DATASETS.map((k) => `<li><strong>${m11.DATASET_LABEL[k]}:</strong> ${recs[k] ? esc(recs[k]) : '<span class="missing">not written yet</span>'}</li>`).join("")}</ul>`
    : '<p class="note">No recommendations yet: fill in <code>Recommendations</code> in DECISIONS.md, run <code>python -m analysis export</code>, then commit and push.</p>';
}

async function render() {
  const key = section();
  document.querySelectorAll("nav a").forEach((a) => (a.hash === `#${key}` ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  const dev = await m11.load(`${state.dataset}_dev`);
  $("model").innerHTML = m11.MODELS.map((m) =>
    `<option value="${m}"${m === state.model ? " selected" : ""}>${m11.MODEL_LABEL[m]}${dev.models[m] ? "" : " (not trained yet)"}</option>`).join("");
  const test = await m11.load(`${state.dataset}_test`);
  const body = dev.models[state.model]
    ? await SECTIONS[key].render(dev, test)
    : `<p class="missing">${m11.MODEL_LABEL[state.model]} has no results yet: run <code>python -m analysis train</code>, then commit and push <code>results/</code>.</p>`;
  $("section").innerHTML = `<h2>${SECTIONS[key].title}</h2>
    <p class="note">${m11.DATASET_LABEL[state.dataset]}: ${dev.n.toLocaleString()} development rows, ${dev.n_pos.toLocaleString()} positive.</p>${body}`;
}

const starterNote = (text) => `<p class="starter">Starter display: ${text} Build this section up to its required minimum.</p>`;

// ---- 1. Beyond Accuracy ------------------------------------------------------
function renderBeyondAccuracy(dev) {
  const [p, y] = m11.oof(dev, state.model);
  const c = m11.confusion(p, y, 0.5);
  const m = m11.metrics(c);
  return `${starterNote("the four counts at threshold 0.50, and accuracy.")}
    <table><caption>${DEV} ${m11.MODEL_LABEL[state.model]}, threshold 0.50</caption>
      <tr><th>TP</th><th>FN</th><th>FP</th><th>TN</th><th class="num">Accuracy</th></tr>
      <tr><td>${c.tp}</td><td>${c.fn}</td><td>${c.fp}</td><td>${c.tn}</td><td class="num">${pct(m.accuracy)}</td></tr>
    </table>`;
}

// ---- 2. A Threshold for the Job ------------------------------------------------
function renderThreshold(dev) {
  const [p, y] = m11.oof(dev, state.model);
  const m = m11.metrics(m11.confusion(p, y, 0.5));
  return `${starterNote("metrics at 0.50 only.")}
    <table><caption>${DEV} ${m11.MODEL_LABEL[state.model]}, threshold 0.50</caption>
      <tr><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th class="num">Flagged per 100</th></tr>
      <tr><td>${pct(m.sensitivity)}</td><td>${pct(m.specificity)}</td><td>${pct(m.precision)}</td><td class="num">${m.flagged_per_100.toFixed(1)}</td></tr>
    </table>`;
}

// ---- 3. Ranking vs. Being Right ------------------------------------------------
function renderRanking(dev) {
  const rows = m11.MODELS.filter((m) => dev.models[m]).map((m) => {
    const [p, y] = m11.oof(dev, m);
    return `<tr><td>${m11.MODEL_LABEL[m]}</td><td class="num">${num(m11.auroc(p, y))}</td></tr>`;
  });
  return `${starterNote("AUROC of the pooled out-of-fold probabilities, in a table.")}
    <table><caption>${DEV} AUROC (ranking, not accuracy)</caption>
      <tr><th>Model</th><th class="num">AUROC</th></tr>${rows.join("")}</table>`;
}

// ---- 4. A Fair Comparison ------------------------------------------------------
function renderFairComparison(dev, test) {
  const s = m11.foldSummary(dev);
  const rows = Object.entries(s.models).map(([m, v]) =>
    `<tr><td>${m11.MODEL_LABEL[m]}</td><td class="num">${num(v.mean_cv)} ± ${num(v.sd_cv)}</td></tr>`);
  return `${starterNote("mean ± SD of the 5 cross-validated AUROCs.")}
    <table><caption>${DEV} CV AUROC across 5 folds</caption>
      <tr><th>Model</th><th class="num">CV AUROC (mean ± SD)</th></tr>${rows.join("")}</table>
    <p class="note">${TEST} ${test ? "Test results are in results/" + state.dataset + "_test.json." : "Locked until you run <code>python -m analysis test</code> (once)."}</p>`;
}

setup();
