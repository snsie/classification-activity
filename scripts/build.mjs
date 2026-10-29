// Builds the static site into dist/. No dependencies: node scripts/build.mjs
//
// Pages are HTML fragments in site/pages/ wrapped in site/layout.html. Tokens like
// {{help auroc}}, {{dl data/heart.csv}} or {{num heart.n_rows}} are expanded here, and the build
// fails on any unknown glossary term, missing download or missing number, so broken links and
// invented numbers can't ship. Every count and share comes from starter/data/manifest.json.
// All links are relative, so the site works under the GitHub Pages project subpath.
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { join, relative } from "node:path";
import { makeZip } from "./zip.mjs";

const ROOT = new URL("..", import.meta.url).pathname;
const SITE = join(ROOT, "site");
const STARTER = join(ROOT, "starter");
const OUT = join(ROOT, "dist");
const REPO = "https://github.com/snsie/classification-activity";
const ZIP_NAME = "judge-the-model-starter.zip";
const ZIP_DIR = "judge-the-model";
const VERSION = (process.env.GITHUB_SHA || (() => {
  try { return execSync("git rev-parse HEAD", { cwd: ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch { return "local"; }
})()).slice(0, 7);

const NAV = [
  { file: "index", label: "Overview" },
  { group: "The work" },
  { file: "models", label: "Models" },
  { file: "analysis", label: "Analysis" },
  { group: "App sections" },
  { file: "beyond-accuracy", label: "1. Beyond Accuracy" },
  { file: "threshold", label: "2. A Threshold for the Job" },
  { file: "ranking", label: "3. Ranking vs. Being Right" },
  { file: "fair-comparison", label: "4. A Fair Comparison" },
  { group: "Resources" },
  { file: "data", label: "Data & files" },
  { file: "hosting", label: "Publish & back up" },
  { file: "submit", label: "Submit" },
  { file: "glossary", label: "Glossary" },
];
const PAGES = ["index", "models", "analysis", "beyond-accuracy", "threshold", "ranking", "fair-comparison",
  "data", "hosting", "submit", "glossary", "404"];

const terms = JSON.parse(readFileSync(join(SITE, "terms.json"), "utf8"));
const dictionary = JSON.parse(readFileSync(join(STARTER, "data", "datasets.json"), "utf8"));
const manifest = JSON.parse(readFileSync(join(STARTER, "data", "manifest.json"), "utf8"));

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const code = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");
const int = (v) => Number(v).toLocaleString("en-US");

function size(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function starterFiles(dir = STARTER) {
  return readdirSync(dir, { withFileTypes: true })
    .flatMap((e) => (e.isDirectory() ? starterFiles(join(dir, e.name)) : [join(dir, e.name)]))
    .filter((p) => !p.includes("__pycache__") && !p.includes(".venv") && !p.endsWith(".DS_Store"))
    .sort();
}

// ---- tokens -------------------------------------------------------------------
function help(id, label) {
  const t = terms[id];
  if (!t) throw new Error(`unknown glossary term: ${id}`);
  const name = label || t.term;
  return `<a class="help" href="glossary.html#${id}" data-help="${id}" aria-label="What is ${esc(name)}?" title="What is ${esc(name)}?"><span aria-hidden="true">?</span></a>`;
}

function download(path, text) {
  const file = join(STARTER, path);
  if (!existsSync(file)) throw new Error(`missing download: starter/${path}`);
  const name = text || path.split("/").pop();
  return `<a class="file" href="starter/${path}" download>${esc(name)}</a> <span class="size">${size(statSync(file).size)}</span>`;
}

function repoLink(path, text) {
  const full = join(ROOT, path);
  if (!existsSync(full)) throw new Error(`missing repository path: ${path}`);
  const kind = statSync(full).isDirectory() ? "tree" : "blob";
  return `<a href="${REPO}/${kind}/main/${path}">${esc(text || path)}</a>`;
}

/** A number from manifest.json: {{num heart.n_rows}}, {{pct diabetes.prevalence}}. */
function fact(path, kind) {
  const [key, ...rest] = path.split(".");
  let v = manifest.datasets[key];
  for (const p of rest) v = v?.[p];
  if (typeof v !== "number") throw new Error(`no number in manifest.json for ${path}`);
  return kind === "pct" ? `${(v * 100).toFixed(1)}%` : int(v);
}

function datasetCard(key) {
  const d = dictionary[key];
  const m = manifest.datasets[key];
  const extra = (c) => {
    const parts = [];
    if (c.categories) parts.push(c.categories.length <= 8 ? c.categories.map((v) => `<code>${esc(v)}</code>`).join(", ") : `${c.categories.length} categories (see data/README.md)`);
    if (c.missing && c.missing !== "never blank") parts.push(esc(c.missing));
    return parts.join("; ");
  };
  const rows = d.columns.filter((c) => c.kind !== "id")
    .map((c) => `<tr><td><code>${esc(c.name)}</code></td><td>${code(c.meaning)}</td><td>${esc(c.units)}</td><td>${extra(c)}</td><td>${esc(c.known)}</td></tr>`).join("");
  const excluded = d.excluded.length ? `<div class="table-wrap"><table class="dictionary">
    <caption>Excluded: leaked columns, not in <code>${key}.csv</code> ${help("leaked-column")}</caption>
    <thead><tr><th scope="col">Column</th><th scope="col">Meaning</th><th scope="col">When is it known?</th></tr></thead>
    <tbody>${d.excluded.map((c) => `<tr><td><code>${esc(c.name)}</code></td><td>${code(c.meaning)}</td><td>${esc(c.known)}</td></tr>`).join("")}</tbody></table></div>` : "";
  return `<dl class="facts">
      <dt>Job</dt><dd>${esc(d.job)}</dd>
      <dt>Positive class</dt><dd>${esc(d.positive)} (<code>target</code> = 1) ${help("positive-class")}</dd>
      <dt>Prediction point</dt><dd>${esc(d.prediction_point)}</dd>
      <dt>Cohort</dt><dd>${esc(d.cohort)}</dd>
      <dt>Framing</dt><dd>${esc(d.framing)}</dd>
    </dl>
    <p class="data-facts"><span class="tag">From the supplied data</span> ${int(m.n_rows)} ${d.units}, ${int(m.n_pos)} positive (${(m.prevalence * 100).toFixed(1)}%).
      Development: ${int(m.n_dev)} (${int(m.n_dev_pos)} positive) in 5 folds. Test: ${int(m.n_test)} (${int(m.n_test_pos)} positive).</p>
    <div class="table-wrap"><table class="dictionary">
    <caption>Predictors in <code>${key}.csv</code>, with “When is it known?” ${help("when-known")}</caption>
    <thead><tr><th scope="col">Column</th><th scope="col">Meaning</th><th scope="col">Units</th><th scope="col">Categories / missing</th><th scope="col">When is it known?</th></tr></thead>
    <tbody>${rows}
      <tr class="target"><td><code>target</code></td><td><strong>Target:</strong> ${esc(d.target.meaning)}</td><td>0/1</td><td></td><td>${esc(d.target.known)}</td></tr>
    </tbody></table></div>
    ${excluded}
    <p class="source">Source: <a href="${esc(d.url)}">${esc(d.source)}</a> (${esc(d.licence)}). ${esc(d.citation)}</p>`;
}

function glossary() {
  const groups = {};
  for (const [id, t] of Object.entries(terms)) (groups[t.group] ||= []).push([id, t]);
  return Object.entries(groups)
    .map(([g, items]) => `<section class="glossary-group"><h2>${esc(g)}</h2><dl>${items
      .map(([id, t]) => `<div class="term" id="${id}"><dt>${esc(t.term)}</dt><dd><p>${esc(t.short)}</p>${t.more ? `<p class="more">${esc(t.more)}</p>` : ""}</dd></div>`)
      .join("")}</dl></section>`)
    .join("");
}

const DATASET_NAMES = { readmission: "30-day readmission", diabetes: "Diabetes screening", heart: "Heart disease" };
const FILE_NOTES = {
  "README.md": "Your README: your app link goes on the first line.",
  "DECISIONS.md": "Your decisions, saved before the results they depend on.",
  "AGENT_BRIEF.md": "Rules for your AI coding agent. Have it read this first.",
  "SETUP.md": "Run, publish and back up (both routes).",
  "check.py": "Integrity checks: must pass before you submit.",
  "m11.py": "Shared metric helpers (Python).",
  "app/m11.js": "Shared metric helpers (browser).",
  "app/app.js": "Route A: the four plain starter sections.",
  "app/index.html": "Route A: the app page.",
  "index.html": "Route A: opens app/ from your site's root link.",
  "package.json": "Route A: npm run serve previews the app.",
  "streamlit_app.py": "Route B: the four plain starter sections.",
  "requirements.txt": "Pinned Python packages (CPU only).",
  ".gitignore": "Keeps caches and secrets out of git.",
  "analysis/__init__.py": "Marks analysis/ as a package.",
  "analysis/__main__.py": "The staged workflow: train, explore, test, export.",
  "analysis/data.py": "Loads the CSVs, folds and predictors.",
  "analysis/models.py": "The six models. You complete the kNN and SVM TODOs here.",
  "analysis/evaluate.py": "Cross-validation and the one-time test fit.",
  "analysis/export.py": "Reads DECISIONS.md and checks git history.",
  "analysis/with_probabilities.py": "Supplied helper that turns an SVM's raw score into a probability.",
  "data/README.md": "Data dictionary, with “When is it known?” for every column.",
  "data/datasets.json": "The same dictionary, as data.",
  "data/manifest.json": "Sources, licences, versions, counts and SHA-256 fingerprints.",
  "results/decisions.json": "Your exported DECISIONS.md (starter: empty).",
  "prompts/README.md": "Optional: save your AI prompts here.",
};

function fileList() {
  const rows = starterFiles().map((f) => {
    const rel = relative(STARTER, f);
    let note = FILE_NOTES[rel];
    if (!note && /^data\/\w+\.csv$/.test(rel)) note = `${DATASET_NAMES[rel.slice(5, -4)]} dataset.`;
    if (!note && /^results\/\w+_dev\.json$/.test(rel)) note = `Development results, ${DATASET_NAMES[rel.slice(8, -9)]} (starter: baseline only).`;
    if (!note) throw new Error(`no description for starter/${rel}: add it to FILE_NOTES`);
    return `<tr><td>${download(rel, rel)}</td><td>${esc(note)}</td></tr>`;
  });
  return `<div class="table-wrap"><table class="files"><thead><tr><th scope="col">File</th><th scope="col">What it is</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function expand(html, zipBytes) {
  return html
    .replace(/\{\{help ([a-z0-9-]+)(?:\|([^}]+))?\}\}/g, (_, id, label) => help(id, label))
    .replace(/\{\{dl ([^|}]+)(?:\|([^}]+))?\}\}/g, (_, p, t) => download(p.trim(), t))
    .replace(/\{\{repo ([^|}]+)(?:\|([^}]+))?\}\}/g, (_, p, t) => repoLink(p.trim(), t))
    .replace(/\{\{(num|pct) ([\w.]+)\}\}/g, (_, kind, p) => fact(p, kind))
    .replace(/\{\{dataset (\w+)\}\}/g, (_, k) => datasetCard(k))
    .replace(/\{\{todo ([^}]+)\}\}/g, (_, t) => `<mark class="todo">TODO(instructor): ${esc(t)}</mark>`)
    .replace(/\{\{glossary\}\}/g, () => glossary())
    .replace(/\{\{files\}\}/g, () => fileList())
    .replace(/\{\{zip\}\}/g, () => `<a class="button primary" href="downloads/${ZIP_NAME}" download>Download the starter (.zip, ${size(zipBytes)})</a>`)
    .replace(/\{\{zipurl\}\}/g, `downloads/${ZIP_NAME}`)
    .replace(/<pre>/g, '<pre tabindex="0">'); // code blocks can scroll sideways, so keyboard users must reach them
}

function nav(current) {
  return NAV.map((n) =>
    n.group
      ? `<li class="nav-group" aria-hidden="true">${n.group}</li>`
      : `<li><a href="${n.file === "index" ? "./" : `${n.file}.html`}"${n.file === current ? ' aria-current="page"' : ""}>${esc(n.label)}</a></li>`,
  ).join("");
}

// ---- build ----------------------------------------------------------------------
rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "downloads"), { recursive: true });
cpSync(join(SITE, "assets"), join(OUT, "assets"), { recursive: true });
cpSync(STARTER, join(OUT, "starter"), { recursive: true, filter: (p) => !p.includes("__pycache__") && !p.includes(".venv") });
writeFileSync(join(OUT, ".nojekyll"), "");
writeFileSync(join(OUT, "assets", "terms.js"), `// Generated from site/terms.json\nexport const TERMS = ${JSON.stringify(terms)};\n`);

const zip = makeZip(starterFiles().map((f) => ({ name: `${ZIP_DIR}/${relative(STARTER, f)}`, data: readFileSync(f) })));
writeFileSync(join(OUT, "downloads", ZIP_NAME), zip);

const layout = readFileSync(join(SITE, "layout.html"), "utf8");
for (const page of PAGES) {
  const src = readFileSync(join(SITE, "pages", `${page}.html`), "utf8");
  const meta = JSON.parse(src.match(/^<!--meta (.*?) -->/)[1]);
  const body = expand(src.replace(/^<!--meta .*? -->\n?/, ""), zip.length);
  const html = layout
    .replaceAll("{{title}}", esc(meta.title))
    .replaceAll("{{description}}", esc(meta.description))
    .replaceAll("{{page}}", page)
    .replace("{{base}}", meta.base ? `<base href="${meta.base}">` : "")
    .replace("{{nav}}", nav(page))
    .replace("{{version}}", VERSION === "local" ? "local build" : `<a href="${REPO}/commit/${VERSION}">version ${VERSION}</a>`)
    .replace("{{content}}", body)
    .replace("{{scripts}}", (meta.scripts || []).map((s) => `<script type="module" src="assets/${s}"></script>`).join(""));
  if (/\{\{[a-z]/.test(html)) throw new Error(`unexpanded token in ${page}: ${html.match(/\{\{[^}]*\}\}/)[0]}`);
  writeFileSync(join(OUT, `${page}.html`), html);
}
console.log(`built ${PAGES.length} pages, ${starterFiles().length} starter files, ${ZIP_NAME} (${size(zip.length)}) → dist/`);
