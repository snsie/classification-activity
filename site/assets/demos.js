// Small illustrative demos for the Models page. All numbers are made up (see demo-math.js),
// never the activity's data. The probe moves with a pointer (drag or tap) or the arrow keys.
import { KNN_DEMO, knnPoints, knnPredict } from "./demo-math.js";

const W = 420, H = 320, PAD = 40;

// The new patient is a white diamond everywhere on the page, so it can't be mistaken for a ring.
const diamond = (x, y, r = 10) => `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`;

// Legend keys use the same SVG classes as the charts, so each symbol always matches its label.
const KEYS = {
  pos: `<circle class="pos" cx="11" cy="11" r="5"/>`,
  neg: `<circle class="neg-ring" cx="11" cy="11" r="5"/>`,
  probe: `<path class="probe" d="${diamond(11, 11, 8)}"/>`,
  ring: `<line class="nbr" x1="-6" y1="11" x2="2" y2="11"/><circle class="ring" cx="11" cy="11" r="8"/>`,
};
const legend = (items) => `<ul class="legend">${items
  .map(([key, label]) => `<li><svg class="key" viewBox="0 0 22 22" aria-hidden="true">${KEYS[key]}</svg>${label}</li>`).join("")}</ul>`;

// Pointer dragging is attached once per chart; the probe is redrawn, so its keys are bound per render.
function draggable(svg, toData, onMove) {
  let dragging = false;
  const move = (e) => {
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    onMove(toData(p.x, p.y));
  };
  svg.addEventListener("pointerdown", (e) => {
    e.preventDefault(); // no text selection, so a second drag can't pick up the selected labels
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    move(e);
  });
  svg.addEventListener("pointermove", (e) => dragging && move(e));
  svg.addEventListener("pointerup", () => (dragging = false));
  svg.addEventListener("pointercancel", () => (dragging = false));
  svg.addEventListener("dragstart", (e) => e.preventDefault());
  svg.style.touchAction = "none";
}

function arrowKeys(probe, onMove) {
  probe.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 5 : 1;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key];
    if (!d) return;
    e.preventDefault();
    onMove(null, d);
  });
}

// ---- kNN: neighbours, k, and why scaling matters ------------------------------------
function knnDemo(root) {
  const pts = knnPoints();
  const [fa, fb] = KNN_DEMO.features;
  const probe = { a: 100, b: 1.0 };
  const st = { k: 7, scaled: true };
  const sx = (a) => PAD + ((a - fa.min) / (fa.max - fa.min)) * (W - 2 * PAD);
  const sy = (b) => H - PAD - ((b - fb.min) / (fb.max - fb.min)) * (H - 2 * PAD);
  root.innerHTML = `
    <p class="demo-intro"><span class="tag illustrative">Illustrative example: made-up numbers</span></p>
    <p><strong>Demo: adjustable k; activity: fixed k = 25.</strong></p>
    <p>Sixty made-up training patients. Feature A runs from 0 to 200, Feature B from 0 to 2. Drag the new patient (the white diamond), or focus it and use the arrow keys.</p>
    <div class="demo-controls">
      <label>k (number of neighbours): <output id="knn-k-out">${st.k}</output>
        <input type="range" id="knn-k" min="0" max="${KNN_DEMO.kValues.length - 1}" step="1" value="${KNN_DEMO.kValues.indexOf(st.k)}"></label>
      <label><span><input type="checkbox" id="knn-scale" checked> Scale both features first</span></label>
    </div>
    ${legend([["pos", "training patient: had the outcome"], ["neg", "training patient: didn't"], ["probe", "new patient"], ["ring", "one of the k nearest (outlined, dashed line)"]])}
    <svg class="chart" viewBox="0 0 ${W} ${H}" id="knn-svg" role="group" aria-label="Scatter plot of 60 made-up patients by Feature A and Feature B, with a movable new patient and lines to its k nearest neighbours" aria-describedby="knn-read"></svg>
    <div class="demo-readout" aria-live="polite" id="knn-read"></div>`;
  const svg = root.querySelector("#knn-svg");
  const axes = `<line class="axis" x1="${PAD}" x2="${W - PAD}" y1="${H - PAD}" y2="${H - PAD}"/><line class="axis" x1="${PAD}" x2="${PAD}" y1="${PAD - 10}" y2="${H - PAD}"/>
    ${[0, 50, 100, 150, 200].map((t) => `<text x="${sx(t)}" y="${H - PAD + 16}" text-anchor="middle">${t}</text>`).join("")}
    ${[0, 0.5, 1, 1.5, 2].map((t) => `<text x="${PAD - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`).join("")}
    <text x="${W / 2}" y="${H - 4}" text-anchor="middle">Feature A (original units, 0 to 200)</text>
    <text transform="translate(15 ${H / 2}) rotate(-90)" text-anchor="middle">Feature B (original units, 0 to 2)</text>`;
  const render = () => {
    const r = knnPredict(pts, probe, st.k, st.scaled);
    const nb = new Set(r.neighbours);
    const px = sx(probe.a), py = sy(probe.b);
    svg.innerHTML = `${axes}
      ${r.neighbours.map((i) => `<line class="nbr" x1="${px}" y1="${py}" x2="${sx(pts[i].a)}" y2="${sy(pts[i].b)}"/>`).join("")}
      ${pts.map((p, i) => `<circle class="${p.y ? "pos" : "neg-ring"}" cx="${sx(p.a)}" cy="${sy(p.b)}" r="5"/>${nb.has(i) ? `<circle class="ring" cx="${sx(p.a)}" cy="${sy(p.b)}" r="9"/>` : ""}`).join("")}
      <path class="probe" id="knn-probe" tabindex="0" role="slider" aria-label="New patient: Feature A ${Math.round(probe.a)}, Feature B ${probe.b.toFixed(2)}" aria-valuetext="A ${Math.round(probe.a)}, B ${probe.b.toFixed(2)}" d="${diamond(px, py)}"/>`;
    root.querySelector("#knn-k-out").textContent = st.k;
    root.querySelector("#knn-read").innerHTML = `
      <p><strong>New patient:</strong> A = ${Math.round(probe.a)}, B = ${probe.b.toFixed(2)}.
        ${r.positives} of its ${st.k} nearest neighbours had the outcome, so kNN's probability is ${r.positives} ÷ ${st.k} = <strong>${r.probability.toFixed(2)}</strong>.</p>
      <p>${st.scaled
        ? "Distances are worked out on scaled values (each feature divided by its standard deviation), so neither feature's units dominate: a step of one SD in A counts as much as one SD in B. The axes still show the original units."
        : "Unscaled, a difference of 1 in A counts as much as a difference of 1 in B, which is half of B's whole range. A decides almost alone, so the neighbours form a narrow vertical band."}
      ${st.k === 1 ? " With k = 1 the answer is always 0 or 1: it follows a single patient." : st.k >= 15 ? " A large k smooths: the answer changes slowly as you move." : ""}</p>`;
    arrowKeys(svg.querySelector("#knn-probe"), move);
  };
  const move = (p, d) => {
    if (p) Object.assign(probe, p);
    else { probe.a += d[0] * 4; probe.b += d[1] * 0.04; }
    probe.a = Math.min(fa.max, Math.max(fa.min, probe.a));
    probe.b = Math.min(fb.max, Math.max(fb.min, probe.b));
    const hadFocus = document.activeElement?.id === "knn-probe";
    render();
    if (hadFocus) svg.querySelector("#knn-probe").focus();
  };
  draggable(svg, (x, y) => ({ a: fa.min + ((x - PAD) / (W - 2 * PAD)) * (fa.max - fa.min), b: fb.min + ((H - PAD - y) / (H - 2 * PAD)) * (fb.max - fb.min) }), move);
  root.querySelector("#knn-k").addEventListener("input", (e) => { st.k = KNN_DEMO.kValues[Number(e.target.value)]; render(); });
  root.querySelector("#knn-scale").addEventListener("change", (e) => { st.scaled = e.target.checked; render(); });
  render();
}

const DEMOS = { "demo-knn": knnDemo };
for (const [id, fn] of Object.entries(DEMOS)) {
  const el = document.getElementById(id);
  if (el) fn(el);
}
