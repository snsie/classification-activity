// Small illustrative demos for the Models page. All numbers are made up (see demo-math.js),
// never the activity's data. Both probes move with a pointer (drag or tap) or the arrow keys.
import { KNN_DEMO, SVM_DEMO, knnPoints, knnPredict, svmDistance, svmLineZ, svmScore } from "./demo-math.js";

const W = 420, H = 320, PAD = 40;

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
  svg.addEventListener("pointerdown", (e) => { dragging = true; svg.setPointerCapture(e.pointerId); move(e); });
  svg.addEventListener("pointermove", (e) => dragging && move(e));
  svg.addEventListener("pointerup", () => (dragging = false));
  svg.addEventListener("pointercancel", () => (dragging = false));
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
    <p>Sixty made-up training patients: filled dots had the outcome, rings didn't. Feature A runs from 0 to 200, Feature B from 0 to 2. Drag the white probe, or focus it and use the arrow keys.</p>
    <div class="demo-controls">
      <label>k (number of neighbours): <output id="knn-k-out">${st.k}</output>
        <input type="range" id="knn-k" min="0" max="${KNN_DEMO.kValues.length - 1}" step="1" value="${KNN_DEMO.kValues.indexOf(st.k)}"></label>
      <label><span><input type="checkbox" id="knn-scale" checked> Scale both features first</span></label>
    </div>
    <ul class="legend"><li><span class="swatch dot" style="background:var(--up)"></span>had the outcome</li><li><span class="swatch ring" style="border-color:var(--down)"></span>didn't</li><li><span class="swatch ring" style="border-color:var(--text)"></span>one of the k nearest</li></ul>
    <svg class="chart" viewBox="0 0 ${W} ${H}" id="knn-svg" role="group" aria-label="Scatter plot of 60 made-up patients by Feature A and Feature B, with a movable new patient and lines to its k nearest neighbours" aria-describedby="knn-read"></svg>
    <div class="demo-readout" aria-live="polite" id="knn-read"></div>`;
  const svg = root.querySelector("#knn-svg");
  const axes = `<line class="axis" x1="${PAD}" x2="${W - PAD}" y1="${H - PAD}" y2="${H - PAD}"/><line class="axis" x1="${PAD}" x2="${PAD}" y1="${PAD - 10}" y2="${H - PAD}"/>
    ${[0, 50, 100, 150, 200].map((t) => `<text x="${sx(t)}" y="${H - PAD + 16}" text-anchor="middle">${t}</text>`).join("")}
    ${[0, 0.5, 1, 1.5, 2].map((t) => `<text x="${PAD - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`).join("")}
    <text x="${W / 2}" y="${H - 4}" text-anchor="middle">Feature A (0 to 200)</text>
    <text transform="translate(12 ${H / 2}) rotate(-90)" text-anchor="middle">Feature B (0 to 2)</text>`;
  const render = () => {
    const r = knnPredict(pts, probe, st.k, st.scaled);
    const nb = new Set(r.neighbours);
    svg.innerHTML = `${axes}
      ${r.neighbours.map((i) => `<line class="nbr" x1="${sx(probe.a)}" y1="${sy(probe.b)}" x2="${sx(pts[i].a)}" y2="${sy(pts[i].b)}"/>`).join("")}
      ${pts.map((p, i) => `<circle class="${p.y ? "pos" : "neg-ring"}" cx="${sx(p.a)}" cy="${sy(p.b)}" r="5"/>${nb.has(i) ? `<circle class="ring" cx="${sx(p.a)}" cy="${sy(p.b)}" r="9"/>` : ""}`).join("")}
      <circle class="probe" id="knn-probe" tabindex="0" role="slider" aria-label="New patient: Feature A ${Math.round(probe.a)}, Feature B ${probe.b.toFixed(2)}" aria-valuetext="A ${Math.round(probe.a)}, B ${probe.b.toFixed(2)}" cx="${sx(probe.a)}" cy="${sy(probe.b)}" r="9"/>`;
    root.querySelector("#knn-k-out").textContent = st.k;
    root.querySelector("#knn-read").innerHTML = `
      <p><strong>New patient:</strong> A = ${Math.round(probe.a)}, B = ${probe.b.toFixed(2)}.
        ${r.positives} of its ${st.k} nearest neighbours had the outcome, so kNN's probability is ${r.positives} ÷ ${st.k} = <strong>${r.probability.toFixed(2)}</strong>.</p>
      <p>${st.scaled
        ? "Both features count: a step of one SD in A matters as much as one SD in B."
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

// ---- SVM: the widest gap, and a raw score that isn't a probability -----------------------
function svmDemo(root) {
  const m = SVM_DEMO;
  const [lo, hi] = m.range;
  const probe = { x: 3.0, z: 3.0 };
  const sx = (x) => PAD + ((x - lo) / (hi - lo)) * (W - 2 * PAD);
  const sy = (z) => H - PAD - ((z - lo) / (hi - lo)) * (H - 2 * PAD);
  const line = (level, cls) => `<line class="${cls}" x1="${sx(lo)}" y1="${sy(svmLineZ(m, lo, level))}" x2="${sx(hi)}" y2="${sy(svmLineZ(m, hi, level))}"/>`;
  root.innerHTML = `
    <p class="demo-intro"><span class="tag illustrative">Illustrative example: made-up numbers</span></p>
    <p>Sixteen made-up training patients, two features. The solid line is the SVM's boundary; the dashed lines mark the gap it keeps as wide as possible. Drag the white probe, or focus it and use the arrow keys.</p>
    <ul class="legend"><li><span class="swatch dot" style="background:var(--up)"></span>had the outcome</li><li><span class="swatch ring" style="border-color:var(--down)"></span>didn't</li><li><span class="swatch ring" style="border-color:var(--text)"></span>support vector: fixes the boundary</li></ul>
    <svg class="chart" viewBox="0 0 ${W} ${H}" id="svm-svg" role="group" aria-label="Scatter plot of 16 made-up patients with the SVM boundary, its margins and a movable new patient" aria-describedby="svm-read"></svg>
    <div class="demo-readout" aria-live="polite" id="svm-read"></div>`;
  const svg = root.querySelector("#svm-svg");
  const sv = new Set(m.supportVectors);
  const render = () => {
    svg.innerHTML = `<defs><clipPath id="svm-clip"><rect x="${PAD}" y="${PAD - 10}" width="${W - 2 * PAD}" height="${H - 2 * PAD + 10}"/></clipPath></defs>
      <line class="axis" x1="${PAD}" x2="${W - PAD}" y1="${H - PAD}" y2="${H - PAD}"/><line class="axis" x1="${PAD}" x2="${PAD}" y1="${PAD - 10}" y2="${H - PAD}"/>
      ${[0, 2, 4, 6].map((t) => `<text x="${sx(t)}" y="${H - PAD + 16}" text-anchor="middle">${t}</text><text x="${PAD - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`).join("")}
      <text x="${W / 2}" y="${H - 4}" text-anchor="middle">Feature 1 (made up)</text>
      <text transform="translate(12 ${H / 2}) rotate(-90)" text-anchor="middle">Feature 2 (made up)</text>
      <g clip-path="url(#svm-clip)">${line(0, "boundary")}${line(1, "margin")}${line(-1, "margin")}</g>
      ${m.points.map((p, i) => `<circle class="${p.y > 0 ? "pos" : "neg-ring"}" cx="${sx(p.x)}" cy="${sy(p.z)}" r="5"/>${sv.has(i) ? `<circle class="sv" cx="${sx(p.x)}" cy="${sy(p.z)}" r="9"/>` : ""}`).join("")}
      <circle class="probe" id="svm-probe" tabindex="0" role="slider" aria-label="New patient: feature 1 ${probe.x.toFixed(1)}, feature 2 ${probe.z.toFixed(1)}" aria-valuetext="${probe.x.toFixed(1)}, ${probe.z.toFixed(1)}" cx="${sx(probe.x)}" cy="${sy(probe.z)}" r="9"/>`;
    const s = svmScore(m, probe.x, probe.z);
    root.querySelector("#svm-read").innerHTML = `
      <p><strong>Raw score: ${s >= 0 ? "+" : "−"}${Math.abs(s).toFixed(2)}</strong>, on the ${s >= 0 ? "outcome" : "no-outcome"} side, ${svmDistance(m, probe.x, probe.z).toFixed(2)} units from the boundary.</p>
      <p>Higher means more likely to have the outcome, but this number is <strong>not a probability</strong>: it can be above 1 or below 0, and a score of +2 isn't twice as likely as +1. In the activity, the supplied <code>with_probabilities()</code> helper turns these scores into probabilities.</p>`;
    arrowKeys(svg.querySelector("#svm-probe"), move);
  };
  const move = (p, d) => {
    if (p) Object.assign(probe, p);
    else { probe.x += d[0] * 0.1; probe.z += d[1] * 0.1; }
    probe.x = Math.min(hi, Math.max(lo, probe.x));
    probe.z = Math.min(hi, Math.max(lo, probe.z));
    const hadFocus = document.activeElement?.id === "svm-probe";
    render();
    if (hadFocus) svg.querySelector("#svm-probe").focus();
  };
  draggable(svg, (x, y) => ({ x: lo + ((x - PAD) / (W - 2 * PAD)) * (hi - lo), z: lo + ((H - PAD - y) / (H - 2 * PAD)) * (hi - lo) }), move);
  render();
}

const DEMOS = { "demo-knn": knnDemo, "demo-svm": svmDemo };
for (const [id, fn] of Object.entries(DEMOS)) {
  const el = document.getElementById(id);
  if (el) fn(el);
}
