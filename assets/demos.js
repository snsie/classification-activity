// Small illustrative demos for the Models page. All numbers are made up (see demo-math.js),
// never the activity's data. The probe moves with a pointer (drag or tap) or the arrow keys.
import { KNN_DEMO, LOGREG_DEMO, knnPoints, knnPredict, knnSurface, logregScore, sigmoid } from "./demo-math.js?v=9617d8c";

const W = 420, H = 320, PAD = 40;

// The new patient is a white diamond everywhere on the page, so it can't be mistaken for a ring.
const diamond = (x, y, r = 10) => `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`;

// Legend keys use the same SVG classes as the charts, so each symbol always matches its label.
const KEYS = {
  pos: `<circle class="pos" cx="11" cy="11" r="5"/>`,
  neg: `<circle class="neg-ring" cx="11" cy="11" r="5"/>`,
  probe: `<path class="probe" d="${diamond(11, 11, 8)}"/>`,
  ring: `<line class="nbr" x1="-6" y1="11" x2="2" y2="11"/><circle class="ring" cx="11" cy="11" r="8"/>`,
  wrong: `<circle class="ring" cx="11" cy="11" r="9"/>`,
  boundary: `<line class="threshold" x1="0" x2="22" y1="11" y2="11"/>`,
  regionPos: `<rect class="region-pos" x="2" y="2" width="18" height="18"/>`,
  regionNeg: `<rect class="region-neg" x="2" y="2" width="18" height="18"/>`,
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

const num = (x, digits = 1) => x.toFixed(digits).replace("-", "−");

// ---- Logistic regression: move the decision boundary by hand ------------------------
function logregDemo(root) {
  const pts = LOGREG_DEMO.points;
  const { slope: sl, intercept: ic } = LOGREG_DEMO;
  const st = { slope: sl.start, intercept: ic.start };
  // Same frames as the static figures: score −6 to 6 on the curve, both features 0 to 10 on the plane.
  const cx = (z) => 172.5 + Math.max(-6, Math.min(6, z)) * 18.75, cy = (p) => 160 - 140 * p;
  const px = (f1) => 50 + f1 * 23.5, py = (f2) => 160 - f2 * 14;
  const curve = Array.from({ length: 49 }, (_, i) => -6 + i * 0.25).map((z, i) => `${i ? "L" : "M"}${cx(z).toFixed(1)},${cy(sigmoid(z)).toFixed(1)}`).join(" ");
  const slider = (id, label, r, v) => `<label>${label}: <output id="${id}-out">${num(v)}</output>
    <input type="range" id="${id}" min="${r.min}" max="${r.max}" step="${r.step}" value="${v}"></label>`;
  root.innerHTML = `
    <div class="demo-controls">
      ${slider("lr-slope", "Slope of the boundary", sl, st.slope)}
      ${slider("lr-int", "Intercept (Feature 2 at Feature 1 = 0)", ic, st.intercept)}
    </div>
    <div class="demo-buttons"><button type="button" id="lr-reset">Reset the line</button></div>
    <div class="illus illus-row">
      <svg viewBox="0 0 300 205" id="lr-curve" role="img"></svg>
      <svg viewBox="0 0 300 205" id="lr-plane" role="img"></svg>
    </div>
    ${legend([["boundary", "probability = 0.5: the threshold (left) and the decision boundary it makes (right)"], ["pos", "had the outcome"], ["neg", "didn't"], ["wrong", "on the wrong side of the boundary"]])}
    <div class="demo-readout" aria-live="polite" id="lr-read"></div>`;
  const curveSvg = root.querySelector("#lr-curve"), plane = root.querySelector("#lr-plane");
  const dot = (p, x, y, wrong) => `<circle class="${p.y ? "pos" : "neg-ring"}" cx="${x}" cy="${y}" r="5"/>${wrong ? `<circle class="ring" cx="${x}" cy="${y}" r="9"/>` : ""}`;
  const render = () => {
    const { slope: m, intercept: b } = st;
    const scored = pts.map((p) => ({ p, z: logregScore(p, m, b) }));
    const wrong = (s) => (s.z >= 0 ? 1 : 0) !== s.p.y;
    const nWrong = scored.filter(wrong).length;
    curveSvg.innerHTML = `
      <line class="axis" x1="60" x2="285" y1="160" y2="160"/><line class="axis" x1="60" x2="60" y1="20" y2="160"/>
      <line class="threshold" x1="60" x2="285" y1="90" y2="90"/>
      <path class="ink" d="${curve}"/>
      ${scored.map((s) => dot(s.p, cx(s.z).toFixed(1), cy(sigmoid(s.z)).toFixed(1), wrong(s))).join("")}
      <text x="54" y="24" text-anchor="end">1</text><text x="54" y="94" text-anchor="end">0.5</text><text x="54" y="164" text-anchor="end">0</text>
      <text x="60" y="176" text-anchor="middle">−6</text><text x="116" y="176" text-anchor="middle">−3</text><text x="172.5" y="176" text-anchor="middle">0</text><text x="229" y="176" text-anchor="middle">3</text><text x="285" y="176" text-anchor="middle">6</text>
      <text x="282" y="84" text-anchor="end">threshold 0.5</text>
      <text x="165" y="196" text-anchor="middle">Model score: weighted sum + intercept</text>
      <text transform="translate(16 90) rotate(-90)" text-anchor="middle">Predicted probability</text>
      <text transform="translate(31 90) rotate(-90)" text-anchor="middle">of the outcome</text>`;
    curveSvg.setAttribute("aria-label", `S-shaped curve from model score to predicted probability, with the 12 patients placed on it at their scores; ${nWrong} of them fall on the wrong side of the 0.5 threshold`);
    // Each region is the half-plane on one side of the line, clipped to the plot.
    const half = (far) => `${px(0)},${py(b)} ${px(10)},${py(10 * m + b)} ${px(10)},${py(far)} ${px(0)},${py(far)}`;
    // Region labels go in the corner deepest inside each region, if the region shows at all.
    const corner = (f2, sign) => [0, 10].map((f1) => ({ f1, z: sign * logregScore({ f1, f2 }, m, b) })).sort((u, v) => v.z - u.z)[0];
    const label = (c, y, text) => c.z > 0 ? `<text class="strong" x="${c.f1 ? 279 : 56}" y="${y}" text-anchor="${c.f1 ? "end" : "start"}">${text}</text>` : "";
    plane.innerHTML = `
      <defs><clipPath id="lr-clip"><rect x="50" y="20" width="235" height="140"/></clipPath></defs>
      <g clip-path="url(#lr-clip)">
        <polygon class="region-pos" points="${half(1000)}"/><polygon class="region-neg" points="${half(-1000)}"/>
        <line class="threshold" x1="${px(0)}" x2="${px(10)}" y1="${py(b)}" y2="${py(10 * m + b)}"/>
      </g>
      <line class="axis" x1="50" x2="285" y1="160" y2="160"/><line class="axis" x1="50" x2="50" y1="20" y2="160"/>
      ${scored.map((s) => dot(s.p, px(s.p.f1), py(s.p.f2), wrong(s))).join("")}
      ${label(corner(10, 1), 36, "predicted ≥ 0.5")}${label(corner(0, -1), 153, "predicted &lt; 0.5")}
      <text x="167.5" y="178" text-anchor="middle">Feature 1 (made up)</text>
      <text transform="translate(36 90) rotate(-90)" text-anchor="middle">Feature 2 (made up)</text>`;
    plane.setAttribute("aria-label", `Twelve made-up patients on two features, split by a straight decision boundary with slope ${num(m)} and intercept ${num(b)}; ${nWrong} patients are on the wrong side`);
    root.querySelector("#lr-slope-out").textContent = num(m);
    root.querySelector("#lr-int-out").textContent = num(b);
    root.querySelector("#lr-read").innerHTML = `
      <p><strong>Score</strong> = Feature 2 − (${num(m)} × Feature 1 ${b < 0 ? "−" : "+"} ${num(Math.abs(b))}): how far a patient sits above the line. Above it the score is positive, so the probability is over 0.5.</p>
      <p><strong>${pts.length - nWrong} of ${pts.length}</strong> patients are on the correct side; ${nWrong} ${nWrong === 1 ? "is" : "are"} outlined on the wrong side.
        The slope tilts the line; the intercept slides it up or down. Either way it stays straight.</p>`;
  };
  const bind = (id, key) => root.querySelector(`#${id}`).addEventListener("input", (e) => { st[key] = Number(e.target.value); render(); });
  bind("lr-slope", "slope");
  bind("lr-int", "intercept");
  root.querySelector("#lr-reset").addEventListener("click", () => {
    st.slope = sl.start;
    st.intercept = ic.start;
    root.querySelector("#lr-slope").value = sl.start;
    root.querySelector("#lr-int").value = ic.start;
    render();
  });
  render();
}

// ---- kNN: neighbours, k, and why scaling matters ------------------------------------
function knnDemo(root) {
  const pts = knnPoints();
  const [fa, fb] = KNN_DEMO.features;
  const probe = { a: 100, b: 1.0 };
  const st = { k: 7, scaled: true };
  const sx = (a) => PAD + ((a - fa.min) / (fa.max - fa.min)) * (W - 2 * PAD);
  const sy = (b) => H - PAD - ((b - fb.min) / (fb.max - fb.min)) * (H - 2 * PAD);
  // The shaded background: kNN's prediction on a grid of 4-pixel cells. The neighbours of every cell
  // are sorted once per scaling setting; each k's shading is then a lookup, kept for reuse.
  const NX = (W - 2 * PAD) / 4, NY = (H - 2 * PAD) / 4;
  const surfaces = {}, shading = new Map();
  const shade = () => {
    const key = `${st.scaled}-${st.k}`;
    if (shading.has(key)) return shading.get(key);
    const surf = (surfaces[st.scaled] ??= knnSurface(pts, st.scaled, NX, NY));
    // One path per class, each a set of horizontal runs of same-class cells.
    const d = { pos: "", neg: "" };
    const cls = (cell) => (surf.probability(cell, st.k) >= 0.5 ? "pos" : "neg");
    for (let j = 0; j < NY; j++) {
      let start = 0;
      for (let i = 1; i <= NX; i++) {
        if (i < NX && cls(j * NX + i) === cls(j * NX + start)) continue;
        d[cls(j * NX + start)] += `M${PAD + start * 4},${H - PAD - (j + 1) * 4}h${(i - start) * 4}v4h${-(i - start) * 4}Z`;
        start = i;
      }
    }
    const out = `<path class="region-pos" shape-rendering="crispEdges" d="${d.pos}"/><path class="region-neg" shape-rendering="crispEdges" d="${d.neg}"/>`;
    shading.set(key, out);
    return out;
  };
  root.innerHTML = `
    <p class="demo-intro"><span class="tag illustrative">Illustrative example: made-up numbers</span></p>
    <p><strong>Demo: adjustable k; activity: fixed k = 25.</strong></p>
    <p>Sixty made-up training patients. Feature A runs from 0 to 200, Feature B from 0 to 2. Drag the new patient (the white diamond), or focus it and use the arrow keys.</p>
    <div class="demo-controls">
      <label>k (number of neighbours): <output id="knn-k-out">${st.k}</output>
        <input type="range" id="knn-k" min="0" max="${KNN_DEMO.kValues.length - 1}" step="1" value="${KNN_DEMO.kValues.indexOf(st.k)}"></label>
      <label><span><input type="checkbox" id="knn-scale" checked> Scale both features first</span></label>
    </div>
    ${legend([["pos", "training patient: had the outcome"], ["neg", "training patient: didn't"], ["probe", "new patient"], ["ring", "one of the k nearest (outlined, dashed line)"],
      ["regionPos", "kNN would predict ≥ 0.5 here"], ["regionNeg", "kNN would predict &lt; 0.5 here"]])}
    <svg class="chart" viewBox="0 0 ${W} ${H}" id="knn-svg" role="group" aria-label="Scatter plot of 60 made-up patients by Feature A and Feature B, shaded by kNN's prediction, with a movable new patient and lines to its k nearest neighbours" aria-describedby="knn-read"></svg>
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
    svg.innerHTML = `${shade()}${axes}
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
      ${st.k === 1 ? " With k = 1 the answer is always 0 or 1: it follows a single patient." : st.k >= 15 ? " A large k smooths: the answer changes slowly as you move." : ""}</p>
      <p>The shading is kNN's prediction for a new patient at each spot on the plot. ${st.k <= 3
        ? "With so few neighbours the border is jagged, with islands around single patients."
        : st.k >= 15 ? "With many neighbours the border is smooth, and small pockets of patients are outvoted." : "Raise or lower k to see the border smooth out or break up."}
      ${st.scaled ? "" : " Unscaled, the border runs almost straight up and down: only A matters."}</p>`;
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

const DEMOS = { "demo-logreg": logregDemo, "demo-knn": knnDemo };
for (const [id, fn] of Object.entries(DEMOS)) {
  const el = document.getElementById(id);
  if (el) fn(el);
}
