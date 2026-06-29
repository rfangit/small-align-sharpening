// ============================================================================
// INTERFACE LAYER — the only file that touches all three worlds.
// ============================================================================
// DESIGN CONTRACT:
//   * The ONLY layer that reads the DOM AND calls the backend (dynamics.js) AND
//     drives the visualization. All cross-layer coupling lives here on purpose.
//   * Backend stays DOM-blind; visualization stays backend-blind.
//   * Orchestration only: read inputs -> call backend -> hand results to viz.
//
// Two widgets are mounted here:
//   1. initPreview(prefix)  — a small, static taste of the curved GD path.
//   2. initMainWidget(prefix) — the full interactive panel: drag p0/p*, choose
//      an imbalance C, see three trajectories (C = 0, +C, -C) plus the circle of
//      norm growth, with loss and sharpness over epochs in a side panel.
// ----------------------------------------------------------------------------

import {
  evolveGradientDescent, paramsToFunction, weightsFromFunction,
  lossOverTrajectory, sharpnessOverTrajectory, sharpeningCircle,
  targetOutputs, layerProduct, V,
} from './dynamics.js';
import { FunctionPlane, EpochPlots, ProbabilityPlot, SharpeningBall, LinePlot } from './visualization.js';
import { PS_TABLE, psProbability, PROD_TABLE, PROD_K } from './distributions.js';
import { TRAJ_DATA } from './trajectories.js';

// architecture for these widgets: two inputs (width d = 2); depth is whatever
// the caller's imbalance list implies (depth n = Carr.length).
const WIDTH = 2;
const GD = { eta: 5e-3, maxSteps: 1500, lossTol: 1e-3, recordEvery: 1 };

// When the C-curves finish at different steps (each hits the loss tolerance at
// its own time), pad the shorter series by holding their last value out to the
// longest series' final step, so the epoch plots show no abruptly-ending lines.
// Set to false to draw each line only up to where its run actually stopped.
const PAD_EPOCHS = true;

const PSTAR_COLOR = '#c8612f';
const P0_COLOR = '#3a5bbf';

// ── whitened dataset: inputs with (1/N) X^T X = I_d; optimum is exactly p*. ──
function whitenedDataset(pStar) {
  const d = pStar.length, X = [];
  for (let j = 0; j < d; j++) {
    const e = V.zeros(d); e[j] = Math.sqrt(d); X.push(e.slice());
    const e2 = V.zeros(d); e2[j] = -Math.sqrt(d); X.push(e2);
  }
  return { X, y: targetOutputs(pStar, X) };
}

// Run GD from initial function p0 toward target p* for a network whose scalar
// layers have conserved quantities Carr (one per layer, so depth = Carr.length).
// A single-layer net is just Carr = [C]. Returns
//   { traj:[[x,y]...], loss:[{step,loss}], sharp:[{step,sharpness}] }.
function runArch(p0, data, Carr) {
  const init = weightsFromFunction(p0, Carr, Carr.map(() => 1));
  const snaps = evolveGradientDescent(init, data, GD);
  const fn = paramsToFunction(snaps, { width: WIDTH, depth: Carr.length });
  return {
    traj: fn.map((q) => [q.p[0], q.p[1]]),
    loss: lossOverTrajectory(snaps),
    sharp: sharpnessOverTrajectory(snaps),
  };
}

function pointById(pts, id) { const p = pts.find((q) => q.id === id); return [p.x, p.y]; }

// Pad a set of [step,value] series so they all extend to the global max step:
// each shorter series gets one extra point at (maxStep, lastValue), flattening
// its tail instead of stopping early. Returns new series (inputs untouched).
// `enabled=false` returns the series unchanged.
function padSeries(series, enabled = true) {
  if (!enabled || !series.length) return series;
  let maxStep = -Infinity;
  for (const s of series) {
    if (s.pts.length) maxStep = Math.max(maxStep, s.pts[s.pts.length - 1][0]);
  }
  return series.map((s) => {
    if (!s.pts.length) return s;
    const last = s.pts[s.pts.length - 1];
    if (last[0] >= maxStep) return s;                 // already the longest
    return { ...s, pts: [...s.pts, [maxStep, last[1]]] };
  });
}

// ════════════════════════════════════════════════════════════════════════════
// 1. PREVIEW WIDGET — small, static. One C = 0 curve, no controls, no circle.
// ════════════════════════════════════════════════════════════════════════════
export function initPreview(prefix) {
  const plane = new FunctionPlane(`${prefix}-canvas`, {
    range: 2.5,
    points: [
      { id: 'pstar', x: 1.4, y: 0.6, color: PSTAR_COLOR, label: 'p*' },
      { id: 'p0',    x: 0.15, y: 0.55, color: P0_COLOR, label: 'p₀' },
    ],
    onDrag: (pts) => redraw(pts),
    onDragEnd: (pts) => redraw(pts),
  });
  function redraw(pts) {
    const pstar = pointById(pts, 'pstar');
    const p0 = pointById(pts, 'p0');
    const data = whitenedDataset(pstar);
    const r = runArch(p0, data, [0]);
    plane.setTrajectories([
      // hypothetical straight-line optimizer: a dashed segment p0 -> p*
      { pts: [p0, pstar], color: '#9aa3b8', width: 1.5, dashed: true },
      // actual (curved) overparameterized GD path
      { pts: r.traj, color: P0_COLOR, width: 2 },
    ]);
  }
  redraw(plane.getPoints());
  return { plane };
}

// ════════════════════════════════════════════════════════════════════════════
// 2. MAIN WIDGET — three C-curves, circle of norm growth, loss/sharpness panel.
// ════════════════════════════════════════════════════════════════════════════
export function initMainWidget(prefix) {
  const el = (id) => document.getElementById(`${prefix}-${id}`);

  // color family for the three imbalances. C=0 is the blue baseline; the two
  // anti-balanced curves get distinct hues, reused across all three panels.
  const COLORS = { zero: '#3a5bbf', plus: '#c8612f', minus: '#2a9d5c' };

  const plane = new FunctionPlane(`${prefix}-canvas`, {
    range: 2.5,
    points: [
      { id: 'pstar', x: 1.4, y: 0.6, color: PSTAR_COLOR, label: 'p*' },
      { id: 'p0',    x: 0.15, y: 0.55, color: P0_COLOR, label: 'p₀' },
    ],
    onDrag: (pts) => recompute(pts),
    onDragEnd: (pts) => recompute(pts),
  });
  const epochs = new EpochPlots(`${prefix}-epochs`, {
    topLabel: 'loss', bottomLabel: 'sharpness', topLog: false,
  });

  function currentC() {
    const v = parseFloat(el('cInput').value);
    return isFinite(v) ? Math.abs(v) : 0;   // we use +-|C|
  }

  function recompute(pts) {
    const pstar = pointById(pts, 'pstar');
    const p0 = pointById(pts, 'p0');
    const data = whitenedDataset(pstar);
    const Cmag = currentC();

    // three imbalances: 0, +C, -C  (each its own color, shared across panels)
    const specs = [
      { C: 0,     color: COLORS.zero },
      { C: +Cmag, color: COLORS.plus },
      { C: -Cmag, color: COLORS.minus },
    ];

    const lines = [], lossSeries = [], sharpSeries = [];
    for (const spec of specs) {
      const r = runArch(p0, data, [spec.C]);
      lines.push({ pts: r.traj, color: spec.color, width: 2 });
      lossSeries.push({ color: spec.color, pts: r.loss.map((d) => [d.step, d.loss]) });
      sharpSeries.push({ color: spec.color, pts: r.sharp.map((d) => [d.step, d.sharpness]) });
    }

    // circle of norm growth (Thales ball, whitened)
    const circ = sharpeningCircle(pstar);
    plane.setCircle({ center: circ.center, radius: circ.radius, color: '#c8612f' });
    plane.setTrajectories(lines);
    epochs.setSeries(padSeries(lossSeries, PAD_EPOCHS), padSeries(sharpSeries, PAD_EPOCHS));
  }

  el('cInput').addEventListener('input', () => recompute(plane.getPoints()));
  recompute(plane.getPoints());
  return { plane, epochs };
}

// ════════════════════════════════════════════════════════════════════════════
// 3. DEPTH WIDGET — three curves for increasing depth.
//    The user gives C1, C2, C3. The three networks use the conserved-quantity
//    lists [C1], [C1,C2], [C1,C2,C3] — i.e. 1, 2, and 3 hidden scalar layers.
// ════════════════════════════════════════════════════════════════════════════
export function initDepthWidget(prefix) {
  const el = (id) => document.getElementById(`${prefix}-${id}`);

  // one color per depth, reused across trajectory / loss / sharpness panels.
  const COLORS = { d1: '#3a5bbf', d2: '#c8612f', d3: '#2a9d5c' };

  const plane = new FunctionPlane(`${prefix}-canvas`, {
    range: 2.5,
    points: [
      { id: 'pstar', x: 1.4, y: 0.6, color: PSTAR_COLOR, label: 'p*' },
      { id: 'p0',    x: 0.15, y: 0.55, color: P0_COLOR, label: 'p₀' },
    ],
    onDrag: (pts) => recompute(pts),
    onDragEnd: (pts) => recompute(pts),
  });
  const epochs = new EpochPlots(`${prefix}-epochs`, {
    topLabel: 'loss', bottomLabel: 'sharpness', topLog: false,
  });

  const readC = (id) => { const v = parseFloat(el(id).value); return isFinite(v) ? v : 0; };

  function recompute(pts) {
    const pstar = pointById(pts, 'pstar');
    const p0 = pointById(pts, 'p0');
    const data = whitenedDataset(pstar);
    const C1 = readC('c1'), C2 = readC('c2'), C3 = readC('c3');

    // increasing depth: [C1] -> [C1,C2] -> [C1,C2,C3]
    const specs = [
      { Carr: [C1],          color: COLORS.d1 },
      { Carr: [C1, C2],      color: COLORS.d2 },
      { Carr: [C1, C2, C3],  color: COLORS.d3 },
    ];

    const lines = [], lossSeries = [], sharpSeries = [];
    for (const spec of specs) {
      const r = runArch(p0, data, spec.Carr);
      lines.push({ pts: r.traj, color: spec.color, width: 2 });
      lossSeries.push({ color: spec.color, pts: r.loss.map((d) => [d.step, d.loss]) });
      sharpSeries.push({ color: spec.color, pts: r.sharp.map((d) => [d.step, d.sharpness]) });
    }

    const circ = sharpeningCircle(pstar);
    plane.setCircle({ center: circ.center, radius: circ.radius, color: '#c8612f' });
    plane.setTrajectories(lines);
    epochs.setSeries(padSeries(lossSeries, PAD_EPOCHS), padSeries(sharpSeries, PAD_EPOCHS));
  }

  for (const id of ['c1', 'c2', 'c3']) {
    el(id).addEventListener('input', () => recompute(plane.getPoints()));
  }
  recompute(plane.getPoints());
  return { plane, epochs };
}

// ════════════════════════════════════════════════════════════════════════════
// 4. SHARPENING-PROBABILITY WIDGET — angular distribution + draggable ball.
//    Left: P(in ball) vs norm ratio for the chosen dimension, with a vertical
//    line at the current norm. Right: the Thales ball with a draggable norm
//    point; the in-ball arc shrinks as the norm grows, and the ball's central
//    opacity rises with dimension (illustrative of hidden volume).
// ════════════════════════════════════════════════════════════════════════════
export function initSharpeningWidget(prefix) {
  const el = (id) => document.getElementById(`${prefix}-${id}`);

  const prob = new ProbabilityPlot(`${prefix}-prob`);
  const ball = new SharpeningBall(`${prefix}-ball`, {
    onDrag: (ratio) => updateMark(ratio),
  });

  function currentDim() { return parseInt(el('dim').value, 10) || 2; }

  function setCurve() {
    const d = currentDim();
    prob.setCurve(PS_TABLE[d]);
    ball.setDim(d);
    el('dimLabel').textContent = d;
    updateMark(ball.getNormRatio());
  }

  function updateMark(ratio) {
    const d = currentDim();
    const P = psProbability(d, ratio);
    prob.setMark(ratio, P);
    el('readRatio').textContent = ratio.toFixed(3);
    el('readProb').textContent = P.toFixed(3);
  }

  el('dim').addEventListener('input', setCurve);
  setCurve();
  return { prob, ball };
}

// ════════════════════════════════════════════════════════════════════════════
// 5. PRODUCT-CONCENTRATION WIDGET — how a product of k unit-variance factors
//    concentrates toward zero. Two static line plots side by side (uniform |
//    normal); each shows the single distribution (k=1) and the k-product, with
//    ±1σ marker bars. The user picks k ∈ {2,3,4}.
// ════════════════════════════════════════════════════════════════════════════
export function initProductWidget(prefix) {
  const el = (id) => document.getElementById(`${prefix}-${id}`);

  // unit-variance factors, so ±1σ is at ±1 for both.
  const SIGMA = 1;
  const SINGLE_COLOR = '#9aa3b8';   // the k=1 baseline
  const PROD_COLOR = '#3a5bbf';     // the k-product
  const BAR_COLOR = '#c8612f';

  // y-range = the clip ceiling for each kind (3x single peak) with a little headroom
  const yMaxUniform = 3.0 * (1 / (2 * Math.sqrt(3))) * 1.08;
  const yMaxNormal = 3.0 * (1 / Math.sqrt(2 * Math.PI)) * 1.08;

  const uni = new LinePlot(`${prefix}-uniform`, {
    xRange: [-4, 4], yRange: [0, yMaxUniform],
    xLabel: 'value', yLabel: 'density', xTicks: [-4, -2, 0, 2, 4],
  });
  const nrm = new LinePlot(`${prefix}-normal`, {
    xRange: [-4, 4], yRange: [0, yMaxNormal],
    xLabel: 'value', yLabel: 'density', xTicks: [-4, -2, 0, 2, 4],
  });

  function currentK() { return parseInt(el('k').value, 10) || 2; }

  function redraw() {
    const k = currentK();
    el('kLabel').textContent = k;

    uni.setCurves([
      { pts: PROD_TABLE.uniform[1], color: SINGLE_COLOR, width: 1.5, dashed: true },
      { pts: PROD_TABLE.uniform[k], color: PROD_COLOR, width: 2.5 },
    ]);
    uni.setBars([{ x: -SIGMA, color: BAR_COLOR }, { x: SIGMA, color: BAR_COLOR }]);

    nrm.setCurves([
      { pts: PROD_TABLE.normal[1], color: SINGLE_COLOR, width: 1.5, dashed: true },
      { pts: PROD_TABLE.normal[k], color: PROD_COLOR, width: 2.5 },
    ]);
    nrm.setBars([{ x: -SIGMA, color: BAR_COLOR }, { x: SIGMA, color: BAR_COLOR }]);
  }

  el('k').addEventListener('input', redraw);
  redraw();
  return { uni, nrm };
}

// ════════════════════════════════════════════════════════════════════════════
// 6. RANDOM-MODELS WIDGET — sample standard-init networks and show trajectories.
//    User picks init (uniform/gaussian), width d, depth n. The trajectories are
//    PRECOMPUTED offline (gen_trajectories.py) and projected into each model's
//    own p0–p* plane; the widget loads and draws them, colored by whether they
//    eventually sharpen. This makes the depth/width competition visible, instantly.
// ════════════════════════════════════════════════════════════════════════════

export function initRandomModelsWidget(prefix) {
  const el = (id) => document.getElementById(`${prefix}-${id}`);
  const SHARP_COLOR = '#c8612f';   // eventually sharpens (norm grows)
  const FLAT_COLOR = '#3a5bbf';    // flattens

  const plane = new FunctionPlane(`${prefix}-canvas`, {
    range: 2.0, points: [],          // no draggable points; static target marker
  });

  function currentKind() { return el('kind').value; }
  function currentD() { return parseInt(el('width').value, 10) || 2; }
  function currentN() { return parseInt(el('depth').value, 10) || 1; }

  // Decode one delta-encoded, integer-quantized path back to [[x,y],...].
  // Format: flat = [x0, y0, dx1, dy1, dx2, dy2, ...] in units of 1/SCALE.
  const SCALE = TRAJ_DATA.meta.scale || 100;
  function decodePath(flat) {
    let x = flat[0], y = flat[1];
    const pts = [[x / SCALE, y / SCALE]];
    for (let i = 2; i < flat.length; i += 2) {
      x += flat[i]; y += flat[i + 1];
      pts.push([x / SCALE, y / SCALE]);
    }
    return pts;
  }

  function recompute() {
    const kind = currentKind(), d = currentD(), n = currentN();
    el('widthLabel').textContent = d;
    el('depthLabel').textContent = n;

    // Trajectories are PRECOMPUTED offline (gen_trajectories.ipynb) in a compact
    // delta-encoded form; here we look up the cell, decode each path, and draw.
    // No live gradient-flow integration, so switching cells is instant.
    const cell = (TRAJ_DATA.data[kind] || {})[d]?.[n];
    const paths = cell ? cell.p : [];
    const inits = cell ? cell.i : [];
    const evers = cell ? cell.e : [];

    const lines = [];
    let nInit = 0, nEver = 0;
    for (let k = 0; k < paths.length; k++) {
      if (inits[k]) nInit++;
      if (evers[k]) nEver++;
      lines.push({ pts: decodePath(paths[k]), color: evers[k] ? SHARP_COLOR : FLAT_COLOR, width: 1.5, alpha: 0.35 });
    }

    // ball of sharpening: p* lies at (1, 0) in every model's plane (e1 = p*
    // direction), so the projected ball is the Thales circle on [0, p*].
    plane.setCircle({ center: [0.5, 0], radius: 0.5, color: '#c8612f' });
    plane.setTrajectories(lines);
    plane.setMarkers([{ p: [1, 0], color: '#c8612f' }]);

    const N = paths.length;
    el('initCount').textContent = `${nInit}/${N}`;
    el('everCount').textContent = `${nEver}/${N}`;
  }

  for (const id of ['kind', 'width', 'depth']) {
    el(id).addEventListener('input', recompute);
  }
  recompute();
  return { plane };
}

// ── boot all widgets when the DOM is ready ──────────────────────────────────
function boot() {
  if (document.getElementById('preview-canvas')) initPreview('preview');
  if (document.getElementById('main-canvas')) initMainWidget('main');
  if (document.getElementById('depth-canvas')) initDepthWidget('depth');
  if (document.getElementById('sharp-prob')) initSharpeningWidget('sharp');
  if (document.getElementById('prod-uniform')) initProductWidget('prod');
  if (document.getElementById('rand-canvas')) initRandomModelsWidget('rand');
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
