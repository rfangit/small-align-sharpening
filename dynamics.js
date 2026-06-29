// ============================================================================
// DYNAMICS — pure numerical core (no DOM, no rendering, no animation).
// ============================================================================
// DESIGN CONTRACT:
//   * Plain functions over plain arrays/objects. No DOM, no canvas, no globals.
//   * Models the general architecture from the writeup: a width-d first layer
//     w1 in R^d, followed by n scalar layers b_1..b_n. The effective function
//     is the row map p = B w1 in R^d, with B = prod_i b_i.
//   * The interface layer (app.js) calls these; nothing here calls outward.
//
// Public surface (the four backend functions the project is organized around):
//   evolveGradientDescent(params, data, opts)  -> parameter snapshots over time
//   paramsToFunction(snapshots, arch)          -> function-space trajectory p(t)
//   conservedQuantities(params)                -> [C_1 .. C_n]
//   targetOutputs(pStar, X)                    -> y for each input row
//
// Plus geometry helpers used by the widgets:
//   weightsFromFunction(p, C, signs)  invert a function value + imbalances to weights
//   functionFromWeights(params)       p = B w1
//   preconditioner(params)            M = B^2 (I + S w1 w1^T)
//   sharpeningCircle(pStar)           Thales ball {center,radius} (whitened)
//
// ----------------------------------------------------------------------------
// PARAMETER REPRESENTATION
//   params = { w1: number[d], b: number[n] }
//   - w1 is the first-layer weight vector (length d = input width).
//   - b  is the list of scalar layers (length n = depth beyond the first layer).
//   A 2-input, 1-hidden-scalar net is d = 2, n = 1.
// ----------------------------------------------------------------------------

// ── small linear-algebra helpers (arbitrary length vectors) ─────────────────
export const V = {
  add: (a, b) => a.map((x, i) => x + b[i]),
  sub: (a, b) => a.map((x, i) => x - b[i]),
  scale: (a, s) => a.map((x) => x * s),
  dot: (a, b) => a.reduce((acc, x, i) => acc + x * b[i], 0),
  norm2: (a) => a.reduce((acc, x) => acc + x * x, 0),
  norm: (a) => Math.sqrt(a.reduce((acc, x) => acc + x * x, 0)),
  zeros: (d) => new Array(d).fill(0),
  clone: (a) => a.slice(),
};

/** Product of the scalar layers, B = prod_i b_i. */
export function layerProduct(b) {
  let B = 1;
  for (let i = 0; i < b.length; i++) B *= b[i];
  return B;
}

// ============================================================================
// FUNCTION <-> WEIGHTS
// ============================================================================

/** Effective function p = B w1  (row map applied to inputs as p^T x). */
export function functionFromWeights(params) {
  const B = layerProduct(params.b);
  return params.w1.map((a) => B * a);
}

// ============================================================================
// 3. CONSERVED QUANTITIES
// ============================================================================
// For each scalar layer i, gradient flow conserves C_i = ||w1||^2 - b_i^2.
// Returns the list [C_1 .. C_n].
// ----------------------------------------------------------------------------
export function conservedQuantities(params) {
  const w2 = V.norm2(params.w1);
  return params.b.map((bi) => w2 - bi * bi);
}

// ============================================================================
// 4. TARGET OUTPUTS
// ============================================================================
// Given a target function pStar (length d) and a dataset X (array of input
// rows, each length d), return the clean outputs y_k = pStar . x_k.
// (Optional gaussian label noise can be added by the caller.)
// ----------------------------------------------------------------------------
export function targetOutputs(pStar, X) {
  return X.map((x) => V.dot(pStar, x));
}

// ============================================================================
// 1. WEIGHT-SPACE GRADIENT DESCENT  (general width d, depth n)
// ============================================================================
// Discrete gradient descent (explicit Euler) on the squared loss
//     L = 1/(2N) sum_k ( f(x_k) - y_k )^2,   f(x) = B (w1 . x).
// over a dataset { X: rows, y: targets }. Returns a list of parameter
// snapshots, one every `recordEvery` steps (plus the first and last).
//
//   params : { w1:[d], b:[n] }     initial parameters
//   data   : { X:[[d]...], y:[...] }
//   opts   : { eta, maxSteps, lossTol, recordEvery }
//            recordEvery = stride between saved snapshots (default 1 = every step).
//
// Snapshot: { step, w1:[d], b:[n], loss }.
// ----------------------------------------------------------------------------
export function evolveGradientDescent(params, data, opts = {}) {
  const eta = opts.eta ?? 1e-3;
  const maxSteps = opts.maxSteps ?? 5000;
  const lossTol = opts.lossTol ?? 1e-4;
  const recordEvery = Math.max(1, opts.recordEvery ?? 1);

  const X = data.X, y = data.y, N = X.length;
  const d = params.w1.length, n = params.b.length;

  let w1 = V.clone(params.w1);
  let b = V.clone(params.b);

  // gradients of L wrt w1 (length d) and each b_i (length n).
  // f(x) = B (w1 . x).  residual r_k = f(x_k) - y_k.
  // dL/dw1 = (1/N) sum_k r_k * B * x_k
  // dL/db_i = (1/N) sum_k r_k * (B / b_i) * (w1 . x_k)
  function gradsAndLoss(w1, b) {
    const B = layerProduct(b);
    const gw1 = V.zeros(d);
    const gb = V.zeros(n);
    let loss = 0;
    // precompute B/b_i (guard tiny b_i)
    const BoverBi = b.map((bi) => B / bi);
    for (let k = 0; k < N; k++) {
      const xk = X[k];
      const wx = V.dot(w1, xk);          // w1 . x_k
      const f = B * wx;
      const r = f - y[k];
      loss += r * r;
      const rB = r * B;
      for (let j = 0; j < d; j++) gw1[j] += rB * xk[j];
      const rwx = r * wx;
      for (let i = 0; i < n; i++) gb[i] += rwx * BoverBi[i];
    }
    const invN = 1 / N;
    for (let j = 0; j < d; j++) gw1[j] *= invN;
    for (let i = 0; i < n; i++) gb[i] *= invN;
    loss *= 0.5 * invN;
    return { gw1, gb, loss };
  }

  const snaps = [];
  let { loss } = gradsAndLoss(w1, b);
  snaps.push({ step: 0, w1: V.clone(w1), b: V.clone(b), loss });

  for (let step = 1; step <= maxSteps; step++) {
    const g = gradsAndLoss(w1, b);
    for (let j = 0; j < d; j++) w1[j] -= eta * g.gw1[j];
    for (let i = 0; i < n; i++) b[i] -= eta * g.gb[i];
    loss = gradsAndLoss(w1, b).loss;     // loss at the new point (cheap enough)

    const record = (step % recordEvery === 0) || step === maxSteps;
    if (record) snaps.push({ step, w1: V.clone(w1), b: V.clone(b), loss });

    if (loss < lossTol) {
      if (snaps[snaps.length - 1].step !== step)
        snaps.push({ step, w1: V.clone(w1), b: V.clone(b), loss });
      break;
    }
    if (!isFinite(loss) || loss > 1e12) break;   // diverged
  }
  return snaps;
}

// ============================================================================
// 2. PARAMETERS -> FUNCTION-SPACE TRAJECTORY
// ============================================================================
// Map a list of weight snapshots to the function-space trajectory p(t) = B w1.
// `arch` is accepted for forward-compatibility / validation; the mapping only
// needs the parameters themselves.
//
//   snapshots : [{ step, w1, b, loss }]
//   arch      : { width, depth }   (optional; validated if present)
// Returns [{ step, p:[d], loss }].
// ----------------------------------------------------------------------------
export function paramsToFunction(snapshots, arch = null) {
  return snapshots.map((s) => {
    if (arch) {
      if (arch.width != null && s.w1.length !== arch.width)
        throw new Error(`width mismatch: arch.width=${arch.width}, w1.length=${s.w1.length}`);
      if (arch.depth != null && s.b.length !== arch.depth)
        throw new Error(`depth mismatch: arch.depth=${arch.depth}, b.length=${s.b.length}`);
    }
    return { step: s.step, p: functionFromWeights(s), loss: s.loss };
  });
}

// ============================================================================
// PRECONDITIONER & SHARPNESS  (general architecture)
// ============================================================================
// M = G G^T = B^2 ( I_d + S w1 w1^T ),  S = sum_i 1/b_i^2.
// Eigenvalues: lambda_par = B^2 (1 + S ||w1||^2) along w1; lambda_perp = B^2
// for the other d-1 directions. (Whitened; for GN sharpness on covariance Sxx
// the relevant operator is M Sxx, handled by the caller in 2D.)
// ----------------------------------------------------------------------------
export function preconditionerEig(params) {
  const B = layerProduct(params.b);
  const B2 = B * B;
  const S = params.b.reduce((acc, bi) => acc + 1 / (bi * bi), 0);
  const w2 = V.norm2(params.w1);
  return {
    lambdaPar: B2 * (1 + S * w2),     // along w1 / p
    lambdaPerp: B2,                   // the other directions
    B2, S,
  };
}

/** Whitened GN sharpness (top eigenvalue of M, = lambda_par here). */
export function gnSharpnessWhitened(params) {
  return preconditionerEig(params).lambdaPar;
}

// ============================================================================
// SHARPNESS / LOSS OVER A TRAJECTORY  (pluggable)
// ============================================================================
// sharpnessOverTrajectory maps weight snapshots -> [{ step, sharpness }] using
// an injectable per-step calculator. The DEFAULT is the analytic whitened top
// eigenvalue of M (gnSharpnessWhitened). To support non-whitened data (or any
// other definition) later, pass a different `calc(params) -> number` — nothing
// else in the pipeline changes. This is the single seam the future
// M*Sigma_xx (power-iteration) sharpness will plug into.
//
//   snapshots : [{ step, w1, b, loss }]
//   calc      : (params:{w1,b}) => number   (default: analytic whitened)
// ----------------------------------------------------------------------------
export function sharpnessOverTrajectory(snapshots, calc = gnSharpnessWhitened) {
  return snapshots.map((s) => ({ step: s.step, sharpness: calc({ w1: s.w1, b: s.b }) }));
}

/** Pull loss(t) out of the snapshots (loss is recorded during GD). */
export function lossOverTrajectory(snapshots) {
  return snapshots.map((s) => ({ step: s.step, loss: s.loss }));
}

// ============================================================================
// INVERSION: function value + imbalances  ->  weights
// ============================================================================
// Given a desired function p (length d) and the conserved quantities C_i, plus
// a sign choice for each scalar layer, recover compatible weights { w1, b }.
//
// Construction (whitened, gradient-flow-consistent):
//   Let u = ||w1||^2. The scalar layers satisfy b_i^2 = u - C_i, so
//     ||p||^2 = ||B w1||^2 = B^2 u = u * prod_i (u - C_i).
//   Solve  g(u) = u * prod_i (u - C_i) - ||p||^2 = 0  for u >= max_i C_i (so all
//   b_i^2 >= 0), by bisection (g is monotone increasing on that half-line).
//   Then b_i = sign_i * sqrt(u - C_i),  B = prod b_i,  w1 = p / B.
//
//   signs : optional array of +/-1 per scalar layer (default all +1).
//
// For p = 0 the direction of w1 is undefined; we return w1 = 0 with b_i from u
// = max(0, max_i C_i) (this is the aligned/degenerate limit).
// ----------------------------------------------------------------------------
export function weightsFromFunction(p, C, signs = null) {
  const n = C.length;
  const s = signs || new Array(n).fill(1);
  const pn2 = V.norm2(p);
  const Cmax = Math.max(0, ...C);

  // g(u) = u * prod_i (u - C_i) - ||p||^2, find root u >= Cmax.
  const g = (u) => {
    let prod = u;
    for (let i = 0; i < n; i++) prod *= (u - C[i]);
    return prod - pn2;
  };

  let u;
  if (pn2 < 1e-30) {
    u = Cmax;                          // degenerate: p = 0
  } else {
    // bracket: at u = Cmax, g = -||p||^2 < 0 (or 0); grow hi until g > 0.
    let lo = Cmax, hi = Math.max(Cmax + 1, 1);
    while (g(hi) < 0) hi *= 2;
    for (let it = 0; it < 200; it++) {
      const mid = 0.5 * (lo + hi);
      if (g(mid) > 0) hi = mid; else lo = mid;
    }
    u = 0.5 * (lo + hi);
  }

  const b = C.map((Ci, i) => s[i] * Math.sqrt(Math.max(u - Ci, 0)));
  const B = layerProduct(b);
  let w1;
  if (pn2 < 1e-30 || Math.abs(B) < 1e-300) {
    w1 = p.map(() => 0);
  } else {
    w1 = p.map((pj) => pj / B);
  }
  return { w1, b };
}

// ============================================================================
// WHITENED-CASE GEOMETRY
// ============================================================================

/** Thales ball of sharpening: {center, radius} for p.(p - pStar)=0 (whitened). */
export function sharpeningCircle(pStar) {
  return { center: V.scale(pStar, 0.5), radius: 0.5 * V.norm(pStar) };
}
