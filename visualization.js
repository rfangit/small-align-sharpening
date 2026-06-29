// ============================================================================
// VISUALIZATION LAYER — all drawing, no math, no DOM-reading, no backend calls.
// ============================================================================
// DESIGN CONTRACT:
//   * DRAWS what it is told and REPORTS user gestures. Nothing else.
//   * Never imports from dynamics.js; never reads control inputs.
//   * Owns the world<->pixel mapping and the render of the function-space plane.
//   * Exposes a small imperative API; the interface layer (app.js) drives it.
//
// Renders: faint integer grid + axes, an optional sharpening circle, any number
// of trajectory polylines, optional playback markers, and the draggable points.
// All share one worldToPx/pxToWorld transform so everything stays pixel-aligned.
// ----------------------------------------------------------------------------

export class FunctionPlane {
  /**
   * @param {string} canvasId
   * @param {object} opts
   *   range    : axes span [-range, range] in data units (default 2.5)
   *   points   : [{ id, x, y, color, label }]  draggable points
   *   onDrag   : (points) => void   continuous, while dragging
   *   onDragEnd: (points) => void   once, on release
   */
  constructor(canvasId, opts = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    this.range = opts.range ?? 2.5;
    this.center = opts.center ?? [0, 0];   // world point at the panel's center
    this.points = opts.points ?? [];
    this.onDrag = opts.onDrag || (() => {});
    this.onDragEnd = opts.onDragEnd || (() => {});

    // data the interface layer sets (draw passes below):
    this.trajectories = [];   // [{ pts:[[x,y]...], color, dashed, width }]
    this.markers = [];        // [{ p:[x,y], color }]
    this.circle = null;       // { center:[x,y], radius, color } or null

    this._DOT_R = 8;
    this._GRAB_R = 16;
    this._dragging = -1;
    this._dpr = 1;

    this._bindEvents();
    this.resize();
  }

  // ── public API ──────────────────────────────────────────────────────────────
  setPoints(points) { this.points = points; this.draw(); }
  getPoints() { return this.points.map((p) => ({ id: p.id, x: p.x, y: p.y, label: p.label })); }
  setPointById(id, x, y) {
    const p = this.points.find((q) => q.id === id);
    if (p) { p.x = x; p.y = y; this.draw(); }
  }
  /** trajectories: array of { pts:[[x,y]...], color, dashed?, width? }. */
  setTrajectories(trajectories) { this.trajectories = trajectories || []; this.draw(); }
  setMarkers(markers) { this.markers = markers || []; this.draw(); }
  setCircle(circle) { this.circle = circle; this.draw(); }
  setRange(range) { this.range = range; this.draw(); }
  redraw() { this.draw(); }

  // ── sizing & transforms ─────────────────────────────────────────────────────
  resize() {
    this._dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this._dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this._dpr);
    this.draw();
  }
  _metrics() {
    const w = this.canvas.width, h = this.canvas.height;
    // Pad as a fraction of the smaller dimension so padding looks even.
    const pad = Math.min(w, h) * 0.08;
    const innerW = w - 2 * pad, innerH = h - 2 * pad;
    // Isometric scale: the same pixels-per-world-unit on BOTH axes, so circles
    // stay circles and angles stay true regardless of the panel's aspect ratio.
    // `range` is the guaranteed half-extent on the SHORTER axis; the longer axis
    // shows proportionally more world. Centered on `this.center` (default origin).
    const minInner = Math.min(innerW, innerH);
    const scale = minInner / (2 * this.range);          // px per world unit
    const cx = this.center ? this.center[0] : 0;
    const cy = this.center ? this.center[1] : 0;
    return { w, h, pad, innerW, innerH, scale, cx, cy,
             x0: pad, y0: pad, x1: pad + innerW, y1: pad + innerH };
  }
  worldToPx(p) {
    const m = this._metrics();
    const X = p.x ?? p[0], Y = p.y ?? p[1];
    return {
      x: m.pad + m.innerW / 2 + (X - m.cx) * m.scale,
      y: m.pad + m.innerH / 2 - (Y - m.cy) * m.scale,    // y up
    };
  }
  pxToWorld(px, py) {
    const m = this._metrics();
    return {
      x: m.cx + (px - m.pad - m.innerW / 2) / m.scale,
      y: m.cy - (py - m.pad - m.innerH / 2) / m.scale,
    };
  }
  /** Visible world half-extents (used for grid bounds). */
  _worldExtent() {
    const m = this._metrics();
    return {
      hx: (m.innerW / 2) / m.scale,                      // half-width in world units
      hy: (m.innerH / 2) / m.scale,
      cx: m.cx, cy: m.cy,
    };
  }

  // ── rendering ───────────────────────────────────────────────────────────────
  draw() {
    const ctx = this.ctx, m = this._metrics(), dpr = this._dpr;
    ctx.clearRect(0, 0, m.w, m.h);
    const x0 = m.x0, x1 = m.x1, y0 = m.y0, y1 = m.y1;

    // grid — span the actual visible world extent (wider on the long axis),
    // not just `range`, so a rectangular panel fills with gridlines.
    const ext = this._worldExtent();
    ctx.lineWidth = 1 * dpr; ctx.strokeStyle = '#ece9e1';
    for (let v = Math.ceil(ext.cx - ext.hx); v <= Math.floor(ext.cx + ext.hx); v++) {
      if (v === 0) continue;
      const gx = this.worldToPx({ x: v, y: 0 }).x;
      ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y1); ctx.stroke();
    }
    for (let v = Math.ceil(ext.cy - ext.hy); v <= Math.floor(ext.cy + ext.hy); v++) {
      if (v === 0) continue;
      const gy = this.worldToPx({ x: 0, y: v }).y;
      ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); ctx.stroke();
    }
    // axes
    const o = this.worldToPx({ x: 0, y: 0 });
    ctx.strokeStyle = '#cfcabd'; ctx.lineWidth = 1.25 * dpr;
    ctx.beginPath(); ctx.moveTo(x0, o.y); ctx.lineTo(x1, o.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(o.x, y0); ctx.lineTo(o.x, y1); ctx.stroke();

    // circle of sharpening (under everything)
    if (this.circle) {
      const c = this.worldToPx(this.circle.center);
      const edge = this.worldToPx({ x: this.circle.center[0] + this.circle.radius, y: this.circle.center[1] });
      const rpx = Math.abs(edge.x - c.x);
      ctx.save();
      ctx.beginPath(); ctx.arc(c.x, c.y, rpx, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(200,97,47,0.10)'; ctx.fill();
      ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = this.circle.color || '#c8612f'; ctx.stroke();
      ctx.restore();
    }

    // trajectory polylines
    for (const t of this.trajectories) {
      const pts = t.pts || [];
      if (pts.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const q = this.worldToPx(pts[i]);
        if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
      }
      ctx.lineWidth = (t.width || 2) * dpr;
      ctx.strokeStyle = t.color || '#3a5bbf';
      ctx.setLineDash(t.dashed ? [5 * dpr, 4 * dpr] : []);
      const prevAlpha = ctx.globalAlpha;
      if (t.alpha != null) ctx.globalAlpha = t.alpha;
      ctx.stroke();
      ctx.globalAlpha = prevAlpha;
      ctx.setLineDash([]);
    }

    // playback markers
    for (const mk of this.markers) {
      const q = this.worldToPx(mk.p);
      ctx.beginPath(); ctx.arc(q.x, q.y, 5 * dpr, 0, 2 * Math.PI);
      ctx.fillStyle = mk.color || '#3a5bbf'; ctx.fill();
      ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = '#fff'; ctx.stroke();
    }

    // draggable points (on top)
    for (const p of this.points) {
      const q = this.worldToPx(p);
      ctx.beginPath(); ctx.arc(q.x, q.y, this._DOT_R * dpr, 0, 2 * Math.PI);
      ctx.fillStyle = p.color || '#3a5bbf'; ctx.fill();
      ctx.lineWidth = 2 * dpr; ctx.strokeStyle = '#fff'; ctx.stroke();
      if (p.label) {
        ctx.fillStyle = '#51596b';
        ctx.font = `${12 * dpr}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(p.label, q.x + (this._DOT_R + 4) * dpr, q.y - (this._DOT_R - 2) * dpr);
      }
    }
  }

  // ── pointer handling ────────────────────────────────────────────────────────
  _bindEvents() {
    const pos = (e) => {
      const r = this.canvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) * this._dpr, y: (e.clientY - r.top) * this._dpr };
    };
    const hit = (px, py) => {
      for (let i = 0; i < this.points.length; i++) {
        const q = this.worldToPx(this.points[i]);
        if (Math.hypot(px - q.x, py - q.y) <= this._GRAB_R * this._dpr) return i;
      }
      return -1;
    };

    this.canvas.addEventListener('pointerdown', (e) => {
      const { x, y } = pos(e); const i = hit(x, y);
      if (i >= 0) {
        this._dragging = i;
        this.canvas.setPointerCapture?.(e.pointerId);
        this.canvas.style.cursor = 'grabbing';
        e.preventDefault();
      }
    });
    this.canvas.addEventListener('pointermove', (e) => {
      const { x, y } = pos(e);
      if (this._dragging >= 0) {
        const w = this.pxToWorld(x, y);
        // clamp to the actually-visible world extent (rectangular-aware)
        const ext = this._worldExtent();
        const cx = (v) => Math.max(ext.cx - ext.hx, Math.min(ext.cx + ext.hx, v));
        const cy = (v) => Math.max(ext.cy - ext.hy, Math.min(ext.cy + ext.hy, v));
        this.points[this._dragging].x = cx(w.x);
        this.points[this._dragging].y = cy(w.y);
        this.draw();
        this.onDrag(this.getPoints());
        e.preventDefault();
      } else {
        this.canvas.style.cursor = hit(x, y) >= 0 ? 'grab' : 'default';
      }
    });
    window.addEventListener('pointerup', () => {
      if (this._dragging >= 0) {
        this._dragging = -1;
        this.canvas.style.cursor = 'grab';
        this.onDragEnd(this.getPoints());
      }
    });
    window.addEventListener('resize', () => this.resize());
  }
}

// ============================================================================
// EpochPlots — two stacked line plots (loss on top, sharpness below) sharing a
// common x-axis (epoch/step). Each series is one colored curve, color-matched
// to the trajectories in the FunctionPlane. Pure drawing: it is handed arrays
// of { step, value } series and renders them.
// ----------------------------------------------------------------------------
export class EpochPlots {
  /**
   * @param {string} canvasId
   * @param {object} opts
   *   topLabel    : y-axis label for the upper plot (default 'loss')
   *   bottomLabel : y-axis label for the lower plot (default 'sharpness')
   *   topLog      : log-scale the upper plot (default true — loss spans decades)
   */
  constructor(canvasId, opts = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.topLabel = opts.topLabel ?? 'loss';
    this.bottomLabel = opts.bottomLabel ?? 'sharpness';
    this.topLog = opts.topLog ?? false;

    // series: { top: [{color, pts:[[step,val]...]}], bottom: [...] }
    this.top = [];
    this.bottom = [];

    this._dpr = 1;
    this._bind();
    this.resize();
  }

  /** top/bottom: arrays of { color, pts:[[step,value]...] }. */
  setSeries(top, bottom) { this.top = top || []; this.bottom = bottom || []; this.draw(); }

  resize() {
    this._dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this._dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this._dpr);
    this.draw();
  }

  _bounds(series, log) {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (const [x, y] of s.pts) {
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      const yy = log ? Math.log10(Math.max(y, 1e-12)) : y;
      if (yy < ymin) ymin = yy; if (yy > ymax) ymax = yy;
    }
    if (!isFinite(xmin)) { xmin = 0; xmax = 1; ymin = 0; ymax = 1; }
    if (xmax === xmin) xmax = xmin + 1;
    if (ymax === ymin) ymax = ymin + 1;
    const pad = 0.08 * (ymax - ymin);
    return { xmin, xmax, ymin: ymin - pad, ymax: ymax + pad };
  }

  _drawPanel(px, py, pw, ph, series, label, log) {
    const ctx = this.ctx, dpr = this._dpr;
    // frame
    ctx.strokeStyle = '#d9d6cd'; ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(px, py, pw, ph);

    const b = this._bounds(series, log);
    const X = (x) => px + ((x - b.xmin) / (b.xmax - b.xmin)) * pw;
    const Y = (y) => {
      const yy = log ? Math.log10(Math.max(y, 1e-12)) : y;
      return py + ph - ((yy - b.ymin) / (b.ymax - b.ymin)) * ph;
    };

    for (const s of series) {
      if (!s.pts.length) continue;
      ctx.beginPath();
      for (let i = 0; i < s.pts.length; i++) {
        const [x, y] = s.pts[i];
        const qx = X(x), qy = Y(y);
        if (i === 0) ctx.moveTo(qx, qy); else ctx.lineTo(qx, qy);
      }
      ctx.lineWidth = 2 * dpr; ctx.strokeStyle = s.color || '#3a5bbf';
      ctx.stroke();
    }

    // y-axis label
    ctx.save();
    ctx.translate(px - 6 * dpr, py + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#666'; ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(log ? `${label} (log)` : label, 0, 0);
    ctx.restore();
  }

  draw() {
    const ctx = this.ctx, dpr = this._dpr;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 40 * dpr, padR = 12 * dpr, padT = 10 * dpr, padB = 26 * dpr;
    const gap = 28 * dpr;
    const pw = W - padL - padR;
    const ph = (H - padT - padB - gap) / 2;

    this._drawPanel(padL, padT, pw, ph, this.top, this.topLabel, this.topLog);
    this._drawPanel(padL, padT + ph + gap, pw, ph, this.bottom, this.bottomLabel, false);

    // shared x-axis label
    ctx.fillStyle = '#666'; ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('time', padL + pw / 2, H - 6 * dpr);
  }

  _bind() { window.addEventListener('resize', () => this.resize()); }
}

// ============================================================================
// ProbabilityPlot — line graph of P(ball | r) vs norm ratio r ∈ [0,1], with a
// vertical readout line at the current norm. Pure renderer: handed a curve and
// a marker position. The curve swaps when the dimension changes.
// ----------------------------------------------------------------------------
export class ProbabilityPlot {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.curve = [];      // [[r, P], ...]
    this.markR = null;    // current norm ratio for the vertical line (or null)
    this.markP = null;    // P at that r (for the dot)
    this.lineColor = '#3a5bbf';
    this._dpr = 1;
    window.addEventListener('resize', () => this.resize());
    this.resize();
  }
  setCurve(curve, color) { this.curve = curve || []; if (color) this.lineColor = color; this.draw(); }
  setMark(r, P) { this.markR = r; this.markP = P; this.draw(); }
  resize() {
    this._dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this._dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this._dpr);
    this.draw();
  }
  draw() {
    const ctx = this.ctx, dpr = this._dpr;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const padL = 44 * dpr, padR = 12 * dpr, padT = 12 * dpr, padB = 30 * dpr;
    const pw = W - padL - padR, ph = H - padT - padB;
    // Axes extend a little beyond the data range so the caps are visually clear:
    // probability tops out at 0.5 (never reaches the top), and the curve runs to
    // zero before the right edge (norm ratios > 1 have zero probability).
    const xmax = 1.15;   // data x ∈ [0,1]; show a bit past 1
    const ymax = 0.58;   // data y ∈ [0,0.5]; show a bit past 0.5
    const X = (r) => padL + (r / xmax) * pw;
    const Y = (p) => padT + ph - (p / ymax) * ph;

    // frame
    ctx.strokeStyle = '#d9d6cd'; ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(padL, padT, pw, ph);
    // reference line at the 0.5 cap, and at r = 1 (where probability hits 0)
    ctx.save();
    ctx.setLineDash([2 * dpr, 3 * dpr]); ctx.lineWidth = 1 * dpr; ctx.strokeStyle = '#cfcabd';
    const capY = Y(0.5);
    ctx.beginPath(); ctx.moveTo(padL, capY); ctx.lineTo(padL + pw, capY); ctx.stroke();
    const oneX = X(1);
    ctx.beginPath(); ctx.moveTo(oneX, padT); ctx.lineTo(oneX, padT + ph); ctx.stroke();
    ctx.restore();
    // small labels for the caps
    ctx.fillStyle = '#aaa'; ctx.font = `${9 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('max 0.5', padL + 4 * dpr, capY - 2 * dpr);

    // gridlines + ticks (x: 0,0.5,1 ; y: 0,0.25,0.5)
    ctx.fillStyle = '#888'; ctx.font = `${10 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const rx of [0, 0.5, 1]) {
      const gx = X(rx);
      ctx.fillText(rx.toFixed(rx === 0 ? 0 : 1), gx, padT + ph + 4 * dpr);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const py of [0, 0.25, 0.5]) {
      const gy = Y(py);
      ctx.fillStyle = '#888'; ctx.fillText(py.toFixed(2), padL - 5 * dpr, gy);
    }

    // the probability curve
    if (this.curve.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < this.curve.length; i++) {
        const [r, p] = this.curve[i];
        const qx = X(r), qy = Y(p);
        if (i === 0) ctx.moveTo(qx, qy); else ctx.lineTo(qx, qy);
      }
      ctx.lineWidth = 2.5 * dpr; ctx.strokeStyle = this.lineColor; ctx.stroke();
    }

    // vertical norm line + dot at (markR, markP)
    if (this.markR != null) {
      const mx = X(this.markR);
      ctx.save();
      ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = '#c8612f';
      ctx.beginPath(); ctx.moveTo(mx, padT); ctx.lineTo(mx, padT + ph); ctx.stroke();
      ctx.restore();
      if (this.markP != null) {
        const my = Y(this.markP);
        ctx.beginPath(); ctx.arc(mx, my, 4.5 * dpr, 0, 2 * Math.PI);
        ctx.fillStyle = '#c8612f'; ctx.fill();
        ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = '#fff'; ctx.stroke();
      }
    }

    // axis labels
    ctx.fillStyle = '#666'; ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('‖p‖ / ‖p*‖', padL + pw / 2, H - 4 * dpr);
    ctx.save();
    ctx.translate(11 * dpr, padT + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'top'; ctx.fillText('P(in ball)', 0, 0);
    ctx.restore();
  }
}

// ============================================================================
// SharpeningBall — the right panel. Target p* fixed at (1,0); the Thales ball
// (center (0.5,0), radius 0.5) is drawn. The user drags a point to set the test
// norm; the widget shades the ARC of in-ball directions at that radius and
// reports the radius. Dimension drives an illustrative radial opacity (the ball
// "gains volume" toward its center as d grows). Emits onDrag(normRatio).
// ----------------------------------------------------------------------------
export class SharpeningBall {
  constructor(canvasId, opts = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.range = opts.range ?? 1.6;          // world half-extent on the shorter axis
    this.onDrag = opts.onDrag || (() => {});
    this.pstar = [1, 0];
    this.dim = 2;                             // for opacity illustration
    this.pointR = 0.35;                       // current dragged radius (norm), along +x by default
    this._dragAngle = 0;                      // remember the drag direction for display
    this._dpr = 1; this._dragging = false;
    this._bind();
    this.resize();
  }
  setDim(d) { this.dim = d; this.draw(); }
  setRadius(r) { this.pointR = Math.max(0, Math.min(this.range, r)); this.draw(); }
  getNormRatio() { return this.pointR / Math.hypot(this.pstar[0], this.pstar[1]); }

  resize() {
    this._dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this._dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this._dpr);
    this.draw();
  }
  _metrics() {
    const w = this.canvas.width, h = this.canvas.height;
    const pad = Math.min(w, h) * 0.08;
    const innerW = w - 2 * pad, innerH = h - 2 * pad;
    const scale = Math.min(innerW, innerH) / (2 * this.range);
    return { w, h, pad, innerW, innerH, scale };
  }
  _w2p(x, y) {
    const m = this._metrics();
    return { x: m.pad + m.innerW / 2 + x * m.scale, y: m.pad + m.innerH / 2 - y * m.scale };
  }
  _p2w(px, py) {
    const m = this._metrics();
    return { x: (px - m.pad - m.innerW / 2) / m.scale, y: -(py - m.pad - m.innerH / 2) / m.scale };
  }

  draw() {
    const ctx = this.ctx, dpr = this._dpr, m = this._metrics();
    ctx.clearRect(0, 0, m.w, m.h);
    const o = this._w2p(0, 0);

    // axes
    ctx.strokeStyle = '#e2ddd2'; ctx.lineWidth = 1 * dpr;
    ctx.beginPath(); ctx.moveTo(m.pad, o.y); ctx.lineTo(m.pad + m.innerW, o.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(o.x, m.pad); ctx.lineTo(o.x, m.pad + m.innerH); ctx.stroke();

    // the sharpening ball: center (0.5,0), radius 0.5. Illustrative radial
    // opacity that intensifies toward the center as dimension grows.
    const c = this._w2p(0.5, 0);
    const rpx = Math.abs(this._w2p(1, 0).x - c.x);
    // base fill + extra central density for d>2
    const extra = Math.min(0.5, (this.dim - 2) * 0.14);  // 0 at d=2, grows with d
    const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, rpx);
    grad.addColorStop(0, `rgba(200,97,47,${0.12 + extra})`);
    grad.addColorStop(1, `rgba(200,97,47,0.08)`);
    ctx.beginPath(); ctx.arc(c.x, c.y, rpx, 0, 2 * Math.PI);
    ctx.fillStyle = grad; ctx.fill();
    ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = '#c8612f'; ctx.stroke();

    // the radius circle of the current test norm, centered at origin
    const Rpx = this.pointR * m.scale;
    if (Rpx > 0.5) {
      ctx.save();
      ctx.setLineDash([3 * dpr, 3 * dpr]); ctx.lineWidth = 1 * dpr; ctx.strokeStyle = '#9aa3b8';
      ctx.beginPath(); ctx.arc(o.x, o.y, Rpx, 0, 2 * Math.PI); ctx.stroke();
      ctx.restore();

      // highlight the ARC of this circle that lies inside the ball:
      // a point at radius R and angle φ is inside when R < ||p*|| cos φ, i.e.
      // cos φ > R/||p*||  ⇒ |φ| < arccos(R/||p*||).
      const ratio = this.getNormRatio();
      if (ratio < 1) {
        const phi = Math.acos(Math.min(1, ratio));
        ctx.lineWidth = 3 * dpr; ctx.strokeStyle = '#c8612f';
        ctx.beginPath(); ctx.arc(o.x, o.y, Rpx, -phi, phi); ctx.stroke();   // canvas y is down; symmetric so fine
      }
    }

    // p* marker
    const ps = this._w2p(this.pstar[0], this.pstar[1]);
    ctx.beginPath(); ctx.arc(ps.x, ps.y, 5 * dpr, 0, 2 * Math.PI);
    ctx.fillStyle = '#c8612f'; ctx.fill(); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#51596b'; ctx.font = `${12 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('p*', ps.x + 8 * dpr, ps.y - 6 * dpr);

    // draggable norm point (on the +x ray by default; user can drag anywhere,
    // we take its distance from origin as the norm)
    const dp = this._w2p(this.pointR * Math.cos(this._dragAngle), this.pointR * Math.sin(this._dragAngle));
    ctx.beginPath(); ctx.arc(dp.x, dp.y, 7 * dpr, 0, 2 * Math.PI);
    ctx.fillStyle = '#3a5bbf'; ctx.fill(); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#51596b'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('p', dp.x + 8 * dpr, dp.y - 6 * dpr);
  }

  _bind() {
    const pos = (e) => { const r = this.canvas.getBoundingClientRect(); return { x: (e.clientX - r.left) * this._dpr, y: (e.clientY - r.top) * this._dpr }; };
    const ptPx = () => this._w2p(this.pointR * Math.cos(this._dragAngle), this.pointR * Math.sin(this._dragAngle));
    this.canvas.addEventListener('pointerdown', (e) => {
      const { x, y } = pos(e); const q = ptPx();
      if (Math.hypot(x - q.x, y - q.y) <= 16 * this._dpr) {
        this._dragging = true; this.canvas.setPointerCapture?.(e.pointerId);
        this.canvas.style.cursor = 'grabbing'; e.preventDefault();
      }
    });
    this.canvas.addEventListener('pointermove', (e) => {
      const { x, y } = pos(e);
      if (this._dragging) {
        const w = this._p2w(x, y);
        this.pointR = Math.max(0, Math.min(this.range, Math.hypot(w.x, w.y)));
        this._dragAngle = Math.atan2(w.y, w.x);
        this.draw();
        this.onDrag(this.getNormRatio());
        e.preventDefault();
      } else {
        const q = ptPx();
        this.canvas.style.cursor = Math.hypot(x - q.x, y - q.y) <= 16 * this._dpr ? 'grab' : 'default';
      }
    });
    window.addEventListener('pointerup', () => {
      if (this._dragging) { this._dragging = false; this.canvas.style.cursor = 'grab'; }
    });
    window.addEventListener('resize', () => this.resize());
  }
}

// ============================================================================
// LinePlot — a small, general static line plot: one or more curves in a framed,
// axis-labeled box with configurable x/y ranges and optional vertical marker
// bars. No interactivity. Built for the product-of-distributions widget, but
// deliberately generic.
//
// NOTE: this overlaps in spirit with ProbabilityPlot (which is hardwired to the
// sharpening-probability axes and has a draggable marker). The two could be
// merged into one configurable plot class later; kept separate for now so the
// working ProbabilityPlot is untouched.
// ----------------------------------------------------------------------------
export class LinePlot {
  constructor(canvasId, opts = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.xRange = opts.xRange ?? [0, 1];     // [xmin, xmax]
    this.yRange = opts.yRange ?? [0, 1];     // [ymin, ymax]
    this.xLabel = opts.xLabel ?? '';
    this.yLabel = opts.yLabel ?? '';
    this.xTicks = opts.xTicks ?? null;       // array of x tick values, or null
    this.curves = [];                        // [{ pts:[[x,y]...], color, width, dashed }]
    this.bars = [];                          // [{ x, color }] vertical marker lines
    this._dpr = 1;
    window.addEventListener('resize', () => this.resize());
    this.resize();
  }
  setCurves(curves) { this.curves = curves || []; this.draw(); }
  setBars(bars) { this.bars = bars || []; this.draw(); }
  resize() {
    this._dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * this._dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * this._dpr);
    this.draw();
  }
  draw() {
    const ctx = this.ctx, dpr = this._dpr;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const padL = 40 * dpr, padR = 12 * dpr, padT = 12 * dpr, padB = 28 * dpr;
    const pw = W - padL - padR, ph = H - padT - padB;
    const [x0, x1] = this.xRange, [y0, y1] = this.yRange;
    const X = (x) => padL + ((x - x0) / (x1 - x0)) * pw;
    const Y = (y) => padT + ph - ((y - y0) / (y1 - y0)) * ph;

    // frame
    ctx.strokeStyle = '#d9d6cd'; ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(padL, padT, pw, ph);

    // x ticks
    const ticks = this.xTicks || [x0, (x0 + x1) / 2, x1];
    ctx.fillStyle = '#888'; ctx.font = `${10 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const tx of ticks) {
      const gx = X(tx);
      ctx.strokeStyle = '#efece5'; ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, padT + ph); ctx.stroke();
      ctx.fillStyle = '#888'; ctx.fillText(String(tx), gx, padT + ph + 4 * dpr);
    }

    // vertical marker bars (e.g. ±σ)
    for (const b of this.bars) {
      const bx = X(b.x);
      ctx.save();
      ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = b.color || '#888';
      ctx.beginPath(); ctx.moveTo(bx, padT); ctx.lineTo(bx, padT + ph); ctx.stroke();
      ctx.restore();
    }

    // curves
    for (const c of this.curves) {
      if (!c.pts || c.pts.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < c.pts.length; i++) {
        const [x, y] = c.pts[i];
        const qx = X(x), qy = Y(y);
        if (i === 0) ctx.moveTo(qx, qy); else ctx.lineTo(qx, qy);
      }
      ctx.lineWidth = (c.width || 2) * dpr;
      ctx.strokeStyle = c.color || '#3a5bbf';
      if (c.dashed) ctx.setLineDash([5 * dpr, 4 * dpr]); else ctx.setLineDash([]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // axis labels
    ctx.fillStyle = '#666'; ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    if (this.xLabel) ctx.fillText(this.xLabel, padL + pw / 2, H - 4 * dpr);
    if (this.yLabel) {
      ctx.save();
      ctx.translate(10 * dpr, padT + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'top'; ctx.fillText(this.yLabel, 0, 0);
      ctx.restore();
    }
  }
}
