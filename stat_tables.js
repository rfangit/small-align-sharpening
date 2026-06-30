// ============================================================================
// STAT TABLES — render width×depth probability tables from ps_stats.json.
// ============================================================================
// DESIGN CONTRACT:
//   * Self-contained view builder: fetches the stats JSON, draws heat-shaded
//     HTML tables into a host element. No dependency on dynamics.js / the
//     widget visualization layer; mirrors their "load precomputed data, render"
//     pattern but for static tables rather than canvas.
//   * Adding a future table block is ONE entry in the `groups` array passed to
//     renderStatTables — no new fetch, no new DOM plumbing. The grid, shading,
//     and MathJax re-typeset are handled here once.
//
// USAGE (from index.html or app.js):
//   import { renderStatTables } from './stat_tables.js';
//   renderStatTables('ps-tables', {
//     src: './ps_stats.json',
//     groups: [
//       { label: 'Probability of <em>initial</em> sharpening', metric: 'init_sharpen',
//         titleFor: (kind) => `${KIND_LABEL[kind]} init — $P(\\text{initially sharpens})$` },
//       ...
//     ],
//   });
//
// The JSON shape (produced by gen_ps_stats.ipynb) is:
//   data.meta.{widths,depths,kinds}
//   data.results[kind][d][n][metric]   // d, n are string keys
// Any metric present per-cell can be tabled (init_sharpen, ever_sharpen,
// end_higher, max_not_end, sharpen_then_flatten, init_sharp, end_sharp, ...).
// ----------------------------------------------------------------------------

const ORANGE = '200,97,47';          // shared accent (matches the widgets' #c8612f)
const SHADE_MIN = 0.06;              // cell background alpha at the table min
const SHADE_MAX = 0.42;              // ...and at the table max

// Default per-kind title prefix; callers can override via group.titleFor.
export const KIND_LABEL = { gaussian: 'Gaussian', uniform: 'Uniform' };

// ── one heat-shaded table for a (kind, group) pair ──────────────────────────
// A group renders EITHER a single metric per cell, or two metrics shown as
// "a → b" (e.g. init_sharp → end_sharp). What drives the CELL TEXT and what
// drives the SHADING are decoupled:
//   cellValue(cell)  -> the display string for the cell
//   shadeValue(cell) -> the number used for the heat color (table-relative)
// Both receive the per-cell record { metric: value, ... }, so shading can be by
// the difference, the end value, or anything else. Non-finite shade values get
// a transparent background; non-finite display values render as "—".
function tableHTML(results, kind, widths, depths, title, cellValue, shadeValue) {
  // collect shade values across the table for min/max normalization
  const sv = [];
  for (const w of widths) for (const n of depths) sv.push(shadeValue(results[kind][w][n]));
  const finite = sv.filter((v) => Number.isFinite(v));
  const vmin = finite.length ? Math.min(...finite) : 0;
  const vmax = finite.length ? Math.max(...finite) : 1;
  const span = vmax - vmin || 1;

  const shade = (v) => {
    if (!Number.isFinite(v)) return 'transparent';
    const t = (v - vmin) / span;
    return `rgba(${ORANGE},${(SHADE_MIN + (SHADE_MAX - SHADE_MIN) * t).toFixed(2)})`;
  };

  const head =
    `<tr><th class="corner">$d \\backslash n$</th>` +
    depths.map((n) => `<th>${n}</th>`).join('') + `</tr>`;
  const body = widths.map((w) =>
    `<tr><th>${w}</th>` +
    depths.map((n) => {
      const cell = results[kind][w][n];
      return `<td style="background:${shade(shadeValue(cell))}">${cellValue(cell)}</td>`;
    }).join('') +
    `</tr>`).join('');

  return `<div class="stat-table-card">
      <div class="stat-table-title">${title}</div>
      <table class="stat-table"><thead>${head}</thead><tbody>${body}</tbody></table>
    </div>`;
}

// Build the (cellValue, shadeValue) pair for a group, resolving the single- vs
// two-metric cases and the various shading options into plain functions.
function resolveGroup(g) {
  const fmt = g.fmt || ((v) => (Number.isFinite(v) ? v.toFixed(2) : '—'));

  if (g.metrics) {
    // two-metric "a → b" cell (e.g. ['init_sharp','end_sharp'])
    const [ma, mb] = g.metrics;
    const arrow = g.arrow || ' → ';
    const cellValue = (cell) =>
      `<span class="cell-from">${fmt(cell[ma])}</span>${arrow}<span class="cell-to">${fmt(cell[mb])}</span>`;
    // default shading: by the END value (mb). Override with shadeBy:
    //   'end' | 'start' | 'diff'  or a function (cell) => number.
    let shadeValue;
    if (typeof g.shadeBy === 'function') shadeValue = g.shadeBy;
    else if (g.shadeBy === 'diff') shadeValue = (cell) => cell[mb] - cell[ma];
    else if (g.shadeBy === 'start') shadeValue = (cell) => cell[ma];
    else shadeValue = (cell) => cell[mb];              // 'end' (default)
    return { cellValue, shadeValue };
  }

  // single-metric cell (unchanged behavior)
  const m = g.metric;
  const cellValue = (cell) => fmt(cell[m]);
  const shadeValue =
    typeof g.shadeBy === 'function' ? g.shadeBy : (cell) => cell[m];
  return { cellValue, shadeValue };
}

// ── public entry point ──────────────────────────────────────────────────────
// hostId  : id of the .stat-table-grid container to fill.
// opts.src    : path to the stats JSON (default './ps_stats.json').
// opts.groups : array of group specs. Each group renders one table per kind.
//   Single-metric group:
//     { label, metric, titleFor?, fmt?, shadeBy? }
//       metric   — key into each cell
//       fmt      — (value) => string  (default: 2 decimals)
//       shadeBy  — (cell) => number   (default: the metric's own value)
//   Two-metric "a → b" group (e.g. init_sharp → end_sharp):
//     { label, metrics:[ma, mb], titleFor?, fmt?, arrow?, shadeBy? }
//       metrics  — [startKey, endKey], shown as "a → b"
//       fmt      — applied to BOTH values (default: 2 decimals)
//       arrow    — separator string (default ' → ')
//       shadeBy  — 'end' (default) | 'start' | 'diff' | (cell) => number
//   Common to both:
//       label    — centered header HTML above the pair of tables
//       titleFor — (kind) => card title HTML (default: "<Kind> — <metric(s)>")
// opts.kinds  : override/limit which kinds to show (default: meta.kinds order).
export async function renderStatTables(hostId, opts = {}) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const src = opts.src || host.dataset.src || './ps_stats.json';

  let data;
  try {
    const resp = await fetch(src);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    data = await resp.json();
  } catch (e) {
    host.innerHTML = `<div class="stat-table-loading">could not load ${src} (${e.message})</div>`;
    console.error('[stat_tables]', e);
    return;
  }

  const results = data.results;
  const widths = data.meta.widths.map(String);
  const depths = data.meta.depths.map(String);
  const kinds = opts.kinds || data.meta.kinds;
  const groups = opts.groups || [];

  const defTitle = (kind, g) =>
    `${KIND_LABEL[kind] || kind} — ${g.metrics ? g.metrics.join(' → ') : g.metric}`;

  let html = '';
  for (const g of groups) {
    const { cellValue, shadeValue } = resolveGroup(g);
    const titleFor = g.titleFor || ((kind) => defTitle(kind, g));
    if (g.label) html += `<div class="stat-table-rowlabel">${g.label}</div>`;
    for (const kind of kinds) {
      html += tableHTML(results, kind, widths, depths, titleFor(kind), cellValue, shadeValue);
    }
  }
  host.innerHTML = html;

  // Re-typeset the injected math ($d \backslash n$, titles) once the DOM is in.
  if (window.MathJax?.typesetPromise) {
    try { await MathJax.typesetPromise([host]); } catch (_) { /* MathJax not ready yet */ }
  }
}
