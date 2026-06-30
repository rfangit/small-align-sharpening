#!/usr/bin/env python3
# ============================================================================
# GENERATOR for the PS-probability tables (audit / re-generation script).
# ============================================================================
# NOT loaded by the website. Run with `python3 gen_distributions.py` to
# regenerate distributions.js. Computes, for each dimension d and norm ratio
# r = ||p|| / ||p*|| in [0, 1], the probability that a uniformly-random
# direction lands inside the sharpening ball:
#
#     P(ball | r) = ∫_0^{arccos r} f_d(θ) dθ,
#     f_d(θ) = Γ(d/2) / [√π Γ((d-1)/2)] · (sin θ)^{d-2}.
#
# f_d is the distribution of the angle between a uniform point on S^{d-1} and a
# fixed reference direction. The integral is done by composite Simpson's rule on
# θ ∈ [0, arccos r]; this is checked against the closed forms for d = 2, 3, 4
# (plus P(0) = 1/2 and P(1) = 0) before anything is written out.
#
# This is the Python port of gen_distributions.mjs; it produces the same table.
# ----------------------------------------------------------------------------

import math


# normalization constant C_d = Γ(d/2) / [√π Γ((d-1)/2)]
# (math.lgamma is the log-gamma in the standard library, so this is exact.)
def norm_const(d):
    return math.exp(math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2))


# f_d(θ) = C_d (sin θ)^{d-2}
def fd(d, theta):
    return norm_const(d) * math.sin(theta) ** (d - 2)


# ∫_a^b g dθ by composite Simpson's rule (n even).
def simpson(g, a, b, n=2000):
    if b <= a:
        return 0.0
    if n % 2:
        n += 1
    h = (b - a) / n
    s = g(a) + g(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * g(a + i * h)
    return s * h / 3


# P(ball | r) for dimension d and norm ratio r ∈ [0, 1].
def p_ball(d, r):
    rc = min(1.0, max(0.0, r))
    upper = math.acos(rc)  # arccos(r)
    return simpson(lambda th: fd(d, th), 0.0, upper)


# ── checks against closed forms before generating ──────────────────────────
def check():
    ok = True

    def approx(a, b, t=1e-6):
        return abs(a - b) <= t

    def log(name, passed, got, want):
        nonlocal ok
        if not passed:
            ok = False
        print(f"{'PASS' if passed else 'FAIL'}: {name}  got={got}  want={want}")

    # d=2: f_2 = 1/π uniform → P = arccos(r)/π
    for r in (0, 0.3, 0.5, 0.8, 1):
        want = math.acos(r) / math.pi
        log(f"d=2 r={r}", approx(p_ball(2, r), want), f"{p_ball(2, r):.6f}", f"{want:.6f}")
    # d=3: f_3 = (1/2) sinθ → P = (1 - r)/2
    for r in (0, 0.3, 0.5, 1):
        want = (1 - r) / 2
        log(f"d=3 r={r}", approx(p_ball(3, r), want), f"{p_ball(3, r):.6f}", f"{want:.6f}")
    # d=4: f_4 = (2/π) sin^2θ → (2/π)[φ/2 - sin(2φ)/4], φ=arccos r
    for r in (0, 0.3, 0.7, 1):
        phi = math.acos(r)
        want = (2 / math.pi) * (phi / 2 - math.sin(2 * phi) / 4)
        log(f"d=4 r={r}", approx(p_ball(4, r), want), f"{p_ball(4, r):.6f}", f"{want:.6f}")
    # P(ball | 0) = 1/2 for every d (forward half-space)
    for d in (2, 3, 4, 5, 6, 7, 8):
        log(f"d={d} r=0 → 0.5", approx(p_ball(d, 0), 0.5), f"{p_ball(d, 0):.6f}", "0.500000")
    # P(ball | 1) = 0 for every d (cone closed)
    for d in (2, 3, 4, 5, 6, 7, 8):
        log(f"d={d} r=1 → 0", approx(p_ball(d, 1), 0, 1e-9), f"{p_ball(d, 1):.2e}", "0")
    return ok


# ============================================================================
# PRODUCT-OF-VARIABLES DENSITIES (for the "random norm" concentration widget).
# ============================================================================
# A depth-n function norm carries a product B = ∏ b_i of scalar layers. Even at
# fixed variance, the product concentrates near zero as n grows. We tabulate the
# density of Z_k = X_1 X_2 ... X_k for k = 1..4, where each X is either a uniform
# or a standard normal, each normalized to unit variance so the curves compare.
#
#   uniform factor: X ~ U[-a, a] with a = √3  (so Var = a²/3 = 1)
#   normal factor:  X ~ N(0, 1)
#
# Since ln|Z_k| = Σ ln|X_i| is a sum, the density of ln|Z_k| is the k-fold
# convolution of the density of ln|X|; we transform back to z. Checked against
# the exact k=2 closed forms (uniform log-density, Gaussian K0) before writing.
# ----------------------------------------------------------------------------

UNIFORM_A = math.sqrt(3.0)   # half-width giving unit variance

# log-|X| grid (shared by both factor types). Resolution chosen for smooth
# visual curves, not high-precision quadrature — the product densities are for
# plotting concentration, so a modest grid keeps the pure-Python convolution fast.
_L_MIN, _L_MAX, _NL = -30.0, 4.0, 1200
_L_DT = (_L_MAX - _L_MIN) / _NL


def _density_log_abs_uniform(t):
    # |X| ~ U[0,a]; L = ln|X| has density e^{L}/a on L < ln a.
    a = UNIFORM_A
    return math.exp(t) / a if t < math.log(a) else 0.0


def _density_log_abs_normal(t):
    # |X| half-normal; L = ln|X| has density sqrt(2/pi) e^{L} exp(-e^{2L}/2).
    return math.sqrt(2.0 / math.pi) * math.exp(t) * math.exp(-math.exp(2.0 * t) / 2.0)


def _convolve(p, q, dt):
    n, m = len(p), len(q)
    out = [0.0] * (n + m - 1)
    for i in range(n):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(m):
            out[i + j] += pi * q[j] * dt
    return out


def _build_conv(kind, k):
    # k-fold convolution (in L = ln|X|) of the factor's log-density. Returns
    # (conv array, startL). Computed once, then sampled at many z.
    base = _density_log_abs_uniform if kind == "uniform" else _density_log_abs_normal
    f = [base(_L_MIN + i * _L_DT) for i in range(_NL + 1)]
    conv = f[:]
    for _ in range(k - 1):
        conv = _convolve(conv, f, _L_DT)
    return conv, k * _L_MIN


def _sample_conv(conv, startL, zs):
    end = startL + (len(conv) - 1) * _L_DT
    out = []
    for z in zs:
        if z <= 0:
            out.append(0.0); continue
        Lz = math.log(z)
        if Lz <= startL or Lz >= end:
            out.append(0.0); continue
        idx = (Lz - startL) / _L_DT
        i0 = int(idx); frac = idx - i0
        fL = conv[i0] * (1 - frac) + conv[i0 + 1] * frac
        out.append(0.5 * fL / z)
    return out


def product_density(kind, k, zs):
    # density of Z_k = ∏ X_i at points zs (z>0; symmetric). Convenience wrapper
    # that builds the convolution then samples — fine for a few points / checks.
    conv, startL = _build_conv(kind, k)
    return _sample_conv(conv, startL, zs)


def base_density(kind, z):
    # the single (k=1) distribution, evaluated directly (no convolution). Unit
    # variance: uniform U[-√3,√3] is flat 1/(2√3) inside, Gaussian is N(0,1).
    if kind == "uniform":
        return 1.0 / (2 * UNIFORM_A) if abs(z) <= UNIFORM_A else 0.0
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def check_products():
    ok = True

    def approx(a, b, t):
        return abs(a - b) <= t

    def log(name, passed, got, want):
        nonlocal ok
        if not passed:
            ok = False
        print(f"{'PASS' if passed else 'FAIL'}: {name}  got={got}  want={want}")

    # k=2 uniform closed form: f_Z(z) = 1/(2a²) ln(a²/|z|), a²=3.
    # Tolerance is loose: these are visual plotting tables on a modest grid, and
    # the uniform factor's hard edge at ln(a) discretizes less cleanly than the
    # smooth Gaussian. A ~1% match confirms the convolution is right.
    a2 = UNIFORM_A ** 2
    for z in (0.1, 0.5, 1.0, 2.0):
        want = (1.0 / (2 * a2)) * math.log(a2 / z) if z < a2 else 0.0
        got = product_density("uniform", 2, [z])[0]
        log(f"uniform k=2 z={z}", approx(got, want, 1e-2), f"{got:.5f}", f"{want:.5f}")

    # k=2 gaussian closed form: f_Z(z) = K0(|z|)/π  (σ=1)
    def K0(x):
        s, N, T = 0.0, 3000, 25.0
        for i in range(N):
            t = (i + 0.5) * T / N
            s += math.exp(-x * math.cosh(t)) * (T / N)
        return s
    for z in (0.2, 0.5, 1.0, 2.0):
        want = K0(z) / math.pi
        got = product_density("normal", 2, [z])[0]
        log(f"normal k=2 z={z}", approx(got, want, 3e-3), f"{got:.5f}", f"{want:.5f}")

    # Shape sanity only — these are visual plots, so we confirm the curve matches
    # the exact k=2 closed forms and don't fuss over total-mass quadrature.
    return ok


# ── emit distributions.js ───────────────────────────────────────────────────
DIMS = [2, 3, 4, 5, 6, 7, 8]   # selectable dimensions (slider 2..8)
STEPS = 201                    # r grid: 0, 0.005, ..., 1


def fmt(x):
    # match the .mjs output: trim trailing zeros, no scientific notation
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def generate():
    table = {}
    for d in DIMS:
        rows = []
        for i in range(STEPS):
            r = i / (STEPS - 1)
            rows.append((round(r, 5), round(p_ball(d, r), 8)))
        table[d] = rows

    header = f"""// ============================================================================
// distributions.js — PRECOMPUTED tables (generated; do not edit).
// ============================================================================
// Generated by gen_distributions.py. To regenerate: `python3 gen_distributions.py`.
// Self-checks against closed forms before writing, so this file is auditable:
// re-run the generator and diff.
//
// PS_TABLE[d]: [r, P] pairs — probability a uniform-random direction lands in
//   the sharpening ball at norm ratio r = ||p||/||p*||, for dimension d.
//     P(ball | r) = ∫_0^{{arccos r}} f_d(θ) dθ,  f_d(θ) ∝ (sin θ)^{{d-2}}.
//   Dimensions: {', '.join(map(str, DIMS))}.  Grid: {STEPS} points on r ∈ [0,1].
//
// PROD_TABLE[kind][k]: [z, density] pairs — density of a product of k unit-
//   variance factors (kind = 'uniform' or 'normal'). Shows the norm
//   concentrating near 0 as k grows. Density clipped to ~3x the single peak (k=1 is the true base, unclipped).
// ----------------------------------------------------------------------------

export const PS_DIMS = [{', '.join(map(str, DIMS))}];

export const PS_TABLE = {{
"""

    body = ""
    for d in DIMS:
        pairs = ", ".join(f"[{fmt(r)},{fmt(p)}]" for (r, p) in table[d])
        body += f"  {d}: [{pairs}],\n"
    body += "};\n\n"

    # ── product-of-variables densities, k = 1..4 factors (unit variance each) ──
    # Plotted on z ∈ [-ZSPAN, ZSPAN]. The k≥2 products have a -ln|z| spike at the
    # origin, clipped at ~3x the single distribution's peak so curves stay
    # readable. The k=1 curve is the TRUE base distribution (flat uniform / N(0,1)
    # bump), evaluated directly — not a convolution, and not clipped.
    PROD_K = [1, 2, 3, 4]
    ZSPAN, ZSTEPS = 4.0, 161
    clip_for = {
        "uniform": 3.0 * (1.0 / (2 * UNIFORM_A)),        # 3x U[-√3,√3] flat height
        "normal": 3.0 * (1.0 / math.sqrt(2 * math.pi)),  # 3x N(0,1) peak
    }
    body += "export const PROD_K = [1, 2, 3, 4];\n\n"
    body += "export const PROD_TABLE = {\n"
    for kind in ("uniform", "normal"):
        clip = clip_for[kind]
        body += f"  {kind}: {{\n"
        zs = [(-ZSPAN + 2 * ZSPAN * i / (ZSTEPS - 1)) for i in range(ZSTEPS)]
        for k in PROD_K:
            if k == 1:
                # true base distribution, no convolution, no clip
                rows = [(round(z, 4), round(base_density(kind, z), 6)) for z in zs]
            else:
                conv, startL = _build_conv(kind, k)      # build convolution once
                pos = [abs(z) if abs(z) > 1e-9 else 1e-9 for z in zs]
                dens = _sample_conv(conv, startL, pos)
                rows = []
                for z, v in zip(zs, dens):
                    vv = clip if abs(z) <= 1e-9 else min(v, clip)
                    rows.append((round(z, 4), round(vv, 6)))
            pairs = ", ".join(f"[{fmt(z)},{fmt(v)}]" for (z, v) in rows)
            body += f"    {k}: [{pairs}],\n"
        body += "  },\n"
    body += "};\n"

    footer = """
/** Linear-interpolated lookup of P(ball | r) for a given dimension d. */
export function psProbability(d, r) {
  const rows = PS_TABLE[d];
  if (!rows) return 0;
  const rc = Math.min(1, Math.max(0, r));
  let lo = 0, hi = rows.length - 1;
  if (rc <= rows[0][0]) return rows[0][1];
  if (rc >= rows[hi][0]) return rows[hi][1];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (rows[mid][0] <= rc) lo = mid; else hi = mid;
  }
  const [r0, p0] = rows[lo], [r1, p1] = rows[hi];
  const t = (rc - r0) / (r1 - r0);
  return p0 + t * (p1 - p0);
}
"""

    with open("distributions.js", "w") as f:
        f.write(header + body + footer)
    print(f"\nWrote distributions.js: PS dims {','.join(map(str, DIMS))}; "
          f"product tables k=[1,2,3,4] for uniform & normal.")


if __name__ == "__main__":
    passed = check() and check_products()
    print("\n" + ("ALL CHECKS PASS — safe to generate" if passed else "CHECKS FAILED — not generating"))
    if passed:
        generate()
