// ============================================================================
// RNG — small seeded pseudo-random generator
// ============================================================================
// Deterministic randomness so a given seed always reproduces the same dataset.
// mulberry32 is a tiny, decent 32-bit generator — fine for visualizations
// (not for cryptography).

export class RNG {
  constructor(seed = 0) {
    // Force to a 32-bit unsigned integer.
    this.state = (seed >>> 0) || 1;
  }

  /** Uniform float in [0, 1). */
  next() {
    let t = (this.state += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform float in [min, max). */
  uniform(min = 0, max = 1) {
    return min + (max - min) * this.next();
  }

  /** Standard normal sample via Box–Muller. */
  gaussian(mean = 0, std = 1) {
    const u1 = Math.max(this.next(), 1e-12);
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }
}
