// One cardiac clock shared by the 3D heartbeat and the ECG strip, so the
// ventricles contract right after every R wave drawn on the monitor.

export const clock = { t0: performance.now() };

export function cardiacPhase(nowMs: number, hr: number): number {
  const beats = ((nowMs - clock.t0) / 1000) * (hr / 60);
  return beats - Math.floor(beats); // 0 = R peak
}

/** Ventricular contraction 0..1 (systole ~ 5-40 % of the cycle). */
export function contraction(phase: number): number {
  if (phase < 0.05 || phase > 0.4) return 0;
  const s = Math.sin((Math.PI * (phase - 0.05)) / 0.35);
  return s * s;
}

export interface EcgFlags {
  stDepression: boolean;
  stElevation: boolean;
  tInversion: boolean;
  qWave: boolean;
  lbbb: boolean;
  rbbb: boolean;
  lvh: boolean;
}

const g = (x: number, mu: number, sd: number, a: number) => a * Math.exp(-((x - mu) ** 2) / (2 * sd * sd));

/**
 * Schematic single-lead ECG for one cardiac cycle (phase 0..1, R peak at 0).
 * Morphology reflects the patient's recorded ECG findings. Illustrative only.
 */
export function ecgSample(phase: number, f: EcgFlags): number {
  // shift so the P wave precedes R: work in [-0.25, 0.75)
  let x = phase;
  if (x > 0.75) x -= 1;
  const wide = f.lbbb || f.rbbb ? 1.9 : 1;
  const rAmp = f.lvh ? 1.45 : 1.05;
  let y = 0;
  y += g(x, -0.16, 0.022, 0.12); // P
  y += g(x, -0.028 * wide, 0.009 * wide, f.qWave ? -0.38 : -0.1); // Q
  y += g(x, 0, 0.011 * wide, rAmp); // R
  if (f.rbbb) y += g(x, 0.045, 0.012, 0.45); // R' (rSR')
  y += g(x, 0.03 * wide, 0.011 * wide, -0.26); // S
  // ST segment level
  const st = f.stElevation ? 0.16 : f.stDepression ? -0.14 : 0;
  if (x > 0.045 * wide && x < 0.2) y += st * Math.min(1, (x - 0.045 * wide) / 0.03) * (x < 0.17 ? 1 : (0.2 - x) / 0.03);
  y += g(x, 0.27, 0.045, f.tInversion || f.lbbb ? -0.24 : 0.28); // T
  return y;
}
