// Shared colour + scale helpers. The risk ramp is identical in the 3D scene,
// the SVG charts and the PDF report (angiolens/report.py), so a colour always
// means the same probability everywhere.

const STOPS: [number, [number, number, number]][] = [
  [0.0, [0x2a, 0x9d, 0x8f]], // teal   - low
  [0.5, [0xe9, 0xa2, 0x3b]], // amber  - intermediate
  [1.0, [0xc8, 0x1d, 0x25]], // crimson - high
];

export function riskRGB(p: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, p));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [p0, c0] = STOPS[i];
    const [p1, c1] = STOPS[i + 1];
    if (x <= p1) {
      const t = (x - p0) / (p1 - p0);
      return [0, 1, 2].map((k) => (c0[k] + (c1[k] - c0[k]) * t) / 255) as [number, number, number];
    }
  }
  return STOPS[STOPS.length - 1][1].map((c) => c / 255) as [number, number, number];
}

export function riskHex(p: number): string {
  return (
    "#" +
    riskRGB(p)
      .map((c) => Math.round(c * 255).toString(16).padStart(2, "0"))
      .join("")
  );
}

export const RAISE = "#d95926"; // SHAP: pushes risk up   (warm pole)
export const LOWER = "#3987e5"; // SHAP: pushes risk down (cool pole)

export const logit = (p: number) => {
  const q = Math.min(1 - 1e-6, Math.max(1e-6, p));
  return Math.log(q / (1 - q));
};
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export const pct = (p: number, digits = 0) => `${(p * 100).toFixed(digits)}%`;

export function fmtValue(v: number, unit = "", kind = "numeric"): string {
  if (kind === "binary") return v >= 0.5 ? "yes" : "no";
  const s = Number.isInteger(v) ? v.toString() : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
  return unit ? `${s} ${unit}` : s;
}

export const CONFORMAL_LABEL: Record<string, string> = {
  positive: "Confident · stenosis likely",
  negative: "Confident · unlikely",
  indeterminate: "Indeterminate",
  atypical: "Atypical profile",
};

export const CONFORMAL_HINT: Record<string, string> = {
  positive: "The 90% conformal prediction set contains only 'stenotic'.",
  negative: "The 90% conformal prediction set contains only 'normal'.",
  indeterminate:
    "Both outcomes remain plausible at 90% confidence. The model is telling you it does not know; imaging decides.",
  atypical: "Neither outcome is plausible at 90% confidence: this profile is unlike the training cohort.",
};
