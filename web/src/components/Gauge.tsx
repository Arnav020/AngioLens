import { riskHex } from "../lib/color";

/** 240° arc gauge: probability (fill), model range (thin outer band), operating threshold (tick). */
export default function Gauge({ p, range, threshold, size = 132 }: { p: number; range: [number, number]; threshold: number; size?: number }) {
  const r = size / 2 - 10;
  const c = size / 2;
  const a0 = 150; // degrees, start (lower-left)
  const sweep = 240;
  const pt = (frac: number, rad: number) => {
    const a = ((a0 + sweep * frac) * Math.PI) / 180;
    return [c + rad * Math.cos(a), c + rad * Math.sin(a)];
  };
  const arc = (f0: number, f1: number, rad: number) => {
    const [x0, y0] = pt(f0, rad);
    const [x1, y1] = pt(f1, rad);
    const large = sweep * (f1 - f0) > 180 ? 1 : 0;
    return `M${x0},${y0} A${rad},${rad} 0 ${large} 1 ${x1},${y1}`;
  };
  const col = riskHex(p);
  const [tx0, ty0] = pt(threshold, r - 9);
  const [tx1, ty1] = pt(threshold, r + 9);
  return (
    <svg width={size} height={size * 0.86} viewBox={`0 0 ${size} ${size * 0.86}`} role="img"
      aria-label={`Overall CAD probability ${Math.round(p * 100)} percent`}>
      <defs>
        <linearGradient id="gtrack" x1="0" x2="1">
          <stop offset="0" stopColor="#2a9d8f" stopOpacity="0.18" />
          <stop offset="0.5" stopColor="#e9a23b" stopOpacity="0.18" />
          <stop offset="1" stopColor="#c81d25" stopOpacity="0.18" />
        </linearGradient>
      </defs>
      <path d={arc(0, 1, r)} stroke="url(#gtrack)" strokeWidth={10} fill="none" strokeLinecap="round" />
      <path d={arc(0, Math.max(0.004, p), r)} stroke={col} strokeWidth={10} fill="none" strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 6px ${col}66)`, transition: "d 0.4s" }} />
      <path d={arc(range[0], Math.max(range[0] + 0.004, range[1]), r + 9)} stroke="#a7b5ba" strokeWidth={2}
        fill="none" strokeLinecap="round" opacity={0.7} />
      <line x1={tx0} y1={ty0} x2={tx1} y2={ty1} stroke="#e6edef" strokeWidth={2} strokeLinecap="round" />
      <text x={c} y={c + 4} textAnchor="middle" fill="#e6edef" style={{ font: "500 30px var(--mono)" }}>
        {Math.round(p * 100)}
        <tspan style={{ font: "500 14px var(--mono)" }} fill="#6c7d84">%</tspan>
      </text>
      <text x={c} y={c + 22} textAnchor="middle" fill="#6c7d84" style={{ font: "500 9px var(--mono)", letterSpacing: "0.14em" }}>
        P(CAD)
      </text>
    </svg>
  );
}
