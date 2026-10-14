import { useEffect, useRef, useState } from "react";
import { useTip } from "./Tip";

export interface Series {
  name: string;
  color: string;
  points: [number, number][];
  width?: number;
  emphasis?: boolean; // drawn on top, thicker
  muted?: boolean; // recessive gray context series
  dots?: boolean;
}

interface Props {
  series: Series[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  yLabel: string;
  height?: number;
  xTicks?: number[];
  yTicks?: number[];
  xFmt?: (v: number) => string;
  yFmt?: (v: number) => string;
  diagonal?: boolean;
  markerX?: number;
  markerLabel?: string;
  directLabels?: boolean;
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(320);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

const interp = (pts: [number, number][], x: number): number | null => {
  if (!pts.length) return null;
  const s = [...pts].sort((a, b) => a[0] - b[0]);
  if (x <= s[0][0]) return s[0][1];
  if (x >= s[s.length - 1][0]) return s[s.length - 1][1];
  for (let i = 1; i < s.length; i++) {
    if (x <= s[i][0]) {
      const [x0, y0] = s[i - 1];
      const [x1, y1] = s[i];
      return x1 === x0 ? y1 : y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
  }
  return null;
};

/** Thin-line chart with hairline grid, crosshair + tooltip hover layer, optional direct labels. */
export default function LineChart(p: Props) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const show = useTip((s) => s.show);
  const hide = useTip((s) => s.hide);
  const [hx, setHx] = useState<number | null>(null);
  const H = p.height ?? 220;
  const m = { l: 40, r: p.directLabels ? 78 : 12, t: 10, b: 34 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const X = (v: number) => m.l + ((v - p.xDomain[0]) / (p.xDomain[1] - p.xDomain[0])) * iw;
  const Y = (v: number) => m.t + ih - ((v - p.yDomain[0]) / (p.yDomain[1] - p.yDomain[0])) * ih;
  const xf = p.xFmt ?? ((v: number) => v.toFixed(1));
  const yf = p.yFmt ?? ((v: number) => v.toFixed(1));
  const xt = p.xTicks ?? [0, 0.25, 0.5, 0.75, 1].map((f) => p.xDomain[0] + f * (p.xDomain[1] - p.xDomain[0]));
  const yt = p.yTicks ?? [0, 0.25, 0.5, 0.75, 1].map((f) => p.yDomain[0] + f * (p.yDomain[1] - p.yDomain[0]));
  const ordered = [...p.series].sort((a, b) => Number(!!a.emphasis) - Number(!!b.emphasis) || Number(!a.muted) - Number(!b.muted));
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(Math.max(p.yDomain[0], Math.min(p.yDomain[1], y))).toFixed(1)}`).join("");

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = (e.target as SVGRectElement).getBoundingClientRect();
    const x = p.xDomain[0] + ((e.clientX - r.left) / r.width) * (p.xDomain[1] - p.xDomain[0]);
    setHx(x);
    const rows: [string, string][] = p.series
      .filter((s) => !s.muted || p.series.length <= 6)
      .map((s) => [s.name, (() => { const v = interp(s.points, x); return v === null ? "–" : yf(v); })()]);
    show(e.clientX, e.clientY, `${p.xLabel}: ${xf(x)}`, rows);
  };

  // direct labels: end of each non-muted series, nudged apart to avoid collisions
  const labels = p.directLabels
    ? (() => {
        const ls = p.series.filter((s) => !s.muted && s.points.length).map((s) => {
          const last = [...s.points].sort((a, b) => a[0] - b[0]).at(-1)!;
          return { name: s.name, color: s.color, y: Y(last[1]) };
        }).sort((a, b) => a.y - b.y);
        for (let i = 1; i < ls.length; i++) if (ls[i].y - ls[i - 1].y < 12) ls[i].y = ls[i - 1].y + 12;
        return ls;
      })()
    : [];

  return (
    <div ref={ref} style={{ width: "100%" }}>
      <svg width={W} height={H} role="img" aria-label={`${p.yLabel} versus ${p.xLabel}`}>
        {yt.map((v) => (
          <g key={`y${v}`}>
            <line x1={m.l} x2={m.l + iw} y1={Y(v)} y2={Y(v)} stroke="#132026" />
            <text x={m.l - 6} y={Y(v) + 3} textAnchor="end" fill="#6c7d84" style={{ font: "10px var(--mono)" }}>{yf(v)}</text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={`x${v}`} x={X(v)} y={m.t + ih + 14} textAnchor="middle" fill="#6c7d84" style={{ font: "10px var(--mono)" }}>{xf(v)}</text>
        ))}
        <line x1={m.l} x2={m.l + iw} y1={m.t + ih} y2={m.t + ih} stroke="#24353d" />
        <text x={m.l + iw / 2} y={H - 4} textAnchor="middle" fill="#a7b5ba" style={{ font: "10.5px var(--sans)" }}>{p.xLabel}</text>
        <text x={10} y={m.t + ih / 2} textAnchor="middle" fill="#a7b5ba" transform={`rotate(-90 10 ${m.t + ih / 2})`} style={{ font: "10.5px var(--sans)" }}>{p.yLabel}</text>
        {p.diagonal && <line x1={X(p.xDomain[0])} y1={Y(p.yDomain[0])} x2={X(p.xDomain[1])} y2={Y(p.yDomain[1])} stroke="#2b3c44" strokeWidth={1} />}
        {p.markerX !== undefined && (
          <g>
            <line x1={X(p.markerX)} x2={X(p.markerX)} y1={m.t} y2={m.t + ih} stroke="#e6edef" strokeOpacity={0.5} />
            {p.markerLabel && <text x={X(p.markerX) + 4} y={m.t + 9} fill="#a7b5ba" style={{ font: "10px var(--mono)" }}>{p.markerLabel}</text>}
          </g>
        )}
        {ordered.map((s) => (
          <g key={s.name}>
            <path d={path(s.points)} fill="none" stroke={s.muted ? "#3a4a52" : s.color}
              strokeWidth={s.width ?? (s.emphasis ? 2.4 : 1.6)} strokeLinejoin="round" strokeLinecap="round" />
            {s.dots && s.points.map(([x, y], i) => (
              <circle key={i} cx={X(x)} cy={Y(y)} r={4} fill={s.color} stroke="#0b1114" strokeWidth={2} />
            ))}
          </g>
        ))}
        {labels.map((l) => (
          <text key={l.name} x={m.l + iw + 6} y={l.y + 3} fill="#a7b5ba" style={{ font: "500 10.5px var(--mono)" }}>
            <tspan fill={l.color}>●</tspan> {l.name}
          </text>
        ))}
        {hx !== null && hx >= p.xDomain[0] && hx <= p.xDomain[1] && (
          <line x1={X(hx)} x2={X(hx)} y1={m.t} y2={m.t + ih} stroke="#a7b5ba" strokeOpacity={0.35} />
        )}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove}
          onMouseLeave={() => { setHx(null); hide(); }} />
      </svg>
    </div>
  );
}
