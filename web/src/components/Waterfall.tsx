import type { Contribution, StageExplanation } from "../api";
import { fmtValue, LOWER, logit, pct, RAISE, sigmoid } from "../lib/color";
import { useTip } from "./Tip";

/**
 * SHAP waterfall on a log-odds axis labelled in probability.
 * Exact additivity: baseline + sum(contributions) = patient's calibrated log-odds.
 */
export default function Waterfall({ stage, n = 10, width = 372 }: { stage: StageExplanation; n?: number; width?: number }) {
  const show = useTip((s) => s.show);
  const hide = useTip((s) => s.hide);
  const top = stage.features.slice(0, n);
  const rest = stage.features.slice(n);
  const restSum = rest.reduce((a, f) => a + f.shap, 0);
  const rows: { label: string; sub: string; d: number; f?: Contribution }[] = top.map((f) => ({
    label: f.label,
    sub: fmtValue(f.value, f.unit, f.kind),
    d: f.shap,
    f,
  }));
  rows.push({ label: `${rest.length} other features`, sub: "combined", d: restSum });

  let cum = stage.base_logit;
  const segs = rows.map((r) => {
    const s = { ...r, a: cum, b: cum + r.d };
    cum += r.d;
    return s;
  });
  const lo = Math.min(stage.base_logit, ...segs.map((s) => Math.min(s.a, s.b)));
  const hi = Math.max(stage.base_logit, ...segs.map((s) => Math.max(s.a, s.b)));
  const pad = Math.max(0.3, (hi - lo) * 0.08);
  const x0 = Math.max(logit(0.003), lo - pad);
  const x1 = Math.min(logit(0.997), hi + pad);

  const labelW = 150;
  const deltaW = 42;
  const plotW = width - labelW - deltaW;
  const rowH = 27;
  const top0 = 22;
  const H = top0 + (segs.length + 2) * rowH + 22;
  const X = (z: number) => labelW + ((Math.min(x1, Math.max(x0, z)) - x0) / (x1 - x0)) * plotW;
  const ticks = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].filter((p) => logit(p) >= x0 && logit(p) <= x1);

  return (
    <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img" aria-label="SHAP waterfall">
      {ticks.map((p) => (
        <g key={p}>
          <line x1={X(logit(p))} x2={X(logit(p))} y1={top0 - 6} y2={H - 20} stroke="#152126" />
          <text x={X(logit(p))} y={H - 6} textAnchor="middle" fill="#6c7d84" style={{ font: "10px var(--mono)" }}>
            {Math.round(p * 100)}%
          </text>
        </g>
      ))}
      {/* baseline row */}
      <text x={labelW - 8} y={top0 + 13} textAnchor="end" fill="#a7b5ba" style={{ font: "500 11px var(--sans)" }}>
        Model baseline
      </text>
      <circle cx={X(stage.base_logit)} cy={top0 + 9} r={4} fill="#a7b5ba" />
      <text x={width - 2} y={top0 + 13} textAnchor="end" fill="#6c7d84" style={{ font: "10.5px var(--mono)" }}>
        {pct(stage.base_prob)}
      </text>
      {segs.map((s, i) => {
        const y = top0 + (i + 1) * rowH;
        const xa = X(s.a);
        const xb = X(s.b);
        const col = s.d >= 0 ? RAISE : LOWER;
        const tipRows: [string, string][] = [
          ["Value", s.sub],
          ["Contribution (log-odds)", `${s.d >= 0 ? "+" : ""}${s.d.toFixed(3)}`],
          ["Running probability", `${pct(sigmoid(s.a))} → ${pct(sigmoid(s.b))}`],
        ];
        return (
          <g key={i}
            onMouseMove={(e) => show(e.clientX, e.clientY, s.label, tipRows, s.f ? undefined : "Sum of all remaining, smaller attributions.")}
            onMouseLeave={hide}>
            <rect x={0} y={y - 2} width={width} height={rowH} fill="transparent" />
            <text x={labelW - 8} y={y + 8} textAnchor="end" fill="#e6edef" style={{ font: "11px var(--sans)" }}>
              {s.label.length > 22 ? s.label.slice(0, 21) + "…" : s.label}
            </text>
            <text x={labelW - 8} y={y + 20} textAnchor="end" fill="#6c7d84" style={{ font: "9.5px var(--mono)" }}>
              {s.sub}
            </text>
            <line x1={xa} x2={xa} y1={y - rowH + 13} y2={y + 2} stroke="#2b3c44" strokeWidth={1} />
            <rect x={Math.min(xa, xb)} y={y + 2} width={Math.max(2, Math.abs(xb - xa))} height={11} rx={3}
              fill={s.f ? col : "#46565d"} />
            <text x={width - 2} y={y + 11} textAnchor="end" fill={s.f ? "#a7b5ba" : "#6c7d84"} style={{ font: "10.5px var(--mono)" }}>
              {s.d >= 0 ? "+" : "−"}{Math.abs(s.d).toFixed(2)}
            </text>
          </g>
        );
      })}
      {/* final */}
      {(() => {
        const y = top0 + (segs.length + 1) * rowH;
        return (
          <g>
            <line x1={labelW} x2={labelW + plotW} y1={y - 3} y2={y - 3} stroke="#24353d" />
            <text x={labelW - 8} y={y + 12} textAnchor="end" fill="#e6edef" style={{ font: "600 11.5px var(--sans)" }}>
              This patient
            </text>
            <line x1={X(stage.logit)} x2={X(stage.logit)} y1={y - rowH + 13} y2={y + 6} stroke="#2b3c44" />
            <circle cx={X(stage.logit)} cy={y + 8} r={5} fill="#e9a23b" />
            <text x={width - 2} y={y + 12} textAnchor="end" fill="#e9a23b" style={{ font: "600 11.5px var(--mono)" }}>
              {pct(stage.prob)}
            </text>
          </g>
        );
      })()}
      <text x={labelW} y={11} fill="#6c7d84" style={{ font: "500 9.5px var(--mono)", letterSpacing: "0.1em" }}>
        ← LOWERS RISK
      </text>
      <text x={labelW + plotW} y={11} textAnchor="end" fill="#6c7d84" style={{ font: "500 9.5px var(--mono)", letterSpacing: "0.1em" }}>
        RAISES RISK →
      </text>
    </svg>
  );
}
