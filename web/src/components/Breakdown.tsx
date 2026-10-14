import { useMemo, useState } from "react";
import type { StageExplanation } from "../api";
import { fmtValue, LOWER, RAISE } from "../lib/color";
import { useStore } from "../store";
import { useTip } from "./Tip";

/** Physiological breakdown: every measurement with its reference range and its exact contribution. */
export default function Breakdown({ stage }: { stage: StageExplanation }) {
  const schema = useStore((s) => s.schema);
  const show = useTip((s) => s.show);
  const hide = useTip((s) => s.hide);
  const [group, setGroup] = useState<string>("all");
  const [sort, setSort] = useState<"impact" | "group">("impact");
  const spec = useMemo(() => Object.fromEntries((schema?.features ?? []).map((f) => [f.id, f])), [schema]);
  const maxAbs = Math.max(...stage.features.map((f) => Math.abs(f.shap)), 1e-6);
  let rows = stage.features.filter((f) => group === "all" || f.group === group);
  if (sort === "group") rows = [...rows].sort((a, b) => a.group.localeCompare(b.group));

  const groupMax = Math.max(...stage.groups.map((g) => Math.abs(g.shap)), 1e-6);
  return (
    <>
      <div className="eyebrow" style={{ marginBottom: 8 }}>Contribution by clinical domain</div>
      <div style={{ display: "grid", gap: 5, marginBottom: 16 }}>
        {stage.groups.map((g) => (
          <div key={g.id} style={{ display: "grid", gridTemplateColumns: "150px 1fr 48px", alignItems: "center", gap: 8, cursor: "pointer" }}
            onClick={() => setGroup(group === g.id ? "all" : g.id)}>
            <span className="small" style={{ color: group === g.id ? "var(--amber)" : "var(--ink-2)" }}>{g.label}</span>
            <div style={{ position: "relative", height: 8 }}>
              <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "#24353d" }} />
              <div style={{
                position: "absolute", top: 0, bottom: 0, borderRadius: 3,
                background: g.shap >= 0 ? RAISE : LOWER,
                left: g.shap >= 0 ? "50%" : `${50 - (Math.abs(g.shap) / groupMax) * 50}%`,
                width: `${(Math.abs(g.shap) / groupMax) * 50}%`,
              }} />
            </div>
            <span className="mono small" style={{ textAlign: "right", color: "var(--ink-2)" }}>
              {g.shap >= 0 ? "+" : "−"}{Math.abs(g.shap).toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      <div className="section-head" style={{ marginBottom: 6 }}>
        <span className="eyebrow">{group === "all" ? "All 55 inputs" : schema?.groups[group]}</span>
        <div className="seg">
          <button className={sort === "impact" ? "on" : ""} onClick={() => setSort("impact")}>impact</button>
          <button className={sort === "group" ? "on" : ""} onClick={() => setSort("group")}>domain</button>
        </div>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Measurement</th>
            <th className="num">Value</th>
            <th>Reference</th>
            <th className="num">Effect</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => {
            const s = spec[f.id];
            const lo = s?.normal ? s.normal[0] : null;
            const hi = s?.normal ? s.normal[1] : null;
            // scale the reference bar to [min(lo, value), max(hi, value)] with margin
            let bar = null;
            if (s && s.kind === "numeric" && lo !== null && hi !== null) {
              const a = Math.min(lo, f.value), b = Math.max(hi, f.value);
              const m = (b - a) * 0.15 || 1;
              const L = a - m, R = b + m;
              const px = (v: number) => `${((v - L) / (R - L)) * 100}%`;
              const mkCol = f.status === "normal" ? "#7fd8c9" : "#f6c27a";
              bar = (
                <div className="refbar" title={`reference ${lo}–${hi} ${f.unit}`}>
                  <div className="band" style={{ left: px(lo), width: `calc(${px(hi)} - ${px(lo)})` }} />
                  <div className="mk" style={{ left: px(f.value), background: mkCol }} />
                </div>
              );
            }
            const w = (Math.abs(f.shap) / maxAbs) * 34;
            return (
              <tr key={f.id}
                onMouseMove={(e) => show(e.clientX, e.clientY, f.label, [
                  ["Value", fmtValue(f.value, f.unit, f.kind)],
                  ...(lo !== null ? [["Reference", `${lo}–${hi} ${f.unit}`] as [string, string]] : []),
                  ["Status", f.status],
                  ["SHAP (log-odds)", `${f.shap >= 0 ? "+" : ""}${f.shap.toFixed(3)}`],
                ], s?.description || undefined)}
                onMouseLeave={hide}>
                <td>{f.label}</td>
                <td className="num">{fmtValue(f.value, "", f.kind)} <span className="muted">{f.kind === "numeric" ? f.unit : ""}</span></td>
                <td>{bar ?? <span className={`status ${f.status}`}>{f.status === "info" ? "–" : f.status}</span>}</td>
                <td className="num">
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <span style={{ display: "inline-block", width: 36, height: 6, position: "relative" }}>
                      <span style={{
                        position: "absolute", top: 0, bottom: 0, borderRadius: 2,
                        background: f.shap >= 0 ? RAISE : LOWER,
                        ...(f.shap >= 0 ? { left: 0 } : { right: 0 }), width: Math.max(1, w),
                      }} />
                    </span>
                    <span style={{ minWidth: 38 }}>{f.shap >= 0 ? "+" : "−"}{Math.abs(f.shap).toFixed(2)}</span>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
