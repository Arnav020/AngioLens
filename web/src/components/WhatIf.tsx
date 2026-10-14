import { useEffect, useMemo, useState } from "react";
import { api, TARGETS, type Explanation, type Sensitivity } from "../api";
import { fmtValue, pct } from "../lib/color";
import { useStore } from "../store";
import LineChart from "./LineChart";

// categorical identity colours (validated dark-mode palette, fixed order)
export const SERIES_COLORS: Record<string, string> = { CAD: "#3987e5", LAD: "#d95926", LCX: "#199e70", RCA: "#c98500" };

type Scenario = { label: string; changes: Record<string, number>; note: string; kind: "probe" | "risk" };
const SCENARIOS: Scenario[] = [
  // diagnostic probes: which findings drive this estimate?
  { label: "Chest pain atypical, not typical", changes: { typical_chest_pain: 0, atypical: 1 }, note: "symptom", kind: "probe" },
  { label: "No ST depression", changes: { st_depression: 0 }, note: "ECG", kind: "probe" },
  { label: "No T-wave inversion", changes: { t_inversion: 0 }, note: "ECG", kind: "probe" },
  { label: "No wall-motion abnormality", changes: { region_rwma: 0 }, note: "echo", kind: "probe" },
  { label: "Ejection fraction 60%", changes: { ef_tte: 60 }, note: "echo", kind: "probe" },
  { label: "No diabetes", changes: { dm: 0 }, note: "history", kind: "probe" },
  // risk-factor targets: shown with an explicit confounding warning
  { label: "Stops smoking", changes: { current_smoker: 0 }, note: "lifestyle", kind: "risk" },
  { label: "Systolic BP 120", changes: { bp: 120 }, note: "therapy target", kind: "risk" },
  { label: "LDL 70 mg/dL", changes: { ldl: 70 }, note: "therapy target", kind: "risk" },
  { label: "Fasting glucose 95", changes: { fbs: 95 }, note: "glycaemic control", kind: "risk" },
  { label: "BMI 24", changes: { bmi: 24, obesity: 0 }, note: "weight loss", kind: "risk" },
];

export default function WhatIf() {
  const values = useStore((s) => s.values);
  const result = useStore((s) => s.result);
  const schema = useStore((s) => s.schema);
  const setValues = useStore((s) => s.setValues);
  const source = useStore((s) => s.source);
  const [feature, setFeature] = useState("age");
  const [sens, setSens] = useState<Sensitivity | null>(null);
  const [preview, setPreview] = useState<Record<string, Explanation>>({});

  const applicable = useMemo(
    () => SCENARIOS.filter((s) => Object.entries(s.changes).some(([k, v]) => values[k] !== undefined && values[k] !== v)),
    [values],
  );

  useEffect(() => {
    if (!Object.keys(values).length) return;
    let alive = true;
    api.sensitivity(values, feature).then((r) => alive && setSens(r)).catch(() => {});
    return () => {
      alive = false;
    };
  }, [values, feature]);

  useEffect(() => {
    if (!Object.keys(values).length) return;
    let alive = true;
    Promise.all(applicable.map((s) => api.predict({ ...values, ...s.changes }).then((r) => [s.label, r] as const)))
      .then((rs) => alive && setPreview(Object.fromEntries(rs)))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [values, applicable]);

  const spec = schema?.features.find((f) => f.id === feature);
  const basedOn = source.kind === "cohort" ? `patient #${source.id}` : source.kind === "preset" ? `${source.name}-risk preset` : source.basedOn;

  return (
    <div className="scroll">
      {(["probe", "risk"] as const).map((kind) => (
        <div className="section" key={kind}>
          <div className="section-head">
            <h3>{kind === "probe" ? "Counterfactual probes" : "Risk-factor targets"}</h3>
            <span className="eyebrow">{kind === "probe" ? "what drives this?" : "model sensitivity"}</span>
          </div>
          <p className="muted small" style={{ margin: "0 0 8px" }}>
            {kind === "probe"
              ? "Change one finding, keep everything else fixed, and see how the model's estimate moves. Click a row to apply it."
              : "These show what the model learned, not what treatment would do. In this cohort LDL is no higher in patients with CAD (many are already on statins), so a lower LDL can co-occur with more disease (confounding by treatment)."}
          </p>
          {result && applicable.filter((sc) => sc.kind === kind).map((sc) => {
            const r = preview[sc.label];
            const d = r ? r.targets.CAD.prob - result.targets.CAD.prob : 0;
            const counter = kind === "risk" && r && d > 0.005;
            return (
              <div key={sc.label} className="sim-row" style={{ gridTemplateColumns: "1fr auto auto" }}
                onClick={() => setValues(sc.changes, basedOn)} title="Apply this change">
                <span>
                  <span style={{ color: "var(--ink)" }}>{sc.label}</span>{" "}
                  <span className="muted small">· {sc.note}</span>
                  {counter && (
                    <span className="badge indeterminate" style={{ marginLeft: 6, padding: "0 5px" }}
                      title="Risk rises when this factor is normalised: an association learned from treated patients, not a causal effect.">
                      counter-intuitive
                    </span>
                  )}
                </span>
                <span className="mono small muted">{r ? `${pct(result.targets.CAD.prob)} →` : ""}</span>
                <span className="mono" style={{ minWidth: 92, textAlign: "right" }}>
                  {r ? (
                    <>
                      {pct(r.targets.CAD.prob)}{" "}
                      <span style={{ color: d < -0.005 ? "#7fd8c9" : d > 0.005 ? "#f6c27a" : "var(--ink-3)", fontSize: 11 }}>
                        {d >= 0 ? "+" : "−"}{Math.abs(d * 100).toFixed(1)}pp
                      </span>
                    </>
                  ) : "…"}
                </span>
              </div>
            );
          })}
          {result && applicable.filter((sc) => sc.kind === kind).length === 0 && (
            <div className="muted small">Nothing to change: already at these values.</div>
          )}
        </div>
      ))}

      <div className="section">
        <div className="section-head">
          <h3>Individual sensitivity curve</h3>
          <select className="search" style={{ width: 180 }} value={feature} onChange={(e) => setFeature(e.target.value)}>
            {schema?.features.filter((f) => f.kind !== "binary").map((f) => (
              <option key={f.id} value={f.id}>{f.label}</option>
            ))}
            <optgroup label="Binary findings">
              {schema?.features.filter((f) => f.kind === "binary").map((f) => (
                <option key={f.id} value={f.id}>{f.label}</option>
              ))}
            </optgroup>
          </select>
        </div>
        {sens && spec && (
          <>
            <LineChart
              height={230}
              series={TARGETS.map((t) => ({
                name: t,
                color: SERIES_COLORS[t],
                emphasis: t === "CAD",
                dots: spec.kind !== "numeric",
                points: sens.grid.map((x, i) => [x, sens.curves[t][i]] as [number, number]),
              }))}
              xDomain={[sens.grid[0], sens.grid[sens.grid.length - 1] === sens.grid[0] ? sens.grid[0] + 1 : sens.grid[sens.grid.length - 1]]}
              yDomain={[0, 1]}
              yTicks={[0, 0.25, 0.5, 0.75, 1]}
              yFmt={(v) => `${Math.round(v * 100)}%`}
              xFmt={(v) => (spec.kind === "binary" ? (v >= 0.5 ? "yes" : "no") : fmtValue(Math.round(v * 10) / 10))}
              xTicks={spec.kind === "numeric" ? undefined : sens.grid}
              xLabel={`${spec.label}${spec.unit ? ` (${spec.unit})` : ""}`}
              yLabel="Probability"
              markerX={sens.current}
              markerLabel="now"
              directLabels
            />
            <div className="legend">
              {TARGETS.map((t) => (
                <span key={t}><i style={{ background: SERIES_COLORS[t], height: t === "CAD" ? 3 : 2 }} />{t === "CAD" ? "Overall CAD" : t}</span>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
