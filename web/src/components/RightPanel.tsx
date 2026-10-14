import { VESSELS, type Conformal, type Target } from "../api";
import { CONFORMAL_HINT, CONFORMAL_LABEL, pct, riskHex } from "../lib/color";
import { useStore, type Panel } from "../store";
import Breakdown from "./Breakdown";
import BullsEye from "./BullsEye";
import Copilot from "./Copilot";
import Gauge from "./Gauge";
import Waterfall from "./Waterfall";
import WhatIf from "./WhatIf";

const FULL: Record<string, string> = {
  LAD: "Left anterior descending",
  LCX: "Left circumflex",
  RCA: "Right coronary artery",
};

function Badge({ c }: { c: Conformal }) {
  return (
    <span className={`badge ${c}`} title={CONFORMAL_HINT[c]}>
      {c === "indeterminate" ? "◐" : c === "positive" ? "●" : c === "negative" ? "○" : "◇"} {CONFORMAL_LABEL[c]}
    </span>
  );
}

function Truth({ t }: { t: Target }) {
  const r = useStore((s) => s.result);
  if (!r?.truth) return null;
  const truth = r.truth[t];
  const pred = r.targets[t].flag ? 1 : 0;
  return (
    <span className={`truth ${truth === pred ? "ok" : "bad"}`} title="Invasive angiography result for this cohort patient vs the model's flag at its operating threshold">
      {truth === pred ? "✓" : "✗"} angio: {truth ? "stenotic" : "normal"}
    </span>
  );
}

function RiskTab() {
  const r = useStore((s) => s.result);
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  const hover = useStore((s) => s.hover);
  const loadCohort = useStore((s) => s.loadCohort);
  const setPanel = useStore((s) => s.setPanel);
  if (!r) return <div className="section"><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></div>;
  const cad = r.targets.CAD;
  const comp = r.stages.CAD.components;
  return (
    <div className="scroll">
      <div className="section">
        <div className="cad-hero">
          <Gauge p={cad.prob} range={cad.model_range} threshold={cad.threshold} />
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            <span className="eyebrow">Overall CAD</span>
            <div style={{ font: "italic 400 24px/1.05 var(--serif)" }}>
              {cad.conformal === "positive" ? "Significant CAD likely" : cad.conformal === "negative" ? "Significant CAD unlikely" : "Uncertain: imaging decides"}
            </div>
            <Badge c={cad.conformal} />
            <span className="decomp" title="Calibrated output of each ensemble member; disagreement signals model uncertainty">
              LR {pct(comp.lr)} · GBM {pct(comp.gbm)} · flag ≥ {pct(cad.threshold)}
            </span>
            <Truth t="CAD" />
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-head">
          <h3>Per-vessel stenosis ≥ 50%</h3>
          <span className="eyebrow" title="Hierarchical model: P(vessel) = P(CAD) × P(vessel | CAD)">P(CAD) × P(v | CAD)</span>
        </div>
        {VESSELS.map((v) => {
          const t = r.targets[v];
          const col = riskHex(t.prob);
          return (
            <div key={v} className={`vcard ${selected === v ? "on" : ""}`} style={{ ["--c" as string]: col }}
              onClick={() => select(selected === v ? null : v)} onMouseEnter={() => hover(v)} onMouseLeave={() => hover(null)}>
              <span className="name" style={{ color: col }}>{v}</span>
              <span>
                <div style={{ fontSize: 12 }}>{FULL[v]}</div>
                <div className="full">{t.territory?.split("(")[0]}</div>
              </span>
              <span className="p" style={{ color: col }}>{pct(t.prob)}</span>
              <div className="track" title={`model range ${pct(t.model_range[0])}–${pct(t.model_range[1])}, flag threshold ${pct(t.threshold)}`}>
                <div className="fill" style={{ width: `${t.prob * 100}%`, background: col }} />
                <div className="range" style={{ left: `${t.model_range[0] * 100}%`, width: `${Math.max(0.5, (t.model_range[1] - t.model_range[0]) * 100)}%` }} />
                <div className="thr" style={{ left: `${t.threshold * 100}%` }} />
              </div>
              <div className="foot">
                <Badge c={t.conformal} />
                <span className="decomp">{pct(cad.prob)} × {pct(t.cond_prob ?? 0)}</span>
              </div>
              {r.truth && <div className="foot" style={{ marginTop: 2 }}><Truth t={v} /><span /></div>}
            </div>
          );
        })}
      </div>

      <div className="section">
        <div className="section-head">
          <h3>Myocardium at risk</h3>
          <span className="eyebrow">AHA 17-segment</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "172px 1fr", gap: 14, alignItems: "center" }}>
          <BullsEye />
          <div className="small muted" style={{ lineHeight: 1.5 }}>
            Each left-ventricular segment takes the probability of its usual supplying artery (Cerqueira 2002;
            right-dominant). Hover a segment to light its artery and territory on the 3D heart.
            <div style={{ marginTop: 8, display: "grid", gap: 3 }}>
              <span><b style={{ color: "var(--ink)" }}>LAD</b> 1, 2, 7, 8, 13, 14, 17</span>
              <span><b style={{ color: "var(--ink)" }}>RCA</b> 3, 4, 9, 10, 15</span>
              <span><b style={{ color: "var(--ink)" }}>LCX</b> 5, 6, 11, 12, 16</span>
            </div>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-head">
          <h3>Most similar cohort patients</h3>
          <span className="eyebrow">their angiography</span>
        </div>
        {r.similar.map((s) => (
          <div key={s.id} className="sim-row" onClick={() => loadCohort(s.id)}>
            <span className="pid mono">#{s.id}</span>
            <span>
              <div className="small">{s.age} y · {s.sex}</div>
              <div className="simbar"><div style={{ width: `${s.similarity * 100}%` }} /></div>
            </span>
            <span className="dots" title="angiography: filled = stenotic">
              {(["CAD", "LAD", "LCX", "RCA"] as Target[]).map((t) => (
                <span key={t} className="mono small" style={{ color: s.truth[t] ? "#ff9c8a" : "var(--ink-4)", width: 30, textAlign: "center" }}>
                  {t === "CAD" ? (s.truth[t] ? "CAD" : "nl") : s.truth[t] ? t : "·"}
                </span>
              ))}
            </span>
          </div>
        ))}
        <button className="btn ghost" style={{ marginTop: 10, width: "100%" }} onClick={() => setPanel("explain")}>
          Why these numbers? →
        </button>
      </div>
    </div>
  );
}

function ExplainTab() {
  const r = useStore((s) => s.result);
  const stage = useStore((s) => s.stage);
  const setStage = useStore((s) => s.setStage);
  const select = useStore((s) => s.select);
  if (!r) return null;
  const st = r.stages[stage] ?? r.stages.CAD;
  const isCad = stage === "CAD";
  return (
    <div className="scroll">
      <div className="section">
        <div className="section-head">
          <h3>{isCad ? "Why this overall CAD estimate" : `Why ${stage.split("|")[0]}, given CAD`}</h3>
          <span className="eyebrow">exact SHAP</span>
        </div>
        <div className="seg" style={{ width: "100%", marginBottom: 12 }}>
          {["CAD", ...VESSELS.map((v) => `${v}|CAD`)].map((s) => (
            <button key={s} style={{ flex: 1 }} className={stage === s ? "on" : ""}
              onClick={() => { setStage(s); select(s === "CAD" ? null : (s.split("|")[0] as typeof VESSELS[number])); }}>
              {s === "CAD" ? "Is there CAD?" : `Where: ${s.split("|")[0]}`}
            </button>
          ))}
        </div>
        <p className="muted small" style={{ margin: "0 0 10px" }}>
          {isCad
            ? "Stage 1 of the hierarchical model. Contributions are exact and additive in log-odds; the axis is labelled in probability."
            : `Stage 2: among patients with CAD, which factors point to the ${stage.split("|")[0]}. Final P(${stage.split("|")[0]}) = P(CAD) × this stage's probability.`}
        </p>
        <Waterfall stage={st} />
      </div>
      <div className="section">
        <div className="section-head">
          <h3>Physiological breakdown</h3>
          <span className="eyebrow">value · reference · effect</span>
        </div>
        <Breakdown stage={st} />
      </div>
    </div>
  );
}

export default function RightPanel() {
  const panel = useStore((s) => s.panel);
  const setPanel = useStore((s) => s.setPanel);
  const tabs: [Panel, string][] = [["risk", "Risk"], ["explain", "Explain"], ["whatif", "What-if"], ["copilot", "Copilot"]];
  return (
    <>
      <div className="tabs" role="tablist">
        {tabs.map(([k, l]) => (
          <button key={k} role="tab" aria-selected={panel === k} className={panel === k ? "on" : ""} onClick={() => setPanel(k)}>
            {l}
          </button>
        ))}
      </div>
      {panel === "risk" && <RiskTab />}
      {panel === "explain" && <ExplainTab />}
      {panel === "whatif" && <WhatIf />}
      {panel === "copilot" && <Copilot />}
    </>
  );
}
