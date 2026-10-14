import { useEffect, useState } from "react";
import { api, TARGETS, type ModelCard as MC, type Target } from "../api";
import { LOWER, pct } from "../lib/color";
import LineChart from "./LineChart";
import { useTip } from "./Tip";
import { SERIES_COLORS } from "./WhatIf";

const NAME: Record<Target, string> = { CAD: "Overall CAD", LAD: "LAD stenosis", LCX: "LCX stenosis", RCA: "RCA stenosis" };
const MODEL_SHORT: Record<string, string> = {
  angiolens: "AngioLens (ours)",
  direct_blend: "Direct LR+GBM blend",
  lr: "L1 logistic regression",
  gbm: "LightGBM",
  rf: "Random forest",
};
const f3 = (v: number) => v.toFixed(3);
const f2 = (v: number) => v.toFixed(2);

function Kpi({ mc, t }: { mc: MC; t: Target }) {
  const s = mc.targets[t].summary.angiolens;
  const tt = mc.targets[t].tuned_threshold;
  return (
    <div className="kpi">
      <div className="eyebrow">{NAME[t]}</div>
      <div className="big">{f3(s.roc_auc.mean)}</div>
      <div className="ci">ROC-AUC · 95% CI {s.roc_auc.ci95 ? `${f3(s.roc_auc.ci95[0])}–${f3(s.roc_auc.ci95[1])}` : "–"}</div>
      <div className="row"><span>Brier</span><span>{f3(s.brier.mean)}</span></div>
      <div className="row"><span>PR-AUC</span><span>{f3(s.pr_auc.mean)}</span></div>
      <div className="row"><span>Precision* · Recall*</span><span>{pct(tt.precision.mean)} · {pct(tt.recall.mean)}</span></div>
      <div className="row"><span>Specificity*</span><span>{pct(tt.specificity.mean)}</span></div>
      <div className="row"><span>Accuracy* · F1*</span><span>{pct(tt.accuracy.mean)} · {f2(tt.f1.mean)}</span></div>
      <div className="row"><span>Prevalence</span><span>{pct(mc.targets[t].prevalence)}</span></div>
    </div>
  );
}

function Bars({ items, max, fmt, color = "#a7b5ba", highlight }: {
  items: { label: string; value: number; sub?: string }[]; max: number; fmt: (v: number) => string; color?: string; highlight?: string;
}) {
  const show = useTip((s) => s.show);
  const hide = useTip((s) => s.hide);
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {items.map((it) => (
        <div key={it.label} style={{ display: "grid", gridTemplateColumns: "170px 1fr 58px", gap: 10, alignItems: "center" }}
          onMouseMove={(e) => show(e.clientX, e.clientY, it.label, [["Value", fmt(it.value)], ...(it.sub ? [["", it.sub] as [string, string]] : [])])}
          onMouseLeave={hide}>
          <span className="small" style={{ color: it.label === highlight ? "var(--amber)" : "var(--ink-2)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{it.label}</span>
          <div style={{ height: 8, background: "#111b20", borderRadius: 4 }}>
            <div style={{ width: `${Math.min(100, (it.value / max) * 100)}%`, height: "100%", borderRadius: 4, background: it.label === highlight ? "#e9a23b" : color }} />
          </div>
          <span className="mono small" style={{ textAlign: "right" }}>{fmt(it.value)}</span>
        </div>
      ))}
    </div>
  );
}

export default function ModelCard() {
  const [mc, setMc] = useState<MC | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.modelCard().then(setMc).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <div className="err">{err}</div>;
  if (!mc) return <div className="skeleton" />;
  const models = Object.keys(mc.models);
  const hasCmp = models.length > 1;
  const m = mc.meta;

  return (
    <div style={{ maxWidth: 1320, margin: "0 auto" }}>
      <div className="eyebrow">
        Validation · {m.repeats}×{m.outer_folds} repeated nested cross-validation · inner {m.inner_folds}-fold · n = {mc.dataset.n_patients}
      </div>
      <h2>How good is AngioLens, honestly?</h2>
      <p className="lede">
        Every number below comes from patients the model had not seen. Hyper-parameters, calibration, operating
        thresholds and conformal sets are all chosen inside each training split, and competing models are scored on
        identical splits. Vessel-level stenosis is genuinely hard to infer from routine clinical data: LCX and RCA
        reach moderate discrimination, and the system says so with conformal "indeterminate" flags instead of
        hiding it. <span className="muted">* metrics at the operating threshold chosen by Youden's J in the inner CV.</span>
      </p>

      <div className="kpis">{TARGETS.map((t) => <Kpi key={t} mc={mc} t={t} />)}</div>

      {hasCmp && (
        <div className="card" style={{ marginBottom: 14 }}>
          <h4>Model comparison (identical outer splits)</h4>
          <div className="cap">Mean ROC-AUC (Brier score) over {m.repeats} repeats. Best per target highlighted. Lower Brier is better.</div>
          <table className="table cmp">
            <thead>
              <tr><th>Model</th>{TARGETS.map((t) => <th key={t} className="num">{NAME[t]}</th>)}<th className="num">Coherent output*</th></tr>
            </thead>
            <tbody>
              {models.map((k) => (
                <tr key={k}>
                  <td style={{ color: k === "angiolens" ? "var(--amber)" : undefined }}>{MODEL_SHORT[k] ?? k}</td>
                  {TARGETS.map((t) => {
                    const best = Math.max(...models.map((mm) => mc.targets[t].summary[mm].roc_auc.mean));
                    const v = mc.targets[t].summary[k];
                    return (
                      <td key={t} className={`num ${Math.abs(v.roc_auc.mean - best) < 5e-4 ? "best" : ""}`}>
                        {f3(v.roc_auc.mean)} <span className="muted">({f3(v.brier.mean)})</span>
                      </td>
                    );
                  })}
                  <td className="num">{mc.coherence_violation_rate ? pct(1 - mc.coherence_violation_rate[k], 1) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="cap" style={{ marginTop: 8, marginBottom: 0 }}>
            * share of patient predictions where every vessel probability ≤ the overall CAD probability (CAD is defined as
            any vessel ≥ 50%). Only the hierarchical model is coherent by construction.
          </div>
        </div>
      )}

      <div className="grid2" style={{ marginBottom: 14 }}>
        {TARGETS.map((t) => (
          <div className="card" key={t}>
            <h4>ROC · {NAME[t]}</h4>
            <div className="cap">Out-of-fold predictions (repeat 1). AngioLens emphasised; baselines in gray.</div>
            <LineChart
              height={230}
              series={models.map((k) => ({
                name: MODEL_SHORT[k] ?? k,
                color: k === "angiolens" ? "#e9a23b" : "#3a4a52",
                emphasis: k === "angiolens",
                muted: k !== "angiolens",
                points: mc.targets[t].curves[k]?.roc ?? [],
              }))}
              xDomain={[0, 1]} yDomain={[0, 1]} diagonal
              xLabel="False-positive rate" yLabel="True-positive rate"
              xFmt={f2} yFmt={f2}
            />
            <div className="legend"><span><i style={{ background: "#e9a23b", height: 3 }} />AngioLens · AUC {f3(mc.targets[t].summary.angiolens.roc_auc.mean)}</span>{hasCmp && <span><i style={{ background: "#3a4a52" }} />other models</span>}</div>
          </div>
        ))}
      </div>

      <div className="grid2" style={{ marginBottom: 14 }}>
        <div className="card">
          <h4>Calibration (reliability diagram)</h4>
          <div className="cap">Predicted probability vs observed frequency in 10 bins. On the diagonal = a 70% means 70%.
            Calibration slope: {TARGETS.map((t) => `${t} ${f2(mc.targets[t].calibration.slope)}`).join(" · ")}.</div>
          <LineChart
            height={260}
            series={TARGETS.map((t) => ({
              name: t, color: SERIES_COLORS[t], dots: true, width: 1.4,
              points: (mc.targets[t].curves.angiolens.reliability ?? []).map((b) => [b.p_mean, b.y_rate] as [number, number]),
            }))}
            xDomain={[0, 1]} yDomain={[0, 1]} diagonal directLabels
            xLabel="Predicted probability" yLabel="Observed frequency" xFmt={f2} yFmt={f2}
          />
          <div className="legend">{TARGETS.map((t) => <span key={t}><i style={{ background: SERIES_COLORS[t] }} />{t}</span>)}</div>
        </div>
        <div className="card">
          <h4>Decision-curve analysis · overall CAD</h4>
          <div className="cap">Net benefit of acting on the model across referral thresholds, versus "refer everyone" and
            "refer no one" (Vickers &amp; Elkin, 2006). Above both lines means using the model adds clinical value.</div>
          <LineChart
            height={260}
            series={[
              { name: "AngioLens", color: "#e9a23b", emphasis: true, points: (mc.targets.CAD.curves.angiolens.dca ?? []).map((d) => [d.threshold, d.model]) },
              { name: "Refer all", color: LOWER, points: (mc.targets.CAD.curves.angiolens.dca ?? []).map((d) => [d.threshold, d.treat_all]) },
              { name: "Refer none", color: "#6c7d84", points: (mc.targets.CAD.curves.angiolens.dca ?? []).map((d) => [d.threshold, 0]) },
            ]}
            xDomain={[0.05, 0.95]} xTicks={[0.05, 0.25, 0.5, 0.75, 0.95]} yDomain={[-0.1, 0.75]} yTicks={[-0.1, 0, 0.25, 0.5, 0.75]} directLabels
            xLabel="Threshold probability" yLabel="Net benefit" xFmt={f2} yFmt={f2}
          />
          <div className="legend"><span><i style={{ background: "#e9a23b", height: 3 }} />AngioLens</span><span><i style={{ background: LOWER }} />Refer all</span><span><i style={{ background: "#6c7d84" }} />Refer none</span></div>
        </div>
      </div>

      <div className="grid3" style={{ marginBottom: 14 }}>
        <div className="card">
          <h4>Conformal prediction: does "90% confident" hold?</h4>
          <div className="cap">Empirical coverage of the 90% prediction sets on outer test folds, and how often the model abstains.</div>
          <table className="table">
            <thead><tr><th>Target</th><th className="num">Coverage</th><th className="num">Indeterminate</th><th className="num">Acc. when decided</th></tr></thead>
            <tbody>
              {TARGETS.map((t) => {
                const c = mc.targets[t].conformal;
                return (
                  <tr key={t}>
                    <td>{t}</td>
                    <td className="num">{pct(c.empirical_coverage, 1)}</td>
                    <td className="num">{pct(c.indeterminate_rate, 0)}</td>
                    <td className="num">{c.accuracy_when_decided !== null ? pct(c.accuracy_when_decided, 1) : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="cap" style={{ marginTop: 8, marginBottom: 0 }}>Target coverage 90%. High indeterminate rates for LCX/RCA are the honest
            signal that routine data cannot localise those lesions reliably.</div>
        </div>
        <div className="card">
          <h4>What drives "is there CAD?"</h4>
          <div className="cap">Mean |SHAP| over the cohort, deployed model, stage 1.</div>
          <Bars items={mc.importance.CAD.features.slice(0, 10).map((f) => ({ label: f.label, value: f.mean_abs_shap }))}
            max={mc.importance.CAD.features[0].mean_abs_shap} fmt={f2} color="#5d7480" />
        </div>
        <div className="card">
          <h4>What localises the lesion?</h4>
          <div className="cap">Top factors of stage 2, P(vessel | CAD), per artery.</div>
          {(["LAD", "LCX", "RCA"] as const).map((v) => (
            <div key={v} style={{ marginBottom: 10 }}>
              <div className="eyebrow" style={{ margin: "4px 0 6px" }}>{v}</div>
              <Bars items={mc.importance[`${v}|CAD`].features.slice(0, 4).map((f) => ({ label: f.label, value: f.mean_abs_shap }))}
                max={mc.importance[`${v}|CAD`].features[0].mean_abs_shap} fmt={f2} color="#5d7480" />
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginBottom: 14 }}>
        <h4>Method in brief</h4>
        <div className="method" style={{ marginTop: 10 }}>
          <p><b>Data.</b> Z-Alizadeh Sani extension (UCI #411): {mc.dataset.n_patients} patients, {mc.dataset.n_model_features} model inputs
            after encoding; no missing values. "Exertional CP" is dropped (constant). LAD, LCX, RCA and Cath are
            targets only and never inputs. {mc.dataset.label_inconsistencies.length} record has Cath = Normal although a vessel is stenotic; it is kept and reported.</p>
          <p><b>Hierarchical, coherent targets.</b> CAD is "any vessel ≥ 50%", so we model P(CAD) and P(vessel | CAD) and
            multiply. Vessel probabilities can never exceed the CAD probability, which direct per-vessel models violate for about 1 in 5 patients.</p>
          <p><b>Exactly explainable ensemble.</b> Each stage blends an L1 logistic regression and a shallow LightGBM in
            log-odds, then applies Platt scaling. All three steps are affine in log-odds, so the SHAP values (linear + TreeSHAP)
            add up exactly to the calibrated prediction.</p>
          <p><b>Honest uncertainty.</b> Mondrian conformal prediction (α = 0.10) gives each output a prediction set;
            "indeterminate" means both outcomes stay plausible. The LR-vs-GBM spread is shown as a model range.</p>
          <p><b>Validation.</b> {m.repeats}×{m.outer_folds} repeated stratified nested CV (strata = CAD × number of diseased vessels).
            C ∈ {"{0.05, 0.1, 0.2, 0.4}"}, blend weight w ∈ {"{0, .25, .5, .75, 1}"}, Platt and conformal thresholds all fitted inside the
            inner loop. 95% CIs mix patient bootstrap with repeat-to-repeat variation.</p>
          <p><b>Honest cohort replay.</b> In the dashboard every cohort patient is explained by the outer-fold model that
            never saw them, so what you browse is genuine out-of-sample behaviour, mistakes included.</p>
        </div>
      </div>
    </div>
  );
}
