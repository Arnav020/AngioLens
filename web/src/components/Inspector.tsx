import { VESSELS, type Vessel } from "../api";
import { CONFORMAL_LABEL, fmtValue, pct, riskHex } from "../lib/color";
import { useStore } from "../store";

const CHAMBER: Record<string, string> = {
  heart_lv: "Left ventricle",
  heart_rv: "Right ventricle",
  heart_la: "Left atrium",
  heart_ra: "Right atrium",
};
const FULL: Record<Vessel, string> = { LAD: "Left anterior descending", LCX: "Left circumflex", RCA: "Right coronary artery" };

/** Floating card in the 3D viewport describing the selected artery or the clicked myocardial region. */
export default function Inspector() {
  const r = useStore((s) => s.result);
  const selected = useStore((s) => s.selected);
  const region = useStore((s) => s.region);
  const select = useStore((s) => s.select);
  const setPanel = useStore((s) => s.setPanel);
  if (!r || !selected) return null;
  const t = r.targets[selected];
  const st = r.stages[`${selected}|CAD`];
  const col = riskHex(t.prob);
  return (
    <div className="glass inspector" style={{ ["--c" as string]: col }}>
      <div className="section-head" style={{ marginBottom: 6 }}>
        <span className="eyebrow">{region ? `${CHAMBER[region.chamber]} · picked point` : "Artery inspector"}</span>
        <button className="x" onClick={() => select(null)} aria-label="close inspector">×</button>
      </div>
      {region && (
        <div className="terr">
          {VESSELS.map((v) => (
            <div key={v} className="terr-row">
              <span className="mono">{v}</span>
              <div className="tb"><div style={{ width: `${region.weights[v] * 100}%`, background: riskHex(r.targets[v].prob) }} /></div>
              <span className="mono small muted">{pct(region.weights[v])}</span>
            </div>
          ))}
          <div className="muted small" style={{ margin: "4px 0 8px" }}>supply share at this point (distance-weighted territory)</div>
        </div>
      )}
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
        <div>
          <div style={{ font: "600 18px var(--mono)", color: col }}>{selected}</div>
          <div className="small">{FULL[selected]}</div>
        </div>
        <div style={{ font: "500 28px var(--mono)", color: col }}>{pct(t.prob)}</div>
      </div>
      <div className="muted small" style={{ margin: "4px 0 8px" }}>{t.territory}</div>
      <div className="small" style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span className={`badge ${t.conformal}`}>{CONFORMAL_LABEL[t.conformal]}</span>
        <span className="mono muted">{pct(r.targets.CAD.prob)} × {pct(t.cond_prob ?? 0)}</span>
      </div>
      <div className="eyebrow" style={{ margin: "8px 0 4px" }}>Top localising factors</div>
      {st.features.slice(0, 3).map((f) => (
        <div key={f.id} className="small" style={{ display: "flex", justifyContent: "space-between", gap: 8, padding: "2px 0" }}>
          <span>{f.label} <span className="muted mono">{fmtValue(f.value, f.unit, f.kind)}</span></span>
          <span className="mono" style={{ color: f.shap > 0 ? "#d95926" : "#3987e5" }}>{f.shap > 0 ? "↑" : "↓"}</span>
        </div>
      ))}
      {r.truth && (
        <div className="small mono" style={{ marginTop: 6, color: r.truth[selected] === (t.flag ? 1 : 0) ? "#7fd8c9" : "#ff9c8a" }}>
          angiography: {r.truth[selected] ? "stenotic" : "normal"} · model flag {t.flag ? "≥" : "<"} {pct(t.threshold)}
        </div>
      )}
      <button className="btn ghost" style={{ width: "100%", marginTop: 10 }} onClick={() => setPanel("explain")}>
        Full explanation →
      </button>
    </div>
  );
}
