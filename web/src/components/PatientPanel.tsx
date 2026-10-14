import { useMemo, useState } from "react";
import { api, TARGETS, type FeatureSpec } from "../api";
import { riskHex } from "../lib/color";
import { useStore } from "../store";

type Filter = "all" | "cad" | "normal" | "missed";

function CohortBrowser() {
  const cohort = useStore((s) => s.cohort);
  const source = useStore((s) => s.source);
  const loadCohort = useStore((s) => s.loadCohort);
  const thr = useStore((s) => s.schema?.thresholds.CAD ?? 0.5);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [sort, setSort] = useState<"id" | "risk">("id");

  const rows = useMemo(() => {
    let r = cohort;
    if (q.trim()) r = r.filter((p) => String(p.id).startsWith(q.trim().replace("#", "")));
    if (filter === "cad") r = r.filter((p) => p.truth.CAD === 1);
    if (filter === "normal") r = r.filter((p) => p.truth.CAD === 0);
    if (filter === "missed") r = r.filter((p) => (p.oof.CAD >= thr ? 1 : 0) !== p.truth.CAD);
    if (sort === "risk") r = [...r].sort((a, b) => b.oof.CAD - a.oof.CAD);
    return r;
  }, [cohort, q, filter, sort, thr]);

  const active = source.kind === "cohort" ? source.id : -1;
  return (
    <>
      <div className="section" style={{ paddingBottom: 10 }}>
        <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
          <input className="search" placeholder="Find patient #id" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="seg">
            <button className={sort === "id" ? "on" : ""} onClick={() => setSort("id")}>#</button>
            <button className={sort === "risk" ? "on" : ""} onClick={() => setSort("risk")}>risk</button>
          </div>
        </div>
        <div className="seg" style={{ width: "100%" }}>
          {(["all", "cad", "normal", "missed"] as Filter[]).map((f) => (
            <button key={f} className={filter === f ? "on" : ""} style={{ flex: 1 }} onClick={() => setFilter(f)}
              title={f === "missed" ? "Patients the model misclassified out-of-fold at its operating threshold" : undefined}>
              {f === "all" ? "All" : f === "cad" ? "CAD+" : f === "normal" ? "Normal" : "Missed"}
            </button>
          ))}
        </div>
        <div className="legend-mini" style={{ marginTop: 9 }}>
          <span>CAD · LAD · LCX · RCA</span>
          <span><span className="sw" style={{ borderRadius: "50%", boxShadow: "0 0 0 1.5px #05080a, 0 0 0 2.5px #e6edef", background: "#e9a23b" }} />= stenotic at angiography</span>
        </div>
      </div>
      <div className="scroll" role="listbox" aria-label="cohort patients">
        {rows.map((p) => (
          <div key={p.id} role="option" aria-selected={p.id === active}
            className={`cohort-row ${p.id === active ? "on" : ""}`} onClick={() => loadCohort(p.id)}>
            <span className="pid">#{p.id}</span>
            <span className="meta">{p.age} y · {p.sex}</span>
            <span className="dots" title="out-of-fold predicted probability (fill) and angiography truth (ring)">
              {TARGETS.map((t) => (
                <span key={t} className={`d ${p.truth[t] ? "truth" : ""}`} style={{ background: riskHex(p.oof[t]) }} />
              ))}
            </span>
          </div>
        ))}
        {rows.length === 0 && <div className="section muted small">No patients match.</div>}
      </div>
    </>
  );
}

function Field({ f }: { f: FeatureSpec }) {
  const v = useStore((s) => s.values[f.id]);
  const setValue = useStore((s) => s.setValue);
  if (v === undefined) return null;
  const status = f.normal ? (v < f.normal[0] ? "low" : v > f.normal[1] ? "high" : "") : "";
  if (f.kind === "binary") {
    return (
      <div className="field">
        <label htmlFor={f.id}>{f.label}</label>
        <button id={f.id} role="switch" aria-checked={v >= 0.5} className={`toggle ${v >= 0.5 ? "on" : ""}`}
          onClick={() => setValue(f.id, v >= 0.5 ? 0 : 1)} />
      </div>
    );
  }
  if (f.kind === "ordinal") {
    const n = Math.round(f.max - f.min) + 1;
    return (
      <div className="field" title={f.description}>
        <label>{f.label}</label>
        <span className="stepper">
          {Array.from({ length: n }, (_, i) => f.min + i).map((k) => (
            <button key={k} className={Math.round(v) === k ? "on" : ""} onClick={() => setValue(f.id, k)}
              title={f.level_labels[k - f.min] ?? String(k)}>
              {f.id === "vhd" ? ["–", "mild", "mod", "sev"][k] : k}
            </button>
          ))}
        </span>
      </div>
    );
  }
  // numeric: slider over a sensible window around the observed range
  return (
    <div className="field" title={f.normal ? `Reference ${f.normal[0]}–${f.normal[1]} ${f.unit}` : f.description}>
      <label htmlFor={f.id}>{f.label}</label>
      <span className={`val ${status}`}>
        {Number.isInteger(f.step) ? Math.round(v) : v.toFixed(1)} <span className="muted">{f.unit}</span>
      </span>
      <input id={f.id} type="range" min={f.min} max={sliderMax(f)} step={f.step} value={v}
        onChange={(e) => setValue(f.id, parseFloat(e.target.value))} />
    </div>
  );
}

// tighter slider ceilings than the hard validation limits, so the thumb moves usefully
const CEIL: Record<string, number> = {
  age: 95, weight: 140, height: 200, bmi: 45, bp: 210, pr: 140, fbs: 400, cr: 4, tg: 800, ldl: 260, hdl: 110,
  bun: 60, esr: 100, hb: 19, k: 7, na: 160, wbc: 20000, plt: 700, ef_tte: 70,
};
const sliderMax = (f: FeatureSpec) => CEIL[f.id] ?? f.max;

/** Minimal CSV parser (quoted fields, comma or semicolon separated). */
function parseCsv(text: string): Record<string, string>[] {
  const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim());
  if (lines.length < 2) return [];
  const sep = lines[0].split(";").length > lines[0].split(",").length ? ";" : ",";
  const split = (l: string) => {
    const out: string[] = [];
    let cur = "", q = false;
    for (const ch of l) {
      if (ch === '"') q = !q;
      else if (ch === sep && !q) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out.map((x) => x.trim());
  };
  const head = split(lines[0]);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}

function ImportExport() {
  const values = useStore((s) => s.values);
  const setValues = useStore((s) => s.setValues);
  const [msg, setMsg] = useState<string | null>(null);
  const onFile = async (file: File) => {
    try {
      const text = await file.text();
      const rows: Record<string, string | number>[] = file.name.toLowerCase().endsWith(".json")
        ? (() => { const j = JSON.parse(text); const v = j.values ?? j; return Array.isArray(v) ? v : [v]; })()
        : parseCsv(text);
      if (!rows.length) throw new Error("no rows found");
      const r = await api.encode(rows.slice(0, 1));
      const first = r.values[0];
      setValues(first.values, file.name);
      setMsg(`Loaded ${file.name}${rows.length > 1 ? ` (row 1 of ${rows.length})` : ""}` +
        (first.filled_with_defaults.length ? ` · ${first.filled_with_defaults.length} missing fields set to cohort median` : ""));
    } catch (e) {
      setMsg(`Import failed: ${(e as Error).message}`);
    }
  };
  const exportJson = () => {
    const blob = new Blob([JSON.stringify({ values }, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "angiolens_patient.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <div className="section" style={{ padding: "10px 16px" }}>
      <div style={{ display: "flex", gap: 6 }}>
        <label className="btn" style={{ flex: 1 }} title="JSON {values:{...}} or CSV with feature ids or raw UCI column names">
          ↑ Import JSON / CSV
          <input type="file" accept=".json,.csv,.txt" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
        <button className="btn" onClick={exportJson}>↓ Export</button>
      </div>
      {msg && <div className="small muted" style={{ marginTop: 6 }}>{msg}</div>}
    </div>
  );
}

function ProfileEditor() {
  const schema = useStore((s) => s.schema);
  const [open, setOpen] = useState<Record<string, boolean>>({ symptom: true, ecg: true });
  if (!schema) return <div className="section"><div className="skeleton" /><div className="skeleton" /></div>;
  return (
    <div className="scroll">
      <ImportExport />
      {Object.entries(schema.groups).map(([g, label]) => {
        const feats = schema.features.filter((f) => f.group === g);
        return (
          <div className="fgroup" key={g}>
            <button onClick={() => setOpen({ ...open, [g]: !open[g] })} aria-expanded={!!open[g]}>
              <span>{label}</span>
              <span className="count">{feats.length} {open[g] ? "▾" : "▸"}</span>
            </button>
            {open[g] && <div style={{ paddingBottom: 8 }}>{feats.map((f) => <Field key={f.id} f={f} />)}</div>}
          </div>
        );
      })}
    </div>
  );
}

export default function PatientPanel() {
  const source = useStore((s) => s.source);
  const loadPreset = useStore((s) => s.loadPreset);
  const cohortN = useStore((s) => s.cohort.length);
  const [tab, setTab] = useState<"cohort" | "profile">("cohort");

  return (
    <>
      <div className="section">
        <div className="section-head">
          <span className="eyebrow">Patient source</span>
        </div>
        <div className="source-switch">
          <button className={tab === "cohort" ? "on" : ""} onClick={() => setTab("cohort")}>
            <b>Cohort</b>{cohortN || 303} real patients
          </button>
          <button className={source.kind === "preset" ? "on" : ""} onClick={() => { loadPreset("high"); setTab("profile"); }}>
            <b>Presets</b>low · mid · high
          </button>
          <button className={tab === "profile" ? "on" : ""} onClick={() => setTab("profile")}>
            <b>Profile</b>edit 55 inputs
          </button>
        </div>
        {tab === "profile" && (
          <div className="seg" style={{ marginTop: 10, width: "100%" }}>
            {["low", "intermediate", "high"].map((p) => (
              <button key={p} style={{ flex: 1 }} className={source.kind === "preset" && source.name === p ? "on" : ""}
                onClick={() => loadPreset(p)}>
                {p === "intermediate" ? "Intermediate" : p[0].toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
        )}
      </div>
      {tab === "cohort" ? <CohortBrowser /> : <ProfileEditor />}
    </>
  );
}
