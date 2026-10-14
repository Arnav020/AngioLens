// Typed client for the AngioLens FastAPI backend.

export type Vessel = "LAD" | "LCX" | "RCA";
export type Target = "CAD" | Vessel;
export const VESSELS: Vessel[] = ["LAD", "LCX", "RCA"];
export const TARGETS: Target[] = ["CAD", "LAD", "LCX", "RCA"];

export interface FeatureSpec {
  id: string;
  raw: string;
  label: string;
  group: string;
  kind: "binary" | "numeric" | "ordinal";
  unit: string;
  min: number;
  max: number;
  step: number;
  normal: [number, number] | null;
  modifiable: boolean;
  description: string;
  level_labels: string[];
}

export interface Schema {
  groups: Record<string, string>;
  features: FeatureSpec[];
  targets: Target[];
  defaults: Record<string, number>;
  presets: Record<string, Record<string, number>>;
  thresholds: Record<Target, number>;
}

export interface Contribution {
  id: string;
  label: string;
  group: string;
  kind: string;
  value: number;
  unit: string;
  shap: number;
  status: "high" | "low" | "normal" | "present" | "absent" | "info";
}

export interface StageExplanation {
  base_logit: number;
  logit: number;
  prob: number;
  base_prob: number;
  features: Contribution[];
  groups: { id: string; label: string; shap: number }[];
  components: { lr: number; gbm: number };
}

export type Conformal = "positive" | "negative" | "indeterminate" | "atypical";

export interface TargetResult {
  prob: number;
  conformal: Conformal;
  threshold: number;
  flag: boolean;
  model_range: [number, number];
  cond_prob?: number;
  aha_segments?: number[];
  territory?: string;
}

export interface Similar {
  id: number;
  distance: number;
  similarity: number;
  age: number;
  sex: string;
  truth: Record<Target, number>;
}

export interface Explanation {
  mode: "live" | "cohort_oof";
  targets: Record<Target, TargetResult>;
  stages: Record<string, StageExplanation>;
  segments: { segment: number; vessel: Vessel; prob: number }[];
  values: Record<string, number>;
  similar: Similar[];
  heart_rate: number;
  patient_id?: number;
  truth?: Record<Target, number>;
  fold?: number;
}

export interface CohortRow {
  id: number;
  age: number;
  sex: string;
  truth: Record<Target, number>;
  oof: Record<Target, number>;
}

export interface CopilotResult {
  text: string;
  source: "llm" | "template";
  model: string | null;
  grounding: { passed: boolean; numbers_checked: number | null; attempts: number; fallback_reason?: string };
  latency_s: number;
}

export interface Sensitivity {
  feature: string;
  grid: number[];
  current: number;
  curves: Record<Target, number[]>;
}

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => j<{ status: string; version: string; llm: boolean }>("/api/health"),
  schema: () => j<Schema>("/api/schema"),
  cohort: () => j<CohortRow[]>("/api/cohort"),
  cohortPatient: (id: number) => j<Explanation>(`/api/cohort/${id}`),
  predict: (values: Record<string, number>, signal?: AbortSignal) =>
    j<Explanation>("/api/predict", { ...post({ values }), signal }),
  sensitivity: (values: Record<string, number>, feature: string) =>
    j<Sensitivity>("/api/sensitivity", post({ values, feature })),
  copilot: (body: { values?: Record<string, number>; patient_id?: number; audience: string; lang: string; question?: string }) =>
    j<CopilotResult>("/api/copilot", post(body)),
  modelCard: () => j<ModelCard>("/api/model-card"),
  encode: (records: Record<string, string | number>[]) =>
    j<{ values: { values: Record<string, number>; filled_with_defaults: string[] }[] }>("/api/encode", post({ records })),
  transcribe: async (blob: Blob) => {
    const fd = new FormData();
    fd.append("file", blob, "question.webm");
    return j<{ text: string }>("/api/transcribe", { method: "POST", body: fd });
  },
  report: async (body: { values?: Record<string, number>; patient_id?: number; narrative?: string }) => {
    const r = await fetch("/api/report", post(body));
    if (!r.ok) throw new Error(`report failed: ${r.status}`);
    return r.blob();
  },
};

// ---------------- model card ----------------
export interface MetricStat {
  mean: number;
  sd: number;
  ci95?: [number, number];
}
export interface TargetCard {
  prevalence: number;
  summary: Record<string, Record<string, MetricStat>>;
  tuned_threshold: Record<string, { mean: number; sd: number }>;
  curves: Record<string, {
    roc: [number, number][];
    pr?: [number, number][];
    reliability?: { p_mean: number; y_rate: number; n: number }[];
    dca?: { threshold: number; model: number; treat_all: number; treat_none: number }[];
    confusion?: number[][];
  }>;
  calibration: { slope: number; intercept: number };
  conformal: {
    target_coverage: number;
    empirical_coverage: number;
    indeterminate_rate: number;
    decided_rate: number;
    accuracy_when_decided: number | null;
  };
}
export interface ModelCard {
  models: Record<string, string>;
  targets: Record<Target, TargetCard>;
  coherence_violation_rate?: Record<string, number>;
  importance: Record<string, {
    features: { id: string; label: string; group: string; mean_abs_shap: number }[];
    groups: Record<string, number>;
    nonzero_lr_coefs: number;
  }>;
  dataset: {
    n_patients: number;
    n_model_features: number;
    prevalence: Record<Target, number>;
    n_diseased_vessels: Record<string, number>;
    label_inconsistencies: unknown[];
    dropped_columns: Record<string, string>;
  };
  meta: Record<string, unknown> & {
    version: string;
    repeats: number;
    outer_folds: number;
    inner_folds: number;
    thresholds: Record<Target, number>;
    stages: Record<string, { C: number; w: number; platt_a: number; platt_b: number }>;
  };
}
