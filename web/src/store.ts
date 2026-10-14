import { create } from "zustand";
import { api, type CohortRow, type Explanation, type Schema, type Target, type Vessel } from "./api";

export type Source =
  | { kind: "cohort"; id: number }
  | { kind: "preset"; name: string }
  | { kind: "custom"; basedOn?: string };

/** A point clicked on the myocardium: which chamber, and how strongly each artery supplies it. */
export interface Region {
  chamber: string;
  weights: Record<Vessel, number>;
  vessel: Vessel;
}

export type Layer = "torso" | "skeleton" | "lungs" | "territory" | "flow" | "labels" | "context";
export type Panel = "risk" | "explain" | "whatif" | "copilot";
export type View = "dashboard" | "model";
export type CameraPreset = "anterior" | "left" | "posterior" | "inferior" | "overview";

interface State {
  schema: Schema | null;
  cohort: CohortRow[];
  llm: boolean;
  source: Source;
  values: Record<string, number>;
  result: Explanation | null;
  loading: boolean;
  error: string | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  region: Region | null;
  stage: string; // "CAD" | "LAD|CAD" | ...
  panel: Panel;
  view: View;
  layers: Record<Layer, boolean>;
  camera: { preset: CameraPreset; nonce: number };
  init: () => Promise<void>;
  loadCohort: (id: number) => Promise<void>;
  loadPreset: (name: string) => Promise<void>;
  setValue: (id: string, v: number) => void;
  setValues: (vals: Record<string, number>, basedOn?: string) => void;
  select: (v: Vessel | null) => void;
  pickRegion: (r: Region | null) => void;
  hover: (v: Vessel | null) => void;
  setStage: (s: string) => void;
  setPanel: (p: Panel) => void;
  setView: (v: View) => void;
  toggleLayer: (l: Layer) => void;
  flyTo: (p: CameraPreset) => void;
}

let ctrl: AbortController | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;

export const useStore = create<State>((set, get) => {
  const runPredict = (values: Record<string, number>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      ctrl?.abort();
      ctrl = new AbortController();
      set({ loading: true });
      try {
        const result = await api.predict(values, ctrl.signal);
        set({ result, loading: false, error: null });
      } catch (e) {
        if ((e as Error).name !== "AbortError") set({ loading: false, error: (e as Error).message });
      }
    }, 90);
  };

  return {
    schema: null,
    cohort: [],
    llm: false,
    source: { kind: "preset", name: "high" },
    values: {},
    result: null,
    loading: false,
    error: null,
    selected: null,
    hovered: null,
    region: null,
    stage: "CAD",
    panel: "risk",
    view: "dashboard",
    layers: { torso: true, skeleton: true, lungs: false, territory: true, flow: true, labels: true, context: true },
    camera: { preset: "overview", nonce: 0 },

    init: async () => {
      try {
        const [schema, cohort, health] = await Promise.all([api.schema(), api.cohort(), api.health()]);
        set({ schema, cohort, llm: health.llm });
        // open on a real, honest example: a cohort patient replayed out-of-fold
        await get().loadCohort(99);
      } catch (e) {
        set({ error: `Cannot reach the AngioLens API (${(e as Error).message}). Is \`python run.py\` running?` });
      }
    },

    loadCohort: async (id) => {
      set({ loading: true, source: { kind: "cohort", id } });
      try {
        const result = await api.cohortPatient(id);
        set({ result, values: result.values, loading: false, error: null });
      } catch (e) {
        set({ loading: false, error: (e as Error).message });
      }
    },

    loadPreset: async (name) => {
      const vals = get().schema?.presets[name];
      if (!vals) return;
      set({ source: { kind: "preset", name }, values: { ...vals } });
      runPredict({ ...vals });
    },

    setValue: (id, v) => {
      const values = { ...get().values, [id]: v };
      const src = get().source;
      const basedOn =
        src.kind === "cohort" ? `patient #${src.id}` : src.kind === "preset" ? `${src.name}-risk preset` : src.basedOn;
      set({ values, source: { kind: "custom", basedOn } });
      runPredict(values);
    },

    setValues: (vals, basedOn) => {
      const values = { ...get().values, ...vals };
      set({ values, source: { kind: "custom", basedOn } });
      runPredict(values);
    },

    select: (v) => set({ selected: v, stage: v ? `${v}|CAD` : "CAD", region: null }),
    pickRegion: (r) => set({ region: r, selected: r ? r.vessel : null, stage: r ? `${r.vessel}|CAD` : "CAD" }),
    hover: (v) => set({ hovered: v }),
    setStage: (s) => set({ stage: s }),
    setPanel: (p) => set({ panel: p }),
    setView: (v) => set({ view: v }),
    toggleLayer: (l) => set({ layers: { ...get().layers, [l]: !get().layers[l] } }),
    flyTo: (p) => set({ camera: { preset: p, nonce: get().camera.nonce + 1 } }),
  };
});

export const targetLabel = (t: Target | string) =>
  ({ CAD: "Overall CAD", LAD: "LAD", LCX: "LCX", RCA: "RCA" } as Record<string, string>)[t] ?? t;
