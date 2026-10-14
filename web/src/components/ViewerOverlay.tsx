import { useState } from "react";
import { useStore, type CameraPreset, type Layer } from "../store";
import Inspector from "./Inspector";

const VIEWS: [CameraPreset, string][] = [
  ["overview", "Torso"],
  ["anterior", "Anterior"],
  ["left", "Left lateral"],
  ["posterior", "Posterior"],
  ["inferior", "Inferior"],
];
const LAYERS: [Layer, string][] = [
  ["flow", "Contrast flow"],
  ["territory", "Territory at risk"],
  ["labels", "Vessel labels"],
  ["context", "Great vessels"],
  ["skeleton", "Ribs & spine"],
  ["torso", "Torso shell"],
  ["lungs", "Lungs"],
];

export default function ViewerOverlay() {
  const result = useStore((s) => s.result);
  const source = useStore((s) => s.source);
  const camera = useStore((s) => s.camera);
  const flyTo = useStore((s) => s.flyTo);
  const layers = useStore((s) => s.layers);
  const toggle = useStore((s) => s.toggleLayer);
  const loading = useStore((s) => s.loading);
  const [showLayers, setShowLayers] = useState(false);

  const age = result?.values.age;
  const sex = result ? (result.values.sex_male >= 0.5 ? "male" : "female") : "";
  const title =
    source.kind === "cohort" ? (
      <>Patient <span>#</span>{source.id}</>
    ) : source.kind === "preset" ? (
      <>{source.name === "intermediate" ? "Intermediate" : source.name[0].toUpperCase() + source.name.slice(1)}-risk <span>profile</span></>
    ) : (
      <>Custom <span>profile</span></>
    );

  return (
    <>
      <div className="overlay ov-tl">
        <div className="patient-id">
          <div className="who">{title}</div>
          <div className="sub">
            {result && <span className="pill">{age} y · {sex}</span>}
            {result?.mode === "cohort_oof" ? (
              <span className="pill honest" title="This patient is explained by the cross-validation fold model that never saw them during training.">
                ● out-of-fold replay · fold {(result.fold ?? 0) + 1}/5 · model never saw this patient
              </span>
            ) : (
              <span className="pill live" title="Prediction from the final model trained on all 303 patients.">
                ● live model{source.kind === "custom" && source.basedOn ? ` · edited from ${source.basedOn}` : ""}
              </span>
            )}
            {loading && <span className="pill">computing…</span>}
          </div>
        </div>
      </div>

      <div className="overlay ov-tr">
        <div className="glass toolbar" role="toolbar" aria-label="camera views">
          {VIEWS.map(([k, label]) => (
            <button key={k} className={camera.preset === k ? "on" : ""} onClick={() => flyTo(k)}>
              {label}
            </button>
          ))}
          <button className={showLayers ? "on" : ""} onClick={() => setShowLayers(!showLayers)} aria-expanded={showLayers}>
            Layers ▾
          </button>
        </div>
        {showLayers && (
          <div className="glass layers">
            {LAYERS.map(([k, label]) => (
              <label key={k}>
                <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
                {label}
              </label>
            ))}
          </div>
        )}
      </div>

      <div className="overlay ov-br">
        <Inspector />
      </div>

      <div className="overlay ov-bl">
        <div className="glass risk-legend">
          <div className="eyebrow">Stenosis probability</div>
          <div className="bar" />
          <div className="ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100%</span></div>
          <div className="hint">
            Drag to orbit · scroll to zoom · click an artery to see why. Flow slows and flickers in high-risk vessels;
            tinted myocardium is the territory each artery supplies.
          </div>
        </div>
      </div>
    </>
  );
}
