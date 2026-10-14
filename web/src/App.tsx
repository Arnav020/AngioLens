import { lazy, Suspense, useEffect, useState } from "react";
import Disclaimer from "./components/Disclaimer";
import EcgStrip from "./components/EcgStrip";
import PatientPanel from "./components/PatientPanel";
import RightPanel from "./components/RightPanel";
import { TipLayer } from "./components/Tip";
import ViewerOverlay from "./components/ViewerOverlay";
import { useStore } from "./store";

const Viewer3D = lazy(() => import("./scene/Viewer3D"));
const ModelCard = lazy(() => import("./components/ModelCard"));

function Logo() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="14.5" fill="none" stroke="#24353d" />
      <path d="M3.5 17h6l2.6-7.5 4.2 14 3.1-9.6 1.6 3.1h7.5" fill="none" stroke="#e9a23b" strokeWidth="2.1"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function App() {
  const init = useStore((s) => s.init);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const llm = useStore((s) => s.llm);
  const error = useStore((s) => s.error);
  const [showDisc, setShowDisc] = useState(() => {
    try {
      return sessionStorage.getItem("angiolens-ack") !== "1";
    } catch {
      return true;
    }
  });

  useEffect(() => {
    init();
  }, [init]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Logo />
          <h1>
            Angio<em>Lens</em>
          </h1>
          <span className="tag">virtual angiogram</span>
        </div>
        <nav className="nav" aria-label="views">
          <button className={view === "dashboard" ? "on" : ""} onClick={() => setView("dashboard")}>
            Patient
          </button>
          <button className={view === "model" ? "on" : ""} onClick={() => setView("model")}>
            Model card &amp; validation
          </button>
        </nav>
        <div className="spacer" />
        <span className="status-chip" title="Grounded LLM copilot (Groq) availability">
          <span className={`dot ${llm ? "" : "off"}`} />
          {llm ? "copilot online" : "copilot offline · template mode"}
        </span>
        <button className="disclaimer-chip" onClick={() => setShowDisc(true)}>
          ⚠ Decision support · not a diagnosis
        </button>
      </header>

      {view === "dashboard" ? (
        <main className="main">
          <aside className="col left">
            <PatientPanel />
          </aside>
          <section className="viewer">
            <Suspense fallback={null}>
              <Viewer3D />
            </Suspense>
            <ViewerOverlay />
            <div className="overlay ov-bottom">
              <EcgStrip />
            </div>
          </section>
          <aside className="col right">
            {error && <div className="err">{error}</div>}
            <RightPanel />
          </aside>
        </main>
      ) : (
        <main className="model-page">
          <Suspense fallback={<div className="skeleton" />}>
            <ModelCard />
          </Suspense>
        </main>
      )}

      <footer className="footer">
        <strong>For decision support and education only.</strong>
        <span>Predictions are not a diagnosis and do not replace invasive or CT coronary angiography.</span>
        <span className="sep">·</span>
        <span>Data: Z-Alizadeh Sani (UCI, CC BY 4.0)</span>
        <span className="sep">·</span>
        <span>Anatomy: BodyParts3D, © The Database Center for Life Science (CC BY 4.0)</span>
      </footer>

      <TipLayer />
      {showDisc && (
        <Disclaimer
          onClose={() => {
            try {
              sessionStorage.setItem("angiolens-ack", "1");
            } catch {
              /* private mode */
            }
            setShowDisc(false);
          }}
        />
      )}
    </div>
  );
}
