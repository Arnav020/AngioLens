import { useEffect, useRef, useState } from "react";
import { api, type CopilotResult } from "../api";
import { useStore } from "../store";

/** Minimal, safe markdown -> React (bold, italic, bullets, paragraphs). No HTML injection. */
function Md({ text }: { text: string }) {
  const inline = (s: string, key: number) => {
    const parts = s.split(/(\*\*[^*]+\*\*|\*[^*]+\*|_[^_]+_)/g).filter(Boolean);
    return (
      <span key={key}>
        {parts.map((p, i) =>
          p.startsWith("**") ? <b key={i}>{p.slice(2, -2)}</b> :
          (p.startsWith("*") && p.endsWith("*")) || (p.startsWith("_") && p.endsWith("_")) ? <i key={i}>{p.slice(1, -1)}</i> :
          p,
        )}
      </span>
    );
  };
  const blocks: JSX.Element[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) blocks.push(<ul key={blocks.length}>{list.map((l, i) => <li key={i}>{inline(l, i)}</li>)}</ul>);
    list = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const m = line.match(/^\s*[-*•]\s+(.*)$/);
    if (m) list.push(m[1]);
    else {
      flush();
      if (line.trim()) blocks.push(<p key={blocks.length}>{inline(line.trim(), 0)}</p>);
    }
  }
  flush();
  return <>{blocks}</>;
}

export default function Copilot() {
  const result = useStore((s) => s.result);
  const values = useStore((s) => s.values);
  const source = useStore((s) => s.source);
  const llm = useStore((s) => s.llm);
  const [audience, setAudience] = useState<"clinician" | "patient">("clinician");
  const [lang, setLang] = useState<"en" | "hi">("en");
  const [out, setOut] = useState<CopilotResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [q, setQ] = useState("");
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const chunks = useRef<Blob[]>([]);

  const subject = source.kind === "cohort" ? { patient_id: source.id } : { values };

  const run = async (question?: string) => {
    setBusy(true);
    setErr(null);
    try {
      setOut(await api.copilot({ ...subject, audience, lang, question: question || undefined }));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // regenerate when the patient changes (only if a narrative is already shown)
  useEffect(() => {
    setOut(null);
    window.speechSynthesis?.cancel();
  }, [result?.patient_id, result?.mode]);

  const toggleMic = async () => {
    if (rec) {
      rec.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const r = new MediaRecorder(stream);
      chunks.current = [];
      r.ondataavailable = (e) => chunks.current.push(e.data);
      r.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRec(null);
        setBusy(true);
        try {
          const { text } = await api.transcribe(new Blob(chunks.current, { type: r.mimeType }));
          setQ(text);
          if (text) await run(text);
        } catch (e) {
          setErr((e as Error).message);
        } finally {
          setBusy(false);
        }
      };
      r.start();
      setRec(r);
    } catch {
      setErr("Microphone unavailable in this browser.");
    }
  };

  const speak = () => {
    const synth = window.speechSynthesis;
    if (!synth || !out) return;
    if (speaking) {
      synth.cancel();
      setSpeaking(false);
      return;
    }
    const clean = out.text.replace(/[*_#>`]/g, "").replace(/^\s*[-•]\s+/gm, "");
    const u = new SpeechSynthesisUtterance(clean);
    u.lang = lang === "hi" ? "hi-IN" : "en-IN";
    const voice = synth.getVoices().find((v) => v.lang.toLowerCase().startsWith(lang === "hi" ? "hi" : "en"));
    if (voice) u.voice = voice;
    u.rate = 1.0;
    u.onend = () => setSpeaking(false);
    setSpeaking(true);
    synth.speak(u);
  };

  const download = async () => {
    const blob = await api.report({ ...subject, narrative: out?.text });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = source.kind === "cohort" ? `angiolens_patient_${source.id}.pdf` : "angiolens_report.pdf";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="scroll">
      <div className="section">
        <div className="section-head">
          <h3>Grounded explanation copilot</h3>
          <span className="eyebrow">{llm ? "groq · gpt-oss-120b" : "offline template"}</span>
        </div>
        <p className="muted small" style={{ margin: "0 0 12px" }}>
          The language model never predicts anything. It rephrases a fact sheet built from the model output, and every
          number it writes is checked against that fact sheet before you see it.
        </p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <div className="seg">
            <button className={audience === "clinician" ? "on" : ""} onClick={() => setAudience("clinician")}>Clinician</button>
            <button className={audience === "patient" ? "on" : ""} onClick={() => setAudience("patient")}>Patient</button>
          </div>
          <div className="seg">
            <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>English</button>
            <button className={lang === "hi" ? "on" : ""} onClick={() => setLang("hi")}>हिन्दी</button>
          </div>
          <button className="btn primary" disabled={busy || !result} onClick={() => run()}>
            {busy ? "Writing…" : "Explain this result"}
          </button>
        </div>
      </div>

      <div className="section">
        {busy && !out && (<><div className="skeleton" /><div className="skeleton" style={{ width: "80%" }} /><div className="skeleton" style={{ width: "60%" }} /></>)}
        {err && <div className="err" style={{ margin: 0 }}>{err}</div>}
        {out && (
          <>
            <div className={`ground ${out.source === "template" ? "fallback" : ""}`} style={{ marginBottom: 10 }}>
              {out.source === "llm" ? (
                <>✓ Number grounding passed · {out.grounding.numbers_checked} numbers verified against model output · {out.latency_s}s</>
              ) : (
                <>◇ Deterministic template · {out.grounding.fallback_reason}</>
              )}
            </div>
            <div className="copilot-out" lang={lang}>
              <Md text={out.text} />
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button className="btn" onClick={speak}>{speaking ? "■ Stop" : "▶ Read aloud"}</button>
              <button className="btn" onClick={download}>↓ PDF report</button>
            </div>
          </>
        )}
        {!out && !busy && !err && (
          <div className="muted small">
            Choose an audience and language, then generate. Ask follow-up questions below by text or voice
            (voice is transcribed with Whisper).
          </div>
        )}
      </div>

      <div className="section">
        <div className="eyebrow" style={{ marginBottom: 8 }}>Ask about this result</div>
        <form className="ask" onSubmit={(e) => { e.preventDefault(); if (q.trim()) run(q); }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. Why is the LAD higher than the RCA?" />
          <button type="button" className={`btn mic ${rec ? "rec" : ""}`} onClick={toggleMic} disabled={!llm || busy}
            title={llm ? "Ask by voice" : "Voice needs the Groq API key"}>
            {rec ? "■" : "🎙"}
          </button>
          <button className="btn" type="submit" disabled={busy || !q.trim()}>Ask</button>
        </form>
        <div className="muted small" style={{ marginTop: 8 }}>
          Questions the model output cannot answer (dosing, targets, prognosis) are declined.
        </div>
      </div>
    </div>
  );
}
