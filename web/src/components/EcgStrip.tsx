import { useEffect, useRef } from "react";
import { cardiacPhase, clock, ecgSample, type EcgFlags } from "../lib/beat";
import { useStore } from "../store";

const FLAG_LIST: [keyof EcgFlags, string][] = [
  ["stDepression", "ST depression"],
  ["stElevation", "ST elevation"],
  ["tInversion", "T inversion"],
  ["qWave", "Q wave"],
  ["lvh", "LVH voltage"],
  ["lbbb", "LBBB"],
  ["rbbb", "RBBB"],
];

/** Sweeping monitor-style ECG. Morphology is schematic, driven by the patient's recorded ECG findings. */
export default function EcgStrip() {
  const result = useStore((s) => s.result);
  const ref = useRef<HTMLCanvasElement>(null);
  const hr = Math.round(result?.heart_rate ?? 72);
  const v = result?.values ?? {};
  const flags: EcgFlags = {
    stDepression: v.st_depression >= 0.5,
    stElevation: v.st_elevation >= 0.5,
    tInversion: v.t_inversion >= 0.5,
    qWave: v.q_wave >= 0.5,
    lvh: v.lvh >= 0.5,
    lbbb: v.bbb_lbbb >= 0.5,
    rbbb: v.bbb_rbbb >= 0.5,
  };
  const flagsRef = useRef(flags);
  flagsRef.current = flags;
  const hrRef = useRef(hr);
  hrRef.current = hr;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    let raf = 0;
    let lastX = -1;
    let lastY = 0;
    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      cv.width = cv.clientWidth * dpr;
      cv.height = cv.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
      lastX = -1;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(cv);
    const pxPerSec = 120;
    const draw = () => {
      const w = cv.clientWidth;
      const h = cv.clientHeight;
      const now = performance.now();
      const x = (((now - clock.t0) / 1000) * pxPerSec) % w;
      const y = h * 0.62 - ecgSample(cardiacPhase(now, hrRef.current), flagsRef.current) * h * 0.42;
      // erase a gap ahead of the pen, like a real bedside monitor
      ctx.clearRect(x, 0, 18, h);
      if (lastX >= 0 && x > lastX) {
        ctx.strokeStyle = "#43f3a5";
        ctx.lineWidth = 1.6;
        ctx.shadowColor = "rgba(67,243,165,0.7)";
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      lastX = x;
      lastY = y;
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="glass ecg" aria-label="schematic ECG">
      <div className="hr">
        <span className="u">HR · BPM</span>
        <span className="n">{hr}</span>
        <span className="u" style={{ marginTop: 4 }}>schematic</span>
      </div>
      <div className="trace"><canvas ref={ref} /></div>
      <div className="flags">
        {[...FLAG_LIST].sort((x, y) => Number(flags[y[0]]) - Number(flags[x[0]])).slice(0, 5).map(([k, label]) => (
          <span key={k} className={`f ${flags[k] ? "on" : ""}`}>{label}</span>
        ))}
      </div>
    </div>
  );
}
