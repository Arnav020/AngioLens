import { create } from "zustand";

// One floating tooltip shared by every chart (hover layer).
interface TipState {
  tip: { x: number; y: number; title: string; rows: [string, string][]; note?: string } | null;
  show: (x: number, y: number, title: string, rows: [string, string][], note?: string) => void;
  hide: () => void;
}
export const useTip = create<TipState>((set) => ({
  tip: null,
  show: (x, y, title, rows, note) => set({ tip: { x, y, title, rows, note } }),
  hide: () => set({ tip: null }),
}));

export function TipLayer() {
  const tip = useTip((s) => s.tip);
  if (!tip) return null;
  const left = Math.min(tip.x + 14, window.innerWidth - 270);
  const top = Math.min(tip.y + 14, window.innerHeight - 120);
  return (
    <div className="chart-tip" style={{ left, top }}>
      <div className="t">{tip.title}</div>
      {tip.rows.map(([k, v]) => (
        <div className="r" key={k}>
          <span>{k}</span>
          <span>{v}</span>
        </div>
      ))}
      {tip.note && <div className="muted small" style={{ marginTop: 4 }}>{tip.note}</div>}
    </div>
  );
}
