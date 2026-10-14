import type { Vessel } from "../api";
import { pct, riskHex } from "../lib/color";
import { useStore } from "../store";
import { useTip } from "./Tip";

const NAMES: Record<number, string> = {
  1: "basal anterior", 2: "basal anteroseptal", 3: "basal inferoseptal", 4: "basal inferior", 5: "basal inferolateral",
  6: "basal anterolateral", 7: "mid anterior", 8: "mid anteroseptal", 9: "mid inferoseptal", 10: "mid inferior",
  11: "mid inferolateral", 12: "mid anterolateral", 13: "apical anterior", 14: "apical septal", 15: "apical inferior",
  16: "apical lateral", 17: "apex",
};

/** AHA 17-segment polar map. Each segment inherits its usual supplying artery's probability. */
export default function BullsEye({ size = 168 }: { size?: number }) {
  const result = useStore((s) => s.result);
  const hovered = useStore((s) => s.hovered);
  const selected = useStore((s) => s.selected);
  const hover = useStore((s) => s.hover);
  const select = useStore((s) => s.select);
  const show = useTip((s) => s.show);
  const hide = useTip((s) => s.hide);
  if (!result) return null;
  const focus = hovered ?? selected;
  const seg = Object.fromEntries(result.segments.map((s) => [s.segment, s]));
  const c = size / 2;
  const R = size / 2 - 3;
  const rings: [number[], number, number][] = [
    [[1, 2, 3, 4, 5, 6], R, R * 0.72],
    [[7, 8, 9, 10, 11, 12], R * 0.72, R * 0.46],
    [[13, 14, 15, 16], R * 0.46, R * 0.2],
  ];
  // SVG y grows downward: AHA anterior at 12 o'clock, septum on the left, numbering counter-clockwise
  const P = (deg: number, r: number) => [c + r * Math.cos((deg * Math.PI) / 180), c - r * Math.sin((deg * Math.PI) / 180)];
  const wedge = (a0: number, a1: number, ro: number, ri: number) => {
    const [x0, y0] = P(a0, ro), [x1, y1] = P(a1, ro), [x2, y2] = P(a1, ri), [x3, y3] = P(a0, ri);
    return `M${x0},${y0} A${ro},${ro} 0 0 0 ${x1},${y1} L${x2},${y2} A${ri},${ri} 0 0 1 ${x3},${y3} Z`;
  };
  const props = (s: number) => {
    const v = seg[s].vessel as Vessel;
    return {
      fill: riskHex(seg[s].prob),
      stroke: "#0b1114",
      strokeWidth: 1.6,
      opacity: focus && focus !== v ? 0.28 : 1,
      style: { cursor: "pointer", transition: "opacity 0.2s" },
      onMouseEnter: (e: React.MouseEvent) => {
        hover(v);
        show(e.clientX, e.clientY, `Segment ${s} · ${NAMES[s]}`, [["Supplied by", v], ["P(stenosis)", pct(seg[s].prob)]]);
      },
      onMouseMove: (e: React.MouseEvent) =>
        show(e.clientX, e.clientY, `Segment ${s} · ${NAMES[s]}`, [["Supplied by", v], ["P(stenosis)", pct(seg[s].prob)]]),
      onMouseLeave: () => {
        hover(null);
        hide();
      },
      onClick: () => select(selected === v ? null : v),
    };
  };
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="AHA 17-segment bull's-eye map">
      {rings.map(([segs, ro, ri]) => {
        const step = 360 / segs.length;
        return segs.map((s, k) => {
          const a0 = 90 - step / 2 + k * step;
          const [lx, ly] = P(a0 + step / 2, (ro + ri) / 2);
          return (
            <g key={s}>
              <path d={wedge(a0, a0 + step, ro, ri)} {...props(s)} />
              <text x={lx} y={ly + 3} textAnchor="middle" fill="#0b1114" style={{ font: "600 9px var(--mono)", pointerEvents: "none" }}>
                {s}
              </text>
            </g>
          );
        });
      })}
      <circle cx={c} cy={c} r={R * 0.2} {...props(17)} />
      <text x={c} y={c + 3} textAnchor="middle" fill="#0b1114" style={{ font: "600 9px var(--mono)", pointerEvents: "none" }}>17</text>
    </svg>
  );
}
