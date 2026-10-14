"""One-page PDF 'virtual angiogram' report (ReportLab, no browser needed)."""

from __future__ import annotations

import io
import math
import re
from datetime import datetime

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String, Wedge
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .copilot import DISCLAIMER
from .model import VESSELS

INK = colors.HexColor("#0E1726")
MUTED = colors.HexColor("#5B6576")
LINE = colors.HexColor("#D5DAE1")
ACCENT = colors.HexColor("#C2410C")
CONF = {"positive": "confident +", "negative": "confident −", "indeterminate": "indeterminate", "atypical": "atypical"}


def risk_color(p: float) -> colors.Color:
    """Same 3-stop ramp as the web UI: teal (low) -> amber -> crimson (high)."""
    stops = [(0.0, (0x2A, 0x9D, 0x8F)), (0.5, (0xE9, 0xA2, 0x3B)), (1.0, (0xC8, 0x1D, 0x25))]
    p = max(0.0, min(1.0, p))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p <= p1:
            t = (p - p0) / (p1 - p0)
            return colors.Color(*[(a + (b - a) * t) / 255 for a, b in zip(c0, c1)])
    return colors.Color(*[c / 255 for c in stops[-1][1]])


def bullseye(segments: list[dict], size: float = 58 * mm) -> Drawing:
    """AHA 17-segment polar map, each segment coloured by its supplying vessel's probability."""
    d = Drawing(size, size)
    cx = cy = size / 2
    R = size / 2 - 2
    prob = {s["segment"]: s["prob"] for s in segments}
    rings = [  # (segments, start angle, outer radius, inner radius)
        (range(1, 7), R, R * 0.72),
        (range(7, 13), R * 0.72, R * 0.46),
        (range(13, 17), R * 0.46, R * 0.2),
    ]
    for segs, r_out, r_in in rings:
        segs = list(segs)
        n = len(segs)
        step = 360 / n
        # AHA convention: anterior segment centred at 12 o'clock, numbering runs counter-clockwise
        # (anterior -> septal on the left -> inferior at the bottom -> lateral on the right)
        start = 90 - step / 2
        for k, s in enumerate(segs):
            a0 = start + k * step
            w = Wedge(cx, cy, r_out, a0, a0 + step, radius1=r_in,
                      fillColor=risk_color(prob[s]), strokeColor=colors.white, strokeWidth=1.2)
            d.add(w)
            am = math.radians(a0 + step / 2)
            rm = (r_out + r_in) / 2
            d.add(String(cx + rm * math.cos(am), cy + rm * math.sin(am) - 3, str(s), fontSize=6.5,
                         fillColor=colors.white, textAnchor="middle", fontName="Helvetica-Bold"))
    d.add(Circle(cx, cy, R * 0.2, fillColor=risk_color(prob[17]), strokeColor=colors.white, strokeWidth=1.2))
    d.add(String(cx, cy - 3, "17", fontSize=6.5, fillColor=colors.white, textAnchor="middle", fontName="Helvetica-Bold"))
    return d


def _value_text(f: dict) -> str:
    if f["kind"] == "binary":
        return "present" if f["value"] >= 0.5 else "absent"
    return f"{f['value']:g} {f['unit']}".strip()


def factor_bars(features: list[dict], width: float = 112 * mm, n: int = 8) -> Drawing:
    """Diverging bar chart of the top-|SHAP| factors (orange raises risk, teal lowers it)."""
    feats = features[:n]
    row = 11
    d = Drawing(width, row * len(feats) + 4)
    mx = max(abs(f["shap"]) for f in feats) or 1.0
    label_w = width * 0.48
    mid = label_w + (width - label_w) / 2
    half = (width - label_w) / 2 - 4
    for i, f in enumerate(feats):
        y = d.height - (i + 1) * row
        label = f["label"] if len(f["label"]) < 26 else f["label"][:24] + "…"
        d.add(String(label_w - 4, y + 1.5, f"{label}  ({_value_text(f)})", fontSize=6.8, textAnchor="end",
                     fillColor=INK, fontName="Helvetica"))
        L = abs(f["shap"]) / mx * half
        col = ACCENT if f["shap"] > 0 else colors.HexColor("#2A9D8F")
        d.add(Rect(mid if f["shap"] > 0 else mid - L, y, L, 7, fillColor=col, strokeColor=None))
    d.add(Line(mid, 0, mid, d.height, strokeColor=MUTED, strokeWidth=0.6))
    return d


_PUNCT = str.maketrans({"‑": "-", "‐": "-", " ": " ", " ": " ", " ": " ",
                        "≈": "~", "−": "-"})


def _md_to_rl(text: str) -> str:
    text = text.translate(_PUNCT)
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.M)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)
    return text.replace("\n", "<br/>")


def build_pdf(expl: dict, narrative: str | None = None, model_card: dict | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="AngioLens virtual angiogram report")
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, textColor=INK, leading=20)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.5, textColor=INK, spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=8, textColor=INK, leading=10.5)
    small = ParagraphStyle("s", parent=body, fontSize=6.8, textColor=MUTED, leading=8.5)
    warn = ParagraphStyle("w", parent=body, fontSize=7.6, textColor=colors.HexColor("#7C2D12"), leading=9.6)

    t = expl["targets"]
    who = f"Cohort patient #{expl['patient_id']} (out-of-fold replay)" if expl.get("mode") == "cohort_oof" \
        else "Custom patient profile (live model)"
    story = [
        Paragraph("AngioLens — virtual angiogram report", h1),
        Paragraph(f"{who} · generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", small),
        Spacer(1, 4),
    ]
    disc = Table([[Paragraph("<b>Not a diagnosis.</b> " + DISCLAIMER, warn)]], colWidths=[182 * mm])
    disc.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF4E5")),
                              ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#F2B37A")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [disc, Spacer(1, 6)]

    rows = [["", "Probability", "Model range", "90 % conformal set", "Operating flag"]]
    for k in ("CAD",) + VESSELS:
        r = t[k]
        lo, hi = r["model_range"]
        rows.append([k if k != "CAD" else "CAD (overall)", f"{r['prob'] * 100:.0f} %", f"{lo * 100:.0f}–{hi * 100:.0f} %",
                     CONF[r["conformal"]], ("≥ " if r["flag"] else "< ") + f"{r['threshold'] * 100:.0f} %"])
    tbl = Table(rows, colWidths=[30 * mm, 22 * mm, 24 * mm, 30 * mm, 24 * mm])
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
             ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE), ("FONT", (0, 1), (0, -1), "Helvetica-Bold", 8)]
    for i, k in enumerate(("CAD",) + VESSELS, start=1):
        style.append(("TEXTCOLOR", (1, i), (1, i), risk_color(t[k]["prob"])))
        style.append(("FONT", (1, i), (1, i), "Helvetica-Bold", 9))
    tbl.setStyle(TableStyle(style))

    eye = bullseye(expl["segments"])
    top = Table([[tbl, eye]], colWidths=[132 * mm, 50 * mm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [Paragraph("Risk summary", h2), top,
              Paragraph("Bull's-eye: AHA 17-segment left-ventricular map; each segment is coloured by the stenosis "
                        "probability of its usual supplying artery (LAD 1,2,7,8,13,14,17 · RCA 3,4,9,10,15 · "
                        "LCX 5,6,11,12,16; right-dominant circulation assumed).", small)]

    cad = expl["stages"]["CAD"]
    story += [Paragraph("Why this overall CAD estimate (exact SHAP, log-odds)", h2),
              Table([[factor_bars(cad["features"]),
                      Paragraph("Orange bars push the estimate up, teal bars push it down. Contributions are exactly "
                                "additive in log-odds: model baseline + all 55 contributions = this patient's "
                                f"{cad['prob'] * 100:.0f} %. They are statistical associations learned from 303 "
                                "patients, not causal effects.", small)]],
                    colWidths=[118 * mm, 64 * mm])]
    loc = []
    for v in VESSELS:
        st = expl["stages"][f"{v}|CAD"]
        tops = " · ".join(f"{f['label']} = {_value_text(f)} ({'↑' if f['shap'] > 0 else '↓'})" for f in st["features"][:3])
        loc.append([v, f"{t[v]['cond_prob'] * 100:.0f} %", Paragraph(tops, body)])
    lt = Table([["", "P(vessel | CAD)", "Top localising factors"]] + loc, colWidths=[16 * mm, 26 * mm, 140 * mm])
    lt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [Paragraph("Where: vessel localisation (stage 2 of the hierarchical model)", h2), lt]

    abn = [f for f in cad["features"] if f["status"] in ("high", "low")]
    if abn:
        story += [Paragraph("Measurements outside adult reference range", h2),
                  Paragraph(" · ".join(f"{f['label']} {f['value']:g} {f['unit']} ({f['status']})" for f in abn), body)]
    if narrative and any(0x0370 <= ord(ch) <= 0x1FFF for ch in narrative):  # Greek..Indic scripts
        story += [Paragraph("Plain-language summary", h2),
                  Paragraph("The copilot summary is in a script this PDF engine cannot typeset (e.g. Devanagari); "
                            "it is shown in the AngioLens dashboard.", small)]
        narrative = None
    if narrative:
        story += [Paragraph("Plain-language summary (LLM copilot, every number verified against the model output)", h2),
                  Paragraph(_md_to_rl(narrative), body)]
    if model_card and "targets" in model_card:
        perf = " · ".join(
            f"{k} AUC {model_card['targets'][k]['summary']['angiolens']['roc_auc']['mean']:.2f}"
            for k in ("CAD",) + VESSELS)
        story += [Spacer(1, 6), Paragraph(
            f"Model: AngioLens v{model_card.get('meta', {}).get('version', '?')} · validated by 10×5 repeated nested "
            f"cross-validation on the Z-Alizadeh Sani dataset (n = 303): {perf}.", small)]
    doc.build(story)
    return buf.getvalue()
