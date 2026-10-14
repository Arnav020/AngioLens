"""Build docs/AngioLens_Technical_Report.pdf (A4, <= 6 pages) from models/metrics.json.

    python scripts/make_figures.py && python scripts/build_report.py

Every number in the report is read from metrics.json, so the PDF can never drift from the model.
Rendering uses headless Chromium (Playwright): `pip install playwright && playwright install chromium`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
M = json.loads((ROOT / "models" / "metrics.json").read_text())
T = ["CAD", "LAD", "LCX", "RCA"]
NAME = {"CAD": "Overall CAD", "LAD": "LAD stenosis", "LCX": "LCX stenosis", "RCA": "RCA stenosis"}
MODEL_NAME = {"angiolens": "<b>AngioLens</b> (hierarchical blend)", "direct_blend": "Direct LR+GBM blend",
              "lr": "L1 logistic regression", "gbm": "LightGBM", "rf": "Random forest"}


def s(t, k, model="angiolens"):
    return M["targets"][t]["summary"][model][k]


def f3(x):
    return f"{x:.3f}"


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


def ci(t, k):
    c = s(t, k).get("ci95")
    return f"{c[0]:.2f}–{c[1]:.2f}" if c else "–"


def results_table():
    rows = []
    for t in T:
        tt = M["targets"][t]["tuned_threshold"]
        rows.append(
            f"<tr><td>{NAME[t]}</td><td>{pc(M['targets'][t]['prevalence'])}</td>"
            f"<td><b>{f3(s(t, 'roc_auc')['mean'])}</b><br><span class=ci>{ci(t, 'roc_auc')}</span></td>"
            f"<td>{f3(s(t, 'pr_auc')['mean'])}</td><td>{f3(s(t, 'brier')['mean'])}</td><td>{f3(s(t, 'ece')['mean'])}</td>"
            f"<td>{pc(tt['precision']['mean'])}</td><td>{pc(tt['recall']['mean'])} / {pc(tt['specificity']['mean'])}</td>"
            f"<td>{pc(tt['accuracy']['mean'])}</td><td>{f3(tt['f1']['mean'])}</td><td>{f3(tt['mcc']['mean'])}</td>"
            f"<td>{pc(s(t, 'accuracy')['mean'])} · {f3(s(t, 'f1')['mean'])}</td></tr>")
    return "\n".join(rows)


def comparison_table():
    models = list(M["models"])
    out = []
    for k in models:
        cells = []
        for t in T:
            best = max(M["targets"][t]["summary"][m]["roc_auc"]["mean"] for m in models)
            v = M["targets"][t]["summary"][k]
            b = abs(v["roc_auc"]["mean"] - best) < 5e-4
            cells.append(f"<td class={'best' if b else ''}>{f3(v['roc_auc']['mean'])} <span class=ci>({f3(v['brier']['mean'])})</span></td>")
        coh = M.get("coherence_violation_rate", {}).get(k)
        out.append(f"<tr><td>{MODEL_NAME[k]}</td>{''.join(cells)}<td>{pc(1 - coh, 1) if coh is not None else '–'}</td></tr>")
    return "\n".join(out)


def conformal_table():
    out = []
    for t in T:
        c = M["targets"][t]["conformal"]
        out.append(f"<tr><td>{t}</td><td>{pc(c['empirical_coverage'], 1)}</td><td>{pc(c['indeterminate_rate'])}</td>"
                   f"<td>{pc(c['accuracy_when_decided'], 1)}</td><td>{f3(M['targets'][t]['calibration']['slope'])}</td></tr>")
    return "\n".join(out)


def main():
    d = M["dataset"]
    meta = M["meta"]
    stages = meta["stages"]
    ctx = {
        "RESULTS": results_table(),
        "COMPARISON": comparison_table(),
        "CONFORMAL": conformal_table(),
        "AUC_CAD": f3(s("CAD", "roc_auc")["mean"]), "AUC_LAD": f3(s("LAD", "roc_auc")["mean"]),
        "AUC_LCX": f3(s("LCX", "roc_auc")["mean"]), "AUC_RCA": f3(s("RCA", "roc_auc")["mean"]),
        "CI_CAD": ci("CAD", "roc_auc"),
        "BRIER_CAD": f3(s("CAD", "brier")["mean"]),
        "COH_DIRECT": pc(M["coherence_violation_rate"]["direct_blend"], 0),
        "COH_RF": pc(M["coherence_violation_rate"]["rf"], 1),
        "N": d["n_patients"], "NF": d["n_model_features"],
        "PREV_CAD": pc(d["prevalence"]["CAD"]), "PREV_LAD": pc(d["prevalence"]["LAD"]),
        "PREV_LCX": pc(d["prevalence"]["LCX"]), "PREV_RCA": pc(d["prevalence"]["RCA"]),
        "V0": d["n_diseased_vessels"].get("0", 0), "V1": d["n_diseased_vessels"].get("1", 0),
        "V2": d["n_diseased_vessels"].get("2", 0), "V3": d["n_diseased_vessels"].get("3", 0),
        "REPEATS": meta["repeats"], "RUNTIME": int(meta["runtime_s"]),
        "STAGES": " · ".join(f"{k}: C={v['C']}, w={v['w']}" for k, v in stages.items()),
        "THR": " · ".join(f"{k} {v:.2f}" for k, v in meta["thresholds"].items()),
        "COV_CAD": pc(M["targets"]["CAD"]["conformal"]["empirical_coverage"], 1),
        "IND_CAD": pc(M["targets"]["CAD"]["conformal"]["indeterminate_rate"]),
        "ACCD_CAD": pc(M["targets"]["CAD"]["conformal"]["accuracy_when_decided"], 1),
        "IND_LCX": pc(M["targets"]["LCX"]["conformal"]["indeterminate_rate"]),
        "IND_RCA": pc(M["targets"]["RCA"]["conformal"]["indeterminate_rate"]),
        "ROOT": ROOT.as_uri(),
    }
    html = (DOCS / "report_template.html").read_text(encoding="utf-8")
    for k, v in ctx.items():
        html = html.replace("{{" + k + "}}", str(v))
    assert "{{" not in html, "unfilled placeholder"
    out_html = DOCS / "report.html"
    out_html.write_text(html, encoding="utf-8")

    from playwright.async_api import async_playwright

    async def render():
        async with async_playwright() as p:
            b = await p.chromium.launch()
            pg = await b.new_page()
            await pg.goto(out_html.as_uri())
            await pg.wait_for_timeout(800)
            await pg.pdf(path=str(DOCS / "AngioLens_Technical_Report.pdf"), format="A4", print_background=True,
                         margin={"top": "13mm", "bottom": "13mm", "left": "14mm", "right": "14mm"},
                         display_header_footer=True, header_template="<span></span>",
                         footer_template='<div style="font:8px Segoe UI,Arial;color:#8a929b;width:100%;text-align:center">'
                                         'AngioLens · Random Forest Rangers · Track A · page <span class="pageNumber"></span>'
                                         ' / <span class="totalPages"></span></div>')
            await b.close()

    asyncio.run(render())
    try:
        from pypdf import PdfReader

        n = len(PdfReader(str(DOCS / "AngioLens_Technical_Report.pdf")).pages)
    except ImportError:
        n = "?"
    print(f"report -> docs/AngioLens_Technical_Report.pdf ({n} pages)")


if __name__ == "__main__":
    main()
