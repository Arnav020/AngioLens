"""Turns raw model outputs into the structured explanation the dashboard shows."""

from __future__ import annotations

import numpy as np

from .model import VESSELS, AngioLensModel, sigmoid
from .schema import BY_ID, FEATURE_IDS, GROUPS

# AHA 17-segment model: standard coronary territory assignment
# (Cerqueira et al., Circulation 2002;105:539-542).
AHA_TERRITORY = {
    "LAD": [1, 2, 7, 8, 13, 14, 17],
    "RCA": [3, 4, 9, 10, 15],
    "LCX": [5, 6, 11, 12, 16],
}
TERRITORY_TEXT = {
    "LAD": "Anterior wall, anterior septum and apex (≈45–55 % of the left-ventricular mass)",
    "LCX": "Lateral and posterolateral walls of the left ventricle",
    "RCA": "Inferior wall, inferior septum and right ventricle (right-dominant circulation)",
}


def status_of(fid: str, value: float) -> str:
    f = BY_ID[fid]
    if f.kind == "binary":
        return "present" if value >= 0.5 else "absent"
    if f.normal is None:
        return "info"
    lo, hi = f.normal
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "normal"


def _contributions(phi: np.ndarray, x: np.ndarray, base: float) -> dict:
    items = []
    for j, fid in enumerate(FEATURE_IDS):
        f = BY_ID[fid]
        items.append({
            "id": fid, "label": f.label, "group": f.group, "kind": f.kind, "value": float(x[j]), "unit": f.unit,
            "shap": float(phi[j]), "status": status_of(fid, float(x[j])),
        })
    items.sort(key=lambda d: -abs(d["shap"]))
    groups = {g: 0.0 for g in GROUPS}
    for it in items:
        groups[it["group"]] += it["shap"]
    total = float(base + phi.sum())
    return {"base_logit": float(base), "logit": total, "prob": float(sigmoid(total)),
            "base_prob": float(sigmoid(base)), "features": items,
            "groups": [{"id": g, "label": GROUPS[g], "shap": float(s)} for g, s in groups.items()]}


def explain_row(model: AngioLensModel, x: np.ndarray) -> dict:
    """Full prediction + exact SHAP explanation for one patient row."""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    p = model.predict(x)
    out = {"targets": {}, "stages": {}}
    base, phi = model.cad.shap(x)
    out["stages"]["CAD"] = _contributions(phi[0], x[0], base)
    out["stages"]["CAD"]["components"] = {k: float(v[0]) for k, v in model.cad.components(x).items()}
    for v in VESSELS:
        st = model.cond[v]
        b, ph = st.shap(x)
        out["stages"][f"{v}|CAD"] = _contributions(ph[0], x[0], b)
        out["stages"][f"{v}|CAD"]["components"] = {k: float(c[0]) for k, c in st.components(x).items()}
    for t in ("CAD",) + VESSELS:
        prob = float(p[t][0])
        q = model.conformal[t]
        pos = (1 - prob) <= q["1"]
        neg = prob <= q["0"]
        cset = "indeterminate" if pos and neg else "positive" if pos else "negative" if neg else "atypical"
        thr = float(model.thresholds.get(t, 0.5))
        rec = {"prob": prob, "conformal": cset, "threshold": thr, "flag": bool(prob >= thr)}
        if t != "CAD":
            rec["cond_prob"] = float(p[f"{t}|CAD"][0])
            rec["aha_segments"] = AHA_TERRITORY[t]
            rec["territory"] = TERRITORY_TEXT[t]
            comp_cad = out["stages"]["CAD"]["components"]
            comp_v = out["stages"][f"{t}|CAD"]["components"]
            lo = min(comp_cad["lr"], comp_cad["gbm"]) * min(comp_v["lr"], comp_v["gbm"])
            hi = max(comp_cad["lr"], comp_cad["gbm"]) * max(comp_v["lr"], comp_v["gbm"])
        else:
            comp = out["stages"]["CAD"]["components"]
            lo, hi = min(comp.values()), max(comp.values())
        rec["model_range"] = [float(min(lo, prob)), float(max(hi, prob))]
        out["targets"][t] = rec
    # 17-segment myocardial "territory at risk": each segment inherits its supplying vessel's probability
    seg = {}
    for v, segs in AHA_TERRITORY.items():
        for s in segs:
            seg[s] = {"vessel": v, "prob": out["targets"][v]["prob"]}
    out["segments"] = [{"segment": s, **seg[s]} for s in sorted(seg)]
    return out
