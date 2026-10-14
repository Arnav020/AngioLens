"""Model-serving layer used by the API (and tests). No web framework code here."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np

from .data import to_feature_vector
from .explain import explain_row
from .model import TARGETS
from .schema import BY_ID, FEATURE_IDS, FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "angiolens.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

PRESETS = {  # hand-written illustrative profiles (not real patients)
    "low": dict(age=42, sex_male=0, weight=62, height=162, bmi=23.6, dm=0, htn=0, current_smoker=0, ex_smoker=0,
                fh=0, obesity=0, crf=0, cva=0, airway_disease=0, thyroid_disease=0, chf=0, dlp=0, bp=118, pr=72,
                edema=0, weak_peripheral_pulse=0, lung_rales=0, systolic_murmur=0, diastolic_murmur=0,
                typical_chest_pain=0, dyspnea=0, function_class=0, atypical=1, nonanginal=0, lowth_ang=0, q_wave=0,
                st_elevation=0, st_depression=0, t_inversion=0, lvh=0, poor_r_progression=0, bbb_lbbb=0, bbb_rbbb=0,
                fbs=88, cr=0.8, tg=110, ldl=95, hdl=52, bun=13, esr=10, hb=13.4, k=4.2, na=140, wbc=6500, lymph=33,
                neut=58, plt=230, ef_tte=60, region_rwma=0, vhd=0),
}
PRESETS["high"] = {**PRESETS["low"], **dict(
    age=68, sex_male=1, weight=84, height=170, bmi=29.1, obesity=1, dm=1, htn=1, current_smoker=1, dlp=1, bp=150,
    pr=84, typical_chest_pain=1, atypical=0, dyspnea=1, function_class=2, st_depression=1, t_inversion=1,
    fbs=168, tg=210, ldl=150, hdl=33, esr=28, ef_tte=40, region_rwma=2, vhd=1)}
PRESETS["intermediate"] = {**PRESETS["low"], **dict(
    age=57, sex_male=1, weight=78, height=172, bmi=26.4, obesity=1, htn=1, dlp=1, bp=136, typical_chest_pain=1,
    atypical=0, fbs=104, tg=165, ldl=128, hdl=38, ef_tte=55)}


class Service:
    def __init__(self, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found - run `python scripts/train.py` first")
        b = joblib.load(model_path)
        self.model = b["model"]
        self.fold_models = b["fold_models"]
        self.fold_of = b["fold_of"]
        self.X = b["X"]
        self.Y = b["Y"]
        self.oof = b["oof_rep0"]
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self._knn_w = self._similarity_weights()

    # ---------------- helpers ----------------
    def _similarity_weights(self) -> np.ndarray:
        """Feature weights for 'similar patients' = how much each feature matters to the CAD stage."""
        _, phi = self.model.cad.shap(self.X)
        w = np.abs(phi).mean(0)
        return w / w.sum()

    @cached_property
    def _scaled_cohort(self) -> np.ndarray:
        return self.model.cad.lin.scaled(self.X)

    def similar(self, x: np.ndarray, k: int = 5, exclude: int | None = None) -> list[dict]:
        z = self.model.cad.lin.scaled(x.reshape(1, -1))[0]
        d = np.sqrt((((self._scaled_cohort - z) ** 2) * self._knn_w).sum(1))
        if exclude is not None:
            d[exclude] = np.inf
        order = np.argsort(d)[:k]
        scale = np.median(d[np.isfinite(d)])
        return [{
            "id": int(i), "distance": float(d[i]), "similarity": float(np.exp(-d[i] / scale)),
            "age": int(self.X[i, FEATURE_IDS.index("age")]), "sex": "M" if self.X[i, FEATURE_IDS.index("sex_male")] else "F",
            "truth": {t: int(self.Y[i, j]) for j, t in enumerate(TARGETS)},
        } for i in order]

    def _package(self, expl: dict, x: np.ndarray, mode: str, exclude: int | None = None) -> dict:
        expl["mode"] = mode
        expl["values"] = {fid: float(v) for fid, v in zip(FEATURE_IDS, x)}
        expl["similar"] = self.similar(x, exclude=exclude)
        expl["heart_rate"] = float(x[FEATURE_IDS.index("pr")])
        return expl

    # ---------------- public API ----------------
    def predict(self, values: dict) -> dict:
        x = to_feature_vector(values)
        return self._package(explain_row(self.model, x), x, mode="live")

    def cohort_list(self) -> list[dict]:
        out = []
        for i in range(len(self.X)):
            out.append({
                "id": i,
                "age": int(self.X[i, FEATURE_IDS.index("age")]),
                "sex": "M" if self.X[i, FEATURE_IDS.index("sex_male")] else "F",
                "truth": {t: int(self.Y[i, j]) for j, t in enumerate(TARGETS)},
                "oof": {t: round(float(self.oof[t][i]), 4) for t in TARGETS},
            })
        return out

    def cohort_patient(self, i: int) -> dict:
        if not 0 <= i < len(self.X):
            raise IndexError(f"patient {i} out of range")
        x = self.X[i]
        fm = self.fold_models[int(self.fold_of[i])]
        expl = self._package(explain_row(fm, x), x, mode="cohort_oof", exclude=i)
        expl["patient_id"] = i
        expl["truth"] = {t: int(self.Y[i, j]) for j, t in enumerate(TARGETS)}
        expl["fold"] = int(self.fold_of[i])
        return expl

    def sensitivity(self, values: dict, feature: str, n: int = 25) -> dict:
        """Individual conditional expectation: how this patient's risks move as one feature varies."""
        f = BY_ID[feature]
        x = to_feature_vector(values)
        if f.kind == "binary":
            grid = np.array([0.0, 1.0])
        elif f.kind == "ordinal":
            grid = np.arange(f.min, f.max + 1)
        else:
            lo, hi = np.percentile(self.X[:, FEATURE_IDS.index(feature)], [1, 99])
            grid = np.linspace(min(lo, x[FEATURE_IDS.index(feature)]), max(hi, x[FEATURE_IDS.index(feature)]), n)
        Xg = np.repeat(x[None], len(grid), 0)
        Xg[:, FEATURE_IDS.index(feature)] = grid
        p = self.model.predict(Xg)
        return {"feature": feature, "grid": grid.tolist(), "current": float(x[FEATURE_IDS.index(feature)]),
                "curves": {t: p[t].tolist() for t in TARGETS}}

    def presets(self) -> dict:
        return {k: {fid: float(v[fid]) for fid in FEATURE_IDS} for k, v in PRESETS.items()}

    def defaults(self) -> dict:
        med = np.median(self.X, axis=0)
        return {f.id: float(round(med[j] / f.step) * f.step) for j, f in enumerate(FEATURES)}

    def model_card(self) -> dict:
        m = dict(self.metrics)
        m.pop("hparams", None)
        return m


assert set(PRESETS["low"]) == set(FEATURE_IDS), set(FEATURE_IDS) ^ set(PRESETS["low"])
