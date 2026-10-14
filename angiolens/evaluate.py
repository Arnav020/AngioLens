"""Repeated nested cross-validation.

Outer loop: 10 x repeated stratified 5-fold (50 train/test splits).  Every
hyper-parameter (C, blend weight w), the Platt calibration and the conformal
thresholds are chosen *inside* each outer training split by an inner 5-fold
CV, so outer test patients never influence any modelling decision.

All competing models are evaluated on the identical outer splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .model import TARGETS, VESSELS, conformal_set, fit_model, fit_stage

MODELS = {
    "angiolens": "AngioLens: hierarchical calibrated LR+GBM blend (ours)",
    "direct_blend": "Direct per-target calibrated LR+GBM blend",
    "lr": "L1 logistic regression (tuned C, calibrated)",
    "gbm": "LightGBM (calibrated)",
    "rf": "Random forest (500 trees)",
}


def _strata(Y: pd.DataFrame) -> np.ndarray:
    # stratify on CAD x number-of-diseased-vessels so every fold sees 0/1/2/3-vessel disease
    return (Y["CAD"].astype(str) + "_" + Y[list(VESSELS)].sum(1).astype(str)).to_numpy()


def _one_fold(X: np.ndarray, Y: pd.DataFrame, tr: np.ndarray, te: np.ndarray, seed: int, compare: bool) -> dict:
    Ytr = Y.iloc[tr]
    model, _ = fit_model(X[tr], Ytr, seed=seed)
    pred = model.predict(X[te])
    out = {"te": te, "pred": {"angiolens": {t: pred[t] for t in TARGETS}}}
    out["conformal"] = {t: conformal_set(pred[t], model.conformal[t]) for t in TARGETS}
    out["yhat_tuned"] = {t: (pred[t] >= model.thresholds[t]).astype(int) for t in TARGETS}
    out["hparams"] = {"CAD": (model.cad.C, model.cad.w),
                      **{f"{v}|CAD": (model.cond[v].C, model.cond[v].w) for v in VESSELS}}
    if compare:
        for key, wg in (("direct_blend", None), ("lr", (1.0,)), ("gbm", (0.0,))):
            out["pred"][key] = {}
            for t in TARGETS:
                if key == "direct_blend" and t == "CAD":  # identical to the angiolens CAD stage
                    out["pred"][key][t] = pred["CAD"]
                    continue
                kw = {} if wg is None else {"w_grid": wg}
                st, _ = fit_stage(t, X[tr], Ytr[t].to_numpy(), seed=seed, **kw)
                out["pred"][key][t] = st.proba(X[te])
        out["pred"]["rf"] = {}
        for t in TARGETS:
            rf = RandomForestClassifier(500, min_samples_leaf=2, n_jobs=1, random_state=seed)
            rf.fit(X[tr], Ytr[t].to_numpy())
            out["pred"]["rf"][t] = rf.predict_proba(X[te])[:, 1]
    return out, model


def nested_cv(X: np.ndarray, Y: pd.DataFrame, repeats: int = 10, folds: int = 5, compare: bool = True,
              n_jobs: int = -1, verbose: int = 0):
    """Returns (oof, conformal, yhat_tuned, hparams, fold_models_rep0).

    oof[model][target] -> array (repeats, n) of out-of-fold probabilities.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    strata = _strata(Y)
    jobs = []
    for r in range(repeats):
        cv = StratifiedKFold(folds, shuffle=True, random_state=1000 + r)
        for k, (tr, te) in enumerate(cv.split(X, strata)):
            jobs.append((r, k, tr, te))
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one_fold)(X, Y, tr, te, seed=r, compare=compare) for r, k, tr, te in jobs
    )
    names = list(MODELS) if compare else ["angiolens"]
    oof = {m: {t: np.full((repeats, n), np.nan) for t in TARGETS} for m in names}
    conf = {t: np.full((repeats, n), -9, dtype=int) for t in TARGETS}
    yhat = {t: np.full((repeats, n), -9, dtype=int) for t in TARGETS}
    hparams = []
    fold_models = []
    for (r, k, _tr, te), (res, model) in zip(jobs, results):
        for m in names:
            for t in TARGETS:
                oof[m][t][r, te] = res["pred"][m][t]
        for t in TARGETS:
            conf[t][r, te] = res["conformal"][t]
            yhat[t][r, te] = res["yhat_tuned"][t]
        hparams.append({"repeat": r, "fold": k, **{s: {"C": c, "w": w} for s, (c, w) in res["hparams"].items()}})
        if r == 0:
            fold_models.append((te, model))
    return oof, conf, yhat, hparams, fold_models
