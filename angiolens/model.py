"""AngioLens prediction model.

Design
------
* **Hierarchical, anatomically coherent targets.**  In this dataset CAD is
  defined as >=50 % stenosis in at least one of LAD / LCX / RCA, so we model

      P(CAD)                      "is there significant coronary disease?"
      P(vessel | CAD)             "given disease, is *this* artery involved?"
      P(vessel) = P(CAD) * P(vessel | CAD)

  which guarantees P(vessel) <= P(CAD) for every patient (a direct per-vessel
  model can output a 70 % LAD risk next to a 40 % CAD risk, which is
  clinically incoherent and confusing on a 3D heart).

* **Exactly-explainable calibrated ensemble per stage.**  Each stage is a
  blend, in log-odds space, of an L1 logistic regression and a shallow
  LightGBM model, followed by Platt scaling:

      z(x) = a * ( w * z_LR(x) + (1 - w) * z_GBM(x) ) + b

  Every operation is affine in log-odds, so SHAP values stay *exactly*
  additive: phi = a * (w * phi_LR + (1 - w) * phi_GBM).  (Linear SHAP for the
  LR, TreeSHAP for LightGBM.)  C, w, a, b are chosen by inner cross-validation.

* **Conformal prediction sets** (Mondrian / class-conditional, alpha = 0.10)
  turn each probability into {positive}, {negative} or {both} = "indeterminate",
  so the UI can say when the model does not know.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .schema import FEATURE_IDS

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")

VESSELS = ("LAD", "LCX", "RCA")
TARGETS = ("CAD",) + VESSELS
LOG_FEATURES = ("fbs", "cr", "tg", "bun", "esr", "wbc", "plt")  # right-skewed labs
C_GRID = (0.05, 0.1, 0.2, 0.4)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA = 0.10  # conformal miscoverage

LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.01, num_leaves=8, min_child_samples=8,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0,
    verbose=-1, n_jobs=1, deterministic=True, force_row_wise=True,
)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- #
#  Linear branch: log-transform skewed labs, standardise, L1 logistic regression
# --------------------------------------------------------------------------- #
@dataclass
class LinearBranch:
    C: float
    seed: int = 0
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        for j, fid in enumerate(FEATURE_IDS):
            if fid in LOG_FEATURES:
                X[:, j] = np.log1p(np.maximum(X[:, j], 0))
        return X

    def fit(self, X, y):
        Z = self._pre(X)
        self.mu_ = Z.mean(0)
        self.sd_ = Z.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        lr = LogisticRegression(C=self.C, penalty="l1", solver="liblinear", max_iter=5000, random_state=self.seed)
        lr.fit((Z - self.mu_) / self.sd_, y)
        self.coef_ = lr.coef_.ravel().copy()
        self.intercept_ = float(lr.intercept_[0])
        return self

    def scaled(self, X) -> np.ndarray:
        return (self._pre(X) - self.mu_) / self.sd_

    def decision(self, X) -> np.ndarray:
        return self.scaled(X) @ self.coef_ + self.intercept_

    def shap(self, X) -> tuple[float, np.ndarray]:
        # Exact interventional linear SHAP w.r.t. the training distribution
        # (scaled training features have mean 0, so E[z] = intercept).
        return self.intercept_, self.scaled(X) * self.coef_


@dataclass
class TreeBranch:
    seed: int = 0
    model: LGBMClassifier | None = None
    _explainer: object = field(default=None, repr=False)

    def fit(self, X, y):
        self.model = LGBMClassifier(random_state=self.seed, **LGBM_PARAMS)
        self.model.fit(pd.DataFrame(X, columns=FEATURE_IDS), y)
        self._explainer = None
        return self

    def decision(self, X) -> np.ndarray:
        return self.model.predict(pd.DataFrame(X, columns=FEATURE_IDS), raw_score=True)

    def shap(self, X) -> tuple[float, np.ndarray]:
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        sv = self._explainer.shap_values(pd.DataFrame(X, columns=FEATURE_IDS))
        if isinstance(sv, list):  # older shap returns [neg, pos]
            sv = sv[1]
        ev = self._explainer.expected_value
        ev = float(np.ravel(ev)[-1])
        return ev, np.asarray(sv)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_explainer"] = None
        return d


# --------------------------------------------------------------------------- #
#  One stage = calibrated log-odds blend of the two branches
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    name: str
    C: float
    w: float
    a: float = 1.0  # Platt slope
    b: float = 0.0  # Platt intercept
    lin: LinearBranch | None = None
    tree: TreeBranch | None = None
    inner_logloss: float = float("nan")

    def components(self, X) -> dict[str, np.ndarray]:
        zl = self.lin.decision(X)
        zt = self.tree.decision(X)
        return {"lr": sigmoid(self.a * zl + self.b), "gbm": sigmoid(self.a * zt + self.b)}

    def decision(self, X) -> np.ndarray:
        return self.a * (self.w * self.lin.decision(X) + (1 - self.w) * self.tree.decision(X)) + self.b

    def proba(self, X) -> np.ndarray:
        return sigmoid(self.decision(X))

    def shap(self, X) -> tuple[float, np.ndarray]:
        """Exact SHAP in calibrated log-odds space: base + phi.sum(1) == decision(X)."""
        bl, pl = self.lin.shap(X)
        bt, pt = self.tree.shap(X)
        base = self.a * (self.w * bl + (1 - self.w) * bt) + self.b
        phi = self.a * (self.w * pl + (1 - self.w) * pt)
        return float(base), phi


def _platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_stage(name: str, X: np.ndarray, y: np.ndarray, seed: int = 0, n_inner: int = 5,
              w_grid: tuple[float, ...] = W_GRID) -> tuple[Stage, np.ndarray]:
    """Select (C, w) and Platt calibration by inner CV, then refit on all rows.

    Returns the fitted stage and its inner out-of-fold calibrated probabilities
    (used to fit conformal thresholds without touching any outer test data).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    oof_lin = {C: np.zeros(len(y)) for C in C_GRID}
    oof_tree = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        if any(w > 0 for w in w_grid):
            for C in C_GRID:
                oof_lin[C][te] = LinearBranch(C, seed).fit(X[tr], y[tr]).decision(X[te])
        if any(w < 1 for w in w_grid):
            oof_tree[te] = TreeBranch(seed).fit(X[tr], y[tr]).decision(X[te])

    best = None
    c_grid = C_GRID if any(w > 0 for w in w_grid) else C_GRID[:1]
    for C in c_grid:
        for w in w_grid:
            if w == 0.0 and C != C_GRID[0]:
                continue  # pure-tree blend does not depend on C
            z = w * oof_lin[C] + (1 - w) * oof_tree
            a, b = _platt(z, y)
            p = sigmoid(a * z + b)
            ll = log_loss(y, p)
            if best is None or ll < best[0] - 1e-9:
                best = (ll, C, w, a, b, p)
    ll, C, w, a, b, p_oof = best
    stage = Stage(name=name, C=C, w=w, a=a, b=b, inner_logloss=ll)
    stage.lin = LinearBranch(C, seed).fit(X, y)
    stage.tree = TreeBranch(seed).fit(X, y)
    return stage, p_oof


# --------------------------------------------------------------------------- #
#  Conformal prediction (Mondrian, class-conditional)
# --------------------------------------------------------------------------- #
def conformal_thresholds(p: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> dict[str, float]:
    """Per-class thresholds on the nonconformity score s = 1 - p(true class)."""
    out = {}
    for cls, scores in ((1, 1 - p[y == 1]), (0, p[y == 0])):
        n = len(scores)
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        out[str(cls)] = float(np.sort(scores)[k - 1])
    return out


def conformal_set(p: np.ndarray, q: dict[str, float]) -> np.ndarray:
    """Return codes: 1 = {positive}, 0 = {negative}, 2 = {both} (indeterminate), -1 = {} (atypical)."""
    p = np.asarray(p, dtype=float)
    pos = (1 - p) <= q["1"]
    neg = p <= q["0"]
    return np.where(pos & neg, 2, np.where(pos, 1, np.where(neg, 0, -1)))


# --------------------------------------------------------------------------- #
#  Full hierarchical model
# --------------------------------------------------------------------------- #
@dataclass
class AngioLensModel:
    cad: Stage
    cond: dict[str, Stage]  # vessel -> P(vessel | CAD) stage
    conformal: dict[str, dict[str, float]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)  # operating points (Youden, inner CV)
    feature_ids: list[str] = field(default_factory=lambda: list(FEATURE_IDS))
    meta: dict = field(default_factory=dict)

    def predict(self, X) -> dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        p_cad = self.cad.proba(X)
        out = {"CAD": p_cad}
        for v in VESSELS:
            out[v] = p_cad * self.cond[v].proba(X)
            out[f"{v}|CAD"] = self.cond[v].proba(X)
        return out


def fit_model(X, Y: pd.DataFrame, seed: int = 0) -> tuple[AngioLensModel, dict[str, np.ndarray]]:
    """Fit the hierarchical model. Returns model + inner-OOF probabilities per target."""
    X = np.asarray(X, dtype=float)
    y_cad = Y["CAD"].to_numpy()
    cad, p_cad_oof = fit_stage("CAD", X, y_cad, seed)
    pos = y_cad == 1
    cond: dict[str, Stage] = {}
    oof = {"CAD": p_cad_oof}
    for v in VESSELS:
        stage, p_cond_pos = fit_stage(f"{v}|CAD", X[pos], Y[v].to_numpy()[pos], seed)
        cond[v] = stage
        # Inner-OOF estimate of P(vessel) for every row: OOF P(CAD) times the
        # conditional probability (OOF for diseased rows, model output otherwise:
        # non-diseased rows were never in the conditional training set).
        p_cond = stage.proba(X)
        p_cond[pos] = p_cond_pos
        oof[v] = p_cad_oof * p_cond
    model = AngioLensModel(cad=cad, cond=cond)
    model.conformal = {t: conformal_thresholds(oof[t], Y[t].to_numpy()) for t in TARGETS}
    model.thresholds = {t: youden_threshold(oof[t], Y[t].to_numpy()) for t in TARGETS}
    return model, oof


def youden_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Operating threshold maximising sensitivity + specificity - 1 on inner out-of-fold predictions."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j[1:])) + 1  # skip the +inf sentinel
    return float(np.clip(thr[k], 0.05, 0.95))
