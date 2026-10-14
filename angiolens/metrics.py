"""Classification, calibration and clinical-utility metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

THRESHOLD = 0.5


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float = THRESHOLD) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(0.5 * (recall_score(y, yhat, zero_division=0) + (tn / max(tn + fp, 1)))),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
    }


def label_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """Threshold metrics from hard labels (used for inner-CV-tuned operating points)."""
    y = np.asarray(y, dtype=int)
    yhat = np.asarray(yhat, dtype=int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    rec = recall_score(y, yhat, zero_division=0)
    spec = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(0.5 * (rec + spec)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)),
    }


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calibration_slope_intercept(y, p) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def curves(y, p) -> dict:
    """Down-sampled curves for the dashboard (ROC, PR, reliability, decision curve)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    def thin(a, b, n=80):
        if len(a) <= n:
            return [[round(float(x), 4), round(float(z), 4)] for x, z in zip(a, b)]
        ix = np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
        return [[round(float(a[i]), 4), round(float(b[i]), 4)] for i in ix]

    # reliability diagram, quantile-free fixed bins
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    rel = []
    for b in range(10):
        m = idx == b
        if m.sum() >= 3:
            rel.append({"p_mean": float(p[m].mean()), "y_rate": float(y[m].mean()), "n": int(m.sum())})

    # decision-curve analysis (Vickers & Elkin 2006): net benefit vs threshold
    n = len(y)
    prev = y.mean()
    dca = []
    for pt in np.arange(0.05, 0.96, 0.05):
        yhat = p >= pt
        tp = (yhat & (y == 1)).sum()
        fp = (yhat & (y == 0)).sum()
        w = pt / (1 - pt)
        dca.append({
            "threshold": round(float(pt), 2),
            "model": float(tp / n - fp / n * w),
            "treat_all": float(prev - (1 - prev) * w),
            "treat_none": 0.0,
        })
    yhat = (p >= THRESHOLD).astype(int)
    cm = [[int(((yhat == 0) & (y == 0)).sum()), int(((yhat == 1) & (y == 0)).sum())],
          [int(((yhat == 0) & (y == 1)).sum()), int(((yhat == 1) & (y == 1)).sum())]]
    return {"roc": thin(fpr, tpr), "pr": thin(rec, prec), "reliability": rel, "dca": dca, "confusion": cm}


def bootstrap_ci(y: np.ndarray, ps: list[np.ndarray], keys, n_boot: int = 1000, seed: int = 0) -> dict:
    """95 % percentile CIs that mix patient resampling with repeat-to-repeat CV variation.

    Each bootstrap draw picks one CV repeat at random and resamples patients with replacement.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        r = rng.integers(len(ps))
        ix = rng.integers(0, n, n)
        if y[ix].min() == y[ix].max():
            continue
        m = point_metrics(y[ix], ps[r][ix])
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}
