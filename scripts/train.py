"""Train, validate and export the AngioLens model.

    python scripts/train.py            # full run: 10x5 nested CV + baselines (~5-10 min on a laptop CPU)
    python scripts/train.py --quick    # 2x5 nested CV, no baselines (smoke test, ~1 min)

Outputs (models/):
    angiolens.joblib   final model (trained on all 303 patients) + repeat-0 fold models for honest cohort replay
    metrics.json       every number shown in the dashboard's Model Card and in docs/REPORT
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from angiolens import __version__  # noqa: E402
from angiolens.data import dataset_audit, load_dataset  # noqa: E402
from angiolens.evaluate import MODELS, nested_cv  # noqa: E402
from angiolens.explain import explain_row  # noqa: E402
from angiolens.metrics import bootstrap_ci, calibration_slope_intercept, curves, label_metrics, point_metrics  # noqa: E402
from angiolens.model import TARGETS, VESSELS, fit_model  # noqa: E402
from angiolens.schema import BY_ID, FEATURE_IDS, GROUPS  # noqa: E402

OUT = ROOT / "models"
CI_METRICS = ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "brier")


def summarise(y: np.ndarray, P: np.ndarray, with_ci: bool) -> dict:
    """P: (repeats, n) out-of-fold probabilities."""
    per_rep = [point_metrics(y, P[r]) for r in range(P.shape[0])]
    keys = per_rep[0].keys()
    out = {k: {"mean": float(np.mean([m[k] for m in per_rep])), "sd": float(np.std([m[k] for m in per_rep]))} for k in keys}
    if with_ci:
        cis = bootstrap_ci(y, [P[r] for r in range(P.shape[0])], CI_METRICS, n_boot=1000)
        for k, ci in cis.items():
            out[k]["ci95"] = ci
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    repeats = 2 if args.quick else args.repeats
    compare = not args.quick
    t0 = time.time()

    X_df, Y, raw = load_dataset()
    X = X_df.to_numpy(dtype=float)
    audit = dataset_audit(raw, Y)
    print(f"[data] {audit['n_patients']} patients, {audit['n_model_features']} features, "
          f"prevalence {', '.join(f'{k}={v:.2f}' for k, v in audit['prevalence'].items())}")

    print(f"[cv] repeated nested CV: {repeats} x 5 outer folds, inner 5-fold, compare={compare}")
    oof, conf, yhat, hparams, fold_models = nested_cv(X, Y, repeats=repeats, compare=compare, n_jobs=args.jobs, verbose=0)
    print(f"[cv] done in {time.time() - t0:.0f}s")

    results = {"models": MODELS if compare else {"angiolens": MODELS["angiolens"]}, "targets": {}}
    for t in TARGETS:
        y = Y[t].to_numpy()
        tres = {"prevalence": float(y.mean()), "summary": {}, "curves": {}}
        for m in oof:
            tres["summary"][m] = summarise(y, oof[m][t], with_ci=(m == "angiolens" or compare))
            tres["curves"][m] = {"roc": curves(y, oof[m][t][0])["roc"]}
        tres["curves"]["angiolens"] = curves(y, oof["angiolens"][t][0])
        per_rep = [label_metrics(y, yhat[t][r]) for r in range(repeats)]
        tres["tuned_threshold"] = {k: {"mean": float(np.mean([m[k] for m in per_rep])),
                                       "sd": float(np.std([m[k] for m in per_rep]))} for k in per_rep[0]}
        tres["calibration"] = calibration_slope_intercept(y, oof["angiolens"][t][0])
        # conformal: empirical coverage of the 90 % prediction sets on outer test folds
        C = conf[t]
        covered = ((C == 2) | (C == y[None, :])).mean()
        decided = (C == 0) | (C == 1)
        tres["conformal"] = {
            "target_coverage": 0.90,
            "empirical_coverage": float(covered),
            "indeterminate_rate": float((C == 2).mean()),
            "empty_rate": float((C == -1).mean()),
            "decided_rate": float(decided.mean()),
            "accuracy_when_decided": float((C[decided] == np.broadcast_to(y, C.shape)[decided]).mean()) if decided.any() else None,
        }
        results["targets"][t] = tres
        s = tres["summary"]["angiolens"]
        print(f"  {t:4s} AUC {s['roc_auc']['mean']:.3f}±{s['roc_auc']['sd']:.3f}  acc {s['accuracy']['mean']:.3f}  "
              f"F1 {s['f1']['mean']:.3f} (tuned thr: F1 {tres['tuned_threshold']['f1']['mean']:.3f}, "
              f"bal.acc {tres['tuned_threshold']['balanced_accuracy']['mean']:.3f})  Brier {s['brier']['mean']:.3f}  conformal cov {covered:.3f}")

    # coherence: how often would a direct per-vessel model claim P(vessel) > P(CAD)?
    if compare:
        coh = {}
        for m in ("direct_blend", "lr", "gbm", "rf", "angiolens"):
            viol = np.zeros_like(oof[m]["CAD"], dtype=bool)
            for v in VESSELS:
                viol |= oof[m][v] > oof[m]["CAD"] + 1e-9
            coh[m] = float(viol.mean())
        results["coherence_violation_rate"] = coh
        print("[coherence] P(vessel) > P(CAD) rate:", {k: round(v, 3) for k, v in coh.items()})

    results["hparams"] = hparams

    # ---------------- final model on all data ----------------
    print("[fit] final model on all patients")
    model, _ = fit_model(X, Y, seed=0)
    model.meta = {
        "version": __version__,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(X)),
        "thresholds": model.thresholds,
        "stages": {s.name: {"C": s.C, "w": s.w, "platt_a": s.a, "platt_b": s.b}
                   for s in [model.cad, *model.cond.values()]},
        "python": platform.python_version(),
    }

    # global importance (mean |SHAP|) of the deployed model, per stage and per clinical group
    importance = {}
    for st in [model.cad, *model.cond.values()]:
        rows = X if st.name == "CAD" else X[Y["CAD"].to_numpy() == 1]
        _, phi = st.shap(rows)
        mabs = np.abs(phi).mean(0)
        order = np.argsort(-mabs)
        importance[st.name] = {
            "features": [{"id": FEATURE_IDS[j], "label": BY_ID[FEATURE_IDS[j]].label,
                          "group": BY_ID[FEATURE_IDS[j]].group, "mean_abs_shap": float(mabs[j])} for j in order[:20]],
            "groups": {g: float(sum(mabs[j] for j, f in enumerate(FEATURE_IDS) if BY_ID[f].group == g)) for g in GROUPS},
            "nonzero_lr_coefs": int((st.lin.coef_ != 0).sum()),
        }
    results["importance"] = importance

    # honest cohort replay: patient i is explained by the repeat-0 fold model that never saw them
    fold_of = np.full(len(X), -1)
    for k, (te, _) in enumerate(fold_models):
        fold_of[te] = k
    assert (fold_of >= 0).all()
    oof_check = np.array([explain_row(fold_models[fold_of[i]][1], X[i])["targets"]["CAD"]["prob"] for i in range(len(X))])
    assert np.allclose(oof_check, oof["angiolens"]["CAD"][0]), "cohort replay does not reproduce OOF predictions"

    bundle = {
        "model": model,
        "fold_models": [m for _, m in fold_models],
        "fold_of": fold_of,
        "X": X,
        "Y": Y.to_numpy(),
        "oof_rep0": {t: oof["angiolens"][t][0] for t in TARGETS},
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT / "angiolens.joblib", compress=3)

    results["dataset"] = audit
    results["meta"] = {**model.meta, "repeats": repeats, "outer_folds": 5, "inner_folds": 5,
                       "runtime_s": round(time.time() - t0, 1), "quick": args.quick}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=1))
    size = (OUT / "angiolens.joblib").stat().st_size / 1e6
    print(f"[done] models/angiolens.joblib ({size:.1f} MB), models/metrics.json  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
