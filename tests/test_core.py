"""Core guarantees of the AngioLens pipeline.

Run:  pytest -q
"""

import numpy as np
import pytest

from angiolens.copilot import check_numbers, fact_sheet, template_narrative
from angiolens.data import dataset_audit, load_dataset, to_feature_vector
from angiolens.explain import AHA_TERRITORY, explain_row
from angiolens.model import TARGETS, VESSELS, conformal_set, conformal_thresholds, fit_model, fit_stage
from angiolens.schema import FEATURE_IDS, FEATURES, LEAKAGE_COLUMNS


@pytest.fixture(scope="module")
def data():
    X, Y, raw = load_dataset()
    return X.to_numpy(float), Y, raw


@pytest.fixture(scope="module")
def small_model(data):
    X, Y, _ = data
    model, _ = fit_model(X, Y, seed=0)
    return model


# ---------------------------------------------------------------- data
def test_dataset_shape_and_no_missing(data):
    X, Y, raw = data
    assert X.shape == (303, len(FEATURE_IDS))
    assert not np.isnan(X).any()
    assert list(Y.columns) == ["CAD", "LAD", "LCX", "RCA"]


def test_no_target_leakage():
    raw_inputs = {f.raw for f in FEATURES}
    assert not raw_inputs & LEAKAGE_COLUMNS, "LAD/LCX/RCA/Cath must never be model inputs"


def test_cad_label_matches_vessels_except_documented_row(data):
    _, Y, raw = data
    audit = dataset_audit(raw, Y)
    assert len(audit["label_inconsistencies"]) == 1  # one source record: vessel stenotic but Cath = Normal


def test_feature_vector_validation():
    good = {f.id: (f.min + f.max) / 2 if f.kind != "binary" else 0 for f in FEATURES}
    assert to_feature_vector(good).shape == (len(FEATURES),)
    with pytest.raises(ValueError):
        to_feature_vector({**good, "age": 500})
    with pytest.raises(ValueError):
        to_feature_vector({**good, "dm": 0.5})
    with pytest.raises(KeyError):
        to_feature_vector({k: v for k, v in good.items() if k != "ldl"})


# ---------------------------------------------------------------- model guarantees
def test_shap_is_exactly_additive(data, small_model):
    X, _, _ = data
    for stage in [small_model.cad, *small_model.cond.values()]:
        base, phi = stage.shap(X[:40])
        np.testing.assert_allclose(base + phi.sum(1), stage.decision(X[:40]), atol=1e-6)


def test_hierarchical_predictions_are_coherent(data, small_model):
    X, _, _ = data
    p = small_model.predict(X)
    for v in VESSELS:
        assert np.all(p[v] <= p["CAD"] + 1e-12), f"P({v}) exceeded P(CAD)"
        assert np.all((p[v] >= 0) & (p[v] <= 1))


def test_training_is_reproducible(data):
    X, Y, _ = data
    idx = np.arange(0, 303, 2)
    a, _ = fit_stage("CAD", X[idx], Y["CAD"].to_numpy()[idx], seed=0)
    b, _ = fit_stage("CAD", X[idx], Y["CAD"].to_numpy()[idx], seed=0)
    np.testing.assert_array_equal(a.decision(X), b.decision(X))


def test_conformal_coverage_on_calibration_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.2, 2000), 0, 1)
    q = conformal_thresholds(p, y, alpha=0.1)
    s = conformal_set(p, q)
    covered = (s == 2) | (s == y)
    for cls in (0, 1):  # Mondrian = class-conditional coverage
        assert covered[y == cls].mean() >= 0.89


def test_explanation_structure(data, small_model):
    X, _, _ = data
    e = explain_row(small_model, X[0])
    assert set(e["targets"]) == set(TARGETS)
    assert {s["segment"] for s in e["segments"]} == set(range(1, 18))
    assert sorted(sum(AHA_TERRITORY.values(), [])) == list(range(1, 18))
    for t in TARGETS:
        lo, hi = e["targets"][t]["model_range"]
        assert lo <= e["targets"][t]["prob"] <= hi
    st = e["stages"]["CAD"]
    assert abs(st["base_logit"] + sum(f["shap"] for f in st["features"]) - st["logit"]) < 1e-6


# ---------------------------------------------------------------- copilot grounding
def test_number_grounding_rejects_invented_numbers(data, small_model):
    X, _, _ = data
    facts = fact_sheet(explain_row(small_model, X[0]))
    cad = facts["overall_CAD_probability"]
    ok, bad, _ = check_numbers(f"The overall probability is {cad}.", facts)
    assert ok and not bad
    ok, bad, _ = check_numbers("Your 10-year risk is 37.5% and LDL target is 55.", facts)
    assert not ok and "37.5" in bad
    # Devanagari digits are normalised before checking
    ok, bad, _ = check_numbers("जोखिम ९९९% है", facts)
    assert not ok and "999" in bad


def test_template_narrative_is_grounded(data, small_model):
    X, _, _ = data
    facts = fact_sheet(explain_row(small_model, X[3]))
    ok, bad, _ = check_numbers(template_narrative(facts), facts)
    assert ok, bad
