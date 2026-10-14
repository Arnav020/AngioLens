"""API contract tests against the shipped model artefact (models/angiolens.joblib)."""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ["GROQ_API_KEY"] = ""  # tests never call the network (dotenv will not override); copilot uses the template

from angiolens.api import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module")
def schema():
    return client.get("/api/schema").json()


def test_health():
    r = client.get("/api/health").json()
    assert r["status"] == "ok"


def test_schema_has_all_features(schema):
    assert len(schema["features"]) == 55
    assert set(schema["presets"]) == {"low", "intermediate", "high"}


def test_predict_presets_are_ordered(schema):
    p = {k: client.post("/api/predict", json={"values": v}).json()["targets"]["CAD"]["prob"]
         for k, v in schema["presets"].items()}
    assert p["low"] < p["intermediate"] < p["high"]


def test_predict_rejects_bad_input(schema):
    bad = {**schema["presets"]["low"], "age": 999}
    assert client.post("/api/predict", json={"values": bad}).status_code == 422
    missing = {k: v for k, v in schema["presets"]["low"].items() if k != "ldl"}
    assert client.post("/api/predict", json={"values": missing}).status_code == 422


def test_cohort_replay_is_out_of_fold():
    from angiolens.service import Service

    svc = Service()
    for i in (0, 57, 150, 302):
        r = client.get(f"/api/cohort/{i}").json()
        assert r["mode"] == "cohort_oof"
        assert np.isclose(r["targets"]["CAD"]["prob"], svc.oof["CAD"][i])
        assert r["patient_id"] not in [s["id"] for s in r["similar"]]
    assert client.get("/api/cohort/999").status_code == 422


def test_sensitivity_curve(schema):
    r = client.post("/api/sensitivity", json={"values": schema["presets"]["high"], "feature": "age"}).json()
    assert len(r["grid"]) == 25 and set(r["curves"]) == {"CAD", "LAD", "LCX", "RCA"}
    r = client.post("/api/sensitivity", json={"values": schema["presets"]["high"], "feature": "dm"}).json()
    assert r["grid"] == [0.0, 1.0]


def test_copilot_offline_fallback_is_grounded():
    r = client.post("/api/copilot", json={"patient_id": 5}).json()
    assert r["source"] == "template"
    assert r["grounding"]["passed"]


def test_pdf_report():
    r = client.post("/api/report", json={"patient_id": 5, "narrative": "**Test** narrative."})
    assert r.status_code == 200 and r.content[:4] == b"%PDF"


def test_model_card(schema):
    mc = client.get("/api/model-card").json()
    assert mc["targets"]["CAD"]["summary"]["angiolens"]["roc_auc"]["mean"] > 0.85
    assert "hparams" not in mc
