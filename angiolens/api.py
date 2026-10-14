"""FastAPI backend: REST API + the pre-built web dashboard (web/dist)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .copilot import narrate, transcribe
from .data import encode_record
from .report import build_pdf
from .schema import BY_ID, schema_payload
from .service import Service

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "web" / "dist"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # optional dependency
    pass

app = FastAPI(title="AngioLens API", version=__version__,
              description="Coronary artery disease and per-vessel stenosis prediction with exact SHAP explanations. "
                          "Educational / decision-support prototype - not a medical device.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_svc: Service | None = None


def svc() -> Service:
    global _svc
    if _svc is None:
        _svc = Service()
    return _svc


class Patient(BaseModel):
    values: dict[str, float] = Field(..., description="feature id -> value (see GET /api/schema)")


class SensitivityReq(Patient):
    feature: str


class CopilotReq(BaseModel):
    values: dict[str, float] | None = None
    patient_id: int | None = None
    audience: str = "clinician"
    lang: str = "en"
    question: str | None = None


class EncodeReq(BaseModel):
    records: list[dict[str, str | float | int]] = Field(..., description="rows keyed by feature id or by raw UCI column name")


class ReportReq(BaseModel):
    values: dict[str, float] | None = None
    patient_id: int | None = None
    narrative: str | None = None


def _explain(values: dict | None, patient_id: int | None) -> dict:
    try:
        if patient_id is not None:
            return svc().cohort_patient(patient_id)
        if values is None:
            raise HTTPException(422, "provide 'values' or 'patient_id'")
        return svc().predict(values)
    except (KeyError, ValueError, IndexError) as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/health")
def health():
    s = svc()
    return {"status": "ok", "version": __version__, "model": s.model.meta.get("version"),
            "llm": bool(os.environ.get("GROQ_API_KEY"))}


@app.get("/api/schema")
def schema():
    s = svc()
    return {**schema_payload(), "defaults": s.defaults(), "presets": s.presets(),
            "thresholds": s.model.thresholds}


@app.get("/api/model-card")
def model_card():
    return svc().model_card()


@app.get("/api/cohort")
def cohort():
    return svc().cohort_list()


@app.get("/api/cohort/{patient_id}")
def cohort_patient(patient_id: int):
    return _explain(None, patient_id)


@app.post("/api/predict")
def predict(p: Patient):
    return _explain(p.values, None)


@app.post("/api/encode")
def encode(r: EncodeReq):
    """Turn uploaded rows (feature ids or raw UCI spreadsheet columns such as 'Sex'='Fmale', 'VHD'='mild') into model values."""
    try:
        return {"values": [encode_record(rec, svc().defaults()) for rec in r.records[:500]]}
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/sensitivity")
def sensitivity(r: SensitivityReq):
    if r.feature not in BY_ID:
        raise HTTPException(422, f"unknown feature {r.feature}")
    try:
        return svc().sensitivity(r.values, r.feature)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/copilot")
def copilot(r: CopilotReq):
    return narrate(_explain(r.values, r.patient_id), audience=r.audience, lang=r.lang, question=r.question)


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 15_000_000:
        raise HTTPException(413, "audio too large")
    try:
        return {"text": transcribe(data, file.filename or "question.webm")}
    except Exception as e:  # noqa: BLE001 - surfaced to the UI as a soft error
        raise HTTPException(503, f"transcription unavailable: {type(e).__name__}") from e


@app.post("/api/report")
def report(r: ReportReq):
    pdf = build_pdf(_explain(r.values, r.patient_id), narrative=r.narrative, model_card=svc().model_card())
    name = f"angiolens_patient_{r.patient_id}.pdf" if r.patient_id is not None else "angiolens_report.pdf"
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---------------- web dashboard ----------------
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
    if (DIST / "anatomy").exists():
        app.mount("/anatomy", StaticFiles(directory=DIST / "anatomy"), name="anatomy")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
