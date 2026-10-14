"""Grounded explanation copilot.

The language model never computes anything. It only rephrases a *fact sheet*
built deterministically from the model output, and every number in its answer
is checked against that fact sheet before it is shown ("number grounding").
If the LLM is unavailable or fails the check twice, a deterministic template
narrative is returned instead, so the feature degrades gracefully offline.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata

import requests

from .model import VESSELS

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
VESSEL_NAMES = {
    "LAD": "left anterior descending artery (LAD)",
    "LCX": "left circumflex artery (LCX)",
    "RCA": "right coronary artery (RCA)",
}
CONFORMAL_TEXT = {
    "positive": "confident: disease likely",
    "negative": "confident: disease unlikely",
    "indeterminate": "indeterminate: the model cannot decide at 90% confidence",
    "atypical": "atypical profile: unlike patients in the training data",
}
N_TRAIN = 303
DISCLAIMER = ("Decision-support and educational prototype only. Not a diagnosis and not a substitute "
              "for coronary angiography, CT coronary angiography or clinical judgement.")


def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def _fmt_val(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _describe(f: dict) -> str:
    if f["kind"] == "binary":
        return f"{f['label']}: {'present' if f['value'] >= 0.5 else 'absent'}"
    return f"{f['label']} = {_fmt_val(f['value'])}{(' ' + f['unit']) if f['unit'] else ''}"


def fact_sheet(expl: dict, top_k: int = 6) -> dict:
    """Compact, human-readable facts. The ONLY source of numbers for the LLM."""
    t = expl["targets"]
    facts = {
        "overall_CAD_probability": _pct(t["CAD"]["prob"]),
        "overall_CAD_confidence": CONFORMAL_TEXT[t["CAD"]["conformal"]],
        "vessels": {},
        "factors_raising_overall_risk": [],
        "factors_lowering_overall_risk": [],
        "abnormal_measurements": [],
        "training_cohort": f"{N_TRAIN} patients who underwent invasive coronary angiography (Z-Alizadeh Sani dataset)",
    }
    for v in VESSELS:
        facts["vessels"][v] = {
            "name": VESSEL_NAMES[v],
            "probability_of_significant_stenosis": _pct(t[v]["prob"]),
            "confidence": CONFORMAL_TEXT[t[v]["conformal"]],
            "myocardial_territory": t[v]["territory"],
            "top_localising_factors": [
                f"{_describe(f)}"
                f" ({'raises' if f['shap'] > 0 else 'lowers'} likelihood that this vessel is involved)"
                for f in expl["stages"][f"{v}|CAD"]["features"][:3]
            ],
        }
    for f in expl["stages"]["CAD"]["features"][: top_k * 2]:
        txt = _describe(f)
        key = "factors_raising_overall_risk" if f["shap"] > 0 else "factors_lowering_overall_risk"
        if len(facts[key]) < top_k:
            facts[key].append(txt)
    for f in expl["stages"]["CAD"]["features"]:
        if f["status"] in ("high", "low"):
            facts["abnormal_measurements"].append(
                f"{f['label']} {_fmt_val(f['value'])} {f['unit']} ({f['status']})".strip())
    most = max(VESSELS, key=lambda v: t[v]["prob"])
    facts["highest_risk_vessel"] = most
    return facts


# --------------------------------------------------------------------------- #
#  number grounding
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
# small integers that are safe in prose ("one", "17-segment", "3 vessels", list numbering)
_SAFE = {str(i) for i in range(0, 18)}


def _normalise_digits(text: str) -> str:
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() and not ch.isascii() else ch for ch in text)


def allowed_numbers(facts: dict) -> set[str]:
    nums = set(_NUM.findall(json.dumps(facts, ensure_ascii=False)))
    out = set(nums)
    for n in nums:
        if "." in n:
            out.add(n.rstrip("0").rstrip("."))
            out.add(str(round(float(n))))
    return out | _SAFE | {"50", "90", "100"}  # 50 % stenosis definition, 90 % conformal level


def check_numbers(text: str, facts: dict) -> tuple[bool, list[str], int]:
    found = _NUM.findall(_normalise_digits(text))
    allowed = allowed_numbers(facts)
    bad = [n for n in found if n not in allowed and n.rstrip("0").rstrip(".") not in allowed]
    return (not bad), bad, len(found)


# --------------------------------------------------------------------------- #
#  prompts
# --------------------------------------------------------------------------- #
AUDIENCE = {
    "clinician": ("a cardiologist. Be concise and technical (use LAD/LCX/RCA, conformal confidence, "
                  "feature attributions). 120-170 words. Use short markdown bullet points."),
    "patient": ("the patient, a non-specialist. Use plain, calm, warm language at a 12-year-old reading level, "
                "explain what the arteries do, avoid jargon, never alarm. 120-170 words."),
}
LANG = {"en": "English", "hi": "Hindi (Devanagari script; keep vessel abbreviations LAD/LCX/RCA in Latin letters)"}


def _system(audience: str, lang: str) -> str:
    return (
        "You explain the output of AngioLens, a machine-learning model that estimates coronary artery disease "
        f"risk from routine clinical data. Write for {AUDIENCE[audience]} Write in {LANG[lang]}.\n"
        "STRICT RULES:\n"
        "1. Use ONLY the facts in the JSON fact sheet. Do not add medical facts about this patient.\n"
        "2. Every number you write must appear verbatim in the fact sheet. Never compute, round differently, "
        "or invent numbers, risks, percentages or statistics.\n"
        "3. Feature effects are statistical associations learned by the model, not proven causes. Say so once.\n"
        "4. Do not diagnose and do not recommend specific drugs or procedures. You may suggest discussing "
        "results with a cardiologist.\n"
        "5. If a confidence says 'indeterminate', say clearly that the model is unsure for that item.\n"
        "6. End with one sentence stating this is a decision-support/educational tool, not a diagnosis."
    )


def _call_groq(messages: list[dict], model: str, timeout: float = 45.0) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not configured")
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 900,
              "reasoning_effort": "low"} if "gpt-oss" in model else
             {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 900},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def template_narrative(facts: dict, audience: str = "clinician") -> str:
    v = facts["vessels"]
    lines = [
        f"**Overall CAD probability: {facts['overall_CAD_probability']}** ({facts['overall_CAD_confidence']}).",
        "",
        "Per-vessel probability of ≥50% stenosis:",
    ]
    for k in VESSELS:
        lines.append(f"- **{k}** {v[k]['probability_of_significant_stenosis']} ({v[k]['confidence']})")
    if facts["factors_raising_overall_risk"]:
        lines += ["", "Main factors raising the estimate: " + "; ".join(facts["factors_raising_overall_risk"][:4]) + "."]
    if facts["factors_lowering_overall_risk"]:
        lines += ["Main factors lowering it: " + "; ".join(facts["factors_lowering_overall_risk"][:3]) + "."]
    lines += ["", f"These are statistical associations learned from {N_TRAIN} angiography patients, not causes.", "",
              f"_{DISCLAIMER}_"]
    return "\n".join(lines)


def narrate(expl: dict, audience: str = "clinician", lang: str = "en", question: str | None = None) -> dict:
    audience = audience if audience in AUDIENCE else "clinician"
    lang = lang if lang in LANG else "en"
    facts = fact_sheet(expl)
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    user = "FACT SHEET (JSON):\n" + json.dumps(facts, ensure_ascii=False, indent=1)
    if question:
        user += ("\n\nThe user asks: " + question.strip()[:500] +
                 "\nAnswer only from the fact sheet. If the fact sheet cannot answer it, say so plainly.")
    else:
        user += "\n\nWrite the explanation."
    messages = [{"role": "system", "content": _system(audience, lang)}, {"role": "user", "content": user}]
    t0 = time.time()
    attempts = []
    try:
        for _ in range(2):
            text = _call_groq(messages, model)
            ok, bad, n = check_numbers(text, facts)
            attempts.append({"ok": ok, "unverified_numbers": bad})
            if ok:
                return {"text": text, "source": "llm", "model": model, "grounding": {
                    "passed": True, "numbers_checked": n, "attempts": len(attempts)},
                    "facts": facts, "latency_s": round(time.time() - t0, 2)}
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"These numbers are not in the fact sheet: {', '.join(bad)}. "
                                                     "Rewrite using only numbers from the fact sheet."}]
        reason = "LLM answer contained numbers not present in the model output"
    except Exception as e:  # network / auth / quota problems -> deterministic fallback
        reason = f"LLM unavailable ({type(e).__name__})"
    return {"text": template_narrative(facts, audience), "source": "template", "model": None,
            "grounding": {"passed": True, "numbers_checked": None, "attempts": len(attempts), "fallback_reason": reason},
            "facts": facts, "latency_s": round(time.time() - t0, 2)}


def transcribe(audio: bytes, filename: str = "question.webm") -> str:
    """Speech -> text with Groq Whisper (voice questions to the copilot)."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not configured")
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (filename, audio)},
        data={"model": "whisper-large-v3-turbo", "response_format": "json"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()
