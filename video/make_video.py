"""Fully automated demo video:  narration -> neural TTS -> scripted screen capture -> subtitles -> MP4.

    python run.py --no-browser            # in another terminal: the app must be running on :8000
    python video/make_video.py            # -> video/out/AngioLens_demo.mp4 (+ .srt)

Pipeline
  1. Each segment's narration is synthesised with Microsoft Edge neural TTS (edge-tts), which also returns
     word-level timestamps.
  2. A headless Chromium (GPU via ANGLE/D3D11 when available) drives the real app. Actions are attached to
     *cue words* in the narration, so a click happens exactly when the narrator says it. Frames are captured
     with the DevTools screencast at ~30 fps.
  3. Frames + audio are muxed per segment with ffmpeg, segments are concatenated, and English subtitles
     built from the word timestamps are burned in (and exported as .srt for YouTube).
Options:  --only intro,dashboard   re-record some segments   --skip-tts   reuse cached audio
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import edge_tts
from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
OUT = ROOT / "out"
APP = "http://127.0.0.1:8000/"
VOICE = "en-US-AndrewMultilingualNeural"
W, H, FPS = 1920, 1080, 30
TAIL = 0.7  # seconds of picture after the narration ends
SUB_FIX = {"Shap": "SHAP"}  # spoken spelling -> displayed spelling

Action = Callable[[Page], Awaitable[None]]


@dataclass
class Segment:
    id: str
    text: str
    page: str  # "app" or "slide:<name>"
    setup: Action | None = None
    cues: list[tuple[str, Action]] = field(default_factory=list)


# ---------------------------------------------------------------- cursor + helpers
CURSOR_JS = """
(() => {
  if (window.__cursor) return;
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;' +
    'border:2px solid rgba(255,255,255,.95);background:rgba(233,162,59,.35);box-shadow:0 0 14px rgba(233,162,59,.7);' +
    'z-index:2147483647;pointer-events:none;transition:transform .12s;transform:translate(-100px,-100px)';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(c));
  if (document.body) document.body.appendChild(c);
  window.addEventListener('mousemove', e => { c.style.transform = `translate(${e.clientX}px,${e.clientY}px)`; }, true);
  window.addEventListener('mousedown', e => {
    const r = document.createElement('div');
    r.style.cssText = `position:fixed;left:${e.clientX - 24}px;top:${e.clientY - 24}px;width:48px;height:48px;border-radius:50%;` +
      'border:2px solid rgba(233,162,59,.9);z-index:2147483646;pointer-events:none;transition:all .45s ease-out;opacity:1';
    document.body.appendChild(r);
    requestAnimationFrame(() => { r.style.transform = 'scale(1.8)'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 600);
  }, true);
  window.__cursor = c;
})();
"""

_mouse = {"x": W / 2, "y": H / 2}


async def move(page: Page, x: float, y: float, steps: int = 28):
    await page.mouse.move(x, y, steps=steps)
    _mouse.update(x=x, y=y)


async def center(page: Page, selector: str | None = None, *, text: str | None = None, nth: int = 0):
    loc = page.get_by_text(text, exact=True).nth(nth) if text else page.locator(selector).nth(nth)
    await loc.scroll_into_view_if_needed(timeout=3000)
    b = await loc.bounding_box()
    if not b:
        raise RuntimeError(f"no box for {selector or text}")
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


async def hover(page: Page, selector: str | None = None, *, text: str | None = None, nth: int = 0, steps: int = 28):
    x, y = await center(page, selector, text=text, nth=nth)
    await move(page, x, y, steps)


async def click(page: Page, selector: str | None = None, *, text: str | None = None, nth: int = 0, steps: int = 28):
    await hover(page, selector, text=text, nth=nth, steps=steps)
    await page.mouse.down()
    await asyncio.sleep(0.06)
    await page.mouse.up()


async def project(page: Page, x: float, y: float, z: float):
    return await page.evaluate(f"window.__angiolensProject({x},{y},{z})")


async def orbit(page: Page, dx: float, dy: float = 0, steps: int = 60):
    cx, cy = 790, 470
    await move(page, cx, cy, 20)
    await page.mouse.down()
    await page.mouse.move(cx + dx, cy + dy, steps=steps)
    await page.mouse.up()


def act(*fns: Action) -> Action:
    async def run(page: Page):
        for f in fns:
            await f(page)
    return run


def wait(s: float) -> Action:
    async def run(_page: Page):
        await asyncio.sleep(s)
    return run


# ---------------------------------------------------------------- the script
SEGMENTS: list[Segment] = [
    Segment("intro", page="slide:title", text=(
        "Coronary artery disease is the world's leading cause of death. A risk score can tell a doctor how likely "
        "disease is, but not where it is, and finding that out still takes an invasive angiogram. "
        "This is AngioLens, our project for Track A. It is a virtual angiogram. From routine clinical data it predicts "
        "coronary artery disease and stenosis in each of the three main arteries, explains every number exactly, "
        "and maps the result onto a real 3D heart.")),

    Segment("dashboard", page="app", text=(
        "Every session opens with a safety notice. Behind it is the dashboard, showing patient ninety-nine from the real "
        "Z-Alizadeh Sani cohort of three hundred and three patients. The green tag matters. This patient is replayed by "
        "the cross-validation model that never saw them in training, so what you see is honest, out-of-sample behaviour. "
        "The model estimates a seventy-four percent probability of coronary disease and points to the LAD, at sixty-five "
        "percent, while confidently ruling out the circumflex. The real angiogram agrees: single-vessel LAD disease."),
        setup=None,
        cues=[
            ("safety", act(wait(1.2), lambda p: click(p, text="I understand, continue"))),
            ("green", lambda p: hover(p, ".pill.honest")),
            ("seventy", lambda p: hover(p, ".cad-hero svg")),
            ("points", lambda p: hover(p, ".vcard", nth=0)),
            ("ruling", lambda p: hover(p, ".vcard .badge", nth=1)),
            ("agrees", lambda p: hover(p, ".vcard .truth", nth=0)),
        ]),

    Segment("anatomy", page="app", text=(
        "The heart is real anatomy from BodyParts3D. The LAD, circumflex and right coronary trees are selected by their "
        "anatomical identifiers and sit on a corrected epicardial surface. Each artery is coloured by its predicted "
        "probability, and contrast boluses flow from the ostium toward the apex. In high-risk vessels the flow slows and "
        "flickers, like delayed filling on an angiogram. Let me click the LAD to isolate it. The inspector shows its "
        "probability, its conformal confidence and the factors that localise disease there, led by the reduced "
        "ejection fraction."),
        cues=[
            ("heart", lambda p: click(p, text="Anterior")),
            ("coloured", lambda p: orbit(p, 160, 10)),
            ("flickers", lambda p: orbit(p, -150, -6)),
            ("click", lambda p: click(p, ".vlabel", nth=0)),
            ("inspector", lambda p: hover(p, ".inspector")),
            ("ejection", lambda p: hover(p, ".inspector .small", nth=4)),
        ]),

    Segment("territory", page="app", text=(
        "The tinted muscle is the territory each artery supplies, linked to the American Heart Association "
        "seventeen-segment bull's-eye. Hover a segment and its artery lights up in 3D. Click any point on the heart muscle, "
        "and AngioLens shows which arteries perfuse it. The heart even beats at the patient's recorded pulse, fifty beats "
        "per minute here, next to a schematic ECG built from their actual findings, including the T-wave inversion."),
        setup=lambda p: click(p, ".inspector .x"),
        cues=[
            ("seventeen", lambda p: hover(p, "svg[aria-label=\"AHA 17-segment bull's-eye map\"] path", nth=2)),
            ("lights", lambda p: hover(p, "svg[aria-label=\"AHA 17-segment bull's-eye map\"] path", nth=7)),
            ("Click", act(lambda p: move(p, 1000, 300, 10), lambda p: _click_lv(p))),
            ("beats", lambda p: hover(p, ".ecg .hr")),
            ("including", lambda p: hover(p, ".ecg .flags .f.on")),
        ]),

    Segment("explain", page="app", text=(
        "Why these numbers? AngioLens is hierarchical. Stage one asks whether there is coronary disease. Stage two asks "
        "which artery, given disease. The vessel probability is the product, so it can never exceed the overall risk, a "
        "rule that independent per-vessel models break for about one patient in five. Each stage blends a sparse logistic "
        "regression with gradient boosting in log-odds, then calibrates it. Because every step is affine, the Shap values "
        "add up exactly to the probability on screen. For this patient, typical chest pain and T-wave inversion push the "
        "risk up, while the absence of hypertension pulls it down. Switching to the LAD stage, the low ejection fraction "
        "is what points to this artery. Below, every measurement appears against its reference range, with its exact "
        "contribution."),
        setup=act(lambda p: click(p, ".inspector .x"), lambda p: click(p, text="Explain")),
        cues=[
            ("hierarchical", lambda p: hover(p, ".stage-pick, .col.right .seg", nth=0)),
            ("Shap", lambda p: hover(p, "svg[aria-label='SHAP waterfall'] circle", nth=1)),
            ("typical", lambda p: hover(p, "svg[aria-label='SHAP waterfall'] g", nth=10)),
            ("Switching", lambda p: click(p, text="Where: LAD")),
            ("Below", lambda p: _scroll_right(p, 650)),
        ]),

    Segment("input", page="app", text=(
        "New patients can be entered in three ways. Pick a preset, edit any of the fifty-five fields, or import a JSON or "
        "CSV file, even a raw row from the original UCI spreadsheet. This one is a seventy-two year old man with "
        "three-vessel disease. Every change re-predicts in about forty milliseconds, so the arteries, the territories and "
        "the explanations update live. Watch the heart as I remove his typical chest pain."),
        setup=act(lambda p: click(p, text="Risk"), lambda p: click(p, text="Torso")),
        cues=[
            ("three", lambda p: click(p, ".source-switch button", nth=2)),
            ("preset", lambda p: click(p, text="High")),
            ("import", lambda p: _import_csv(p)),
            ("Every", lambda p: hover(p, ".vcard", nth=0)),
            ("Watch", lambda p: _open_group_and_toggle(p)),
        ]),

    Segment("whatif", page="app", text=(
        "The What-if tab probes the model, here on patient five. Counterfactual probes show what drives an estimate. "
        "Risk-factor targets reveal something important. Setting LDL to seventy slightly raises the modelled risk, as it "
        "does for most patients in this cohort. That is confounding by treatment: LDL is no higher in diseased patients, "
        "because many are already on statins. So AngioLens flags these as associations, not treatment effects. "
        "Sensitivity curves show how all four risks move with any single feature."),
        setup=act(lambda p: _close_inspector(p), lambda p: click(p, ".source-switch button", nth=0),
                  lambda p: _pick_patient(p, 5), lambda p: _close_inspector(p),
                  lambda p: click(p, text="What-if"), wait(1.5)),
        cues=[
            ("Counterfactual", lambda p: hover(p, ".col.right .sim-row", nth=0)),
            ("Setting", lambda p: _hover_text(p, "LDL 70 mg/dL")),
            ("flags", lambda p: hover(p, ".col.right .badge.indeterminate")),
            ("Sensitivity", act(lambda p: _select_feature(p, "ef_tte"), lambda p: _hover_chart(p))),
        ]),

    Segment("honesty", page="app", text=(
        "We also show where the model fails. The Missed filter lists patients it got wrong out-of-fold, like patient "
        "twenty: a true coronary case with no typical chest pain, rated at just seven percent. Localising disease from "
        "routine data is genuinely hard, so conformal prediction marks uncertain vessel calls as indeterminate instead of "
        "painting a falsely confident heart, and its ninety percent coverage holds out-of-sample. When the evidence is "
        "strong, as in patient fifteen with three-vessel disease, every artery lights up with a confident call."),
        setup=act(lambda p: click(p, text="Risk"), lambda p: click(p, ".source-switch button", nth=0), lambda p: _clear_search(p)),
        cues=[
            ("Missed", lambda p: click(p, text="Missed")),
            ("twenty", lambda p: click(p, text="#20")),
            ("conformal", lambda p: hover(p, ".vcard .badge", nth=1)),
            ("strong", act(lambda p: click(p, ".col.left .seg button", nth=2), lambda p: _pick_patient(p, 15))),
            ("lights", lambda p: hover(p, ".vcard", nth=2)),
        ]),

    Segment("copilot", page="app", text=(
        "The copilot turns all of this into language, for clinicians or patients, in English or Hindi, by text or by "
        "voice. The language model never predicts anything. It rephrases a fact sheet produced by the model, and every "
        "number it writes is automatically checked against that sheet. Here, every one is verified. Questions that the "
        "model output cannot answer, such as drug dosing, are declined. And one click exports a one-page PDF report."),
        setup=lambda p: click(p, text="Copilot"),
        cues=[
            ("patients", lambda p: click(p, text="Patient")),
            ("Hindi", lambda p: click(p, text="हिन्दी")),
            ("voice", lambda p: click(p, text="Explain this result")),
            ("verified", lambda p: hover(p, ".ground")),
            ("Questions", lambda p: _ask(p, "What statin dose should this patient take?")),
            ("PDF", lambda p: hover(p, text="↓ PDF report")),
        ]),

    Segment("modelcard", page="app", text=(
        "Finally, the model card. Everything is validated with ten-times repeated, nested, five-fold cross-validation, so "
        "hyper-parameters, calibration, thresholds and conformal sets never touch the test fold. Overall coronary disease "
        "reaches an AUC of 0.93, the LAD 0.84, and the circumflex and right coronary artery about 0.71 to 0.74. Against four "
        "strong baselines, AngioLens matches the best discrimination, has the best or tied-best Brier score, and is the "
        "only model that is always anatomically coherent. Decision-curve analysis shows clinical net benefit across "
        "referral thresholds."),
        setup=lambda p: click(p, text="Model card & validation"),
        cues=[
            ("reaches", lambda p: hover(p, ".kpi", nth=0)),
            ("Against", lambda p: _scroll_model(p, 520)),
            ("only", lambda p: hover(p, ".cmp tr", nth=1)),
            ("Decision", lambda p: _scroll_model(p, 1450)),
        ]),

    Segment("architecture", page="slide:architecture", text=(
        "Under the hood, a single feature registry drives training, validation and the interface. A FastAPI backend "
        "returns predictions with exact Shap for all four stages in about forty milliseconds, and a React Three Fiber "
        "front end with custom shaders runs at sixty frames per second on integrated laptop graphics. Twenty automated "
        "tests guard against leakage and check exact additivity, coherence and reproducibility. The whole pipeline, from "
        "training to the 3D asset, the report and even this video, is scripted and reproducible.")),

    Segment("closing", page="slide:closing", text=(
        "AngioLens is a decision-support and educational prototype, not a diagnosis. It is trained on a single centre, "
        "uses generic anatomy, and shows associations rather than causes. But it shows how a prediction can become a "
        "picture that a clinician can question. Thank you, from Random Forest Rangers.")),
]


# ---------------------------------------------------------------- scene helpers
async def _click_lv(p: Page):
    x, y = await project(p, 0.42, -0.22, 0.25)
    await move(p, x, y, 30)
    await p.mouse.down()
    await asyncio.sleep(0.06)
    await p.mouse.up()


async def _scroll_right(p: Page, dy: int):
    x, y = await center(p, ".col.right .scroll")
    await move(p, x, y, 18)
    for _ in range(8):
        await p.mouse.wheel(0, dy / 8)
        await asyncio.sleep(0.06)


async def _scroll_model(p: Page, top: int):
    await p.evaluate(f"document.querySelector('.model-page').scrollTo({{top:{top}, behavior:'smooth'}})")
    await asyncio.sleep(0.9)


async def _import_csv(p: Page):
    await hover(p, "label.btn")
    await p.set_input_files("input[type=file]", str(ROOT / "assets" / "sample_patient_uci.csv"))


async def _open_group_and_toggle(p: Page):
    await click(p, "#typical_chest_pain")


async def _hover_text(p: Page, t: str):
    await hover(p, text=t)


async def _select_feature(p: Page, fid: str):
    await hover(p, "select.search")
    await p.select_option("select.search", fid)
    await asyncio.sleep(0.8)


async def _hover_chart(p: Page):
    box = await p.locator(".col.right svg[role=img]").last.bounding_box()
    if box:
        for f in (0.25, 0.5, 0.75):
            await move(p, box["x"] + box["width"] * f, box["y"] + box["height"] * 0.45, 22)
            await asyncio.sleep(0.5)


async def _close_inspector(p: Page):
    if await p.locator(".inspector .x").count():
        await p.click(".inspector .x")


async def _clear_search(p: Page):
    await p.fill("input.search", "")


async def _pick_patient(p: Page, pid: int):
    await click(p, "input.search")
    await p.fill("input.search", "")
    await p.keyboard.type(str(pid), delay=90)
    await asyncio.sleep(0.3)
    await click(p, text=f"#{pid}")


async def _ask(p: Page, q: str):
    await click(p, ".ask input")
    await p.keyboard.type(q, delay=28)
    await click(p, text="Ask")


# ---------------------------------------------------------------- TTS
_TOKEN = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _TOKEN.sub("", s.lower())


async def synth(seg: Segment) -> dict:
    d = BUILD / "audio"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((VOICE + seg.text).encode()).hexdigest()[:10]
    mp3, js = d / f"{seg.id}_{key}.mp3", d / f"{seg.id}_{key}.json"
    if not mp3.exists() or not js.exists():
        com = edge_tts.Communicate(seg.text, VOICE, rate="+4%", boundary="WordBoundary")
        words = []
        with open(mp3, "wb") as f:
            async for ch in com.stream():
                if ch["type"] == "audio":
                    f.write(ch["data"])
                elif ch["type"] == "WordBoundary":
                    words.append({"t": ch["offset"] / 1e7, "d": ch["duration"] / 1e7, "w": ch["text"]})
        js.write_text(json.dumps(words))
    words = json.loads(js.read_text())
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)]).decode().strip())
    return {"mp3": mp3, "words": words, "dur": dur}


def align_display(text: str, words: list[dict]) -> list[dict]:
    """Attach the original (punctuated) tokens of `text` to the TTS word timings."""
    tokens = text.split()
    out, j, rem = [], 0, ""
    for w in words:
        nw = _norm(w["w"])
        if not rem and j < len(tokens):
            rem = _norm(tokens[j])
            disp = tokens[j]
        else:
            disp = ""
        if nw and rem.startswith(nw):
            rem = rem[len(nw):]
        else:
            rem = ""
        if disp:
            out.append({**w, "disp": SUB_FIX.get(disp.strip(".,;:!?"), disp.strip(".,;:!?")) + disp[len(disp.rstrip(".,;:!?")):]})
        elif out:
            out[-1]["d"] = w["t"] + w["d"] - out[-1]["t"]
        if not rem:
            j += 1
    for t in tokens[j:]:  # never drop text
        out.append({"t": out[-1]["t"] + out[-1]["d"], "d": 0.3, "w": t, "disp": t})
    return out


def cue_time(words: list[dict], cue: str, after: float) -> float:
    c = _norm(cue)
    for w in words:
        if w["t"] >= after - 1e-6 and _norm(w["w"]).startswith(c[: max(3, len(c))]) or (
                w["t"] >= after - 1e-6 and c and _norm(w["w"]) == c):
            return w["t"]
    raise KeyError(f"cue '{cue}' not found after {after:.2f}s")


# ---------------------------------------------------------------- capture
async def record(page: Page, seg: Segment, audio: dict) -> Path:
    fdir = BUILD / "frames" / seg.id
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    cdp = await page.context.new_cdp_session(page)
    frames: list[tuple[float, str]] = []

    def on_frame(ev):
        name = f"{len(frames):06d}.jpg"
        (fdir / name).write_bytes(base64.b64decode(ev["data"]))
        frames.append((time.time(), name))  # arrival clock: same clock as the cue scheduler
        asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))

    cdp.on("Page.screencastFrame", on_frame)
    await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 86, "maxWidth": W, "maxHeight": H,
                                           "everyNthFrame": 2})
    t0 = time.time()
    # schedule cue actions against the narration clock (slightly early so the effect lands on the word)
    last = 0.0
    for cue, fn in seg.cues:
        try:
            t = cue_time(audio["words"], cue, last)
        except KeyError as e:
            print("   !", e)
            continue
        last = t + 0.01
        delay = t0 + max(0.0, t - 0.35) - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            await asyncio.wait_for(fn(page), timeout=8)
        except Exception as e:  # keep recording even if one action misses
            print(f"   ! action '{cue}' failed: {type(e).__name__}: {str(e)[:120]}")
    remaining = t0 + audio["dur"] + TAIL - time.time()
    if remaining > 0:
        await asyncio.sleep(remaining)
    await cdp.send("Page.stopScreencast")
    await asyncio.sleep(0.2)
    await cdp.detach()
    if not frames:
        raise RuntimeError(f"no frames captured for {seg.id}")
    # constant-frame-rate concat list (each frame shown until the next one arrived)
    lst = fdir / "frames.txt"
    with open(lst, "w") as f:
        for k, (ts, name) in enumerate(frames):
            nxt = frames[k + 1][0] if k + 1 < len(frames) else t0 + audio["dur"] + TAIL
            start = max(ts, t0) if k == 0 else ts
            f.write(f"file '{name}'\nduration {max(0.001, nxt - start):.4f}\n")
        f.write(f"file '{frames[-1][1]}'\n")
    lead = max(0.0, frames[0][0] - t0)  # time before the first frame arrived
    clip = BUILD / "clips" / f"{seg.id}.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    total = audio["dur"] + TAIL
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-itsoffset", "0", "-i", str(audio["mp3"]),
        "-filter_complex", f"[0:v]tpad=start_duration={lead:.3f}:start_mode=clone,fps={FPS},scale={W}:{H},format=yuv420p[v];"
                           f"[1:a]apad,atrim=0:{total:.3f},asetpts=N/SR/TB[a]",
        "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(clip),
    ], check=True)
    print(f"   {seg.id}: {len(frames)} frames, {total:.1f}s -> {clip.name}")
    return clip


# ---------------------------------------------------------------- subtitles
def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_srt(timeline: list[tuple[float, list[dict]]], max_chars: int = 84, max_dur: float = 5.5) -> str:
    cues = []
    for offset, words in timeline:
        cur: list[dict] = []
        for w in words:
            cand = " ".join(x["disp"] for x in cur + [w])
            too_long = len(cand) > max_chars or (cur and w["t"] + w["d"] - cur[0]["t"] > max_dur)
            if cur and too_long:
                cues.append((offset, cur))
                cur = []
            cur.append(w)
            if w["disp"].endswith((".", "?", "!")) and len(" ".join(x["disp"] for x in cur)) > 28:
                cues.append((offset, cur))
                cur = []
        if cur:
            cues.append((offset, cur))
    lines = []
    for i, (off, ws) in enumerate(cues, 1):
        start = off + ws[0]["t"]
        end = off + ws[-1]["t"] + ws[-1]["d"] + 0.15
        if i < len(cues):
            nxt = cues[i][0] + cues[i][1][0]["t"]
            end = min(end, nxt - 0.02)
        text = " ".join(w["disp"] for w in ws)
        if len(text) > 44:  # balance onto two lines
            mid = len(text) // 2
            cut = min((k for k, ch in enumerate(text) if ch == " "), key=lambda k: abs(k - mid))
            text = text[:cut] + "\n" + text[cut + 1:]
        lines.append(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{text}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
async def main(only: set[str] | None):
    audio = {}
    for seg in SEGMENTS:
        audio[seg.id] = await synth(seg)
        print(f"[tts] {seg.id}: {audio[seg.id]['dur']:.1f}s, {len(audio[seg.id]['words'])} words")
    total = sum(a["dur"] + TAIL for a in audio.values())
    print(f"[tts] total narration {total / 60:.2f} min")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist",
                                                 "--force-device-scale-factor=1"])
        ctx = await browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1)
        await ctx.add_init_script(CURSOR_JS)
        page = await ctx.new_page()
        on_app = False
        for seg in SEGMENTS:
            if seg.page.startswith("slide:"):
                await page.goto((ROOT / "slides" / f"{seg.page[6:]}.html").as_uri())
                await page.wait_for_timeout(150)
                on_app = False
            elif not on_app:
                await page.goto(APP)
                await page.wait_for_selector(".vcard", timeout=30000)
                await page.wait_for_timeout(2500)
                on_app = True
            if seg.setup:
                try:
                    await seg.setup(page)
                except Exception as e:
                    print(f"   ! setup {seg.id}: {type(e).__name__}: {str(e)[:120]}")
                await page.wait_for_timeout(900)
            if only and seg.id not in only and (BUILD / "clips" / f"{seg.id}.mp4").exists():
                # replay cues quickly without recording so app state stays consistent
                for _, fn in seg.cues:
                    try:
                        await asyncio.wait_for(fn(page), timeout=8)
                    except Exception:
                        pass
                print(f"   {seg.id}: reused")
                continue
            print(f"[rec] {seg.id}")
            await record(page, seg, audio[seg.id])
        await browser.close()

    # concat + subtitles
    OUT.mkdir(exist_ok=True)
    lst = BUILD / "clips.txt"
    lst.write_text("".join(f"file 'clips/{s.id}.mp4'\n" for s in SEGMENTS))
    raw = BUILD / "demo_nosubs.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(raw)], check=True)
    timeline, off = [], 0.0
    for seg in SEGMENTS:
        clip_dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                                  "csv=p=0", str(BUILD / "clips" / f"{seg.id}.mp4")]).decode().strip())
        timeline.append((off, align_display(seg.text, audio[seg.id]["words"])))
        off += clip_dur
    srt = OUT / "AngioLens_demo.en.srt"
    srt.write_text(build_srt(timeline), encoding="utf-8")
    shutil.copy(srt, BUILD / "subs.srt")
    style = ("FontName=Segoe UI,FontSize=12,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,BackColour=&H90000000,"
             "BorderStyle=4,Outline=0,Shadow=0,MarginV=16,Alignment=2")
    final = OUT / "AngioLens_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "demo_nosubs.mp4", "-vf",
                    f"subtitles=subs.srt:force_style='{style}'", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
                    "-c:a", "copy", "-movflags", "+faststart", str(final)], check=True, cwd=BUILD)
    print(f"[done] {final}  ({off / 60:.2f} min)  subtitles: {srt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated segment ids to re-record")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(set(filter(None, a.only.split(","))) or None)))
