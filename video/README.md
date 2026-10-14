# Automated demo video

`make_video.py` produces the submission video end to end, without manual editing:

1. **Script.** `SEGMENTS` holds the narration for each scene and the UI actions attached to *cue words*. A click
   fires when the narrator says the matching word.
2. **Voice.** Microsoft Edge neural TTS (`edge-tts`, voice `en-US-AndrewMultilingualNeural`) returns audio plus
   word-level timestamps. Audio is cached in `build/audio/`.
3. **Capture.** Headless Chromium drives the real app at 1920×1080 using the GPU (ANGLE/D3D11). Frames come from
   the DevTools screencast at about 30 fps. A visible cursor with click ripples is injected so viewers can follow
   the interaction.
4. **Assembly.** ffmpeg builds a constant-frame-rate clip per segment, muxes the narration, concatenates the
   segments and burns in English subtitles generated from the TTS word timings. The same subtitles are written
   to `out/AngioLens_demo.en.srt` for upload to YouTube.

```bash
python run.py --no-browser          # terminal 1: the app on :8000
python video/make_video.py          # terminal 2: -> video/out/AngioLens_demo.mp4
python video/make_video.py --only explain,copilot   # re-record selected segments
```

Requirements: `pip install edge-tts playwright && playwright install chromium`, plus ffmpeg with libass on PATH.
The slides in `slides/` (title, architecture, closing) are plain HTML and use the same fonts as the app.
