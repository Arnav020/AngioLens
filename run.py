"""Start AngioLens:  python run.py  ->  http://127.0.0.1:8000"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not (ROOT / "models" / "angiolens.joblib").exists():
        sys.exit("models/angiolens.joblib not found - run `python scripts/train.py` first (about 5 minutes).")
    if not (ROOT / "web" / "dist" / "index.html").exists():
        print("[warn] web/dist missing - the API will run but the dashboard needs `cd web && npm install && npm run build`.")
    if not a.no_browser:
        webbrowser.open(f"http://{a.host}:{a.port}")
    uvicorn.run("angiolens.api:app", host=a.host, port=a.port)
