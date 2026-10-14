# Free deployment: Render (Docker, free tier)

AngioLens uses about **190 MB of RAM** under load (measured in the Docker container), so it fits Render's free
web service (512 MB). Predictions run on the server, and the 3D view renders in each visitor's own browser.

Free-tier behaviour: the service sleeps after 15 minutes without traffic, and the first visit after that takes
about a minute to wake it. Open the link a minute before judges or a demo.

## Option A: one click (Blueprint)

1. Sign up at https://render.com with your GitHub account. Render's docs only ask for a payment method if you exceed the free allowances.
2. Open: **https://render.com/deploy?repo=https://github.com/Arnav020/AngioLens**
   Render reads `render.yaml` from the repo. It creates a free Docker web service named `angiolens`.
3. When it asks for `GROQ_API_KEY`, paste your key, or leave it empty: the copilot then uses its offline template.
4. Click **Apply** and wait for the build (about 5–8 minutes). Your app is live at
   `https://angiolens-XXXX.onrender.com`; the exact URL appears in the dashboard.

## Option B: manual

Render dashboard → **New → Web Service** → connect the `AngioLens` GitHub repo → Language **Docker** →
Instance type **Free** → *(optional)* add environment variable `GROQ_API_KEY` → **Deploy**.
Health check path: `/api/health`.

The Dockerfile listens on `$PORT`, which Render sets automatically.

Every later `git push` to `main` redeploys the service automatically.

## Notes
- Free allowance: 750 instance hours a month; sleeping services use none. Running one service around the clock fits.
- Hugging Face Docker/Gradio Spaces now require billing, so they are no longer a free option.
- Local Docker check: `docker build -t angiolens . && docker run -p 8000:7860 angiolens` → http://127.0.0.1:8000
