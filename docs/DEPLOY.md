# Free deployment: Hugging Face Spaces (Docker)

The free **CPU basic** tier (2 vCPU, 16 GB RAM) runs AngioLens comfortably: predictions take about 50 ms, and the
3D view renders in each visitor's own browser. The Space sleeps after about 48 h without visits and wakes on the
next visit (about 1 minute). The image was tested locally: `docker build -t angiolens . && docker run -p 8000:7860 angiolens`.

## Steps (about 10 minutes)

1. Create a free account at https://huggingface.co, then go to **New → Space**.
   - Space name: `angiolens` · License: MIT · **SDK: Docker** (Blank template) · Hardware: **CPU basic (free)** · Public.
2. Clone the empty Space next to your project folder:
   ```bash
   git clone https://huggingface.co/spaces/<your-hf-username>/angiolens hf-angiolens
   ```
3. Copy the contents of `AngioLens/` into `hf-angiolens/`, except the `.git` folder and your `.env`.
4. In `hf-angiolens/README.md`, add this block at the **very top** (Spaces reads its settings from it):
   ```yaml
   ---
   title: AngioLens
   emoji: 🫀
   colorFrom: red
   colorTo: yellow
   sdk: docker
   app_port: 7860
   pinned: false
   license: mit
   short_description: Virtual angiogram - CAD and LAD/LCX/RCA risk on a 3D heart
   ---
   ```
5. Push to the Space:
   ```bash
   cd hf-angiolens
   git add .
   git commit -m "Deploy AngioLens"
   git push
   ```
   When asked for a password, use a Hugging Face **access token** with write permission (Settings → Access Tokens).
   Every file is under 10 MB, so Git LFS is not needed.
6. *(Optional: enables the LLM copilot and voice questions)* In the Space, open **Settings → Variables and secrets → New secret**,
   set Name `GROQ_API_KEY` and paste your key as the value. Without it, the copilot uses its offline template.
7. Wait for the build (about 4–6 minutes, shown in the **Logs** tab). The app is then live at
   `https://<your-hf-username>-angiolens.hf.space`.

Add that URL to the README and to the Devpost "Try it out" links.

## Alternatives

- **Render** (free web service): Docker deploy from the GitHub repo. It uses `$PORT` automatically, but the 512 MB
  free instance is tight for this stack, and the instance sleeps after 15 minutes.
- **Koyeb / Railway**: Docker deploys that honour `$PORT`. Their free allowances change often.

Hugging Face Spaces is the recommended option: it is free, has enough memory, and is familiar to ML judges.
