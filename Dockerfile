# AngioLens: one-container build (frontend is built in a node stage, served by FastAPI)
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY angiolens/ angiolens/
COPY scripts/ scripts/
COPY models/ models/
COPY data/raw/ data/raw/
COPY run.py .
COPY --from=web /web/dist web/dist
# Hugging Face Spaces expects 7860; Render/Koyeb/Railway inject $PORT.
ENV PORT=7860
EXPOSE 7860
RUN useradd -m -u 1000 app && chown -R app /app
USER app
CMD ["sh", "-c", "python run.py --host 0.0.0.0 --port ${PORT} --no-browser"]
