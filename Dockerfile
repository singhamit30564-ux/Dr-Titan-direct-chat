# Dr-Titan BioNeMo Web Console — CPU image.
#
# This image runs the *console*. It does not contain the BioNeMo NIMs: those are
# GPU-only NVIDIA containers that speak HTTP, and the console is designed to talk
# to them over the network. That split is deliberate and is what lets you run a
# fast, small front end next to heavy GPU services.
#
#   docker build -t dr-titan-console .
#   docker run --rm -p 8501:8501 \
#     -e BIONEMO_NIM_URL=http://host.docker.internal:8000 \
#     -e NGC_API_KEY=$NGC_API_KEY \
#     -v "$PWD/.runtime:/app/.runtime" \
#     dr-titan-console
#
# With docker compose (see docker-compose.yml):
#   docker compose up --build

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    BIONEMO_CONSOLE_HOME=/app/.runtime

WORKDIR /app

# libexpat/libgomp are what the RDKit wheels link against; curl is used by the
# health check below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Requirements first so the dependency layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# The vendored 3Dmol.js viewer must be present, otherwise the 3D panes degrade
# to text; fail the build early instead of surprising the user at runtime.
RUN test -s bionemo_web/static/3Dmol-min.js \
    || (echo "bionemo_web/static/3Dmol-min.js is missing: run scripts/setup.sh before building" && exit 1)

# Streamlit config lives in .streamlit/config.toml (headless, 0.0.0.0, port 8501).
# Create the state directory owned by the runtime user.
RUN mkdir -p /app/.runtime && useradd --create-home --uid 10001 console \
    && chown -R console:console /app
USER console

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
