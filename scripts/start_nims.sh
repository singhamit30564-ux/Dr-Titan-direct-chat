#!/usr/bin/env bash
# Start NVIDIA BioNeMo NIM containers locally.
#
#   export NGC_API_KEY=nvapi-xxxxxxxx     # free key: https://build.nvidia.com
#   bash scripts/start_nims.sh            # all default NIMs
#   bash scripts/start_nims.sh esmfold    # just one
#
# Requirements on the HOST: an NVIDIA GPU + driver, Docker (or Podman) with
# nvidia-container-toolkit, and enough VRAM for the chosen NIM. These containers
# serve the same HTTP API the console speaks, so once one is up and the console
# points at http://localhost:8000 the backend badge flips from SIMULATED to
# LOCAL NIM - no code change.
#
# Port layout: the first NIM takes 8000 (the console's default), each additional
# one gets the next free port; register those ports on the Administration page.
set -euo pipefail

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY is not set. Get a free key at https://build.nvidia.com" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to the Docker daemon (is it running, and do you have permission?)." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found - NIM containers need an NVIDIA GPU and driver." >&2
  exit 1
fi
if ! docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: the GPU is not visible inside a test container."
  echo "         Install nvidia-container-toolkit and restart Docker. Continuing anyway."
fi

login() {
  echo "==> Logging in to nvcr.io"
  echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
}

# image tag : friendly name : cache dir suffix
declare -A IMAGES=(
  [esmfold]="nvcr.io/nim/meta/esmfold:latest"
  [alphafold2]="nvcr.io/nim/deepmind/alphafold2:latest"
  [molmim]="nvcr.io/nim/nvidia/molmim:latest"
  [genmol]="nvcr.io/nim/nvidia/genmol:latest"
  [diffdock]="nvcr.io/nim/mit/diffdock:latest"
  [proteinmpnn]="nvcr.io/nim/ipd/proteinmpnn:latest"
  [rfdiffusion]="nvcr.io/nim/ipd/rfdiffusion:latest"
  [evo2]="nvcr.io/nim/arc/evo2-40b:latest"
)

REQUESTED=("$@")
if [ ${#REQUESTED[@]} -eq 0 ]; then
  REQUESTED=(esmfold molmim diffdock proteinmpnn)
fi

login

PORT=8000
for name in "${REQUESTED[@]}"; do
  image="${IMAGES[$name]:-}"
  if [ -z "$image" ]; then
    echo "!! unknown NIM '$name' (known: ${!IMAGES[*]})" >&2
    continue
  fi
  cache="$HOME/.cache/nim/$name"
  mkdir -p "$cache"
  echo "==> Starting $name on port $PORT  ($image)"
  echo "    first run downloads several GB; subsequent starts are fast"
  docker run -d --rm \
    --name "nim-$name" \
    --gpus all \
    --shm-size=16g \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -e NIM_CACHE_PATH=/opt/nim/.cache \
    -u "$(id -u)" \
    -v "$cache:/opt/nim/.cache" \
    -p "${PORT}:8000" \
    "$image"
  echo "    waiting for readiness on http://localhost:${PORT}/v1/health/ready"
  for _ in $(seq 1 60); do
    if curl -fsS "http://localhost:${PORT}/v1/health/ready" >/dev/null 2>&1; then
      echo "    ✔ $name ready on port $PORT"
      break
    fi
    sleep 5
  done
  echo "    register this URL in the console: http://localhost:${PORT}"
  PORT=$((PORT + 1))
done

cat <<'MSG'

==> Containers started.

Check them:

    docker ps --filter name=nim-
    curl -s http://localhost:8000/v1/models

In the console, open Administration -> Backends, set the endpoint for each model
to the port printed above, then press "Probe every configured endpoint". The
badges on your results will read LOCAL NIM from then on.

Stop everything:

    docker ps --filter name=nim- -q | xargs -r docker stop
MSG
