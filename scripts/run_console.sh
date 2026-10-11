#!/usr/bin/env bash
# Launch the Streamlit console.
#
#   bash scripts/run_console.sh                 # defaults: 0.0.0.0:8501
#   PORT=8600 bash scripts/run_console.sh        # custom port
#   BIONEMO_BACKEND=local bash scripts/run_console.sh
#
# Binding to 0.0.0.0 (not 127.0.0.1) is deliberate: it makes the console
# reachable from another machine, a container host or a remote preview URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d .venv ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"
BACKEND="${BIONEMO_BACKEND:-auto}"

echo "Dr-Titan BioNeMo Web Console"
echo "  address : http://${ADDRESS}:${PORT}"
echo "  backend : ${BACKEND}  (override with BIONEMO_BACKEND=local|hosted|simulate)"
if [ -n "${NGC_API_KEY:-}" ] || [ -n "${NVIDIA_API_KEY:-}" ]; then
  echo "  NGC key : present (hosted NIMs enabled)"
else
  echo "  NGC key : not set (hosted NIMs and nvcr.io pulls will be unavailable)"
fi

exec streamlit run app.py \
  --server.address "$ADDRESS" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
