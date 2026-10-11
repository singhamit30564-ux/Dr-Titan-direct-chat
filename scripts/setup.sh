#!/usr/bin/env bash
# One-command local setup for the Dr-Titan BioNeMo Web Console.
#
#   bash scripts/setup.sh
#
# Creates a virtual environment, installs the runtime requirements, and makes
# sure the vendored 3Dmol.js viewer is present (it is committed, but this
# restores it if the tree was trimmed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-.venv}"

echo "==> Python"
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required (3.11 recommended).")
PY

echo "==> Virtual environment ($VENV)"
if [ ! -d "$VENV" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

echo "==> Runtime requirements (CPU-only; no GPU needed)"
pip install -r requirements.txt

echo "==> 3D viewer asset"
if [ ! -s bionemo_web/static/3Dmol-min.js ]; then
  echo "    3Dmol-min.js is missing; fetching 3Dmol.js 2.5.5 from npm"
  TMPDIR_3DMOL="$(mktemp -d)"
  ( cd "$TMPDIR_3DMOL" && npm pack 3dmol >/dev/null 2>&1 && tar xzf 3dmol-*.tgz )
  mkdir -p bionemo_web/static
  cp "$TMPDIR_3DMOL/package/build/3Dmol-min.js" bionemo_web/static/
  cp "$TMPDIR_3DMOL/package/build/3Dmol-min.js.LICENSE.txt" bionemo_web/static/ 2>/dev/null || true
  cp "$TMPDIR_3DMOL/package/LICENSE" bionemo_web/static/3Dmol-LICENSE 2>/dev/null || true
  rm -rf "$TMPDIR_3DMOL"
else
  echo "    present ($(wc -c < bionemo_web/static/3Dmol-min.js) bytes)"
fi

cat <<'MSG'

==> Done.

Start the console:

    source .venv/bin/activate
    bash scripts/run_console.sh

Then open http://localhost:8501

No GPU? The console still works: it uses its own documented simulation backends
and every result says so. With a GPU, start NVIDIA NIM containers first
(see scripts/start_nims.sh) and the same UI will use them automatically.
MSG
