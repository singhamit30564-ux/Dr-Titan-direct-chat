#!/usr/bin/env bash
# Fetch the NVIDIA BioNeMo Framework source for reference (optional).
#
#   bash scripts/fetch_bionemo.sh          # sparse clone into third_party/
#   BRANCH=main bash scripts/fetch_bionemo.sh
#
# Why this exists: BioNeMo has two faces.
#
#   * The NIM microservices are what this console *drives* over HTTP (inference).
#   * The BioNeMo Framework repository is the training / recipe code (models,
#     datasets, distributed training configurations).
#
# The framework is a large, GPU-oriented research tree (it is not needed to run
# this app and is deliberately NOT vendored into the repository), but having it
# locally is useful when you want to train, fine-tune, or read the reference
# implementation behind a NIM. ./docs/ARCHITECTURE.md explains the relationship.
#
# Everything lands in third_party/bionemo-framework/ which .gitignore excludes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/bionemo-framework"
REPO="${REPO:-https://github.com/NVIDIA/bionemo-framework.git}"
BRANCH="${BRANCH:-main}"
TAG="${TAG:-}"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed." >&2
  exit 1
fi

if [ -d "$DEST/.git" ]; then
  echo "==> Updating existing clone at $DEST"
  git -C "$DEST" fetch --depth 1 origin "${TAG:-$BRANCH}"
  git -C "$DEST" checkout --detach FETCH_HEAD
else
  echo "==> Cloning $REPO (${TAG:-$BRANCH}) into $DEST"
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 --branch "${TAG:-$BRANCH}" "$REPO" "$DEST"
fi

echo "==> Version / contents"
[ -f "$DEST/VERSION" ] && echo "    VERSION: $(cat "$DEST/VERSION")"
[ -d "$DEST/models" ] && echo "    models : $(ls "$DEST/models" 2>/dev/null | tr '\n' ' ')"
[ -d "$DEST/recipes" ] && echo "    recipes: $(ls "$DEST/recipes" 2>/dev/null | head -8 | tr '\n' ' ')"

cat <<'MSG'

==> Done.

Read it:

    ls third_party/bionemo-framework
    less third_party/bionemo-framework/README.md

Train / run recipes (needs a GPU and a matching NGC container, e.g.
nvcr.io/nvidia/clara/bionemo-framework:nightly - see
third_party/bionemo-framework/README.md for the current image and mount layout).
This console is the inference front end; recipes are the training path.
MSG
