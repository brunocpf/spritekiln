#!/usr/bin/env bash
# SpriteKiln installer for a local ComfyUI.
#   ./install.sh                       # install nodes + workflows + examples (no model download)
#   ./install.sh --models core anim    # ...and download models: core | anim | hq-rotate | all (default: core anim)
# Env overrides: COMFYUI_DIR (folder containing main.py), MODELS_DIR, INPUT_DIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${COMFYUI_DIR:-}" ]; then
  # Comfy Desktop standalone install, then a plain git checkout
  for c in "$HOME/ComfyUI-Installs/ComfyUI/ComfyUI" "$HOME/ComfyUI" "$HOME/Documents/ComfyUI"; do
    [ -f "$c/main.py" ] && COMFYUI_DIR="$c" && break
  done
fi
COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
[ -f "$COMFYUI_DIR/main.py" ] || { echo "ComfyUI not found at $COMFYUI_DIR (set COMFYUI_DIR)"; exit 1; }
# Comfy Desktop keeps models/inputs in ~/ComfyUI-Shared; a git checkout keeps them inside ComfyUI
SHARED="$HOME/ComfyUI-Shared"
MODELS_DIR="${MODELS_DIR:-$([ -d "$SHARED/models" ] && echo "$SHARED/models" || echo "$COMFYUI_DIR/models")}"
INPUT_DIR="${INPUT_DIR:-$([ -d "$SHARED/input" ] && echo "$SHARED/input" || echo "$COMFYUI_DIR/input")}"

retire() {  # move a leftover from the old "PixelForge" name out of the way (Trash on macOS)
  [ -e "$1" ] || return 0
  if [ -d "$HOME/.Trash" ]; then mv "$1" "$HOME/.Trash/$(basename "$1")-$(date +%s)"; else rm -rf "$1"; fi
  echo "   removed old $(basename "$1")"
}
retire "$COMFYUI_DIR/custom_nodes/ComfyUI-PixelForge"
retire "$COMFYUI_DIR/user/default/workflows/PixelForge"
for f in "$INPUT_DIR"/pixelforge_example_*.png; do retire "$f"; done
PY="$COMFYUI_DIR/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"

echo "== SpriteKiln nodes -> $COMFYUI_DIR/custom_nodes/ComfyUI-SpriteKiln"
dest="$COMFYUI_DIR/custom_nodes/ComfyUI-SpriteKiln"
[ -L "$dest" ] && rm "$dest"
mkdir -p "$dest"
rsync -a --delete --exclude __pycache__ --exclude palettes "$HERE/ComfyUI-SpriteKiln/" "$dest/"
mkdir -p "$dest/palettes"   # user palettes (.hex / .gpl) survive re-installs

echo "== ComfyUI-GGUF (needed by the Wan 14B and Qwen workflows: 07, 10, 11, 16, 17, 19, 20)"
if [ ! -d "$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF" ]; then
  git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF"
fi
if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$UV" pip install --python "$PY" -q -r "$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"
else
  "$PY" -m pip install -q -r "$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"
fi

chmod +x "$HERE/scripts/"*.py "$HERE/scripts/"*.sh 2>/dev/null || true

echo "== workflows -> ComfyUI workflow library (sidebar: Workflows / SpriteKiln)"
wf="$COMFYUI_DIR/user/default/workflows/SpriteKiln"
mkdir -p "$wf"
cp "$HERE"/workflows/SpriteKiln_*.json "$wf/"

echo "== example inputs -> $INPUT_DIR"
mkdir -p "$INPUT_DIR"
cp "$HERE"/examples/*.png "$INPUT_DIR/"

if [ "${1:-}" = "--models" ]; then
  shift
  [ $# -eq 0 ] && set -- core anim
  MODELS_DIR="$MODELS_DIR" "$HERE/scripts/download_models.sh" "$@"
fi
echo "Done. Restart ComfyUI, then open Workflows → SpriteKiln."
