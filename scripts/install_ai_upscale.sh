#!/usr/bin/env bash
set -euo pipefail

# Installs Real-ESRGAN dependencies and downloads a model checkpoint.
# Usage:
#   ./scripts/install_ai_upscale.sh
#   ./scripts/install_ai_upscale.sh x2plus
#   ./scripts/install_ai_upscale.sh x4plus /custom/output/dir

MODEL_VARIANT="${1:-x4plus}"
OUT_DIR="${2:-models/upscale}"

case "$MODEL_VARIANT" in
  x2plus)
    MODEL_FILE="RealESRGAN_x2plus.pth"
    MODEL_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    ;;
  x4plus)
    MODEL_FILE="RealESRGAN_x4plus.pth"
    MODEL_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ;;
  *)
    echo "Unsupported model variant: $MODEL_VARIANT"
    echo "Supported values: x2plus, x4plus"
    exit 1
    ;;
esac

if [[ ! -d ".venv" ]]; then
  echo "Warning: .venv not found in repo root. Using current Python environment."
fi

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

echo "Installing AI upscale dependencies..."
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "numpy<2" "opencv-python<4.12" "realesrgan>=0.3.0" "basicsr>=1.4.2"

mkdir -p "$OUT_DIR"
TARGET_PATH="$OUT_DIR/$MODEL_FILE"

if command -v curl >/dev/null 2>&1; then
  echo "Downloading $MODEL_FILE to $TARGET_PATH"
  curl -L --fail "$MODEL_URL" -o "$TARGET_PATH"
elif command -v wget >/dev/null 2>&1; then
  echo "Downloading $MODEL_FILE to $TARGET_PATH"
  wget -O "$TARGET_PATH" "$MODEL_URL"
else
  echo "Error: neither curl nor wget is installed; cannot download model."
  echo "Download manually from: $MODEL_URL"
  exit 1
fi

echo
echo "Done."
echo "Use this in the UI AI Model Path field:"
echo "  $TARGET_PATH"
