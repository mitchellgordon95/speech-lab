#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
else
  python3 -m venv .bootstrap
  .bootstrap/bin/python -m pip install uv
  uv_bin="$PWD/.bootstrap/bin/uv"
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.cache/uv"
"$uv_bin" sync --locked --python 3.12 --group dev
printf '\nReady. Run ./start.sh and open http://127.0.0.1:8765\n'
