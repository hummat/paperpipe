#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv sync --group dev
  exit 0
fi

python -m pip install -e ".[dev]"
