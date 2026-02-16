#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv sync --group dev
else
  python -m pip install -e ".[dev]"
fi

# Use tracked hooks directory so hooks are version-controlled.
git config core.hooksPath scripts/hooks
