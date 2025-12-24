#!/usr/bin/env bash
set -euo pipefail

rm -rf dist/

if command -v uv >/dev/null 2>&1; then
  uv build --sdist --wheel --out-dir dist --clear
  uv run twine check dist/*
  exit 0
fi

python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
