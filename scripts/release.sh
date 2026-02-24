#!/usr/bin/env bash
set -euo pipefail

VERSION_ARG="${1:-}"

if [[ "${VERSION_ARG}" == "-h" || "${VERSION_ARG}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  scripts/release.sh [VERSION]

Notes:
  - VERSION is optional; if provided, it must match pyproject.toml.
  - Requires: git, uv.
  - Runs: make check, make build, creates/pushes tag. CI handles GitHub release.
EOF
  exit 0
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

require_clean_git() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is dirty. Commit or stash changes before releasing." >&2
    exit 2
  fi
}

project_version() {
  uv run python - <<'PY'
from __future__ import annotations

import pathlib
import sys

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
sys.stdout.write(data["project"]["version"])
PY
}


require_cmd git
require_cmd uv

require_clean_git

VERSION="$(project_version)"
if [[ -n "$VERSION_ARG" && "$VERSION_ARG" != "$VERSION" ]]; then
  echo "VERSION arg ($VERSION_ARG) does not match pyproject.toml ($VERSION)" >&2
  exit 2
fi

# Single source of truth is pyproject.toml; CLI version is derived from package metadata.

TAG="v$VERSION"

echo "Release: $TAG"
echo "Running checks..."
make check

echo "Building dist..."
make build

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Tag already exists locally: $TAG"
else
  git tag -a "$TAG" -m "$TAG"
fi

echo "Pushing commit + tag..."
git push
git push origin "$TAG"

echo "Done. Tag push will trigger release.yml -> publish.yml workflows."
