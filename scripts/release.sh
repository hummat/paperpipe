#!/usr/bin/env bash
set -euo pipefail

VERSION_ARG="${1:-}"

if [[ "${VERSION_ARG}" == "-h" || "${VERSION_ARG}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  scripts/release.sh [VERSION]

Notes:
  - VERSION is optional; if provided, it must match pyproject.toml.
  - Requires: git, python, gh; and gh auth login.
  - Runs: make check, make build, creates/pushes tag, then gh release create.
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
  python - <<'PY'
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

verify_version_synced() {
  local expected="$1"
  local click_ver
  click_ver="$(python - <<'PY'
import re
from pathlib import Path

text = Path("paperpipe/cli.py").read_text(encoding="utf-8")
m = re.search(r"@click\.version_option\(\s*version\s*=\s*['\"]([^'\"]+)['\"]\s*\)", text)
print(m.group(1) if m else "")
PY
)"
  if [[ -z "$click_ver" || "$click_ver" != "$expected" ]]; then
    echo "Version mismatch: pyproject.toml=$expected, paperpipe/cli.py=$click_ver" >&2
    exit 2
  fi
}

require_cmd git
require_cmd python
require_cmd gh

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run: gh auth login" >&2
  exit 2
fi

require_clean_git

VERSION="$(project_version)"
if [[ -n "$VERSION_ARG" && "$VERSION_ARG" != "$VERSION" ]]; then
  echo "VERSION arg ($VERSION_ARG) does not match pyproject.toml ($VERSION)" >&2
  exit 2
fi

verify_version_synced "$VERSION"

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

echo "Creating GitHub release (published) via gh..."
gh release create "$TAG" --generate-notes --verify-tag

echo "Done. This should trigger the Publish workflow."
