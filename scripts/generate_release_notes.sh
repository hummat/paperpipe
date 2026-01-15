#!/usr/bin/env bash
# Generate structured release notes from git commits between two tags
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/generate_release_notes.sh <from-tag> <to-tag>

Examples:
  scripts/generate_release_notes.sh v1.0.0 v1.1.0
  scripts/generate_release_notes.sh v1.1.0 HEAD

Generates structured release notes by parsing commits between tags.
Categorizes by commit message prefixes (feat:, fix:, docs:, etc.)
EOF
  exit 0
}

if [[ $# -lt 2 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  usage
fi

FROM_TAG="$1"
TO_TAG="$2"

# Get commits between tags
mapfile -t COMMITS < <(git log --oneline "${FROM_TAG}..${TO_TAG}" --no-merges | grep -v "^[a-f0-9]\+ Merge pull request")

# Initialize category arrays
declare -a BREAKING=()
declare -a FEATURES=()
declare -a FIXES=()
declare -a DOCS=()
declare -a INTERNAL=()
declare -a OTHER=()

# Categorize commits
for commit in "${COMMITS[@]}"; do
  sha="${commit%% *}"
  msg="${commit#* }"

  # Check for breaking changes (BREAKING or !)
  if [[ "$msg" =~ BREAKING ]] || [[ "$msg" =~ ^[a-z]+\!: ]]; then
    BREAKING+=("- $msg")
  # Conventional commit prefixes
  elif [[ "$msg" =~ ^feat: ]] || [[ "$msg" =~ ^feature: ]]; then
    FEATURES+=("- ${msg#*: }")
  elif [[ "$msg" =~ ^fix: ]] || [[ "$msg" =~ ^bugfix: ]]; then
    FIXES+=("- ${msg#*: }")
  elif [[ "$msg" =~ ^docs?: ]]; then
    DOCS+=("- ${msg#*: }")
  elif [[ "$msg" =~ ^chore: ]] || [[ "$msg" =~ ^refactor: ]] || [[ "$msg" =~ ^Refactor: ]] || [[ "$msg" =~ ^Bump ]] || [[ "$msg" =~ ^Update ]] || [[ "$msg" =~ ^Use ]] || [[ "$msg" =~ ^Migrate ]]; then
    INTERNAL+=("- $msg")
  # Heuristics for features/fixes without prefixes
  elif [[ "$msg" =~ ^Add\  ]] || [[ "$msg" =~ ^Implement\  ]] || [[ "$msg" =~ ^Support\  ]] || [[ "$msg" =~ ^Enable\  ]]; then
    FEATURES+=("- $msg")
  elif [[ "$msg" =~ ^Fix\  ]] || [[ "$msg" =~ ^Resolve\  ]] || [[ "$msg" =~ ^Correct\  ]]; then
    FIXES+=("- $msg")
  else
    OTHER+=("- $msg")
  fi
done

# Generate notes
echo "## What's Changed"
echo

if [[ ${#BREAKING[@]} -gt 0 ]]; then
  echo "### ⚠️ Breaking Changes"
  printf '%s\n' "${BREAKING[@]}"
  echo
fi

if [[ ${#FEATURES[@]} -gt 0 ]]; then
  echo "### Features"
  printf '%s\n' "${FEATURES[@]}"
  echo
fi

if [[ ${#FIXES[@]} -gt 0 ]]; then
  echo "### Bug Fixes"
  printf '%s\n' "${FIXES[@]}"
  echo
fi

if [[ ${#DOCS[@]} -gt 0 ]]; then
  echo "### Documentation"
  printf '%s\n' "${DOCS[@]}"
  echo
fi

if [[ ${#INTERNAL[@]} -gt 0 ]]; then
  echo "### Internal"
  printf '%s\n' "${INTERNAL[@]}"
  echo
fi

if [[ ${#OTHER[@]} -gt 0 ]]; then
  echo "### Other Changes"
  printf '%s\n' "${OTHER[@]}"
  echo
fi

# Get repo info for changelog link
REMOTE_URL=$(git remote get-url origin)
if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^/.]+) ]]; then
  OWNER="${BASH_REMATCH[1]}"
  REPO="${BASH_REMATCH[2]%.git}"
  echo "**Full Changelog**: https://github.com/$OWNER/$REPO/compare/${FROM_TAG}...${TO_TAG}"
fi
