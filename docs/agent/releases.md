# Releases

## Prerequisites

```bash
# Check if git-cliff is installed
git-cliff --version || cargo install git-cliff  # or: brew install git-cliff
```

## Creating a Release

```bash
make release                 # uses pyproject.toml version
make release VERSION=1.2.0   # explicit version
```

Runs checks → builds → tags → creates GitHub release → triggers PyPI publish.

## Changelog Generation

```bash
# Generate full changelog
git cliff -o CHANGELOG.md

# Preview unreleased changes
git cliff --unreleased

# Generate notes for specific release
git cliff --latest --strip header
```

## Version Numbering

Pre-1.0: `0.x.y` where x = breaking/behavior change, y = bugfix/internal.
Post-1.0: Semantic versioning (major.minor.patch).

## Conventional Commits

**Required format:** `type(scope): description`

| Type | Purpose | Version Impact |
|------|---------|----------------|
| `feat:` | New feature | Minor bump |
| `fix:` | Bug fix | Patch bump |
| `docs:` | Documentation only | Patch bump |
| `refactor:` | Code restructure, no behavior change | Patch bump |
| `perf:` | Performance improvement | Patch bump |
| `test:` | Adding/updating tests | Patch bump |
| `chore:` | Build, CI, tooling | Patch bump |
| `ci:` | CI configuration | Patch bump |
| `revert:` | Revert previous commit | Depends |

**Scope** (optional): Component affected, e.g., `fix(cli):`, `feat(mcp):`

**Breaking changes:** Add `!` after type or include `BREAKING CHANGE:` in body:
```
feat!: Remove deprecated config option
feat(api)!: Change return type
```

## Examples

```
feat: Add semantic search for papers
fix(mcp): Handle empty query gracefully
docs: Update installation instructions
chore: Bump dependencies
refactor(index): Simplify embedding pipeline
feat!: Remove legacy PDF parser
```

## Commit Validation

Conventional commit format is enforced at two levels:

- **Local hook** (`scripts/hooks/commit-msg`) — rejects non-conforming messages instantly. Auto-installed by `make deps` (via `scripts/deps.sh`).
- **CI check** (`commit-lint` job in `.github/workflows/ci.yml`) — validates all PR commits. Safety net for contributors who bypass the local hook.

## Key Files

- `cliff.toml` — git-cliff changelog config
- `scripts/release.sh` — main automation
- `scripts/hooks/commit-msg` — local commit validation
- `.github/workflows/publish.yml` — PyPI workflow
