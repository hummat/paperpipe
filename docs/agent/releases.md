# Releases

## Creating a Release

```bash
make release                 # uses pyproject.toml version
make release VERSION=1.2.0   # explicit version
```

Runs checks → builds → tags → creates GitHub release → triggers PyPI publish.

## Version Numbering

Pre-1.0: `0.x.y` where x = breaking/behavior change, y = bugfix/internal.
Post-1.0: Semantic versioning (major.minor.patch).

## Commit Prefixes

- `feat:`/`fix:`/`docs:`/`chore:`/`refactor:` → auto-categorized in release notes
- `BREAKING` anywhere → Breaking Changes section

Heuristics also work: `Add`/`Implement` → Features, `Fix`/`Resolve` → Bug Fixes.

## Manual Release Notes

```bash
scripts/generate_release_notes.sh v1.0.0 v1.1.0   # between tags
scripts/generate_release_notes.sh v1.1.0 HEAD     # preview next
```

## Key Files

- `scripts/release.sh` — main automation
- `.github/release.yml` — auto-generated notes config
- `.github/workflows/publish.yml` — PyPI workflow
