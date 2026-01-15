# Release Process

This document explains how releases work and how to generate release notes.

## Automated Release Process

The project uses GitHub's auto-generated release notes with custom categorization.

### Configuration

- **`.github/release.yml`**: Configures how commits/PRs are categorized in auto-generated notes
  - Categories: Breaking Changes, Features, Bug Fixes, Documentation, Internal, Other Changes
  - Works with PR labels and commit message patterns

### Creating a Release

```bash
# Standard release (uses pyproject.toml version)
make release

# Or specify version explicitly
make release VERSION=1.2.0
```

The `scripts/release.sh` script will:
1. Run checks (`make check`)
2. Build distribution (`make build`)
3. Create and push git tag (`v{version}`)
4. Create GitHub release with auto-generated notes (`gh release create`)
5. Trigger PyPI publish workflow (`.github/workflows/publish.yml`)

## Manual Release Notes Generation

For retroactive updates or custom formatting, use the script:

```bash
# Generate notes for a version
scripts/generate_release_notes.sh v1.0.0 v1.1.0

# Preview next release notes
scripts/generate_release_notes.sh v1.1.0 HEAD
```

This script:
- Parses commits between tags
- Categorizes by message patterns (feat:, fix:, Add, Refactor, etc.)
- Generates structured markdown output

### Updating Existing Releases

```bash
# Generate notes and update release
NOTES=$(scripts/generate_release_notes.sh v1.0.0 v1.1.0)
gh release edit v1.1.0 --notes "$NOTES"
```

## Commit Message Conventions

While not strictly enforced, using conventional commit prefixes improves auto-categorization:

- `feat:` or `feature:` → Features section
- `fix:` or `bugfix:` → Bug Fixes section
- `docs:` or `doc:` → Documentation section
- `chore:`, `refactor:`, `test:`, `ci:` → Internal section
- `BREAKING` anywhere in message → Breaking Changes section

Heuristics also work:
- Starts with `Add`, `Implement`, `Support`, `Enable` → Features
- Starts with `Fix`, `Resolve`, `Correct` → Bug Fixes
- Starts with `Refactor`, `Bump`, `Update`, `Migrate` → Internal

## PR Labels

Add labels to PRs for better categorization:

- `breaking-change`, `breaking` → Breaking Changes
- `feature`, `enhancement`, `feat` → Features
- `bug`, `fix`, `bugfix` → Bug Fixes
- `documentation`, `docs` → Documentation
- `internal`, `refactor`, `chore` → Internal

## Version Numbering

Pre-1.0: `0.x.y` where:
- `x` (minor): breaking changes or significant new behavior
- `y` (patch): bug fixes, docs, internal changes

Post-1.0: Semantic versioning (major.minor.patch)

## See Also

- `scripts/release.sh` - Main release automation script
- `scripts/generate_release_notes.sh` - Manual notes generation
- `.github/release.yml` - Auto-generated notes configuration
- `.github/workflows/publish.yml` - PyPI publish workflow
