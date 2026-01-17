# Feature Workflow

**Read this file before starting any feature or non-trivial change.**

## New Features

1. **Discuss/plan** — clarify requirements, identify affected files
2. **Create GitHub issue** — use the appropriate issue template (bug report or feature request)
3. **Create branch** — `git checkout -b feat/<short-name>`
4. **Implement** — follow Code Workflow in `AGENTS.md` (or `CLAUDE.md`/`GEMINI.md`, which are symlinks)
5. **Create PR** — use the PR template, reference issue (`Closes #N`), fill out all sections

## Trivial Changes

Skip issue for typos, small fixes, docs-only changes. Branch + PR is still recommended.

## Branch Naming

- `feat/<name>` — new features
- `fix/<name>` — bug fixes
- `refactor/<name>` — internal improvements
- `docs/<name>` — documentation only

## Templates

- **Issues**: Use `.github/ISSUE_TEMPLATE/` templates (bug_report.yml, feature_request.yml)
- **PRs**: Use `.github/PULL_REQUEST_TEMPLATE.md` — fill out Summary, Changes, Type, Testing, Checklist
- **Contributing**: See `.github/CONTRIBUTING.md` for dev setup and code style
