# Feature Workflow

**Read this file before starting any feature or non-trivial change.**

## New Features

1. **Discuss/plan** — clarify requirements, identify affected files
2. **Create GitHub issue** — document scope, acceptance criteria, add labels (see below)
3. **Create branch** — `git checkout -b feat/<short-name>`
4. **Implement** — follow Code Workflow in `AGENTS.md`
5. **Create PR** — reference issue (`Closes #N`), describe changes

## Issue Labels

Always add appropriate labels when creating issues. Common labels:

| Label | Use for |
|-------|---------|
| `enhancement` | New features or improvements |
| `bug` | Something isn't working |
| `CLI` | Changes to CLI commands/flags |
| `UX` | User experience improvements |
| `documentation` | Docs-only changes |
| `recovery` | Data recovery / repair features |

Check existing issues for label patterns: `gh issue list --repo hummat/paperpipe`

## Trivial Changes

Skip issue for typos, small fixes, docs-only changes. Branch + PR is still recommended.

## Branch Naming

- `feat/<name>` — new features
- `fix/<name>` — bug fixes
- `refactor/<name>` — internal improvements
- `docs/<name>` — documentation only
