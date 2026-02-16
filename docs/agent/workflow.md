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

## Creating Issues via CLI/API (for agents)

GitHub issue templates are a UI feature — neither `gh` CLI nor GitHub API/MCP supports creating issues from templates directly. When creating issues programmatically:

1. **Read the template** in `.github/ISSUE_TEMPLATE/` to see required fields
2. **Structure the body** to match the template sections (use markdown headers)
3. **Add labels** that the template would auto-apply (e.g., `bug` for bug_report.yml)

Example for bug_report.yml:
```markdown
## Description
[Clear description of the bug]

## Steps to Reproduce
1. ...
2. ...

## Command Used
\`\`\`bash
papi ...
\`\`\`

## Installation Method
[uv tool install / pip install / From source]

## Feature Area
[Paper fetching / RAG / Agent integration / etc.]

## Python Version
[e.g., 3.12]

## Operating System
[e.g., Linux, macOS, Windows]

## Error Logs
\`\`\`
[paste logs here]
\`\`\`

## Checklist
- [x] I have searched existing issues
- [x] I have tried with the latest version
```

## Creating PRs via CLI/API (for agents)

Unlike issues, `gh pr create` supports `--template` to use `.github/PULL_REQUEST_TEMPLATE.md`. However, if you pass `--body`, it overrides the template. The GitHub API/MCP also doesn't support templates.

When creating PRs programmatically, structure the body to match `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Summary
[Brief description]

Closes: #[issue number]

## Changes
- [Change 1]
- [Change 2]

## Type of Change
- [x] Bug fix / New feature / etc.

## Testing
- [x] Ran `make check` (or equivalent)
- [x] Added/updated tests
- [x] Tested CLI manually

## Checklist
- [x] Code follows existing style
- [x] Documentation updated if needed
```

## Title Conventions

Use conventional commit format for issue and PR titles (used by git-cliff for changelog):

```
type(scope): short description
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`

**Scopes:** `cli`, `leann`, `pqa`, `search`, `agent`, `mcp`, or omit for broad changes

**Examples:**
- `feat(leann): incremental indexing for new papers`
- `fix(cli): handle missing PDF gracefully`
- `docs(agent): add MCP server setup guide`
- `refactor: split cli.py into submodules`

## Commit Validation

Commit messages must follow conventional commits (see `docs/agent/releases.md`).

- **Locally**: A `commit-msg` hook rejects non-conforming messages. Auto-installed by `make deps`.
- **CI**: The `commit-lint` job validates all PR commits. If it fails, fix the offending commit with `git rebase -i` and amend the message.

## Labels

Defined in `.github/labels.yml`, synced automatically via `sync-labels.yml` workflow.

| Label | Use for |
|-------|---------|
| `bug` | Bug reports (auto-applied by template) |
| `enhancement` | Feature requests (auto-applied by template) |
| `documentation` | Docs-only changes |
| `question` | Questions needing clarification |
| `good first issue` | Newcomer-friendly tasks |
| `help wanted` | Needs external contribution |
| `wontfix` | Won't be addressed |
| `duplicate` | Already exists |
| `invalid` | Not valid/applicable |
