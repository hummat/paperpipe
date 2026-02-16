# Feature Workflow

**Read this file before creating issues, PRs, or branches.**

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

- **Issues**: `.github/ISSUE_TEMPLATE/` (bug_report.yml, feature_request.yml)
- **PRs**: `.github/PULL_REQUEST_TEMPLATE.md`
- **Contributing**: `.github/CONTRIBUTING.md`

## Creating Issues via CLI/API (for agents)

GitHub issue templates are a UI feature — neither `gh` CLI nor GitHub API/MCP supports creating issues from templates directly. When creating issues programmatically:

1. **Read the template** in `.github/ISSUE_TEMPLATE/` to see required fields
2. **Structure the body** to match the template sections (use markdown headers matching each template field)
3. **Add labels** that the template would auto-apply (e.g., `bug` for bug_report.yml, `enhancement` for feature_request.yml)

## Creating PRs via CLI/API (for agents)

`gh pr create --body` overrides the template. When creating PRs programmatically:

1. **Read** `.github/PULL_REQUEST_TEMPLATE.md` to see required sections
2. **Structure the body** to match the template (use the same markdown headers)

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
