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
