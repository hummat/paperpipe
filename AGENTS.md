# Repository Guidelines (for developing paperpipe)

This file is for agents working ON the paperpipe codebase. `CLAUDE.md` and `GEMINI.md` are symlinks to this file.

For agents USING paperpipe in other repos, see `AGENT_INTEGRATION.md` (or run `papi docs`).

## Conventions

Read relevant `docs/agent/` files before proceeding:
- `workflow.md` — **read before starting any feature** (issues with labels, branching, PRs)
- `code_conventions.md` — **read before writing code** (style, typing, minimal diffs)
- `architecture.md` — read before adding modules/restructuring
- `testing_patterns.md` — read before writing tests
- `releases.md` — read before releasing

**REQUIRED: Read `docs/agent/workflow.md` before implementing, updating, fixing, or changing anything.**

---

## Project Overview

- **Type:** Python CLI application
- **CLI entry point:** `papi` (defined in `pyproject.toml` via `[project.scripts]`)
- **GitHub:** `hummat/paperpipe`
- **Goal:** Maintain a local paper database (PDF + LaTeX + summaries/equations) for coding agents and PaperQA2.

## Architecture & Runtime Data

Package-based Click CLI (`paperpipe/`). Runtime state lives in `~/.paperpipe/` (do not commit):

- `~/.paperpipe/index.json`: quick lookup index mapping paper name → metadata
- `~/.paperpipe/papers/{name}/`: per-paper directory (PDF, LaTeX, summary, equations, metadata)

Key flows:

- `add`: fetch arXiv/Semantic Scholar metadata → download PDF + LaTeX → generate summary/equations/tags → update index
- `regenerate`: re-run summary/equation generation for an existing paper
- `export`: copy selected content to a destination directory
- `ask`: route to PaperQA2 for RAG queries over the stored PDFs (if installed)

## Commands

**Use `uv run` (not bare `python`/`pytest`) and `make` targets for all commands.**

```bash
# Verify after changes — ALWAYS run this:
make check                    # fmt + lint + type + test

# Install
uv sync --group dev           # via uv (matches CI)
pip install -e ".[dev]"       # alternative: editable + dev tools

# Individual steps (if needed)
make fmt                      # ruff format
make lint                     # ruff check
make type                     # pyright
make test                     # pytest (skip integration)

# CLI
papi --help

# Tests (pyproject.toml sets -m 'not integration' in addopts)
uv run pytest                              # default (skip integration)
uv run pytest -m integration               # network-dependent tests only
uv run pytest -m "integration or not integration"  # all tests
```

## Project-Specific Style

- Python >= 3.10; lines ≤ 120 chars (see `[tool.ruff]`)
- Keep CLI behavior stable; update `README.md` when adding flags/commands

## LLM & PaperQA2 Integration

- PaperQA2 uses `pqa` CLI if installed; model selection via LiteLLM identifiers
- Summaries/equations/tags generated via LiteLLM; fallback to regex/metadata if unavailable
- API keys checked: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `VOYAGE_API_KEY`, `OPENROUTER_API_KEY`

## Testing Notes

- Use `tmp_path`/`monkeypatch` to avoid touching `~/.paperpipe/`
- Mark network tests with `@pytest.mark.integration` (and `@pytest.mark.slow` if long-running)
- New functionality: test success + one common failure mode
- Pre-1.0: minor = breaking/behavior change, patch = bugfix/internal

## Commit Guidelines

- Short, imperative subjects (optionally `feat:`, `fix:`, etc.)
- PRs: what/why, how to test, CLI/database behavior changes

## Code Workflow

1. **Before editing**: read files first; understand existing code
2. **After code changes**: run `make check` (runs: ruff format → ruff check → pyright → pytest via uv)
3. **Doc check**: explicitly verify if docs/prompts need updating (even if "no doc impact")
4. **CLI changes**: update `README.md`, `AGENT_INTEGRATION.md`, `skills/papi/SKILL.md`, `skills/papi/commands.md`
5. **Doc style**: don't document default behavior (it's default); keep agent-facing docs KISS and concise

If `uv run` fails (sandbox/offline): fall back to `.venv/bin/*` or set `UV_CACHE_DIR=$PWD/.uv-cache` and `UV_LINK_MODE=copy`.
