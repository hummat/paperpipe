# Repository Guidelines

This file is the single source of truth for agent instructions in this repo.
`CLAUDE.md` and `GEMINI.md` are symlinks to this file.

## Project Overview

- **Type:** Python CLI application
- **CLI entry point:** `papi` (defined in `pyproject.toml` via `[project.scripts]`)
- **Goal:** Maintain a local paper database (PDF + LaTeX + summaries/equations) for coding agents and PaperQA2.

## Architecture & Runtime Data Layout

Package-based Click CLI (`paperpipe/`). Runtime state lives in `~/.paperpipe/` (do not commit generated data):

- `~/.paperpipe/index.json`: quick lookup index mapping paper name → metadata
- `~/.paperpipe/papers/{name}/`: per-paper directory (PDF, LaTeX, summary, equations, metadata)

Key flows:

- `add`: fetch arXiv or Semantic Scholar metadata → download PDF + (if available) LaTeX source → generate summary/equations/tags
  → update index
- `regenerate`: re-run summary/equation generation for an existing paper
- `export`: copy selected content to a destination directory
- `ask`: route to PaperQA2 for RAG queries over the stored PDFs (if installed)

## Project Structure & Module Organization

- `paperpipe/`: package containing the core logic and the Click CLI (installed as `papi`).
- `test_paperpipe.py`: pytest suite (unit tests + optional integration checks).
- `pyproject.toml`: packaging (Hatchling), dependencies, and tool configuration (pytest markers, Ruff).
- `README.md`: end-user documentation and CLI examples.

Runtime data is stored outside the repo in `~/.paperpipe/` (PDFs, LaTeX, summaries, index). Do not commit generated paper databases.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: editable install with dev tools (`pytest`, `ruff`, `pyright`).
- `uv sync --group dev`: install dev deps via uv (matches CI).
- `pip install -e ".[all]"`: editable install with all optional features (LLM + PaperQA2 integration).
- `papi --help`: show CLI commands and options.
- `pytest`: run the default test suite (note: `pyproject.toml` sets `-m 'not integration'` in `addopts`).
- `pytest -m "integration or not integration"`: run all tests (including integration) while keeping other `addopts`.
- `pytest -m "not integration"`: skip tests that may require network or external CLIs.
- `pytest -m integration`: run network-dependent integration tests.
- `ruff check .`: run lint (configured for 100-char lines and import sorting).
- `ruff check . --fix`: apply safe auto-fixes.
- `ruff format .`: format code.
- `pyright`: type check (basic mode; configuration in `pyproject.toml`).

## Coding Style & Naming Conventions

- Python >= 3.10; 4-space indentation; keep lines ≤ 120 chars (see `[tool.ruff]`).
- Use type hints for public helpers and any non-trivial return values.
- Naming: `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, tests as `test_*`.
- Keep CLI behavior stable: when adding flags/commands, update `README.md` examples.

## LLM & PaperQA2 Integration Notes

- PaperQA2 integration uses the `pqa` CLI if installed.
- PaperQA2 model selection uses LiteLLM identifiers (via `papi ask --pqa-llm/--pqa-embedding`).
- Summaries/equations/tags are generated via LiteLLM when available; otherwise paperpipe
  falls back to simpler non-LLM extraction (metadata-based summaries, regex equation extraction).
- Common API keys paperpipe checks for:
  - Google: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
  - Anthropic: `ANTHROPIC_API_KEY`
  - OpenAI: `OPENAI_API_KEY` (or `AZURE_OPENAI_API_KEY`)
  - Voyage: `VOYAGE_API_KEY` (for Voyage embeddings)
  - OpenRouter: `OPENROUTER_API_KEY` (access 200+ models via unified API)

## Testing Guidelines

- Prefer unit tests that use `tmp_path`/`monkeypatch` to avoid touching `~/.paperpipe/`.
- Mark network/external-tool tests with `@pytest.mark.integration` (and `@pytest.mark.slow` when appropriate).
- New functionality should include a focused test covering success + a common failure mode.
- Versioning (pre-1.0): treat minor as breaking/behavior changes and patch as bugfix/internal/no behavior change.
- Pre-1.0 stability: no backward compatibility guarantees unless explicitly requested.

## Test Markers

- `@pytest.mark.integration`: requires network access (arXiv / model APIs / external CLIs)
- `@pytest.mark.slow`: long-running tests (e.g., LLM calls)

## Commit & Pull Request Guidelines

- Git history currently contains only `Initial commit` (no established conventions yet).
- Use short, imperative commit subjects (optionally Conventional Commits like `feat: …`, `fix: …`).
- PRs should include: what/why, how to test (`pytest`, `ruff check .`), and any behavior changes to the CLI or database format.

## Agent-Specific Instructions

- When Python code or tooling (`pyproject.toml`, CI) is touched, run `uv run ruff format .` (first),
  then `uv run ruff check .`, `uv run pyright`, and `uv run pytest -m "not integration"` (or note what you skipped).
- Always use `uv run` for running dev tools (`ruff`, `pyright`, `pytest`) to ensure consistent environments.
- In sandboxed/offline environments, `uv run` may fail (e.g., it needs to fetch build requirements like `hatchling`,
  or cannot use the default `~/.cache/uv`); in that case either re-run with network/permissions or fall back to
  a pre-provisioned venv (`.venv/bin/ruff`, `.venv/bin/pyright`, `.venv/bin/pytest`). If needed, set
  `UV_CACHE_DIR=$PWD/.uv-cache` and `UV_LINK_MODE=copy` to avoid cache/FS issues.
- For any code change, explicitly check whether any docs/prompts need updating (even if the conclusion is “no doc impact”).
- When changing CLI surface area (commands/options/output) or user-facing behavior (env vars, database layout),
  update `README.md`, `AGENT_INTEGRATION.md`, `skill/SKILL.md`, and `skill/commands.md` as needed.
