# Repository Guidelines

This file is the single source of truth for agent instructions in this repo.
`CLAUDE.md` and `GEMINI.md` are symlinks to this file.

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

```bash
# Install
pip install -e ".[dev]"       # editable + dev tools
pip install -e ".[all]"       # editable + all optional features
uv sync --group dev           # via uv (matches CI)

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
2. **After code changes**: `make check` (or: `ruff format .` → `ruff check .` → `pyright` → `pytest`)
3. **Doc check**: explicitly verify if docs/prompts need updating (even if "no doc impact")
4. **CLI changes**: update `README.md`, `AGENT_INTEGRATION.md`, `skills/papi/SKILL.md`, `skills/papi/commands.md`
5. **Doc style**: don't document default behavior (it's default); keep agent-facing docs KISS and concise

If `uv run` fails (sandbox/offline): fall back to `.venv/bin/*` or set `UV_CACHE_DIR=$PWD/.uv-cache` and `UV_LINK_MODE=copy`.

## Paper References (PaperPipe)

This repo implements methods from scientific papers. Papers are managed via `papi` (PaperPipe).

- Paper DB root: run `papi path` (default `~/.paperpipe/`; override via `PAPER_DB_PATH`).
- Add a paper: `papi add <arxiv_id_or_url>` or `papi add <s2_id_or_url>`.
- Inspect a paper (prints to stdout):
  - Equations (verification): `papi show <paper> -l eq`
  - Definitions (LaTeX): `papi show <paper> -l tex`
  - Overview: `papi show <paper> -l summary`
  - Quick TL;DR: `papi show <paper> -l tldr`
- Direct files (if needed): `<paper_db>/papers/{paper}/equations.md`, `source.tex`, `summary.md`, `tldr.md`, `figures/`

MCP Tools (if configured):
- `leann_search(index_name, query, top_k)` - Fast semantic search, returns snippets + file paths
- `retrieve_chunks(query, index_name, k)` - Detailed retrieval with formal citations (DOI, page numbers)
  - `embedding_model` is optional (auto-inferred from index metadata)
  - If specified, must match index's embedding model (check via `list_pqa_indexes()`)
- **Embedding priority** (prefer in order): Voyage AI → Google/Gemini → OpenAI → Local (Ollama)
  - Check available indexes: `leann_list()` or `list_pqa_indexes()`
- **When to use:** `leann_search` for exploration, `retrieve_chunks` for verification/citations

Rules:
- For "does this match the paper?", use `papi show <paper> -l eq` / `-l tex` and compare symbols step-by-step.
- For "which paper mentions X?":
  - Exact string hits (fast): `papi search --rg "X"` (case-insensitive literal by default)
  - Regex patterns: `papi search --rg --regex "pattern"` (for complex patterns like `BRDF\|material`)
  - Ranked search (BM25): `papi index --backend search --search-rebuild` then `papi search "X"`
  - Hybrid (ranked + exact boost): `papi search --hybrid "X"`
  - MCP semantic search: `leann_search()` or `retrieve_chunks()`
- If the agent can't read `~/.paperpipe/`, export context into the repo: `papi export <papers...> --level equations --to ./paper-context/`.
- Use `papi ask "..."` only when you explicitly want RAG synthesis (PaperQA2 default if installed; optional `--backend leann`).
  - For cheaper/deterministic queries: `papi ask "..." --pqa-agent-type fake`
  - For machine-readable evidence: `papi ask "..." --format evidence-blocks`
  - For debugging PaperQA2 output: `papi ask "..." --pqa-raw`
