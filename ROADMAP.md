# Roadmap

This file tracks planned features and intended CLI surface area for paperpipe (`papi`).
It is not a commitment to specific timelines.

## Principles

- Prefer one mental model: `papi add` adds papers (arXiv or local files).
- Keep the local database format stable and easy to inspect/edit.
- Avoid API-heavy features unless they are clearly optional and cached.
- Prefer local-first workflows when feasible (no cloud/API keys required).
- Precedence for configuration: **CLI flags > env vars > config.toml > defaults**.

## Planned (next)

### 1) Refactor: split `paperpipe.py` into a `src/` package

Goal: make the codebase maintainable without changing CLI behavior or on-disk database formats.

- **Constraints**
  - Keep CLI surface area stable (commands/options/output).
  - Keep runtime data layout stable (`~/.paperpipe/…`, `papers/*/{paper.pdf,source.tex,meta.json,…}`).
  - Avoid behavior changes during the refactor; keep diffs mechanical.
- **Packaging**
  - Move to a `src/` layout (e.g., `src/paperpipe/*.py`) and update `pyproject.toml` accordingly.
  - Keep `papi` entrypoint stable (likely `paperpipe.cli:cli`).
  - Consider keeping a thin `paperpipe.py` shim (imports + delegates to the package).
- **Proposed module split (initial cut)**
  - `paperpipe/cli.py`: Click group + command wiring
  - `paperpipe/config.py`: config/env precedence + defaults
  - `paperpipe/store.py`: paths, index/meta read/write, paper directory helpers
  - `paperpipe/arxiv.py`: arXiv fetch/download/source extraction
  - `paperpipe/llm.py`: LiteLLM wrappers + prompts
  - `paperpipe/paperqa.py`: `papi ask` orchestration + staging/index handling
  - `paperpipe/search.py`: local search helpers
  - `paperpipe/audit.py`: summary/equation audit logic
  - `paperpipe/util.py`: small shared helpers (logging, IO limits, etc.)
- **Test plan**
  - Keep existing tests; add a minimal “CLI smokes” unit test for import/entrypoint regressions if needed.

### 2) Local LLM via Ollama (core + `papi ask`)

Goal: make a zero-cloud setup the default happy path: local generation for `add/regenerate` and local RAG via
PaperQA2 or LEANN (when installed).

- **Core paperpipe generation**
  - Support LiteLLM Ollama model ids (exact ids can vary by LiteLLM version; validate via `papi models --model ...`).
  - Document a working baseline:
    - LLM: `PAPERPIPE_LLM_MODEL=ollama/<chat-model>`
    - Embeddings (optional): `PAPERPIPE_EMBEDDING_MODEL=ollama/<embed-model>` (commonly a `nomic-embed-text`-style model)
  - Make Ollama base URL configuration “just work”:
    - Detect/normalize common env var names (e.g., `OLLAMA_HOST` vs `OLLAMA_API_BASE`) so users don’t have to guess.
    - Fail with a clear error when Ollama isn’t reachable (connection refused, wrong host, missing model, etc.).
- **PaperQA2 via `papi ask`**
  - Ensure `papi ask --pqa-llm ... --pqa-embedding ...` works cleanly with Ollama identifiers.
  - Add docs/examples for local-only `papi ask` (including an embedding model choice).

### 3) `papi ask`: PaperQA2 output stream hygiene

Goal: make `papi ask` output match the rest of paperpipe: quiet by default, high-signal, and debuggable when needed.

- **Default behavior**
  - Suppress PaperQA2’s verbose streaming logs by default.
  - Print a concise summary aligned with paperpipe’s style:
    - progress: “Indexing N PDFs…”, “Querying…”
    - result: final answer text
    - sources: a short, stable list of cited papers/files (and any missing/failed docs)
- **Debug/verbose mode**
  - `-v/--verbose` should surface PaperQA2’s raw output for troubleshooting.
  - Preserve color when possible:
    - If passthrough to a real TTY preserves color, keep it.
    - Otherwise accept “no color” and provide an explicit `--pqa-raw` passthrough mode.
- **Failure handling**
  - Detect common PaperQA2 failure patterns (crash loops, bad PDFs) and show actionable guidance, not walls of logs.
  - Keep the existing crash triage (identify the crashing document) but present it in a compact, paperpipe-style report.

### 4) `papi attach` (upgrade/attach files)

Goal: let users fix missing/low-quality assets after initial ingest.

- **Command**
  - `papi attach PAPER --pdf /path/to/better.pdf`
  - `papi attach PAPER --source /path/to/main.tex`
- **Behavior**
  - Replace/attach the specified file(s)
  - Update `meta.json` (`has_pdf` / `has_source`)
  - Regeneration policy:
    - Default: do not regenerate anything (fast + predictable)
    - `--regen auto`: regenerate only artifacts affected by the attached file(s) (e.g., equations when `--source` changes)
    - `--regen equations|summary|tags|all`: explicit
- **Options (TBD)**
  - `--regen equations|summary|tags|all`
  - `--backup` (keep `paper.pdf.bak`, `source.tex.bak`, etc.)

### 5) `papi bibtex` (export)

Goal: easy citation export that integrates with LaTeX workflows.

- **Command**
  - `papi bibtex PAPER...`
- **Output**
  - Prints BibTeX entries derived from stored metadata.
- **Behavior details**
  - Missing fields: emit best-effort entries (always include a stable key + title when available).
  - Key collisions: deterministic suffixing (`_2`, `_3`, …).
- **Options (TBD)**
  - `--to library.bib` (write/append)
  - `--key-style name|doi|arxiv|slug`

### 6) `papi import-bib` (bulk ingest)

Goal: bootstrap a library from an existing BibTeX file.

- **Command**
  - `papi import-bib /path/to/library.bib`
- **Dependency**
  - Use `bibtexparser` (BibTeX is irregular; avoid hand-rolled parsing). Prefer making this an optional extra.
- **Behavior (MVP)**
  - Create metadata-only paper entries (PDF can be attached later with `papi attach`)
  - Dedup/match order: `doi` > `arxiv_id` > bibtex key
- **Options (TBD)**
  - `--dry-run`
  - `--update-existing`
  - `--tag TAG` (apply to all imported)
  - `--name-from key|slug(title)`

## Later (after the above stabilizes)

- `papi rename OLD NEW` (safe rename + index/meta updates)
- `papi rebuild-index` (recover `index.json` from on-disk state)
- `papi stats` (tags over time, has_pdf/has_source, storage usage)
- arXiv version tracking + update checks (`papi check-updates`, `papi update`)
- `papi diff` (start as text diff; avoid semantic parsing in MVP)
  - Define diff targets explicitly (e.g., `summary.md` vs regenerated summary, or two papers).
  - Likely depends on `papi attach --backup`/snapshots to be meaningful.

## Out of scope for now (high scope creep)

- Citation graph / related paper discovery across multiple APIs
- Semantic embedding search with a dedicated local vector index
- Watch/notifications for new papers
- Zotero/Mendeley integration

## Completed

### Non-arXiv ingestion via `papi add --pdf` (MVP)

Implemented (see `README.md` → “Non-arXiv Papers” for usage and examples).
