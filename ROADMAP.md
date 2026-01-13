# Roadmap

This file tracks planned features and intended CLI surface area for paperpipe (`papi`).
It is not a commitment to specific timelines.

## Principles

- Prefer one mental model: `papi add` adds papers (arXiv or local files).
- Keep the local database format stable and easy to inspect/edit.
- Avoid API-heavy features unless they are clearly optional and cached.
- Prefer local-first workflows when feasible (no cloud/API keys required).
- Precedence for configuration: **CLI flags > env vars > config.toml > defaults**.
- **Leverage existing backends**: PaperQA2 and LEANN already provide sophisticated retrieval. Expose their features rather than reimplementing.

## Planned (next)

### 0) TL;DR summaries on `papi add` (Semantic Scholar style)

Goal: auto-generate a short, one-paragraph TL;DR when adding a paper (ideally using metadata + abstract, optionally LLM).

Ideas:
- Add `--tldr/--no-tldr` flag on `papi add` (default on).
- Store as `tldr.md` alongside `summary.md` and `equations.md`.
- If LLM available, ask for a 2–3 sentence TL;DR; otherwise fall back to a heuristic (title + abstract sentence).
- Expose in `papi show` output (optional).
- Provide a lightweight update path that detects missing artifacts (e.g., TL;DR) and backfills them without full regenerate.

Status: planned.

### 1) Retrieval quality improvements (expose existing backend features)

Goal: improve RAG quality for paper implementation workflows without adding new vector DBs.

**Context from analysis (Jan 2025):**
- PaperQA2 already has: tantivy (lexical), dense retrieval, LLM reranking, evidence summarization, optional Qdrant backend.
- LEANN already has: metadata filtering (post-search), grep-based exact text search, AST-aware code chunking.
- Adding Weaviate/Qdrant/Vespa as a new backend would duplicate existing functionality — not worth the complexity.

**What to expose:**

- **A) LEANN metadata filtering**
  - LEANN has rich post-search filtering (`==`, `!=`, `<`, `>`, `in`, `contains`, `starts_with`, etc.) that paperpipe doesn't surface.
  - Add `--leann-filter` option to `papi ask --backend leann` (e.g., `--leann-filter "paper_name in ['lora', 'attention']"`).
  - Status (Jan 2026): LEANN supports filtering in its Python API (`metadata_filters=...`) but the `leann ask` CLI does not expose it yet. Cleanest path: PR to LEANN to add a CLI flag (likely JSON), then paperpipe can pass through.

- **B) LEANN + grep fusion for hybrid-ish retrieval**
  - LEANN has grep search (`use_grep=True`) for exact text matching.
  - For papers, exact string hits matter: hyperparams ("λ=0.1"), symbols ("Eq. 7"), dataset names.
  - Expose via `--leann-grep` or similar; fuse grep + vector results.
  - Status (Jan 2026): LEANN supports grep in its Python API (`use_grep=True`) but the `leann ask` CLI does not expose it yet. Cleanest path: PR to LEANN to add `--use-grep`, then paperpipe can pass through.

- **C) PaperQA2 "fake" agent mode** — ✅ DONE
  - Implemented as `papi ask --pqa-agent-type fake`.

- **D) Better evidence block formatting** — ✅ DONE
  - The actual hallucination reduction comes from forcing the agent to cite evidence.
  - Add `papi ask --format evidence-blocks` that outputs structured JSON with `{paper, section, page, snippet}`.
  - Useful for agent integrations that want to enforce "no claim without citation".
  - Status (Jan 2026): implemented for PaperQA2 backend only (`--backend pqa`). LEANN does not currently expose the same citation/evidence structure in paperpipe.

- **E) Optional API reranking for LEANN**
  - LEANN's "reranking" is ANN-internal (approx → exact distance), not cross-encoder.
  - Add optional `--leann-rerank cohere|voyage` that calls an external reranker after retrieval.
  - Medium complexity: ~50-100 lines; retrieve `top_k=50` → rerank → keep `top_n=10`.
  - Requires API key; clearly optional.

**Not doing:**
- Adding Weaviate/Qdrant/Vespa as backends (duplicates PaperQA2's Qdrant option, high complexity).
- Building a custom vector index (PaperQA2 and LEANN already exist).

### 2) Non-LLM search improvements (`papi search`)

Goal: make `papi search` fast and useful without requiring LLM/embedding APIs.

**Completed:**

- **A) ripgrep (rg) for exact text search** — ✅ DONE

  Implemented as `papi search --grep "pattern"` with options:
  - `--fixed-strings` / `--regex` (literal vs regex)
  - `--context N` (context lines)
  - `--ignore-case` / `--case-sensitive`
  - `--max-matches N`
  - `--json` (machine-readable output)

  Falls back to `grep` if `rg` not installed.

- **B) SQLite FTS5 for ranked search (BM25)** — ✅ DONE

  Implemented as:
  - `papi search-index` — builds/updates `~/.paperpipe/search.db`
  - `papi search --fts` (enabled by default) — queries with BM25 ranking
  - `--include-tex` option for indexing LaTeX source

- **C) Hybrid: ripgrep + FTS5 fusion** — ✅ DONE

  Implemented as `papi search --hybrid "query"` (optionally `--show-grep-hits`).

**Remaining:**

- **D) Optional: reuse PaperQA2's tantivy index**

  **Utility: MEDIUM** — avoids duplicate indexing if user already ran `papi ask`.
  **Complexity: MEDIUM** — need to understand PaperQA2's index format.
  **Deps:** `paperqa` must be installed.

  [tantivy-py](https://github.com/quickwit-oss/tantivy-py) is what PaperQA2 uses internally. If the index exists at `~/.paperpipe/.pqa_index/`, we could query it directly for `papi search`.

  Deferred until FTS5 is implemented — may not be worth the complexity.

**Not doing:**
- Adding tantivy as a standalone dep for `papi search` (FTS5 is good enough, no new deps).

### 3) Local LLM via Ollama (core + `papi ask`) — ✅ DONE

Implemented:
- Detects `ollama/` model IDs and normalizes env vars (`OLLAMA_HOST`, `OLLAMA_API_BASE`, `OLLAMA_BASE_URL`, `OLLAMA_API_BASE_URL`)
- Reachability check with clear error message: "Start Ollama (`ollama serve`) or set OLLAMA_HOST..."
- Works for both core generation (`papi add/regenerate`) and RAG (`papi ask --pqa-llm ollama/...`)
- LEANN defaults to Ollama (`DEFAULT_LEANN_EMBEDDING_MODE = "ollama"`)

**Remaining:** Documentation/examples for local-only workflows.

### 4) `papi ask`: PaperQA2 output stream hygiene — PARTIAL

**Implemented:**
- `--pqa-raw` flag for full passthrough
- `-v/--verbose` enables raw output
- Default behavior suppresses PaperQA2 streaming logs and prints the filtered output (answer + citations) on completion

**Remaining:**
- Failure detection (crash loops, bad PDFs) with actionable guidance

### 5) `papi attach` (upgrade/attach files)

Goal: let users fix missing/low-quality assets after initial ingest.

**Utility: MEDIUM** — useful but not blocking for most workflows.
**Complexity: LOW** — straightforward file operations + meta.json updates.

- `papi attach PAPER --pdf /path/to/better.pdf`
- `papi attach PAPER --source /path/to/main.tex`
- Options: `--regen auto|equations|summary|tags|all`, `--backup`

### 6) `papi bibtex` (export)

Goal: easy citation export for LaTeX workflows.

**Utility: MEDIUM** — nice for academic users.
**Complexity: LOW** — metadata already stored, just formatting.

- `papi bibtex PAPER...` → prints BibTeX entries
- Options: `--to library.bib`, `--key-style name|doi|arxiv|slug`

## Later (lower priority or higher complexity)

### `papi import-bib` (bulk ingest from BibTeX)

**Utility: MEDIUM** — useful for bootstrapping from existing libraries.
**Complexity: MEDIUM** — BibTeX parsing is irregular, needs `bibtexparser` dependency.

- `papi import-bib /path/to/library.bib`
- Creates metadata-only entries (PDF via `papi attach` later).
- Dedup order: `doi` > `arxiv_id` > bibtex key.
- Optional extra: `paperpipe[bibtex]`.

### `papi rebuild-index` (recovery)

**Utility: HIGH** (for recovery) — but rare need.
**Complexity: LOW** — iterate paper dirs, rebuild `index.json`.

- Recover `index.json` from on-disk paper directories.
- Useful when index is corrupted or manually edited.

### `papi rename OLD NEW`

**Utility: LOW** — users can do this manually.
**Complexity: LOW** — rename dir + update index.

### Refactor: split `paperpipe.py` into a `src/` package

**Utility: MEDIUM** — maintainability for contributors.
**Complexity: MEDIUM** — mechanical but tedious, risk of regressions.

Deferred until the single-file approach becomes a real bottleneck. Current size (~3500 lines) is still manageable.

## Reconsidered (moved from "Later")

### `papi stats`

**Utility: LOW** — nice to have, not blocking.
**Complexity: LOW**.

Moved to "maybe later" — not a priority.

### arXiv version tracking + update checks

**Utility: LOW** — most users don't need this.
**Complexity: MEDIUM** — needs arXiv API polling, version comparison logic.

Deferred indefinitely. Users can manually re-add papers if needed.

### `papi diff`

**Utility: LOW** — unclear use case.
**Complexity: MEDIUM** — needs snapshot/backup infrastructure.

Deferred indefinitely. The original motivation (compare regenerated summaries) is better served by just regenerating and using git diff.

## Out of scope (won't do)

### Dedicated local vector index / new vector DB backend

PaperQA2 already provides tantivy + dense + LLM reranking + optional Qdrant.
LEANN already provides efficient local vector search with metadata filtering.
Adding Weaviate/Qdrant/Vespa as a paperpipe-native backend would:
- Duplicate existing functionality
- Add significant complexity (~500-1000 lines)
- Require users to run/manage another service

**Instead:** Expose existing backend features (see "Retrieval quality improvements").

### Citation graph / related paper discovery

High complexity, requires multiple API integrations. Out of scope for a local paper database tool.

### Watch/notifications for new papers

Medium complexity, low utility. Users can set up their own arXiv alerts.

### Zotero/Mendeley integration

High complexity. Users can export from Zotero to BibTeX and use `papi import-bib`.

## Completed

### Non-arXiv ingestion via `papi add --pdf` (MVP)

Implemented (see README.md → "Non-arXiv Papers").

### ripgrep exact text search (`papi search --grep`)

Implemented with full option support: `--fixed-strings`, `--context`, `--ignore-case`, `--max-matches`, `--json`. Falls back to `grep` if `rg` not installed.

### SQLite FTS5 ranked search (`papi search --fts`, `papi search-index`)

Implemented with BM25 ranking, field weighting, porter stemmer. Index stored at `~/.paperpipe/search.db`.

### Local LLM via Ollama

Implemented with env var normalization (`OLLAMA_HOST`, etc.), reachability checks, clear error messages. Works for core generation and `papi ask`.

### PaperQA2 "fake" agent mode (`papi ask --pqa-agent-type`)

Implemented — passes through to `pqa --agent.agent_type`.

---

## Appendix: Backend capability reference

This informed the "Retrieval quality improvements" decisions.

### PaperQA2 (verified Jan 2025)

| Feature | Status |
|---------|--------|
| Multi-stage pipeline (Search → Evidence → Answer) | ✅ |
| LiteLLM-compatible models | ✅ |
| Metadata from Crossref + Semantic Scholar + OpenAlex | ✅ |
| tantivy for full-text/lexical search | ✅ |
| LLM-based evidence reranking | ✅ |
| "Fake" agent mode (deterministic, low-token) | ✅ |
| Optional Qdrant backend | ✅ (not exposed in paperpipe yet) |
| Grobid for parsing | ❌ (uses PyPDF/PyMuPDF/Docling/nemotron instead) |

### LEANN (verified Jan 2025)

| Feature | Status |
|---------|--------|
| Tiny on-disk footprint (97% reduction) | ✅ |
| AST-aware code chunking (Python, Java, C#, TS, JS) | ✅ |
| OpenAI-compatible endpoints (LM Studio, vLLM, Ollama) | ✅ |
| ANN-internal reranking (approx → exact distance) | ✅ |
| Metadata filtering (post-search) | ✅ (not exposed in paperpipe) |
| Grep search (exact text) | ✅ (not exposed in paperpipe) |
| Hybrid BM25 search | ❌ |
| Cross-encoder reranking | ❌ |
| Native MCP server | ✅ |
