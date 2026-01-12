# papi Command Reference

## Core Commands

| Command | Description |
|---------|-------------|
| `papi path` | Print database location |
| `papi list` | List all papers with tags |
| `papi list --tag TAG` | List papers filtered by tag |
| `papi tags` | List all tags with counts |
| `papi search "query"` | Search by title, tag, or content (scan) |
| `papi search --grep QUERY` | Exact text search (ripgrep/grep) |
| `papi search-index --rebuild` | Build ranked search index (`search.db`) |
| `papi search --fts "query"` | Ranked search (SQLite FTS5 / BM25; requires `search.db`) |
| `papi search --hybrid "query"` | Ranked search with exact-hit boost (FTS + grep) |
| `papi show <papers...>` | Show paper details or print stored content |
| `papi notes <paper>` | Open or print per-paper implementation notes |
| `papi install [components...]` | Install integrations (components: `skill`, `prompts`, `mcp`) |
| `papi uninstall [components...]` | Uninstall integrations (components: `skill`, `prompts`, `mcp`) |
| `papi index` | Build/update retrieval index (`--backend pqa|leann`) |

### `papi install` Options

- Components: `skill`, `prompts`, `mcp` (default: installs all 3).
- Targets: `--claude`, `--codex`, `--gemini` (plus `--repo` for `mcp` only).
- Prompts: `--copy` copies files (no symlinks).
- MCP: `--name`, `--leann-name`, `--embedding`.
- `--force` overwrites existing entries.

Gemini CLI note: skills are currently experimental; enable them in `~/.gemini/settings.json`:
`{"experimental": {"skills": true}}`.

## Paper Management

| Command | Description |
|---------|-------------|
| `papi add <arxiv-id-or-url>` | Add paper (name auto-generated; idempotent by arXiv ID) |
| `papi add --pdf PATH --title TEXT` | Add local PDF as a first-class paper |
| `papi add <arxiv> --name <n> --tags t1,t2` | Add with explicit name/tags |
| `papi add <arxiv> --update [--name <n>]` | Refresh an existing paper in-place |
| `papi add <arxiv> --duplicate` | Add another copy even if it already exists |
| `papi regenerate <name>` | Regenerate summaries/equations/tags |
| `papi regenerate <name> --overwrite name` | Regenerate auto-name |
| `papi regenerate --all` | Regenerate all papers |
| `papi remove <name>` | Remove a paper |

## Audit

| Command | Description |
|---------|-------------|
| `papi audit` | Audit all papers and flag obvious issues in generated content |
| `papi audit <names...>` | Audit only specific papers |
| `papi audit --limit N --seed S` | Audit a random sample (reproducible with `--seed`) |
| `papi audit --regenerate` | Regenerate all flagged papers (default overwrite: `summary,equations,tags`) |
| `papi audit --interactive` | Interactively pick which flagged papers to regenerate |
| `papi audit --regenerate --no-llm -o summary,equations` | Regenerate flagged papers without LLM (overwrite selected fields) |

## Export

| Command | Description |
|---------|-------------|
| `papi export <names...> --to ./dir` | Export to directory |
| `papi export <names...> --level summary` | Export summaries only |
| `papi export <names...> --level equations` | Export equations (best for code verification) |
| `papi export <names...> --level full` | Export full LaTeX source |

## Show Levels (stdout)

| Command | Description |
|---------|-------------|
| `papi show <names...>` | Show metadata (default) |
| `papi show <names...> --level summary` | Print summaries |
| `papi show <names...> --level equations` | Print equations (best for agent sessions) |
| `papi show <names...> --level tex` | Print LaTeX source |

## Notes

| Command | Description |
|---------|-------------|
| `papi notes <name>` | Open `{paper}/notes.md` in `$EDITOR` (creates if missing) |
| `papi notes <name> --print` | Print notes to stdout |

paperpipe supports two RAG backends for `papi ask`/`papi index`: PaperQA2 (`--backend pqa`) and LEANN (`--backend leann`).

## PaperQA2 Integration

| Command | Description |
|---------|-------------|
| `papi ask "question"` | Query papers via PaperQA2 RAG (default backend: `pqa`, if installed) |
| `papi ask "q" --pqa-llm MODEL --pqa-embedding EMB` | Specify PaperQA2 models |
| `papi ask "q" --pqa-summary-llm MODEL` | Use cheaper model for summarization |
| `papi ask "q" --pqa-verbosity 2 --pqa-evidence-k 15` | More verbose, more evidence |
| `papi ask "q" --pqa-rebuild-index` | Force full index rebuild |
| `papi ask "q" --format evidence-blocks` | Output JSON with `{answer, evidence[]}` |
| `papi models` | Probe which models work with your API keys |

First-class options: `--pqa-llm`, `--pqa-summary-llm`, `--pqa-embedding`, `--pqa-temperature`, `--pqa-verbosity`,
`--pqa-answer-length`, `--pqa-evidence-k`, `--pqa-max-sources`, `--pqa-timeout`, `--pqa-concurrency`,
`--pqa-rebuild-index`, `--pqa-retry-failed`.
Any other `pqa` args are passed through (e.g., `--agent.search_count 10`).

Notes:
- The first `papi ask` may take a while while PaperQA2 builds its index; by default it is cached under `<paper_db>/.pqa_index/`.
- By default, `papi ask` stages PDFs under `<paper_db>/.pqa_papers/` so RAG backends don't index generated Markdown.
- By default, `papi ask` syncs the PaperQA2 index with the staged PDFs (so newly added papers get indexed on the next ask).
- Override the index directory by passing `--agent.index.index_directory ...` through to `pqa`, or with `PAPERPIPE_PQA_INDEX_DIR`.
- Override PaperQA2's summarization/enrichment models with `PAPERPIPE_PQA_SUMMARY_LLM` and `PAPERPIPE_PQA_ENRICHMENT_LLM`
  (or use `--pqa-summary-llm` / `--parsing.enrichment_llm`).
- If PaperQA2 previously failed to index some PDFs, it records them as `ERROR` and won't retry automatically; re-run with
  `papi ask "..." --pqa-retry-failed` (or `--pqa-rebuild-index`).

### Index Build (No Question)

- `papi index` builds/updates the default retrieval index (PaperQA2 backend `pqa` by default if installed; same `--pqa-*` flags as `papi ask`).
- `papi index --backend leann` builds/updates the LEANN index (PDF-only) and passes extra args to `leann build` (except
  `--docs` / `--file-types`, which paperpipe controls).
  - Common LEANN build flags are also exposed as first-class options (e.g., `--leann-embedding-model`, `--leann-embedding-mode`,
    `--leann-doc-chunk-size`, `--leann-doc-chunk-overlap`, `--leann-num-threads`).

### Model combinations (practical examples)

See `README.md` → “PaperQA2 configuration” → “Model combinations” for copy/paste-ready examples.

## LEANN Integration (Local)

| Command | Description |
|---------|-------------|
| `papi leann-index` | Build/update LEANN index over staged PDFs (`<paper_db>/.pqa_papers/*.pdf`, PDF-only) |
| `papi index --backend leann` | Same as `papi leann-index` (plus `leann build` passthrough) |
| `papi ask "q" --backend leann` | Ask using LEANN RAG |
| `papi ask "q" --backend leann --leann-provider ollama --leann-model qwen3:8b` | Use local Ollama model |

Notes:
- If you use `--leann-provider anthropic`, your `leann` install must include the `anthropic` Python package
  (`pip install anthropic` in the same environment that runs `leann`).

Defaults:
- Indexing defaults come from `config.toml` / env vars (`PAPERPIPE_LEANN_EMBEDDING_MODEL`, `PAPERPIPE_LEANN_EMBEDDING_MODE`)
  unless you override via `papi index --backend leann --leann-embedding-*` (or pass raw `leann build --embedding-*` args).
- If `PAPERPIPE_LEANN_*` / `[leann]` are unset, paperpipe derives `--leann-provider/--leann-model` and
  `--leann-embedding-mode/--leann-embedding-model` from your global `[llm]` / `[embedding]` model settings when compatible
  (Ollama `ollama/...`, OpenAI `gpt-*`/`text-embedding-*`).
  Gemini `gemini/...` is supported for `--leann-provider/--leann-model` via OpenAI-compatible endpoint, but is not mapped
  for embeddings by default due to Gemini's 100-items-per-request embedding batch limit vs LEANN's current OpenAI batch size.
  If not compatible, it falls back to `ollama` + `olmo-3:7b` and `nomic-embed-text`.
- You can also override LEANN defaults via `config.toml`:
  ```toml
  [leann]
  llm_provider = "ollama"
  llm_model = "qwen3:8b"
  embedding_model = "nomic-embed-text"
  embedding_mode = "ollama"
  ```
- Env vars (override `config.toml`): `PAPERPIPE_LEANN_LLM_PROVIDER`, `PAPERPIPE_LEANN_LLM_MODEL`,
  `PAPERPIPE_LEANN_EMBEDDING_MODEL`, `PAPERPIPE_LEANN_EMBEDDING_MODE`

Common LEANN ask flags:
- `--leann-index`, `--leann-provider`, `--leann-model`
- `--leann-host` (Ollama), `--leann-api-base`/`--leann-api-key` (OpenAI-compatible)
- Retrieval tuning: `--leann-top-k`, `--leann-complexity`, `--leann-beam-width`, `--leann-prune-ratio`,
  `--leann-recompute/--leann-no-recompute`, `--leann-pruning-strategy`, `--leann-thinking-budget`,
  `--leann-interactive`
- Indexing behavior: `--leann-auto-index/--leann-no-auto-index` (default: auto-build index if missing)
- Passthrough: add `-- <leann args...>` to forward extra flags directly to `leann` (useful for debugging)

Embedding provider examples (indexing):
See `README.md` → “LEANN configuration” → “Embedding provider examples”.

## Per-Paper Files

Located at `<paper_db>/papers/{name}/`:

| File | Purpose | Best For |
|------|---------|----------|
| `equations.md` | Key equations with explanations | Code verification |
| `summary.md` | Coding-context overview | Understanding approach |
| `source.tex` | Full LaTeX source | Exact definitions |
| `meta.json` | Metadata + tags | Programmatic access |
| `paper.pdf` | PDF file | PaperQA2/LEANN RAG |
| `notes.md` | Your implementation notes | Gotchas/snippets |

## LLM Configuration (Optional)

```bash
export PAPERPIPE_LLM_MODEL="gemini/gemini-3-flash-preview"  # LiteLLM identifier
export PAPERPIPE_LLM_TEMPERATURE=0.3
```
