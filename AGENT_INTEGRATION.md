# Agent Integration Snippet (PaperPipe)

Run `papi docs` to output this snippet, or copy/paste into your repo's agent instructions file (`AGENTS.md`, or `CLAUDE.md` / `GEMINI.md` / etc).

**Tip:** Use `/papi-init` to automatically add/update this snippet in your project's agent instructions file.

```markdown
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
```

<details>
<summary>Glossary (optional)</summary>

- **RAG** = retrieval‑augmented generation: retrieve passages first, then generate an answer grounded in those passages.
- **Embeddings** = vector representations used for semantic retrieval; changing the embedding model implies a new index.
- **MCP** = Model Context Protocol: agent/tool integration for retrieval without pasting PDFs into chat.

</details>
