# Agent Integration Snippet (PaperPipe)

Run `papi docs` to output this snippet, or copy/paste into your repo's agent instructions file (`AGENTS.md`, or `CLAUDE.md` / `GEMINI.md` / etc).

**Tip:** Use `/papi-init` to automatically add/update this snippet in your project's agent instructions file.

```markdown
## Paper References (PaperPipe)

This repo implements methods from scientific papers. Papers are managed via `papi` (PaperPipe).

**Tool priority: Use `papi` CLI first. MCP RAG tools only when CLI is insufficient.**

### Commands

- `papi path` — DB location (default `~/.paperpipe/`; override via `PAPER_DB_PATH`)
- `papi list` — available papers; `papi list | grep -i "keyword"` to check if paper exists
- `papi add <arxiv_id_or_url>` — add a paper
- `papi show <paper> -l eq|tex|summary|tldr` — inspect paper content
- `papi search --rg "X"` — exact text; `--regex "pattern"` for OR/wildcards
- Direct files: `<paper_db>/papers/{paper}/equations.md`, `source.tex`, `summary.md`, `tldr.md`, `figures/`

### Decision Tree

| Question | Tool |
|----------|------|
| "What does paper X say about Y?" | `papi show X -l summary`, then `papi search --rg "Y"` |
| "Does my code match the paper?" | `/papi-verify` skill (uses `papi show -l eq`) |
| "Which paper mentions X?" | `papi search --rg "X"` first, then `leann_search()` if no hits |
| "Compare approaches across papers" | `/papi-compare` skill or `papi ask` |
| "Need citable quote with page number" | `retrieve_chunks()` (PQA MCP) |
| "Cross-paper synthesis" | `papi ask "..."` |

### When NOT to Use MCP RAG

- Paper name known → `papi show <paper> -l summary`
- Exact term search → `papi search --rg "term"`
- Checking equations → `papi show <paper> -l eq`
- Only use RAG when above methods fail or semantic matching required

### Search Escalation (cheapest first)

1. `papi search --rg "X"` — exact text, fast, no LLM
2. `papi search "X"` — ranked BM25 (requires `papi index --backend search` first)
3. `papi search --hybrid "X"` — ranked + exact boost
4. `leann_search()` — semantic search, returns file paths for follow-up
5. `retrieve_chunks()` — formal citations (DOI, page numbers)
6. `papi ask "..."` — full RAG synthesis

### MCP Tool Selection (when papi CLI insufficient)

| Tool | Speed | Output | Best For |
|------|-------|--------|----------|
| `leann_search(index, query, top_k)` | Fast | Snippets + file paths | Exploration, finding which paper to dig into |
| `retrieve_chunks(query, index, k)` | Slower | Chunks + formal citations | Verification, citing specific claims |
| `papi ask "..."` | Slowest | Synthesized answer | Cross-paper questions, "what does literature say" |

- Check indexes: `leann_list()` or `list_pqa_indexes()`
- Embedding priority: Voyage AI → Google/Gemini → OpenAI → Ollama

### Export (if agent can't read ~/.paperpipe/)

```
papi export <papers...> --level equations --to ./paper-context/
papi export <papers...> --figures --to ./paper-context/  # include figures
```
```

<details>
<summary>Glossary (optional)</summary>

- **RAG** = retrieval‑augmented generation: retrieve passages first, then generate an answer grounded in those passages.
- **Embeddings** = vector representations used for semantic retrieval; changing the embedding model implies a new index.
- **MCP** = Model Context Protocol: agent/tool integration for retrieval without pasting PDFs into chat.

</details>
