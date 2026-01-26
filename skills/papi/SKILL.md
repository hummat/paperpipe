---
name: papi
description: CLI reference for paperpipe (papi). Use BEFORE MCP RAG tools. For listing, searching, showing, adding papers.
allowed-tools: Read, Bash, Glob, Grep
---

# Paper Reference Assistant (CLI)

**Entry point skill.** Use `papi` CLI first; MCP RAG tools only when CLI is insufficient.

For specialized workflows, invoke dedicated skills:
- `/papi-ask` — RAG queries requiring synthesis
- `/papi-verify` — verify code against paper
- `/papi-compare` — compare papers for decision
- `/papi-ground` — ground responses with citations
- `/papi-curate` — create project notes

## Setup

```bash
papi path   # DB location (default ~/.paperpipe/; override via PAPER_DB_PATH)
papi list   # available papers
papi list | grep -i "keyword"  # check if paper exists before searching
```

## When NOT to Use MCP RAG

- Paper name known → `papi show <paper> -l summary`
- Exact term search → `papi search --rg "term"`
- Checking equations → `papi show <paper> -l eq`
- Only use RAG when above methods fail or semantic matching required

## Decision Tree

| Question | Tool |
|----------|------|
| "What does paper X say about Y?" | `papi show X -l summary`, then `papi search --rg "Y"` |
| "Does my code match the paper?" | `/papi-verify` skill |
| "Which paper mentions X?" | `papi search --rg "X"` first, then `leann_search()` if no hits |
| "Compare approaches across papers" | `/papi-compare` skill or `papi ask` |
| "Need citable quote with page number" | `retrieve_chunks()` (PQA MCP) |
| "Cross-paper synthesis" | `papi ask "..."` |

## Search Commands

```bash
papi search --rg "query"              # exact text (fast, no LLM)
papi search --rg --regex "pattern"    # regex (OR, wildcards)
papi search "query"                   # ranked BM25
papi search --hybrid "query"          # ranked + exact boost
papi ask "question"                   # PaperQA2 RAG
papi ask "question" --backend leann   # LEANN RAG
papi notes {name}                     # open/print implementation notes
```

## Search Escalation (cheapest first)

1. `papi search --rg "X"` — exact text, fast, no LLM
2. `papi search "X"` — ranked BM25 (requires `papi index --backend search` first)
3. `papi search --hybrid "X"` — ranked + exact boost
4. `leann_search()` — semantic search, returns file paths for follow-up
5. `retrieve_chunks()` — formal citations (DOI, page numbers)
6. `papi ask "..."` — full RAG synthesis

## MCP Tool Selection (when papi CLI insufficient)

| Tool | Speed | Output | Best For |
|------|-------|--------|----------|
| `leann_search(index, query, top_k)` | Fast | Snippets + file paths | Exploration, finding which paper to dig into |
| `retrieve_chunks(query, index, k)` | Slower | Chunks + formal citations | Verification, citing specific claims |
| `papi ask "..."` | Slowest | Synthesized answer | Cross-paper questions, "what does literature say" |

- Check indexes: `leann_list()` or `list_pqa_indexes()`
- Embedding priority: Voyage AI → Google/Gemini → OpenAI → Ollama

## Adding Papers

```bash
papi add 2303.13476                   # arXiv ID
papi add https://arxiv.org/abs/...    # URL
papi add --pdf /path/to.pdf           # local PDF
papi add --pdf "https://..."          # PDF from URL
papi add --from-file papers.bib       # bulk import
```

## Per-Paper Files

Located at `{db}/papers/{name}/`:

| File | Best For |
|------|----------|
| `equations.md` | Code verification |
| `summary.md` | Understanding approach |
| `source.tex` | Exact definitions |
| `notes.md` | Implementation gotchas |
| `figures/` | Architecture diagrams, plots |

If agent can't read `~/.paperpipe/`, export to repo: `papi export <papers...> --level equations --to ./paper-context/`
Use `--figures` to include extracted figures in export.

See `commands.md` for full reference.
