---
name: papi
description: Help with paper references using paperpipe (papi). Use when user asks about papers, wants to verify code against a paper, needs paper context for implementation, or asks about equations/methods from literature.
allowed-tools: Read, Bash, Glob, Grep
---

# Paper Reference Assistant

Use when user mentions papers, arXiv IDs, equations, or "does this match the paper?"

## Setup

```bash
papi path   # DB location (default ~/.paperpipe/; override via PAPER_DB_PATH)
papi list   # available papers
```

## Decision Rules (Cheapest First)

1. `papi show <paper> -l eq|tex|summary` — prints to stdout
2. MCP tools for "top passages about X":
   - `leann_search(index_name, query, top_k)` — fast, returns snippets
   - `retrieve_chunks(query, index_name, k)` — slower, formal citations
   - Check indexes: `leann_list()` or `list_pqa_indexes()`
   - Embedding priority: Voyage AI → Google/Gemini → OpenAI → Ollama
3. `papi ask` — only when user explicitly requests RAG synthesis

## Code Verification

1. `papi show {name} -l eq` — compare symbols with implementation
2. `papi show {name} -l tex` — exact definitions if ambiguous
3. `papi notes {name}` — implementation gotchas

## Implementation Guidance

1. `papi show {name} -l summary` — high-level approach
2. `papi show {name} -l eq` — formulas to implement

## Cross-Paper Search

```bash
papi search --rg "query"              # exact text (fast, no LLM)
papi search --rg --regex "pattern"    # regex (OR, wildcards)
papi search "query"                   # ranked BM25
papi search --hybrid "query"          # ranked + exact boost
papi ask "question"                   # PaperQA2 RAG
papi ask "question" --backend leann   # LEANN RAG
```

## Adding Papers

```bash
papi add 2303.13476                   # arXiv ID
papi add https://arxiv.org/abs/...    # URL
papi add --pdf /path/to.pdf --title "Title"  # local PDF
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
