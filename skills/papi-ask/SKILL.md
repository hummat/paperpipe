---
name: papi-ask
description: Query papers using RAG (PaperQA2 or LEANN). Use when user needs synthesized answers from papers, asks "what does paper X say about Y", or needs cited responses.
---

# Query Papers via RAG

Use `papi ask` for questions requiring synthesis across papers or cited answers.

## Cost-Aware Retrieval

Before using RAG, consider cheaper alternatives:

1. **Exact match**: `papi search --rg "query"` — fast, no LLM
2. **Ranked search**: `papi search "query"` — BM25 ranking
3. **Direct read**: `papi show <paper> -l eq|tex|summary` — if you know the paper

Use `papi ask` only when:
- User explicitly requests RAG/synthesis
- Question spans multiple papers
- Search/show cannot answer

## Commands

```bash
# PaperQA2 (default) — full RAG with citations
papi ask "question"

# LEANN — faster semantic search + LLM
papi ask "question" --backend leann

# Structured output for programmatic use
papi ask "question" --format evidence-blocks
```

## MCP Tools (if available)

For quick retrieval without full RAG:
- `leann_search(index_name, query, top_k)` — fast semantic search
- `retrieve_chunks(query, index_name, k)` — PaperQA2 chunks with citations

Check available indexes: `leann_list()` or `list_pqa_indexes()`

## Output

RAG answers include:
- Synthesized response
- Citations with page/section references
- Confidence indicators
