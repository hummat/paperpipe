# Agent Integration Snippet (PaperPipe)

Copy/paste this into your repo’s agent instructions file (`AGENTS.md`, or `CLAUDE.md` / `GEMINI.md` / etc):

```markdown
## Paper References (PaperPipe)

This repo implements methods from scientific papers. Papers are managed via `papi` (PaperPipe).

- Paper DB root: run `papi path` (default `~/.paperpipe/`; override via `PAPER_DB_PATH`).
- Inspect a paper (prints to stdout):
  - Equations (verification): `papi show <paper> -l eq`
  - Definitions (LaTeX): `papi show <paper> -l tex`
  - Overview: `papi show <paper> -l summary`
- Direct files (if needed): `<paper_db>/papers/{paper}/equations.md`, `source.tex`, `summary.md`

Rules:
- For “does this match the paper?”, use `papi show <paper> -l eq` / `-l tex` and compare symbols step-by-step.
- For “which paper mentions X?”:
  - Exact string hits (fast): `papi search --grep --fixed-strings "X"`
  - Ranked search (BM25): `papi search-index --rebuild` then `papi search --fts "X"`
  - Hybrid (ranked + exact boost): `papi search --hybrid "X"`
- If the agent can’t read `~/.paperpipe/`, export context into the repo: `papi export <papers...> --level equations --to ./paper-context/`.
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
