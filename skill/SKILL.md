---
name: papi
description: Help with paper references using paperpipe (papi). Use when user asks about papers, wants to verify code against a paper, needs paper context for implementation, or asks about equations/methods from literature.
allowed-tools: Read, Bash, Glob, Grep
---

# Paper Reference Assistant

Use this skill when the user:
- Asks "does this match the paper?" or similar verification questions
- Mentions a paper name, arXiv ID, or method from literature
- Wants to implement or verify an algorithm from a paper
- Asks about equations, formulas, or methods from scientific papers
- Needs paper context loaded into the conversation

## Workflow

### Decision Rules (Use the Cheapest Thing That Works)

1. Prefer `papi show <paper> -l eq|tex|summary` (prints to stdout); use direct files under `{db}/papers/{name}/` if needed.
2. Prefer retrieval-only MCP chunks over full `papi ask` when you only need “top passages about X”.
3. Use `papi ask` only when explicitly requested to run a RAG backend (PaperQA2 via `--backend pqa`, or LEANN via `--backend leann`).

### 1. Find the paper database location

```bash
papi path
```

### 2. List available papers

```bash
papi list
```

Or search for specific topics:

```bash
papi search "surface reconstruction"
papi search --grep --fixed-strings "AdamW"
papi search-index --rebuild
papi search --fts "surface reconstruction"
```

### 2b. Audit generated content (optional)

If summaries/equations/tags look suspicious, run an audit to flag obvious issues:

```bash
papi audit
papi audit --limit 10 --seed 0
```

### 2c. LLM configuration (optional)

```bash
export PAPERPIPE_LLM_MODEL="gemini/gemini-3-flash-preview"
export PAPERPIPE_LLM_TEMPERATURE=0.3
```

### 3. For code verification

1. Identify which paper(s) the code references (check comments, function names, README)
2. Run `papi show {name} -l eq` — compare symbol-by-symbol with implementation
3. If ambiguous, run `papi show {name} -l tex` for exact definitions
4. Check `{db}/papers/{name}/notes.md` for local implementation gotchas (or run `papi notes {name}`)

### 4. For implementation guidance

1. Run `papi show {name} -l summary` for high-level approach
2. Run `papi show {name} -l eq` for formulas to implement
3. Cross-reference with `papi show {name} -l tex` if equation details are unclear

### 5. For cross-paper questions

```bash
papi search --grep --fixed-strings "query"  # exact text search (fast, no LLM)
papi search-index --rebuild                 # build/update ranked search index (SQLite FTS5 / BM25)
papi search --fts "query"                   # ranked search (BM25)
papi search --hybrid "query"                # ranked + exact-hit boost (FTS + grep)
papi index               # build/update PaperQA2 index (backend: pqa)
papi ask "question"      # PaperQA2 RAG (backend: pqa, if installed)
papi ask "question" --pqa-agent-type fake   # cheaper/deterministic retrieval
papi ask "question" --format evidence-blocks  # JSON output: answer + cited evidence snippets
papi ask "question" --pqa-raw               # debugging: show raw PaperQA2 output
papi index --backend leann
papi ask "question" --backend leann
```

## Adding New Papers

```bash
papi add 2303.13476                           # name auto-generated
papi add https://arxiv.org/abs/2303.13476     # URLs work too
papi add 2303.13476 --name my-custom-name     # override auto-name
papi add 2303.13476 --update                  # refresh existing paper in-place
papi add 2303.13476 --duplicate               # add a second copy (-2/-3 suffix)
papi add --pdf /path/to/paper.pdf --title "Some Paper" --tags my-project  # local PDF ingest
```

## See Also

Read `commands.md` in this skill directory for the full command reference.
