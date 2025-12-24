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
2. Read `{db}/papers/{name}/equations.md` — compare symbol-by-symbol with implementation
3. If ambiguous, check `{db}/papers/{name}/source.tex` for exact definitions
4. Check `{db}/papers/{name}/notes.md` for local implementation gotchas (or run `papi notes {name}`)

### 4. For implementation guidance

1. Read `{db}/papers/{name}/summary.md` for high-level approach
2. Read `{db}/papers/{name}/equations.md` for formulas to implement
3. Cross-reference with `source.tex` if equation details are unclear

### 5. For cross-paper questions

```bash
papi search "query"      # fast text search
papi ask "question"      # PaperQA2 RAG (if installed)
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
