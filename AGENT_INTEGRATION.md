# Agent Integration Snippet (PaperPipe)

Add this section to your project's agent instructions file:
- Preferred: `AGENTS.md`
- Also works: `CLAUDE.md`, `GEMINI.md`, or your agent’s equivalent

---

## Paper References (PaperPipe)

This project implements methods from scientific papers. Papers are managed via `papi` (paperpipe).

### Paper Database Location

Default database root is `~/.paperpipe/`, but it may be overridden (e.g. via `PAPER_DB_PATH`).
Prefer discovering the active location with:

```bash
papi path
```

Per-paper files live at: `<paper_db>/papers/{paper}/`

- `meta.json` — metadata + tags
- `summary.md` — coding-context overview
- `equations.md` — key equations + explanations (best for implementation verification)
- `notes.md` — implementation notes (yours; created automatically)
- `source.tex` — full LaTeX (if available)
- `paper.pdf` — PDF (used by PaperQA2)
- `<paper_db>/.pqa_papers/` — PaperQA2 input staging (PDF-only; created on first `papi ask`)
- `<paper_db>/.pqa_index/` — PaperQA2 index cache (created on first `papi ask`; override via `PAPERPIPE_PQA_INDEX_DIR`)

### When to Use What

| Task | Best source |
|------|-------------|
| “Does my code match the paper?” | Read `{paper}/equations.md` (and/or `{paper}/source.tex`) |
| “What’s the high-level approach?” | Read `{paper}/summary.md` |
| “Find the exact formulation / definitions” | Read `{paper}/source.tex` |
| “Which papers discuss X?” | Run `papi search "X"` (fast) or `papi ask "X"` (default backend: PaperQA2; optional: `--backend leann`) |
| “Compare methods across papers” | Load multiple `{paper}/equations.md` files |
| “Do the generated summaries/equations look sane?” | Run `papi audit` (and optionally regenerate flagged papers) |

### Useful Commands

```bash
# List papers and tags
papi list
papi tags

# Search by title, tag, or content
papi search "sdf loss"

# Export equations/summaries into the repo for a coding session
papi export neuralangelo neus --level equations --to ./paper-context/

# Or print directly to stdout for pasting into a terminal agent session
papi show neuralangelo neus --level eq

# Open or print per-paper implementation notes
papi notes neuralangelo
papi notes neuralangelo --print

# Add papers (arXiv) / regenerate; use --no-llm to avoid LLM calls
papi add 2303.13476                      # name auto-generated
papi add 2303.13476 --name neuralangelo  # or explicit name
papi add 2303.13476 --update             # refresh existing paper in-place
papi add 2303.13476 --duplicate          # add a second copy (-2/-3 suffix)
papi add --pdf ./paper.pdf --title "Some Paper" --tags my-project  # local PDF ingest
papi regenerate neuralangelo --no-llm

# Audit generated content for obvious issues (and optionally regenerate flagged papers)
papi audit
papi audit --limit 5 --seed 0
papi audit --regenerate --no-llm -o summary,equations,tags
```

### LLM Configuration (Optional)

```bash
export PAPERPIPE_LLM_MODEL="gemini/gemini-3-flash-preview"  # any LiteLLM identifier
export PAPERPIPE_LLM_TEMPERATURE=0.3                        # default: 0.3
```

Without LLM, paperpipe falls back to metadata + section headings + regex equation extraction.

### Code Verification Workflow

1. Identify the referenced paper(s) (comments, function names, README, etc.)
2. Read `{paper}/equations.md` and compare symbol-by-symbol with the implementation
3. If ambiguous, confirm definitions/assumptions in `{paper}/source.tex`
4. If the question is broad or spans multiple papers, run `papi ask "..."` (default backend: PaperQA2; optional: `--backend leann`)

### Optional: Shared Prompts / Commands

paperpipe ships prompt templates you can install into your agent CLI:

```bash
papi install prompts
papi install prompts --gemini
```

Usage:
- Claude Code: `/papi`, `/verify-with-paper`, `/ground-with-paper`, `/compare-papers`, `/curate-paper-note`
- Codex CLI: `/prompts:papi`, `/prompts:verify-with-paper`, `/prompts:ground-with-paper`, `/prompts:compare-papers`, `/prompts:curate-paper-note`
- Gemini CLI: `/papi`, `/papi-run`, `/verify-with-paper`, `/ground-with-paper`, `/compare-papers`, `/curate-paper-note`
- Gemini CLI (papi helpers): `/papi-path`, `/papi-list`, `/papi-tags`, `/papi-search`, `/papi-show-summary`, `/papi-show-eq`, `/papi-show-tex`

Notes:
- Codex CLI: attach exported context with `@...` or paste output from `papi show ... --level ...`.
- Gemini CLI: inject files/directories with `@{...}` or paste output from `papi show ... --level ...`.
- Gemini CLI skills are experimental; enable them in `~/.gemini/settings.json`: `{"experimental": {"skills": true}}`.

### Decision Rules (Use the Cheapest Thing That Works)

1. If you know the paper: read `{paper}/equations.md` (and `source.tex` for definitions).
2. If you need to pull paper content into chat quickly: use Gemini `/papi-show-eq` / `/papi-show-tex`.
3. If you need cross-paper retrieval (raw chunks + citations): use the MCP tool (`retrieve_chunks`).
4. Avoid `papi ask` unless explicitly requested (it runs an LLM loop).

### Optional: MCP Server (Retrieval-Only)

paperpipe can install MCP servers for retrieval-only workflows:
- `papi mcp-server` (PaperQA2 retrieval: raw chunks + citations)
- `papi leann-mcp-server` (LEANN search: wraps `leann_mcp` from the paper DB directory)

```bash
papi install mcp          # Claude (via `claude mcp add`) + Codex (via `codex mcp add`) + Gemini (via `gemini mcp add`)
papi install mcp --repo   # Repo-local .mcp.json (Claude) + .gemini/settings.json (Gemini)
```

Useful flags:
- Targets: `--claude`, `--codex`, `--gemini`, `--repo`
- Names: `--name <paperqa>` and `--leann-name <leann>`
- PaperQA2 embedding: `--embedding <model-id>` (sets `PAPERQA_EMBEDDING`)
- `--force` overwrites existing entries

Build indexes outside MCP:

```bash
papi index                 # PaperQA2 index
papi index --backend leann  # LEANN index
```
