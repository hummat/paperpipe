# papi Command Reference

## Core Commands

| Command | Description |
|---------|-------------|
| `papi path` | Print database location |
| `papi list` | List all papers with tags |
| `papi list --tag TAG` | List papers filtered by tag |
| `papi tags` | List all tags with counts |
| `papi search "query"` | Search by title, tag, or content |
| `papi show <papers...>` | Show paper details or print stored content |

## Paper Management

| Command | Description |
|---------|-------------|
| `papi add <arxiv-id-or-url>` | Add paper (name auto-generated; idempotent by arXiv ID) |
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

## PaperQA2 Integration

| Command | Description |
|---------|-------------|
| `papi ask "question"` | Query papers via PaperQA2 RAG |
| `papi ask "q" --llm MODEL --embedding EMB` | Specify models |
| `papi models` | Probe which models work with your API keys |

## Per-Paper Files

Located at `<paper_db>/papers/{name}/`:

| File | Purpose | Best For |
|------|---------|----------|
| `equations.md` | Key equations with explanations | Code verification |
| `summary.md` | Coding-context overview | Understanding approach |
| `source.tex` | Full LaTeX source | Exact definitions |
| `meta.json` | Metadata + tags | Programmatic access |
| `paper.pdf` | PDF file | PaperQA2 RAG |

## LLM Configuration (Optional)

```bash
export PAPERPIPE_LLM_MODEL="gemini/gemini-3-flash-preview"  # LiteLLM identifier
export PAPERPIPE_LLM_TEMPERATURE=0.3
```
