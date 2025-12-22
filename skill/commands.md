# papi Command Reference

## Core Commands

| Command | Description |
|---------|-------------|
| `papi path` | Print database location |
| `papi list` | List all papers with tags |
| `papi list --tag TAG` | List papers filtered by tag |
| `papi tags` | List all tags with counts |
| `papi search "query"` | Search by title, tag, or content |
| `papi show <name>` | Show paper details |

## Paper Management

| Command | Description |
|---------|-------------|
| `papi add <arxiv-id-or-url>` | Add paper (name auto-generated) |
| `papi add <arxiv> --name <n> --tags t1,t2` | Add with explicit name/tags |
| `papi regenerate <name>` | Regenerate summaries/equations/tags |
| `papi regenerate <name> --overwrite name` | Regenerate auto-name |
| `papi regenerate --all` | Regenerate all papers |
| `papi remove <name>` | Remove a paper |

## Export

| Command | Description |
|---------|-------------|
| `papi export <names...> --to ./dir` | Export to directory |
| `papi export <names...> --level summary` | Export summaries only |
| `papi export <names...> --level equations` | Export equations (best for code verification) |
| `papi export <names...> --level full` | Export full LaTeX source |

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
