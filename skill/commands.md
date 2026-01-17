# papi Command Reference

## Core Commands

| Command | Description |
|---------|-------------|
| `papi path` | Print database location |
| `papi docs` | Print agent integration snippet |
| `papi list [--tag TAG]` | List papers (optionally filtered) |
| `papi tags` | List all tags with counts |
| `papi search "query"` | Ranked search (BM25 if `search.db` exists) |
| `papi search --rg QUERY` | Exact text via ripgrep |
| `papi search --rg --regex PATTERN` | Regex patterns |
| `papi show <papers...> [-l eq\|tex\|summary]` | Print paper content |
| `papi notes <paper> [--print]` | Open/print implementation notes |
| `papi index [--backend pqa\|leann\|search]` | Build retrieval index |

## Paper Management

| Command | Description |
|---------|-------------|
| `papi add <arxiv-id-or-url>` | Add paper (idempotent) |
| `papi add --pdf PATH --title TEXT` | Add local PDF |
| `papi add --from-file <file>` | Import from JSON/BibTeX/text |
| `papi add <id> --update` | Refresh existing paper |
| `papi add <id> --figures` | Extract figures from LaTeX/PDF |
| `papi regenerate <name> [--all]` | Regenerate summaries/equations |
| `papi remove <name>` | Remove a paper |
| `papi audit [--regenerate]` | Flag/fix issues in generated content |
| `papi rebuild-index [--dry-run] [--validate]` | Rebuild index from paper directories |

## Export

| Command | Description |
|---------|-------------|
| `papi export <names...> --to ./dir` | Export to directory |
| `papi export ... --level summary\|equations\|full` | Control export depth |
| `papi export ... --figures` | Include extracted figures |

## RAG Queries

| Command | Description |
|---------|-------------|
| `papi ask "question"` | PaperQA2 RAG (default) |
| `papi ask "q" --backend leann` | LEANN RAG |
| `papi ask "q" --format evidence-blocks` | JSON output with citations |
| `papi ask "q" --pqa-agent-type fake` | Cheaper/deterministic |

Common flags: `--pqa-llm`, `--pqa-embedding`, `--pqa-rebuild-index`, `--leann-provider`, `--leann-model`.

## Per-Paper Files

Located at `<paper_db>/papers/{name}/`:

| File | Best For |
|------|----------|
| `equations.md` | Code verification |
| `summary.md` | Understanding approach |
| `source.tex` | Exact definitions |
| `notes.md` | Implementation gotchas |
| `paper.pdf` | RAG backends |
| `figures/` | Architecture diagrams, plots |
