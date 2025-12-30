# paperpipe

A unified paper database for coding agents + [PaperQA2](https://github.com/Future-House/paper-qa).

**The problem:** You want AI coding assistants (Claude Code, Codex CLI, Gemini CLI) to reference scientific papers while implementing algorithms. But:
- PDFs are token-heavy and lose equation fidelity
- PaperQA2 is great for research but not optimized for code verification
- No simple way to ask "does my code match equation 7?"

**The solution:** A local database that stores:
- PDFs (for PaperQA2 RAG queries)
- LaTeX source (for exact equation comparison)
- Summaries optimized for coding context
- Extracted equations with explanations

## Installation

Install from PyPI (use `uv pip` if you use uv; otherwise use `pip`):

```bash
# Basic installation
pip install paperpipe

# With LLM support (for better summaries/equations)
pip install 'paperpipe[llm]'

# With PaperQA2 integration
pip install 'paperpipe[paperqa]'

# With PaperQA2 + multimodal PDF parsing (images/tables; installs Pillow)
pip install 'paperpipe[paperqa-media]'

# Everything
pip install 'paperpipe[all]'
```

Install from source:
```bash
git clone https://github.com/hummat/paperpipe
cd paperpipe
pip install -e ".[all]"  # or: uv pip install -e ".[all]"
```

## Release (GitHub + PyPI)

This repo publishes to PyPI when a GitHub Release is published (see `.github/workflows/publish.yml`).

```bash
# Bump versions first (pyproject.toml + paperpipe.py), then:
make release
```

## Quick Start

```bash
# Add papers (names auto-generated from title; auto-tags from arXiv + LLM)
papi add 2303.13476 2106.10689 2112.03907

# Override auto-generated name with --name (single paper only):
papi add https://arxiv.org/abs/1706.03762 --name attention

# Re-adding the same arXiv ID is idempotent (skips). Use --update to refresh, or --duplicate for another copy:
papi add 1706.03762
papi add 1706.03762 --update --name attention
papi add 1706.03762 --duplicate

# Add local PDFs (non-arXiv)
papi add --pdf /path/to/paper.pdf --title "Some Paper" --tags my-project

# List papers
papi list
papi list --tag sdf

# Search
papi search "surface reconstruction"

# Export for coding session
papi export neuralangelo neus --level equations --to ./paper-context/

# Query with PaperQA2 (if installed)
papi ask "What are the key differences between NeuS and Neuralangelo loss functions?"
```

`papi ask` runs PaperQA2 (`pqa`) directly on your local paper database. The first query may take a while
while PaperQA2 builds its index; subsequent queries reuse it (stored at `<paper_db>/.pqa_index/` by default).
Override the index location by passing `--agent.index.index_directory ...` through to `pqa`, or with
`PAPERPIPE_PQA_INDEX_DIR`.
By default, `papi ask` indexes **PDFs only** (it avoids indexing paperpipe’s generated `summary.md` / `equations.md`
Markdown files by staging PDFs under `<paper_db>/.pqa_papers/`). If you previously ran `papi ask` and PaperQA2
indexed Markdown, delete `<paper_db>/.pqa_index/` once to force a clean rebuild.
If PaperQA2 previously failed to index a PDF, it records it as `ERROR` and will not retry automatically; re-run
with `papi ask "..." --retry-failed` (or `--rebuild-index` to rebuild the whole index).
You can also override the models PaperQA2 uses for summarization/enrichment with
`PAPERPIPE_PQA_SUMMARY_LLM` and `PAPERPIPE_PQA_ENRICHMENT_LLM` (or use `--summary-llm` / `--parsing.enrichment_llm`).

## Database Structure

Default database root is `~/.paperpipe/` (override with `PAPER_DB_PATH`; see `papi path`).

```
<paper_db>/
├── index.json                    # Quick lookup index
├── .pqa_papers/                  # PaperQA2 input staging (PDF-only; created on first `papi ask`)
├── .pqa_index/                   # PaperQA2 index cache (created on first `papi ask`)
├── papers/
│   ├── neuralangelo/
│   │   ├── meta.json             # Metadata + tags
│   │   ├── paper.pdf             # For PaperQA2
│   │   ├── source.tex            # Full LaTeX (if available)
│   │   ├── summary.md            # Coding-context summary
│   │   ├── equations.md          # Key equations extracted
│   │   └── notes.md              # Your implementation notes (created automatically)
│   └── neus/
│       └── ...
```

## Integration with Coding Agents

> **Tip:** See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) for a ready-to-use snippet you can append to your
> repo's agent instructions file (for example `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`).

### Claude Code / Codex CLI Skill

paperpipe includes a skill that automatically activates when you ask about papers,
verification, or equations. Install it for Claude Code and/or Codex CLI:

```bash
# Install for both Claude Code and Codex CLI
papi install-skill

# Or install for a specific CLI only
papi install-skill --claude
papi install-skill --codex
```

Restart your CLI after installing the skill.

### Optional: Shared Prompts / Commands

paperpipe also ships lightweight prompt templates you can invoke as:
- Claude Code: slash commands (from `~/.claude/commands/`)
- Codex CLI: prompts (from `~/.codex/prompts/`)

Install them with:

```bash
papi install-prompts
papi install-prompts --claude
papi install-prompts --codex
```

Usage:
- Claude Code: `/ground-with-paper`, `/compare-papers`, `/curate-paper-note`
- Codex CLI: `/prompts:ground-with-paper`, `/prompts:compare-papers`, `/prompts:curate-paper-note`

For Codex CLI prompts, attach exported context with `@...` (or paste output from `papi show ... --level ...`).

Most coding-agent CLIs can read local files directly. The best workflow is:

1. Use `papi` to build/manage your paper collection.
2. For code verification, have the agent read `{paper}/equations.md` (and `source.tex` when needed).
3. For research-y questions across many papers, use `papi ask` (PaperQA2).

Minimal snippet to add to your agent instructions:

```markdown
## Paper References (PaperPipe)

PaperPipe manages papers via `papi`. Find the active database root with:
`papi path`

Per-paper files are under `<paper_db>/papers/{paper}/`:
- `equations.md` — best for implementation verification
- `summary.md` — high-level overview
- `source.tex` — exact definitions (if available)

Use `papi search "query"` to find papers/tags quickly.
Use `papi ask "question"` for PaperQA2 multi-paper queries (if installed).
```

If you want paper context inside your repo (useful for agents that can’t access `~`), export it:

```bash
papi export neuralangelo neus --level equations --to ./paper-context/
```

If you want to paste context directly into a terminal agent session, print to stdout:

```bash
papi show neuralangelo neus --level eq
```

## Commands

| Command | Description |
|---------|-------------|
| `papi add <ids-or-urls...>` | Add one or more arXiv papers (idempotent by arXiv ID; use `--update`/`--duplicate` for existing) |
| `papi add --pdf PATH --title TEXT` | Add a local PDF as a first-class paper |
| `papi regenerate <papers...>` | Regenerate summary/equations/tags (use `--overwrite name` to rename) |
| `papi regenerate --all` | Regenerate for all papers |
| `papi audit [papers...]` | Audit generated summaries/equations and optionally regenerate flagged papers |
| `papi remove <papers...>` | Remove one or more papers (by name or arXiv ID/URL) |
| `papi list [--tag TAG]` | List papers, optionally filtered by tag |
| `papi search <query>` | Exact search (with fuzzy fallback if no exact matches) across title/tags/metadata + local summaries/equations (use `--exact` to disable fallback; `--tex` includes LaTeX) |
| `papi show <papers...>` | Show paper details or print stored content |
| `papi notes <paper>` | Open or print per-paper implementation notes (`notes.md`) |
| `papi export <papers...>` | Export context files to a directory |
| `papi ask <query> [opts]` | Query papers via PaperQA2 (first-class opts + passthrough) |
| `papi models` | Probe which models work with your API keys |
| `papi tags` | List all tags with counts |
| `papi path` | Print database location |
| `papi install-skill` | Install the papi skill for Claude Code / Codex CLI |
| `papi install-prompts` | Install shared prompts (Claude commands + Codex prompts) |
| `--quiet/-q` | Suppress progress messages |
| `--verbose/-v` | Enable debug output |

## Tagging

Papers are automatically tagged from three sources:

1. **arXiv categories** → human-readable tags (cs.CV → computer-vision)
2. **LLM-generated** → semantic tags from title/abstract
3. **User-provided** → via `--tags` flag

```bash
# Auto-tags from arXiv + LLM
papi add 2303.13476
# → name: neuralangelo, tags: computer-vision, graphics, neural-radiance-field, sdf, hash-encoding

# Add custom tags (and override auto-name)
papi add 2303.13476 --name my-neuralangelo --tags my-project,priority
```

## Export Levels

```bash
# Just summaries (smallest, good for overview)
papi export neuralangelo neus --level summary

# Equations only (best for code verification)
papi export neuralangelo neus --level equations

# Full LaTeX source (most complete)
papi export neuralangelo neus --level full
```

## Show Levels (stdout)

```bash
# Metadata (default)
papi show neuralangelo

# Print equations (for piping into agent sessions)
papi show neuralangelo neus --level eq

# Print summary / LaTeX
papi show neuralangelo --level summary
papi show neuralangelo --level tex
```

## Notes (per paper)

paperpipe creates a `notes.md` per paper for implementation notes, gotchas, and code snippets.

```bash
# Open in $EDITOR (creates notes.md if missing)
papi notes neuralangelo

# Print notes to stdout (useful for piping into an agent session)
papi notes neuralangelo --print
```

## Workflow Example

```bash
# 1. Build your paper collection (names auto-generated)
papi add 2303.13476 2106.10689 2104.06405
# → neuralangelo, neus, volsdf

# 2. Research phase: use PaperQA2
papi ask "Compare the volume rendering approaches in NeuS, VolSDF, and Neuralangelo"

# 3. Implementation phase: export equations to project
cd ~/my-neural-surface-project
papi export neuralangelo neus volsdf --level equations --to ./paper-context/

# 4. In Claude Code / Codex / Gemini:
# "Compare my eikonal_loss() implementation with the formulations in paper-context/"

# 5. Clean up: remove papers you no longer need
papi remove volsdf neus
```

## Configuration

Set custom database location:
```bash
export PAPER_DB_PATH=/path/to/your/papers
```

### config.toml

In addition to env vars, you can use a persistent config file at `<paper_db>/config.toml`
(override the location with `PAPERPIPE_CONFIG_PATH`).

Precedence is: **CLI flags > env vars > config.toml > built-in defaults**.

Example:
```toml
[llm]
model = "gemini/gemini-3-flash-preview"
temperature = 0.3

[embedding]
model = "gemini/gemini-embedding-001"

[paperqa]
settings = "default"
index_dir = "~/.paperpipe/.pqa_index"
summary_llm = "gemini/gemini-3-flash-preview"
enrichment_llm = "gemini/gemini-3-flash-preview"

[tags.aliases]
cv = "computer-vision"
nerf = "neural-radiance-field"
```

## Environment Setup

To use PaperQA2 via `papi ask` with the built-in default models, set the environment variables for your
chosen provider (PaperQA2 uses LiteLLM identifiers for `--llm` and `--embedding`).

| Provider | Required Env Var | Used For |
|----------|------------------|----------|
| **Google** | `GEMINI_API_KEY` | Gemini models & embeddings |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude models |
| **Voyage AI** | `VOYAGE_API_KEY` | Embeddings (recommended when using Claude) |
| **OpenAI** | `OPENAI_API_KEY` | GPT models & embeddings |
| **OpenRouter** | `OPENROUTER_API_KEY` | Access to 200+ models via unified API |

## LLM Support

For better summaries and equation extraction, install with LLM support:

```bash
pip install 'paperpipe[llm]'  # or: uv pip install 'paperpipe[llm]'
```

This installs LiteLLM, which supports many providers. Set the appropriate API key:

```bash
export GEMINI_API_KEY=...      # For Gemini (default)
export OPENAI_API_KEY=...      # For OpenAI/GPT
export ANTHROPIC_API_KEY=...   # For Claude
export OPENROUTER_API_KEY=...  # For OpenRouter (200+ models)
```

paperpipe defaults to `gemini/gemini-3-flash-preview`. Override via:
```bash
export PAPERPIPE_LLM_MODEL=gpt-4o  # or any LiteLLM model identifier
```

You can also tune LLM generation:
```bash
export PAPERPIPE_LLM_TEMPERATURE=0.3  # default: 0.3
```

Without LLM support, paperpipe falls back to:
- Metadata + section headings from LaTeX
- Regex-based equation extraction

## PaperQA2 Integration

When both paperpipe and [PaperQA2](https://github.com/Future-House/paper-qa) are installed, they share the same PDFs:

```bash
# paperpipe stores PDFs in <paper_db>/papers/*/paper.pdf (see `papi path`)
# `papi ask` stages PDFs under <paper_db>/.pqa_papers/ so PaperQA2 doesn't index generated Markdown.
# paperpipe ask routes to PaperQA2 for complex queries

papi ask "What optimizer settings do these papers recommend?"
```

### First-Class Options

The most common PaperQA2 options are exposed as first-class `papi ask` flags:

| Flag | Description |
|------|-------------|
| `--llm MODEL` | LLM for answer generation (LiteLLM id) |
| `--summary-llm MODEL` | LLM for evidence summarization (often cheaper) |
| `--embedding MODEL` | Embedding model for text chunks |
| `-t, --temperature FLOAT` | LLM temperature (0.0-1.0) |
| `-v, --verbosity INT` | Logging level (0-3; 3 = log all LLM calls) |
| `--answer-length TEXT` | Target answer length (e.g., "about 200 words") |
| `--evidence-k INT` | Number of evidence pieces to retrieve (default: 10) |
| `--max-sources INT` | Max sources to cite in answer (default: 5) |
| `--timeout FLOAT` | Agent timeout in seconds (default: 500) |
| `--concurrency INT` | Indexing concurrency (default: 1) |
| `--rebuild-index` | Force full index rebuild |
| `--retry-failed` | Retry previously failed documents |

Any additional arguments are passed through to `pqa` (e.g., `--agent.search_count 10`).

```bash
# Examples with first-class options:

# Use a cheaper model for summarization
papi ask "Compare the loss functions" --llm gpt-4o --summary-llm gpt-4o-mini

# Increase verbosity and evidence retrieval
papi ask "Explain eq. 4" -v 2 --evidence-k 15 --max-sources 8

# Shorter answers with lower temperature
papi ask "Summarize the methods" --answer-length "about 50 words" -t 0.1

# Force index rebuild after adding new papers
papi ask "What's new?" --rebuild-index

# Specific model combinations:
# Gemini 3 Flash + Google Embeddings
papi ask "Explain the architecture" --llm "gemini/gemini-3-flash-preview" --embedding "gemini/gemini-embedding-001"

# Claude Sonnet 4.5 + Voyage AI Embeddings
papi ask "Compare the loss functions" --llm "claude-sonnet-4-5" --embedding "voyage/voyage-3-large"

# GPT-5.2 + OpenAI Embeddings
papi ask "How to implement eq 4?" --llm "gpt-5.2" --embedding "text-embedding-3-large"

# OpenRouter (access 200+ models via unified API)
papi ask "Explain the method" --llm "openrouter/anthropic/claude-sonnet-4" --embedding "openrouter/openai/text-embedding-3-large"
```

By default, `papi ask` uses `pqa --settings default` to avoid failures caused by stale user
settings files; pass `-s/--settings <name>` to use a specific PaperQA2 settings profile.
If Pillow is not installed, `papi ask` forces `--parsing.multimodal OFF` to avoid PDF
image extraction errors; pass your own `--parsing...` args to override.

### Model Probing

To see which model ids work with your currently configured API keys (this makes small live API calls):

```bash
papi models
# (default: probes one "latest" completion model and one embedding model per provider for
# which you have an API key set; pass `latest` (or `--preset latest`) to probe a broader list.)
# or probe specific models only:
papi models --kind completion --model gemini/gemini-3-flash-preview --model gemini/gemini-2.5-flash --model gpt-4o-mini
papi models --kind embedding --model gemini/gemini-embedding-001 --model text-embedding-3-small
# probe "latest" defaults (gpt-5.2/5.1, gemini 3 preview, claude-sonnet-4-5; plus text-embedding-3-large if enabled):
papi models latest
# probe "last-gen" defaults (gpt-4.1/4o, gemini 2.5, older/smaller embeddings; Claude 3.5 is retired):
papi models last-gen
# probe a broader superset:
papi models all
# show underlying provider errors (noisy):
papi models --verbose
```

## Non-arXiv Papers

You can ingest local PDFs as first-class entries:

```bash
papi add --pdf /path/to/paper.pdf --title "Some Paper"
papi add --pdf ./paper.pdf --title "Some Paper" --name some-paper --tags my-project
```

## Development

```bash
# Install app + dev tooling (ruff, pyright, pytest)
make deps

# Format + lint + typecheck + unit tests
make check
```

## Credits

- **[PaperQA2](https://github.com/Future-House/paper-qa)** by Future House — the RAG engine powering `papi ask`.
  *Skarlinski et al., "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge", 2024.*
  [arXiv:2409.13740](https://arxiv.org/abs/2409.13740)

## License

MIT (see [LICENSE](LICENSE))
