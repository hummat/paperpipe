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

### Global install (recommended)

Install as a standalone CLI tool using [uv](https://docs.astral.sh/uv/):

```bash
# Basic installation
uv tool install paperpipe

# With optional features
uv tool install paperpipe --with "paperpipe[llm]"
uv tool install paperpipe --with "paperpipe[all]"
```

### Project install

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

# With MCP server for Claude Code (requires Python 3.11+)
pip install 'paperpipe[mcp]'

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

`papi ask` defaults to PaperQA2 (`pqa`) and can also use LEANN (`--backend leann`).
For explicit index builds without asking a question, use `papi index` (see below).

The first PaperQA2 query may take a while while it builds its index; subsequent queries reuse it
(stored at `<paper_db>/.pqa_index/` by default).
Override the index location by passing `--agent.index.index_directory ...` through to `pqa`, or with
`PAPERPIPE_PQA_INDEX_DIR`.
By default, `papi ask` indexes **PDFs only** (it avoids indexing paperpipe’s generated `summary.md` / `equations.md`
Markdown files by staging PDFs under `<paper_db>/.pqa_papers/`). If you previously ran `papi ask` and PaperQA2
indexed Markdown, delete `<paper_db>/.pqa_index/` once to force a clean rebuild.
If PaperQA2 previously failed to index a PDF, it records it as `ERROR` and will not retry automatically; re-run
with `papi ask "..." --pqa-retry-failed` (or `--pqa-rebuild-index` to rebuild the whole index).
You can also override the models PaperQA2 uses for summarization/enrichment with
`PAPERPIPE_PQA_SUMMARY_LLM` and `PAPERPIPE_PQA_ENRICHMENT_LLM` (or use `--pqa-summary-llm` / `--parsing.enrichment_llm`).

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
# Install everything (skill + prompts + mcp)
papi install

# Or install for a specific CLI only
papi install skill --claude
papi install skill --codex
papi install skill --gemini
```

Gemini CLI note: skills are currently experimental; enable them in `~/.gemini/settings.json`:
`{"experimental": {"skills": true}}`.

Restart your CLI after installing the skill.

### Optional: Shared Prompts / Commands

paperpipe also ships lightweight prompt templates you can invoke as:
- Claude Code: slash commands (from `~/.claude/commands/`)
- Codex CLI: prompts (from `~/.codex/prompts/`)
- Gemini CLI: custom commands (from `~/.gemini/commands/`, can run `papi` via `!{...}`)

Install them with:

```bash
papi install prompts
papi install prompts --claude
papi install prompts --codex
papi install prompts --gemini
```

Usage:
- Claude Code: `/papi`, `/verify-with-paper`, `/ground-with-paper`, `/compare-papers`, `/curate-paper-note`
- Codex CLI: `/prompts:papi`, `/prompts:verify-with-paper`, `/prompts:ground-with-paper`, `/prompts:compare-papers`, `/prompts:curate-paper-note`
- Gemini CLI (prompts): `/papi`, `/papi-run`, `/verify-with-paper`, `/ground-with-paper`, `/compare-papers`, `/curate-paper-note`
- Gemini CLI (papi helpers): `/papi-path`, `/papi-list`, `/papi-tags`, `/papi-search`, `/papi-show-summary`, `/papi-show-eq`, `/papi-show-tex`

For Codex CLI prompts, attach exported context with `@...` (or paste output from `papi show ... --level ...`).
For Gemini CLI commands, inject files/directories with `@{...}` (or paste output from `papi show ... --level ...`).

### When to Use What (User + Agent)

Prefer the cheapest/highest-fidelity mechanism first:

- **Read files** (best fidelity): use `{paper}/equations.md` and `{paper}/source.tex` for implementation correctness.
- **Gemini custom commands** (fast, local): use `/papi-search` + `/papi-show-eq` to pull exact snippets into the chat.
- **Skill** (workflow guardrails): keeps the agent in the “read files / cite evidence / verify symbols” mode.
- **MCP retrieval** (cross-paper search): use when you need “top chunks about X” without running PaperQA2’s LLM loop.
- **`papi ask`** (slow/expensive): only when you explicitly want a RAG backend to do synthesis/answering.

### Optional: MCP Server Install

If you want fast retrieval-only search tools exposed to your coding agent, install MCP server config(s).
`papi install mcp` installs whichever servers are available in your environment (PaperQA2 and/or LEANN):

```bash
# Install for Claude Code (via `claude mcp add`) + Codex CLI (via `codex mcp add`) + Gemini CLI (via `gemini mcp add`)
papi install mcp

# Repo-local config files for Claude Code (.mcp.json) + Gemini CLI (.gemini/settings.json)
papi install mcp --repo

# Only install for specific targets
papi install mcp --codex
papi install mcp --claude
papi install mcp --gemini

# Customize server names (defaults: --name paperqa, --leann-name leann)
papi install mcp --name paperqa --leann-name leann

# Set the embedding model used by the PaperQA2 MCP server
papi install mcp --embedding text-embedding-3-small

# Overwrite existing entries
papi install mcp --force
```

Most coding-agent CLIs can read local files directly. The best workflow is:

1. Use `papi` to build/manage your paper collection.
2. For code verification, have the agent read `{paper}/equations.md` (and `source.tex` when needed).
3. For research-y questions across many papers, use `papi ask` (default backend: PaperQA2; optional: `--backend leann`).

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
Use `papi index` to build/update the retrieval index (PaperQA2 by default; `--backend leann` for LEANN).
Use `papi ask "question"` for multi-paper queries (default backend: PaperQA2; `--backend leann` optional).
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
| `papi leann-index` | Build/update a LEANN index over stored PDFs (PDF-only) |
| `papi index` | Build/update the default index (`--backend pqa|leann`) |
| `papi ask <query> [opts]` | Query papers via PaperQA2 (default) or LEANN (`--backend leann`) |
| `papi models` | Probe which models work with your API keys |
| `papi tags` | List all tags with counts |
| `papi path` | Print database location |
| `papi install [components...]` | Install integrations (components: `skill`, `prompts`, `mcp`) |
| `papi uninstall [components...]` | Uninstall integrations (components: `skill`, `prompts`, `mcp`) |
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
chosen provider (PaperQA2 uses LiteLLM identifiers for `--pqa-llm` and `--pqa-embedding`).

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
| `--pqa-llm MODEL` | LLM for answer generation (LiteLLM id) |
| `--pqa-summary-llm MODEL` | LLM for evidence summarization (often cheaper) |
| `--pqa-embedding MODEL` | Embedding model for text chunks |
| `--pqa-temperature FLOAT` | LLM temperature (0.0-1.0) |
| `--pqa-verbosity INT` | Logging level (0-3; 3 = log all LLM calls) |
| `--pqa-answer-length TEXT` | Target answer length (e.g., "about 200 words") |
| `--pqa-evidence-k INT` | Number of evidence pieces to retrieve (default: 10) |
| `--pqa-max-sources INT` | Max sources to cite in answer (default: 5) |
| `--pqa-timeout FLOAT` | Agent timeout in seconds (default: 500) |
| `--pqa-concurrency INT` | Indexing concurrency (default: 1) |
| `--pqa-rebuild-index` | Force full index rebuild |
| `--pqa-retry-failed` | Retry previously failed documents |

Any additional arguments are passed through to `pqa` (e.g., `--agent.search_count 10`).

### Index Builds (No Question)

Use `papi index` to build/update the retrieval index without asking a question:

- PaperQA2 (default): supports the same `--pqa-*` flags as `papi ask`; extra `pqa` args are passed through.
- LEANN: `papi index --backend leann [--leann-index NAME] [--leann-force]` (PDF-only); extra args are passed to
  `leann build` (except `--docs` / `--file-types`, which paperpipe controls).

```bash
# Examples with first-class options:

# Use a cheaper model for summarization
papi ask "Compare the loss functions" --pqa-llm gpt-4o --pqa-summary-llm gpt-4o-mini

# Increase verbosity and evidence retrieval
papi ask "Explain eq. 4" --pqa-verbosity 2 --pqa-evidence-k 15 --pqa-max-sources 8

# Shorter answers with lower temperature
papi ask "Summarize the methods" --pqa-answer-length "about 50 words" --pqa-temperature 0.1

# Build/update the PaperQA2 index explicitly (no question)
papi index
papi index --pqa-rebuild-index

# Specific model combinations:
# Gemini 3 Flash + Google Embeddings
papi ask "Explain the architecture" --pqa-llm "gemini/gemini-3-flash-preview" --pqa-embedding "gemini/gemini-embedding-001"

# Claude Sonnet 4.5 + Voyage AI Embeddings
papi ask "Compare the loss functions" --pqa-llm "claude-sonnet-4-5" --pqa-embedding "voyage/voyage-3-large"

# GPT-5.2 + OpenAI Embeddings
papi ask "How to implement eq 4?" --pqa-llm "gpt-5.2" --pqa-embedding "text-embedding-3-large"

# OpenRouter (access 200+ models via unified API)
papi ask "Explain the method" --pqa-llm "openrouter/anthropic/claude-sonnet-4" --pqa-embedding "openrouter/openai/text-embedding-3-large"
```

By default, `papi ask` uses `pqa --settings default` to avoid failures caused by stale user
settings files; pass `-s/--settings <name>` to use a specific PaperQA2 settings profile.
If Pillow is not installed, `papi ask` forces `--parsing.multimodal OFF` to avoid PDF
image extraction errors; pass your own `--parsing...` args to override.

## LEANN Integration (Local)

LEANN uses a separate local index stored under `<paper_db>/.leann/` (by default, `papers`). paperpipe indexes
only staged PDFs from `<paper_db>/.pqa_papers/` (no `*.md`).

Defaults can be overridden via env vars or `config.toml`:

```toml
[leann]
llm_provider = "ollama"
llm_model = "qwen3:8b"
embedding_model = "nomic-embed-text"
embedding_mode = "ollama"
```

Env vars (override `config.toml`):
- `PAPERPIPE_LEANN_LLM_PROVIDER`
- `PAPERPIPE_LEANN_LLM_MODEL`
- `PAPERPIPE_LEANN_EMBEDDING_MODEL`
- `PAPERPIPE_LEANN_EMBEDDING_MODE`

```bash
# Build/update the LEANN index explicitly
papi index --backend leann
# or:
papi leann-index

# Ask via LEANN (defaults: ollama + olmo-3:7b; embeddings: nomic-embed-text via ollama)
papi ask "..." --backend leann

# Common LEANN knobs (see `papi ask --help` for the full set)
papi ask "..." --backend leann --leann-provider ollama --leann-model qwen3:8b
papi ask "..." --backend leann --leann-host http://localhost:11434
papi ask "..." --backend leann --leann-top-k 12 --leann-complexity 64

# By default, `papi ask --backend leann` auto-builds the index if missing (disable with --leann-no-auto-index)
papi ask "..." --backend leann --leann-no-auto-index
```

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

## MCP Server (Claude Code / Codex CLI / Gemini CLI)

paperpipe supports MCP (Model Context Protocol) servers for retrieval-only workflows:
- **PaperQA2 retrieval** (`papi mcp-server`): raw chunks + citations over the cached PaperQA2 index.
- **LEANN search** (`papi leann-mcp-server`): wraps LEANN's `leann_mcp` server, running from your paper DB directory.

### Installation

```bash
# PaperQA2 MCP support (requires Python 3.11+)
pip install 'paperpipe[mcp]'

# LEANN MCP support (requires compiled LEANN backend)
pip install 'paperpipe[leann]'
```

### Setup (Recommended)

Use the installer (it installs all available MCP servers):

```bash
papi install mcp
papi install mcp --repo
```

### Setup (Manual)

Claude Code (project `.mcp.json`):

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "papi",
      "args": ["mcp-server"],
      "env": {
        "PAPERQA_EMBEDDING": "text-embedding-3-small"
      }
    },
    "leann": {
      "command": "papi",
      "args": ["leann-mcp-server"],
      "env": {}
    }
  }
}
```

Claude Code (user scope via CLI):

```bash
claude mcp add --transport stdio --env PAPERQA_EMBEDDING=text-embedding-3-small --scope user paperqa -- papi mcp-server
```

Codex CLI:

```bash
codex mcp add paperqa --env PAPERQA_EMBEDDING=text-embedding-3-small -- papi mcp-server
codex mcp add leann -- papi leann-mcp-server
```

Gemini CLI (user `~/.gemini/settings.json` or project `.gemini/settings.json`):

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "papi",
      "args": ["mcp-server"],
      "env": {
        "PAPERQA_EMBEDDING": "text-embedding-3-small"
      }
    },
    "leann": {
      "command": "papi",
      "args": ["leann-mcp-server"],
      "env": {}
    }
  }
}
```

Gemini CLI (user scope via CLI):

```bash
gemini mcp add --scope user --transport stdio --env PAPERQA_EMBEDDING=text-embedding-3-small paperqa -- papi mcp-server
gemini mcp add --scope user --transport stdio leann -- papi leann-mcp-server
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PAPERPIPE_PQA_INDEX_DIR` | `~/.paperpipe/.pqa_index` | Root directory containing PaperQA2 indices |
| `PAPERPIPE_PQA_INDEX_NAME` | `paperpipe_<embedding>` | Index name to query (subfolder under index dir) |
| `PAPERQA_EMBEDDING` | (from env or `papi` config) | Embedding model id (must match the index you built) |
| `PAPERQA_LLM` | `gpt-4o-mini` | Unused for retrieval-only tools |

### Available Tools

| Tool | Description |
|------|-------------|
| `retrieve_chunks` | Retrieve raw chunks + citations (no LLM answering) |
| `list_pqa_indexes` | List available PaperQA2 indices under the index dir |
| `get_pqa_index_status` | Show basic index stats (files, failures) |

### Usage in Claude Code

```
# Build/update the PaperQA2 index once (outside MCP)
> Run: papi index --pqa-embedding text-embedding-3-small

# Retrieve chunks (Claude does synthesis)
> Retrieve chunks: What methods exist for neural surface reconstruction?

# Check what's indexed
> Get pqa index status
```

Debug with: `claude --debug` (note: `--mcp-debug` is deprecated)

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
