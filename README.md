# paperpipe

paperpipe is a local paper database optimized for coding work: keep PDFs + (when available) LaTeX source, and
generate coding-oriented summaries + extracted equations. Optional RAG backends: [PaperQA2](https://github.com/Future-House/paper-qa) and LEANN.

**Most users use paperpipe for:**
- building a curated local library (`papi add`, `papi list`, `papi search`)
- pulling high-fidelity context for implementation and verification (`papi show --level eq`, `papi export --level equations`)
- keeping per-paper implementation notes (`papi notes`)

## Installation

### Install a global CLI (recommended)

Install as a standalone CLI tool using [uv](https://docs.astral.sh/uv/):

```bash
# Basic installation
uv tool install paperpipe

# Optional features (pick what you need)
uv tool install paperpipe --with "paperpipe[llm]"      # better summaries/equations via LLMs
uv tool install paperpipe --with "paperpipe[paperqa]"  # `papi ask` via PaperQA2
uv tool install paperpipe --with "paperpipe[leann]"    # `papi ask` via local LEANN backend
uv tool install paperpipe --with "paperpipe[mcp]"      # MCP server integrations (Python 3.11+)
uv tool install paperpipe --with "paperpipe[all]"
```

### Project install

Install from PyPI (use `uv pip` if you use uv; otherwise use `pip`):

```bash
# Basic installation
pip install paperpipe

pip install 'paperpipe[llm]'         # better summaries/equations via LLMs
pip install 'paperpipe[paperqa]'     # `papi ask` via PaperQA2
pip install 'paperpipe[paperqa-media]'  # PaperQA2 + multimodal PDF parsing (installs Pillow)
pip install 'paperpipe[leann]'       # `papi ask` via LEANN (local)
pip install 'paperpipe[mcp]'         # MCP server integrations (Python 3.11+)
pip install 'paperpipe[all]'
```

Install from source:

```bash
git clone https://github.com/hummat/paperpipe
cd paperpipe
pip install -e ".[all]"  # or: uv pip install -e ".[all]"
```

## Quick Start

```bash
# Add papers from arXiv (names auto-generated from title; tags from arXiv + optional LLM)
papi add 2303.13476 2106.10689

# List and search
papi list
papi list --tag sdf
papi search "surface reconstruction"

# Print equations to stdout (good for pasting into agent sessions / piping)
papi show neuralangelo --level eq

# Export context into a repo (good when your agent can't read from `~`)
papi export neuralangelo neus --level equations --to ./paper-context/
```

If you installed a RAG backend, you can also ask cross-paper questions:

```bash
papi ask "What are the key differences between NeuS and Neuralangelo loss functions?"
```

## Where data lives

Default database root is `~/.paperpipe/` (override with `PAPER_DB_PATH`; see `papi path`).

```
<paper_db>/
├── index.json                    # Quick lookup index
├── .pqa_papers/                  # Staged PDFs for RAG backends (PDF-only; created on first `papi ask`)
├── .pqa_index/                   # PaperQA2 index cache (created on first `papi ask`)
├── papers/
│   ├── neuralangelo/
│   │   ├── meta.json             # Metadata + tags
│   │   ├── paper.pdf             # For RAG backends (PaperQA2/LEANN)
│   │   ├── source.tex            # Full LaTeX (if available)
│   │   ├── summary.md            # Coding-context summary
│   │   ├── equations.md          # Key equations extracted
│   │   └── notes.md              # Your implementation notes (created automatically)
│   └── neus/
│       └── ...
```

## Core commands (most-used)

| Command | Description |
|---------|-------------|
| `papi add <ids-or-urls...>` | Add arXiv papers (idempotent by arXiv ID; use `--update`/`--duplicate` for existing) |
| `papi add --pdf PATH --title TEXT` | Add a local PDF as a first-class paper |
| `papi list [--tag TAG]` | List papers (optionally filter by tag) |
| `papi search <query>` | Search across title/tags/metadata + stored summaries/equations (use `--tex` to include LaTeX) |
| `papi show <papers...>` | Show metadata or print stored content (`--level eq|summary|tex`) |
| `papi export <papers...> --to DIR` | Export context files into a directory (`--level summary|equations|full`) |
| `papi notes <paper>` | Open or print per-paper implementation notes (`notes.md`) |
| `papi regenerate <papers...>` | Regenerate summary/equations/tags |
| `papi remove <papers...>` | Remove papers |
| `papi ask <query>` | Ask cross-paper questions (requires PaperQA2 and/or LEANN) |
| `papi index` | Build/update the retrieval index (for `papi ask`) |
| `papi tags` | List tags with counts |
| `papi path` | Print database location |

For everything else: `papi --help` and `papi <command> --help`.

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

## Export levels

```bash
# Just summaries (smallest, good for overview)
papi export neuralangelo neus --level summary

# Equations only (best for code verification)
papi export neuralangelo neus --level equations

# Full LaTeX source (most complete)
papi export neuralangelo neus --level full
```

## Show levels (stdout)

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

## Optional: LLM support (better summaries/equations)

Install with LLM support:

```bash
pip install 'paperpipe[llm]'  # or: uv pip install 'paperpipe[llm]'
```

This installs LiteLLM, which supports many providers. Set the appropriate API key:

```bash
export GEMINI_API_KEY=...      # For Gemini (default)
export OPENAI_API_KEY=...      # For OpenAI/GPT
export ANTHROPIC_API_KEY=...   # For Claude
export VOYAGE_API_KEY=...      # For Voyage embeddings (recommended when using Claude)
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

## Optional: `papi ask` (RAG backends)

If you installed a backend, `papi ask` can answer cross-paper questions:

```bash
papi ask "What optimizer settings do these papers recommend?"
```

Backends:
- **PaperQA2** (`--backend pqa`, default if installed): agentic synthesis with citations.
- **LEANN** (`--backend leann`): local retrieval/search backend (no PaperQA2 dependency).

For explicit index builds without asking a question, use `papi index`.
PaperQA2 uses LiteLLM identifiers for `--pqa-llm` / `--pqa-embedding` and the same API key env vars as the LLM
section (including `VOYAGE_API_KEY` if you use Voyage embeddings).

### PaperQA2 (default backend if installed)

When both paperpipe and [PaperQA2](https://github.com/Future-House/paper-qa) are installed, they share the same PDFs:

```bash
# paperpipe stores PDFs in <paper_db>/papers/*/paper.pdf (see `papi path`)
# `papi ask` stages PDFs under <paper_db>/.pqa_papers/ so PaperQA2 doesn't index generated Markdown.
# `papi ask --backend pqa` routes to PaperQA2 for agentic synthesis/citations

papi ask "What optimizer settings do these papers recommend?"
```

Index/caching notes:
- First run builds an index under `<paper_db>/.pqa_index/` and stages PDFs under `<paper_db>/.pqa_papers/` (PDF-only).
- Override the index location with `PAPERPIPE_PQA_INDEX_DIR`.
- If you accidentally indexed the wrong content (or changed embeddings), delete `<paper_db>/.pqa_index/` once to force a clean rebuild.
- If some PDFs failed indexing and are recorded as `ERROR`, re-run with `--pqa-retry-failed` (or `--pqa-rebuild-index` to rebuild everything).

#### Common options

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

#### Index builds (no question)

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

### LEANN (local backend)

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

## Optional: model probing

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

## Optional: coding agent integrations

See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) for a ready-to-paste snippet for your repo’s agent instructions.

### Skill + prompts

```bash
# Install everything (skill + prompts + mcp)
papi install

# Or install for a specific CLI only
papi install skill --claude
papi install skill --codex
papi install skill --gemini
```

Install only prompt templates/commands (no skill):

```bash
papi install prompts
```

Gemini CLI note: skills are currently experimental; enable them in `~/.gemini/settings.json`:
`{"experimental": {"skills": true}}`.

Restart your CLI after installing the skill.

### MCP servers (retrieval-only tools)

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

### Usage (Claude Code / Codex CLI / Gemini CLI)

1. Build/update the PaperQA2 index once (outside MCP): `papi index --pqa-embedding text-embedding-3-small`
2. In your agent CLI, call the MCP tool `retrieve_chunks` with your query (the agent does synthesis)
3. If retrieval looks wrong, call `get_pqa_index_status` to inspect what’s indexed

If your CLI supports a debug flag, enable it. For Claude Code: `claude --debug` (note: `--mcp-debug` is deprecated).

## Non-arXiv Papers

You can ingest local PDFs as first-class entries:

```bash
papi add --pdf /path/to/paper.pdf --title "Some Paper"
papi add --pdf ./paper.pdf --title "Some Paper" --name some-paper --tags my-project
```

## Development (contributors)

```bash
# Install app + dev tooling (ruff, pyright, pytest)
make deps

# Format + lint + typecheck + unit tests
make check
```

## Release (maintainers)

This repo publishes to PyPI when a GitHub Release is published (see `.github/workflows/publish.yml`).

```bash
# Bump versions first (pyproject.toml + paperpipe.py), then:
make release
```

## Credits

- **[PaperQA2](https://github.com/Future-House/paper-qa)** by Future House — the RAG backend powering `papi ask --backend pqa`.
  *Skarlinski et al., "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge", 2024.*
  [arXiv:2409.13740](https://arxiv.org/abs/2409.13740)
- **LEANN** (`leann-core`, `leann-backend-hnsw`) — the local RAG backend powering `papi ask --backend leann`.

## License

MIT (see [LICENSE](LICENSE))
