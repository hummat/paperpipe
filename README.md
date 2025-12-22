# paperpipe

A unified paper database for coding agents + PaperQA2.

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

### With uv (recommended)

```bash
# Basic installation
uv pip install paperpipe

# With LLM support (for better summaries/equations)
uv pip install 'paperpipe[llm]'

# With PaperQA2 integration
uv pip install 'paperpipe[paperqa]'

# Everything
uv pip install 'paperpipe[all]'
```

Or install from source:
```bash
git clone https://github.com/hummat/paperpipe
cd paperpipe
uv pip install -e ".[all]"
```

### With pip

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

Or install from source:
```bash
git clone https://github.com/hummat/paperpipe
cd paperpipe
pip install -e ".[all]"
```

## Development

```bash
# Install app + dev tooling (ruff, pyright, pytest)
uv sync --group dev

uv run ruff check .
uv run pyright
uv run pytest -m "not integration"
```

## Quick Start

```bash
# Add papers (auto-tags from arXiv categories + LLM-generated semantic tags)
papi add 2303.13476 --name neuralangelo
papi add 2106.10689 --name neus
papi add 2112.03907 --name ref-nerf

# You can also pass arXiv URLs directly:
# papi add https://arxiv.org/abs/1706.03762 --name attention

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

## Database Structure

Default database root is `~/.paperpipe/` (override with `PAPER_DB_PATH`; see `papi path`).

```
<paper_db>/
├── index.json                    # Quick lookup index
├── papers/
│   ├── neuralangelo/
│   │   ├── meta.json             # Metadata + tags
│   │   ├── paper.pdf             # For PaperQA2
│   │   ├── source.tex            # Full LaTeX (if available)
│   │   ├── summary.md            # Coding-context summary
│   │   └── equations.md          # Key equations extracted
│   └── neus/
│       └── ...
```

## Integration with Coding Agents

> **Tip:** See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) for a ready-to-use snippet you can append to your
> repo's agent instructions file (for example `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`).

### Claude Code Skill

For [Claude Code](https://github.com/anthropics/claude-code) users, paperpipe includes a skill that
automatically activates when you ask about papers, verification, or equations.

**Install globally** (available in all projects):

```bash
# Symlink the skill to your global skills directory
mkdir -p ~/.claude/skills
ln -s /path/to/paperpipe/skill ~/.claude/skills/papi
```

### Codex CLI Skill

Codex CLI reads skills from `$CODEX_HOME/skills` (defaults to `~/.codex/skills`).

**Install globally** (available in all projects):

```bash
mkdir -p ~/.codex/skills
ln -s /path/to/paperpipe/skill ~/.codex/skills/papi
```

Restart Codex CLI after installing a new skill.

The skill lives in `skill/` and can also be copied directly to other projects.

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

## Commands

| Command | Description |
|---------|-------------|
| `papi add <arxiv-id-or-url>` | Add a paper (downloads PDF, LaTeX, generates summary) |
| `papi regenerate <name-or-arxiv-id-or-url>` | Regenerate summary/equations (and LLM tags when enabled) |
| `papi regenerate --all` | Regenerate summary/equations for all papers |
| `papi list [--tag TAG]` | List papers, optionally filtered by tag |
| `papi search <query>` | Search by title, tag, or ID |
| `papi show <name>` | Show paper details |
| `papi export <papers...>` | Export context files to a directory |
| `papi ask <query> [args]` | Query papers via PaperQA2 (supports all pqa args) |
| `papi models` | Probe which models work with your API keys |
| `papi tags` | List all tags with counts |
| `papi remove <name-or-arxiv-id-or-url>` | Remove a paper |
| `papi path` | Print database location |
| `--quiet/-q` | Suppress progress messages |
| `--verbose/-v` | Enable debug output |

## Tagging

Papers are automatically tagged from three sources:

1. **arXiv categories** → human-readable tags (cs.CV → computer-vision)
2. **LLM-generated** → semantic tags from title/abstract
3. **User-provided** → via `--tags` flag

```bash
# Auto-tags from arXiv + LLM
papi add 2303.13476 --name neuralangelo
# → tags: computer-vision, graphics, neural-radiance-field, sdf, hash-encoding

# Add custom tags
papi add 2303.13476 --name neuralangelo --tags my-project,priority
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

## Workflow Example

```bash
# 1. Build your paper collection
papi add 2303.13476 --name neuralangelo
papi add 2106.10689 --name neus  
papi add 2104.06405 --name volsdf

# 2. Research phase: use PaperQA2
papi ask "Compare the volume rendering approaches in NeuS, VolSDF, and Neuralangelo"

# 3. Implementation phase: export equations to project
cd ~/my-neural-surface-project
papi export neuralangelo neus volsdf --level equations --to ./paper-context/

# 4. In Claude Code / Codex / Gemini:
# "Compare my eikonal_loss() implementation with the formulations in paper-context/"
```

## Configuration

Set custom database location:
```bash
export PAPER_DB_PATH=/path/to/your/papers
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

## LLM Support

For better summaries and equation extraction, install with LLM support:

```bash
pip install 'paperpipe[llm]'
# or with uv:
uv pip install 'paperpipe[llm]'
```

This installs LiteLLM, which supports many providers. Set the appropriate API key:

```bash
export GEMINI_API_KEY=...      # For Gemini (default)
export OPENAI_API_KEY=...      # For OpenAI/GPT
export ANTHROPIC_API_KEY=...   # For Claude
```

paperpipe defaults to `gemini/gemini-3-flash-preview`. Override via:
```bash
export PAPERPIPE_LLM_MODEL=gpt-4o  # or any LiteLLM model identifier
```

Without LLM support, paperpipe falls back to:
- Metadata-based summaries
- Regex-based equation extraction

## PaperQA2 Integration

When both paperpipe and PaperQA2 are installed, they share the same PDFs:

```bash
# paperpipe stores PDFs in <paper_db>/papers/*/paper.pdf (see `papi path`)
# paperpipe ask routes to PaperQA2 for complex queries

papi ask "What optimizer settings do these papers recommend?"

# PaperQA uses LiteLLM model identifiers for `--llm` and `--embedding`.
# You can also pass through any other `pqa ask` flags after the query/options.
# By default, `papi ask` uses `pqa --settings default` to avoid failures caused by stale user
# settings files; pass `-s/--settings <name>` to use a specific PaperQA2 settings profile.
# `papi ask` also defaults to `--llm gemini/gemini-3-flash-preview` and `--embedding gemini/gemini-embedding-001`
# unless you pick a PaperQA2 settings profile with `-s/--settings` (in that case, the profile controls).
# If Pillow is not installed, `papi ask` also forces `--parsing.multimodal OFF` to avoid PDF
# image extraction errors; pass your own `--parsing...` args to override.
#
# Examples (specify LLM + embedding):
# Gemini 3 Flash + Google Embeddings
papi ask "Explain the architecture" --llm "gemini/gemini-3-flash-preview" --embedding "gemini/gemini-embedding-001"

# Gemini 3 Pro + Google Embeddings
papi ask "Give a detailed derivation of eq. 4 and explain implementation pitfalls" --llm "gemini/gemini-3-pro-preview" --embedding "gemini/gemini-embedding-001"

# Claude Sonnet 4.5 + Voyage AI Embeddings
papi ask "Compare the loss functions" --llm "claude-sonnet-4-5" --embedding "voyage/voyage-3-large"

# GPT-5.2 + OpenAI Embeddings
papi ask "How to implement eq 4?" --llm "gpt-5.2" --embedding "text-embedding-3-large"

# Pass any arbitrary PaperQA2 arguments (e.g., temperature, verbosity)
papi ask "Summarize the methods" --summary-llm gpt-4o-mini --temperature 0.2 --verbosity 2
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

## Non-arXiv Papers

PaperPipe currently focuses on arXiv ingestion (`papi add <arxiv-id-or-url>`). For papers not on arXiv you can still
store files for agents to read, but they will not show up in `papi list/search` unless you also add index/meta
entries.

```bash
PAPER_DB="$(papi path)"
mkdir -p "$PAPER_DB/papers/my-paper"
cp /path/to/paper.pdf "$PAPER_DB/papers/my-paper/paper.pdf"
# Create:
# - "$PAPER_DB/papers/my-paper/summary.md"
# - "$PAPER_DB/papers/my-paper/equations.md"
# (optional) "$PAPER_DB/papers/my-paper/source.tex"
```

## License

MIT
