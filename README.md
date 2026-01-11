# paperpipe

**The problem:** You're implementing a paper. You need the exact equations, want to verify your code matches the math, and your coding agent keeps hallucinating details. Reading PDFs is slow; copy-pasting LaTeX is tedious.

**The solution:** paperpipe maintains a local paper database with PDFs, LaTeX source (when available), extracted equations, and coding-oriented summaries. It integrates with coding agents (Claude Code, Codex, Gemini CLI) so they can ground their responses in actual paper content.

## Typical workflow

```bash
# 1. Add papers you're implementing
papi add 2303.08813                    # LoRA paper
papi add https://arxiv.org/abs/1706.03762  # Attention paper

# 2. Check what equations you need to implement
papi show lora --level eq             # prints equations to stdout

# 3. Verify your code matches the paper
#    (or let your coding agent do this via the /papi skill)
papi show lora --level tex            # exact LaTeX definitions

# 4. Ask cross-paper questions (requires RAG backend)
papi ask "How does LoRA differ from full fine-tuning in terms of parameter count?"

# 5. Keep implementation notes
papi notes lora                       # opens notes.md in $EDITOR
```

## Installation

```bash
# Basic (uv recommended)
uv tool install paperpipe

# With features
uv tool install paperpipe --with "paperpipe[llm]"      # better summaries via LLMs
uv tool install paperpipe --with "paperpipe[paperqa]"  # RAG via PaperQA2
uv tool install paperpipe --with "paperpipe[leann]"    # local RAG via LEANN
uv tool install paperpipe --with "paperpipe[mcp]"      # MCP server integrations (Python 3.11+)
uv tool install paperpipe --with "paperpipe[all]"      # everything
```

<details>
<summary>Alternative: pip install</summary>

```bash
pip install paperpipe
pip install 'paperpipe[llm]'
pip install 'paperpipe[paperqa]'
pip install 'paperpipe[paperqa-media]'  # PaperQA2 + multimodal PDF parsing (installs Pillow)
pip install 'paperpipe[leann]'
pip install 'paperpipe[mcp]'
pip install 'paperpipe[all]'
```
</details>

<details>
<summary>From source</summary>

```bash
git clone https://github.com/hummat/paperpipe && cd paperpipe
pip install -e ".[all]"
```
</details>

## What paperpipe stores

```
~/.paperpipe/                         # override with PAPER_DB_PATH
├── index.json
├── .pqa_papers/                      # staged PDFs for RAG (created on first `papi ask`)
├── .pqa_index/                       # PaperQA2 index cache
├── .leann/                           # LEANN index cache
├── papers/
│   └── lora/
│       ├── paper.pdf                 # for RAG backends
│       ├── source.tex                # full LaTeX (if available from arXiv)
│       ├── equations.md              # extracted equations with context
│       ├── summary.md                # coding-oriented summary
│       ├── meta.json                 # metadata + tags
│       └── notes.md                  # your implementation notes
```

**Why this structure matters:**
- `equations.md` — Key equations with variable definitions. Use for code verification.
- `source.tex` — Original LaTeX. Use when you need exact notation or the equation extraction missed something.
- `summary.md` — High-level overview focused on implementation (not literature review). Use for understanding the approach.
- `.pqa_papers/` — Staged PDFs only (no markdown) so RAG backends don't index generated content.

## Core commands

| Command | Purpose |
|---------|---------|
| `papi add <arxiv-id-or-url>` | Add a paper (downloads PDF + LaTeX, generates summary/equations) |
| `papi add --pdf file.pdf --title "..."` | Add a local PDF |
| `papi list` | List papers (filter with `--tag`) |
| `papi search "query"` | Search across titles, tags, summaries, equations (`--grep` exact, `--fts` ranked BM25) |
| `papi search-index` | Build/update ranked search index (`search.db`) |
| `papi show <paper> --level eq` | Print equations (best for agent sessions) |
| `papi show <paper> --level tex` | Print LaTeX source |
| `papi show <paper> --level summary` | Print summary |
| `papi export <papers...> --to ./dir` | Export context files into a repo (`--level summary\|equations\|full`) |
| `papi notes <paper>` | Open/print implementation notes |
| `papi regenerate <papers...>` | Regenerate summary/equations/tags |
| `papi remove <papers...>` | Remove papers |
| `papi ask "question"` | Cross-paper RAG query (requires PaperQA2 or LEANN) |
| `papi index` | Build/update the retrieval index |
| `papi tags` | List all tags |
| `papi path` | Print database location |

Run `papi --help` or `papi <command> --help` for full options.

Exact text search (fast, no LLM required):

```bash
papi search --grep "AdamW"
papi search --grep "Eq\\. 7"          # regex mode (escape if needed)
papi search --grep --fixed-strings "λ=0.1"
```

Ranked search (BM25 via SQLite FTS5, no LLM required):

```bash
papi search-index --rebuild           # builds <paper_db>/search.db
papi search --fts "surface reconstruction"
# Force the old in-memory scan (slower, no sqlite):
papi search --no-fts "surface reconstruction"
```

Hybrid ranked+exact search:

```bash
papi search --hybrid "surface reconstruction"
papi search --hybrid --show-grep-hits "surface reconstruction"
```

### What are FTS and BM25?

- **FTS** = *Full-Text Search*. Here it means SQLite’s FTS5 extension, which builds an inverted index so searches don’t
  have to rescan every file on every query.
- **BM25** = *Okapi BM25*, a standard relevance-ranking function used by many search engines. It ranks results based on
  term frequency, inverse document frequency, and document length normalization.

References (external):
```text
https://sqlite.org/fts5.html
https://en.wikipedia.org/wiki/Okapi_BM25
```

<details>
<summary>Glossary (RAG, embeddings, MCP, LiteLLM)</summary>

- **RAG** = retrieval‑augmented generation: retrieve relevant paper passages first, then generate an answer grounded in
  those passages.
- **Embedding model** = turns text into vectors for semantic search; changing it usually requires rebuilding an index.
- **LiteLLM model id** = the model string you pass to LiteLLM (provider/model routing), e.g. `gpt-4o`, `gemini/...`,
  `ollama/...`.
- **MCP** = Model Context Protocol: lets tools/agents call into paperpipe’s retrieval helpers (e.g. “retrieve chunks”)
  without copying PDFs into the chat.
- **Staging dir** (`.pqa_papers/`) = PDF-only mirror used so RAG backends don’t index generated Markdown.

</details>

<details>
<summary>Config: default search mode</summary>

Set a default for `papi search` (CLI flags still win):

```bash
export PAPERPIPE_SEARCH_MODE=auto   # auto|fts|scan|hybrid
```

Or in `config.toml`:

```toml
[search]
mode = "auto" # auto|fts|scan|hybrid
```

</details>

## Agent integration

paperpipe is designed to work with coding agents. Install the skill and MCP servers:

```bash
papi install                          # installs skill + prompts + MCP for detected CLIs
# or be specific:
papi install skill --claude --codex --gemini
papi install mcp --claude --codex --gemini
```

After installation, your agent can:
- Use `/papi` to get paper context (skill)
- Call MCP tools like `retrieve_chunks` for RAG retrieval
- Verify code against paper equations

For a ready-to-paste snippet for your repo's agent instructions, see [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md).

### What the agent sees

When you (or your agent) run `papi show <paper> --level eq`, you get structured output like:

```markdown
## Equation 1: LoRA Update
$$h = W_0 x + \Delta W x = W_0 x + BA x$$
where:
- $W_0 \in \mathbb{R}^{d \times k}$: pretrained weight matrix (frozen)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: low-rank matrices
- $r \ll \min(d, k)$: the rank (typically 1-64)
```

This is what makes verification possible — the agent can compare your code symbol-by-symbol.

<details>
<summary>MCP server setup (manual)</summary>

### MCP servers

paperpipe provides MCP servers for retrieval-only workflows:
- **PaperQA2 retrieval** (`papi mcp-server`): raw chunks + citations over the cached index
- **LEANN search** (`papi leann-mcp-server`): wraps LEANN's MCP server

**Claude Code** (project `.mcp.json`):
```json
{
  "mcpServers": {
    "paperqa": {
      "command": "papi",
      "args": ["mcp-server"],
      "env": { "PAPERQA_EMBEDDING": "text-embedding-3-small" }
    },
    "leann": {
      "command": "papi",
      "args": ["leann-mcp-server"],
      "env": {}
    }
  }
}
```

**Claude Code** (user scope via CLI):
```bash
claude mcp add --transport stdio --env PAPERQA_EMBEDDING=text-embedding-3-small --scope user paperqa -- papi mcp-server
```

**Codex CLI**:
```bash
codex mcp add paperqa --env PAPERQA_EMBEDDING=text-embedding-3-small -- papi mcp-server
codex mcp add leann -- papi leann-mcp-server
```

**Gemini CLI** (`~/.gemini/settings.json` or `.gemini/settings.json`):
```json
{
  "mcpServers": {
    "paperqa": {
      "command": "papi",
      "args": ["mcp-server"],
      "env": { "PAPERQA_EMBEDDING": "text-embedding-3-small" }
    },
    "leann": {
      "command": "papi",
      "args": ["leann-mcp-server"],
      "env": {}
    }
  }
}
```

### MCP environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERPIPE_PQA_INDEX_DIR` | `~/.paperpipe/.pqa_index` | Root directory for PaperQA2 indices |
| `PAPERPIPE_PQA_INDEX_NAME` | `paperpipe_<embedding>` | Index name (subfolder under index dir) |
| `PAPERQA_EMBEDDING` | (from config) | Embedding model (must match the index you built) |

### MCP tools

| Tool | Description |
|------|-------------|
| `retrieve_chunks` | Retrieve raw chunks + citations (no LLM answering) |
| `list_pqa_indexes` | List available PaperQA2 indices |
| `get_pqa_index_status` | Show index stats (files, failures) |

### MCP usage

1. Build the index first: `papi index --pqa-embedding text-embedding-3-small`
2. In your agent, call `retrieve_chunks` with your query
3. If retrieval looks wrong, call `get_pqa_index_status` to inspect

</details>

## RAG backends (`papi ask`)

paperpipe supports two RAG backends for cross-paper questions:

| Backend | Install | Best for |
|---------|---------|----------|
| [PaperQA2](https://github.com/Future-House/paper-qa) | `paperpipe[paperqa]` | Agentic synthesis with citations (cloud LLMs) |
| [LEANN](https://github.com/yichuan-w/LEANN) | `paperpipe[leann]` | Local retrieval (Ollama) |

```bash
# PaperQA2 (default if installed)
papi ask "What regularization techniques do these papers use?"

# LEANN (local)
papi ask "..." --backend leann
```

The first query builds an index (cached under `.pqa_index/` or `.leann/`). Use `papi index` to pre-build.

<details>
<summary>PaperQA2 configuration</summary>

### Common options

| Flag | Description |
|------|-------------|
| `--pqa-llm MODEL` | LLM for answer generation (LiteLLM id) |
| `--pqa-summary-llm MODEL` | LLM for evidence summarization (often cheaper) |
| `--pqa-embedding MODEL` | Embedding model for text chunks |
| `--pqa-temperature FLOAT` | LLM temperature (0.0-1.0) |
| `--pqa-verbosity INT` | Logging level (0-3; 3 = log all LLM calls) |
| `--pqa-agent-type TEXT` | Agent type (e.g., `fake` for deterministic low-token retrieval) |
| `--pqa-answer-length TEXT` | Target answer length (e.g., "about 200 words") |
| `--pqa-evidence-k INT` | Number of evidence pieces to retrieve (default: 10) |
| `--pqa-max-sources INT` | Max sources to cite in answer (default: 5) |
| `--pqa-timeout FLOAT` | Agent timeout in seconds (default: 500) |
| `--pqa-concurrency INT` | Indexing concurrency (default: 1) |
| `--pqa-rebuild-index` | Force full index rebuild |
| `--pqa-retry-failed` | Retry previously failed documents |
| `--format evidence-blocks` | Output JSON with `{answer, evidence[]}` (requires PaperQA2 Python package) |
| `--pqa-raw` | Show raw PaperQA2 output (streaming logs + answer); disables `papi ask` output filtering (also enabled by global `-v/--verbose`) |

Any additional arguments are passed through to `pqa` (e.g., `--agent.search_count 10`).

### Model combinations

```bash
# Gemini + Google Embeddings
papi ask "Explain the architecture" --pqa-llm "gemini/gemini-2.5-flash" --pqa-embedding "gemini/gemini-embedding-001"

# Claude + Voyage Embeddings
papi ask "Compare the loss functions" --pqa-llm "claude-sonnet-4-5" --pqa-embedding "voyage/voyage-3-large"

# GPT + OpenAI Embeddings
papi ask "How to implement eq 4?" --pqa-llm "gpt-4o" --pqa-embedding "text-embedding-3-large"

# OpenRouter (200+ models)
papi ask "Explain the method" --pqa-llm "openrouter/anthropic/claude-sonnet-4" --pqa-embedding "openrouter/openai/text-embedding-3-large"

# Cheaper summarization model
papi ask "Compare methods" --pqa-llm gpt-4o --pqa-summary-llm gpt-4o-mini
```

### Index/caching notes

- First run builds an index under `<paper_db>/.pqa_index/` and stages PDFs under `<paper_db>/.pqa_papers/`.
- Override index location with `PAPERPIPE_PQA_INDEX_DIR`.
- If you indexed wrong content (or changed embeddings), delete `.pqa_index/` to force rebuild.
- If PDFs failed indexing (recorded as `ERROR`), re-run with `--pqa-retry-failed` or `--pqa-rebuild-index`.
- By default, `papi ask` uses `--settings default` to avoid stale user settings; pass `-s/--settings <name>` to override.
- If Pillow is not installed, `papi ask` forces `--parsing.multimodal OFF`; pass your own `--parsing...` args to override.

</details>

<details>
<summary>LEANN configuration</summary>

### Common options

```bash
papi ask "..." --backend leann --leann-provider ollama --leann-model qwen3:8b
papi ask "..." --backend leann --leann-host http://localhost:11434
papi ask "..." --backend leann --leann-top-k 12 --leann-complexity 64
```

### Defaults

LEANN defaults to `ollama` with `olmo-3:7b` for answering and `nomic-embed-text` for embeddings.

Override via `config.toml`:
```toml
[leann]
llm_provider = "ollama"
llm_model = "qwen3:8b"
embedding_model = "nomic-embed-text"
embedding_mode = "ollama"
```

Or env vars: `PAPERPIPE_LEANN_LLM_PROVIDER`, `PAPERPIPE_LEANN_LLM_MODEL`, `PAPERPIPE_LEANN_EMBEDDING_MODEL`, `PAPERPIPE_LEANN_EMBEDDING_MODE`.

### Index builds

```bash
papi index --backend leann
# or:
papi leann-index
```

By default, `papi ask --backend leann` auto-builds the index if missing (disable with `--leann-no-auto-index`).

</details>

## LLM configuration

paperpipe uses LLMs for generating summaries, extracting equations, and tagging. Without an LLM, it falls back to regex extraction and metadata-based summaries.

```bash
# Set your API key (pick one)
export GEMINI_API_KEY=...       # default provider
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...       # for Voyage embeddings (recommended with Claude)
export OPENROUTER_API_KEY=...   # 200+ models

# Override the default model
export PAPERPIPE_LLM_MODEL=gpt-4o
export PAPERPIPE_LLM_TEMPERATURE=0.3  # default: 0.3
```

### Local-only via Ollama

```bash
export PAPERPIPE_LLM_MODEL=ollama/qwen3:8b
export PAPERPIPE_EMBEDDING_MODEL=ollama/nomic-embed-text

# Either env var name works (paperpipe normalizes both):
export OLLAMA_HOST=http://localhost:11434
# export OLLAMA_API_BASE=http://localhost:11434
```

Check which models work with your keys:
```bash
papi models                    # probe default models for your configured keys
papi models latest             # probe latest models (gpt-4o, gemini-2.5, claude-sonnet-4-5)
papi models last-gen           # probe previous generation
papi models all                # probe broader superset
papi models --verbose          # show underlying provider errors
```

## Tagging

Papers are auto-tagged from:
1. arXiv categories (cs.CV → computer-vision)
2. LLM-generated semantic tags
3. Your `--tags` flag

```bash
papi add 1706.03762 --tags my-project,priority
papi list --tag attention
```

## Non-arXiv papers

```bash
papi add --pdf ./paper.pdf --title "Some Conference Paper" --tags local
```

## Configuration file

For persistent settings, create `~/.paperpipe/config.toml` (override location with `PAPERPIPE_CONFIG_PATH`):

```toml
[llm]
model = "gemini/gemini-2.5-flash"
temperature = 0.3

[embedding]
model = "gemini/gemini-embedding-001"

[paperqa]
settings = "default"
index_dir = "~/.paperpipe/.pqa_index"
summary_llm = "gpt-4o-mini"
enrichment_llm = "gpt-4o-mini"

[leann]
llm_provider = "ollama"
llm_model = "qwen3:8b"

[tags.aliases]
cv = "computer-vision"
nlp = "natural-language-processing"
```

Precedence: **CLI flags > env vars > config.toml > built-in defaults**.

## Development

```bash
git clone https://github.com/hummat/paperpipe && cd paperpipe
pip install -e ".[dev]"
make check                            # format + lint + typecheck + test
```

<details>
<summary>Release (maintainers)</summary>

This repo publishes to PyPI when a GitHub Release is published (see `.github/workflows/publish.yml`).

```bash
# Bump versions first (pyproject.toml + paperpipe.py), then:
make release
```

</details>

## Credits

- [PaperQA2](https://github.com/Future-House/paper-qa) by Future House — RAG backend.
  *Skarlinski et al., "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge", 2024.*
  [arXiv:2409.13740](https://arxiv.org/abs/2409.13740)
- [LEANN](https://github.com/yichuan-w/LEANN) — local RAG backend

## License

MIT
