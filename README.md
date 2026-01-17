# paperpipe

![image](https://repository-images.githubusercontent.com/1120503046/8ba4e2ed-30ef-4d0d-996f-68a48962cb9b)

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
uv tool install paperpipe --with "paperpipe[figures]"  # figure extraction from LaTeX/PDF
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
pip install 'paperpipe[figures]'        # figure extraction from LaTeX/PDF
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
│       ├── tldr.md                   # one-paragraph TL;DR
│       ├── meta.json                 # metadata + tags
│       ├── notes.md                  # your implementation notes
│       └── figures/                  # extracted figures (if available)
│           ├── figure1.png
│           └── figure2.pdf
```

**Why this structure matters:**
- `equations.md` — Key equations with variable definitions. Use for code verification.
- `source.tex` — Original LaTeX. Use when you need exact notation or the equation extraction missed something.
- `summary.md` — High-level overview focused on implementation (not literature review). Use for understanding the approach.
- `tldr.md` — Quick 2-3 sentence overview of the paper's contribution.
- `figures/` — Architecture diagrams, network structures, and result plots extracted from LaTeX source or PDF.
- `.pqa_papers/` — Staged PDFs only (no markdown) so RAG backends don't index generated content.

## Core commands

| Command | Purpose |
|---------|---------|
| `papi add <arxiv-id-or-url>` | Add a paper (downloads PDF + LaTeX, generates summary/equations/TL;DR) |
| `papi add --pdf file.pdf --title "..."` | Add a local PDF |
| `papi add --from-file list.json` | Import papers from a JSON list or text file |
| `papi list` | List papers (filter with `--tag`) |
| `papi search "query"` | Search across titles, tags, summaries, equations (`--rg` for grep-style, default uses ranked BM25 if indexed) |
| `papi index --backend search` | Build/update ranked search index (`search.db`) |
| `papi show <paper> --level eq` | Print equations (best for agent sessions) |
| `papi show <paper> --level tex` | Print LaTeX source |
| `papi show <paper> --level summary` | Print summary |
| `papi show <paper> --level tldr` | Print TL;DR |
| `papi export <papers...> --to ./dir` | Export context files into a repo (`--level summary\|equations\|full`) |
| `papi notes <paper>` | Open/print implementation notes |
| `papi regenerate <papers...>` | Regenerate summary/equations/tags/TL;DR |
| `papi remove <papers...>` | Remove papers |
| `papi ask "question"` | Cross-paper RAG query (requires PaperQA2 or LEANN) |
| `papi index` | Build/update the retrieval index |
| `papi tags` | List all tags |
| `papi path` | Print database location |
| `papi docs` | Print agent integration snippet (for CLAUDE.md/AGENTS.md) |
| `papi rebuild-index` | Rebuild index.json from on-disk paper directories (recovery) |

Run `papi --help` or `papi <command> --help` for full options.

## Import/Export

Share your paper collection with others or back it up.

**Export:**
```bash
# Export full list to JSON
papi list --json > my_papers.json

# Export specific tag
papi list --tag "computer-vision" --json > cv_papers.json
```

**Import:**
```bash
# Import from JSON (preserves custom names and tags)
papi add --from-file my_papers.json

# Import from text file (one arXiv ID per line)
papi add --from-file paper_ids.txt --tags "imported"

# Import from BibTeX file (requires bibtexparser)
papi add --from-file papers.bib
# or install with BibTeX support:
# uv tool install paperpipe --with "paperpipe[bibtex]"
```

**Semantic Scholar Support:**
```bash
# Add papers from Semantic Scholar
papi add https://www.semanticscholar.org/paper/...
papi add 0123456789abcdef0123456789abcdef01234567  # S2 paper ID
```

Exact text search (fast, no LLM required):

```bash
papi search --rg "AdamW"              # case-insensitive, literal string (default)
papi search --rg --case-sensitive "NeRF"  # match exact case
papi search --rg --regex "Eq\\. [0-9]+"   # regex mode (opt-in)
```

Ranked search (BM25 via SQLite FTS5, no LLM required):

```bash
papi index --backend search --search-rebuild    # builds <paper_db>/search.db
papi search "surface reconstruction"             # uses FTS if available (default)
papi search --no-fts "surface reconstruction"    # force in-memory scan (disables FTS, uses fuzzy matching)
papi search --no-fts --exact "exact phrase"      # force scan with exact matching only
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

### Custom prompts

| Prompt | Description |
|--------|-------------|
| `/papi` | Route questions to the cheapest papi command |
| `/papi-init` | Add/update PaperPipe integration in your project's AGENTS.md/CLAUDE.md |
| `/verify-with-paper` | Verify code against paper equations |
| `/ground-with-paper` | Ground responses in paper excerpts |
| `/compare-papers` | Compare multiple papers for a decision |
| `/curate-paper-note` | Create a project note from paper excerpts |

For a ready-to-paste snippet for your repo's agent instructions, run `papi docs` or see [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md).

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
- **PaperQA2 retrieval**: raw chunks + citations (via `paperqa_mcp_server`)
- **LEANN search**: fast semantic search over papers (via `leann_mcp`)

MCP servers are configured automatically when you run `papi install mcp`. The install command creates the appropriate configuration files for your agent (Claude Code, Codex CLI, or Gemini CLI).

**Installation**:
```bash
# Install MCP servers for all supported agents (user scope)
papi install mcp

# Install for specific agents
papi install mcp --claude
papi install mcp --codex
papi install mcp --gemini

# Install repo-local MCP configs (project scope)
papi install mcp --repo

# Customize embedding model
papi install mcp --embedding text-embedding-3-small
```

The MCP servers are automatically launched by your agent when needed. You don't need to manually start them.

### MCP environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERPIPE_PQA_INDEX_DIR` | `~/.paperpipe/.pqa_index` | Root directory for PaperQA2 indices |
| `PAPERPIPE_PQA_INDEX_NAME` | `paperpipe_<embedding>` | Index name (subfolder under index dir) |
| `PAPERQA_EMBEDDING` | (from config) | Embedding model (must match index for PaperQA2) |

### MCP tools

| Tool | Backend | Description |
|------|---------|-------------|
| `retrieve_chunks` | PaperQA2 | Retrieve raw chunks + citations (no LLM answering) |
| `list_pqa_indexes` | PaperQA2 | List available PaperQA2 indices with embedding model metadata |
| `get_pqa_index_status` | PaperQA2 | Show index stats (files, failures) |
| `leann_search` | LEANN | Semantic search over papers (faster, simpler output) |
| `leann_list` | LEANN | List available LEANN indexes |

### MCP usage

1. Build indexes: `papi index --backend pqa --pqa-embedding text-embedding-3-small`
2. In your agent: `leann_search()` (fast) or `retrieve_chunks()` (with citations)
3. For PaperQA2: embedding model is **automatically inferred** from index metadata (or index name for backward compatibility)

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

<details>
<summary><strong>Model combination examples</strong></summary>

**Indexing:**

```bash
# API keys should be in env
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export VOYAGE_API_KEY=...

# Ollama (local) + Ollama embeddings
papi index --backend pqa --pqa-llm ollama/olmo-3:7b --pqa-embedding ollama/nomic-embed-text

# GPT + OpenAI Embeddings
papi index --backend pqa --pqa-llm gpt-4.1 --pqa-summary-llm gpt-4.1-mini --pqa-embedding text-embedding-3-small

# Gemini + Google Embeddings
papi index --backend pqa --pqa-llm gemini/gemini-3-flash-preview --pqa-embedding gemini/gemini-embedding-001

# Claude + Voyage Embeddings
papi index --backend pqa --pqa-llm claude-sonnet-4-5 --pqa-summary-llm claude-haiku-4-5 --pqa-embedding voyage/voyage-3.5
```

**Asking:**

```bash
# Ollama (local)
papi ask "how is neus different from nerf?" --backend pqa --pqa-llm ollama/olmo-3:7b --pqa-embedding ollama/nomic-embed-text

# GPT
papi ask "how is neus different from nerf?" --backend pqa --pqa-llm gpt-4.1 --pqa-summary-llm gpt-4.1-mini --pqa-embedding text-embedding-3-small

# Gemini
papi ask "how is neus different from nerf?" --backend pqa --pqa-llm gemini/gemini-3-flash-preview --pqa-embedding gemini/gemini-embedding-001

# Claude
papi ask "how is neus different from nerf?" --backend pqa --pqa-llm claude-sonnet-4-5 --pqa-summary-llm claude-haiku-4-5 --pqa-embedding voyage/voyage-3.5
```

</details>

<details>
<summary><strong>Embedding provider examples (indexing)</strong></summary>

#### OpenAI

```bash
export OPENAI_API_KEY=...
papi index --backend pqa --pqa-embedding text-embedding-3-small
```

#### Gemini (native LiteLLM id)

```bash
export GEMINI_API_KEY=...
papi index --backend pqa --pqa-embedding gemini/gemini-embedding-001
```

#### Voyage (native LiteLLM id)

```bash
export VOYAGE_API_KEY=...
papi index --backend pqa --pqa-embedding voyage/voyage-3.5
```

#### OpenAI-compatible endpoints (advanced)

If you want to hit an OpenAI-compatible endpoint directly (instead of a native LiteLLM provider id), set
`OPENAI_API_BASE` and `OPENAI_API_KEY` and use an `openai/...` embedding id.

```bash
export OPENAI_API_BASE=https://api.voyageai.com/v1
export OPENAI_API_KEY="$VOYAGE_API_KEY"
papi index --backend pqa --pqa-embedding openai/voyage-3.5
```

</details>

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

Notes:
- If you use `--leann-provider anthropic`, your `leann` install must include the `anthropic` Python package
  (`pip install anthropic` in the same environment that runs `leann`).
- You can pass through extra `leann` CLI flags after `--` (useful for debugging), e.g.:
  `papi -v ask "..." --backend leann -- ...`

### Model combinations

<details>
<summary><strong>Model combination examples</strong></summary>

**Indexing:**

```bash
# API keys should be in env
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export VOYAGE_API_KEY=...

# Ollama (local) + Ollama embeddings
papi index --backend leann --leann-embedding-mode ollama --leann-embedding-model nomic-embed-text

# OpenAI + OpenAI embeddings
papi index --backend leann --leann-embedding-mode openai --leann-embedding-model text-embedding-3-small --leann-embedding-api-key $OPENAI_API_KEY

# Gemini + Gemini embeddings (OpenAI-compatible)
papi index --backend leann --leann-embedding-mode openai --leann-embedding-model gemini-embedding-001 --leann-embedding-api-base https://generativelanguage.googleapis.com/v1beta/openai/ --leann-embedding-api-key $GEMINI_API_KEY

# Voyage embeddings (OpenAI-compatible)
papi index --backend leann --leann-embedding-mode openai --leann-embedding-model voyage-3.5 --leann-embedding-api-base https://api.voyageai.com/v1 --leann-embedding-api-key $VOYAGE_API_KEY
```

**Asking:**

```bash
# Ollama (local)
papi ask "how is neus different from nerf?" --backend leann --leann-provider ollama --leann-model olmo-3:7b --leann-index papers_ollama_nomic-embed-text

# OpenAI
papi ask "how is neus different from nerf?" --backend leann --leann-provider openai --leann-model gpt-4.1 --leann-api-key $OPENAI_API_KEY --leann-index papers_openai_text-embedding-3-small

# Anthropic + Voyage embeddings
papi ask "how is neus different from nerf?" --backend leann --leann-provider anthropic --leann-model claude-sonnet-4-5 --leann-api-key $ANTHROPIC_API_KEY --leann-index papers_openai_voyage-3.5

# Gemini (OpenAI-compatible)
papi ask "how is neus different from nerf?" --backend leann --leann-provider openai --leann-model gemini-3-flash-preview --leann-api-base https://generativelanguage.googleapis.com/v1beta/openai/ --leann-api-key $GEMINI_API_KEY --leann-index papers_openai_gemini-embedding-001
```

</details>

<details>
<summary><strong>Embedding provider examples</strong></summary>

**Note:** For `--leann-embedding-mode openai`, LEANN defaults the API key to `OPENAI_API_KEY` unless you pass `--leann-embedding-api-key`.

```bash
# Ollama (local)
papi index --backend leann --leann-embedding-mode ollama --leann-embedding-model nomic-embed-text

# OpenAI
export OPENAI_API_KEY=...
papi index --backend leann --leann-embedding-mode openai --leann-embedding-model text-embedding-3-small --leann-embedding-api-key $OPENAI_API_KEY

# Gemini (OpenAI-compatible)
export GEMINI_API_KEY=...
papi index --backend leann --leann-embedding-mode openai --leann-embedding-model gemini-embedding-001 --leann-embedding-api-base https://generativelanguage.googleapis.com/v1beta/openai/ --leann-embedding-api-key $GEMINI_API_KEY

# Voyage (OpenAI-compatible)
export VOYAGE_API_KEY=...
papi index --backend leann --leann-embedding-mode openai --leann-embedding-model voyage-3.5 --leann-embedding-api-base https://api.voyageai.com/v1 --leann-embedding-api-key $VOYAGE_API_KEY
```

**Gemini notes:**
- May hit quota/rate limits (HTTP 429). Retry after suggested delay.
- Some LEANN versions batch too many inputs per request for Gemini (hard limit: 100 inputs/request) and fail with HTTP 400; update LEANN or reduce chunk counts (e.g., larger `--leann-doc-chunk-size`).

</details>

### Defaults

By default, paperpipe derives LEANN's defaults from your global `[llm]` / `[embedding]` model settings when they are
LEANN-compatible:
- `ollama/...` → `--llm ollama` / `--embedding-mode ollama`
- `gpt-*` / `text-embedding-*` → `--llm openai` / `--embedding-mode openai`
- `gemini/...` → `--llm openai` (Gemini OpenAI-compatible endpoint)

For Gemini, paperpipe defaults `--leann-api-base` to `https://generativelanguage.googleapis.com/v1beta/openai/` and uses
`GEMINI_API_KEY`/`GOOGLE_API_KEY` if set.

Note: LEANN's current CLI batches OpenAI-compatible embeddings in chunks of up to ~500-800 texts per request; Gemini's
embedding endpoint hard-limits batches to 100, so paperpipe does *not* auto-map `gemini/...` embeddings to LEANN by
default. Use `PAPERPIPE_LEANN_EMBEDDING_*` / `[leann]` to override (and expect to tune batch behavior upstream in LEANN).

### Multiple indices

LEANN supports multiple index names under `<paper_db>/.leann/indexes/`.

By default, paperpipe auto-derives the LEANN index name from the embedding mode/model (similar to PaperQA2).

To disable and always use a single LEANN index named `papers`, set:

```toml
[leann]
index_by_embedding = false
```

or `export PAPERPIPE_LEANN_INDEX_BY_EMBEDDING=0`.

When enabled, the default LEANN index name becomes `papers_<mode>_<model>` (with `/` and `:` replaced by `_`).

If model ids are not recognized as compatible, it falls back to `ollama` with `olmo-3:7b` (LLM) and `nomic-embed-text`
(embeddings).

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

# Override common LEANN build knobs (maps to `leann build ...`):
papi index --backend leann --leann-embedding-mode ollama --leann-embedding-model nomic-embed-text
papi index --backend leann --leann-embedding-mode ollama --leann-embedding-host http://localhost:11434
papi index --backend leann --leann-doc-chunk-size 350 --leann-doc-chunk-overlap 128
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

# Optional: override LEANN separately (otherwise it follows [llm]/[embedding] for openai/ollama model ids)
[leann]
llm_provider = "ollama"
llm_model = "qwen3:8b"
embedding_model = "nomic-embed-text"
embedding_mode = "ollama"

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
# Bump version in pyproject.toml, then:
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
