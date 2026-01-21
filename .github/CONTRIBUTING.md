# Contributing to paperpipe

Thanks for your interest in contributing! This document covers development setup and guidelines.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/hummat/paperpipe.git
cd paperpipe

# Install development dependencies
uv sync --group dev

# Run all checks
make check
```

### Running Tests

```bash
# Run unit tests (default, skips integration tests)
uv run pytest

# Run specific test
uv run pytest tests/ -k test_name -v

# Run integration tests (requires network/API keys)
uv run pytest -m integration

# Run all tests
uv run pytest -m "integration or not integration"
```

## Code Style

### Python

- Python >= 3.10
- 120-character line limit (see `[tool.ruff]` in pyproject.toml)
- Type hints for function signatures
- Run `ruff format .` and `ruff check .` before committing
- Run `pyright` for type checking

### Workflow

1. Read files before editing — understand existing code
2. After changes: `uv run ruff format .` → `uv run ruff check .` → `uv run pyright` → `uv run pytest`
3. Check if docs need updating (README.md, AGENT_INTEGRATION.md)

## Architecture

Before making changes, read the architecture docs:

- `docs/agent/architecture.md` — Package structure and key flows
- `docs/agent/code_conventions.md` — Detailed style guide
- `docs/agent/testing_patterns.md` — Testing approach
- `docs/agent/workflow.md` — Issues, branching, PRs

### Key Directories

- `paperpipe/` — Main package (Click CLI)
- `tests/` — Test suite
- `skill/` — Agent skill definitions
- `prompts/` — Agent prompt templates

## Pull Request Process

1. **Create an issue first** for non-trivial changes
2. **Fork and branch** from `main`
3. **Make your changes** following the style guide
4. **Run `make check`** — all checks must pass
5. **Update documentation** if adding/changing CLI flags or behavior
6. **Submit PR** using the template

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description

feat: Add semantic search for papers
fix(mcp): Handle empty query gracefully
docs: Update installation guide
refactor(index): Simplify embedding pipeline
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`
**Scope:** Optional component name in parentheses
**Breaking changes:** Add `!` after type, e.g., `feat!: Remove legacy API`

See `docs/agent/releases.md` for full guidelines.

## What to Contribute

### Good First Issues

- Documentation improvements
- Adding test coverage
- Bug fixes with clear reproduction steps

### Feature Ideas

- New export formats
- Additional metadata sources
- CLI improvements

### Before Starting Large Features

Please open an issue first to discuss the approach. This helps avoid duplicate work and ensures the feature aligns with project goals.

## Questions?

- Open a [Discussion](https://github.com/hummat/paperpipe/discussions) for questions
- Check existing [Issues](https://github.com/hummat/paperpipe/issues) for known problems
