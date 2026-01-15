# Architecture

## Layout
```
.
├── AGENTS.md                 # agent instructions (CLAUDE.md/GEMINI.md symlink here)
├── pyproject.toml            # project config, dependencies, tool settings
├── paperpipe/                # package root
│   ├── __init__.py           # version, public API
│   ├── __main__.py           # CLI entry point (python -m paperpipe)
│   ├── cli/                  # Click command groups
│   └── *.py                  # core modules
└── tests/
    ├── conftest.py           # shared fixtures
    └── test_*.py             # test files
```

## Conventions
- **Flat layout**: package code directly under `paperpipe/`
- **Entry point**: `__main__.py` for CLI; use `if __name__ == "__main__":`
- **Config**: all tool config in `pyproject.toml` (ruff, pyright, pytest, coverage)
- **Dependencies**: managed via `uv`; lock file is `uv.lock`

## Package manager
```bash
uv sync                       # install deps from lock
uv add <pkg>                  # add dependency
uv add --group dev <pkg>      # add dev dependency
uv run <cmd>                  # run command in venv
```

## Adding new modules
1. Create `paperpipe/<module>.py`
2. Create corresponding `tests/test_<module>.py`
3. Export public API in `__init__.py` if needed
