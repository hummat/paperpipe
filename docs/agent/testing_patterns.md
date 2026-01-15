# Testing Patterns

## Toolchain
- **Runner**: pytest
- **Coverage**: pytest-cov (threshold in `pyproject.toml`)
- **Types**: pyright
- **Lint/format**: ruff

## Commands
```bash
uv run pytest                      # run all tests
uv run pytest tests/test_foo.py    # run single file
uv run pytest -k "test_name"       # run by name pattern
uv run pytest --cov                # with coverage
uv run pyright                     # type check
uv run ruff check .                # lint
uv run ruff format .               # format
```

## Test organization
- Tests live in `tests/`
- File naming: `test_<module>.py`
- Function naming: `test_<behavior>_<scenario>()`

## Writing tests
- One assertion per test when practical
- Use fixtures for shared setup; define in `conftest.py`
- Prefer real objects over mocks; mock only external I/O
- Test behavior, not implementation

## Running tests (agent guidance)
- Run single test file during development, not full suite
- After changes: run affected tests → types → lint
- Before commit: full suite with coverage
