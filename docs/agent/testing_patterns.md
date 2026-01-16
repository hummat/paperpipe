# Testing Patterns

Toolchain: pytest, pytest-cov, pyright, ruff (config in `pyproject.toml`).

## Organization
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
