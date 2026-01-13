.PHONY: help deps fmt lint type test check clean build release

help:
	@echo "Targets:"
	@echo "  deps    Install dev dependencies (uv sync or pip -e)"
	@echo "  fmt     Format (ruff)"
	@echo "  lint    Lint (ruff)"
	@echo "  type    Type check (pyright)"
	@echo "  test    Tests (pytest, not integration)"
	@echo "  check   fmt + lint + type + test"
	@echo "  build   Build dist/ and run twine check"
	@echo "  release Create a GitHub Release via gh (VERSION optional)"

deps:
	@bash scripts/deps.sh

fmt:
	@uv run ruff format .

lint:
	@uv run ruff check .

type:
	@uv run pyright

test:
	@uv run pytest -m "not integration"

check: fmt lint type test

clean:
	@rm -rf dist/

build: clean
	@bash scripts/build.sh

release:
	@bash scripts/release.sh "$(VERSION)"
