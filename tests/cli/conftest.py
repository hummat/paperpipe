"""Shared fixtures and imports for CLI tests."""

from __future__ import annotations

import importlib.util
import sys
from importlib import import_module
from pathlib import Path

import pytest
from click.testing import CliRunner

# Import parent conftest by path - pytest does not auto-inherit fixtures from parent conftest.py
_tests_dir = Path(__file__).resolve().parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

_parent_conftest_spec = importlib.util.spec_from_file_location("tests_conftest", _tests_dir / "conftest.py")
_parent_conftest = importlib.util.module_from_spec(_parent_conftest_spec)  # type: ignore[arg-type]
_parent_conftest_spec.loader.exec_module(_parent_conftest)  # type: ignore[union-attr]

# Re-export fixtures and helpers from parent conftest
MockPopen = _parent_conftest.MockPopen
fts5_available = _parent_conftest.fts5_available
litellm_available = _parent_conftest.litellm_available
pqa_available = _parent_conftest.pqa_available
temp_db = _parent_conftest.temp_db

# Import the CLI module explicitly (avoid resolving to the package's cli function).
cli_mod = import_module("paperpipe.cli")

# "Attention Is All You Need" - stable arXiv paper for integration tests
TEST_ARXIV_ID = "1706.03762"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CLI runner."""
    return CliRunner()
