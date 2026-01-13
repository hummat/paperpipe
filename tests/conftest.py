"""Shared fixtures and helpers for tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import paperpipe
import paperpipe.config as config


def litellm_available() -> bool:
    """Check if LiteLLM is installed and an API key is configured."""
    try:
        import litellm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )


def pqa_available() -> bool:
    """Check if PaperQA2 CLI is installed."""
    return shutil.which("pqa") is not None


def fts5_available() -> bool:
    """Check if SQLite FTS5 is available in this Python/SQLite build."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_test USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_test")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a temporary paper database."""
    db_path = tmp_path / ".paperpipe"
    monkeypatch.setattr(config, "PAPER_DB", db_path)
    monkeypatch.setattr(config, "PAPERS_DIR", db_path / "papers")
    monkeypatch.setattr(config, "INDEX_FILE", db_path / "index.json")
    paperpipe.ensure_db()
    return db_path


class MockPopenProcess:
    """Mock Popen process object for testing."""

    def __init__(self, returncode: int, stdout: str):
        self._returncode = returncode
        self._stdout_lines = stdout.splitlines(keepends=True) if stdout else []
        self.stdout = iter(self._stdout_lines)

    def wait(self) -> int:
        return self._returncode


class MockPopen:
    """Mock subprocess.Popen for testing pqa command construction."""

    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.calls: list[tuple[list[str], dict]] = []
        self._returncode = returncode
        self._stdout = stdout

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return MockPopenProcess(self._returncode, self._stdout)
