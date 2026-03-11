"""Tests for paperpipe/cli/ask.py (ask command) with LEANN backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from .conftest import MockPopen, cli_mod


class TestAskLeannCommand:
    def test_ask_leann_constructs_correct_command(self, temp_db: Path, monkeypatch):
        # Mock leann availability
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)
        
        # Mock subprocess.Popen
        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        # Create dummy index meta to avoid "Build it first" error
        index_dir = temp_db / ".leann" / "indexes" / "my-index"
        index_dir.mkdir(parents=True)
        (index_dir / "documents.leann.meta.json").write_text("{}")

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--backend",
                "leann",
                "--leann-index",
                "my-index",
                "--leann-filter",
                "paper_name == 'lora'",
                "--leann-grep",
            ],
        )

        assert result.exit_code == 0, result.output

        # Verify leann ask was called with correct args
        leann_call, _ = next(c for c in mock_popen.calls if c[0][0] == "leann")
        assert "leann" in leann_call
        assert "ask" in leann_call
        assert "my-index" in leann_call
        assert "query" in leann_call
        assert "--filter" in leann_call
        assert "paper_name == 'lora'" in leann_call
        assert "--use-grep" in leann_call

    def test_ask_leann_without_filter_and_grep(self, temp_db: Path, monkeypatch):
        # Mock leann availability
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)
        
        # Mock subprocess.Popen
        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        # Create dummy index meta
        index_dir = temp_db / ".leann" / "indexes" / "my-index"
        index_dir.mkdir(parents=True)
        (index_dir / "documents.leann.meta.json").write_text("{}")

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--backend",
                "leann",
                "--leann-index",
                "my-index",
            ],
        )

        assert result.exit_code == 0

        # Verify leann ask was called without filter or grep
        leann_call, _ = next(c for c in mock_popen.calls if c[0][0] == "leann")
        assert "my-index" in leann_call
        assert "--filter" not in leann_call
        assert "--use-grep" not in leann_call
