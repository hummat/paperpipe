"""Tests for paperpipe/cli/ask.py (ask command) with LEANN backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from .conftest import MockPopen, cli_mod


class TestAskLeannCommand:
    def test_ask_help_does_not_list_unsupported_leann_filter_flags(self):
        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "--help"])

        assert result.exit_code == 0
        assert "--leann-filter" not in result.output
        assert "--leann-grep" not in result.output

    def test_ask_rejects_removed_leann_filter_alias(self):
        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli, ["ask", "query", "--backend", "leann", "--leann-filter", "paper_name == 'lora'"]
        )

        assert result.exit_code != 0
        assert "--leann-filter is unavailable" in result.output

    def test_ask_rejects_unsupported_leann_passthrough_flags(self):
        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli, ["ask", "query", "--backend", "leann", "--", "--filter", "paper_name == 'lora'"]
        )

        assert result.exit_code != 0
        assert "Installed LEANN CLI does not support --filter" in result.output

    def test_ask_rejects_removed_leann_grep_alias(self):
        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli, ["ask", "query", "--backend", "leann", "--leann-grep"]
        )

        assert result.exit_code != 0
        assert "--leann-grep is unavailable" in result.output

    def test_ask_rejects_unsupported_use_grep_passthrough(self):
        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli, ["ask", "query", "--backend", "leann", "--", "--use-grep"]
        )

        assert result.exit_code != 0
        assert "does not support --use-grep" in result.output

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
