"""Tests for paperpipe/cli/index.py (index command)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import paperpipe.config as config
import paperpipe.paperqa as paperqa

from .conftest import MockPopen, cli_mod


class TestIndexCommand:
    def test_index_backend_pqa_runs_pqa_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        noisy = (
            "/home/x/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:\n"
            "  PydanticSerializationUnexpectedValue(Expected 10 fields but got 7)\n"
            "  return self.__pydantic_serializer__.to_python(\n"
        )
        mock_popen = MockPopen(returncode=0, stdout=f"{noisy}Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed"])
        assert result.exit_code == 0, result.output
        assert "Indexed" in result.output
        assert "PydanticSerializationUnexpectedValue" not in result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "index" in pqa_call
        assert str(temp_db / ".pqa_papers") in pqa_call
        assert "--agent.index.paper_directory" in pqa_call
        assert "--index" in pqa_call and "paperpipe_my-embed" in pqa_call
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

    def test_index_backend_pqa_ollama_embedding_strips_prefix_and_forces_provider(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "ollama/nomic-embed-text"])
        assert result.exit_code == 0, result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--index" in pqa_call
        assert "paperpipe_ollama_nomic-embed-text" in pqa_call
        assert "--embedding" in pqa_call
        assert pqa_call[pqa_call.index("--embedding") + 1] == "nomic-embed-text"
        assert "--embedding_config" in pqa_call
        cfg = pqa_call[pqa_call.index("--embedding_config") + 1]
        assert '"custom_llm_provider":"ollama"' in cfg

    def test_index_backend_pqa_pqa_raw_prints_noisy_output(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        noisy = (
            "/home/x/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:\n"
            "  PydanticSerializationUnexpectedValue(Expected 10 fields but got 7)\n"
            "  return self.__pydantic_serializer__.to_python(\n"
        )
        mock_popen = MockPopen(returncode=0, stdout=f"{noisy}Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed", "--pqa-raw"])
        assert result.exit_code == 0, result.output
        assert "PydanticSerializationUnexpectedValue" in result.output

    def test_index_backend_pqa_ollama_models_prepare_env(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-llm", "ollama/qwen3:8b"])
        assert result.exit_code == 0, result.output

        _, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        env = pqa_kwargs.get("env") or {}
        assert env.get("OLLAMA_API_BASE") == "http://localhost:11434"
        assert env.get("OLLAMA_HOST") == "http://localhost:11434"

    def test_index_rejects_pqa_concurrency_zero(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)
        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-concurrency", "0"])
        assert result.exit_code != 0
        assert "--pqa-concurrency must be >= 1" in result.output
