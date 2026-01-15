from __future__ import annotations

import shutil
import subprocess
import types
from importlib import import_module
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import MockPopen

import paperpipe
import paperpipe.config as config
import paperpipe.paperqa as paperqa

# Import the CLI module explicitly (avoid resolving to the package's cli function).
cli_mod = import_module("paperpipe.cli")


class TestLeannCli:
    def test_index_help_includes_leann_build_flags(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "--leann-embedding-model" in result.output
        assert "--leann-embedding-mode" in result.output
        assert "--leann-doc-chunk-size" in result.output
        assert "--leann-doc-chunk-overlap" in result.output
        assert "--pqa-raw" in result.output

    def test_index_leann_build_args_forwarded(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_refresh(*, staging_dir: Path) -> None:
            return

        def fake_build(*, index_name: str, docs_dir: Path, force: bool, extra_args: list[str]) -> None:
            captured["index_name"] = index_name
            captured["docs_dir"] = docs_dir
            captured["force"] = force
            captured["extra_args"] = list(extra_args)

        monkeypatch.setattr(paperqa, "_refresh_pqa_pdf_staging_dir", fake_refresh)
        monkeypatch.setattr(cli_mod, "_leann_build_index", fake_build)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "index",
                "--backend",
                "leann",
                "--leann-index",
                "papers",
                "--leann-embedding-model",
                "nomic-embed-text",
                "--leann-doc-chunk-size",
                "1200",
                "--leann-doc-chunk-overlap",
                "200",
                "--some-raw-leann-arg",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["index_name"] == "papers"
        assert captured["docs_dir"] == temp_db / ".pqa_papers"
        assert captured["force"] is False
        assert captured["extra_args"] == [
            "--embedding-model",
            "nomic-embed-text",
            "--doc-chunk-size",
            "1200",
            "--doc-chunk-overlap",
            "200",
            "--some-raw-leann-arg",
        ]


class TestLeannAsk:
    def test_ask_backend_leann_allows_passthrough_args(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        meta = temp_db / ".leann" / "indexes" / "papers" / "documents.leann.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}")

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--backend",
                "leann",
                "--leann-index",
                "papers",
                "--leann-no-auto-index",
                "--leann-provider",
                "openai",
                "--leann-model",
                "gpt-5.2",
                "--",
                "--verbose",
            ],
        )
        assert result.exit_code == 0, result.output

        leann_call, _ = next(c for c in mock_popen.calls if c[0][0] == "leann")
        assert leann_call[:3] == ["leann", "ask", "papers"]
        assert "--verbose" in leann_call


class TestLeannCommands:
    def test_leann_index_runs_leann_build(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append((args, kwargs))
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "leann"])
        assert result.exit_code == 0, result.output

        cmd, kwargs = calls[0]
        assert cmd[:2] == ["leann", "build"]
        assert "papers_ollama_nomic-embed-text" in cmd
        assert "--docs" in cmd and str(temp_db / ".pqa_papers") in cmd
        assert "--file-types" in cmd and ".pdf" in cmd
        assert "--embedding-model" in cmd and "nomic-embed-text" in cmd
        assert "--embedding-mode" in cmd and "ollama" in cmd
        assert kwargs.get("cwd") == temp_db
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

    def test_leann_index_can_derive_name_from_embedding(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[leann]",
                    "index_by_embedding = true",
                    'embedding_mode = "openai"',
                    'embedding_model = "text-embedding-3-small"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append((args, kwargs))
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "leann"])
        assert result.exit_code == 0, result.output

        cmd, kwargs = calls[0]
        assert cmd[:2] == ["leann", "build"]
        assert "papers_openai_text-embedding-3-small" in cmd
        assert kwargs.get("cwd") == temp_db

    def test_leann_index_rejects_file_types_override(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "leann", "--file-types", ".txt"])
        assert result.exit_code != 0
        assert "PDF-only" in result.output

    def test_ask_backend_leann_runs_leann_ask(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        meta = temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text" / "documents.leann.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}")

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "what is x",
                "--backend",
                "leann",
                "--leann-provider",
                "ollama",
                "--leann-model",
                "qwen3:8b",
                "--leann-top-k",
                "3",
                "--leann-no-recompute",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "OUT" in result.output

        cmd, kwargs = mock_popen.calls[0]
        assert cmd[:3] == ["leann", "ask", "papers_ollama_nomic-embed-text"]
        assert "what is x" in cmd
        assert "--llm" in cmd and "ollama" in cmd
        assert "--model" in cmd and "qwen3:8b" in cmd
        assert "--top-k" in cmd and "3" in cmd
        assert "--no-recompute" in cmd
        assert kwargs.get("cwd") == temp_db

    def test_ask_backend_leann_defaults_to_gemini_openai_compat(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        meta = temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text" / "documents.leann.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}")

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "what is x", "--backend", "leann", "--leann-no-auto-index"])
        assert result.exit_code == 0, result.output
        assert "OUT" in result.output

        cmd, _ = mock_popen.calls[0]
        assert cmd[:3] == ["leann", "ask", "papers_ollama_nomic-embed-text"]
        assert "--llm" in cmd and "openai" in cmd
        assert "--model" in cmd and "gemini-3-flash-preview" in cmd
        assert "--api-base" in cmd and paperpipe.GEMINI_OPENAI_COMPAT_BASE_URL in cmd
        assert "--api-key" in cmd and "test-key" in cmd

    def test_ask_backend_leann_requires_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        build_calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            build_calls.append((args, kwargs))
            # Simulate that `leann build` created the index metadata file.
            meta = temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text" / "documents.leann.meta.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text('{"backend_name":"hnsw"}')
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "q", "--backend", "leann"])
        assert result.exit_code == 0, result.output
        assert "OUT" in result.output
        assert build_calls, "Expected `leann build` to run when index is missing"

    def test_ask_backend_leann_can_disable_auto_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "q", "--backend", "leann", "--leann-no-auto-index"])
        assert result.exit_code != 0
        assert "Build it first" in result.output


class TestLeannIndexCommand:
    def test_index_backend_leann_runs_leann_build(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append((args, kwargs))
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "leann"])
        assert result.exit_code == 0, result.output

        cmd, kwargs = calls[0]
        assert cmd[:2] == ["leann", "build"]
        assert "papers_ollama_nomic-embed-text" in cmd
        assert "--docs" in cmd and str(temp_db / ".pqa_papers") in cmd
        assert "--file-types" in cmd and ".pdf" in cmd
        assert kwargs.get("cwd") == temp_db
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()
