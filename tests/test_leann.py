from __future__ import annotations

import shutil
import subprocess
import types
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner
from conftest import MockPopen

import paperpipe
import paperpipe.config as config
import paperpipe.paperqa as paperqa
from paperpipe.leann import FileEntry, LeannManifest, _redact_cmd

# Import the CLI module explicitly (avoid resolving to the package's cli function).
cli_mod = import_module("paperpipe.cli")
# Import the index submodule for patching _leann_build_index
cli_index_mod = import_module("paperpipe.cli.index")
cli_ask_mod = import_module("paperpipe.cli.ask")


def _make_manifest(*, files: dict[str, FileEntry] | None = None, is_compact: bool = False) -> LeannManifest:
    return cast(
        LeannManifest,
        {
            "version": paperpipe.leann.MANIFEST_VERSION,
            "is_compact": is_compact,
            "embedding_mode": "ollama",
            "embedding_model": "nomic-embed-text",
            "created_at": "2026-01-26T10:00:00Z",
            "updated_at": "2026-01-26T10:00:00Z",
            "files": files or {},
        },
    )


def _write_leann_index_stub(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "documents.leann.meta.json").write_text("{}")
    (index_dir / "documents.index").write_text("")


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

        def fake_build(
            *, index_name: str, docs_dir: Path, force: bool, no_compact: bool, extra_args: list[str]
        ) -> None:
            captured["index_name"] = index_name
            captured["docs_dir"] = docs_dir
            captured["force"] = force
            captured["no_compact"] = no_compact
            captured["extra_args"] = list(extra_args)

        monkeypatch.setattr(paperqa, "_refresh_pqa_pdf_staging_dir", fake_refresh)
        monkeypatch.setattr(cli_index_mod, "_leann_build_index", fake_build)

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
        assert captured["no_compact"] is True  # Default is True (non-compact for incremental)
        assert captured["extra_args"] == [
            "--embedding-model",
            "nomic-embed-text",
            "--doc-chunk-size",
            "1200",
            "--doc-chunk-overlap",
            "200",
            "--some-raw-leann-arg",
        ]


class TestLeannExecutable:
    def test_uses_path_command_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(paperpipe.leann.shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        assert paperpipe.leann._leann_executable() == "leann"

    def test_finds_script_next_to_python_for_uv_tool_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bin_dir = tmp_path / "tool" / "bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python"
        leann = bin_dir / "leann"
        python.write_text("")
        leann.write_text("#!/bin/sh\n")
        leann.chmod(0o755)

        monkeypatch.setattr(paperpipe.leann.shutil, "which", lambda cmd: None)
        monkeypatch.setattr(paperpipe.leann.sys, "executable", str(python))

        assert paperpipe.leann._leann_executable() == str(leann)


class TestLeannAsk:
    def test_ask_backend_leann_allows_passthrough_args(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "papers")

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

        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text")

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
        # Ensure default LLM model is used (not overridden by env or cached config)
        monkeypatch.delenv("PAPERPIPE_LLM_MODEL", raising=False)
        monkeypatch.delenv("PAPERPIPE_LEANN_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("PAPERPIPE_LEANN_LLM_MODEL", raising=False)
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text")

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

    def test_ask_backend_leann_defaults_to_openrouter_openai_compat(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)
        monkeypatch.setenv("PAPERPIPE_LLM_MODEL", "openrouter/google/gemini-3.5-flash")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        monkeypatch.delenv("PAPERPIPE_LEANN_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("PAPERPIPE_LEANN_LLM_MODEL", raising=False)
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text")

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "what is x", "--backend", "leann", "--leann-no-auto-index"])
        assert result.exit_code == 0, result.output

        cmd, _ = mock_popen.calls[0]
        assert "--llm" in cmd and "openai" in cmd
        assert "--model" in cmd and "google/gemini-3.5-flash" in cmd
        assert "--api-base" in cmd and paperpipe.OPENROUTER_OPENAI_COMPAT_BASE_URL in cmd
        assert "--api-key" in cmd and "openrouter-key" in cmd

    def test_ask_backend_leann_openrouter_from_leann_model(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        # OpenRouter selected via the LEANN model id itself, even when [llm].model is not openrouter.
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)
        monkeypatch.delenv("PAPERPIPE_LLM_MODEL", raising=False)
        monkeypatch.setenv("PAPERPIPE_LEANN_LLM_MODEL", "openrouter/deepseek/deepseek-v4-pro")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        monkeypatch.delenv("PAPERPIPE_LEANN_LLM_PROVIDER", raising=False)
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text")

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        result = CliRunner().invoke(cli_mod.cli, ["ask", "what is x", "--backend", "leann", "--leann-no-auto-index"])
        assert result.exit_code == 0, result.output

        cmd, _ = mock_popen.calls[0]
        assert "--llm" in cmd and "openai" in cmd
        assert "--model" in cmd and "deepseek/deepseek-v4-pro" in cmd
        assert "openrouter/deepseek/deepseek-v4-pro" not in cmd  # prefix stripped
        assert "--api-base" in cmd and paperpipe.OPENROUTER_OPENAI_COMPAT_BASE_URL in cmd
        assert "--api-key" in cmd and "openrouter-key" in cmd

    def test_ask_backend_leann_requires_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        build_calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            build_calls.append((args, kwargs))
            # Simulate that `leann build` created the index metadata file.
            _write_leann_index_stub(temp_db / ".leann" / "indexes" / "papers_ollama_nomic-embed-text")
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

    def test_ask_backend_leann_auto_build_infers_embedding_from_explicit_index(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        captured: dict[str, object] = {}

        def fake_build(
            *, index_name: str, docs_dir: Path, force: bool, no_compact: bool, extra_args: list[str]
        ) -> None:
            captured["index_name"] = index_name
            captured["extra_args"] = list(extra_args)

        def fake_ask(**kwargs: object) -> None:
            captured["ask_index_name"] = kwargs["index_name"]

        monkeypatch.setattr(cli_ask_mod, "_leann_build_index", fake_build)
        monkeypatch.setattr(cli_ask_mod, "_ask_leann", fake_ask)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "q", "--backend", "leann", "--leann-index", "papers_openai_voyage-4"],
        )
        assert result.exit_code == 0, result.output
        assert captured["index_name"] == "papers_openai_voyage-4"
        assert captured["ask_index_name"] == "papers_openai_voyage-4"
        assert captured["extra_args"] == [
            "--embedding-mode",
            "openai",
            "--embedding-model",
            "voyage-4",
            "--embedding-api-base",
            "https://api.voyageai.com/v1",
            "--embedding-api-key",
            "voyage-key",
        ]


class TestLeannVoyageEmbeddingArgs:
    """Voyage embeddings route through LEANN's openai mode; shared by `papi ask` and `papi index`."""

    def test_voyage_openai_mode_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from paperpipe.leann import _leann_voyage_embedding_args

        monkeypatch.setenv("VOYAGE_API_KEY", "vk")
        assert _leann_voyage_embedding_args(embedding_mode="openai", embedding_model="voyage-4") == [
            "--embedding-api-base",
            "https://api.voyageai.com/v1",
            "--embedding-api-key",
            "vk",
        ]

    def test_voyage_without_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from paperpipe.leann import _leann_voyage_embedding_args

        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        assert _leann_voyage_embedding_args(embedding_mode="openai", embedding_model="voyage-4") == []

    def test_non_voyage_or_non_openai_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from paperpipe.leann import _leann_voyage_embedding_args

        monkeypatch.setenv("VOYAGE_API_KEY", "vk")
        assert _leann_voyage_embedding_args(embedding_mode="openai", embedding_model="text-embedding-3-small") == []
        assert _leann_voyage_embedding_args(embedding_mode="ollama", embedding_model="voyage-4") == []


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

    def test_index_backend_leann_auto_routes_voyage_embedding(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `papi index` must inject Voyage's endpoint + key (like `papi ask`) so an openai/voyage-*
        # index builds from VOYAGE_API_KEY instead of failing on a missing OPENAI_API_KEY.
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["index", "--backend", "leann", "--leann-embedding-mode", "openai", "--leann-embedding-model", "voyage-4"],
        )
        assert result.exit_code == 0, result.output

        cmd = calls[0]
        assert "--embedding-api-base" in cmd
        assert cmd[cmd.index("--embedding-api-base") + 1] == "https://api.voyageai.com/v1"
        assert "--embedding-api-key" in cmd
        assert cmd[cmd.index("--embedding-api-key") + 1] == "voyage-key"

    def test_index_backend_leann_passes_no_compact_flag(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify --no-compact is passed to LEANN CLI by default."""
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

        cmd, _ = calls[0]
        assert "--no-compact" in cmd, "Default build should include --no-compact for incremental updates"

    def test_index_backend_leann_compact_flag_overrides(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify --leann-compact overrides the default --no-compact."""
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append((args, kwargs))
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "leann", "--leann-compact"])
        assert result.exit_code == 0, result.output

        cmd, _ = calls[0]
        assert "--no-compact" not in cmd, "--leann-compact should prevent --no-compact"

    def test_index_backend_leann_forces_rebuild_when_manifest_exists_but_index_files_missing(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        from paperpipe.leann import _save_leann_manifest

        docs_dir = temp_db / ".pqa_papers"
        docs_dir.mkdir(parents=True)
        (docs_dir / "test-paper.pdf").touch()
        _save_leann_manifest("test-index", _make_manifest())

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object):
            calls.append(args)
            _write_leann_index_stub(temp_db / ".leann" / "indexes" / "test-index")
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "leann", "--leann-index", "test-index"])
        assert result.exit_code == 0, result.output
        assert "--force" in calls[0]


class TestLeannManifest:
    """Tests for LEANN incremental indexing manifest functionality."""

    def test_manifest_path(self, temp_db: Path) -> None:
        from paperpipe.leann import _leann_manifest_path

        path = _leann_manifest_path("test-index")
        assert path == temp_db / ".leann" / "indexes" / "test-index" / "paperpipe_manifest.json"

    def test_load_manifest_missing_returns_none(self, temp_db: Path) -> None:
        from paperpipe.leann import _load_leann_manifest

        assert _load_leann_manifest("nonexistent") is None

    def test_save_and_load_manifest_roundtrip(self, temp_db: Path) -> None:
        from paperpipe.leann import MANIFEST_VERSION, _load_leann_manifest, _save_leann_manifest

        manifest = _make_manifest()

        assert _save_leann_manifest("test-index", manifest)
        loaded = _load_leann_manifest("test-index")

        assert loaded is not None
        assert loaded["version"] == MANIFEST_VERSION
        assert loaded["is_compact"] is False
        assert loaded["embedding_mode"] == "ollama"
        assert loaded["embedding_model"] == "nomic-embed-text"

    def test_load_manifest_wrong_version_returns_none(self, temp_db: Path) -> None:
        import json

        from paperpipe.leann import _leann_manifest_path, _load_leann_manifest

        path = _leann_manifest_path("test-index")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 999, "files": {}}))

        assert _load_leann_manifest("test-index") is None

    def test_load_manifest_corrupt_json_returns_none(self, temp_db: Path) -> None:
        from paperpipe.leann import _leann_manifest_path, _load_leann_manifest

        path = _leann_manifest_path("test-index")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json {{{")

        assert _load_leann_manifest("test-index") is None

    def test_manifest_key_is_staging_basename(self) -> None:
        from paperpipe.leann import _manifest_key

        assert _manifest_key(Path("/anywhere/.paperpipe/.pqa_papers/neus2.pdf")) == "neus2.pdf"

    def test_migrate_manifest_key_resolved_symlink(self) -> None:
        from paperpipe.leann import _migrate_manifest_key

        # Legacy key from a resolved symlink: every paper's file is named paper.pdf,
        # so the portable key must come from the parent (paper-name) directory.
        assert _migrate_manifest_key("/home/me/.paperpipe/papers/neus2/paper.pdf") == "neus2.pdf"
        assert _migrate_manifest_key("/Users/me/.paperpipe/papers/soft-rasterizer/paper.pdf") == "soft-rasterizer.pdf"

    def test_migrate_manifest_key_unresolved_and_idempotent(self) -> None:
        from paperpipe.leann import _migrate_manifest_key

        # Unresolved staging path and already-portable key both collapse to the basename.
        assert _migrate_manifest_key("/home/me/.paperpipe/.pqa_papers/neus2.pdf") == "neus2.pdf"
        assert _migrate_manifest_key("neus2.pdf") == "neus2.pdf"

    def test_load_manifest_migrates_legacy_absolute_keys(self, temp_db: Path) -> None:
        import json

        from paperpipe.leann import _leann_manifest_path, _load_leann_manifest

        path = _leann_manifest_path("test-index")
        path.parent.mkdir(parents=True, exist_ok=True)
        legacy = _make_manifest(
            files={
                "/home/me/.paperpipe/papers/neus2/paper.pdf": {
                    "mtime": 1.0,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                },
                "/home/me/.paperpipe/papers/nerf/paper.pdf": {
                    "mtime": 2.0,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                },
            }
        )
        path.write_text(json.dumps(legacy))

        loaded = _load_leann_manifest("test-index")
        assert loaded is not None
        assert set(loaded["files"]) == {"neus2.pdf", "nerf.pdf"}

        # Migration must be persisted to disk, not just applied in memory.
        on_disk = json.loads(path.read_text())
        assert set(on_disk["files"]) == {"neus2.pdf", "nerf.pdf"}

    def test_load_backend_meta_helpers(self, temp_db: Path) -> None:
        import json

        from paperpipe.leann import _leann_index_meta_path, _load_leann_backend_kwargs, _load_leann_backend_name

        meta_path = _leann_index_meta_path("test-index")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps({"backend_name": "hnsw", "backend_kwargs": {"graph_degree": 32, "complexity": 64}})
        )

        assert _load_leann_backend_name("test-index") == "hnsw"
        assert _load_leann_backend_kwargs("test-index") == {"graph_degree": 32, "complexity": 64}

    def test_load_backend_meta_helpers_invalid(self, temp_db: Path) -> None:
        from paperpipe.leann import _leann_index_meta_path, _load_leann_backend_kwargs, _load_leann_backend_name

        meta_path = _leann_index_meta_path("test-index")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("not valid json {{{")

        assert _load_leann_backend_name("test-index") is None
        assert _load_leann_backend_kwargs("test-index") == {}

    def test_create_initial_manifest(self, temp_db: Path) -> None:
        from paperpipe.leann import _create_initial_manifest, _load_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        (docs_dir / "paper1.pdf").touch()
        (docs_dir / "paper2.pdf").touch()

        manifest = _create_initial_manifest(
            index_name="test-index",
            docs_dir=docs_dir,
            is_compact=False,
            embedding_mode="ollama",
            embedding_model="nomic-embed-text",
        )

        assert manifest["is_compact"] is False
        assert len(manifest["files"]) == 2
        assert all(f.endswith(".pdf") for f in manifest["files"])

        # Verify it was persisted
        loaded = _load_leann_manifest("test-index")
        assert loaded is not None
        assert len(loaded["files"]) == 2


class TestLeannIndexDelta:
    """Tests for computing index delta (new/changed/removed files)."""

    def test_compute_delta_all_new_files(self, temp_db: Path) -> None:
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        (docs_dir / "new1.pdf").touch()
        (docs_dir / "new2.pdf").touch()

        delta = _compute_index_delta(docs_dir, None)

        assert len(delta.new_files) == 2
        assert len(delta.changed_files) == 0
        assert len(delta.removed_files) == 0
        assert delta.unchanged_count == 0

    def test_compute_delta_unchanged_files(self, temp_db: Path) -> None:
        import time

        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.touch()

        # Small delay to ensure mtime is stable
        time.sleep(0.01)

        manifest = _make_manifest(
            files={
                pdf.name: {
                    "mtime": pdf.stat().st_mtime,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        assert len(delta.new_files) == 0
        assert len(delta.changed_files) == 0
        assert delta.unchanged_count == 1

    def test_compute_delta_changed_mtime(self, temp_db: Path) -> None:
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.touch()

        manifest = _make_manifest(
            files={
                pdf.name: {
                    "mtime": pdf.stat().st_mtime - 100,  # Old mtime
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        assert len(delta.new_files) == 0
        assert len(delta.changed_files) == 1
        assert delta.unchanged_count == 0

    def test_compute_delta_tolerates_subsecond_mtime_drift(self, temp_db: Path) -> None:
        # A synced index loses sub-second mtime precision; such files must stay "unchanged".
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.touch()

        manifest = _make_manifest(
            files={
                pdf.name: {
                    "mtime": pdf.stat().st_mtime + 0.9,  # sub-second drift from precision loss
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        assert len(delta.changed_files) == 0
        assert delta.unchanged_count == 1

    def test_compute_delta_size_is_primary_signal(self, temp_db: Path) -> None:
        # When size is recorded, it decides change — a differing size means changed even if mtime
        # matches, and a matching size means unchanged even if mtime drifts far (cross-sync case).
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.write_bytes(b"hello world")
        st = pdf.stat()

        # Same size, but mtime wildly off: size wins -> unchanged.
        same_size = _make_manifest(
            files={
                pdf.name: {
                    "mtime": st.st_mtime - 10_000,
                    "size": st.st_size,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )
        delta = _compute_index_delta(docs_dir, same_size)
        assert delta.unchanged_count == 1
        assert len(delta.changed_files) == 0

        # Different size, but mtime identical: size wins -> changed.
        diff_size = _make_manifest(
            files={
                pdf.name: {
                    "mtime": st.st_mtime,
                    "size": st.st_size + 1,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )
        delta = _compute_index_delta(docs_dir, diff_size)
        assert len(delta.changed_files) == 1
        assert delta.unchanged_count == 0

    def test_compute_delta_backfills_size_for_legacy_unchanged(self, temp_db: Path) -> None:
        # A migrated legacy entry (no size) judged unchanged must gain a size in place so future
        # comparisons use the robust filesystem-invariant signal instead of the mtime window.
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.write_bytes(b"hello world")

        manifest = _make_manifest(
            files={
                pdf.name: {
                    "mtime": pdf.stat().st_mtime,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        assert delta.unchanged_count == 1
        assert delta.backfilled_count == 1
        assert manifest["files"][pdf.name].get("size") == pdf.stat().st_size

    def test_compute_delta_removed_files(self, temp_db: Path) -> None:
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        manifest = _make_manifest(
            files={
                "removed.pdf": {
                    "mtime": 12345.0,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        assert len(delta.removed_files) == 1
        assert delta.removed_files[0] == "removed.pdf"

    def test_compute_delta_skips_error_status(self, temp_db: Path) -> None:
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "failed.pdf"
        pdf.touch()

        manifest = _make_manifest(
            files={
                pdf.name: {
                    "mtime": 0,  # Different mtime, but status is error
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "error",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        # Error files should be counted as unchanged (skipped), not changed
        assert len(delta.changed_files) == 0
        assert delta.unchanged_count == 1


class TestLeannIncrementalUpdate:
    """Tests for incremental update functionality."""

    def test_leann_pdf_text_sanitizers_clean_surrogates_before_chunking(self) -> None:
        from paperpipe.leann import _install_leann_pdf_text_sanitizers

        class FakeDoc:
            def __init__(self) -> None:
                self.text = "reader \ud835 text"
                self.metadata = {"source": "meta \ud835"}

            def get_content(self) -> str:
                return self.text

        class FakeReader:
            def load_data(self) -> list[FakeDoc]:
                return [FakeDoc()]

        def bad_pymupdf(_path: str) -> str:
            return "bad \ud835\udc4f text"

        def bad_pdfplumber(_path: str) -> str:
            return "also \ud835 bad"

        leann_cli_module = types.SimpleNamespace(
            extract_pdf_text_with_pymupdf=bad_pymupdf,
            extract_pdf_text_with_pdfplumber=bad_pdfplumber,
            SimpleDirectoryReader=FakeReader,
        )

        _install_leann_pdf_text_sanitizers(leann_cli_module)

        assert leann_cli_module.extract_pdf_text_with_pymupdf("paper.pdf").encode("utf-8")
        assert leann_cli_module.extract_pdf_text_with_pdfplumber("paper.pdf").encode("utf-8")
        doc = FakeReader().load_data()[0]
        assert doc.get_content().encode("utf-8")
        assert doc.metadata["source"].encode("utf-8")

    def test_incremental_update_error_no_manifest(self, temp_db: Path) -> None:
        from paperpipe.leann import IncrementalUpdateError, _leann_incremental_update

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        with pytest.raises(IncrementalUpdateError, match="No manifest found"):
            _leann_incremental_update(
                index_name="nonexistent",
                docs_dir=docs_dir,
                embedding_mode="ollama",
                embedding_model="nomic-embed-text",
            )

    def test_incremental_update_error_compact_index(self, temp_db: Path) -> None:
        from paperpipe.leann import IncrementalUpdateError, _leann_incremental_update, _save_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        manifest = _make_manifest(is_compact=True)
        _save_leann_manifest("test-index", manifest)

        with pytest.raises(IncrementalUpdateError, match="compact"):
            _leann_incremental_update(
                index_name="test-index",
                docs_dir=docs_dir,
                embedding_mode="ollama",
                embedding_model="nomic-embed-text",
            )

    def test_incremental_update_error_embedding_mismatch(self, temp_db: Path) -> None:
        from paperpipe.leann import IncrementalUpdateError, _leann_incremental_update, _save_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        manifest = _make_manifest()
        _save_leann_manifest("test-index", manifest)
        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "test-index")

        with pytest.raises(IncrementalUpdateError, match="mismatch"):
            _leann_incremental_update(
                index_name="test-index",
                docs_dir=docs_dir,
                embedding_mode="openai",  # Different mode
                embedding_model="nomic-embed-text",
            )

    def test_incremental_update_removed_files_warns_and_cleans_manifest(
        self, temp_db: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from paperpipe.leann import _leann_incremental_update, _load_leann_manifest, _save_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        manifest = _make_manifest(
            files={
                "removed.pdf": {
                    "mtime": 12345.0,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )
        _save_leann_manifest("test-index", manifest)
        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "test-index")

        # Should not raise; removed files trigger a warning and manifest cleanup instead
        added, unchanged, errors = _leann_incremental_update(
            index_name="test-index",
            docs_dir=docs_dir,
            embedding_mode="ollama",
            embedding_model="nomic-embed-text",
        )
        assert added == 0
        assert errors == 0

        captured = capsys.readouterr()
        assert "stale vectors" in captured.err or "removed paper" in captured.err

        # Removed entry must be pruned from the manifest
        updated = _load_leann_manifest("test-index")
        assert updated is not None
        assert "removed.pdf" not in updated["files"]

    def test_incremental_update_no_changes_returns_zero(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from paperpipe.leann import _leann_incremental_update, _save_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.touch()

        manifest = _make_manifest(
            files={
                pdf.name: {
                    "mtime": pdf.stat().st_mtime,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )
        _save_leann_manifest("test-index", manifest)
        _write_leann_index_stub(temp_db / ".leann" / "indexes" / "test-index")

        # Mock LEANN API import to avoid requiring LEANN
        class MockLeannBuilder:
            def __init__(self, **kwargs):
                pass

            def add_document(self, path):
                pass

            def update_index(self, path):
                pass

        mock_leann = types.ModuleType("leann")
        mock_leann_api = types.ModuleType("leann.api")
        cast(Any, mock_leann_api).LeannBuilder = MockLeannBuilder
        monkeypatch.setitem(__import__("sys").modules, "leann", mock_leann)
        monkeypatch.setitem(__import__("sys").modules, "leann.api", mock_leann_api)

        added, unchanged, errors = _leann_incremental_update(
            index_name="test-index",
            docs_dir=docs_dir,
            embedding_mode="ollama",
            embedding_model="nomic-embed-text",
        )

        assert added == 0
        assert unchanged == 1
        assert errors == 0


class TestRedactCmd:
    """Tests for _redact_cmd (Fix #4: API key leakage in logs)."""

    def test_redacts_api_key(self):
        cmd = ["leann", "ask", "papers", "query", "--api-key", "sk-secret-123"]
        result = _redact_cmd(cmd)
        assert "sk-secret-123" not in result
        assert "***" in result

    def test_redacts_embedding_api_key(self):
        cmd = ["leann", "build", "papers", "--embedding-api-key", "sk-embed-456"]
        result = _redact_cmd(cmd)
        assert "sk-embed-456" not in result
        assert "***" in result

    def test_no_false_redaction(self):
        cmd = ["leann", "ask", "papers", "query", "--model", "gpt-4o"]
        result = _redact_cmd(cmd)
        assert "gpt-4o" in result
        assert "***" not in result

    def test_key_at_end(self):
        """Flag at end of list with no value should not crash."""
        cmd = ["leann", "ask", "--api-key"]
        result = _redact_cmd(cmd)
        assert "--api-key" in result

    def test_redacts_equals_form(self):
        cmd = ["leann", "ask", "--api-key=sk-secret"]
        result = _redact_cmd(cmd)
        assert "sk-secret" not in result
        assert "--api-key=***" in result
