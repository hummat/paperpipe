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
from paperpipe.leann import FileEntry, LeannManifest

# Import the CLI module explicitly (avoid resolving to the package's cli function).
cli_mod = import_module("paperpipe.cli")
# Import the index submodule for patching _leann_build_index
cli_index_mod = import_module("paperpipe.cli.index")


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
                str(pdf.resolve()): {
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
                str(pdf.resolve()): {
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

    def test_compute_delta_removed_files(self, temp_db: Path) -> None:
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        manifest = _make_manifest(
            files={
                "/nonexistent/removed.pdf": {
                    "mtime": 12345.0,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )

        delta = _compute_index_delta(docs_dir, manifest)

        assert len(delta.removed_files) == 1
        assert delta.removed_files[0] == "/nonexistent/removed.pdf"

    def test_compute_delta_skips_error_status(self, temp_db: Path) -> None:
        from paperpipe.leann import _compute_index_delta

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "failed.pdf"
        pdf.touch()

        manifest = _make_manifest(
            files={
                str(pdf.resolve()): {
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

        with pytest.raises(IncrementalUpdateError, match="mismatch"):
            _leann_incremental_update(
                index_name="test-index",
                docs_dir=docs_dir,
                embedding_mode="openai",  # Different mode
                embedding_model="nomic-embed-text",
            )

    def test_incremental_update_error_removed_files(self, temp_db: Path) -> None:
        from paperpipe.leann import IncrementalUpdateError, _leann_incremental_update, _save_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)

        manifest = _make_manifest(
            files={
                "/nonexistent/removed.pdf": {
                    "mtime": 12345.0,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )
        _save_leann_manifest("test-index", manifest)

        with pytest.raises(IncrementalUpdateError, match="Removed files"):
            _leann_incremental_update(
                index_name="test-index",
                docs_dir=docs_dir,
                embedding_mode="ollama",
                embedding_model="nomic-embed-text",
            )

    def test_incremental_update_no_changes_returns_zero(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from paperpipe.leann import _leann_incremental_update, _save_leann_manifest

        docs_dir = temp_db / "docs"
        docs_dir.mkdir(parents=True)
        pdf = docs_dir / "paper.pdf"
        pdf.touch()

        manifest = _make_manifest(
            files={
                str(pdf.resolve()): {
                    "mtime": pdf.stat().st_mtime,
                    "indexed_at": "2026-01-26T10:00:00Z",
                    "status": "ok",
                }
            }
        )
        _save_leann_manifest("test-index", manifest)

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
