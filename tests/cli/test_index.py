"""Tests for paperpipe/cli/index.py (index command)."""

from __future__ import annotations

import csv
import pickle
import shutil
import subprocess
import zlib
from pathlib import Path

import pytest

import paperpipe.config as config
import paperpipe.paperqa as paperqa

from .conftest import MockPopen, cli_mod


class _HookedPopenProcess:
    def __init__(self, returncode: int, stdout: str, on_wait) -> None:
        self._returncode = returncode
        self._stdout_lines = stdout.splitlines(keepends=True) if stdout else []
        self.stdout = iter(self._stdout_lines)
        self._on_wait = on_wait

    def wait(self) -> int:
        self._on_wait()
        return self._returncode


class _HookedPopen:
    def __init__(self, returncode: int, stdout: str, on_wait) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self._returncode = returncode
        self._stdout = stdout
        self._on_wait = on_wait

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return _HookedPopenProcess(self._returncode, self._stdout, self._on_wait)


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
        assert "--agent.index.manifest_file" in pqa_call
        assert "--parsing.multimodal" in pqa_call
        assert "OFF" in pqa_call
        assert "--parsing.use_doc_details" in pqa_call
        assert "false" in pqa_call
        assert "--parsing.reader_config" in pqa_call
        reader_config = pqa_call[pqa_call.index("--parsing.reader_config") + 1]
        assert reader_config == '{"chunk_chars":5000,"overlap":250,"use_block_parsing":true}'
        assert "--index" in pqa_call and "paperpipe_my-embed" in pqa_call
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()
        assert (temp_db / ".pqa_papers_manifest.csv").exists()

    def test_index_backend_pqa_writes_manifest_from_paper_metadata(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").touch()
        (paper_dir / "meta.json").write_text(
            '{"title": "Test Paper", "authors": ["Ada Lovelace", "Grace Hopper"], "published": "2024-01-02"}'
        )

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed"])
        assert result.exit_code == 0, result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        manifest_path = temp_db / ".pqa_papers_manifest.csv"
        assert pqa_call[pqa_call.index("--agent.index.manifest_file") + 1] == str(manifest_path)

        rows = list(csv.DictReader(manifest_path.read_text().splitlines()))
        assert rows == [
            {
                "file_location": "test-paper.pdf",
                "docname": "test-paper",
                "dockey": "test-paper",
                "citation": "Ada Lovelace et al. (2024). Test Paper.",
                "title": "Test Paper",
                "fields_to_overwrite_from_metadata": "[]",
            }
        ]

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

    def test_index_bad_pdf_failure_marks_and_removes_managed_staging(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)

        paper_dir = temp_db / "papers" / "objaverse"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n% fake\n")

        mock_popen = MockPopen(
            returncode=1,
            stdout=(
                "New file to index: objaverse.pdf...\n"
                "Traceback (most recent call last):\n"
                "ImpossibleParsingError: The text in page 3 of 15 was 16120357 chars long, "
                "which exceeds the 1280000 char limit for the PDF at path /tmp/objaverse.pdf.\n"
            ),
        )
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed"])

        assert result.exit_code == 1
        assert "PaperQA2 hit a PDF parsing failure while indexing: objaverse" in result.output
        assert "Replace or repair the PDF" in result.output
        assert "papi remove objaverse" in result.output
        assert not (temp_db / ".pqa_papers" / "objaverse.pdf").exists()

        files_zip = temp_db / ".pqa_index" / "paperpipe_my-embed" / "files.zip"
        mapping = pickle.loads(zlib.decompress(files_zip.read_bytes()))
        assert mapping == {str(temp_db / ".pqa_papers" / "objaverse.pdf"): "ERROR"}

    def test_index_bad_pdf_successful_pqa_exit_still_reports_partial_failure(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)

        paper_dir = temp_db / "papers" / "objaverse"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n% fake\n")
        files_zip = temp_db / ".pqa_index" / "paperpipe_my-embed" / "files.zip"

        def write_pqa_error_marker() -> None:
            files_zip.parent.mkdir(parents=True, exist_ok=True)
            mapping = {str(temp_db / ".pqa_papers" / "objaverse.pdf"): "ERROR"}
            files_zip.write_bytes(zlib.compress(pickle.dumps(mapping, protocol=pickle.HIGHEST_PROTOCOL)))

        mock_popen = _HookedPopen(
            returncode=0,
            stdout=(
                "New file to index: objaverse.pdf...\n"
                "Error parsing objaverse.pdf, skipping index for this file.\n"
                "ImpossibleParsingError: The text in page 3 of 15 was 16120357 chars long, "
                "which exceeds the 1280000 char limit for the PDF at path /tmp/objaverse.pdf.\n"
            ),
            on_wait=write_pqa_error_marker,
        )
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed", "--pqa-retry-failed"])

        assert result.exit_code == 1
        assert "PaperQA2 hit a PDF parsing failure while indexing: objaverse" in result.output
        assert not (temp_db / ".pqa_papers" / "objaverse.pdf").exists()
