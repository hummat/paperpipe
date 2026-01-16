"""Tests for paperpipe/cli/ask.py (ask command)."""

from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import types
import zlib
from pathlib import Path

import click
import pytest

import paperpipe.config as config
import paperpipe.paperqa as paperqa

from .conftest import MockPopen, cli_mod


class TestAskCommand:
    def test_ask_constructs_correct_command(self, temp_db: Path, monkeypatch):
        # Mock pqa availability
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        # Mock subprocess.Popen (used for pqa with streaming output)
        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        # Add a dummy paper PDF so PaperQA has something to index.
        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--pqa-llm",
                "my-llm",
                "--pqa-embedding",
                "my-embed",
                "--pqa-temperature",
                "0.7",
                "--pqa-verbosity",
                "2",
            ],
        )

        assert result.exit_code == 0

        # Verify pqa was called with correct args
        pqa_call, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "pqa" in pqa_call
        assert "--settings" in pqa_call
        assert "default" in pqa_call
        assert "--parsing.multimodal" in pqa_call
        assert "OFF" in pqa_call
        assert "--agent.index.index_directory" in pqa_call
        assert str(temp_db / ".pqa_index") in pqa_call
        assert "--agent.index.paper_directory" in pqa_call
        assert str(temp_db / ".pqa_papers") in pqa_call
        assert "--agent.index.sync_with_paper_directory" in pqa_call
        assert "ask" in pqa_call
        assert "query" in pqa_call
        assert "--llm" in pqa_call
        assert "my-llm" in pqa_call
        assert "--embedding" in pqa_call
        assert "my-embed" in pqa_call
        assert "--summary_llm" in pqa_call
        summary_idx = pqa_call.index("--summary_llm") + 1
        assert pqa_call[summary_idx] == "my-llm"
        assert "--parsing.enrichment_llm" in pqa_call
        enrich_idx = pqa_call.index("--parsing.enrichment_llm") + 1
        assert pqa_call[enrich_idx] == "my-llm"
        assert "--temperature" in pqa_call
        assert "0.7" in pqa_call
        assert "--verbosity" in pqa_call
        assert "2" in pqa_call
        assert pqa_kwargs.get("cwd") == config.PAPERS_DIR
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

    def test_ask_injects_ollama_timeout_config_by_default(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        def fake_prepare_ollama_env(env: dict[str, str]) -> None:
            env["OLLAMA_API_BASE"] = "http://127.0.0.1:11434"

        monkeypatch.setattr(config, "_prepare_ollama_env", fake_prepare_ollama_env)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda api_base: None)
        monkeypatch.setenv("PAPERPIPE_PQA_OLLAMA_TIMEOUT", "123")

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--pqa-llm",
                "ollama/olmo-3:7b",
                "--pqa-embedding",
                "voyage/voyage-3.5",
            ],
        )
        assert result.exit_code == 0

        pqa_call, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--llm_config" in pqa_call
        assert "--summary_llm_config" in pqa_call
        assert "--parsing.enrichment_llm_config" in pqa_call

        llm_cfg = pqa_call[pqa_call.index("--llm_config") + 1]
        assert '"timeout": 123.0' in llm_cfg

        env = pqa_kwargs.get("env") or {}
        assert env.get("PQA_LITELLM_MAX_CALLBACKS") == "1000"
        assert env.get("LMI_LITELLM_MAX_CALLBACKS") == "1000"
        assert env.get("OLLAMA_API_BASE") == "http://127.0.0.1:11434"

    def test_ask_does_not_override_user_llm_config(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        def fake_prepare_ollama_env(env: dict[str, str]) -> None:
            env["OLLAMA_API_BASE"] = "http://127.0.0.1:11434"

        monkeypatch.setattr(config, "_prepare_ollama_env", fake_prepare_ollama_env)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda api_base: None)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--pqa-llm",
                "ollama/olmo-3:7b",
                "--pqa-embedding",
                "voyage/voyage-3.5",
                # Prevent summary/enrichment defaults from being ollama, so we only exercise llm_config behavior.
                "--pqa-summary-llm",
                "gpt-4o-mini",
                "--parsing.enrichment_llm",
                "gpt-4o-mini",
                "--llm_config",
                '{"router_kwargs":{"timeout":999}}',
            ],
        )
        assert result.exit_code == 0

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert pqa_call.count("--llm_config") == 1
        assert pqa_call[pqa_call.index("--llm_config") + 1] == '{"router_kwargs":{"timeout":999}}'
        assert "--summary_llm_config" not in pqa_call
        assert "--parsing.enrichment_llm_config" not in pqa_call

    def test_paperqa_ask_evidence_blocks_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummySettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DummyText:
            name = "paper1"
            pages = "1-2"
            section = "sec"

        class DummyContext:
            text = DummyText()
            context = "snippet"

        class DummySession:
            contexts = [DummyContext()]

        class DummyResponse:
            answer = "answer"
            session = DummySession()

        def dummy_ask(_query: str, *, settings):
            assert isinstance(settings, DummySettings)
            return DummyResponse()

        dummy_mod = types.ModuleType("paperqa")
        dummy_mod.Settings = DummySettings  # type: ignore[attr-defined]
        dummy_mod.ask = dummy_ask  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "paperqa", dummy_mod)

        payload = paperqa._paperqa_ask_evidence_blocks(
            cmd=[
                "pqa",
                "--llm",
                "my-llm",
                "--embedding",
                "my-embed",
                "--summary_llm",
                "my-summary",
                "--temperature",
                "0.1",
                "--verbosity",
                "2",
                "--parsing.multimodal",
                "OFF",
                "--agent.agent_type",
                "fake",
                "--agent.timeout",
                "12",
                "--agent.rebuild_index",
                "true",
                "--agent.index.paper_directory",
                "/papers",
                "--agent.index.index_directory",
                "/index",
                "--agent.index.name",
                "idx",
                "--agent.index.sync_with_paper_directory",
                "true",
                "--agent.index.concurrency",
                "2",
                "--answer.answer_length",
                "short",
                "--answer.evidence_k",
                "10",
                "--answer.answer_max_sources",
                "5",
            ],
            query="q",
        )

        assert payload["backend"] == "pqa"
        assert payload["question"] == "q"
        assert payload["answer"] == "answer"
        evidence = payload["evidence"]
        assert isinstance(evidence, list) and evidence
        assert evidence[0]["paper"] == "paper1"
        assert evidence[0]["page"] == "1-2"
        assert evidence[0]["section"] == "sec"
        assert evidence[0]["snippet"] == "snippet"

    def test_ask_pqa_agent_type_flag_is_passed(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "--pqa-agent-type", "fake"])
        assert result.exit_code == 0, result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--agent.agent_type" in pqa_call
        idx = pqa_call.index("--agent.agent_type") + 1
        assert pqa_call[idx] == "fake"

    def test_ask_filters_noisy_pqa_output_by_default(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="New file to index: test-paper.pdf...\nAnswer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query"])
        assert result.exit_code == 0, result.output
        assert "Answer" in result.output
        assert "New file to index:" not in result.output

    def test_ask_pqa_raw_disables_output_filtering(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="New file to index: test-paper.pdf...\nAnswer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "--pqa-raw"])
        assert result.exit_code == 0, result.output
        assert "Answer" in result.output
        assert "New file to index:" in result.output

    def test_ask_evidence_blocks_outputs_json(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)
        monkeypatch.setattr(
            paperqa,
            "_paperqa_ask_evidence_blocks",
            lambda **kwargs: {"backend": "pqa", "question": "q", "answer": "a", "evidence": []},
        )

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "--format", "evidence-blocks"])
        assert result.exit_code == 0, result.output
        assert '"backend": "pqa"' in result.output

    def test_ask_evidence_blocks_rejects_passthrough_args(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--format", "evidence-blocks", "--agent.search_count", "10"],
        )
        assert result.exit_code != 0
        assert "--format evidence-blocks does not support extra passthrough args" in result.output

    def test_ask_evidence_blocks_reports_errors(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        def boom(**_kwargs):
            raise click.ClickException("boom")

        monkeypatch.setattr(paperqa, "_paperqa_ask_evidence_blocks", boom)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "--format", "evidence-blocks"])
        assert result.exit_code != 0
        assert "Error: boom" in result.output

    def test_ask_ollama_models_prepare_env_for_pqa(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "--pqa-llm", "ollama/qwen3:8b"])
        assert result.exit_code == 0, result.output

        _, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        env = pqa_kwargs.get("env") or {}
        assert env.get("OLLAMA_API_BASE") == "http://localhost:11434"
        assert env.get("OLLAMA_HOST") == "http://localhost:11434"

    def test_ask_retry_failed_index_clears_error_markers(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        # Pretend PaperQA2 previously marked this staged file as ERROR.
        # Also create the staged file so the clear logic can match it.
        staging_dir = temp_db / ".pqa_papers"
        staging_dir.mkdir(parents=True)
        (staging_dir / "test-paper.pdf").touch()
        index_dir = temp_db / ".pqa_index" / "paperpipe_my-embed"
        index_dir.mkdir(parents=True)
        files_zip = index_dir / "files.zip"
        # pickle is required for PaperQA2 index format
        files_zip.write_bytes(
            zlib.compress(
                pickle.dumps({str(staging_dir / "test-paper.pdf"): "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)
            )
        )

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--pqa-llm", "my-llm", "--pqa-embedding", "my-embed", "--pqa-retry-failed"],
        )
        assert result.exit_code == 0

        mapping = paperqa._paperqa_load_index_files_map(files_zip)
        assert mapping == {}

    def test_ask_does_not_override_user_settings(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "-s", "my-settings"])

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--settings" not in pqa_call
        assert "default" not in pqa_call
        assert "-s" in pqa_call
        assert "my-settings" in pqa_call

    def test_ask_does_not_override_user_parsing(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--parsing.multimodal", "ON_WITHOUT_ENRICHMENT"],
        )

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--parsing.multimodal" in pqa_call
        assert "ON_WITHOUT_ENRICHMENT" in pqa_call
        assert "OFF" not in pqa_call

    def test_ask_does_not_force_multimodal_when_pillow_available(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query"])

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--parsing.multimodal" not in pqa_call

    def test_ask_does_not_override_user_index_directory(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--agent.index.index_directory", "/custom/index"],
        )

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--agent.index.index_directory" in pqa_call
        # User-provided value should be preserved; paperpipe should not append its default.
        assert "/custom/index" in pqa_call
        assert str(temp_db / ".pqa_index") not in pqa_call

    def test_ask_does_not_override_user_paper_directory(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--agent.index.paper_directory", "/custom/papers"],
        )

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--agent.index.paper_directory" in pqa_call
        assert "/custom/papers" in pqa_call
        assert str(temp_db / ".pqa_papers") not in pqa_call

    def test_ask_marks_crashing_doc_error_when_custom_paper_directory_used(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        custom_papers = temp_db / "custom_papers"
        custom_papers.mkdir(parents=True)
        crashing_pdf = custom_papers / "crashy.pdf"
        crashing_pdf.write_bytes(b"%PDF-1.4\n% fake\n")

        mock_popen = MockPopen(
            returncode=1,
            stdout="New file to index: crashy.pdf...\nTraceback (most recent call last):\nBoom\n",
        )
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--pqa-embedding", "my-embed", "--agent.index.paper_directory", str(custom_papers)],
        )

        assert result.exit_code == 1
        assert crashing_pdf.exists()

        files_zip = temp_db / ".pqa_index" / "paperpipe_my-embed" / "files.zip"
        raw = zlib.decompress(files_zip.read_bytes())
        mapping = pickle.loads(raw)
        assert mapping[str(crashing_pdf)] == "ERROR"

    def test_ask_first_class_options(self, temp_db: Path, monkeypatch):
        """Test all first-class options are passed correctly to pqa."""
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "test query",
                "--pqa-llm",
                "gpt-4o",
                "--pqa-summary-llm",
                "gpt-4o-mini",
                "--pqa-embedding",
                "text-embedding-3-small",
                "--pqa-temperature",
                "0.5",
                "--pqa-verbosity",
                "3",
                "--pqa-answer-length",
                "about 100 words",
                "--pqa-evidence-k",
                "15",
                "--pqa-max-sources",
                "8",
                "--pqa-timeout",
                "300",
                "--pqa-concurrency",
                "4",
                "--pqa-rebuild-index",
            ],
        )

        assert result.exit_code == 0
        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")

        # Verify all first-class options are passed
        assert "--llm" in pqa_call and "gpt-4o" in pqa_call
        assert "--summary_llm" in pqa_call and "gpt-4o-mini" in pqa_call
        assert "--embedding" in pqa_call and "text-embedding-3-small" in pqa_call
        assert "--temperature" in pqa_call and "0.5" in pqa_call
        assert "--verbosity" in pqa_call and "3" in pqa_call
        assert "--answer.answer_length" in pqa_call and "about 100 words" in pqa_call
        assert "--answer.evidence_k" in pqa_call and "15" in pqa_call
        assert "--answer.answer_max_sources" in pqa_call and "8" in pqa_call
        assert "--agent.timeout" in pqa_call and "300.0" in pqa_call
        assert "--agent.index.concurrency" in pqa_call and "4" in pqa_call
        assert "--agent.rebuild_index" in pqa_call and "true" in pqa_call

    def test_ask_concurrency_defaults_to_one(self, temp_db: Path, monkeypatch):
        """Concurrency should default to 1 when not specified."""
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)
        # Ensure no env override
        monkeypatch.delenv("PAPERPIPE_PQA_CONCURRENCY", raising=False)
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "test"])

        assert result.exit_code == 0
        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")

        # Concurrency should be set to 1 by default
        concurrency_idx = pqa_call.index("--agent.index.concurrency") + 1
        assert pqa_call[concurrency_idx] == "1"


class TestAskErrorHandling:
    def test_ask_excludes_failed_files_from_staging(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        # Create a test paper that exists in the DB
        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        # Create a fake PaperQA2 index where this paper is marked as ERROR
        embedding_model = "my-embed"
        index_name = f"paperpipe_{embedding_model}"

        index_dir = temp_db / ".pqa_index"
        index_path = index_dir / index_name
        index_path.mkdir(parents=True)
        files_zip = index_path / "files.zip"

        # The key in the index map is typically the full path to the staged file.
        staged_path_str = str(temp_db / ".pqa_papers" / "test-paper.pdf")

        # Write the index file with ERROR status (pickle is required for PaperQA2 compatibility)
        files_zip.write_bytes(zlib.compress(pickle.dumps({staged_path_str: "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)))

        runner = pytest.importorskip("click.testing").CliRunner()
        # Run ask WITHOUT --pqa-retry-failed
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "query", "--pqa-embedding", embedding_model, "--agent.index.index_directory", str(index_dir)],
        )

        assert result.exit_code == 0

        # Verify that test-paper.pdf is NOT in the staging directory
        staging_dir = temp_db / ".pqa_papers"
        assert staging_dir.exists()
        assert not (staging_dir / "test-paper.pdf").exists(), "Failed file should not be staged"

        # Verify pqa was still called
        assert len(mock_popen.calls) > 0

    def test_ask_stages_failed_files_with_retry_failed(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        embedding_model = "my-embed"
        index_name = f"paperpipe_{embedding_model}"
        index_dir = temp_db / ".pqa_index"
        index_path = index_dir / index_name
        index_path.mkdir(parents=True)
        files_zip = index_path / "files.zip"

        staged_path_str = str(temp_db / ".pqa_papers" / "test-paper.pdf")
        # pickle is required for PaperQA2 index format compatibility
        files_zip.write_bytes(zlib.compress(pickle.dumps({staged_path_str: "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)))

        runner = pytest.importorskip("click.testing").CliRunner()
        # Run ask WITH --pqa-retry-failed
        result = runner.invoke(
            cli_mod.cli,
            [
                "ask",
                "query",
                "--pqa-embedding",
                embedding_model,
                "--agent.index.index_directory",
                str(index_dir),
                "--pqa-retry-failed",
            ],
        )

        assert result.exit_code == 0

        # Verify that test-paper.pdf IS in the staging directory (re-staged)
        staging_dir = temp_db / ".pqa_papers"
        assert (staging_dir / "test-paper.pdf").exists(), "Failed file should be staged when retrying"

        # Also verify that the ERROR marker was cleared from the index file
        mapping = paperqa._paperqa_load_index_files_map(files_zip)
        assert mapping == {}, "ERROR markers should be cleared"
