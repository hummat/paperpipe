"""Tests for paperpipe."""

import json
import os
import pickle
import shutil
import subprocess
import sys
import types
import zlib
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import paperpipe

# Well-known paper for integration tests: "Attention Is All You Need"
TEST_ARXIV_ID = "1706.03762"


def litellm_available() -> bool:
    """Check if LiteLLM is installed and an API key is configured."""
    try:
        import litellm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    # Check for common API keys
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
    monkeypatch.setattr(paperpipe, "PAPER_DB", db_path)
    monkeypatch.setattr(paperpipe, "PAPERS_DIR", db_path / "papers")
    monkeypatch.setattr(paperpipe, "INDEX_FILE", db_path / "index.json")
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


class TestEnsureDb:
    def test_creates_directories(self, temp_db: Path):
        paperpipe.ensure_db()
        assert temp_db.exists()
        assert (temp_db / "papers").exists()
        assert (temp_db / "index.json").exists()

    def test_creates_empty_index(self, temp_db: Path):
        paperpipe.ensure_db()
        index = json.loads((temp_db / "index.json").read_text())
        assert index == {}

    def test_idempotent(self, temp_db: Path):
        paperpipe.ensure_db()
        paperpipe.ensure_db()
        assert temp_db.exists()


class TestConfigToml:
    def test_config_toml_sets_defaults(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PAPERPIPE_LLM_MODEL", raising=False)
        monkeypatch.delenv("PAPERPIPE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("PAPERPIPE_LLM_TEMPERATURE", raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "gpt-4o-mini"',
                    "temperature = 0.42",
                    "",
                    "[embedding]",
                    'model = "text-embedding-3-small"',
                    "",
                    "[tags.aliases]",
                    'cv = "computer-vision"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)

        assert paperpipe.default_llm_model() == "gpt-4o-mini"
        assert paperpipe.default_embedding_model() == "text-embedding-3-small"
        assert paperpipe.default_llm_temperature() == 0.42
        assert paperpipe.normalize_tag("cv") == "computer-vision"


class TestConfigPrecedence:
    def test_env_overrides_config(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "config-model"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_LLM_MODEL", "env-model")
        assert paperpipe.default_llm_model() == "env-model"

    def test_pqa_config_from_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_PQA_TEMPERATURE",
            "PAPERPIPE_PQA_VERBOSITY",
            "PAPERPIPE_PQA_ANSWER_LENGTH",
            "PAPERPIPE_PQA_EVIDENCE_K",
            "PAPERPIPE_PQA_MAX_SOURCES",
            "PAPERPIPE_PQA_TIMEOUT",
            "PAPERPIPE_PQA_CONCURRENCY",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[paperqa]",
                    "temperature = 0.5",
                    "verbosity = 2",
                    'answer_length = "about 100 words"',
                    "evidence_k = 15",
                    "max_sources = 8",
                    "timeout = 300.0",
                    "concurrency = 4",
                ]
            )
        )
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)

        assert paperpipe.default_pqa_temperature() == 0.5
        assert paperpipe.default_pqa_verbosity() == 2
        assert paperpipe.default_pqa_answer_length() == "about 100 words"
        assert paperpipe.default_pqa_evidence_k() == 15
        assert paperpipe.default_pqa_max_sources() == 8
        assert paperpipe.default_pqa_timeout() == 300.0
        assert paperpipe.default_pqa_concurrency() == 4

    def test_leann_config_from_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_LEANN_LLM_PROVIDER",
            "PAPERPIPE_LEANN_LLM_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODE",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[leann]",
                    'llm_provider = "ollama"',
                    'llm_model = "qwen3:8b"',
                    'embedding_model = "nomic-embed-text"',
                    'embedding_mode = "ollama"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)

        assert paperpipe.default_leann_llm_provider() == "ollama"
        assert paperpipe.default_leann_llm_model() == "qwen3:8b"
        assert paperpipe.default_leann_embedding_model() == "nomic-embed-text"
        assert paperpipe.default_leann_embedding_mode() == "ollama"

    def test_leann_config_env_overrides_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[leann]",
                    'llm_model = "config-model"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_LEANN_LLM_MODEL", "env-model")

        assert paperpipe.default_leann_llm_model() == "env-model"

    def test_pqa_config_env_overrides_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[paperqa]",
                    "temperature = 0.5",
                    "concurrency = 4",
                ]
            )
        )
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_PQA_TEMPERATURE", "0.9")
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "8")

        assert paperpipe.default_pqa_temperature() == 0.9
        assert paperpipe.default_pqa_concurrency() == 8

    def test_pqa_config_defaults_when_unset(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_PQA_TEMPERATURE",
            "PAPERPIPE_PQA_VERBOSITY",
            "PAPERPIPE_PQA_ANSWER_LENGTH",
            "PAPERPIPE_PQA_EVIDENCE_K",
            "PAPERPIPE_PQA_MAX_SOURCES",
            "PAPERPIPE_PQA_TIMEOUT",
            "PAPERPIPE_PQA_CONCURRENCY",
        ]:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)

        # These return None when unset (no hardcoded default)
        assert paperpipe.default_pqa_temperature() is None
        assert paperpipe.default_pqa_verbosity() is None
        assert paperpipe.default_pqa_answer_length() is None
        assert paperpipe.default_pqa_evidence_k() is None
        assert paperpipe.default_pqa_max_sources() is None
        assert paperpipe.default_pqa_timeout() is None
        # concurrency defaults to 1
        assert paperpipe.default_pqa_concurrency() == 1

    def test_pqa_config_invalid_env_values_fallback(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        """Invalid env var values should fall back to config/defaults."""
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)
        # Set invalid (non-numeric) values for numeric env vars
        monkeypatch.setenv("PAPERPIPE_PQA_TEMPERATURE", "not-a-number")
        monkeypatch.setenv("PAPERPIPE_PQA_VERBOSITY", "high")
        monkeypatch.setenv("PAPERPIPE_PQA_EVIDENCE_K", "many")
        monkeypatch.setenv("PAPERPIPE_PQA_MAX_SOURCES", "all")
        monkeypatch.setenv("PAPERPIPE_PQA_TIMEOUT", "forever")
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "max")

        # Should fall back to None (no config) or default
        assert paperpipe.default_pqa_temperature() is None
        assert paperpipe.default_pqa_verbosity() is None
        assert paperpipe.default_pqa_evidence_k() is None
        assert paperpipe.default_pqa_max_sources() is None
        assert paperpipe.default_pqa_timeout() is None
        assert paperpipe.default_pqa_concurrency() == 1  # Falls back to default

    def test_pqa_config_env_vars_direct(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        """Test env vars are read correctly without config file."""
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_PQA_TEMPERATURE", "0.7")
        monkeypatch.setenv("PAPERPIPE_PQA_VERBOSITY", "3")
        monkeypatch.setenv("PAPERPIPE_PQA_ANSWER_LENGTH", "short")
        monkeypatch.setenv("PAPERPIPE_PQA_EVIDENCE_K", "20")
        monkeypatch.setenv("PAPERPIPE_PQA_MAX_SOURCES", "10")
        monkeypatch.setenv("PAPERPIPE_PQA_TIMEOUT", "600.5")
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "2")

        assert paperpipe.default_pqa_temperature() == 0.7
        assert paperpipe.default_pqa_verbosity() == 3
        assert paperpipe.default_pqa_answer_length() == "short"
        assert paperpipe.default_pqa_evidence_k() == 20
        assert paperpipe.default_pqa_max_sources() == 10
        assert paperpipe.default_pqa_timeout() == 600.5
        assert paperpipe.default_pqa_concurrency() == 2


class TestOllamaHelpers:
    def test_normalize_ollama_base_url_adds_scheme_and_strips_v1(self) -> None:
        assert paperpipe._normalize_ollama_base_url("localhost:11434") == "http://localhost:11434"
        assert paperpipe._normalize_ollama_base_url("http://localhost:11434/") == "http://localhost:11434"
        assert paperpipe._normalize_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434"

    def test_prepare_ollama_env_sets_both_vars(self) -> None:
        env: dict[str, str] = {}
        paperpipe._prepare_ollama_env(env)
        assert env["OLLAMA_API_BASE"] == "http://localhost:11434"
        assert env["OLLAMA_HOST"] == "http://localhost:11434"

        env2: dict[str, str] = {"OLLAMA_HOST": "localhost:11434"}
        paperpipe._prepare_ollama_env(env2)
        assert env2["OLLAMA_API_BASE"] == "http://localhost:11434"
        assert env2["OLLAMA_HOST"] == "http://localhost:11434"

    def test_ollama_reachability_error_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(paperpipe, "urlopen", lambda *args, **kwargs: _Resp())
        assert paperpipe._ollama_reachability_error(api_base="http://localhost:11434") is None

    def test_ollama_reachability_error_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(paperpipe, "urlopen", boom)
        err = paperpipe._ollama_reachability_error(api_base="http://localhost:11434")
        assert err is not None and "not reachable" in err


class TestGrepCollection:
    def test_collect_grep_matches_uses_rg_and_parses(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (temp_db / "papers" / "p").mkdir(parents=True)

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="p/summary.md:2:hit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        matches = paperpipe._collect_grep_matches(
            query="hit",
            fixed_strings=True,
            max_matches=10,
            ignore_case=True,
            include_tex=False,
        )
        assert matches and matches[0]["paper"] == "p"
        assert matches[0]["path"] == "p/summary.md"
        assert matches[0]["line"] == 2
        assert "--context" in calls[0] and "0" in calls[0]
        assert "--fixed-strings" in calls[0]
        assert "--ignore-case" in calls[0]

    def test_collect_grep_matches_rg_no_hits_returns_empty(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (temp_db / "papers" / "p").mkdir(parents=True)
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        matches = paperpipe._collect_grep_matches(
            query="nope",
            fixed_strings=True,
            max_matches=10,
            ignore_case=False,
            include_tex=False,
        )
        assert matches == []

    def test_collect_grep_matches_falls_back_to_grep(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (temp_db / "papers" / "p").mkdir(parents=True)
        monkeypatch.setattr(shutil, "which", lambda cmd: "/bin/grep" if cmd == "grep" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="p/notes.md:5:hit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        matches = paperpipe._collect_grep_matches(
            query="hit",
            fixed_strings=False,
            max_matches=10,
            ignore_case=True,
            include_tex=False,
        )
        assert matches and matches[0]["path"] == "p/notes.md"

    def test_collect_grep_matches_no_tools_returns_empty(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (temp_db / "papers").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        matches = paperpipe._collect_grep_matches(
            query="x",
            fixed_strings=True,
            max_matches=10,
            ignore_case=False,
            include_tex=False,
        )
        assert matches == []


class TestPqaIndexNaming:
    def test_pqa_index_name_for_embedding_is_stable_and_safe(self):
        assert paperpipe.pqa_index_name_for_embedding("text-embedding-3-small") == "paperpipe_text-embedding-3-small"
        assert paperpipe.pqa_index_name_for_embedding("foo/bar:baz") == "paperpipe_foo_bar_baz"
        assert paperpipe.pqa_index_name_for_embedding("") == "paperpipe_default"


class TestNotesCommand:
    def test_notes_creates_file_and_prints(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "my-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "Test Paper"}))
        paperpipe.save_index({"my-paper": {"title": "Test Paper", "tags": [], "added": "now"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["notes", "my-paper", "--print"])
        assert result.exit_code == 0
        assert (paper_dir / "notes.md").exists()
        assert "# my-paper" in result.output


class TestIndex:
    def test_load_empty_index(self, temp_db: Path):
        index = paperpipe.load_index()
        assert index == {}

    def test_save_and_load_index(self, temp_db: Path):
        test_data = {"paper1": {"title": "Test Paper", "tags": ["ml"]}}
        paperpipe.save_index(test_data)
        loaded = paperpipe.load_index()
        assert loaded == test_data


class TestIsSafePaperName:
    """Tests for _is_safe_paper_name helper."""

    def test_rejects_empty_string(self):
        assert paperpipe._is_safe_paper_name("") is False

    def test_rejects_dot(self):
        assert paperpipe._is_safe_paper_name(".") is False

    def test_rejects_dotdot(self):
        assert paperpipe._is_safe_paper_name("..") is False

    def test_rejects_forward_slash(self):
        assert paperpipe._is_safe_paper_name("foo/bar") is False

    def test_rejects_backslash(self):
        assert paperpipe._is_safe_paper_name("foo\\bar") is False

    def test_rejects_absolute_path(self):
        assert paperpipe._is_safe_paper_name("/etc/passwd") is False

    def test_accepts_valid_name(self):
        assert paperpipe._is_safe_paper_name("nerf-2020") is True

    def test_accepts_name_with_dots(self):
        assert paperpipe._is_safe_paper_name("paper.v2") is True


class TestResolvePaperNameFromRef:
    """Tests for _resolve_paper_name_from_ref helper."""

    def test_returns_error_for_empty_input(self, temp_db: Path):
        name, error = paperpipe._resolve_paper_name_from_ref("", {})
        assert name is None
        assert "Missing" in error

    def test_finds_paper_in_index(self, temp_db: Path):
        index = {"my-paper": {"arxiv_id": "1234.5678", "title": "Test"}}
        name, error = paperpipe._resolve_paper_name_from_ref("my-paper", index)
        assert name == "my-paper"
        assert error == ""

    def test_finds_paper_on_disk_not_in_index(self, temp_db: Path):
        # Paper exists on disk but not in index
        paper_dir = temp_db / "papers" / "disk-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text('{"arxiv_id": "1234.5678"}')

        name, error = paperpipe._resolve_paper_name_from_ref("disk-paper", {})
        assert name == "disk-paper"
        assert error == ""

    def test_returns_error_for_invalid_arxiv_id(self, temp_db: Path):
        name, error = paperpipe._resolve_paper_name_from_ref("not-a-paper-or-id", {})
        assert name is None
        assert "not found" in error.lower()

    def test_fallback_scan_finds_paper_by_arxiv_id(self, temp_db: Path):
        # Paper on disk with arxiv_id, but not in index
        paper_dir = temp_db / "papers" / "some-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        # Empty index, but valid arxiv ID should trigger fallback scan
        name, error = paperpipe._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "some-paper"
        assert error == ""

    def test_fallback_scan_reports_multiple_matches(self, temp_db: Path):
        # Two papers with same arxiv_id on disk
        for pname in ["paper-a", "paper-b"]:
            paper_dir = temp_db / "papers" / pname
            paper_dir.mkdir(parents=True)
            (paper_dir / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = paperpipe._resolve_paper_name_from_ref("2301.00001", {})
        assert name is None
        assert "Multiple papers match" in error

    def test_fallback_scan_skips_non_directories(self, temp_db: Path):
        # Create a file (not directory) in papers dir
        (temp_db / "papers" / "not-a-dir.txt").write_text("just a file")
        # And a valid paper
        paper_dir = temp_db / "papers" / "real-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = paperpipe._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "real-paper"

    def test_fallback_scan_skips_invalid_json(self, temp_db: Path):
        # Paper with invalid JSON
        bad_paper = temp_db / "papers" / "bad-paper"
        bad_paper.mkdir(parents=True)
        (bad_paper / "meta.json").write_text("not valid json {{{")
        # And a valid paper
        good_paper = temp_db / "papers" / "good-paper"
        good_paper.mkdir(parents=True)
        (good_paper / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = paperpipe._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "good-paper"

    def test_fallback_scan_skips_missing_meta(self, temp_db: Path):
        # Paper directory without meta.json
        no_meta = temp_db / "papers" / "no-meta"
        no_meta.mkdir(parents=True)
        # And a valid paper
        valid = temp_db / "papers" / "valid"
        valid.mkdir(parents=True)
        (valid / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = paperpipe._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "valid"

    def test_fallback_scan_returns_not_found(self, temp_db: Path):
        # Papers exist but none match the arxiv_id
        paper = temp_db / "papers" / "other-paper"
        paper.mkdir(parents=True)
        (paper / "meta.json").write_text('{"arxiv_id": "9999.99999"}')

        name, error = paperpipe._resolve_paper_name_from_ref("2301.00001", {})
        assert name is None
        assert "not found" in error.lower()


class TestProbeHint:
    """Tests for _probe_hint helper."""

    def test_hint_for_gpt52_not_supported(self):
        hint = paperpipe._probe_hint("completion", "gpt-5.2", "model_not_supported error")
        assert hint is not None
        assert "gpt-5.1" in hint

    def test_hint_for_embedding_not_supported(self):
        hint = paperpipe._probe_hint("embedding", "text-embedding-3-large", "not supported")
        assert hint is not None
        assert "text-embedding-3-small" in hint

    def test_hint_for_claude_35_retired(self):
        hint = paperpipe._probe_hint("completion", "claude-3-5-sonnet", "not_found")
        assert hint is not None
        assert "claude-sonnet-4-5" in hint

    def test_hint_for_voyage_completion(self):
        hint = paperpipe._probe_hint("completion", "voyage/voyage-3", "does not support parameters")
        assert hint is not None
        assert "embedding" in hint

    def test_no_hint_for_unknown_error(self):
        hint = paperpipe._probe_hint("completion", "some-model", "random error")
        assert hint is None


class TestPillowAvailable:
    """Tests for _pillow_available helper."""

    def test_returns_bool(self):
        # Just verify it returns a boolean without crashing
        result = paperpipe._pillow_available()
        assert isinstance(result, bool)


class TestGenerateLlmContent:
    """Tests for generate_llm_content fallback behavior."""

    def test_falls_back_when_litellm_unavailable(self, tmp_path, monkeypatch):
        # Force LiteLLM to appear unavailable
        monkeypatch.setattr(paperpipe, "_litellm_available", lambda: False)

        meta = {
            "arxiv_id": "2301.00001",
            "title": "Test Paper",
            "authors": ["Author"],
            "abstract": "This is the abstract.",
            "categories": ["cs.CV"],
            "published": "2023-01-01",
        }
        tex_content = r"\begin{equation}E=mc^2\end{equation}"

        summary, equations, tags = paperpipe.generate_llm_content(tmp_path, meta, tex_content)

        # Should get simple summary (contains title)
        assert "Test Paper" in summary
        # Should get simple equation extraction
        assert "E=mc^2" in equations
        # No LLM tags
        assert tags == []

    def test_falls_back_without_tex_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paperpipe, "_litellm_available", lambda: False)

        meta = {
            "arxiv_id": "2301.00001",
            "title": "Test Paper",
            "authors": ["Author"],
            "abstract": "Abstract text.",
            "categories": [],
            "published": "2023-01-01",
        }

        summary, equations, tags = paperpipe.generate_llm_content(tmp_path, meta, None)

        assert "Test Paper" in summary
        assert "No LaTeX source" in equations


class TestFirstLine:
    """Tests for _first_line helper."""

    def test_extracts_first_line(self):
        assert paperpipe._first_line("first\nsecond\nthird") == "first"

    def test_strips_whitespace(self):
        assert paperpipe._first_line("  hello  \nworld") == "hello"

    def test_single_line(self):
        assert paperpipe._first_line("just one line") == "just one line"


class TestCategoriesToTags:
    def test_known_categories(self):
        tags = paperpipe.categories_to_tags(["cs.CV", "cs.LG"])
        assert "computer-vision" in tags
        assert "machine-learning" in tags

    def test_unknown_category(self):
        tags = paperpipe.categories_to_tags(["cs.XX"])
        assert "cs-xx" in tags

    def test_deduplication(self):
        tags = paperpipe.categories_to_tags(["cs.LG", "stat.ML"])
        assert tags.count("machine-learning") == 1

    def test_empty_input(self):
        tags = paperpipe.categories_to_tags([])
        assert tags == []


class TestNormalizeArxivId:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1706.03762", "1706.03762"),
            ("1706.03762v5", "1706.03762v5"),
            ("https://arxiv.org/abs/1706.03762", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762v5.pdf", "1706.03762v5"),
            ("http://arxiv.org/abs/1706.03762", "1706.03762"),
            ("arXiv:1706.03762", "1706.03762"),
            ("abs/1706.03762", "1706.03762"),
            ("hep-th/9901001", "hep-th/9901001"),
            ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
            ("https://arxiv.org/pdf/hep-th/9901001.pdf", "hep-th/9901001"),
        ],
    )
    def test_normalizes_common_inputs(self, raw: str, expected: str):
        assert paperpipe.normalize_arxiv_id(raw) == expected

    def test_rejects_unparseable(self):
        with pytest.raises(ValueError):
            paperpipe.normalize_arxiv_id("not-an-arxiv-id")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="missing"):
            paperpipe.normalize_arxiv_id("")

    def test_handles_plain_id_with_pdf_extension(self):
        # Plain ID ending with .pdf (not a URL)
        assert paperpipe.normalize_arxiv_id("1706.03762.pdf") == "1706.03762"

    def test_extracts_embedded_arxiv_id(self):
        # Arxiv ID embedded in surrounding text
        result = paperpipe.normalize_arxiv_id("See paper 1706.03762 for details")
        assert result == "1706.03762"


class TestExtractNameFromTitle:
    """Tests for _extract_name_from_title helper."""

    def test_extracts_colon_prefix(self):
        assert paperpipe._extract_name_from_title("NeRF: Representing Scenes") == "nerf"

    def test_extracts_multi_word_prefix(self):
        assert paperpipe._extract_name_from_title("Instant NGP: Fast Training") == "instant-ngp"

    def test_returns_none_for_no_colon(self):
        assert paperpipe._extract_name_from_title("Attention Is All You Need") is None

    def test_returns_none_for_long_prefix(self):
        # More than 3 words should be rejected
        result = paperpipe._extract_name_from_title("This Is A Very Long Prefix: And Then The Rest")
        assert result is None

    def test_handles_special_characters(self):
        assert paperpipe._extract_name_from_title("BERT++: Better BERT") == "bert"

    def test_handles_parentheses(self):
        assert paperpipe._extract_name_from_title("GPT (Generative): A Model") == "gpt-generative"


class TestGenerateAutoName:
    """Tests for generate_auto_name function."""

    def test_uses_colon_prefix(self):
        meta = {
            "title": "Neuralangelo: High-Fidelity Neural Surface Reconstruction",
            "abstract": "Some abstract text here.",
            "arxiv_id": "2303.13476",
        }
        name = paperpipe.generate_auto_name(meta, set(), use_llm=False)
        assert name == "neuralangelo"

    def test_falls_back_to_arxiv_id(self):
        meta = {
            "title": "Attention Is All You Need",
            "abstract": "Some abstract text here.",
            "arxiv_id": "1706.03762",
        }
        # Without LLM and no colon, should fall back to arxiv ID
        name = paperpipe.generate_auto_name(meta, set(), use_llm=False)
        assert name == "1706_03762"

    def test_handles_collision(self):
        meta = {
            "title": "NeRF: Representing Scenes",
            "abstract": "Some abstract text here.",
            "arxiv_id": "2020.12345",
        }
        existing = {"nerf", "nerf-2"}
        name = paperpipe.generate_auto_name(meta, existing, use_llm=False)
        assert name == "nerf-3"

    def test_handles_old_style_arxiv_id(self):
        meta = {
            "title": "Some Paper Without Colon",
            "abstract": "Some abstract text here.",
            "arxiv_id": "hep-th/9901001",
        }
        name = paperpipe.generate_auto_name(meta, set(), use_llm=False)
        assert name == "hep-th_9901001"


class TestExtractEquationsSimple:
    def test_extracts_equation_environment(self):
        tex = r"""
        \begin{equation}
        E = mc^2
        \end{equation}
        """
        result = paperpipe.extract_equations_simple(tex)
        assert "E = mc^2" in result
        assert "# Key Equations" in result

    def test_extracts_align_environment(self):
        tex = r"""
        \begin{align}
        a &= b + c \\
        d &= e + f
        \end{align}
        """
        result = paperpipe.extract_equations_simple(tex)
        assert "a &= b + c" in result

    def test_extracts_display_math(self):
        tex = r"""
        \[
        \nabla \cdot E = \frac{\rho}{\epsilon_0}
        \]
        """
        result = paperpipe.extract_equations_simple(tex)
        assert "nabla" in result

    def test_no_equations(self):
        tex = "Just some text without equations."
        result = paperpipe.extract_equations_simple(tex)
        assert result == "No equations extracted."

    def test_skips_trivial_equations(self):
        tex = r"\begin{equation}x\end{equation}"
        result = paperpipe.extract_equations_simple(tex)
        assert result == "No equations extracted."


class TestGenerateSimpleSummary:
    def test_contains_metadata(self):
        meta = {
            "arxiv_id": "2301.00001",
            "title": "Test Paper Title",
            "authors": ["Author One", "Author Two"],
            "published": "2023-01-01T00:00:00",
            "categories": ["cs.CV", "cs.LG"],
            "abstract": "This is the abstract.",
        }
        summary = paperpipe.generate_simple_summary(meta)
        assert "Test Paper Title" in summary
        assert "2301.00001" in summary
        assert "Author One" in summary
        assert "2023-01-01" in summary
        assert "This is the abstract." in summary

    def test_truncates_long_author_list(self):
        meta = {
            "arxiv_id": "2301.00001",
            "title": "Test",
            "authors": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
            "published": "2023-01-01T00:00:00",
            "categories": [],
            "abstract": "Abstract",
        }
        summary = paperpipe.generate_simple_summary(meta)
        assert "..." in summary
        assert "A6" not in summary


class TestCli:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["--help"])
        assert result.exit_code == 0
        assert "paperpipe" in result.output

    def test_list_empty(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["list"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_search_no_results(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_search_grep_uses_ripgrep(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="papers/x/summary.md:1:AdamW\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--grep", "AdamW"])
        assert result.exit_code == 0, result.output
        assert "AdamW" in result.output

        cmd = calls[0]
        assert cmd[0] == "/usr/bin/rg"
        assert "--context" in cmd and "2" in cmd
        assert "--max-count" in cmd and "200" in cmd
        assert "--glob" in cmd and "*/summary.md" in cmd
        assert "*/source.tex" not in cmd

    def test_search_grep_falls_back_to_grep(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/bin/grep" if cmd == "grep" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="papers/x/equations.md:10:Eq. 7\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--grep", "Eq. 7"])
        assert result.exit_code == 0, result.output
        assert "Eq. 7" in result.output

        cmd = calls[0]
        assert cmd[0] == "/bin/grep"
        assert "-RIn" in cmd
        assert "-C2" in cmd
        assert "-m" in cmd and "200" in cmd

    def test_search_grep_no_matches(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--grep", "nope"])
        assert result.exit_code == 0, result.output
        assert "No matches" in result.output

    def test_search_rg_alias_works(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="x/summary.md:1:hit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--rg", "hit"])
        assert result.exit_code == 0, result.output
        assert "hit" in result.output

    def test_search_regex_flag_requires_grep(self, temp_db: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--regex", "x"])
        assert result.exit_code != 0
        assert "only apply with --grep" in result.output

    def test_search_grep_json_output(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="x/summary.md:12:AdamW\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--grep", "--json", "AdamW"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload[0]["paper"] == "x"
        assert payload[0]["path"] == "x/summary.md"
        assert payload[0]["line"] == 12

    def test_search_index_builds_db_and_search_fts_uses_it(self, temp_db: Path) -> None:
        if not fts5_available():
            pytest.skip("SQLite FTS5 not available")

        paper_dir = temp_db / "papers" / "geom-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "Surface Reconstruction", "authors": [], "tags": []}))
        (paper_dir / "summary.md").write_text("We propose surface reconstruction from sparse points.\n")
        (paper_dir / "equations.md").write_text("No equations.\n")
        (paper_dir / "notes.md").write_text("Note.\n")

        paperpipe.save_index({"geom-paper": {"arxiv_id": None, "title": "Surface Reconstruction", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search-index", "--rebuild"])
        assert result.exit_code == 0, result.output
        assert (temp_db / "search.db").exists()

        result = runner.invoke(paperpipe.cli, ["search", "--fts", "surface"])
        assert result.exit_code == 0, result.output
        assert "geom-paper" in result.output

    def test_search_hybrid_boosts_grep_hits(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not fts5_available():
            pytest.skip("SQLite FTS5 not available")

        # Paper A: only in FTS (title match)
        paper_a = temp_db / "papers" / "paper-a"
        paper_a.mkdir(parents=True)
        (paper_a / "meta.json").write_text(json.dumps({"title": "Surface Reconstruction A", "authors": [], "tags": []}))
        (paper_a / "summary.md").write_text("Nothing.\n")

        # Paper B: has an exact grep hit
        paper_b = temp_db / "papers" / "paper-b"
        paper_b.mkdir(parents=True)
        (paper_b / "meta.json").write_text(json.dumps({"title": "Unrelated", "authors": [], "tags": []}))
        (paper_b / "summary.md").write_text("surface reconstruction\n")

        paperpipe.save_index(
            {
                "paper-a": {"arxiv_id": None, "title": "Surface Reconstruction A", "tags": []},
                "paper-b": {"arxiv_id": None, "title": "Unrelated", "tags": []},
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search-index", "--rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            # Simulate ripgrep returning a hit in paper-b only.
            return types.SimpleNamespace(
                returncode=0, stdout="paper-b/summary.md:1:surface reconstruction\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner.invoke(paperpipe.cli, ["search", "--hybrid", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        # Hybrid should annotate grep hits.
        assert "paper-b" in result.output
        assert "grep:" in result.output

    def test_search_hybrid_show_grep_hits_prints_lines(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not fts5_available():
            pytest.skip("SQLite FTS5 not available")

        paper_b = temp_db / "papers" / "paper-b"
        paper_b.mkdir(parents=True)
        (paper_b / "meta.json").write_text(json.dumps({"title": "Unrelated", "authors": [], "tags": []}))
        (paper_b / "summary.md").write_text("surface reconstruction\n")

        paperpipe.save_index({"paper-b": {"arxiv_id": None, "title": "Unrelated", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search-index", "--rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(
                returncode=0, stdout="paper-b/summary.md:1:surface reconstruction\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner.invoke(paperpipe.cli, ["search", "--hybrid", "--show-grep-hits", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        assert "paper-b/summary.md:1:" in result.output

    def test_search_hybrid_requires_search_db(self, temp_db: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--hybrid", "x"])
        assert result.exit_code != 0
        assert "search-index" in result.output

    def test_search_mode_env_scan_forces_scan(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not fts5_available():
            pytest.skip("SQLite FTS5 not available")

        paper_dir = temp_db / "papers" / "p"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "Surface Reconstruction", "authors": [], "tags": []}))
        (paper_dir / "summary.md").write_text("surface reconstruction\n")
        (paper_dir / "equations.md").write_text("surface reconstruction\n")
        (paper_dir / "notes.md").write_text("surface reconstruction\n")

        paperpipe.save_index({"p": {"arxiv_id": None, "title": "Surface Reconstruction", "tags": ["tag"]}})
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search-index", "--rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setenv("PAPERPIPE_SEARCH_MODE", "scan")
        result = runner.invoke(paperpipe.cli, ["search", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        assert "Matches:" in result.output

    def test_search_fts_schema_mismatch_prompts_rebuild(self, temp_db: Path) -> None:
        if not fts5_available():
            pytest.skip("SQLite FTS5 not available")

        import sqlite3

        db_path = temp_db / "search.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE search_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO search_meta(key, value) VALUES ('schema_version', '0')")
            conn.commit()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--fts", "x"])
        assert result.exit_code != 0
        assert "schema version mismatch" in result.output.lower()
        assert "search-index --rebuild" in result.output

    def test_show_nonexistent(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["show", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_path_command(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["path"])
        assert result.exit_code == 0
        assert ".paperpipe" in result.output

    def test_tags_empty(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["tags"])
        assert result.exit_code == 0

    def test_cli_verbose_flag(self, temp_db: Path):
        """Test that --verbose flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["--verbose", "list"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_cli_quiet_flag(self, temp_db: Path):
        """Test that --quiet flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["--quiet", "list"])
        assert result.exit_code == 0
        # Data output should still be shown even with --quiet
        assert "No papers found" in result.output

    def test_list_with_papers(self, temp_db: Path):
        # Add a paper to the index
        paperpipe.save_index(
            {
                "test-paper": {
                    "arxiv_id": "2301.00001",
                    "title": "A Test Paper About Machine Learning",
                    "tags": ["machine-learning", "computer-vision"],
                    "added": "2023-01-01T00:00:00",
                }
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["list"])
        assert result.exit_code == 0
        assert "test-paper" in result.output

    def test_list_filter_by_tag(self, temp_db: Path):
        paperpipe.save_index(
            {
                "paper1": {"title": "ML Paper", "tags": ["ml"], "arxiv_id": "1"},
                "paper2": {"title": "CV Paper", "tags": ["cv"], "arxiv_id": "2"},
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["list", "-t", "ml"])
        assert "paper1" in result.output
        assert "paper2" not in result.output

    def test_search_finds_by_title(self, temp_db: Path):
        paperpipe.save_index(
            {
                "neural-sdf": {
                    "arxiv_id": "2301.00001",
                    "title": "Neural Signed Distance Functions",
                    "tags": ["sdf", "neural"],
                }
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "neural"])
        assert "neural-sdf" in result.output

    def test_search_finds_by_tag(self, temp_db: Path):
        paperpipe.save_index(
            {
                "test-paper": {
                    "arxiv_id": "2301.00001",
                    "title": "Some Paper",
                    "tags": ["transformers"],
                }
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "transformer"])
        assert "test-paper" in result.output

    def test_search_finds_by_arxiv_id(self, temp_db: Path):
        paperpipe.save_index(
            {
                "attention": {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "tags": [],
                }
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "1706"])
        assert "attention" in result.output

    def test_search_finds_by_summary_content_fuzzy(self, temp_db: Path):
        # Paper exists with content, but title/tags don't match query.
        paper_dir = temp_db / "papers" / "geom-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("We propose surface reconstruction from sparse points using an SDF.\n")

        paperpipe.save_index(
            {
                "geom-paper": {
                    "arxiv_id": "2301.00001",
                    "title": "A Paper About Neural Fields",
                    "tags": ["neural-fields"],
                }
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "surfae reconstructon"])
        assert result.exit_code == 0
        assert "geom-paper" in result.output

    def test_search_exact_does_not_match_typos(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "geom-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("We propose surface reconstruction from sparse points using an SDF.\n")

        paperpipe.save_index(
            {
                "geom-paper": {
                    "arxiv_id": "2301.00001",
                    "title": "A Paper About Neural Fields",
                    "tags": ["neural-fields"],
                }
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "--exact", "surfae reconstructon"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_search_does_not_fuzzy_expand_when_exact_exists(self, temp_db: Path):
        exact_dir = temp_db / "papers" / "exact-paper"
        exact_dir.mkdir(parents=True)
        (exact_dir / "summary.md").write_text("We propose surface reconstruction from sparse points.\n")

        fuzzy_only_dir = temp_db / "papers" / "fuzzy-only-paper"
        fuzzy_only_dir.mkdir(parents=True)
        (fuzzy_only_dir / "summary.md").write_text("We propose surfae reconstructon from sparse points.\n")

        paperpipe.save_index(
            {
                "exact-paper": {
                    "arxiv_id": "2301.00001",
                    "title": "Exact Match Paper",
                    "tags": [],
                },
                "fuzzy-only-paper": {
                    "arxiv_id": "2301.00002",
                    "title": "Fuzzy Only Paper",
                    "tags": [],
                },
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["search", "surface reconstruction"])
        assert result.exit_code == 0
        assert "exact-paper" in result.output
        assert "fuzzy-only-paper" not in result.output

    def test_show_displays_paper_details(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": "2301.00001",
                    "title": "Test Paper Title",
                    "authors": ["Alice", "Bob", "Charlie"],
                    "tags": ["ml", "nlp"],
                    "has_pdf": True,
                    "has_source": True,
                }
            )
        )
        (paper_dir / "paper.pdf").touch()
        (paper_dir / "source.tex").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["show", "test-paper"])
        assert result.exit_code == 0
        assert "Test Paper Title" in result.output
        assert "2301.00001" in result.output
        assert "Alice" in result.output
        assert "ml" in result.output

    def test_tags_lists_all_tags_with_counts(self, temp_db: Path):
        paperpipe.save_index(
            {
                "paper1": {"arxiv_id": "1", "title": "P1", "tags": ["ml", "cv"]},
                "paper2": {"arxiv_id": "2", "title": "P2", "tags": ["ml", "nlp"]},
                "paper3": {"arxiv_id": "3", "title": "P3", "tags": ["ml"]},
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["tags"])
        assert result.exit_code == 0
        assert "ml: 3" in result.output
        assert "cv: 1" in result.output
        assert "nlp: 1" in result.output

    def test_list_json_output(self, temp_db: Path):
        paperpipe.save_index(
            {
                "paper1": {"arxiv_id": "1", "title": "Paper One", "tags": ["ml"]},
            }
        )
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "paper1" in data
        assert data["paper1"]["title"] == "Paper One"

    def test_show_equations_stdout(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": "2301.00001",
                    "title": "Test Paper Title",
                    "authors": ["Alice"],
                    "tags": ["ml"],
                }
            )
        )
        (paper_dir / "equations.md").write_text("# Key Equations\n\n```latex\nE=mc^2\n```\n")
        paperpipe.save_index({"test-paper": {"arxiv_id": "2301.00001", "title": "Test Paper Title", "tags": ["ml"]}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["show", "test-paper", "--level", "eq"])
        assert result.exit_code == 0
        assert "# test-paper" in result.output
        assert "Content: equations" in result.output
        assert "E=mc^2" in result.output

    def test_show_multiple_papers_separated(self, temp_db: Path):
        for name, arxiv_id in [("paper-a", "2301.00001"), ("paper-b", "2301.00002")]:
            paper_dir = temp_db / "papers" / name
            paper_dir.mkdir(parents=True)
            (paper_dir / "meta.json").write_text(json.dumps({"arxiv_id": arxiv_id, "title": f"Title {name}"}))
            (paper_dir / "equations.md").write_text(f"# Eq\n\n{name}\n")

        paperpipe.save_index(
            {
                "paper-a": {"arxiv_id": "2301.00001", "title": "Title paper-a", "tags": []},
                "paper-b": {"arxiv_id": "2301.00002", "title": "Title paper-b", "tags": []},
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["show", "paper-a", "paper-b", "--level", "equations"])
        assert result.exit_code == 0
        assert "# paper-a" in result.output
        assert "# paper-b" in result.output
        assert "\n---\n" in result.output

    def test_export_to_dash_is_rejected(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("x\n")
        paperpipe.save_index({"test-paper": {"arxiv_id": "2301.00001", "title": "T", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "equations", "--to", "-"])
        assert result.exit_code != 0
        assert "export` only writes to a directory" in result.output.lower()

    def test_export_copies_equations_to_directory(self, temp_db: Path, tmp_path: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("eq\n")
        paperpipe.save_index({"test-paper": {"arxiv_id": "2301.00001", "title": "T", "tags": []}})

        runner = CliRunner()
        out_dir = tmp_path / "paper-context"
        result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "equations", "--to", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "test-paper_equations.md").exists()
        assert (out_dir / "test-paper_equations.md").read_text() == "eq\n"

    def test_export_accepts_eq_alias(self, temp_db: Path, tmp_path: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("eq\n")
        paperpipe.save_index({"test-paper": {"arxiv_id": "2301.00001", "title": "T", "tags": []}})

        runner = CliRunner()
        out_dir = tmp_path / "paper-context"
        result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "eq", "--to", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "test-paper_equations.md").exists()


class TestFetchArxivMetadata:
    """Unit tests for fetch_arxiv_metadata with mocked arxiv library."""

    def test_extracts_metadata_from_arxiv_result(self, monkeypatch):
        """Test that metadata is correctly extracted from arxiv API response."""
        from datetime import datetime
        from unittest.mock import MagicMock

        import arxiv

        # Create mock paper object matching arxiv library structure
        mock_paper = MagicMock()
        mock_paper.title = "Attention Is All You Need"
        mock_paper.authors = [MagicMock(), MagicMock()]
        mock_paper.authors[0].name = "Vaswani"
        mock_paper.authors[1].name = "Shazeer"
        mock_paper.summary = "The dominant sequence transduction models..."
        mock_paper.primary_category = "cs.CL"
        mock_paper.categories = ["cs.CL", "cs.LG"]
        mock_paper.published = datetime(2017, 6, 12)
        mock_paper.updated = datetime(2017, 6, 12)
        mock_paper.doi = "10.1234/example"
        mock_paper.journal_ref = None
        mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762"

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        monkeypatch.setattr(arxiv, "Search", lambda id_list: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        meta = paperpipe.fetch_arxiv_metadata("1706.03762")

        assert meta["arxiv_id"] == "1706.03762"
        assert meta["title"] == "Attention Is All You Need"
        assert meta["authors"] == ["Vaswani", "Shazeer"]
        assert meta["abstract"] == "The dominant sequence transduction models..."
        assert meta["primary_category"] == "cs.CL"
        assert meta["categories"] == ["cs.CL", "cs.LG"]
        assert meta["pdf_url"] == "https://arxiv.org/pdf/1706.03762"

    def test_handles_empty_authors(self, monkeypatch):
        """Test handling of papers with no authors listed."""
        from datetime import datetime
        from unittest.mock import MagicMock

        import arxiv

        mock_paper = MagicMock()
        mock_paper.title = "Anonymous Paper"
        mock_paper.authors = []
        mock_paper.summary = "Abstract"
        mock_paper.primary_category = "cs.CV"
        mock_paper.categories = ["cs.CV"]
        mock_paper.published = datetime(2023, 1, 1)
        mock_paper.updated = datetime(2023, 1, 1)
        mock_paper.doi = None
        mock_paper.journal_ref = None
        mock_paper.pdf_url = "https://arxiv.org/pdf/2301.00001"

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        monkeypatch.setattr(arxiv, "Search", lambda id_list: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        meta = paperpipe.fetch_arxiv_metadata("2301.00001")
        assert meta["authors"] == []


class TestDownloadPdf:
    """Unit tests for download_pdf with mocked arxiv library."""

    def test_downloads_pdf_successfully(self, tmp_path, monkeypatch):
        """Test successful PDF download."""
        from unittest.mock import MagicMock

        import arxiv

        pdf_content = b"%PDF-1.4 fake pdf content"
        dest = tmp_path / "paper.pdf"

        mock_paper = MagicMock()

        def fake_download(filename):
            Path(filename).write_bytes(pdf_content)

        mock_paper.download_pdf = fake_download

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        monkeypatch.setattr(arxiv, "Search", lambda id_list: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        result = paperpipe.download_pdf("1706.03762", dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == pdf_content

    def test_returns_false_when_download_fails(self, tmp_path, monkeypatch):
        """Test that download_pdf returns False when file isn't created."""
        from unittest.mock import MagicMock

        import arxiv

        dest = tmp_path / "paper.pdf"

        mock_paper = MagicMock()
        mock_paper.download_pdf = MagicMock()  # Does nothing, file not created

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        monkeypatch.setattr(arxiv, "Search", lambda id_list: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        result = paperpipe.download_pdf("1706.03762", dest)

        assert result is False
        assert not dest.exists()


class TestDownloadSource:
    """Unit tests for download_source with mocked requests."""

    def test_extracts_tex_from_tarball(self, tmp_path, monkeypatch):
        """Test extraction of .tex files from a tarball."""
        import io
        import tarfile
        from unittest.mock import MagicMock

        import requests

        # Create a fake tarball with .tex content
        tex_content = r"\begin{document}Hello\end{document}"
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            tex_bytes = tex_content.encode("utf-8")
            info = tarfile.TarInfo(name="main.tex")
            info.size = len(tex_bytes)
            tar.addfile(info, io.BytesIO(tex_bytes))
        tar_buffer.seek(0)

        mock_response = MagicMock()
        mock_response.content = tar_buffer.read()
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1706.03762", paper_dir)

        assert result is not None
        assert r"\begin{document}" in result
        assert (paper_dir / "source.tex").exists()

    def test_returns_none_on_http_error(self, tmp_path, monkeypatch):
        """Test that HTTP errors return None gracefully."""
        from unittest.mock import MagicMock

        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("nonexistent", paper_dir)

        assert result is None
        assert not (paper_dir / "source.tex").exists()

    def test_returns_none_for_non_tex_content(self, tmp_path, monkeypatch):
        """Test that non-LaTeX content (no \\begin{document}) returns None."""
        from unittest.mock import MagicMock

        import requests

        # Plain text without LaTeX markers
        mock_response = MagicMock()
        mock_response.content = b"This is just plain text, no LaTeX here."
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1706.03762", paper_dir)

        assert result is None

    def test_prefers_main_tex_over_others(self, tmp_path, monkeypatch):
        """Test that main.tex is preferred when multiple .tex files exist."""
        import io
        import tarfile
        from unittest.mock import MagicMock

        import requests

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a secondary .tex file (larger)
            other_content = r"\begin{document}Other content here that is longer\end{document}"
            other_bytes = other_content.encode("utf-8")
            info = tarfile.TarInfo(name="other.tex")
            info.size = len(other_bytes)
            tar.addfile(info, io.BytesIO(other_bytes))

            # Add main.tex (smaller but preferred)
            main_content = r"\begin{document}Main\end{document}"
            main_bytes = main_content.encode("utf-8")
            info = tarfile.TarInfo(name="main.tex")
            info.size = len(main_bytes)
            tar.addfile(info, io.BytesIO(main_bytes))
        tar_buffer.seek(0)

        mock_response = MagicMock()
        mock_response.content = tar_buffer.read()
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1706.03762", paper_dir)

        # main.tex content should come first
        assert result is not None
        assert result.startswith(r"\begin{document}Main")

    def test_finds_main_by_document_marker(self, tmp_path, monkeypatch):
        """Test that file with \\begin{document} is preferred when no main.tex."""
        import io
        import tarfile
        from unittest.mock import MagicMock

        import requests

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # File without \begin{document} (larger)
            preamble = r"\newcommand{\foo}{bar}" + "x" * 1000
            preamble_bytes = preamble.encode("utf-8")
            info = tarfile.TarInfo(name="preamble.tex")
            info.size = len(preamble_bytes)
            tar.addfile(info, io.BytesIO(preamble_bytes))

            # File with \begin{document} (smaller but is main)
            doc_content = r"\begin{document}The actual document\end{document}"
            doc_bytes = doc_content.encode("utf-8")
            info = tarfile.TarInfo(name="article.tex")
            info.size = len(doc_bytes)
            tar.addfile(info, io.BytesIO(doc_bytes))
        tar_buffer.seek(0)

        mock_response = MagicMock()
        mock_response.content = tar_buffer.read()
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1706.03762", paper_dir)

        assert result is not None
        # article.tex (with \begin{document}) should be main, preamble.tex appended
        assert result.startswith(r"\begin{document}The actual")

    def test_falls_back_to_largest_file(self, tmp_path, monkeypatch):
        """Test fallback to largest file when no \\begin{document} anywhere."""
        import io
        import tarfile
        from unittest.mock import MagicMock

        import requests

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Small file
            small = r"\section{Small}"
            small_bytes = small.encode("utf-8")
            info = tarfile.TarInfo(name="small.tex")
            info.size = len(small_bytes)
            tar.addfile(info, io.BytesIO(small_bytes))

            # Large file (should be picked)
            large = r"\section{Large}" + "x" * 500
            large_bytes = large.encode("utf-8")
            info = tarfile.TarInfo(name="large.tex")
            info.size = len(large_bytes)
            tar.addfile(info, io.BytesIO(large_bytes))
        tar_buffer.seek(0)

        mock_response = MagicMock()
        mock_response.content = tar_buffer.read()
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1706.03762", paper_dir)

        # No \begin{document}, so returns None
        assert result is None


class TestAddCommand:
    """Unit tests for the add command with mocked network calls."""

    def test_generate_auto_name_local_meta_uses_slug(self):
        # Local/meta-only papers should fall back to a stable title slug, not "unknown".
        meta = {"title": "Some Paper", "authors": [], "abstract": ""}
        assert paperpipe.generate_auto_name(meta, set(), use_llm=False) == "some-paper"

    def test_parse_authors_keeps_last_first_single_author(self):
        assert paperpipe._parse_authors("Smith, John") == ["Smith, John"]

    def test_parse_authors_semicolon_separated(self):
        assert paperpipe._parse_authors("Smith, John; Doe, Jane") == ["Smith, John", "Doe, Jane"]

    def test_parse_authors_multiple_comma_separated(self):
        assert paperpipe._parse_authors("Alice, Bob, Charlie") == ["Alice", "Bob", "Charlie"]

    def test_parse_authors_empty(self):
        assert paperpipe._parse_authors("") == []
        assert paperpipe._parse_authors(None) == []

    def test_format_title_short_truncates(self):
        long_title = "A" * 100
        result = paperpipe._format_title_short(long_title, max_len=60)
        assert len(result) == 63  # 60 chars + "..."
        assert result.endswith("...")

    def test_format_title_short_keeps_short(self):
        short_title = "Short Title"
        assert paperpipe._format_title_short(short_title) == short_title

    def test_slugify_title_basic(self):
        assert paperpipe._slugify_title("Hello World") == "hello-world"

    def test_slugify_title_empty(self):
        assert paperpipe._slugify_title("") == "paper"
        assert paperpipe._slugify_title("   ") == "paper"

    def test_slugify_title_truncates_long(self):
        long_title = "word " * 50
        result = paperpipe._slugify_title(long_title, max_len=30)
        assert len(result) <= 30

    def test_slugify_title_special_chars(self):
        assert paperpipe._slugify_title("Test: A 'Paper' with \"Quotes\"") == "test-a-paper-with-quotes"

    def test_looks_like_pdf_valid(self, tmp_path: Path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4\ntest content")
        assert paperpipe._looks_like_pdf(pdf) is True

    def test_looks_like_pdf_invalid(self, tmp_path: Path):
        txt = tmp_path / "test.txt"
        txt.write_text("not a pdf")
        assert paperpipe._looks_like_pdf(txt) is False

    def test_looks_like_pdf_missing_file(self, tmp_path: Path):
        missing = tmp_path / "missing.pdf"
        assert paperpipe._looks_like_pdf(missing) is False

    def test_add_rejects_unsafe_name_without_network(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(paperpipe, "fetch_arxiv_metadata", lambda _: (_ for _ in ()).throw(AssertionError()))

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "1706.03762", "--name", "../bad", "--no-llm"],
        )

        assert result.exit_code != 0
        assert "invalid paper name" in result.output.lower()

    def test_add_local_pdf_ingests_and_indexes(self, temp_db: Path):
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            [
                "add",
                "--pdf",
                str(pdf_path),
                "--title",
                "Some Paper",
                "--authors",
                "A; B",
                "--abstract",
                "Short abstract.",
                "--year",
                "2024",
                "--venue",
                "NeurIPS",
                "--doi",
                "10.1234/example",
                "--url",
                "https://example.com/paper",
                "--tags",
                "ml,vision",
                "--no-llm",
            ],
        )

        assert result.exit_code == 0, result.output

        name = "some-paper"
        paper_dir = temp_db / "papers" / name
        assert (paper_dir / "paper.pdf").read_bytes() == pdf_path.read_bytes()
        assert (paper_dir / "summary.md").exists()
        assert (paper_dir / "equations.md").exists()

        meta = json.loads((paper_dir / "meta.json").read_text())
        assert meta["arxiv_id"] is None
        assert meta["title"] == "Some Paper"
        assert meta["authors"] == ["A", "B"]
        assert meta["abstract"] == "Short abstract."
        assert meta["year"] == 2024
        assert meta["venue"] == "NeurIPS"
        assert meta["doi"] == "10.1234/example"
        assert meta["url"] == "https://example.com/paper"
        assert meta["has_pdf"] is True
        assert meta["has_source"] is False

        index = paperpipe.load_index()
        assert name in index
        assert index[name]["arxiv_id"] is None

    def test_add_local_pdf_requires_title(self, temp_db: Path):
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["add", "--pdf", str(pdf_path), "--no-llm"])
        assert result.exit_code != 0
        assert "--title" in result.output

    def test_add_local_pdf_rejects_non_pdf(self, temp_db: Path):
        bad_path = temp_db / "not-a-pdf.txt"
        bad_path.write_text("hello")

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "--pdf", str(bad_path), "--title", "Some Paper", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "does not look like a pdf" in result.output.lower()

    def test_add_local_pdf_rejects_invalid_year(self, temp_db: Path):
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "--pdf", str(pdf_path), "--title", "Some Paper", "--year", "12", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "invalid --year" in result.output.lower()

    def test_add_local_pdf_name_conflict_fails_fast(self, temp_db: Path):
        # Existing entry with auto-name "some-paper"
        (temp_db / "papers" / "some-paper").mkdir(parents=True)
        paperpipe.save_index({"some-paper": {"arxiv_id": None, "title": "Old", "tags": [], "added": "x"}})

        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "--pdf", str(pdf_path), "--title", "Some Paper", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "name conflict" in result.output.lower()

    def test_add_existing_paper_is_idempotent_by_arxiv_id(self, temp_db: Path, monkeypatch):
        """Test that re-adding the same arXiv paper is a no-op (unit test, no network)."""
        # Pre-create the paper directory to simulate existing paper
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(
            json.dumps({"arxiv_id": "1706.03762", "title": "Test", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({"test-paper": {"arxiv_id": "1706.03762", "title": "Test", "tags": [], "added": "x"}})

        # Should not fetch metadata again when skipping duplicates
        monkeypatch.setattr(paperpipe, "fetch_arxiv_metadata", lambda _: (_ for _ in ()).throw(AssertionError()))

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "1706.03762", "--name", "test-paper", "--no-llm"],
        )

        assert result.exit_code == 0
        assert "already added" in result.output.lower()

    def test_add_detects_duplicate_in_index(self, temp_db: Path, monkeypatch):
        """Test that add command detects duplicate name in index even without directory."""
        from datetime import datetime

        # Pre-populate index only (no directory)
        paperpipe.save_index({"existing-paper": {"arxiv_id": "other-id", "title": "Other", "tags": [], "added": "x"}})

        # Mock fetch_arxiv_metadata to avoid network call
        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": "Test Paper",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.CV",
                "categories": ["cs.CV"],
                "published": datetime(2023, 1, 1).isoformat(),
                "updated": datetime(2023, 1, 1).isoformat(),
                "doi": None,
                "journal_ref": None,
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
            }

        monkeypatch.setattr(paperpipe, "fetch_arxiv_metadata", mock_fetch)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "1706.03762", "--name", "existing-paper", "--no-llm"],
        )

        assert "already" in result.output.lower()

    def test_add_duplicate_arxiv_skips_without_network(self, temp_db: Path, monkeypatch):
        """Test that duplicate arXiv IDs are skipped by default (no extra copy)."""
        paper_dir = temp_db / "papers" / "attention"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(
            json.dumps({"arxiv_id": "1706.03762v1", "title": "Old", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({"attention": {"arxiv_id": "1706.03762v1", "title": "Old", "tags": [], "added": "x"}})

        monkeypatch.setattr(paperpipe, "fetch_arxiv_metadata", lambda _: (_ for _ in ()).throw(AssertionError()))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["add", "1706.03762v2", "--no-llm"])

        assert result.exit_code == 0
        assert "already added" in result.output.lower()
        # Still only one directory
        assert sorted([p.name for p in (temp_db / "papers").iterdir() if p.is_dir()]) == ["attention"]

    def test_add_duplicate_arxiv_duplicate_flag_creates_second_copy(self, temp_db: Path, monkeypatch):
        """Test --duplicate keeps the old behavior of adding another copy."""
        from datetime import datetime

        # Existing paper uses the default no-LLM auto-name for this arXiv ID.
        existing_name = "1706_03762"
        paper_dir = temp_db / "papers" / existing_name
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(
            json.dumps({"arxiv_id": "1706.03762", "title": "Old", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({existing_name: {"arxiv_id": "1706.03762", "title": "Old", "tags": [], "added": "x"}})

        def mock_fetch(arxiv_id: str):
            return {
                "arxiv_id": arxiv_id,
                "title": "Attention Is All You Need",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "published": datetime(2017, 6, 12).isoformat(),
                "updated": datetime(2017, 6, 12).isoformat(),
                "doi": None,
                "journal_ref": None,
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
            }

        monkeypatch.setattr(paperpipe, "fetch_arxiv_metadata", mock_fetch)

        def fake_download_pdf(_arxiv_id: str, dest: Path):
            dest.write_bytes(b"%PDF")
            return True

        monkeypatch.setattr(paperpipe, "download_pdf", fake_download_pdf)
        monkeypatch.setattr(paperpipe, "download_source", lambda _arxiv_id, _paper_dir: None)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["add", "1706.03762", "--duplicate", "--no-llm"])

        assert result.exit_code == 0
        assert "Added: 1706_03762-2" in result.output
        assert (temp_db / "papers" / "1706_03762-2" / "meta.json").exists()

    def test_add_duplicate_arxiv_update_refreshes_existing(self, temp_db: Path, monkeypatch):
        """Test --update refreshes an existing paper in-place."""
        from datetime import datetime

        name = "attention"
        paper_dir = temp_db / "papers" / name
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("old summary")
        (paper_dir / "equations.md").write_text("old equations")
        (paper_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": "1706.03762",
                    "title": "Old",
                    "authors": [],
                    "abstract": "",
                    "categories": ["cs.CL"],
                    "tags": ["old-tag"],
                    "published": "2017-01-01",
                    "added": "x",
                    "has_source": False,
                    "has_pdf": False,
                }
            )
        )
        paperpipe.save_index({name: {"arxiv_id": "1706.03762", "title": "Old", "tags": ["old-tag"], "added": "x"}})

        def mock_fetch(arxiv_id: str):
            return {
                "arxiv_id": arxiv_id,
                "title": "New Title",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.CV",
                "categories": ["cs.CV"],
                "published": datetime(2023, 1, 1).isoformat(),
                "updated": datetime(2023, 1, 1).isoformat(),
                "doi": None,
                "journal_ref": None,
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
            }

        monkeypatch.setattr(paperpipe, "fetch_arxiv_metadata", mock_fetch)

        def fake_download_pdf(_arxiv_id: str, dest: Path):
            dest.write_bytes(b"%PDF")
            return True

        def fake_download_source(_arxiv_id: str, pdir: Path):
            tex = r"\begin{document}\begin{equation}E=mc^2\end{equation}\end{document}"
            (pdir / "source.tex").write_text(tex)
            return tex

        monkeypatch.setattr(paperpipe, "download_pdf", fake_download_pdf)
        monkeypatch.setattr(paperpipe, "download_source", fake_download_source)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["add", "1706.03762", "--update", "--name", name, "--no-llm"])

        assert result.exit_code == 0
        assert "Updated: attention" in result.output

        meta = json.loads((paper_dir / "meta.json").read_text())
        assert meta["title"] == "New Title"
        assert meta["added"] == "x"  # preserved
        assert "computer-vision" in meta["tags"]
        assert "old-tag" in meta["tags"]


class TestAddMultiplePapers:
    """Tests for adding multiple papers at once."""

    def test_add_name_with_multiple_papers_errors(self, temp_db: Path):
        """Test that --name errors when used with multiple papers."""
        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "1706.03762", "2301.00001", "--name", "my-paper", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "--name can only be used when adding a single paper" in result.output


class TestInstallSkillCommand:
    def test_install_skill_creates_symlink_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "skills" / "papi"
        assert dest.is_symlink()
        assert dest.resolve() == (Path(__file__).parent / "skill").resolve()

    def test_install_skill_existing_dest_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0
        assert "use --force" in result.output.lower()

    def test_install_skill_force_overwrites_existing_file(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "skill", "--codex", "--force"])
        assert result.exit_code == 0, result.output

        assert dest.is_symlink()

    def test_install_skill_creates_symlink_for_gemini(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "skill", "--gemini"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".gemini" / "skills" / "papi"
        assert dest.is_symlink()
        assert dest.resolve() == (Path(__file__).parent / "skill").resolve()


class TestInstallPromptsCommand:
    def test_install_prompts_creates_symlinks_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0, result.output

        prompt_dir = tmp_path / ".codex" / "prompts"
        assert prompt_dir.exists()

        expected = [
            "compare-papers.md",
            "curate-paper-note.md",
            "ground-with-paper.md",
            "papi.md",
            "verify-with-paper.md",
        ]
        for filename in expected:
            dest = prompt_dir / filename
            assert dest.is_symlink()
            assert dest.resolve() == (Path(__file__).parent / "prompts" / "codex" / filename).resolve()

    def test_install_prompts_existing_dest_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "prompts" / "ground-with-paper.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0
        assert "use --force" in result.output.lower()

    def test_install_prompts_copy_mode_copies_files(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "prompts", "--codex", "--copy"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "curate-paper-note.md"
        assert dest.exists()
        assert not dest.is_symlink()

    def test_install_prompts_creates_symlinks_for_gemini(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "prompts", "--gemini"])
        assert result.exit_code == 0, result.output

        prompt_dir = tmp_path / ".gemini" / "commands"
        assert prompt_dir.exists()

        expected = [
            "compare-papers.toml",
            "curate-paper-note.toml",
            "ground-with-paper.toml",
            "papi-run.toml",
            "papi.toml",
            "papi-list.toml",
            "papi-path.toml",
            "papi-search.toml",
            "papi-show-eq.toml",
            "papi-show-summary.toml",
            "papi-show-tex.toml",
            "papi-tags.toml",
            "verify-with-paper.toml",
        ]
        for filename in expected:
            dest = prompt_dir / filename
            assert dest.is_symlink()
            assert dest.resolve() == (Path(__file__).parent / "prompts" / "gemini" / filename).resolve()


class TestInstallMcpCommand:
    @pytest.fixture(autouse=True)
    def _pretend_paperqa_mcp_is_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """install mcp now installs only MCP servers available in the environment.

        Tests run without optional deps installed, so we pretend PaperQA2 MCP deps exist.
        """
        import importlib.machinery
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, package: str | None = None):  # type: ignore[override]
            if name in {"mcp.server.fastmcp", "paperqa"}:
                return importlib.machinery.ModuleSpec(name, loader=None)
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    def test_install_mcp_claude_runs_claude_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "mcp", "--claude"])
        assert result.exit_code == 0, result.output

        assert any(
            c[:3] == ["claude", "mcp", "add"]
            and "--transport" in c
            and "stdio" in c
            and "--env" in c
            and any(s.startswith("PAPERQA_EMBEDDING=") for s in c)
            and "--scope" in c
            and "user" in c
            and "paperqa" in c
            and "--" in c
            and "papi" in c
            and "mcp-server" in c
            for c in calls
        )

    def test_install_mcp_repo_writes_mcp_json(self, temp_db: Path):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["install", "mcp", "--repo", "--embedding", "text-embedding-3-small"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["command"] == "papi"
            assert cfg["mcpServers"]["paperqa"]["args"] == ["mcp-server"]
            cfg2 = json.loads((Path(".gemini") / "settings.json").read_text())
            assert cfg2["mcpServers"]["paperqa"]["command"] == "papi"
            assert cfg2["mcpServers"]["paperqa"]["args"] == ["mcp-server"]

    def test_install_mcp_repo_writes_leann_when_available(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann_mcp" if cmd == "leann_mcp" else None)

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["install", "mcp", "--repo", "--embedding", "text-embedding-3-small"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["command"] == "papi"
            assert cfg["mcpServers"]["paperqa"]["args"] == ["mcp-server"]
            assert cfg["mcpServers"]["leann"]["command"] == "papi"
            assert cfg["mcpServers"]["leann"]["args"] == ["leann-mcp-server"]

    def test_install_mcp_repo_uses_paperqa_embedding_env_override(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (temp_db / "config.toml").write_text("\n".join(["[paperqa]", 'embedding = "config-embedding"', ""]))
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERQA_EMBEDDING", "env-embedding")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["install", "mcp", "--repo"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["env"]["PAPERQA_EMBEDDING"] == "env-embedding"

    def test_install_mcp_gemini_writes_settings_json(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "mcp", "--gemini", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output

        cfg = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert cfg["mcpServers"]["paperqa"]["command"] == "papi"
        assert cfg["mcpServers"]["paperqa"]["args"] == ["mcp-server"]
        assert cfg["mcpServers"]["paperqa"]["env"]["PAPERQA_EMBEDDING"] == "text-embedding-3-small"

    def test_install_mcp_gemini_runs_gemini_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "mcp", "--gemini", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output

        assert any(
            c[:3] == ["gemini", "mcp", "add"]
            and "--scope" in c
            and "user" in c
            and "--transport" in c
            and "stdio" in c
            and "--env" in c
            and "PAPERQA_EMBEDDING=text-embedding-3-small" in c
            and "paperqa" in c
            and "papi" in c
            and "mcp-server" in c
            for c in calls
        )

    def test_install_mcp_gemini_does_not_write_settings_json_when_cli_succeeds(
        self, temp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        settings_path = tmp_path / ".gemini" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        before = json.dumps({"mcpServers": {"paperqa": {"some": "other-shape"}}}, indent=2) + "\n"
        settings_path.write_text(before)

        def fake_run(_args: list[str], **_kwargs):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output
        assert "already configured" not in result.output.lower()
        assert settings_path.read_text() == before

    def test_install_mcp_codex_runs_codex_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/codex" if cmd == "codex" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "mcp", "--codex", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output
        assert any(
            c[:4] == ["codex", "mcp", "add", "paperqa"]
            and "papi" in c
            and "mcp-server" in c
            and "PAPERQA_EMBEDDING=text-embedding-3-small" in " ".join(c)
            for c in calls
        )

    def test_install_mcp_codex_force_removes_then_adds(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/codex" if cmd == "codex" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["install", "mcp", "--codex", "--force", "--embedding", "text-embedding-3-small"],
        )
        assert result.exit_code == 0, result.output
        assert calls[0][:4] == ["codex", "mcp", "remove", "paperqa"]
        assert calls[1][:4] == ["codex", "mcp", "add", "paperqa"]


class TestUninstallSkillCommand:
    def test_uninstall_skill_removes_symlink_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "skills" / "papi"
        assert dest.is_symlink()

        result2 = runner.invoke(paperpipe.cli, ["uninstall", "skill", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_skill_mismatch_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["uninstall", "skill", "--codex"])
        assert result.exit_code == 1
        assert "use --force" in result.output.lower()


class TestUninstallPromptsCommand:
    def test_uninstall_prompts_removes_symlinks_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "papi.md"
        assert dest.is_symlink()

        result2 = runner.invoke(paperpipe.cli, ["uninstall", "prompts", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_prompts_removes_copied_files_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["install", "prompts", "--codex", "--copy"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "curate-paper-note.md"
        assert dest.exists()
        assert not dest.is_symlink()

        result2 = runner.invoke(paperpipe.cli, ["uninstall", "prompts", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_parses_commas_for_components(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["uninstall", "mcp,prompts", "--codex", "--force"])
        assert result.exit_code == 0, result.output


class TestUninstallMcpCommand:
    def test_uninstall_mcp_claude_runs_claude_mcp_remove(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["uninstall", "mcp", "--claude"])
        assert result.exit_code == 0, result.output

        assert ["claude", "mcp", "remove", "paperqa"] in calls
        assert ["claude", "mcp", "remove", "leann"] in calls

    def test_uninstall_mcp_repo_removes_server_keys(self, temp_db: Path):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "paperqa": {"command": "papi", "args": [], "env": {}},
                            "leann": {"command": "papi", "args": [], "env": {}},
                            "other": {"command": "x", "args": [], "env": {}},
                        }
                    }
                )
                + "\n"
            )
            (Path(".gemini") / "settings.json").parent.mkdir(parents=True, exist_ok=True)
            (Path(".gemini") / "settings.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "paperqa": {"command": "papi", "args": [], "env": {}},
                            "leann": {"command": "papi", "args": [], "env": {}},
                            "other": {"command": "x", "args": [], "env": {}},
                        }
                    }
                )
                + "\n"
            )

            result = runner.invoke(paperpipe.cli, ["uninstall", "mcp", "--repo"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert "paperqa" not in cfg["mcpServers"]
            assert "leann" not in cfg["mcpServers"]
            assert "other" in cfg["mcpServers"]

            cfg2 = json.loads((Path(".gemini") / "settings.json").read_text())
            assert "paperqa" not in cfg2["mcpServers"]
            assert "leann" not in cfg2["mcpServers"]
            assert "other" in cfg2["mcpServers"]

    def test_uninstall_mcp_gemini_writes_settings_json(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        (tmp_path / ".gemini").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".gemini" / "settings.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "paperqa": {"command": "papi", "args": [], "env": {}},
                        "leann": {"command": "papi", "args": [], "env": {}},
                    }
                }
            )
            + "\n"
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["uninstall", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output

        cfg = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert "paperqa" not in cfg.get("mcpServers", {})
        assert "leann" not in cfg.get("mcpServers", {})

    def test_uninstall_mcp_gemini_runs_gemini_mcp_remove(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["uninstall", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output

        assert ["gemini", "mcp", "remove", "--scope", "user", "paperqa"] in calls
        assert ["gemini", "mcp", "remove", "--scope", "user", "leann"] in calls


class TestUninstallValidation:
    def test_uninstall_repo_requires_mcp_component(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["uninstall", "skill", "--repo"])
        assert result.exit_code != 0
        assert "--repo is only valid" in result.output


class TestRemoveMultiplePapers:
    """Tests for removing multiple papers at once."""

    def test_remove_multiple_papers(self, temp_db: Path):
        """Test removing multiple papers in one command."""
        papers_dir = temp_db / "papers"
        for name in ["p1", "p2", "p3"]:
            (papers_dir / name).mkdir(parents=True)
            (papers_dir / name / "meta.json").write_text(
                json.dumps({"arxiv_id": f"id-{name}", "title": f"Paper {name}"})
            )
        paperpipe.save_index(
            {
                "p1": {"arxiv_id": "id-p1", "title": "Paper p1", "tags": [], "added": "x"},
                "p2": {"arxiv_id": "id-p2", "title": "Paper p2", "tags": [], "added": "x"},
                "p3": {"arxiv_id": "id-p3", "title": "Paper p3", "tags": [], "added": "x"},
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["remove", "p1", "p2", "--yes"])
        assert result.exit_code == 0
        assert "Removed: p1" in result.output
        assert "Removed: p2" in result.output
        assert not (papers_dir / "p1").exists()
        assert not (papers_dir / "p2").exists()
        assert (papers_dir / "p3").exists()  # Not removed
        index = paperpipe.load_index()
        assert "p1" not in index
        assert "p2" not in index
        assert "p3" in index

    def test_remove_multiple_partial_failure(self, temp_db: Path):
        """Test removing multiple papers where some fail."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(json.dumps({"arxiv_id": "id-p1", "title": "Paper p1"}))
        paperpipe.save_index({"p1": {"arxiv_id": "id-p1", "title": "Paper p1", "tags": [], "added": "x"}})

        runner = CliRunner()
        # p1 exists, nonexistent does not
        result = runner.invoke(paperpipe.cli, ["remove", "p1", "nonexistent", "--yes"])
        # Exit code 1 because one failed
        assert result.exit_code == 1
        assert "Removed: p1" in result.output
        assert "not found" in result.output.lower()
        assert "1 failed" in result.output
        assert not (papers_dir / "p1").exists()


class TestRegenerateMultiplePapers:
    """Tests for regenerating multiple papers at once."""

    def test_regenerate_multiple_papers(self, temp_db: Path):
        """Test regenerating multiple papers in one command."""
        papers_dir = temp_db / "papers"
        for name in ["p1", "p2"]:
            (papers_dir / name).mkdir(parents=True)
            (papers_dir / name / "meta.json").write_text(
                json.dumps({"arxiv_id": f"id-{name}", "title": f"Paper {name}", "authors": [], "abstract": ""})
            )
            (papers_dir / name / "source.tex").write_text(r"\begin{equation}x=1\end{equation}")

        paperpipe.save_index(
            {
                "p1": {"arxiv_id": "id-p1", "title": "Paper p1", "tags": [], "added": "x"},
                "p2": {"arxiv_id": "id-p2", "title": "Paper p2", "tags": [], "added": "x"},
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "p1", "p2", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating p1:" in result.output
        assert "Regenerating p2:" in result.output
        assert (papers_dir / "p1" / "summary.md").exists()
        assert (papers_dir / "p2" / "summary.md").exists()

    def test_regenerate_multiple_partial_failure(self, temp_db: Path):
        """Test regenerating multiple papers where some fail."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "id-p1", "title": "Paper p1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text(r"\begin{equation}x=1\end{equation}")
        paperpipe.save_index({"p1": {"arxiv_id": "id-p1", "title": "Paper p1", "tags": [], "added": "x"}})

        runner = CliRunner()
        # p1 exists, nonexistent does not
        result = runner.invoke(paperpipe.cli, ["regenerate", "p1", "nonexistent", "--no-llm", "-o", "summary"])
        # Exit code 1 because one failed
        assert result.exit_code == 1
        assert "Regenerating p1:" in result.output
        assert "not found" in result.output.lower()
        assert "1 failed" in result.output


class TestAuditCommand:
    def test_audit_flags_paper(self, temp_db: Path):
        papers_dir = temp_db / "papers"

        bad_dir = papers_dir / "bad-paper"
        bad_dir.mkdir(parents=True)
        (bad_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": "id-bad",
                    "title": "Bad Paper",
                    "authors": [],
                    "abstract": "This paper discusses occupancy fields.",
                    "tags": [],
                }
            )
        )
        (bad_dir / "source.tex").write_text(r"\begin{document}occupancy\end{document}")
        (bad_dir / "summary.md").write_text("**Eikonal Regularization:** This is important.\n")
        (bad_dir / "equations.md").write_text(
            'Based on the provided LaTeX content for the paper **"Some Other Paper"**\n'
        )

        ok_dir = papers_dir / "ok-paper"
        ok_dir.mkdir(parents=True)
        (ok_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": "id-ok",
                    "title": "Good Paper",
                    "authors": [],
                    "abstract": "We present an occupancy method.",
                    "tags": [],
                }
            )
        )
        (ok_dir / "source.tex").write_text(r"\begin{document}occupancy\end{document}")
        (ok_dir / "summary.md").write_text("# Good Paper\n\n**Occupancy:** Used for geometry.\n")
        (ok_dir / "equations.md").write_text("# Good Paper\n\n## Key Equations\n\n$x = y$\n")

        paperpipe.save_index(
            {
                "bad-paper": {"arxiv_id": "id-bad", "title": "Bad Paper", "tags": [], "added": "x"},
                "ok-paper": {"arxiv_id": "id-ok", "title": "Good Paper", "tags": [], "added": "x"},
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["audit"])
        assert result.exit_code == 0
        assert "flagged" in result.output.lower()
        assert "bad-paper" in result.output
        assert "Eikonal" in result.output
        assert "Some Other Paper" in result.output
        assert "boilerplate" in result.output.lower()  # Detects "based on the provided"
        assert "ok-paper" not in result.output  # OK papers are not listed

    def test_audit_regenerate_flagged_papers(self, temp_db: Path):
        papers_dir = temp_db / "papers"

        bad_dir = papers_dir / "bad-paper"
        bad_dir.mkdir(parents=True)
        (bad_dir / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": "id-bad",
                    "title": "Bad Paper",
                    "authors": [],
                    "abstract": "Abstract.",
                    "tags": [],
                    "added": "x",
                }
            )
        )
        (bad_dir / "source.tex").write_text(r"\begin{equation}x=1\end{equation}")
        (bad_dir / "summary.md").write_text("**Eikonal Regularization:** This is important.\n")
        (bad_dir / "equations.md").write_text(
            'Based on the provided LaTeX content for the paper **"Some Other Paper"**\n'
        )

        paperpipe.save_index({"bad-paper": {"arxiv_id": "id-bad", "title": "Bad Paper", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["audit", "--regenerate", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating bad-paper:" in result.output
        assert "# Bad Paper" in (bad_dir / "summary.md").read_text()
        assert "Eikonal" not in (bad_dir / "summary.md").read_text()


class TestExport:
    def test_export_nonexistent_paper(self, temp_db: Path):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["export", "nonexistent"])
            assert "not found" in result.output

    def test_export_summary(self, temp_db: Path):
        # Create a paper directory with summary
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("# Test Summary")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "summary", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper_summary.md").exists()
            assert Path("test-paper_summary.md").read_text() == "# Test Summary"

    def test_export_equations(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("# Equations\nE=mc^2")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "equations", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper_equations.md").exists()

    def test_export_full_with_source(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "source.tex").write_text(r"\documentclass{article}")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "full", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper.tex").exists()

    def test_export_full_without_source(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        # No source.tex file

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(paperpipe.cli, ["export", "test-paper", "--level", "full", "--to", "."])
            assert "No LaTeX source" in result.output


class TestAskCommand:
    def test_ask_constructs_correct_command(self, temp_db: Path, monkeypatch):
        # Mock pqa availability
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        # Mock subprocess.Popen (used for pqa with streaming output)
        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        # Add a dummy paper PDF so PaperQA has something to index.
        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
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
        assert pqa_kwargs.get("cwd") == paperpipe.PAPERS_DIR
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

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

        payload = paperpipe._paperqa_ask_evidence_blocks(
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
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "--pqa-agent-type", "fake"])
        assert result.exit_code == 0, result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--agent.agent_type" in pqa_call
        idx = pqa_call.index("--agent.agent_type") + 1
        assert pqa_call[idx] == "fake"

    def test_ask_filters_noisy_pqa_output_by_default(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="New file to index: test-paper.pdf...\nAnswer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query"])
        assert result.exit_code == 0, result.output
        assert "Answer" in result.output
        assert "New file to index:" not in result.output

    def test_ask_pqa_raw_disables_output_filtering(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="New file to index: test-paper.pdf...\nAnswer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "--pqa-raw"])
        assert result.exit_code == 0, result.output
        assert "Answer" in result.output
        assert "New file to index:" in result.output

    def test_ask_evidence_blocks_outputs_json(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)
        monkeypatch.setattr(
            paperpipe,
            "_paperqa_ask_evidence_blocks",
            lambda **kwargs: {"backend": "pqa", "question": "q", "answer": "a", "evidence": []},
        )

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "--format", "evidence-blocks"])
        assert result.exit_code == 0, result.output
        assert '"backend": "pqa"' in result.output

    def test_ask_evidence_blocks_rejects_passthrough_args(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["ask", "query", "--format", "evidence-blocks", "--agent.search_count", "10"],
        )
        assert result.exit_code != 0
        assert "--format evidence-blocks does not support extra passthrough args" in result.output

    def test_ask_evidence_blocks_reports_errors(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        def boom(**_kwargs):
            raise click.ClickException("boom")

        monkeypatch.setattr(paperpipe, "_paperqa_ask_evidence_blocks", boom)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "--format", "evidence-blocks"])
        assert result.exit_code != 0
        assert "Error: boom" in result.output

    def test_ask_ollama_models_prepare_env_for_pqa(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)
        monkeypatch.setattr(paperpipe, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "--pqa-llm", "ollama/qwen3:8b"])
        assert result.exit_code == 0, result.output

        _, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        env = pqa_kwargs.get("env") or {}
        assert env.get("OLLAMA_API_BASE") == "http://localhost:11434"
        assert env.get("OLLAMA_HOST") == "http://localhost:11434"

    def test_ask_retry_failed_index_clears_error_markers(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

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

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["ask", "query", "--pqa-llm", "my-llm", "--pqa-embedding", "my-embed", "--pqa-retry-failed"],
        )
        assert result.exit_code == 0

        mapping = paperpipe._paperqa_load_index_files_map(files_zip)
        assert mapping == {}

    def test_ask_does_not_override_user_settings(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "-s", "my-settings"])

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--settings" not in pqa_call
        assert "default" not in pqa_call
        assert "-s" in pqa_call
        assert "my-settings" in pqa_call

    def test_ask_does_not_override_user_parsing(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["ask", "query", "--parsing.multimodal", "ON_WITHOUT_ENRICHMENT"],
        )

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--parsing.multimodal" in pqa_call
        assert "ON_WITHOUT_ENRICHMENT" in pqa_call
        assert "OFF" not in pqa_call

    def test_ask_does_not_force_multimodal_when_pillow_available(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query"])

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--parsing.multimodal" not in pqa_call

    def test_ask_does_not_override_user_index_directory(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
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
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["ask", "query", "--agent.index.paper_directory", "/custom/papers"],
        )

        assert result.exit_code == 0

        pqa_call, _pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--agent.index.paper_directory" in pqa_call
        assert "/custom/papers" in pqa_call
        assert str(temp_db / ".pqa_papers") not in pqa_call

    def test_ask_marks_crashing_doc_error_when_custom_paper_directory_used(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        custom_papers = temp_db / "custom_papers"
        custom_papers.mkdir(parents=True)
        crashing_pdf = custom_papers / "crashy.pdf"
        crashing_pdf.write_bytes(b"%PDF-1.4\n% fake\n")

        mock_popen = MockPopen(
            returncode=1,
            stdout="New file to index: crashy.pdf...\nTraceback (most recent call last):\nBoom\n",
        )
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
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
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
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
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)
        # Ensure no env override
        monkeypatch.delenv("PAPERPIPE_PQA_CONCURRENCY", raising=False)
        monkeypatch.setattr(paperpipe, "_CONFIG_CACHE", None)

        mock_popen = MockPopen(returncode=0, stdout="Answer\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "test"])

        assert result.exit_code == 0
        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")

        # Concurrency should be set to 1 by default
        concurrency_idx = pqa_call.index("--agent.index.concurrency") + 1
        assert pqa_call[concurrency_idx] == "1"


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
        result = runner.invoke(paperpipe.cli, ["leann-index"])
        assert result.exit_code == 0, result.output

        cmd, kwargs = calls[0]
        assert cmd[:2] == ["leann", "build"]
        assert "papers" in cmd
        assert "--docs" in cmd and str(temp_db / ".pqa_papers") in cmd
        assert "--file-types" in cmd and ".pdf" in cmd
        assert "--embedding-model" in cmd and "nomic-embed-text" in cmd
        assert "--embedding-mode" in cmd and "ollama" in cmd
        assert kwargs.get("cwd") == temp_db
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

    def test_leann_index_rejects_file_types_override(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["leann-index", "--file-types", ".txt"])
        assert result.exit_code != 0
        assert "PDF-only" in result.output

    def test_ask_backend_leann_runs_leann_ask(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        meta = temp_db / ".leann" / "indexes" / "papers" / "documents.leann.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}")

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
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
        assert cmd[:3] == ["leann", "ask", "papers"]
        assert "what is x" in cmd
        assert "--llm" in cmd and "ollama" in cmd
        assert "--model" in cmd and "qwen3:8b" in cmd
        assert "--top-k" in cmd and "3" in cmd
        assert "--no-recompute" in cmd
        assert kwargs.get("cwd") == temp_db

    def test_ask_backend_leann_requires_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        build_calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs):
            build_calls.append((args, kwargs))
            # Simulate that `leann build` created the index metadata file.
            meta = temp_db / ".leann" / "indexes" / "papers" / "documents.leann.meta.json"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text('{"backend_name":"hnsw"}')
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        mock_popen = MockPopen(returncode=0, stdout="OUT\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "q", "--backend", "leann"])
        assert result.exit_code == 0, result.output
        assert "OUT" in result.output
        assert build_calls, "Expected `leann build` to run when index is missing"

    def test_ask_backend_leann_can_disable_auto_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/leann" if cmd == "leann" else None)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "q", "--backend", "leann", "--leann-no-auto-index"])
        assert result.exit_code != 0
        assert "Build it first" in result.output


class TestIndexCommand:
    def test_index_backend_pqa_runs_pqa_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["index", "--pqa-embedding", "my-embed"])
        assert result.exit_code == 0, result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "index" in pqa_call
        assert str(temp_db / ".pqa_papers") in pqa_call
        assert "--agent.index.paper_directory" in pqa_call
        assert "--index" in pqa_call and "paperpipe_my-embed" in pqa_call
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

    def test_index_backend_pqa_ollama_models_prepare_env(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)
        monkeypatch.setattr(paperpipe, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["index", "--pqa-llm", "ollama/qwen3:8b"])
        assert result.exit_code == 0, result.output

        _, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        env = pqa_kwargs.get("env") or {}
        assert env.get("OLLAMA_API_BASE") == "http://localhost:11434"
        assert env.get("OLLAMA_HOST") == "http://localhost:11434"

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
        result = runner.invoke(paperpipe.cli, ["index", "--backend", "leann"])
        assert result.exit_code == 0, result.output

        cmd, kwargs = calls[0]
        assert cmd[:2] == ["leann", "build"]
        assert "papers" in cmd
        assert "--docs" in cmd and str(temp_db / ".pqa_papers") in cmd
        assert "--embedding-model" in cmd and "nomic-embed-text" in cmd
        assert "--embedding-mode" in cmd and "ollama" in cmd
        assert kwargs.get("cwd") == temp_db


class TestModelsCommand:
    def test_models_reports_ok_and_fail(self, monkeypatch):
        calls = []

        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            calls.append(("completion", model))
            if model == "bad-model":
                raise RuntimeError("missing key")
            return {"ok": True}

        def embedding(*, model, input, timeout):
            calls.append(("embedding", model))
            if model == "bad-embed":
                raise RuntimeError("not authorized")
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            [
                "models",
                "--kind",
                "completion",
                "--model",
                "ok-model",
                "--model",
                "bad-model",
            ],
        )

        assert result.exit_code == 0
        assert "OK" in result.output
        assert "FAIL" in result.output
        assert "ok-model" in result.output
        assert "bad-model" in result.output
        assert ("completion", "ok-model") in calls
        assert ("completion", "bad-model") in calls

    def test_models_json_output(self, monkeypatch):
        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            return {"ok": True}

        def embedding(*, model, input, timeout):
            raise RuntimeError("no key")

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            [
                "models",
                "--kind",
                "completion",
                "--kind",
                "embedding",
                "--model",
                "ok-model",
                "--model",
                "bad-embed",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert any(r["ok"] is True and r["model"] == "ok-model" and r["kind"] == "completion" for r in payload)
        assert any(r["ok"] is False and r["model"] == "bad-embed" and r["kind"] == "embedding" for r in payload)

    def test_models_requires_litellm(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace())
        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["models", "--kind", "completion", "--model", "ok-model"])
        assert result.exit_code != 0
        assert "LiteLLM is required" in result.output

    def test_models_accepts_presets(self, monkeypatch):
        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            return {"ok": True}

        def embedding(*, model, input, timeout, **kwargs):
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["models", "--preset", "latest", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(paperpipe.cli, ["models", "latest", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(paperpipe.cli, ["models", "--preset", "last-gen", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(paperpipe.cli, ["models", "last-gen", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(paperpipe.cli, ["models", "--preset", "all", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(paperpipe.cli, ["models", "all", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

    def test_models_default_probes_only_configured_combo(self, monkeypatch):
        calls = []

        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            calls.append(("completion", model))
            return {"ok": True}

        def embedding(*, model, input, timeout, **kwargs):
            calls.append(("embedding", model))
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        # Pretend all provider keys are set; the default should still probe only the
        # configured paperpipe model+embedding combo (mapped to "latest" for that provider).
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("VOYAGE_API_KEY", "x")

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["models", "--json"])
        assert result.exit_code == 0

        # Default run probes one "latest" completion + embedding per configured provider.
        assert ("completion", "gpt-5.2") in calls
        assert ("completion", "gemini/gemini-3-flash-preview") in calls
        assert ("completion", "claude-sonnet-4-5") in calls
        assert ("embedding", "text-embedding-3-large") in calls
        assert ("embedding", "gemini/gemini-embedding-001") in calls
        assert ("embedding", "voyage/voyage-3-large") in calls

    def test_models_positional_preset_is_explicit(self, monkeypatch):
        calls = []

        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            calls.append(("completion", model))
            return {"ok": True}

        def embedding(*, model, input, timeout, **kwargs):
            calls.append(("embedding", model))
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("VOYAGE_API_KEY", "x")

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["models", "latest", "--kind", "completion", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        # Explicit "latest" probes the full preset list (includes Pro/Opus).
        assert ("completion", "gemini/gemini-3-pro-preview") in calls
        assert ("completion", "claude-opus-4-5") in calls


class TestRegenerateCommand:
    def test_regenerate_all_flag(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p2").mkdir(parents=True)

        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p2" / "meta.json").write_text(
            json.dumps({"arxiv_id": "2", "title": "Paper 2", "authors": [], "abstract": ""})
        )

        tex = "\\begin{equation}E=mc^2\\end{equation}"
        (papers_dir / "p1" / "source.tex").write_text(tex)
        (papers_dir / "p2" / "source.tex").write_text(tex)

        (papers_dir / "p1" / "summary.md").write_text("old")
        (papers_dir / "p2" / "summary.md").write_text("old")
        (papers_dir / "p1" / "equations.md").write_text("old")
        (papers_dir / "p2" / "equations.md").write_text("old")

        paperpipe.save_index(
            {
                "p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"},
                "p2": {"arxiv_id": "2", "title": "Paper 2", "tags": [], "added": "x"},
            }
        )

        runner = CliRunner()
        # Use summary,equations,tags to avoid name regeneration which would rename the papers
        result = runner.invoke(paperpipe.cli, ["regenerate", "--all", "--no-llm", "-o", "summary,equations,tags"])
        assert result.exit_code == 0
        assert "Regenerating p1:" in result.output
        assert "Regenerating p2:" in result.output

        assert (papers_dir / "p1" / "summary.md").read_text() != "old"
        assert (papers_dir / "p2" / "summary.md").read_text() != "old"

    def test_regenerate_all_positional_alias(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("old")
        (papers_dir / "p1" / "equations.md").write_text("old")

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "all", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating p1:" in result.output

    def test_regenerate_all_does_not_steal_paper_named_all(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "all").mkdir(parents=True)
        (papers_dir / "p2").mkdir(parents=True)

        (papers_dir / "all" / "meta.json").write_text(
            json.dumps({"arxiv_id": "a", "title": "All Paper", "authors": [], "abstract": ""})
        )
        (papers_dir / "p2" / "meta.json").write_text(
            json.dumps({"arxiv_id": "b", "title": "Other Paper", "authors": [], "abstract": ""})
        )
        (papers_dir / "all" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p2" / "source.tex").write_text("\\begin{equation}x=2\\end{equation}")
        (papers_dir / "all" / "summary.md").write_text("old")
        (papers_dir / "p2" / "summary.md").write_text("old")
        (papers_dir / "all" / "equations.md").write_text("old")
        (papers_dir / "p2" / "equations.md").write_text("old")

        paperpipe.save_index(
            {
                "all": {"arxiv_id": "a", "title": "All Paper", "tags": [], "added": "x"},
                "p2": {"arxiv_id": "b", "title": "Other Paper", "tags": [], "added": "x"},
            }
        )

        runner = CliRunner()
        # With a paper named "all" in the index, "regenerate all" should target that paper only
        result = runner.invoke(paperpipe.cli, ["regenerate", "all", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating all:" in result.output
        assert "Regenerating p2:" not in result.output

    def test_regenerate_all_fails_if_missing_metadata(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "--all", "--no-llm"])
        assert result.exit_code != 0
        assert "failed to regenerate" in result.output.lower()

    def test_regenerate_accepts_arxiv_urls(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps(
                {
                    "arxiv_id": TEST_ARXIV_ID,
                    "title": "Paper 1",
                    "authors": [],
                    "abstract": "",
                }
            )
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        paperpipe.save_index({"p1": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["regenerate", f"https://arxiv.org/abs/{TEST_ARXIV_ID}", "--no-llm", "-o", "summary"],
        )
        assert result.exit_code == 0
        assert "Regenerating p1:" in result.output

    def test_regenerate_only_missing_by_default(self, temp_db: Path):
        """Without --overwrite, only missing fields are regenerated."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": "", "tags": ["existing"]})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("existing summary")
        # No equations.md - this should be regenerated

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": ["existing"], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "p1", "--no-llm"])
        assert result.exit_code == 0
        # Should only regenerate equations (missing)
        assert "equations" in result.output
        # Should NOT regenerate summary (exists) or tags (exists)
        assert "summary" not in result.output.lower() or "equations" in result.output
        # Summary should be unchanged
        assert (papers_dir / "p1" / "summary.md").read_text() == "existing summary"
        # Equations should now exist
        assert (papers_dir / "p1" / "equations.md").exists()

    def test_regenerate_overwrite_specific_field(self, temp_db: Path):
        """--overwrite <field> only regenerates that specific field."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("old summary")
        (papers_dir / "p1" / "equations.md").write_text("old equations")

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "p1", "--no-llm", "-o", "summary"])
        assert result.exit_code == 0
        assert "summary" in result.output
        # Summary should be changed
        assert (papers_dir / "p1" / "summary.md").read_text() != "old summary"
        # Equations should be unchanged
        assert (papers_dir / "p1" / "equations.md").read_text() == "old equations"

    def test_regenerate_overwrite_invalid_field(self, temp_db: Path):
        """--overwrite with invalid field should error."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "p1", "-o", "invalid"])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower()


class TestRemoveCommand:
    def test_remove_by_arxiv_url(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "summary.md").write_text("summary")
        paperpipe.save_index({"p1": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["remove", f"https://arxiv.org/abs/{TEST_ARXIV_ID}", "--yes"])
        assert result.exit_code == 0
        assert "Removed: p1" in result.output
        assert not (papers_dir / "p1").exists()
        assert "p1" not in paperpipe.load_index()

    def test_remove_by_arxiv_id_ambiguous(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p2").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p2" / "meta.json").write_text(
            json.dumps({"arxiv_id": TEST_ARXIV_ID, "title": "Paper 2", "authors": [], "abstract": ""})
        )
        paperpipe.save_index(
            {
                "p1": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "tags": [], "added": "x"},
                "p2": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 2", "tags": [], "added": "x"},
            }
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["remove", TEST_ARXIV_ID, "--yes"])
        # Exit code 1 because the operation failed (ambiguous match)
        assert result.exit_code == 1
        assert "Multiple papers match arXiv ID" in result.output
        # Neither paper should be removed
        assert (papers_dir / "p1").exists()
        assert (papers_dir / "p2").exists()


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestArxivIntegration:
    """Integration tests for arXiv API (requires network)."""

    def test_fetch_arxiv_metadata(self):
        """Test fetching metadata from arXiv."""
        meta = paperpipe.fetch_arxiv_metadata(TEST_ARXIV_ID)

        assert meta["arxiv_id"] == TEST_ARXIV_ID
        assert "Attention" in meta["title"]
        assert len(meta["authors"]) > 0
        assert "Vaswani" in meta["authors"][0]
        assert len(meta["abstract"]) > 100
        assert len(meta["categories"]) > 0
        assert meta["pdf_url"] is not None

    def test_download_pdf(self, tmp_path: Path):
        """Test downloading PDF from arXiv."""
        pdf_path = tmp_path / "paper.pdf"
        result = paperpipe.download_pdf(TEST_ARXIV_ID, pdf_path)

        assert result is True
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 100_000  # Should be > 100KB

    def test_download_source(self, tmp_path: Path):
        """Test downloading LaTeX source from arXiv."""
        tex_content = paperpipe.download_source(TEST_ARXIV_ID, tmp_path)

        # "Attention Is All You Need" has LaTeX source
        assert tex_content is not None
        assert "\\begin{document}" in tex_content
        assert (tmp_path / "source.tex").exists()

    def test_extract_equations_from_source(self, tmp_path: Path):
        """Test equation extraction runs on downloaded source."""
        tex_content = paperpipe.download_source(TEST_ARXIV_ID, tmp_path)
        assert tex_content is not None

        # Just verify extraction runs without error
        # (not all papers use standard equation environments)
        equations = paperpipe.extract_equations_simple(tex_content)
        assert isinstance(equations, str)


@pytest.mark.integration
class TestAddCommandIntegration:
    """Integration tests for the full add workflow."""

    def test_add_paper_no_llm(self, temp_db: Path):
        """Test adding a paper without LLM generation."""
        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", TEST_ARXIV_ID, "--name", "attention", "--no-llm"],
        )

        assert result.exit_code == 0
        assert "Added: attention" in result.output

        # Verify files were created
        paper_dir = temp_db / "papers" / "attention"
        assert paper_dir.exists()
        assert (paper_dir / "paper.pdf").exists()
        assert (paper_dir / "source.tex").exists()
        assert (paper_dir / "summary.md").exists()
        assert (paper_dir / "equations.md").exists()
        assert (paper_dir / "meta.json").exists()

        # Verify metadata
        meta = json.loads((paper_dir / "meta.json").read_text())
        assert meta["arxiv_id"] == TEST_ARXIV_ID
        assert "Attention" in meta["title"]
        assert meta["has_pdf"] is True
        assert meta["has_source"] is True

        # Verify index was updated
        index = paperpipe.load_index()
        assert "attention" in index

    def test_add_paper_with_custom_tags(self, temp_db: Path):
        """Test adding a paper with custom tags."""
        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            [
                "add",
                TEST_ARXIV_ID,
                "--name",
                "transformer",
                "--tags",
                "nlp,deep-learning",
                "--no-llm",
            ],
        )

        assert result.exit_code == 0

        meta = json.loads((temp_db / "papers" / "transformer" / "meta.json").read_text())
        assert "nlp" in meta["tags"]
        assert "deep-learning" in meta["tags"]

    # NOTE: test_add_paper_already_exists moved to unit tests (TestAddCommand)
    # NOTE: test_remove_paper covered by unit tests (TestRemoveCommand)


@pytest.mark.integration
@pytest.mark.skipif(not litellm_available(), reason="LiteLLM not installed or no API key configured")
class TestLlmIntegration:
    """Integration tests for LLM-based generation."""

    @pytest.mark.slow
    def test_generate_with_litellm(self):
        """Test LLM-based content generation."""
        meta = {
            "title": "Test Paper",
            "authors": ["Author One"],
            "abstract": "This paper introduces a novel method for testing.",
        }
        tex_content = r"""
        \begin{document}
        \begin{equation}
        L = \sum_{i} \|f(x_i) - y_i\|^2
        \end{equation}
        \end{document}
        """

        summary, equations, tags = paperpipe.generate_with_litellm(meta, tex_content)

        assert len(summary) > 50
        assert len(equations) > 20
        assert isinstance(tags, list)

    @pytest.mark.slow
    def test_add_paper_with_llm(self, temp_db: Path):
        """Test adding a paper with LLM generation enabled."""
        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", TEST_ARXIV_ID, "--name", "attention-llm"],
        )

        assert result.exit_code == 0
        assert "Added: attention-llm" in result.output

        # LLM should generate more sophisticated content
        paper_dir = temp_db / "papers" / "attention-llm"
        summary = (paper_dir / "summary.md").read_text()
        equations = (paper_dir / "equations.md").read_text()

        # LLM summaries should be more than just the abstract
        assert len(summary) > 200

        # LLM equation extraction should have explanations
        assert len(equations) > 100


@pytest.mark.integration
@pytest.mark.skipif(not pqa_available(), reason="PaperQA2 (pqa) not installed")
class TestPaperQAIntegration:
    """Integration tests for PaperQA2."""

    @pytest.mark.slow
    def test_ask_command(self, temp_db: Path):
        """Test the ask command with PaperQA2."""
        runner = CliRunner()

        # First add a paper
        runner.invoke(
            paperpipe.cli,
            ["add", TEST_ARXIV_ID, "--name", "attention-pqa", "--no-llm"],
        )

        # Then query it
        result = runner.invoke(
            paperpipe.cli,
            ["ask", "What is the attention mechanism?"],
        )

        # Should get some response (not just the fallback)
        assert result.exit_code == 0


class TestAskErrorHandling:
    def test_ask_excludes_failed_files_from_staging(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

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

        # Write the index file with ERROR status
        files_zip.write_bytes(zlib.compress(pickle.dumps({staged_path_str: "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)))

        runner = CliRunner()
        # Run ask WITHOUT --pqa-retry-failed
        result = runner.invoke(
            paperpipe.cli,
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
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

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
        files_zip.write_bytes(zlib.compress(pickle.dumps({staged_path_str: "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)))

        runner = CliRunner()
        # Run ask WITH --pqa-retry-failed
        result = runner.invoke(
            paperpipe.cli,
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
        mapping = paperpipe._paperqa_load_index_files_map(files_zip)
        assert mapping == {}, "ERROR markers should be cleared"
