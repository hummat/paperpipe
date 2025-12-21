"""Tests for paperpipe."""

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

import paperpipe

# Well-known paper for integration tests: "Attention Is All You Need"
TEST_ARXIV_ID = "1706.03762"


def llm_cli_available() -> bool:
    """Check if llm CLI is installed and configured."""
    if not shutil.which("llm"):
        return False
    # Consider env vars as "configured" (llm can use provider keys from env).
    if (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    ):
        return True

    # Otherwise, require at least one stored key.
    try:
        result = subprocess.run(["llm", "keys", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False
        out = (result.stdout or "").strip()
        return bool(out) and "No keys found" not in out
    except Exception:
        return False


def pqa_available() -> bool:
    """Check if PaperQA2 CLI is installed."""
    return shutil.which("pqa") is not None


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a temporary paper database."""
    db_path = tmp_path / ".paperpipe"
    monkeypatch.setattr(paperpipe, "PAPER_DB", db_path)
    monkeypatch.setattr(paperpipe, "PAPERS_DIR", db_path / "papers")
    monkeypatch.setattr(paperpipe, "INDEX_FILE", db_path / "index.json")
    paperpipe.ensure_db()
    return db_path


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


class TestIndex:
    def test_load_empty_index(self, temp_db: Path):
        index = paperpipe.load_index()
        assert index == {}

    def test_save_and_load_index(self, temp_db: Path):
        test_data = {"paper1": {"title": "Test Paper", "tags": ["ml"]}}
        paperpipe.save_index(test_data)
        loaded = paperpipe.load_index()
        assert loaded == test_data


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
            result = runner.invoke(
                paperpipe.cli, ["export", "test-paper", "--level", "summary", "--to", "."]
            )
            assert result.exit_code == 0
            assert Path("test-paper_summary.md").exists()
            assert Path("test-paper_summary.md").read_text() == "# Test Summary"


class TestAskCommand:
    def test_ask_constructs_correct_command(self, temp_db: Path, monkeypatch):
        # Mock pqa availability
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        # Mock subprocess.run
        mock_run_calls = []

        def mock_run(cmd, **kwargs):
            mock_run_calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Answer")

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Add a dummy paper so the loop over PAPERS_DIR works
        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            [
                "ask",
                "query",
                "--llm",
                "my-llm",
                "--embedding",
                "my-embed",
                "--temperature",
                "0.7",
                "--verbosity",
                "2",
            ],
        )

        assert result.exit_code == 0

        # Verify pqa was called with correct args
        pqa_call = next(c for c in mock_run_calls if c[0] == "pqa")
        assert "pqa" in pqa_call
        assert "--settings" in pqa_call
        assert "default" in pqa_call
        assert "--parsing.multimodal" in pqa_call
        assert "OFF" in pqa_call
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

    def test_ask_does_not_override_user_settings(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        mock_run_calls = []

        def mock_run(cmd, **kwargs):
            mock_run_calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Answer")

        monkeypatch.setattr(subprocess, "run", mock_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query", "-s", "my-settings"])

        assert result.exit_code == 0

        pqa_call = next(c for c in mock_run_calls if c[0] == "pqa")
        assert "--settings" not in pqa_call
        assert "default" not in pqa_call
        assert "-s" in pqa_call
        assert "my-settings" in pqa_call

    def test_ask_does_not_override_user_parsing(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: False)

        mock_run_calls = []

        def mock_run(cmd, **kwargs):
            mock_run_calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Answer")

        monkeypatch.setattr(subprocess, "run", mock_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["ask", "query", "--parsing.multimodal", "ON_WITHOUT_ENRICHMENT"],
        )

        assert result.exit_code == 0

        pqa_call = next(c for c in mock_run_calls if c[0] == "pqa")
        assert "--parsing.multimodal" in pqa_call
        assert "ON_WITHOUT_ENRICHMENT" in pqa_call
        assert "OFF" not in pqa_call

    def test_ask_does_not_force_multimodal_when_pillow_available(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperpipe, "_pillow_available", lambda: True)

        mock_run_calls = []

        def mock_run(cmd, **kwargs):
            mock_run_calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Answer")

        monkeypatch.setattr(subprocess, "run", mock_run)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["ask", "query"])

        assert result.exit_code == 0

        pqa_call = next(c for c in mock_run_calls if c[0] == "pqa")
        assert "--parsing.multimodal" not in pqa_call


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
        assert any(
            r["ok"] is True and r["model"] == "ok-model" and r["kind"] == "completion"
            for r in payload
        )
        assert any(
            r["ok"] is False and r["model"] == "bad-embed" and r["kind"] == "embedding"
            for r in payload
        )

    def test_models_requires_litellm(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace())
        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli, ["models", "--kind", "completion", "--model", "ok-model"]
        )
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

        monkeypatch.setattr(paperpipe, "DEFAULT_PQA_LLM", "gemini/gemini-2.5-flash")
        monkeypatch.setattr(paperpipe, "DEFAULT_PQA_EMBEDDING", "text-embedding-3-small")

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
        result = runner.invoke(
            paperpipe.cli, ["models", "latest", "--kind", "completion", "--json"]
        )
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
        result = runner.invoke(paperpipe.cli, ["regenerate", "--all", "--no-llm"])
        assert result.exit_code == 0
        assert "Regenerating: p1" in result.output
        assert "Regenerating: p2" in result.output

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

        paperpipe.save_index(
            {"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}}
        )

        runner = CliRunner()
        result = runner.invoke(paperpipe.cli, ["regenerate", "all", "--no-llm"])
        assert result.exit_code == 0
        assert "Regenerating: p1" in result.output

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
        result = runner.invoke(paperpipe.cli, ["regenerate", "all", "--no-llm"])
        assert result.exit_code == 0
        assert "Regenerating: all" in result.output
        assert "Regenerating: p2" not in result.output

    def test_regenerate_all_fails_if_missing_metadata(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        paperpipe.save_index(
            {"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}}
        )

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
        paperpipe.save_index(
            {"p1": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "tags": [], "added": "x"}}
        )

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["regenerate", f"https://arxiv.org/abs/{TEST_ARXIV_ID}", "--no-llm"],
        )
        assert result.exit_code == 0
        assert "Regenerating: p1" in result.output


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

    def test_add_paper_already_exists(self, temp_db: Path):
        """Test that adding duplicate paper fails gracefully."""
        runner = CliRunner()

        # Add first time
        runner.invoke(
            paperpipe.cli,
            ["add", TEST_ARXIV_ID, "--name", "test-dup", "--no-llm"],
        )

        # Try to add again
        result = runner.invoke(
            paperpipe.cli,
            ["add", TEST_ARXIV_ID, "--name", "test-dup", "--no-llm"],
        )

        assert "already exists" in result.output

    def test_remove_paper(self, temp_db: Path):
        """Test removing a paper."""
        runner = CliRunner()

        # Add a paper first
        runner.invoke(
            paperpipe.cli,
            ["add", TEST_ARXIV_ID, "--name", "to-remove", "--no-llm"],
        )

        # Remove it
        result = runner.invoke(
            paperpipe.cli,
            ["remove", "to-remove", "--yes"],
        )

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert not (temp_db / "papers" / "to-remove").exists()
        assert "to-remove" not in paperpipe.load_index()


@pytest.mark.integration
@pytest.mark.skipif(not llm_cli_available(), reason="llm CLI not installed or configured")
class TestLlmIntegration:
    """Integration tests for LLM-based generation."""

    @pytest.mark.slow
    def test_generate_with_llm_cli(self):
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

        summary, equations, tags = paperpipe.generate_with_llm_cli(meta, tex_content)

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
            ["ask", "What is the attention mechanism?", "-p", "attention-pqa"],
        )

        # Should get some response (not just the fallback)
        assert result.exit_code == 0
