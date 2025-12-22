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

    def test_add_paper_already_exists(self, temp_db: Path):
        """Test that adding duplicate paper fails gracefully (unit test, no network)."""
        # Pre-create the paper directory to simulate existing paper
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(
            json.dumps({"arxiv_id": "1706.03762", "title": "Test", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({"test-paper": {"arxiv_id": "1706.03762", "title": "Test", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(
            paperpipe.cli,
            ["add", "1706.03762", "--name", "test-paper", "--no-llm"],
        )

        assert "already exists" in result.output

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
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "id-p1", "title": "Paper p1"})
        )
        paperpipe.save_index(
            {"p1": {"arxiv_id": "id-p1", "title": "Paper p1", "tags": [], "added": "x"}}
        )

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
        paperpipe.save_index(
            {"p1": {"arxiv_id": "id-p1", "title": "Paper p1", "tags": [], "added": "x"}}
        )

        runner = CliRunner()
        # p1 exists, nonexistent does not
        result = runner.invoke(paperpipe.cli, ["regenerate", "p1", "nonexistent", "--no-llm", "-o", "summary"])
        # Exit code 1 because one failed
        assert result.exit_code == 1
        assert "Regenerating p1:" in result.output
        assert "not found" in result.output.lower()
        assert "1 failed" in result.output


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

        monkeypatch.setattr(paperpipe, "DEFAULT_LLM_MODEL", "gemini/gemini-3-flash-preview")
        monkeypatch.setattr(paperpipe, "DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

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
            ["ask", "What is the attention mechanism?", "-p", "attention-pqa"],
        )

        # Should get some response (not just the fallback)
        assert result.exit_code == 0
