"""Integration tests for CLI commands (require network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import litellm_available, pqa_available

import paperpipe

from .conftest import TEST_ARXIV_ID, cli_mod


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
            cli_mod.cli,
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
            cli_mod.cli,
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
class TestSemanticScholarIntegration:
    """Integration tests for Semantic Scholar API (requires network)."""

    def test_fetch_semantic_scholar_metadata(self):
        """Test fetching metadata from Semantic Scholar."""
        from paperpipe.cli.helpers import _fetch_semantic_scholar_metadata

        # "Attention Is All You Need" S2 paper ID
        s2_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        meta = _fetch_semantic_scholar_metadata(s2_id)

        if meta is None:
            pytest.skip("Semantic Scholar API rate limited or unavailable")

        assert "Attention" in meta["title"]
        assert len(meta["authors"]) > 0
        assert meta.get("arxiv_id") == "1706.03762"


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

        summary, equations, tags, _ = paperpipe.generate_with_litellm(meta, tex_content)

        assert len(summary) > 50
        assert len(equations) > 20
        assert isinstance(tags, list)

    @pytest.mark.slow
    def test_add_paper_with_llm(self, temp_db: Path):
        """Test adding a paper with LLM generation enabled."""
        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
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
            cli_mod.cli,
            ["add", TEST_ARXIV_ID, "--name", "attention-pqa", "--no-llm"],
        )

        # Then query it
        result = runner.invoke(
            cli_mod.cli,
            ["ask", "What is the attention mechanism?"],
        )

        # Just verify it runs - actual answer quality depends on PaperQA2
        assert result.exit_code == 0
