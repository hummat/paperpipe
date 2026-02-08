from __future__ import annotations

import logging
import tarfile
import tempfile
from pathlib import Path

import pytest

import paperpipe
import paperpipe.paper as paper_mod


class TestGenerateLlmContent:
    """Tests for generate_llm_content fallback behavior."""

    def test_falls_back_when_litellm_unavailable(self, tmp_path, monkeypatch):
        # Force LiteLLM to appear unavailable
        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: False)

        meta = {
            "arxiv_id": "2301.00001",
            "title": "Test Paper",
            "authors": ["Author"],
            "abstract": "This is the abstract.",
            "categories": ["cs.CV"],
            "published": "2023-01-01",
        }
        tex_content = r"\begin{equation}E=mc^2\end{equation}"

        summary, equations, tags, tldr = paperpipe.generate_llm_content(tmp_path, meta, tex_content)

        # Should get simple summary (contains title)
        assert "Test Paper" in summary
        # Should get simple equation extraction
        assert "E=mc^2" in equations
        # No LLM tags
        assert tags == []
        # Should get simple TL;DR
        assert "Test Paper" in tldr

    def test_falls_back_without_tex_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: False)

        meta = {
            "arxiv_id": "2301.00001",
            "title": "Test Paper",
            "authors": ["Author"],
            "abstract": "Abstract text.",
            "categories": [],
            "published": "2023-01-01",
        }

        summary, equations, tags, tldr = paperpipe.generate_llm_content(tmp_path, meta, None)

        assert "Test Paper" in summary
        assert "No LaTeX source" in equations
        assert "Test Paper" in tldr

    def test_extracts_pdf_text_when_no_latex_source(self, tmp_path, monkeypatch):
        """Integration test: PDF text should be extracted when tex_content is None.

        This test catches the bug where local PDFs had no content for LLM summarization.
        """
        pytest.importorskip("fitz")
        import fitz

        # Create a PDF with recognizable content
        pdf_path = tmp_path / "paper.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This paper introduces a novel deep learning method.")
        doc.save(pdf_path)
        doc.close()

        meta = {
            "arxiv_id": None,
            "title": "Test Paper",
            "authors": ["Author"],
            "abstract": "No abstract available (local PDF).",  # Placeholder - the bug scenario
            "categories": [],
            "published": None,
        }

        # Track what content is passed to the LLM
        captured_content = []

        def mock_generate_with_litellm(meta, tex_content, **kwargs):
            captured_content.append(tex_content)
            # Return valid output structure
            return ("summary", "equations", [], "tldr")

        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: True)
        monkeypatch.setattr(paper_mod, "generate_with_litellm", mock_generate_with_litellm)

        paperpipe.generate_llm_content(tmp_path, meta, None)

        # The critical assertion: LLM should receive PDF text, not None/empty
        assert len(captured_content) == 1
        assert captured_content[0] is not None, "LLM received no content - PDF extraction failed"
        assert "deep learning" in captured_content[0], "PDF text was not extracted"

    def test_rejects_placeholder_only_content(self, tmp_path, monkeypatch):
        """Verify that placeholder abstracts don't constitute meaningful content."""
        # No PDF, no tex, just placeholder abstract - this should NOT go to LLM with content
        meta = {
            "title": "Test",
            "authors": [],
            "abstract": "No abstract available (local PDF).",
        }

        captured_content = []

        def mock_generate_with_litellm(meta, tex_content, **kwargs):
            captured_content.append(tex_content)
            return ("summary", "equations", [], "tldr")

        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: True)
        monkeypatch.setattr(paper_mod, "generate_with_litellm", mock_generate_with_litellm)

        paperpipe.generate_llm_content(tmp_path, meta, None)

        # With no PDF and no tex, content should be None
        assert len(captured_content) == 1
        assert captured_content[0] is None, "Expected no content when PDF doesn't exist"


class TestExtractPdfText:
    """Tests for _extract_pdf_text function."""

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        """Should return None for files that don't exist."""
        result = paper_mod._extract_pdf_text(tmp_path / "nonexistent.pdf")
        assert result is None

    def test_returns_none_for_invalid_pdf(self, tmp_path):
        """Should return None for invalid PDF files."""
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_text("not a pdf")
        result = paper_mod._extract_pdf_text(bad_pdf)
        assert result is None

    def test_extracts_text_from_valid_pdf(self, tmp_path):
        """Should extract text from a valid PDF (if fitz is available)."""
        pytest.importorskip("fitz")

        # Create a minimal valid PDF using fitz
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello World")
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_pdf_text(pdf_path)
        assert result is not None
        assert "Hello" in result

    def test_extracts_full_text_from_multipage_pdf(self, tmp_path):
        """Should extract all text from multi-page PDFs without truncation."""
        pytest.importorskip("fitz")

        import fitz

        pdf_path = tmp_path / "long.pdf"
        doc = fitz.open()
        for i in range(5):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page{i}" + "X" * 500)
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_pdf_text(pdf_path)
        assert result is not None
        # Should contain text from all pages
        for i in range(5):
            assert f"Page{i}" in result

    def test_returns_none_for_empty_pdf(self, tmp_path):
        """Should return None for PDFs with no extractable text (e.g., image-only)."""
        pytest.importorskip("fitz")

        import fitz

        # Create PDF with blank page (no text)
        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # Blank page with no text
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_pdf_text(pdf_path)
        assert result is None  # Empty string filtered to None

    def test_falls_back_to_fitz_when_pymupdf4llm_unavailable(self, tmp_path, monkeypatch):
        """Should use fitz fallback when pymupdf4llm is not installed."""
        pytest.importorskip("fitz")
        import sys

        import fitz

        # Block pymupdf4llm import
        monkeypatch.setitem(sys.modules, "pymupdf4llm", None)

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Fallback Test")
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_pdf_text(pdf_path)
        assert result is not None
        assert "Fallback" in result


class TestExtractFirstPageText:
    """Tests for _extract_first_page_text helper."""

    def test_extracts_first_page_text(self, tmp_path):
        """Should extract text from the first page of a PDF."""
        pytest.importorskip("fitz")
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "First Page Title")
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_first_page_text(pdf_path)
        assert result is not None
        assert "First Page" in result

    def test_falls_back_to_fitz_when_pymupdf4llm_unavailable(self, tmp_path, monkeypatch):
        """Should use fitz fallback when pymupdf4llm is not installed."""
        pytest.importorskip("fitz")
        import sys

        import fitz

        # Block pymupdf4llm import
        monkeypatch.setitem(sys.modules, "pymupdf4llm", None)

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Fallback Title")
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_first_page_text(pdf_path)
        assert result is not None
        assert "Fallback" in result

    def test_returns_none_for_empty_pdf(self, tmp_path):
        """Should return None for PDFs with no text."""
        pytest.importorskip("fitz")
        import fitz

        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(pdf_path)
        doc.close()

        result = paper_mod._extract_first_page_text(pdf_path)
        assert result is None


class TestGetFallbackContextWindow:
    """Tests for _get_fallback_context_window helper."""

    def test_exact_match(self):
        """Should return context window for exact model name match."""
        result = paper_mod._get_fallback_context_window("gpt-4o")
        assert result == 128_000

    def test_prefix_match(self):
        """Should return context window for prefix match."""
        result = paper_mod._get_fallback_context_window("gemini-2.5-flash-preview")
        assert result == 1_048_576

    def test_extracts_base_name_from_provider_prefix(self):
        """Should extract base model name from provider/vendor/model format."""
        result = paper_mod._get_fallback_context_window("openrouter/google/gemini-2.5-flash")
        assert result == 1_048_576

    def test_returns_none_for_unknown_model(self):
        """Should return None for unknown models."""
        result = paper_mod._get_fallback_context_window("unknown/model")
        assert result is None


class TestCheckContextLimit:
    """Tests for _check_context_limit helper."""

    def test_returns_ok_when_under_limit(self, monkeypatch):
        """Should return (True, None) when content is under limit."""
        import types

        mock_litellm = types.SimpleNamespace(
            get_model_info=lambda m: {"max_input_tokens": 100_000},
            token_counter=lambda model, messages: 1000,
        )

        ok, err = paper_mod._check_context_limit([{"role": "user", "content": "test"}], "gpt-4o", mock_litellm)
        assert ok is True
        assert err is None

    def test_returns_error_when_over_limit(self):
        """Should return (False, error_msg) when content exceeds limit."""
        import types

        mock_litellm = types.SimpleNamespace(
            get_model_info=lambda m: {"max_input_tokens": 100},
            token_counter=lambda model, messages: 200,
        )

        ok, err = paper_mod._check_context_limit([{"role": "user", "content": "x" * 1000}], "gpt-4o", mock_litellm)
        assert ok is False
        assert err is not None and "exceeds model context limit" in err

    def test_uses_fallback_when_litellm_fails(self):
        """Should use fallback context window when litellm.get_model_info fails."""
        import types

        def raise_error(m):
            raise Exception("Model not found")

        mock_litellm = types.SimpleNamespace(
            get_model_info=raise_error,
            token_counter=lambda model, messages: 1000,
        )

        # gpt-4o has 128k in fallback map
        ok, err = paper_mod._check_context_limit([{"role": "user", "content": "test"}], "gpt-4o", mock_litellm)
        assert ok is True
        assert err is None

    def test_estimates_tokens_when_counter_fails(self):
        """Should estimate tokens from chars when token_counter fails."""
        import types

        def raise_error(*args, **kwargs):
            raise Exception("Token counter failed")

        mock_litellm = types.SimpleNamespace(
            get_model_info=lambda m: {"max_input_tokens": 100},
            token_counter=raise_error,
        )

        # 1000 chars ≈ 250 tokens, which exceeds 100 limit
        ok, err = paper_mod._check_context_limit([{"role": "user", "content": "x" * 1000}], "gpt-4o", mock_litellm)
        assert ok is False
        assert err is not None and "exceeds" in err


class TestGenerateSimpleTldr:
    """Tests for generate_simple_tldr function."""

    def test_generate_simple_tldr_with_abstract(self):
        meta = {
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer.",
        }
        tldr = paper_mod.generate_simple_tldr(meta)
        assert (
            "Attention Is All You Need: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
            in tldr
        )

    def test_generate_simple_tldr_no_abstract(self):
        meta = {"title": "Test Title", "abstract": ""}
        tldr = paper_mod.generate_simple_tldr(meta)
        assert tldr == "Test Title"


class TestExtractNameFromTitle:
    """Tests for _extract_name_from_title helper."""

    def test_extracts_colon_prefix(self):
        assert paper_mod._extract_name_from_title("NeRF: Representing Scenes") == "nerf"

    def test_extracts_multi_word_prefix(self):
        assert paper_mod._extract_name_from_title("Instant NGP: Fast Training") == "instant-ngp"

    def test_returns_none_for_no_colon(self):
        assert paper_mod._extract_name_from_title("Attention Is All You Need") is None

    def test_returns_none_for_long_prefix(self):
        # More than 3 words should be rejected
        result = paper_mod._extract_name_from_title("This Is A Very Long Prefix: And Then The Rest")
        assert result is None

    def test_handles_special_characters(self):
        assert paper_mod._extract_name_from_title("BERT++: Better BERT") == "bert"

    def test_handles_parentheses(self):
        assert paper_mod._extract_name_from_title("GPT (Generative): A Model") == "gpt-generative"


class TestExtractTitleAndNameFromPdf:
    """Tests for extract_title_and_name_from_pdf combined extraction."""

    def _make_pdf(self, tmp_path: Path) -> Path:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        return pdf

    def _mock_fitz(self, monkeypatch, page_text: str):
        import sys

        class MockPage:
            def get_text(self):
                return page_text

        class MockDoc:
            def __init__(self, *_a, **_kw):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, _i):
                return MockPage()

            def close(self):
                pass

        mock_fitz = type("MockFitz", (), {"open": MockDoc})()
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    def test_parses_title_and_name(self, tmp_path: Path, monkeypatch):
        pdf = self._make_pdf(tmp_path)
        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: True)
        self._mock_fitz(monkeypatch, "Attention Is All You Need\nAbstract...")
        monkeypatch.setattr(
            paper_mod,
            "_run_llm",
            lambda prompt, *, purpose, model=None: "TITLE: Attention Is All You Need\nNAME: transformer",
        )
        title, name = paper_mod.extract_title_and_name_from_pdf(pdf)
        assert title == "Attention Is All You Need"
        assert name == "transformer"

    def test_returns_none_on_llm_failure(self, tmp_path: Path, monkeypatch):
        pdf = self._make_pdf(tmp_path)
        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: True)
        self._mock_fitz(monkeypatch, "Some text")
        monkeypatch.setattr(paper_mod, "_run_llm", lambda prompt, *, purpose, model=None: None)
        title, name = paper_mod.extract_title_and_name_from_pdf(pdf)
        assert title is None
        assert name is None

    def test_returns_none_when_litellm_unavailable(self, tmp_path: Path, monkeypatch):
        pdf = self._make_pdf(tmp_path)
        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: False)
        title, name = paper_mod.extract_title_and_name_from_pdf(pdf)
        assert title is None
        assert name is None

    def test_name_none_when_too_short(self, tmp_path: Path, monkeypatch):
        pdf = self._make_pdf(tmp_path)
        monkeypatch.setattr(paper_mod, "_litellm_available", lambda: True)
        self._mock_fitz(monkeypatch, "Some Title\nAbstract...")
        monkeypatch.setattr(paper_mod, "_run_llm", lambda prompt, *, purpose, model=None: "TITLE: Some Title\nNAME: ab")
        title, name = paper_mod.extract_title_and_name_from_pdf(pdf)
        assert title == "Some Title"
        assert name is None  # "ab" is < 3 chars


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


class TestSearchArxivByTitle:
    """Unit tests for search_arxiv_by_title with mocked arxiv library."""

    def test_returns_sorted_results_by_similarity(self, monkeypatch):
        """Test that results are sorted by similarity score."""
        from datetime import datetime
        from unittest.mock import MagicMock

        arxiv = pytest.importorskip("arxiv")

        # Create mock papers with varying similarity to "Attention Is All You Need"
        mock_papers = []
        for title, entry_id in [
            ("Neural Networks for Pattern Recognition", "https://arxiv.org/abs/1234.56789"),
            ("Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
            ("Self-Attention Mechanisms", "https://arxiv.org/abs/2001.00001"),
        ]:
            mock_paper = MagicMock()
            mock_paper.title = title
            mock_paper.entry_id = entry_id
            mock_paper.authors = [MagicMock()]
            mock_paper.authors[0].name = "Test Author"
            mock_paper.published = datetime(2023, 1, 1)
            mock_papers.append(mock_paper)

        mock_client = MagicMock()
        mock_client.results.return_value = iter(mock_papers)

        monkeypatch.setattr(arxiv, "Search", lambda query, max_results, sort_by: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        results = paper_mod.search_arxiv_by_title("Attention Is All You Need")

        # Should be sorted by similarity (exact match first)
        assert len(results) == 3
        assert results[0]["title"] == "Attention Is All You Need"
        assert results[0]["similarity"] == 1.0
        assert results[0]["arxiv_id"] == "1706.03762"

    def test_strips_version_suffix_from_arxiv_id(self, monkeypatch):
        """Test that version suffixes (v1, v2) are stripped from arXiv IDs."""
        from datetime import datetime
        from unittest.mock import MagicMock

        arxiv = pytest.importorskip("arxiv")

        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.entry_id = "https://arxiv.org/abs/2301.00001v3"
        mock_paper.authors = []
        mock_paper.published = datetime(2023, 1, 1)

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        monkeypatch.setattr(arxiv, "Search", lambda query, max_results, sort_by: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        results = paper_mod.search_arxiv_by_title("Test Paper")

        assert results[0]["arxiv_id"] == "2301.00001"  # No v3 suffix

    def test_returns_empty_list_for_no_results(self, monkeypatch):
        """Test that empty list is returned when no papers match."""
        from unittest.mock import MagicMock

        arxiv = pytest.importorskip("arxiv")

        mock_client = MagicMock()
        mock_client.results.return_value = iter([])

        monkeypatch.setattr(arxiv, "Search", lambda query, max_results, sort_by: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        results = paper_mod.search_arxiv_by_title("Nonexistent Paper Title XYZ123")

        assert results == []

    def test_limits_authors_to_five(self, monkeypatch):
        """Test that only first 5 authors are included in results."""
        from datetime import datetime
        from unittest.mock import MagicMock

        arxiv = pytest.importorskip("arxiv")

        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.entry_id = "https://arxiv.org/abs/2301.00001"
        mock_paper.authors = [MagicMock() for _ in range(10)]
        for i, author in enumerate(mock_paper.authors):
            author.name = f"Author {i}"
        mock_paper.published = datetime(2023, 1, 1)

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])

        monkeypatch.setattr(arxiv, "Search", lambda query, max_results, sort_by: MagicMock())
        monkeypatch.setattr(arxiv, "Client", lambda: mock_client)

        results = paper_mod.search_arxiv_by_title("Test Paper")

        assert len(results[0]["authors"]) == 5
        assert results[0]["authors"][4] == "Author 4"


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
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found", response=mock_response)

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


class TestDownloadSourceTempFileCleanup:
    """Tests for temp file cleanup in download_source (Finding 2)."""

    def test_temp_file_cleaned_up_on_tarfile_open_error(self, tmp_path, monkeypatch):
        """Temp file should be cleaned up even when tarfile.open raises OSError."""
        import requests

        # Create content that will pass size checks but fail tarfile.open
        bad_content = b"not a valid tar or gzip file at all" * 100

        class MockResponse:
            def __init__(self):
                self.content = bad_content
                self.status_code = 200
                self.headers: dict[str, str] = {}

            def raise_for_status(self):
                pass

        monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: MockResponse())

        # Track temp files created
        created_temps: list[Path] = []
        original_named_temp = tempfile.NamedTemporaryFile

        def tracking_temp(*args, **kwargs):
            f = original_named_temp(*args, **kwargs)
            created_temps.append(Path(f.name))
            return f

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking_temp)

        # Force tarfile.open to raise OSError (not tarfile.ReadError, so it propagates)
        monkeypatch.setattr(tarfile, "open", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk error")))

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        # OSError propagates, but the finally block should still clean up the temp file
        with pytest.raises(OSError, match="disk error"):
            paper_mod.download_source("1234.56789", paper_dir)

        # The temp file should have been cleaned up by the finally block
        assert len(created_temps) == 1
        assert not created_temps[0].exists(), "Temp file was not cleaned up after OSError"


class TestSetupDebugLogging:
    """Tests for duplicate handler prevention in _setup_debug_logging (Finding 5)."""

    def test_no_duplicate_handlers_on_repeated_calls(self):
        """Calling _setup_debug_logging twice should not add duplicate handlers."""
        from paperpipe.output import _debug_logger, _setup_debug_logging

        # Count non-NullHandler handlers before
        initial_count = sum(1 for h in _debug_logger.handlers if not isinstance(h, logging.NullHandler))

        _setup_debug_logging()
        after_first = sum(1 for h in _debug_logger.handlers if not isinstance(h, logging.NullHandler))

        _setup_debug_logging()
        after_second = sum(1 for h in _debug_logger.handlers if not isinstance(h, logging.NullHandler))

        # Should have added exactly one handler, regardless of how many times called
        assert after_first == max(initial_count, 1)
        assert after_second == after_first

    def test_replaces_stale_handler(self):
        """A second call should replace the handler, allowing recovery from closed streams."""
        import io

        from paperpipe.output import _debug_logger, _setup_debug_logging

        # Remove existing non-NullHandlers for a clean test
        for h in _debug_logger.handlers[:]:
            if not isinstance(h, logging.NullHandler):
                _debug_logger.removeHandler(h)

        # Simulate a stale handler pointing to a closed stream
        closed_stream = io.StringIO()
        closed_stream.close()
        stale_handler = logging.StreamHandler(closed_stream)
        _debug_logger.addHandler(stale_handler)

        # _setup_debug_logging should replace the stale handler with a fresh one
        _setup_debug_logging()

        non_null = [h for h in _debug_logger.handlers if not isinstance(h, logging.NullHandler)]
        assert len(non_null) == 1
        assert non_null[0] is not stale_handler, "Stale handler was not replaced"


class TestDownloadSourceSizeLimits:
    """Tests for download size limits (Fix #5b: bounded download size)."""

    def test_oversized_content_length_rejected(self, tmp_path, monkeypatch):
        """Content-Length header exceeding limit should cause download to be skipped."""
        from unittest.mock import MagicMock

        import requests

        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": str(600 * 1024 * 1024)}  # 600 MB
        mock_response.content = b"small"
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1234.56789", paper_dir)
        assert result is None

    def test_malformed_content_length_ignored(self, tmp_path, monkeypatch):
        """Malformed Content-Length header should not crash; download proceeds normally."""
        from unittest.mock import MagicMock

        import requests

        # Monkeypatch to small limit so the body size check triggers
        monkeypatch.setattr(paper_mod, "_MAX_DOWNLOAD_SIZE", 100)

        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "invalid"}
        mock_response.content = b"x" * 200  # exceeds monkeypatched limit
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        # Should not raise ValueError; falls through to body size check
        result = paperpipe.download_source("1234.56789", paper_dir)
        assert result is None

    def test_oversized_content_rejected(self, tmp_path, monkeypatch):
        """Actual content exceeding limit should cause download to be skipped."""
        from unittest.mock import MagicMock

        import requests

        # Simulate content that exceeds the limit (we'll monkeypatch the constant for test speed)
        monkeypatch.setattr(paper_mod, "_MAX_DOWNLOAD_SIZE", 100)

        mock_response = MagicMock()
        mock_response.headers = {}  # No Content-Length
        mock_response.content = b"x" * 200  # 200 bytes > 100 byte limit
        mock_response.raise_for_status = MagicMock()

        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        paper_dir = tmp_path / "test-paper"
        paper_dir.mkdir()

        result = paperpipe.download_source("1234.56789", paper_dir)
        assert result is None
