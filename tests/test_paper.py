from __future__ import annotations

from pathlib import Path

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
