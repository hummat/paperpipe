"""Tests for figure extraction from LaTeX and PDF sources."""

import io
import tarfile
from pathlib import Path

import pytest

from paperpipe import paper as paper_mod


class TestExtractFiguresFromLatex:
    """Tests for _extract_figures_from_latex function."""

    def test_extracts_figure_with_explicit_extension(self, tmp_path):
        """Test extraction of figure with explicit file extension."""
        # Create a mock tarball with a figure and tex referencing it
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a PNG file
            png_data = b"\x89PNG\r\n\x1a\n" + b"fake png data"
            png_info = tarfile.TarInfo(name="figure1.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            # Add tex file referencing the figure
            tex_content = r"\begin{document}\includegraphics{figure1.png}\end{document}"
            tex_info = tarfile.TarInfo(name="paper.tex")
            tex_info.size = len(tex_content)
            tar.addfile(tex_info, io.BytesIO(tex_content.encode()))

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        # Extract figures
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 1
        assert (paper_dir / "figures" / "figure1.png").exists()
        assert (paper_dir / "figures" / "figure1.png").read_bytes().startswith(b"\x89PNG")

    def test_extracts_figure_without_extension(self, tmp_path):
        """Test extraction when LaTeX omits file extension."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a PNG file
            png_data = b"\x89PNG\r\n\x1a\n" + b"fake png data"
            png_info = tarfile.TarInfo(name="diagram.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            # Tex references without extension (LaTeX allows this)
            tex_content = r"\begin{document}\includegraphics{diagram}\end{document}"

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 1
        assert (paper_dir / "figures" / "diagram.png").exists()

    def test_extracts_figure_with_options(self, tmp_path):
        """Test extraction with includegraphics options."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            pdf_data = b"%PDF-1.4\nfake pdf data"
            pdf_info = tarfile.TarInfo(name="plot.pdf")
            pdf_info.size = len(pdf_data)
            tar.addfile(pdf_info, io.BytesIO(pdf_data))

            # Tex with options
            tex_content = r"\begin{document}\includegraphics[width=0.5\textwidth]{plot.pdf}\end{document}"

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 1
        assert (paper_dir / "figures" / "plot.pdf").exists()

    def test_extracts_figure_from_subdirectory(self, tmp_path):
        """Test extraction when figure is in a subdirectory."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            jpg_data = b"\xff\xd8\xff\xe0fake jpeg data"
            jpg_info = tarfile.TarInfo(name="figures/architecture.jpg")
            jpg_info.size = len(jpg_data)
            tar.addfile(jpg_info, io.BytesIO(jpg_data))

            tex_content = r"\begin{document}\includegraphics{figures/architecture.jpg}\end{document}"

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 1
        assert (paper_dir / "figures" / "architecture.jpg").exists()

    def test_extracts_multiple_figures(self, tmp_path):
        """Test extraction of multiple figures."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add multiple figures
            for i, ext in enumerate(["png", "pdf", "jpg"], 1):
                data = f"fake {ext} data".encode()
                info = tarfile.TarInfo(name=f"fig{i}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            tex_content = r"""
            \begin{document}
            \includegraphics{fig1.png}
            \includegraphics{fig2.pdf}
            \includegraphics[scale=0.5]{fig3.jpg}
            \end{document}
            """

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 3
        assert (paper_dir / "figures" / "fig1.png").exists()
        assert (paper_dir / "figures" / "fig2.pdf").exists()
        assert (paper_dir / "figures" / "fig3.jpg").exists()

    def test_returns_zero_when_no_includegraphics(self, tmp_path):
        """Test function returns 0 when no figures referenced."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a figure but tex doesn't reference it
            png_data = b"\x89PNG\r\n\x1a\n"
            png_info = tarfile.TarInfo(name="unused.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            tex_content = r"\begin{document}No figures here\end{document}"

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 0
        # figures directory should not be created when no figures extracted
        assert not (paper_dir / "figures").exists()

    def test_skips_nonexistent_figure(self, tmp_path):
        """Test graceful handling when referenced figure doesn't exist in tarball."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add one figure
            png_data = b"\x89PNG\r\n\x1a\n"
            png_info = tarfile.TarInfo(name="exists.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            # Tex references both existing and missing figure
            tex_content = r"""
            \begin{document}
            \includegraphics{exists.png}
            \includegraphics{missing.png}
            \end{document}
            """

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        # Should extract the one that exists
        assert count == 1
        assert (paper_dir / "figures" / "exists.png").exists()
        assert not (paper_dir / "figures" / "missing.png").exists()

    def test_handles_tarball_with_directory_prefix(self, tmp_path):
        """Test extraction when tarball has directory prefix (real arXiv behavior)."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Real arXiv tarballs often have a prefix directory
            png_data = b"\x89PNG\r\n\x1a\n"
            png_info = tarfile.TarInfo(name="arxiv_submission/image.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            # Tex references without prefix
            tex_content = r"\begin{document}\includegraphics{image.png}\end{document}"

        tar_buffer.seek(0)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            count = paper_mod._extract_figures_from_latex(tex_content, tar, paper_dir)

        assert count == 1
        # Should use basename for output
        assert (paper_dir / "figures" / "image.png").exists()


class TestExtractFiguresFromPdf:
    """Tests for _extract_figures_from_pdf function."""

    def test_returns_zero_when_pymupdf_unavailable(self, tmp_path, monkeypatch):
        """Test graceful degradation when PyMuPDF not installed."""
        # Mock fitz module to not exist
        import sys

        # Remove fitz from sys.modules if it exists
        monkeypatch.setitem(sys.modules, "fitz", None)

        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()
        pdf_path = paper_dir / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake pdf")

        # Should return 0 and not crash
        count = paper_mod._extract_figures_from_pdf(pdf_path, paper_dir)
        assert count == 0
        assert not (paper_dir / "figures").exists()

    @pytest.mark.skipif(
        not hasattr(paper_mod, "fitz") and paper_mod.__dict__.get("fitz") is None, reason="PyMuPDF not installed"
    )
    def test_extracts_images_from_pdf(self, tmp_path, monkeypatch):
        """Test basic PDF image extraction with mocked PyMuPDF."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        # Create a minimal PDF with an image (mocked)
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()
        pdf_path = paper_dir / "paper.pdf"

        # Mock PyMuPDF to return a fake image
        class MockImage:
            def __init__(self):
                self.image = b"fake image data that is longer than 1KB" + b"x" * 1024
                self.ext = "png"

        class MockPage:
            def get_images(self):
                return [(1, 0, 0, 0, 0, 0, 0)]  # (xref, ...)

        class MockDoc:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return MockPage()

            def extract_image(self, xref):
                return {"image": b"fake image data" + b"x" * 1024, "ext": "png"}

            def close(self):
                pass

        monkeypatch.setattr(paper_mod, "fitz", type("fitz", (), {"open": MockDoc}))

        count = paper_mod._extract_figures_from_pdf(pdf_path, paper_dir)

        # With mock, should extract 1 image
        assert count == 1
        assert (paper_dir / "figures").exists()

    def test_handles_pdf_open_failure(self, tmp_path, monkeypatch):
        """Test returns 0 when PDF cannot be opened."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()
        pdf_path = paper_dir / "corrupt.pdf"
        pdf_path.write_bytes(b"not a real pdf")

        # Should handle exception and return 0
        count = paper_mod._extract_figures_from_pdf(pdf_path, paper_dir)
        assert count == 0


class TestFigureExtractionIntegration:
    """Integration tests for figure extraction in add command."""

    def test_download_source_extracts_figures_by_default(self, tmp_path, monkeypatch):
        """Test that download_source extracts figures when extract_figures=True."""
        import requests

        # Create a mock tarball with figure
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            png_data = b"\x89PNG\r\n\x1a\n"
            png_info = tarfile.TarInfo(name="fig.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            tex_content = r"\begin{document}\includegraphics{fig.png}\end{document}"
            tex_info = tarfile.TarInfo(name="paper.tex")
            tex_info.size = len(tex_content)
            tar.addfile(tex_info, io.BytesIO(tex_content.encode()))

        tar_bytes = tar_buffer.getvalue()

        # Mock requests
        class MockResponse:
            def __init__(self):
                self.content = tar_bytes
                self.status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        # Call download_source with extract_figures=True (default)
        result = paper_mod.download_source("1234.5678", paper_dir, extract_figures=True)

        assert result is not None
        assert (paper_dir / "source.tex").exists()
        assert (paper_dir / "figures" / "fig.png").exists()

    def test_download_source_skips_figures_when_disabled(self, tmp_path, monkeypatch):
        """Test that download_source skips figures when extract_figures=False."""
        import requests

        # Create a mock tarball with figure
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            png_data = b"\x89PNG\r\n\x1a\n"
            png_info = tarfile.TarInfo(name="fig.png")
            png_info.size = len(png_data)
            tar.addfile(png_info, io.BytesIO(png_data))

            tex_content = r"\begin{document}\includegraphics{fig.png}\end{document}"
            tex_info = tarfile.TarInfo(name="paper.tex")
            tex_info.size = len(tex_content)
            tar.addfile(tex_info, io.BytesIO(tex_content.encode()))

        tar_bytes = tar_buffer.getvalue()

        class MockResponse:
            def __init__(self):
                self.content = tar_bytes
                self.status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir()

        # Call download_source with extract_figures=False
        result = paper_mod.download_source("1234.5678", paper_dir, extract_figures=False)

        assert result is not None
        assert (paper_dir / "source.tex").exists()
        # Figures should NOT be extracted
        assert not (paper_dir / "figures").exists()
