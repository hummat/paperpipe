"""Tests for paperpipe/cli/papers.py (add, notes, regenerate, remove, show commands)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
import requests
from click.testing import CliRunner

import paperpipe
import paperpipe.core as core
import paperpipe.paper as paper_mod

from .conftest import TEST_ARXIV_ID, cli_mod


class TestNotesCommand:
    def test_notes_creates_file_and_prints(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "my-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "Test Paper"}))
        paperpipe.save_index({"my-paper": {"title": "Test Paper", "tags": [], "added": "now"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["notes", "my-paper", "--print"])
        assert result.exit_code == 0
        assert (paper_dir / "notes.md").exists()
        assert "# my-paper" in result.output


class TestShowCommand:
    def test_show_nonexistent(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["show", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

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
        result = runner.invoke(cli_mod.cli, ["show", "test-paper"])
        assert result.exit_code == 0
        assert "Test Paper Title" in result.output
        assert "2301.00001" in result.output
        assert "Alice" in result.output
        assert "ml" in result.output

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
        result = runner.invoke(cli_mod.cli, ["show", "test-paper", "--level", "eq"])
        assert result.exit_code == 0
        assert "# test-paper" in result.output
        assert "Content: equations" in result.output
        assert "E=mc^2" in result.output

    def test_show_tldr_stdout(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "T"}))
        (paper_dir / "tldr.md").write_text("Very short summary.")
        paperpipe.save_index({"test-paper": {"title": "T", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["show", "test-paper", "--level", "tldr"])
        assert result.exit_code == 0
        assert "Very short summary." in result.output

    def test_show_meta_includes_tldr(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "T"}))
        (paper_dir / "tldr.md").write_text("Short TLDR")
        paperpipe.save_index({"test-paper": {"title": "T", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["show", "test-paper"])
        assert result.exit_code == 0
        assert "- TL;DR: Short TLDR" in result.output

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
        result = runner.invoke(cli_mod.cli, ["show", "paper-a", "paper-b", "--level", "equations"])
        assert result.exit_code == 0
        assert "# paper-a" in result.output
        assert "# paper-b" in result.output
        assert "\n---\n" in result.output


class TestAddCommand:
    """Unit tests for the add command with mocked network calls."""

    def test_generate_auto_name_local_meta_uses_slug(self):
        # Local/meta-only papers should fall back to a stable title slug, not "unknown".
        meta = {"title": "Some Paper", "authors": [], "abstract": ""}
        assert paperpipe.generate_auto_name(meta, set(), use_llm=False) == "some-paper"

    def test_parse_authors_keeps_last_first_single_author(self):
        assert core._parse_authors("Smith, John") == ["Smith, John"]

    def test_parse_authors_semicolon_separated(self):
        assert core._parse_authors("Smith, John; Doe, Jane") == ["Smith, John", "Doe, Jane"]

    def test_parse_authors_multiple_comma_separated(self):
        assert core._parse_authors("Alice, Bob, Charlie") == ["Alice", "Bob", "Charlie"]

    def test_parse_authors_empty(self):
        assert core._parse_authors("") == []
        assert core._parse_authors(None) == []

    def test_format_title_short_truncates(self):
        long_title = "A" * 100
        result = core._format_title_short(long_title, max_len=60)
        assert len(result) == 63  # 60 chars + "..."
        assert result.endswith("...")

    def test_format_title_short_keeps_short(self):
        short_title = "Short Title"
        assert core._format_title_short(short_title) == short_title

    def test_slugify_title_basic(self):
        assert core._slugify_title("Hello World") == "hello-world"

    def test_slugify_title_empty(self):
        assert core._slugify_title("") == "paper"
        assert core._slugify_title("   ") == "paper"

    def test_slugify_title_truncates_long(self):
        long_title = "word " * 50
        result = core._slugify_title(long_title, max_len=30)
        assert len(result) <= 30

    def test_slugify_title_special_chars(self):
        assert core._slugify_title("Test: A 'Paper' with \"Quotes\"") == "test-a-paper-with-quotes"

    def test_looks_like_pdf_valid(self, tmp_path: Path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4\ntest content")
        assert core._looks_like_pdf(pdf) is True

    def test_looks_like_pdf_invalid(self, tmp_path: Path):
        txt = tmp_path / "test.txt"
        txt.write_text("not a pdf")
        assert core._looks_like_pdf(txt) is False

    def test_looks_like_pdf_missing_file(self, tmp_path: Path):
        missing = tmp_path / "missing.pdf"
        assert core._looks_like_pdf(missing) is False

    def test_add_rejects_unsafe_name_without_network(self, temp_db: Path, monkeypatch):
        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", lambda _: (_ for _ in ()).throw(AssertionError()))

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["add", "1706.03762", "--name", "../bad", "--no-llm"],
        )

        assert result.exit_code != 0
        assert "invalid paper name" in result.output.lower()

    def test_add_local_pdf_ingests_and_indexes(self, temp_db: Path):
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
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
        assert (paper_dir / "tldr.md").exists()
        assert "Some Paper" in (paper_dir / "tldr.md").read_text()

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

    def test_add_local_pdf_requires_title_with_no_llm(self, temp_db: Path):
        """--title is required when --no-llm is specified."""
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--pdf", str(pdf_path), "--no-llm"])
        assert result.exit_code != 0
        assert "--title" in result.output
        assert "--no-llm" in result.output

    def test_add_local_pdf_extracts_title_with_llm(self, temp_db: Path, monkeypatch):
        """In LLM mode, title can be extracted from PDF automatically."""
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        # Mock the title extraction function
        monkeypatch.setattr(paper_mod, "extract_title_from_pdf", lambda _: "Extracted Paper Title")

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                str(pdf_path),
                # No --title provided
                "--no-llm",  # But we still skip LLM for summary/equations
            ],
        )
        # Should fail because --no-llm requires --title
        assert result.exit_code != 0

        # Now test without --no-llm (LLM mode)
        # We need to mock generate_llm_content too since it will be called
        monkeypatch.setattr(
            paper_mod,
            "generate_llm_content",
            lambda *args, **kwargs: ("Summary", "Equations", ["tag"], "TL;DR"),
        )

        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                str(pdf_path),
                # No --title - will be extracted
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Extracted Paper Title" in result.output or "extracted-paper-title" in result.output

    def test_add_local_pdf_rejects_non_pdf(self, temp_db: Path):
        bad_path = temp_db / "not-a-pdf.txt"
        bad_path.write_text("hello")

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["add", "--pdf", str(bad_path), "--title", "Some Paper", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "does not look like a pdf" in result.output.lower()

    def test_add_local_pdf_rejects_invalid_year(self, temp_db: Path):
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
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
            cli_mod.cli,
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
        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", lambda _: (_ for _ in ()).throw(AssertionError()))

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["add", "1706.03762", "--name", "test-paper", "--no-llm"],
        )

        assert result.exit_code == 0
        assert "already added" in result.output.lower()

    def test_add_detects_duplicate_in_index(self, temp_db: Path, monkeypatch):
        """Test that add command detects duplicate name in index even without directory."""
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

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
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

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", lambda _: (_ for _ in ()).throw(AssertionError()))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "1706.03762v2", "--no-llm"])

        assert result.exit_code == 0
        assert "already added" in result.output.lower()
        # Still only one directory
        assert sorted([p.name for p in (temp_db / "papers").iterdir() if p.is_dir()]) == ["attention"]

    def test_add_duplicate_arxiv_duplicate_flag_creates_second_copy(self, temp_db: Path, monkeypatch):
        """Test --duplicate keeps the old behavior of adding another copy."""
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

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)

        def fake_download_pdf(_arxiv_id: str, dest: Path):
            dest.write_bytes(b"%PDF")
            return True

        monkeypatch.setattr(paper_mod, "download_pdf", fake_download_pdf)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "1706.03762", "--duplicate", "--no-llm"])

        assert result.exit_code == 0
        assert "Added: 1706_03762-2" in result.output
        assert (temp_db / "papers" / "1706_03762-2" / "meta.json").exists()

    def test_add_duplicate_arxiv_update_refreshes_existing(self, temp_db: Path, monkeypatch):
        """Test --update refreshes an existing paper in-place."""
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

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)

        def fake_download_pdf(_arxiv_id: str, dest: Path):
            dest.write_bytes(b"%PDF")
            return True

        def fake_download_source(_arxiv_id: str, pdir: Path, **kwargs):
            tex = r"\begin{document}\begin{equation}E=mc^2\end{equation}\end{document}"
            (pdir / "source.tex").write_text(tex)
            return tex

        monkeypatch.setattr(paper_mod, "download_pdf", fake_download_pdf)
        monkeypatch.setattr(paper_mod, "download_source", fake_download_source)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "1706.03762", "--update", "--name", name, "--no-llm"])

        assert result.exit_code == 0
        assert "Updated: attention" in result.output

        meta = json.loads((paper_dir / "meta.json").read_text())
        assert meta["title"] == "New Title"
        assert meta["added"] == "x"  # preserved
        assert "computer-vision" in meta["tags"]
        assert "old-tag" in meta["tags"]

    def test_add_figures_flag_enables_extraction(self, temp_db: Path, monkeypatch):
        """Test --figures flag enables figure extraction during add."""
        # Track whether extract_figures was passed as True
        extract_figures_args = []

        def mock_fetch(arxiv_id: str):
            return {
                "arxiv_id": arxiv_id,
                "title": "Test Paper",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "published": datetime(2023, 1, 1).isoformat(),
                "updated": datetime(2023, 1, 1).isoformat(),
                "doi": None,
                "journal_ref": None,
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)

        def fake_download_pdf(_arxiv_id: str, dest: Path):
            dest.write_bytes(b"%PDF")
            return True

        def fake_download_source(_arxiv_id: str, pdir: Path, *, extract_figures=False):
            extract_figures_args.append(extract_figures)
            tex = r"\begin{document}\includegraphics{fig.png}\end{document}"
            (pdir / "source.tex").write_text(tex)
            return tex

        monkeypatch.setattr(paper_mod, "download_pdf", fake_download_pdf)
        monkeypatch.setattr(paper_mod, "download_source", fake_download_source)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "1706.03762", "--figures", "--no-llm"])

        assert result.exit_code == 0
        # Verify extract_figures=True was passed to download_source
        assert extract_figures_args == [True]

    def test_add_without_figures_flag_skips_extraction(self, temp_db: Path, monkeypatch):
        """Test that figure extraction is skipped by default (without --figures flag)."""
        # Track whether extract_figures was passed
        extract_figures_args = []

        def mock_fetch(arxiv_id: str):
            return {
                "arxiv_id": arxiv_id,
                "title": "Test Paper No Figures",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "published": datetime(2023, 1, 1).isoformat(),
                "updated": datetime(2023, 1, 1).isoformat(),
                "doi": None,
                "journal_ref": None,
                "pdf_url": "https://arxiv.org/pdf/1706.03763",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)

        def fake_download_pdf(_arxiv_id: str, dest: Path):
            dest.write_bytes(b"%PDF")
            return True

        def fake_download_source(_arxiv_id: str, pdir: Path, *, extract_figures=False):
            extract_figures_args.append(extract_figures)
            tex = r"\begin{document}\includegraphics{fig.png}\end{document}"
            (pdir / "source.tex").write_text(tex)
            return tex

        monkeypatch.setattr(paper_mod, "download_pdf", fake_download_pdf)
        monkeypatch.setattr(paper_mod, "download_source", fake_download_source)

        runner = CliRunner()
        # No --figures flag = figures extraction disabled
        result = runner.invoke(cli_mod.cli, ["add", "1706.03763", "--no-llm"])

        assert result.exit_code == 0
        # Verify extract_figures=False was passed to download_source (default)
        assert extract_figures_args == [False]

    def test_add_from_file_json(self, temp_db: Path, monkeypatch):
        """Test adding papers from a JSON file (export format)."""
        papers_json = temp_db / "papers.json"
        papers_json.write_text(
            json.dumps(
                {
                    "paper1": {"arxiv_id": "1111.1111", "tags": ["tag1"]},
                    "paper2": {"arxiv_id": "2222.2222", "tags": ["tag2", "tag3"]},
                }
            )
        )

        # Mock fetch_metadata and download
        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2023-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2023-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_json), "--no-llm"])
        assert result.exit_code == 0, result.output
        assert "Importing 2 papers" in result.output
        assert "Added: paper1" in result.output
        assert "Added: paper2" in result.output

        index = paperpipe.load_index()
        assert "paper1" in index
        assert "tag1" in index["paper1"]["tags"]
        assert "paper2" in index
        assert "tag2" in index["paper2"]["tags"]

    def test_add_from_file_json_list_with_tags(self, temp_db: Path, monkeypatch):
        """Test adding papers from a JSON list of objects with tags (common export format)."""
        papers_json = temp_db / "papers.json"
        # JSON list format with tags as arrays (the natural JSON shape)
        papers_json.write_text(
            json.dumps(
                [
                    {"arxiv_id": "1111.1111", "name": "paper-one", "tags": ["ml", "transformers"]},
                    {"arxiv_id": "2222.2222", "tags": ["nlp"]},  # name is optional
                    {"arxiv_id": "3333.3333"},  # tags is optional
                ]
            )
        )

        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2023-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2023-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_json), "--no-llm"])
        assert result.exit_code == 0, result.output

        index = paperpipe.load_index()
        assert "paper-one" in index
        assert "ml" in index["paper-one"]["tags"]
        assert "transformers" in index["paper-one"]["tags"]
        # Second paper should have its tags
        paper2_name = [k for k in index if index[k].get("arxiv_id") == "2222.2222"][0]
        assert "nlp" in index[paper2_name]["tags"]

    def test_add_from_file_json_list_merges_cli_tags(self, temp_db: Path, monkeypatch):
        """Test that CLI --tags are merged with JSON list tags."""
        papers_json = temp_db / "papers.json"
        papers_json.write_text(json.dumps([{"arxiv_id": "1111.1111", "tags": ["from-json"]}]))

        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2023-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2023-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_json), "--tags", "from-cli", "--no-llm"])
        assert result.exit_code == 0, result.output

        index = paperpipe.load_index()
        paper_name = list(index.keys())[0]
        # Both JSON tags and CLI tags should be present
        assert "from-json" in index[paper_name]["tags"]
        assert "from-cli" in index[paper_name]["tags"]

    def test_add_from_file_text(self, temp_db: Path, monkeypatch):
        """Test adding papers from a text file (one ID per line)."""
        papers_txt = temp_db / "papers.txt"
        papers_txt.write_text("1111.1111\n2222.2222\n# Comment")

        # Mock fetch_metadata and download
        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2023-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2023-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_txt), "--no-llm", "--tags", "batch"])
        assert result.exit_code == 0, result.output
        assert "Importing 2 papers" in result.output

        index = paperpipe.load_index()
        assert len(index) == 2
        for info in index.values():
            assert "batch" in info["tags"]

    def test_add_from_file_bibtex(self, temp_db: Path, monkeypatch):
        """Test adding papers from a BibTeX file."""
        papers_bib = temp_db / "papers.bib"
        papers_bib.write_text("""
@article{test2023,
  title={Test Paper Title},
  author={Doe, John and Smith, Jane},
  journal={Test Journal},
  year={2023},
  eprint={1111.1111},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}

@article{another2023,
  title={Another Test Paper},
  author={Brown, Alice},
  journal={Another Journal},
  year={2023},
  doi={10.1234/567890},
  url={https://arxiv.org/abs/2222.2222}
}
""")

        # Mock fetch_metadata and download
        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2023-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2023-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_bib), "--no-llm"])
        # Skip test if bibtexparser is not installed
        if "bibtexparser not installed" in result.output:
            pytest.skip("bibtexparser not installed")
        assert result.exit_code == 0, result.output
        # Should find at least one paper with arXiv ID
        assert "Added:" in result.output or "already added" in result.output.lower()
        # Should show progress message
        assert "Adding" in result.output

    def test_add_from_file_bibtex_malformed(self, temp_db: Path, monkeypatch):
        """Test error handling for malformed BibTeX files."""
        papers_bib = temp_db / "papers.bib"
        # Write malformed BibTeX content
        papers_bib.write_text("""
@article{test2023,
  title={Test Paper Title},
  author={Doe, John and Smith, Jane},
  journal={Test Journal,
  year={2023},
  eprint={1111.1111}
  # Missing closing brace
""")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_bib), "--no-llm"])
        # Skip test if bibtexparser is not installed
        if "bibtexparser not installed" in result.output:
            pytest.skip("bibtexparser not installed")
        # Should handle parsing errors gracefully
        assert result.exit_code != 0 or "Failed to parse BibTeX file" in result.output

    def test_add_from_file_bibtex_missing_fields(self, temp_db: Path, monkeypatch):
        """Test BibTeX entries with missing required fields."""
        papers_bib = temp_db / "papers.bib"
        # BibTeX entry with no arXiv ID or DOI
        papers_bib.write_text("""
@article{test2023,
  title={Test Paper Title},
  author={Doe, John and Smith, Jane},
  journal={Test Journal},
  year={2023}
}
""")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--from-file", str(papers_bib), "--no-llm"])
        # Skip test if bibtexparser is not installed
        if "bibtexparser not installed" in result.output:
            pytest.skip("bibtexparser not installed")
        # Should handle missing fields gracefully (no papers added, but no crash)
        assert result.exit_code == 0

    def test_add_semantic_scholar(self, temp_db: Path, monkeypatch):
        """Test adding papers from Semantic Scholar IDs."""

        # Mock Semantic Scholar API response
        def mock_semantic_scholar_request(url, params=None, timeout=None):
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self._json = {
                        "title": "Test Paper from Semantic Scholar",
                        "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
                        "abstract": "This is a test abstract.",
                        "year": 2023,
                        "venue": "Test Conference",
                        "url": "https://www.semanticscholar.org/paper/test-id",
                        "externalIds": {"ArXiv": "3333.3333", "DOI": "10.1234/test-doi"},
                    }

                def json(self):
                    return self._json

                def raise_for_status(self):
                    pass

            return MockResponse()

        monkeypatch.setattr("requests.get", mock_semantic_scholar_request)

        # Mock arXiv fetch
        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2023-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2023-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "https://www.semanticscholar.org/paper/test-id", "--no-llm"])
        assert result.exit_code == 0, result.output
        assert "Added:" in result.output or "already added" in result.output.lower()

    def test_add_semantic_scholar_api_error(self, temp_db: Path, monkeypatch):
        """Test error handling for Semantic Scholar API failures."""

        # Mock Semantic Scholar API response with 404 error
        def mock_semantic_scholar_request_404(url, params=None, timeout=None):
            class MockResponse:
                def __init__(self):
                    self.status_code = 404
                    self._json = {}

                def json(self):
                    return self._json

                def raise_for_status(self):
                    raise requests.exceptions.HTTPError("404 Not Found")

            return MockResponse()

        monkeypatch.setattr("requests.get", mock_semantic_scholar_request_404)
        monkeypatch.setattr("requests.exceptions.HTTPError", requests.exceptions.HTTPError)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "https://www.semanticscholar.org/paper/nonexistent", "--no-llm"])
        # Should handle API errors gracefully
        assert result.exit_code != 0

    def test_add_semantic_scholar_rate_limit(self, temp_db: Path, monkeypatch):
        """Test handling of Semantic Scholar API rate limiting."""

        # Mock Semantic Scholar API response with 429 error
        def mock_semantic_scholar_request_429(url, params=None, timeout=None):
            class MockResponse:
                def __init__(self):
                    self.status_code = 429
                    self._json = {}

                def json(self):
                    return self._json

                def raise_for_status(self):
                    raise requests.exceptions.HTTPError("429 Too Many Requests")

            return MockResponse()

        monkeypatch.setattr("requests.get", mock_semantic_scholar_request_429)
        monkeypatch.setattr("requests.exceptions.HTTPError", requests.exceptions.HTTPError)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "https://www.semanticscholar.org/paper/test-id", "--no-llm"])
        # Should handle rate limiting gracefully
        assert result.exit_code != 0

    def test_add_semantic_scholar_no_arxiv_id_fails(self, temp_db: Path, monkeypatch):
        """Test that S2 papers without arXiv IDs fail with non-zero exit code."""

        # Mock Semantic Scholar API response WITHOUT arXiv ID
        def mock_semantic_scholar_request(url, params=None, timeout=None):
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self._json = {
                        "title": "Non-arXiv Paper",
                        "authors": [{"name": "John Doe"}],
                        "abstract": "This paper is not on arXiv.",
                        "year": 2023,
                        "venue": "Some Journal",
                        "url": "https://www.semanticscholar.org/paper/no-arxiv-id",
                        "externalIds": {"DOI": "10.1234/no-arxiv"},  # No ArXiv key
                    }

                def json(self):
                    return self._json

                def raise_for_status(self):
                    pass

            return MockResponse()

        monkeypatch.setattr("requests.get", mock_semantic_scholar_request)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "https://www.semanticscholar.org/paper/no-arxiv-id", "--no-llm"])

        # Should fail with non-zero exit code
        assert result.exit_code != 0
        assert "does not have an arXiv ID" in result.output
        assert "No papers to add" in result.output

    def test_add_semantic_scholar_mixed_with_arxiv_reports_failures(self, temp_db: Path, monkeypatch):
        """Test that mixing S2 papers (some with, some without arXiv IDs) reports failures correctly."""

        call_count = [0]

        def mock_semantic_scholar_request(url, params=None, timeout=None):
            call_count[0] += 1

            class MockResponse:
                def __init__(self, has_arxiv: bool):
                    self.status_code = 200
                    external_ids = {"DOI": "10.1234/test"}
                    if has_arxiv:
                        external_ids["ArXiv"] = "2401.00001"
                    self._json = {
                        "title": "Test Paper",
                        "authors": [{"name": "John Doe"}],
                        "abstract": "Abstract",
                        "year": 2024,
                        "externalIds": external_ids,
                    }

                def json(self):
                    return self._json

                def raise_for_status(self):
                    pass

            # First call has arXiv ID, second doesn't
            return MockResponse(has_arxiv=(call_count[0] == 1))

        monkeypatch.setattr("requests.get", mock_semantic_scholar_request)

        # Mock arXiv fetch for the one that has an arXiv ID
        def mock_fetch(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "title": f"Title {arxiv_id}",
                "authors": [],
                "abstract": "Abstract",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published": "2024-01-01",
                "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
                "updated": "2024-01-01",
            }

        monkeypatch.setattr(paper_mod, "fetch_arxiv_metadata", mock_fetch)
        monkeypatch.setattr(paper_mod, "download_pdf", lambda *args: True)
        monkeypatch.setattr(paper_mod, "download_source", lambda *args, **kwargs: None)

        runner = CliRunner()
        # Add two S2 papers: first has arXiv ID, second doesn't
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "https://www.semanticscholar.org/paper/has-arxiv",
                "https://www.semanticscholar.org/paper/no-arxiv",
                "--no-llm",
            ],
        )

        # Should exit with failure due to the paper without arXiv ID
        assert result.exit_code != 0
        assert "does not have an arXiv ID" in result.output
        # Should still report the successful addition
        assert "added 1" in result.output
        assert "1 failed" in result.output

    def test_add_llm_flag_overrides_model(self, temp_db: Path, monkeypatch):
        """Test that --llm flag passes model to generate_llm_content."""
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        captured_model = []

        def mock_generate_llm_content(*args, **kwargs):
            captured_model.append(kwargs.get("model"))
            return ("Summary", "Equations", ["tag"], "TL;DR")

        monkeypatch.setattr(paper_mod, "generate_llm_content", mock_generate_llm_content)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                str(pdf_path),
                "--title",
                "Test Paper",
                "--llm",
                "gpt-4o-mini",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(captured_model) == 1
        assert captured_model[0] == "gpt-4o-mini"


class TestAddMultiplePapers:
    """Tests for adding multiple papers at once."""

    def test_add_name_with_multiple_papers_errors(self, temp_db: Path):
        """Test that --name errors when used with multiple papers."""
        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["add", "1706.03762", "2301.00001", "--name", "my-paper", "--no-llm"],
        )
        assert result.exit_code != 0
        assert "--name can only be used when adding a single paper" in result.output


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
        result = runner.invoke(cli_mod.cli, ["remove", f"https://arxiv.org/abs/{TEST_ARXIV_ID}", "--yes"])
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
        result = runner.invoke(cli_mod.cli, ["remove", TEST_ARXIV_ID, "--yes"])
        # Exit code 1 because the operation failed (ambiguous match)
        assert result.exit_code == 1
        assert "Multiple papers match arXiv ID" in result.output
        # Neither paper should be removed
        assert (papers_dir / "p1").exists()
        assert (papers_dir / "p2").exists()

    def test_remove_yes_flag_skips_confirmation(self, temp_db: Path):
        """Test that -y short flag skips confirmation prompt."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({"p1": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["remove", "p1", "-y"])
        assert result.exit_code == 0
        assert "Removed: p1" in result.output
        assert not (papers_dir / "p1").exists()

    def test_remove_without_yes_prompts(self, temp_db: Path):
        """Test that missing -y/--yes causes abort without input."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "authors": [], "abstract": ""})
        )
        paperpipe.save_index({"p1": {"arxiv_id": TEST_ARXIV_ID, "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        # Provide 'n' as input to decline confirmation
        result = runner.invoke(cli_mod.cli, ["remove", "p1"], input="n\n")
        assert result.exit_code == 1
        assert "Are you sure you want to remove these paper(s)?" in result.output
        # Paper should NOT be removed
        assert (papers_dir / "p1").exists()


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
        result = runner.invoke(cli_mod.cli, ["remove", "p1", "p2", "--yes"])
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
        result = runner.invoke(cli_mod.cli, ["remove", "p1", "nonexistent", "--yes"])
        # Exit code 1 because one failed
        assert result.exit_code == 1
        assert "Removed: p1" in result.output
        assert "not found" in result.output.lower()
        assert "1 failed" in result.output
        assert not (papers_dir / "p1").exists()


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
        result = runner.invoke(cli_mod.cli, ["regenerate", "--all", "--no-llm", "-o", "summary,equations,tags"])
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
        result = runner.invoke(cli_mod.cli, ["regenerate", "all", "--no-llm", "-o", "summary,equations"])
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
        result = runner.invoke(cli_mod.cli, ["regenerate", "all", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating all:" in result.output
        assert "Regenerating p2:" not in result.output

    def test_regenerate_all_fails_if_missing_metadata(self, temp_db: Path):
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["regenerate", "--all", "--no-llm"])
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
            cli_mod.cli,
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
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "--no-llm"])
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
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "--no-llm", "-o", "summary"])
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
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "-o", "invalid"])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower()

    def test_regenerate_extracts_figures_with_overwrite_flag(self, temp_db: Path, monkeypatch):
        """Test that regenerate extracts figures from PDF when --overwrite figures is passed."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("summary")
        (papers_dir / "p1" / "equations.md").write_text("equations")

        # Create a PDF
        pdf_path = papers_dir / "p1" / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake pdf")

        # Mock PyMuPDF module
        class MockPage:
            def get_images(self):
                return [(1, 0, 0, 0, 0, 0, 0)]

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

        mock_fitz = type("MockFitz", (), {"open": MockDoc})()
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        # Figure extraction requires --overwrite figures (opt-in)
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "--no-llm", "--overwrite", "figures"])
        assert result.exit_code == 0
        # Should extract figures and show warning about PDF extraction
        assert "Extracting figures from PDF" in result.output
        assert "source tarball not cached during add" in result.output
        assert (papers_dir / "p1" / "figures").exists()

    def test_regenerate_skips_figures_by_default(self, temp_db: Path, monkeypatch):
        """Test that regenerate skips figure extraction by default (opt-in only)."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("summary")
        (papers_dir / "p1" / "equations.md").write_text("equations")

        # Create a PDF
        pdf_path = papers_dir / "p1" / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake pdf")

        # Mock PyMuPDF - should NOT be called since figure extraction is opt-in
        extract_called = []

        class MockDoc:
            def __init__(self, *args, **kwargs):
                extract_called.append(True)

        mock_fitz = type("MockFitz", (), {"open": MockDoc})()
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        # No --overwrite figures = no figure extraction
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "--no-llm"])
        assert result.exit_code == 0
        # Should NOT extract figures (opt-in only)
        assert "Extracting figures" not in result.output
        # Mock should not have been called
        assert len(extract_called) == 0
        # No figures directory should be created
        assert not (papers_dir / "p1" / "figures").exists()

    def test_regenerate_overwrite_figures_forces_extraction(self, temp_db: Path, monkeypatch):
        """Test that --overwrite figures forces re-extraction even when figures exist."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("summary")
        (papers_dir / "p1" / "equations.md").write_text("equations")

        # Create figures directory with existing figure
        figures_dir = papers_dir / "p1" / "figures"
        figures_dir.mkdir()
        (figures_dir / "old.png").write_bytes(b"old figure")

        # Create a PDF
        pdf_path = papers_dir / "p1" / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake pdf")

        # Mock PyMuPDF module
        class MockPage:
            def get_images(self):
                return [(1, 0, 0, 0, 0, 0, 0)]

        class MockDoc:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return MockPage()

            def extract_image(self, xref):
                return {"image": b"new image data" + b"x" * 1024, "ext": "png"}

            def close(self):
                pass

        mock_fitz = type("MockFitz", (), {"open": MockDoc})()
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "--no-llm", "-o", "figures"])
        assert result.exit_code == 0
        # Should extract figures with warning
        assert "Extracting figures from PDF" in result.output
        assert (figures_dir / "figure_01.png").exists()
        # Old figures should be cleared (stale file removal)
        assert not (figures_dir / "old.png").exists(), "Stale figures should be cleared on overwrite"

    def test_regenerate_figures_warning_message(self, temp_db: Path, monkeypatch):
        """Test that warning mentions using 'papi add --update' for LaTeX extraction."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text("\\begin{equation}x=1\\end{equation}")
        (papers_dir / "p1" / "summary.md").write_text("summary")
        (papers_dir / "p1" / "equations.md").write_text("equations")

        # Create a PDF
        pdf_path = papers_dir / "p1" / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nfake pdf")

        # Mock PyMuPDF module
        class MockPage:
            def get_images(self):
                return [(1, 0, 0, 0, 0, 0, 0)]

        class MockDoc:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return MockPage()

            def extract_image(self, xref):
                return {"image": b"fake" + b"x" * 1024, "ext": "png"}

            def close(self):
                pass

        mock_fitz = type("MockFitz", (), {"open": MockDoc})()
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        # Need --overwrite figures to trigger figure extraction (opt-in)
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "--no-llm", "--overwrite", "figures"])
        assert result.exit_code == 0
        # Check for warning message
        assert "Extracting figures from PDF" in result.output
        assert "source tarball not cached" in result.output
        assert "papi add --update" in result.output

    def test_regenerate_llm_flag_overrides_model(self, temp_db: Path, monkeypatch):
        """Test that --llm flag passes model to generate_llm_content."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text(r"\begin{equation}x=1\end{equation}")
        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        captured_model = []

        def mock_generate_llm_content(*args, **kwargs):
            captured_model.append(kwargs.get("model"))
            return ("Summary", "Equations", ["tag"], "TL;DR")

        monkeypatch.setattr(paper_mod, "generate_llm_content", mock_generate_llm_content)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["regenerate", "p1", "--llm", "gpt-4o-mini", "-o", "summary"],
        )

        assert result.exit_code == 0, result.output
        assert len(captured_model) == 1
        assert captured_model[0] == "gpt-4o-mini"


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
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "p2", "--no-llm", "-o", "summary,equations"])
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
        result = runner.invoke(cli_mod.cli, ["regenerate", "p1", "nonexistent", "--no-llm", "-o", "summary"])
        # Exit code 1 because one failed
        assert result.exit_code == 1
        assert "Regenerating p1:" in result.output
        assert "not found" in result.output.lower()
        assert "1 failed" in result.output


class TestAddPdfUrl:
    """Tests for adding papers from PDF URLs."""

    def test_add_pdf_url_downloads_and_ingests(self, temp_db: Path, monkeypatch):
        """Test that --pdf with a URL downloads the PDF and ingests it."""

        # Mock requests.get for the download
        def mock_requests_get(url, timeout=None, stream=None):
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self._content = b"%PDF-1.4\n%test pdf content\n"

                def raise_for_status(self):
                    pass

                def iter_content(self, chunk_size=8192):
                    yield self._content

            return MockResponse()

        monkeypatch.setattr("requests.get", mock_requests_get)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                "https://example.com/paper.pdf",
                "--title",
                "Test Paper from URL",
                "--no-llm",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Downloading PDF from https://example.com/paper.pdf" in result.output

        # Verify the paper was added
        name = "test-paper-from-url"
        paper_dir = temp_db / "papers" / name
        assert paper_dir.exists()
        assert (paper_dir / "paper.pdf").read_bytes().startswith(b"%PDF")

    def test_add_pdf_url_handles_download_failure(self, temp_db: Path, monkeypatch):
        """Test that URL download failures are handled gracefully."""

        def mock_requests_get(url, timeout=None, stream=None):
            raise requests.exceptions.ConnectionError("Connection refused")

        monkeypatch.setattr("requests.get", mock_requests_get)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                "https://example.com/paper.pdf",
                "--title",
                "Test Paper",
                "--no-llm",
            ],
        )

        assert result.exit_code != 0
        assert "Failed to download PDF" in result.output

    def test_add_pdf_url_handles_http_error(self, temp_db: Path, monkeypatch):
        """Test that HTTP errors (404, 500) are handled gracefully."""

        def mock_requests_get(url, timeout=None, stream=None):
            class MockResponse:
                def __init__(self):
                    self.status_code = 404

                def raise_for_status(self):
                    # Create HTTPError with a response attribute
                    err = requests.exceptions.HTTPError("404 Not Found")
                    err.response = self
                    raise err

            return MockResponse()

        monkeypatch.setattr("requests.get", mock_requests_get)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                "https://example.com/nonexistent.pdf",
                "--title",
                "Test Paper",
                "--no-llm",
            ],
        )

        assert result.exit_code != 0
        assert "HTTP 404" in result.output

    def test_add_pdf_url_handles_timeout(self, temp_db: Path, monkeypatch):
        """Test that timeouts are handled gracefully."""

        def mock_requests_get(url, timeout=None, stream=None):
            raise requests.exceptions.Timeout("Request timed out")

        monkeypatch.setattr("requests.get", mock_requests_get)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                "https://example.com/slow.pdf",
                "--title",
                "Test Paper",
                "--no-llm",
            ],
        )

        assert result.exit_code != 0
        assert "Timed out" in result.output

    def test_add_pdf_local_file_still_works(self, temp_db: Path):
        """Test that local file paths still work after URL support was added."""
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local test\n")

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                str(pdf_path),
                "--title",
                "Local Paper",
                "--no-llm",
            ],
        )

        assert result.exit_code == 0, result.output
        paper_dir = temp_db / "papers" / "local-paper"
        assert paper_dir.exists()
        assert (paper_dir / "paper.pdf").read_bytes() == pdf_path.read_bytes()

    def test_add_pdf_nonexistent_local_file_errors(self, temp_db: Path):
        """Test that nonexistent local files give a clear error."""
        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "add",
                "--pdf",
                "/nonexistent/path/to/paper.pdf",
                "--title",
                "Test Paper",
                "--no-llm",
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_is_url_helper(self):
        """Test the _is_url helper function."""
        from paperpipe.cli.papers import _is_url

        assert _is_url("https://example.com/paper.pdf") is True
        assert _is_url("http://example.com/paper.pdf") is True
        assert _is_url("HTTP://EXAMPLE.COM/paper.pdf") is False  # Must be lowercase
        assert _is_url("/local/path/to/paper.pdf") is False
        assert _is_url("./relative/path.pdf") is False
        assert _is_url("paper.pdf") is False


class TestCommandAliases:
    """Tests for command aliases (rm, ls, regen, s, idx)."""

    def test_rm_alias_removes_paper(self, temp_db: Path):
        """Test that 'rm' is an alias for 'remove'."""
        papers_dir = temp_db / "papers"
        (papers_dir / "test-paper").mkdir(parents=True)
        (papers_dir / "test-paper" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1234.5678", "title": "Test Paper"})
        )
        paperpipe.save_index({"test-paper": {"arxiv_id": "1234.5678", "title": "Test Paper", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rm", "test-paper", "--yes"])
        assert result.exit_code == 0
        assert "Removed: test-paper" in result.output
        assert not (papers_dir / "test-paper").exists()

    def test_ls_alias_lists_papers(self, temp_db: Path):
        """Test that 'ls' is an alias for 'list'."""
        papers_dir = temp_db / "papers"
        (papers_dir / "paper-a").mkdir(parents=True)
        (papers_dir / "paper-a" / "meta.json").write_text(json.dumps({"title": "Paper A"}))
        paperpipe.save_index({"paper-a": {"title": "Paper A", "tags": [], "added": "2024-01-01"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ls"])
        assert result.exit_code == 0
        assert "paper-a" in result.output

    def test_regen_alias_regenerates_paper(self, temp_db: Path):
        """Test that 'regen' is an alias for 'regenerate'."""
        papers_dir = temp_db / "papers"
        (papers_dir / "p1").mkdir(parents=True)
        (papers_dir / "p1" / "meta.json").write_text(
            json.dumps({"arxiv_id": "1", "title": "Paper 1", "authors": [], "abstract": ""})
        )
        (papers_dir / "p1" / "source.tex").write_text(r"\begin{equation}x=1\end{equation}")
        paperpipe.save_index({"p1": {"arxiv_id": "1", "title": "Paper 1", "tags": [], "added": "x"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["regen", "p1", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating p1:" in result.output
        assert (papers_dir / "p1" / "summary.md").exists()

    def test_s_alias_searches_papers(self, temp_db: Path):
        """Test that 's' is an alias for 'search'."""
        papers_dir = temp_db / "papers"
        (papers_dir / "attention-paper").mkdir(parents=True)
        (papers_dir / "attention-paper" / "meta.json").write_text(
            json.dumps({"title": "Attention Is All You Need", "tags": ["transformers"]})
        )
        paperpipe.save_index(
            {"attention-paper": {"title": "Attention Is All You Need", "tags": ["transformers"], "added": "x"}}
        )

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["s", "attention"])
        assert result.exit_code == 0
        assert "attention-paper" in result.output

    def test_idx_alias_shows_index_status(self, temp_db: Path):
        """Test that 'idx' is an alias for 'index'."""
        runner = CliRunner()
        # Just check that the command is recognized and runs
        result = runner.invoke(cli_mod.cli, ["idx", "--help"])
        assert result.exit_code == 0
        assert "index" in result.output.lower() or "pqa" in result.output.lower()
