"""Tests for paperpipe/cli/export.py (audit, export commands)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import paperpipe

from .conftest import cli_mod


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
        result = runner.invoke(cli_mod.cli, ["audit"])
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
        result = runner.invoke(cli_mod.cli, ["audit", "--regenerate", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating bad-paper:" in result.output
        assert "# Bad Paper" in (bad_dir / "summary.md").read_text()
        assert "Eikonal" not in (bad_dir / "summary.md").read_text()

    def test_audit_yes_flag_skips_prompts_without_regenerating(self, temp_db: Path):
        """Test that --yes/-y flag skips prompts but does NOT regenerate without --regenerate."""
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
        original_summary = "**Eikonal Regularization:** This is important.\n"
        (bad_dir / "summary.md").write_text(original_summary)
        (bad_dir / "equations.md").write_text(
            'Based on the provided LaTeX content for the paper **"Some Other Paper"**\n'
        )

        paperpipe.save_index({"bad-paper": {"arxiv_id": "id-bad", "title": "Bad Paper", "tags": [], "added": "x"}})

        runner = CliRunner()
        # --yes alone should NOT regenerate, just skip prompts and exit
        result = runner.invoke(cli_mod.cli, ["audit", "-y"])
        assert result.exit_code == 0
        assert "FLAGGED" in result.output
        assert "Regenerating" not in result.output
        # Summary should be unchanged
        assert (bad_dir / "summary.md").read_text() == original_summary

    def test_audit_yes_with_regenerate_skips_prompts_and_regenerates(self, temp_db: Path):
        """Test that --yes --regenerate skips prompts and regenerates flagged papers."""
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
        # --yes --regenerate should regenerate without prompts
        result = runner.invoke(cli_mod.cli, ["audit", "-y", "--regenerate", "--no-llm", "-o", "summary,equations"])
        assert result.exit_code == 0
        assert "Regenerating bad-paper:" in result.output
        assert "# Bad Paper" in (bad_dir / "summary.md").read_text()
        assert "Eikonal" not in (bad_dir / "summary.md").read_text()


class TestExportCommand:
    def test_export_nonexistent_paper(self, temp_db: Path):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["export", "nonexistent"])
            assert "not found" in result.output

    def test_export_summary(self, temp_db: Path):
        # Create a paper directory with summary
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("# Test Summary")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "summary", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper_summary.md").exists()
            assert Path("test-paper_summary.md").read_text() == "# Test Summary"

    def test_export_equations(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("# Equations\nE=mc^2")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "equations", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper_equations.md").exists()

    def test_export_full_with_source(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "source.tex").write_text(r"\documentclass{article}")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "full", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper.tex").exists()

    def test_export_full_without_source(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        # No source.tex file

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "full", "--to", "."])
            assert "No LaTeX source" in result.output

    def test_export_to_dash_is_rejected(self, temp_db: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("x\n")
        paperpipe.save_index({"test-paper": {"arxiv_id": "2301.00001", "title": "T", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "equations", "--to", "-"])
        assert result.exit_code != 0
        assert "export` only writes to a directory" in result.output.lower()

    def test_export_copies_equations_to_directory(self, temp_db: Path, tmp_path: Path):
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "equations.md").write_text("eq\n")
        paperpipe.save_index({"test-paper": {"arxiv_id": "2301.00001", "title": "T", "tags": []}})

        runner = CliRunner()
        out_dir = tmp_path / "paper-context"
        result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "equations", "--to", str(out_dir)])
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
        result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "eq", "--to", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "test-paper_equations.md").exists()

    def test_export_with_figures_flag(self, temp_db: Path):
        """Test --figures flag exports figures directory."""
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("# Test Summary")

        # Create figures directory with files
        figures_dir = paper_dir / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (figures_dir / "fig2.pdf").write_bytes(b"%PDF-1.4\n")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli_mod.cli, ["export", "test-paper", "--level", "summary", "--to", ".", "--figures"]
            )
            assert result.exit_code == 0
            assert Path("test-paper_summary.md").exists()
            assert Path("test-paper_figures").exists()
            assert (Path("test-paper_figures") / "fig1.png").exists()
            assert (Path("test-paper_figures") / "fig2.pdf").exists()

    def test_export_without_figures_flag(self, temp_db: Path):
        """Test figures not exported when --figures flag omitted."""
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("# Test Summary")

        # Create figures directory
        figures_dir = paper_dir / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["export", "test-paper", "--level", "summary", "--to", "."])
            assert result.exit_code == 0
            assert Path("test-paper_summary.md").exists()
            # Figures should NOT be exported
            assert not Path("test-paper_figures").exists()

    def test_export_figures_when_no_figures_directory(self, temp_db: Path):
        """Test --figures flag gracefully handles missing figures directory."""
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("# Test Summary")
        # No figures directory

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli_mod.cli, ["export", "test-paper", "--level", "summary", "--to", ".", "--figures"]
            )
            assert result.exit_code == 0
            assert Path("test-paper_summary.md").exists()
            # No error, just no figures directory created
            assert not Path("test-paper_figures").exists()

    def test_export_figures_overwrites_existing(self, temp_db: Path):
        """Test --figures overwrites existing figures directory."""
        paper_dir = temp_db / "papers" / "test-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "summary.md").write_text("# Test Summary")

        figures_dir = paper_dir / "figures"
        figures_dir.mkdir()
        (figures_dir / "new.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create old figures directory
            old_figures = Path("test-paper_figures")
            old_figures.mkdir()
            (old_figures / "old.png").write_bytes(b"old data")

            result = runner.invoke(
                cli_mod.cli, ["export", "test-paper", "--level", "summary", "--to", ".", "--figures"]
            )
            assert result.exit_code == 0

            # Old file should be gone, new file should exist
            assert not (old_figures / "old.png").exists()
            assert (old_figures / "new.png").exists()

    def test_export_multiple_papers_with_figures(self, temp_db: Path):
        """Test exporting multiple papers with figures."""
        for name in ["paper1", "paper2"]:
            paper_dir = temp_db / "papers" / name
            paper_dir.mkdir(parents=True)
            (paper_dir / "summary.md").write_text(f"# {name} Summary")

            figures_dir = paper_dir / "figures"
            figures_dir.mkdir()
            (figures_dir / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli_mod.cli, ["export", "paper1", "paper2", "--level", "summary", "--to", ".", "--figures"]
            )
            assert result.exit_code == 0
            assert Path("paper1_summary.md").exists()
            assert Path("paper2_summary.md").exists()
            assert (Path("paper1_figures") / "fig.png").exists()
            assert (Path("paper2_figures") / "fig.png").exists()
