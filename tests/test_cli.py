from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import types
import zlib
from importlib import import_module
from pathlib import Path

import click
import conftest
import pytest
import requests
from click.testing import CliRunner
from conftest import MockPopen

import paperpipe
import paperpipe.config as config
import paperpipe.core as core
import paperpipe.paper as paper_mod
import paperpipe.paperqa as paperqa

# Import the CLI module explicitly (avoid resolving to the package's cli function).
cli_mod = import_module("paperpipe.cli")

# Well-known paper for integration tests: \"Attention Is All You Need\"
TEST_ARXIV_ID = "1706.03762"
REPO_ROOT = Path(__file__).resolve().parents[1]


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


class TestCli:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["--help"])
        assert result.exit_code == 0
        assert "paperpipe" in result.output

    def test_list_empty(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["list"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_search_no_results(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "nonexistent"])
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
        result = runner.invoke(cli_mod.cli, ["search", "--grep", "AdamW"])
        assert result.exit_code == 0, result.output
        assert "AdamW" in result.output

        cmd = calls[0]
        assert cmd[0] == "/usr/bin/rg"
        assert "--context" in cmd and "2" in cmd
        assert "--max-count" in cmd and "200" in cmd
        assert "--glob" in cmd and "**/summary.md" in cmd
        assert "**/source.tex" not in cmd

    def test_search_grep_falls_back_to_grep(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/bin/grep" if cmd == "grep" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="papers/x/equations.md:10:Eq. 7\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "--grep", "Eq. 7"])
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
        result = runner.invoke(cli_mod.cli, ["search", "--grep", "nope"])
        assert result.exit_code == 0, result.output
        assert "No matches" in result.output

    def test_search_rg_alias_works(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="x/summary.md:1:hit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "--rg", "hit"])
        assert result.exit_code == 0, result.output
        assert "hit" in result.output

    def test_search_regex_flag_requires_grep(self, temp_db: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "--regex", "x"])
        assert result.exit_code != 0
        assert "only apply with --grep" in result.output

    def test_search_grep_json_output(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="x/summary.md:12:AdamW\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "--grep", "--json", "AdamW"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload[0]["paper"] == "x"
        assert payload[0]["path"] == "x/summary.md"
        assert payload[0]["line"] == 12

    def test_search_index_builds_db_and_search_fts_uses_it(self, temp_db: Path) -> None:
        if not conftest.fts5_available():
            pytest.skip("SQLite FTS5 not available")

        paper_dir = temp_db / "papers" / "geom-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "Surface Reconstruction", "authors": [], "tags": []}))
        (paper_dir / "summary.md").write_text("We propose surface reconstruction from sparse points.\n")
        (paper_dir / "equations.md").write_text("No equations.\n")
        (paper_dir / "notes.md").write_text("Note.\n")

        paperpipe.save_index({"geom-paper": {"arxiv_id": None, "title": "Surface Reconstruction", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "search", "--search-rebuild"])
        assert result.exit_code == 0, result.output
        assert (temp_db / "search.db").exists()

        result = runner.invoke(cli_mod.cli, ["search", "--fts", "surface"])
        assert result.exit_code == 0, result.output
        assert "geom-paper" in result.output

    def test_index_search_include_tex_requires_rebuild(self, temp_db: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "search", "--search-include-tex"])
        assert result.exit_code != 0
        assert "--search-include-tex only applies with --search-rebuild" in result.output

    def test_search_hybrid_boosts_grep_hits(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not conftest.fts5_available():
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
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "search", "--search-rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            # Simulate ripgrep returning a hit in paper-b only.
            return types.SimpleNamespace(
                returncode=0, stdout="paper-b/summary.md:1:surface reconstruction\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner.invoke(cli_mod.cli, ["search", "--hybrid", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        # Hybrid should annotate grep hits.
        assert "paper-b" in result.output
        assert "grep:" in result.output

    def test_search_hybrid_show_grep_hits_prints_lines(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not conftest.fts5_available():
            pytest.skip("SQLite FTS5 not available")

        paper_b = temp_db / "papers" / "paper-b"
        paper_b.mkdir(parents=True)
        (paper_b / "meta.json").write_text(json.dumps({"title": "Unrelated", "authors": [], "tags": []}))
        (paper_b / "summary.md").write_text("surface reconstruction\n")

        paperpipe.save_index({"paper-b": {"arxiv_id": None, "title": "Unrelated", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "search", "--search-rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(
                returncode=0, stdout="paper-b/summary.md:1:surface reconstruction\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = runner.invoke(cli_mod.cli, ["search", "--hybrid", "--show-grep-hits", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        assert "paper-b/summary.md:1:" in result.output

    def test_search_hybrid_requires_search_db(self, temp_db: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "--hybrid", "x"])
        assert result.exit_code != 0
        assert "index --backend search" in result.output

    def test_search_mode_env_scan_forces_scan(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not conftest.fts5_available():
            pytest.skip("SQLite FTS5 not available")

        paper_dir = temp_db / "papers" / "p"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text(json.dumps({"title": "Surface Reconstruction", "authors": [], "tags": []}))
        (paper_dir / "summary.md").write_text("surface reconstruction\n")
        (paper_dir / "equations.md").write_text("surface reconstruction\n")
        (paper_dir / "notes.md").write_text("surface reconstruction\n")

        paperpipe.save_index({"p": {"arxiv_id": None, "title": "Surface Reconstruction", "tags": ["tag"]}})
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "search", "--search-rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setenv("PAPERPIPE_SEARCH_MODE", "scan")
        result = runner.invoke(cli_mod.cli, ["search", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        assert "Matches:" in result.output

    def test_search_fts_schema_mismatch_prompts_rebuild(self, temp_db: Path) -> None:
        if not conftest.fts5_available():
            pytest.skip("SQLite FTS5 not available")

        import sqlite3
        from contextlib import closing

        db_path = temp_db / "search.db"
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("CREATE TABLE search_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO search_meta(key, value) VALUES ('schema_version', '0')")
            conn.commit()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "--fts", "x"])
        assert result.exit_code != 0
        assert "schema version mismatch" in result.output.lower()
        assert "index --backend search --search-rebuild" in result.output

    def test_show_nonexistent(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["show", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_path_command(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["path"])
        assert result.exit_code == 0
        assert ".paperpipe" in result.output

    def test_tags_empty(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["tags"])
        assert result.exit_code == 0

    def test_cli_verbose_flag(self, temp_db: Path):
        """Test that --verbose flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["--verbose", "list"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

    def test_cli_quiet_flag(self, temp_db: Path):
        """Test that --quiet flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["--quiet", "list"])
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
        result = runner.invoke(cli_mod.cli, ["list"])
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
        result = runner.invoke(cli_mod.cli, ["list", "-t", "ml"])
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
        result = runner.invoke(cli_mod.cli, ["search", "neural"])
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
        result = runner.invoke(cli_mod.cli, ["search", "transformer"])
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
        result = runner.invoke(cli_mod.cli, ["search", "1706"])
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
        result = runner.invoke(cli_mod.cli, ["search", "surfae reconstructon"])
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
        result = runner.invoke(cli_mod.cli, ["search", "--exact", "surfae reconstructon"])
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
        result = runner.invoke(cli_mod.cli, ["search", "surface reconstruction"])
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
        result = runner.invoke(cli_mod.cli, ["show", "test-paper"])
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
        result = runner.invoke(cli_mod.cli, ["tags"])
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
        result = runner.invoke(cli_mod.cli, ["list", "--json"])
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

    def test_add_local_pdf_requires_title(self, temp_db: Path):
        pdf_path = temp_db / "local.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%local\n")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["add", "--pdf", str(pdf_path), "--no-llm"])
        assert result.exit_code != 0
        assert "--title" in result.output

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
        from datetime import datetime

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
        from datetime import datetime

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


class TestInstallSkillCommand:
    def test_install_skill_creates_symlink_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "skills" / "papi"
        assert dest.is_symlink()
        assert dest.resolve() == (REPO_ROOT / "skill").resolve()

    def test_install_skill_existing_dest_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0
        assert "use --force" in result.output.lower()

    def test_install_skill_force_overwrites_existing_file(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex", "--force"])
        assert result.exit_code == 0, result.output

        assert dest.is_symlink()

    def test_install_skill_creates_symlink_for_gemini(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--gemini"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".gemini" / "skills" / "papi"
        assert dest.is_symlink()
        assert dest.resolve() == (REPO_ROOT / "skill").resolve()


class TestInstallPromptsCommand:
    def test_install_prompts_creates_symlinks_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex"])
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
            assert dest.resolve() == (REPO_ROOT / "prompts" / "codex" / filename).resolve()

    def test_install_prompts_existing_dest_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "prompts" / "ground-with-paper.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0
        assert "use --force" in result.output.lower()

    def test_install_prompts_copy_mode_copies_files(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex", "--copy"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "curate-paper-note.md"
        assert dest.exists()
        assert not dest.is_symlink()

    def test_install_prompts_creates_symlinks_for_gemini(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--gemini"])
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
            assert dest.resolve() == (REPO_ROOT / "prompts" / "gemini" / filename).resolve()


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
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--claude"])
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
            and "paperqa_mcp_server" in c
            for c in calls
        )

    def test_install_mcp_repo_writes_mcp_json(self, temp_db: Path):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["install", "mcp", "--repo", "--embedding", "text-embedding-3-small"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["command"] == "paperqa_mcp_server"
            assert cfg["mcpServers"]["paperqa"]["args"] == []
            cfg2 = json.loads((Path(".gemini") / "settings.json").read_text())
            assert cfg2["mcpServers"]["paperqa"]["command"] == "paperqa_mcp_server"
            assert cfg2["mcpServers"]["paperqa"]["args"] == []

    def test_install_mcp_repo_writes_config(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["install", "mcp", "--repo", "--embedding", "text-embedding-3-small"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["command"] == "paperqa_mcp_server"
            assert cfg["mcpServers"]["paperqa"]["args"] == []
            # paperqa server now includes both PaperQA2 and LEANN tools
            assert "leann" not in cfg["mcpServers"]  # No separate LEANN server anymore

    def test_install_mcp_repo_uses_paperqa_embedding_env_override(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (temp_db / "config.toml").write_text("\n".join(["[paperqa]", 'embedding = "config-embedding"', ""]))
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERQA_EMBEDDING", "env-embedding")

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["install", "mcp", "--repo"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["env"]["PAPERQA_EMBEDDING"] == "env-embedding"

    def test_install_mcp_gemini_writes_settings_json(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--gemini", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output

        cfg = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert cfg["mcpServers"]["paperqa"]["command"] == "paperqa_mcp_server"
        assert cfg["mcpServers"]["paperqa"]["args"] == []
        assert cfg["mcpServers"]["paperqa"]["env"]["PAPERQA_EMBEDDING"] == "text-embedding-3-small"

    def test_install_mcp_gemini_runs_gemini_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--gemini", "--embedding", "text-embedding-3-small"])
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
            and "paperqa_mcp_server" in c
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
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--gemini"])
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
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--codex", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output
        assert any(
            c[:4] == ["codex", "mcp", "add", "paperqa"]
            and "paperqa_mcp_server" in c
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
            cli_mod.cli,
            ["install", "mcp", "--codex", "--force", "--embedding", "text-embedding-3-small"],
        )
        assert result.exit_code == 0, result.output
        assert calls[0][:4] == ["codex", "mcp", "remove", "paperqa"]
        assert calls[1][:4] == ["codex", "mcp", "add", "paperqa"]


class TestUninstallSkillCommand:
    def test_uninstall_skill_removes_symlink_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "skills" / "papi"
        assert dest.is_symlink()

        result2 = runner.invoke(cli_mod.cli, ["uninstall", "skill", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_skill_mismatch_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "skill", "--codex"])
        assert result.exit_code == 1
        assert "use --force" in result.output.lower()


class TestUninstallPromptsCommand:
    def test_uninstall_prompts_removes_symlinks_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "papi.md"
        assert dest.is_symlink()

        result2 = runner.invoke(cli_mod.cli, ["uninstall", "prompts", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_prompts_removes_copied_files_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex", "--copy"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "curate-paper-note.md"
        assert dest.exists()
        assert not dest.is_symlink()

        result2 = runner.invoke(cli_mod.cli, ["uninstall", "prompts", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_parses_commas_for_components(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp,prompts", "--codex", "--force"])
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
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--claude"])
        assert result.exit_code == 0, result.output

        assert ["claude", "mcp", "remove", "paperqa"] in calls
        # No separate leann server anymore

    def test_uninstall_mcp_repo_removes_server_keys(self, temp_db: Path):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "paperqa": {"command": "papi", "args": [], "env": {}},
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
                            "other": {"command": "x", "args": [], "env": {}},
                        }
                    }
                )
                + "\n"
            )

            result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--repo"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert "paperqa" not in cfg["mcpServers"]
            # No separate leann server anymore
            assert "other" in cfg["mcpServers"]

            cfg2 = json.loads((Path(".gemini") / "settings.json").read_text())
            assert "paperqa" not in cfg2["mcpServers"]
            # No separate leann server anymore
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
                    }
                }
            )
            + "\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output

        cfg = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert "paperqa" not in cfg.get("mcpServers", {})
        # No separate leann server anymore

    def test_uninstall_mcp_gemini_runs_gemini_mcp_remove(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output

        assert ["gemini", "mcp", "remove", "--scope", "user", "paperqa"] in calls
        # No separate leann server anymore


class TestUninstallValidation:
    def test_uninstall_repo_requires_mcp_component(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "skill", "--repo"])
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


class TestExport:
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "query", "--format", "evidence-blocks"])
        assert result.exit_code == 0, result.output
        assert '"backend": "pqa"' in result.output

    def test_ask_evidence_blocks_rejects_passthrough_args(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
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

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["ask", "test"])

        assert result.exit_code == 0
        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")

        # Concurrency should be set to 1 by default
        concurrency_idx = pqa_call.index("--agent.index.concurrency") + 1
        assert pqa_call[concurrency_idx] == "1"


class TestIndexCommand:
    def test_index_backend_pqa_runs_pqa_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        noisy = (
            "/home/x/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:\n"
            "  PydanticSerializationUnexpectedValue(Expected 10 fields but got 7)\n"
            "  return self.__pydantic_serializer__.to_python(\n"
        )
        mock_popen = MockPopen(returncode=0, stdout=f"{noisy}Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed"])
        assert result.exit_code == 0, result.output
        assert "Indexed" in result.output
        assert "PydanticSerializationUnexpectedValue" not in result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "index" in pqa_call
        assert str(temp_db / ".pqa_papers") in pqa_call
        assert "--agent.index.paper_directory" in pqa_call
        assert "--index" in pqa_call and "paperpipe_my-embed" in pqa_call
        assert (temp_db / ".pqa_papers" / "test-paper.pdf").exists()

    def test_index_backend_pqa_ollama_embedding_strips_prefix_and_forces_provider(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "ollama/nomic-embed-text"])
        assert result.exit_code == 0, result.output

        pqa_call, _ = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        assert "--index" in pqa_call
        assert "paperpipe_ollama_nomic-embed-text" in pqa_call
        assert "--embedding" in pqa_call
        assert pqa_call[pqa_call.index("--embedding") + 1] == "nomic-embed-text"
        assert "--embedding_config" in pqa_call
        cfg = pqa_call[pqa_call.index("--embedding_config") + 1]
        assert '"custom_llm_provider":"ollama"' in cfg

    def test_index_backend_pqa_pqa_raw_prints_noisy_output(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)

        noisy = (
            "/home/x/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:\n"
            "  PydanticSerializationUnexpectedValue(Expected 10 fields but got 7)\n"
            "  return self.__pydantic_serializer__.to_python(\n"
        )
        mock_popen = MockPopen(returncode=0, stdout=f"{noisy}Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-embedding", "my-embed", "--pqa-raw"])
        assert result.exit_code == 0, result.output
        assert "PydanticSerializationUnexpectedValue" in result.output

    def test_index_backend_pqa_ollama_models_prepare_env(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: False)
        monkeypatch.setattr(config, "_ollama_reachability_error", lambda **kwargs: None)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        (temp_db / "papers" / "test-paper").mkdir(parents=True)
        (temp_db / "papers" / "test-paper" / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-llm", "ollama/qwen3:8b"])
        assert result.exit_code == 0, result.output

        _, pqa_kwargs = next(c for c in mock_popen.calls if c[0][0] == "pqa")
        env = pqa_kwargs.get("env") or {}
        assert env.get("OLLAMA_API_BASE") == "http://localhost:11434"
        assert env.get("OLLAMA_HOST") == "http://localhost:11434"

    def test_index_rejects_pqa_concurrency_zero(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pqa" if cmd == "pqa" else None)
        monkeypatch.setattr(paperqa, "_pillow_available", lambda: True)
        mock_popen = MockPopen(returncode=0, stdout="Indexed\n")
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["index", "--pqa-concurrency", "0"])
        assert result.exit_code != 0
        assert "--pqa-concurrency must be >= 1" in result.output


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
            cli_mod.cli,
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
            cli_mod.cli,
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
        result = runner.invoke(cli_mod.cli, ["models", "--kind", "completion", "--model", "ok-model"])
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
        result = runner.invoke(cli_mod.cli, ["models", "--preset", "latest", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "latest", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "--preset", "last-gen", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "last-gen", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "--preset", "all", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "all", "--json"])
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
        result = runner.invoke(cli_mod.cli, ["models", "--json"])
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
        result = runner.invoke(cli_mod.cli, ["models", "latest", "--kind", "completion", "--json"])
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
        import sys

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
        import sys

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
        import sys

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
        import sys

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
@pytest.mark.skipif(not conftest.litellm_available(), reason="LiteLLM not installed or no API key configured")
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
@pytest.mark.skipif(not conftest.pqa_available(), reason="PaperQA2 (pqa) not installed")
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

        # Should get some response (not just the fallback)
        assert result.exit_code == 0


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

        # Write the index file with ERROR status
        files_zip.write_bytes(zlib.compress(pickle.dumps({staged_path_str: "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)))

        runner = CliRunner()
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
        files_zip.write_bytes(zlib.compress(pickle.dumps({staged_path_str: "ERROR"}, protocol=pickle.HIGHEST_PROTOCOL)))

        runner = CliRunner()
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


class TestRebuildIndexCommand:
    def test_rebuild_index_basic(self, temp_db: Path):
        """Test basic rebuild-index from paper directories."""
        # Create some paper directories with meta.json
        paper1 = temp_db / "papers" / "paper-one"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(
            json.dumps({"title": "First Paper", "authors": ["Alice"], "tags": ["ml"], "added": "2024-01-01"})
        )
        (paper1 / "paper.pdf").touch()

        paper2 = temp_db / "papers" / "paper-two"
        paper2.mkdir(parents=True)
        (paper2 / "meta.json").write_text(
            json.dumps({"title": "Second Paper", "arxiv_id": "2401.00001", "tags": [], "added": "2024-01-02"})
        )
        (paper2 / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Rebuilt index with 2 paper(s)" in result.output

        # Verify the index was rebuilt correctly
        index = paperpipe.load_index()
        assert "paper-one" in index
        assert "paper-two" in index
        assert index["paper-one"]["title"] == "First Paper"
        assert index["paper-two"]["arxiv_id"] == "2401.00001"

    def test_rebuild_index_dry_run(self, temp_db: Path):
        """Test dry run doesn't modify the index."""
        paper1 = temp_db / "papers" / "paper-one"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(json.dumps({"title": "First Paper"}))

        # Save a different index
        paperpipe.save_index({"old-paper": {"title": "Old"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "paper-one" in result.output

        # Verify the original index is unchanged
        index = paperpipe.load_index()
        assert "old-paper" in index
        assert "paper-one" not in index

    def test_rebuild_index_with_backup(self, temp_db: Path):
        """Test that backup is created when --backup is used."""
        paper1 = temp_db / "papers" / "paper-one"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(json.dumps({"title": "First Paper"}))

        # Save an existing index
        paperpipe.save_index({"old-paper": {"title": "Old"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index", "--backup"])

        assert result.exit_code == 0
        assert "Backed up existing index" in result.output

        # Find the backup file
        backups = list(temp_db.glob("index.json.backup.*"))
        assert len(backups) == 1
        backup_content = json.loads(backups[0].read_text())
        assert "old-paper" in backup_content

    def test_rebuild_index_no_backup(self, temp_db: Path):
        """Test that backup is skipped when --no-backup is used."""
        paper1 = temp_db / "papers" / "paper-one"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(json.dumps({"title": "First Paper"}))

        paperpipe.save_index({"old-paper": {"title": "Old"}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index", "--no-backup"])

        assert result.exit_code == 0
        assert "Backed up" not in result.output

        # Verify no backup was created
        backups = list(temp_db.glob("index.json.backup.*"))
        assert len(backups) == 0

    def test_rebuild_index_with_validation(self, temp_db: Path):
        """Test validation reports issues."""
        # Paper with missing PDF
        paper1 = temp_db / "papers" / "paper-no-pdf"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(json.dumps({"title": "Missing PDF"}))
        # No paper.pdf

        # Paper with all files
        paper2 = temp_db / "papers" / "paper-complete"
        paper2.mkdir(parents=True)
        (paper2 / "meta.json").write_text(json.dumps({"title": "Complete Paper"}))
        (paper2 / "paper.pdf").touch()

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index", "--validate"])

        assert result.exit_code == 0
        assert "Validation issues" in result.output
        assert "paper-no-pdf" in result.output
        assert "missing paper.pdf" in result.output

    def test_rebuild_index_skips_invalid_directories(self, temp_db: Path):
        """Test that directories without meta.json are skipped."""
        # Valid paper
        paper1 = temp_db / "papers" / "valid-paper"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(json.dumps({"title": "Valid Paper"}))

        # Invalid directory (no meta.json)
        invalid = temp_db / "papers" / "invalid-dir"
        invalid.mkdir(parents=True)
        (invalid / "some-file.txt").write_text("not a paper")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Skipped 1 directory" in result.output
        assert "invalid-dir" in result.output

        index = paperpipe.load_index()
        assert "valid-paper" in index
        assert "invalid-dir" not in index

    def test_rebuild_index_empty_papers_dir(self, temp_db: Path):
        """Test handling of empty papers directory."""
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "No paper directories found" in result.output

    def test_rebuild_index_empty_papers_dir_backs_up_existing_index(self, temp_db: Path):
        """Test that empty rebuild still backs up existing index."""
        # Save an existing index with data
        paperpipe.save_index({"existing-paper": {"title": "Existing Paper", "tags": []}})

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "No paper directories found" in result.output
        assert "Backed up existing index" in result.output

        # Verify backup was created
        backups = list(temp_db.glob("index.json.backup.*"))
        assert len(backups) == 1
        backup_content = json.loads(backups[0].read_text())
        assert "existing-paper" in backup_content

        # Verify index is now empty
        index = paperpipe.load_index()
        assert index == {}

    def test_rebuild_index_preserves_all_metadata_fields(self, temp_db: Path):
        """Test that all common metadata fields are preserved."""
        paper1 = temp_db / "papers" / "full-meta"
        paper1.mkdir(parents=True)
        meta = {
            "title": "Full Metadata Test",
            "authors": ["Alice", "Bob"],
            "arxiv_id": "2401.00001",
            "doi": "10.1234/test",
            "tags": ["ml", "nlp"],
            "added": "2024-01-01T00:00:00",
            "year": 2024,
            "venue": "NeurIPS",
            "tldr": "A test paper.",
            "abstract": "This is the abstract.",
            "url": "https://example.com/paper",
            "semantic_scholar_id": "abc123",
            "citation_count": 42,
            "categories": ["cs.LG", "cs.CL"],
        }
        (paper1 / "meta.json").write_text(json.dumps(meta))

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index"])

        assert result.exit_code == 0

        index = paperpipe.load_index()
        entry = index["full-meta"]
        assert entry["title"] == "Full Metadata Test"
        assert entry["authors"] == ["Alice", "Bob"]
        assert entry["arxiv_id"] == "2401.00001"
        assert entry["doi"] == "10.1234/test"
        assert entry["tags"] == ["ml", "nlp"]
        assert entry["year"] == 2024
        assert entry["venue"] == "NeurIPS"
        assert entry["tldr"] == "A test paper."
        assert entry["semantic_scholar_id"] == "abc123"
        assert entry["citation_count"] == 42

    def test_rebuild_index_handles_corrupt_meta_json(self, temp_db: Path):
        """Test handling of corrupt meta.json files."""
        # Valid paper
        paper1 = temp_db / "papers" / "valid-paper"
        paper1.mkdir(parents=True)
        (paper1 / "meta.json").write_text(json.dumps({"title": "Valid Paper"}))

        # Corrupt meta.json
        corrupt = temp_db / "papers" / "corrupt-paper"
        corrupt.mkdir(parents=True)
        (corrupt / "meta.json").write_text("not valid json {")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["rebuild-index"])

        assert result.exit_code == 0
        assert "Skipped" in result.output
        assert "corrupt-paper" in result.output

        index = paperpipe.load_index()
        assert "valid-paper" in index
        assert "corrupt-paper" not in index
