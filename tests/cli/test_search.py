"""Tests for paperpipe/cli/search_cli.py (list, search, tags commands)."""

from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

import paperpipe

from .conftest import cli_mod, fts5_available


class TestListCommand:
    def test_list_empty(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["list"])
        assert result.exit_code == 0
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


class TestSearchCommand:
    def test_search_no_results(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No papers found" in result.output

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
        if not fts5_available():
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
        result = runner.invoke(cli_mod.cli, ["index", "--backend", "search", "--search-rebuild"])
        assert result.exit_code == 0, result.output

        monkeypatch.setenv("PAPERPIPE_SEARCH_MODE", "scan")
        result = runner.invoke(cli_mod.cli, ["search", "surface reconstruction"])
        assert result.exit_code == 0, result.output
        assert "Matches:" in result.output

    def test_search_fts_schema_mismatch_prompts_rebuild(self, temp_db: Path) -> None:
        if not fts5_available():
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


class TestTagsCommand:
    def test_tags_empty(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["tags"])
        assert result.exit_code == 0

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


class TestCliBasics:
    """Basic CLI tests (help, verbose, quiet flags)."""

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["--help"])
        assert result.exit_code == 0
        assert "paperpipe" in result.output

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

    def test_path_command(self, temp_db: Path):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["path"])
        assert result.exit_code == 0
        assert ".paperpipe" in result.output
