"""Tests for paperpipe/cli/rebuild.py (rebuild-index command)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import paperpipe

from .conftest import cli_mod


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
