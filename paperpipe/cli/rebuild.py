"""Rebuild-index command for index recovery."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from .. import config
from ..core import save_index
from ..output import echo_error, echo_progress, echo_success, echo_warning


def _scan_paper_directory(paper_dir: Path) -> Optional[dict]:
    """Scan a paper directory and extract index entry from meta.json.

    Returns None if the directory is invalid or meta.json is missing/corrupt.
    """
    meta_path = paper_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as e:
        echo_warning(f"Could not read {meta_path}: {e}")
        return None

    if not isinstance(meta, dict):
        echo_warning(f"Invalid meta.json format in {paper_dir.name}: expected dict")
        return None

    # Build index entry from meta.json - copy known fields that exist
    index_fields = (
        "title",
        "authors",
        "arxiv_id",
        "doi",
        "tags",
        "added",
        "year",
        "venue",
        "tldr",
        "abstract",
        "url",
        "semantic_scholar_id",
        "citation_count",
        "categories",
    )
    return {key: meta[key] for key in index_fields if key in meta}


def _validate_paper_directory(paper_dir: Path) -> list[str]:
    """Validate a paper directory and return list of issues found.

    Note: Only called for directories that passed _scan_paper_directory,
    so meta.json is guaranteed to exist and be valid.
    """
    issues: list[str] = []

    # Check for PDF (expected but not strictly required)
    if not (paper_dir / "paper.pdf").exists():
        issues.append("missing paper.pdf")

    return issues


def _backup_index(backup_path: Path) -> bool:
    """Create a backup of the current index.json.

    Returns True if backup was created, False if index doesn't exist or backup failed.
    """
    if not config.INDEX_FILE.exists():
        return False

    try:
        shutil.copy2(config.INDEX_FILE, backup_path)
        return True
    except (PermissionError, OSError) as e:
        echo_error(f"Failed to create backup at {backup_path}: {e}")
        return False


def _safe_save_index(index: dict, backup_path: Optional[Path] = None) -> bool:
    """Save index with error handling, providing recovery guidance on failure.

    Returns True on success, False on failure (after printing error message).
    """
    try:
        save_index(index)
        return True
    except (PermissionError, OSError) as e:
        echo_error(f"Failed to save index: {e}")
        if backup_path and backup_path.exists():
            echo_error(f"Your previous index was backed up to: {backup_path}")
        raise SystemExit(1)


@click.command("rebuild-index")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be rebuilt without modifying the index.",
)
@click.option(
    "--backup/--no-backup",
    default=True,
    show_default=True,
    help="Create a timestamped backup of the existing index before rebuilding.",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Run validation checks and report issues after rebuild.",
)
def rebuild_index(dry_run: bool, backup: bool, validate: bool) -> None:
    """Rebuild index.json from on-disk paper directories.

    Useful for recovery when the index is corrupted, manually edited incorrectly,
    or when migrating from a backup or different machine.

    By default, creates a timestamped backup of the existing index before rebuilding.
    """
    papers_dir = config.PAPERS_DIR

    if not papers_dir.exists():
        echo_error(f"Papers directory does not exist: {papers_dir}")
        raise SystemExit(1)

    # Scan paper directories
    paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]

    if not paper_dirs:
        echo_warning("No paper directories found.")
        if not dry_run:
            # Backup before overwriting with empty index
            backup_path: Optional[Path] = None
            if backup and config.INDEX_FILE.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = config.PAPER_DB / f"index.json.backup.{timestamp}"
                if _backup_index(backup_path):
                    echo_progress(f"Backed up existing index to {backup_path}")
            _safe_save_index({}, backup_path)
            echo_success("Created empty index.")
        return

    # Build new index
    new_index: dict = {}
    skipped: list[str] = []
    validation_issues: dict[str, list[str]] = {}

    for paper_dir in sorted(paper_dirs):
        name = paper_dir.name
        entry = _scan_paper_directory(paper_dir)

        if entry is None:
            skipped.append(name)
            continue

        new_index[name] = entry

        if validate:
            issues = _validate_paper_directory(paper_dir)
            if issues:
                validation_issues[name] = issues

    # Report what was found
    echo_progress(f"Found {len(new_index)} paper(s) with valid metadata.")
    if skipped:
        echo_warning(f"Skipped {len(skipped)} directory(ies) without valid meta.json: {', '.join(skipped)}")

    if dry_run:
        echo_progress("Dry run - index not modified.")
        click.echo("\nPapers that would be indexed:")
        for name in sorted(new_index.keys()):
            title = new_index[name].get("title", "(no title)")
            click.echo(f"  {name}: {title}")
        return

    # Backup existing index
    backup_path: Optional[Path] = None
    if backup and config.INDEX_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config.PAPER_DB / f"index.json.backup.{timestamp}"
        if _backup_index(backup_path):
            echo_progress(f"Backed up existing index to {backup_path}")

    # Save new index
    _safe_save_index(new_index, backup_path)
    echo_success(f"Rebuilt index with {len(new_index)} paper(s).")

    # Report validation issues
    if validate and validation_issues:
        echo_warning(f"\nValidation issues found in {len(validation_issues)} paper(s):")
        for name, issues in sorted(validation_issues.items()):
            echo_warning(f"  {name}: {', '.join(issues)}")
