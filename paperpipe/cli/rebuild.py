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
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        echo_warning(f"Could not read {meta_path}: {e}")
        return None

    if not isinstance(meta, dict):
        echo_warning(f"Invalid meta.json format in {paper_dir.name}: expected dict")
        return None

    # Build index entry from meta.json
    entry: dict = {}

    # Required/common fields
    if "title" in meta:
        entry["title"] = meta["title"]
    if "authors" in meta:
        entry["authors"] = meta["authors"]
    if "arxiv_id" in meta:
        entry["arxiv_id"] = meta["arxiv_id"]
    if "doi" in meta:
        entry["doi"] = meta["doi"]
    if "tags" in meta:
        entry["tags"] = meta["tags"]
    if "added" in meta:
        entry["added"] = meta["added"]
    if "year" in meta:
        entry["year"] = meta["year"]
    if "venue" in meta:
        entry["venue"] = meta["venue"]
    if "tldr" in meta:
        entry["tldr"] = meta["tldr"]
    if "abstract" in meta:
        entry["abstract"] = meta["abstract"]
    if "url" in meta:
        entry["url"] = meta["url"]
    if "semantic_scholar_id" in meta:
        entry["semantic_scholar_id"] = meta["semantic_scholar_id"]
    if "citation_count" in meta:
        entry["citation_count"] = meta["citation_count"]
    if "categories" in meta:
        entry["categories"] = meta["categories"]

    return entry


def _validate_paper_directory(paper_dir: Path) -> list[str]:
    """Validate a paper directory and return list of issues found."""
    issues: list[str] = []

    # Check for meta.json
    if not (paper_dir / "meta.json").exists():
        issues.append("missing meta.json")

    # Check for PDF (expected but not strictly required)
    pdf_path = paper_dir / "paper.pdf"
    if not pdf_path.exists():
        issues.append("missing paper.pdf")

    return issues


def _backup_index(backup_path: Path) -> bool:
    """Create a backup of the current index.json.

    Returns True if backup was created, False if index doesn't exist.
    """
    if not config.INDEX_FILE.exists():
        return False

    shutil.copy2(config.INDEX_FILE, backup_path)
    return True


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
            save_index({})
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
    if backup and config.INDEX_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config.PAPER_DB / f"index.json.backup.{timestamp}"
        if _backup_index(backup_path):
            echo_progress(f"Backed up existing index to {backup_path}")

    # Save new index
    save_index(new_index)
    echo_success(f"Rebuilt index with {len(new_index)} paper(s).")

    # Report validation issues
    if validate and validation_issues:
        echo_warning(f"\nValidation issues found in {len(validation_issues)} paper(s):")
        for name, issues in sorted(validation_issues.items()):
            echo_warning(f"  {name}: {', '.join(issues)}")
