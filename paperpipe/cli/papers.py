"""Paper management commands: add, regenerate, show, notes, remove."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import click

from .. import config
from ..cli.helpers import (
    _extract_semantic_scholar_id,
    _fetch_semantic_scholar_metadata,
    _is_semantic_scholar_id,
    _parse_bibtex_file,
)
from ..config import normalize_tags
from ..core import (
    _arxiv_base_from_any,
    _index_arxiv_base_to_names,
    _is_safe_paper_name,
    _parse_overwrite_option,
    _resolve_paper_name_from_ref,
    ensure_notes_file,
    load_index,
    normalize_arxiv_id,
    save_index,
)
from ..output import echo_error, echo_progress, echo_success, echo_warning
from ..paper import _add_local_pdf, _add_single_paper, _regenerate_one_paper
from ..search import _maybe_delete_from_search_index, _maybe_update_search_index


@click.command()
@click.argument("arxiv_ids", nargs=-1, required=False)
@click.option("--pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Ingest a local PDF.")
@click.option("--title", help="Title for local PDF ingest (required with --pdf).")
@click.option(
    "--authors",
    help="Authors for local PDF ingest (use ';' as separator; supports single 'Last, First' without splitting).",
)
@click.option("--abstract", help="Abstract for local PDF ingest.")
@click.option("--year", type=int, help="Year for local PDF ingest (YYYY).")
@click.option("--venue", help="Venue/journal for local PDF ingest.")
@click.option("--doi", help="DOI for local PDF ingest.")
@click.option("--url", help="URL for the paper (publisher/project page).")
@click.option("--name", "-n", help="Short name for the paper (only valid with single paper)")
@click.option("--tags", "-t", help="Additional comma-separated tags (applied to all papers)")
@click.option("--no-llm", is_flag=True, help="Skip LLM-based generation")
@click.option("--tldr/--no-tldr", default=True, show_default=True, help="Generate a one-paragraph TL;DR.")
@click.option("--figures", is_flag=True, help="Extract figures from LaTeX source or PDF")
@click.option(
    "--duplicate",
    is_flag=True,
    help="Allow adding a second copy even if this arXiv ID already exists (creates a new name like -2/-3).",
)
@click.option(
    "--update", is_flag=True, help="If this arXiv ID already exists, refresh it in-place instead of skipping."
)
@click.option(
    "--from-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Import papers from a JSON file (exported via `papi list --json`), BibTeX file (.bib), or text file (one ID per line).",
)
def add(
    arxiv_ids: tuple[str, ...],
    pdf: Optional[Path],
    title: Optional[str],
    authors: Optional[str],
    abstract: Optional[str],
    year: Optional[int],
    venue: Optional[str],
    doi: Optional[str],
    url: Optional[str],
    name: Optional[str],
    tags: Optional[str],
    no_llm: bool,
    tldr: bool,
    figures: bool,
    duplicate: bool,
    update: bool,
    from_file: Optional[Path],
):
    """Add one or more papers to the database."""
    if pdf:
        if arxiv_ids or from_file:
            raise click.UsageError("Use either arXiv IDs/URLs/--from-file OR `--pdf`, not both.")
        if not title or not title.strip():
            raise click.UsageError("Missing required option: --title (required with --pdf).")
        if duplicate or update:
            raise click.UsageError("--duplicate/--update are only supported for arXiv ingestion.")
        success, paper_name = _add_local_pdf(
            pdf=pdf,
            title=title,
            name=name,
            tags=tags,
            authors=authors,
            abstract=abstract,
            year=year,
            venue=venue,
            doi=doi,
            url=url,
            no_llm=no_llm,
            tldr=tldr,
        )
        if not success:
            raise SystemExit(1)
        if paper_name:
            _maybe_update_search_index(name=paper_name)
        return

    if not arxiv_ids and not from_file:
        raise click.UsageError("Missing arXiv ID/URL argument(s) or --from-file (or pass `--pdf`).")

    if name and (len(arxiv_ids) > 1 or from_file):
        raise click.UsageError("--name can only be used when adding a single paper via CLI arguments.")
    if duplicate and update:
        raise click.UsageError("Use either --duplicate or --update, not both.")

    # Collect tasks: list of (arxiv_id, name_override, tags_override)
    tasks: list[tuple[str, Optional[str], Optional[str]]] = []
    # Track S2 papers without arXiv IDs (these are failures we can't process)
    s2_no_arxiv_failures: list[str] = []

    # 1. From CLI args
    for identifier in arxiv_ids:
        # Check if this is a Semantic Scholar ID
        if _is_semantic_scholar_id(identifier):
            # Fetch metadata from Semantic Scholar
            s2_id = _extract_semantic_scholar_id(identifier)
            metadata = _fetch_semantic_scholar_metadata(s2_id)

            if metadata and metadata.get("arxiv_id"):
                # If we found an arXiv ID, use that for paperpipe
                tasks.append((metadata["arxiv_id"], name, tags))
            elif metadata:
                # If we have metadata but no arXiv ID, track as failure
                echo_warning(
                    f"Paper {identifier} does not have an arXiv ID. Currently only arXiv papers are supported."
                )
                s2_no_arxiv_failures.append(identifier)
                continue
            else:
                # Failed to fetch metadata
                echo_error(f"Failed to fetch metadata for Semantic Scholar paper: {identifier}")
                raise SystemExit(1)
        else:
            # Regular arXiv ID processing
            tasks.append((identifier, name, tags))

    # 2. From file
    if from_file:
        content = from_file.read_text("utf-8")
        file_extension = from_file.suffix.lower()

        # Handle BibTeX files
        if file_extension == ".bib":
            try:
                tasks.extend(_parse_bibtex_file(content, tags))
            except ImportError:
                echo_error("bibtexparser not installed. Install with: pip install 'paperpipe[bibtex]'")
                raise SystemExit(1)
            except Exception as e:
                # Provide more specific error messages based on the type of error
                error_msg = str(e)
                if "EOF" in error_msg or "Unexpected end of file" in error_msg:
                    echo_error("Failed to parse BibTeX file: File appears to be incomplete or corrupted")
                elif "Expected" in error_msg and "got" in error_msg:
                    echo_error(f"Failed to parse BibTeX file: Syntax error - {error_msg}")
                elif "Unknown field" in error_msg:
                    echo_error(f"Failed to parse BibTeX file: Unknown field encountered - {error_msg}")
                else:
                    echo_error(f"Failed to parse BibTeX file: {error_msg}")
                    # For debugging, show the first few lines of the file
                    lines = content.split("\n")[:5]
                    echo_error(f"First few lines of file: {lines}")
                raise SystemExit(1)
        else:
            # Existing JSON and text file handling
            try:
                # Try parsing as JSON (export format: dict[name, info])
                data = json.loads(content)
                if isinstance(data, dict):
                    echo_progress(f"Importing {len(data)} papers from JSON...")
                    for key, info in data.items():
                        # Handle both export format (dict) and list of IDs (if user made a list)
                        if isinstance(info, dict):
                            aid = info.get("arxiv_id")
                            if not aid:
                                continue
                            # If importing from JSON, we use the key as the name
                            # and merge JSON tags with CLI tags
                            item_tags = info.get("tags", [])
                            if isinstance(item_tags, list):
                                item_tags_str = ",".join(item_tags)
                            else:
                                item_tags_str = str(item_tags) if item_tags else ""

                            # Merge with CLI tags if present
                            final_tags = item_tags_str
                            if tags:
                                final_tags = f"{final_tags},{tags}" if final_tags else tags

                            tasks.append((aid, key, final_tags))
                        elif isinstance(info, str):
                            # Simple dict {"name": "arxiv_id"} style? unlikely but possible
                            tasks.append((info, key, tags))
                elif isinstance(data, list):
                    # JSON list of IDs or objects?
                    for item in data:
                        if isinstance(item, str):
                            tasks.append((item, None, tags))
                        elif isinstance(item, dict) and "arxiv_id" in item:
                            # Normalize tags from list to comma-separated string
                            item_tags = item.get("tags", [])
                            if isinstance(item_tags, list):
                                item_tags_str = ",".join(item_tags)
                            else:
                                item_tags_str = str(item_tags) if item_tags else ""

                            # Merge with CLI tags if present
                            final_tags = item_tags_str
                            if tags:
                                final_tags = f"{final_tags},{tags}" if final_tags else tags

                            tasks.append((item["arxiv_id"], item.get("name"), final_tags))
            except json.JSONDecodeError:
                # Fallback: Treat as line-separated text file
                # Filter empty lines and comments
                lines = [
                    line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")
                ]
                echo_progress(f"Importing {len(lines)} papers from text file...")
                for line in lines:
                    tasks.append((line, None, tags))

    if not tasks:
        if s2_no_arxiv_failures:
            # All inputs were S2 papers without arXiv IDs - this is a failure
            echo_error(
                f"No papers to add: {len(s2_no_arxiv_failures)} Semantic Scholar paper(s) "
                "had no arXiv ID and could not be processed."
            )
            raise SystemExit(1)
        click.echo("No papers to add.")
        return

    index = load_index()
    existing_names = set(index.keys())
    base_to_names = _index_arxiv_base_to_names(index)

    added = 0
    updated = 0
    skipped = 0
    # Include S2 papers without arXiv IDs in the failure count
    failures = len(s2_no_arxiv_failures)

    for i, (arxiv_id, p_name, p_tags) in enumerate(tasks, 1):
        if len(tasks) > 1:
            echo_progress(f"[{i}/{len(tasks)}] Adding {arxiv_id}...")
        else:
            echo_progress(f"Adding paper: {arxiv_id}")

        success, paper_name, action = _add_single_paper(
            arxiv_id,
            p_name,
            p_tags,
            no_llm,
            tldr,
            duplicate,
            update,
            figures,
            index,
            existing_names,
            base_to_names,
        )
        if success:
            if action == "added":
                added += 1
                if paper_name:
                    _maybe_update_search_index(name=paper_name)
            elif action == "updated":
                updated += 1
                if paper_name:
                    _maybe_update_search_index(name=paper_name)
            elif action == "skipped":
                skipped += 1
        else:
            failures += 1

    # Print summary for multiple papers (including S2 failures that weren't added to tasks)
    total_inputs = len(tasks) + len(s2_no_arxiv_failures)
    if total_inputs > 1:
        click.echo()
        if failures == 0:
            parts = []
            if added:
                parts.append(f"added {added}")
            if updated:
                parts.append(f"updated {updated}")
            if skipped:
                parts.append(f"skipped {skipped}")
            echo_success(", ".join(parts) if parts else "No changes")
        else:
            parts = []
            if added:
                parts.append(f"added {added}")
            if updated:
                parts.append(f"updated {updated}")
            if skipped:
                parts.append(f"skipped {skipped}")
            if not parts:
                parts.append("no changes")
            echo_warning(f"{', '.join(parts)}, {failures} failed")

    if failures > 0:
        raise SystemExit(1)


@click.command()
@click.argument("papers", nargs=-1)
@click.option("--all", "regenerate_all", is_flag=True, help="Regenerate all papers")
@click.option("--no-llm", is_flag=True, help="Skip LLM-based regeneration")
@click.option(
    "--overwrite",
    "-o",
    default=None,
    help="Overwrite fields: 'all' or comma-separated list (summary,equations,tags,name,figures)",
)
@click.option("--name", "-n", "set_name", default=None, help="Set name directly (single paper only)")
@click.option("--tags", "-t", "set_tags", default=None, help="Add tags (comma-separated)")
def regenerate(
    papers: tuple[str, ...],
    regenerate_all: bool,
    no_llm: bool,
    overwrite: Optional[str],
    set_name: Optional[str],
    set_tags: Optional[str],
):
    """Regenerate summary/equations/figures for existing papers (by name or arXiv ID).

    By default, only missing fields are generated. Use --overwrite to force regeneration:

    \b
      --overwrite all           Regenerate everything
      --overwrite name          Regenerate name only
      --overwrite tags,tldr     Regenerate tags and TL;DR
      --overwrite figures       Extract figures from PDF

    Use --name or --tags to set values directly (no LLM):

    \b
      --name neus-w             Rename paper to 'neus-w'
      --tags nerf,3d            Add tags 'nerf' and '3d'
    """
    index = load_index()

    # Validate set options
    if set_name and (regenerate_all or len(papers) != 1):
        raise click.UsageError("--name can only be used with a single paper.")
    if (set_name or set_tags) and regenerate_all:
        raise click.UsageError("--name/--tags cannot be used with --all.")

    # Parse overwrite option
    overwrite_fields, overwrite_all = _parse_overwrite_option(overwrite)

    if regenerate_all and papers:
        raise click.UsageError("Use either paper(s)/arXiv id(s) OR `--all`, not both.")

    def resolve_name(target: str) -> Optional[str]:
        if target in index:
            return target
        try:
            normalized = normalize_arxiv_id(target)
        except ValueError:
            normalized = target

        base = _arxiv_base_from_any(normalized)
        matches = [n for n, info in index.items() if _arxiv_base_from_any(info.get("arxiv_id", "")) == base]
        if not matches:
            return None
        if len(matches) > 1:
            echo_error(f"Multiple papers match arXiv ID {base}: {', '.join(sorted(matches))}")
            return None
        return matches[0]

    # Handle --all flag or "all" as positional argument (when no paper named "all" exists)
    if regenerate_all or (len(papers) == 1 and papers[0] == "all" and "all" not in index):
        names = sorted(index.keys())
        if not names:
            click.echo("No papers found.")
            return

        failures = 0
        renames: list[tuple[str, str]] = []
        for i, name in enumerate(names, 1):
            echo_progress(f"[{i}/{len(names)}] {name}")
            success, new_name = _regenerate_one_paper(
                name,
                index,
                no_llm=no_llm,
                overwrite_fields=overwrite_fields,
                overwrite_all=overwrite_all,
            )
            if not success:
                failures += 1
            elif new_name:
                renames.append((name, new_name))

        if renames:
            click.echo(f"\nRenamed {len(renames)} paper(s):")
            for old, new in renames:
                click.echo(f"  {old} → {new}")

        if failures:
            raise click.ClickException(f"{failures} paper(s) failed to regenerate.")
        return

    if not papers:
        raise click.UsageError("Missing PAPER argument(s) (or pass `--all`).")

    # Handle direct set operations (--name, --tags) for single paper
    if set_name or set_tags:
        paper_ref = papers[0]
        name = resolve_name(paper_ref)
        if not name:
            raise click.ClickException(f"Paper not found: {paper_ref}")

        paper_dir = config.PAPERS_DIR / name
        meta_path = paper_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Handle --name
        if set_name:
            set_name = set_name.strip().lower()
            set_name = re.sub(r"[^a-z0-9-]", "", set_name).strip("-")
            if not set_name:
                raise click.UsageError("Invalid name")
            if set_name == name:
                echo_warning(f"Name unchanged: {name}")
            elif set_name in index:
                raise click.ClickException(f"Name '{set_name}' already exists")
            else:
                new_dir = config.PAPERS_DIR / set_name
                paper_dir.rename(new_dir)
                del index[name]
                index[set_name] = {
                    "arxiv_id": meta.get("arxiv_id"),
                    "title": meta.get("title"),
                    "tags": meta.get("tags", []),
                    "added": meta.get("added"),
                }
                save_index(index)
                echo_success(f"Renamed: {name} → {set_name}")
                name = set_name
                paper_dir = new_dir
                meta_path = paper_dir / "meta.json"

        # Handle --tags
        if set_tags:
            new_tags = [t.strip().lower() for t in set_tags.split(",") if t.strip()]
            existing_tags = meta.get("tags", [])
            all_tags = normalize_tags([*existing_tags, *new_tags])
            meta["tags"] = all_tags
            meta_path.write_text(json.dumps(meta, indent=2))
            index[name]["tags"] = all_tags
            save_index(index)
            echo_success(f"Tags: {', '.join(all_tags)}")

        # If no --overwrite, we're done
        if not overwrite:
            return

    # Process multiple papers
    successes = 0
    failures = 0
    renames: list[tuple[str, str]] = []

    for i, paper_ref in enumerate(papers, 1):
        if len(papers) > 1:
            echo_progress(f"[{i}/{len(papers)}] {paper_ref}")

        name = resolve_name(paper_ref)
        if not name:
            echo_error(f"Paper not found: {paper_ref}")
            failures += 1
            continue

        success, new_name = _regenerate_one_paper(
            name,
            index,
            no_llm=no_llm,
            overwrite_fields=overwrite_fields,
            overwrite_all=overwrite_all,
        )
        if success:
            successes += 1
            if new_name:
                renames.append((name, new_name))
        else:
            failures += 1

    # Print summary for multiple papers
    if len(papers) > 1:
        if renames:
            click.echo(f"\nRenamed {len(renames)} paper(s):")
            for old, new in renames:
                click.echo(f"  {old} → {new}")

        click.echo()
        if failures == 0:
            echo_success(f"Regenerated {successes} paper(s)")
        else:
            echo_warning(f"Regenerated {successes} paper(s), {failures} failed")
    elif renames:
        # Single paper case
        old, new = renames[0]
        click.echo(f"Paper renamed: {old} → {new}")

    if failures > 0:
        raise SystemExit(1)


@click.command()
@click.argument("papers", nargs=-1, required=True)
@click.option(
    "--level",
    "-l",
    type=click.Choice(["meta", "summary", "equations", "eq", "tex", "latex", "full", "tldr"], case_sensitive=False),
    default="meta",
    show_default=True,
    help="What to show (prints to stdout).",
)
def show(papers: tuple[str, ...], level: str):
    """Show paper details or print saved content (summary/equations/LaTeX/TL;DR)."""
    index = load_index()

    level_norm = (level or "").strip().lower()
    if level_norm == "eq":
        level_norm = "equations"
    if level_norm in {"latex", "tex", "full"}:
        level_norm = "tex"

    if level_norm == "summary":
        src_name = "summary.md"
        missing_msg = "No summary found"
    elif level_norm == "equations":
        src_name = "equations.md"
        missing_msg = "No equations found"
    elif level_norm == "tex":
        src_name = "source.tex"
        missing_msg = "No LaTeX source found"
    elif level_norm == "tldr":
        src_name = "tldr.md"
        missing_msg = "No TL;DR found"
    else:
        src_name = ""
        missing_msg = ""

    first_output = True
    for paper_ref in papers:
        name, error = _resolve_paper_name_from_ref(paper_ref, index)
        if not name:
            echo_error(error)
            continue

        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            echo_error(f"Paper not found: {paper_ref}")
            continue

        if not first_output:
            click.echo("\n\n---\n")
        first_output = False

        meta_path = paper_dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as e:
                echo_warning(f"Could not read metadata for {name}: {e}")
                meta = {}

        click.echo(f"# {name}")

        if level_norm == "meta":
            title = (meta.get("title") or "").strip()
            arxiv_id = (meta.get("arxiv_id") or "").strip()
            authors = meta.get("authors") or []
            tags = meta.get("tags") or []
            has_pdf = bool(meta.get("has_pdf", False))
            has_source = bool(meta.get("has_source", False))

            if title:
                click.echo(f"- Title: {title}")
            if arxiv_id:
                click.echo(f"- arXiv: {arxiv_id}")
            if authors:
                click.echo(f"- Authors: {', '.join([str(a) for a in authors[:8]])}")
            if tags:
                click.echo(f"- Tags: {', '.join([str(t) for t in tags])}")
            click.echo(f"- Has PDF: {has_pdf}")
            click.echo(f"- Has LaTeX: {has_source}")

            tldr_path = paper_dir / "tldr.md"
            if tldr_path.exists():
                tldr_text = tldr_path.read_text(errors="ignore").strip()
                if tldr_text:
                    click.echo(f"- TL;DR: {tldr_text}")

            click.echo(f"- Location: {paper_dir}")
            try:
                click.echo(f"- Files: {', '.join(sorted(f.name for f in paper_dir.iterdir()))}")
            except Exception as e:
                click.echo(f"- Files: [unable to list: {e}]")
            continue

        src = paper_dir / src_name
        if not src.exists():
            echo_error(f"{missing_msg}: {name}")
            continue

        click.echo(f"- Content: {level_norm}")
        click.echo()
        click.echo(src.read_text(errors="ignore").rstrip("\n"))


@click.command()
@click.argument("papers", nargs=-1, required=True)
@click.option("--print", "print_", is_flag=True, help="Print notes to stdout instead of opening an editor.")
@click.option(
    "--edit/--no-edit",
    default=None,
    help="Open notes in $EDITOR (default: edit for a single paper; otherwise print paths).",
)
def notes(papers: tuple[str, ...], print_: bool, edit: Optional[bool]):
    """Open or print per-paper implementation notes (notes.md)."""
    index = load_index()

    effective_edit = edit
    if effective_edit is None:
        effective_edit = (not print_) and (len(papers) == 1)

    if effective_edit and len(papers) != 1:
        raise click.UsageError("--edit can only be used with a single paper. Use --print for multiple.")

    first_output = True
    for paper_ref in papers:
        name, error = _resolve_paper_name_from_ref(paper_ref, index)
        if not name:
            raise click.ClickException(error)

        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            raise click.ClickException(f"Paper not found: {paper_ref}")

        meta_path = paper_dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception as e:
                echo_warning(f"Could not read metadata for {name}: {e}")
                meta = {}

        notes_path = ensure_notes_file(paper_dir, meta)

        if print_:
            if not first_output:
                click.echo("\n\n---\n")
            first_output = False
            click.echo(f"# {name} ({notes_path})")
            click.echo()
            click.echo(notes_path.read_text(errors="ignore").rstrip("\n"))
            continue

        if effective_edit:
            try:
                click.edit(filename=str(notes_path))
            except Exception as exc:
                raise click.ClickException(f"Failed to open editor for {notes_path}: {exc}") from exc
        else:
            click.echo(str(notes_path))


@click.command()
@click.argument("papers", nargs=-1, required=True)
@click.confirmation_option(prompt="Are you sure you want to remove these paper(s)?")
def remove(papers: tuple[str, ...]):
    """Remove one or more papers from the database (by name or arXiv ID/URL)."""
    index = load_index()

    successes = 0
    failures = 0

    for i, paper_ref in enumerate(papers, 1):
        if len(papers) > 1:
            echo_progress(f"[{i}/{len(papers)}] Removing {paper_ref}...")

        name, error = _resolve_paper_name_from_ref(paper_ref, index)
        if not name:
            echo_error(error)
            failures += 1
            continue

        if not _is_safe_paper_name(name):
            echo_error(f"Invalid paper name: {name!r}")
            failures += 1
            continue

        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            echo_error(f"Paper not found: {paper_ref}")
            failures += 1
            continue

        shutil.rmtree(paper_dir)

        if name in index:
            del index[name]
            save_index(index)

        _maybe_delete_from_search_index(name=name)

        echo_success(f"Removed: {name}")
        successes += 1

    # Print summary for multiple papers
    if len(papers) > 1:
        click.echo()
        if failures == 0:
            echo_success(f"Removed {successes} paper(s)")
        else:
            echo_warning(f"Removed {successes} paper(s), {failures} failed")

    if failures > 0:
        raise SystemExit(1)
