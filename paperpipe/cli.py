"""Click CLI entry points."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import click

from . import config, paperqa
from .config import (
    DEFAULT_LEANN_INDEX_NAME,
    _effective_leann_index_name,
    _is_ollama_model_id,
    _strip_ollama_prefix,
    default_embedding_model,
    default_llm_model,
    default_pqa_answer_length,
    default_pqa_concurrency,
    default_pqa_embedding_model,
    default_pqa_enrichment_llm,
    default_pqa_evidence_k,
    default_pqa_index_dir,
    default_pqa_llm_model,
    default_pqa_max_sources,
    default_pqa_ollama_timeout,
    default_pqa_settings_name,
    default_pqa_summary_llm,
    default_pqa_temperature,
    default_pqa_timeout,
    default_pqa_verbosity,
    default_search_mode,
    normalize_tags,
    pqa_index_name_for_embedding,
)
from .core import (
    _arxiv_base_from_any,
    _fuzzy_text_score,
    _index_arxiv_base_to_names,
    _is_safe_paper_name,
    _parse_overwrite_option,
    _read_text_limited,
    _resolve_paper_name_from_ref,
    ensure_db,
    ensure_notes_file,
    load_index,
    normalize_arxiv_id,
    save_index,
)
from .install import (
    _install_mcp,
    _install_prompts,
    _install_skill,
    _parse_components,
    _uninstall_mcp,
    _uninstall_prompts,
    _uninstall_skill,
)
from .leann import _ask_leann, _leann_build_index, _leann_index_meta_path
from .output import (
    _setup_debug_logging,
    debug,
    echo_error,
    echo_progress,
    echo_success,
    echo_warning,
    set_quiet,
)
from .paper import _add_local_pdf, _add_single_paper, _regenerate_one_paper
from .search import (
    _audit_paper_dir,
    _collect_grep_matches,
    _ensure_search_index_schema,
    _maybe_delete_from_search_index,
    _maybe_update_search_index,
    _parse_selection_spec,
    _search_db_path,
    _search_fts,
    _search_grep,
    _search_index_rebuild,
    _search_index_upsert,
    _sqlite_connect,
)


class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, min_interval: float = 0.1):
        self.min_interval = min_interval
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if the minimum interval hasn't passed since last call."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            self.last_call_time = time.time()

    def update_from_headers(self, headers: dict) -> None:
        """Update rate limiting parameters from API response headers."""
        # This can be extended to parse actual rate limit headers if available
        pass


def _parse_bibtex_file(content: str, cli_tags: Optional[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Parse a BibTeX file and extract arXiv IDs or DOIs for paper import.

    Returns a list of tuples: (identifier, name_override, tags_override)
    """
    try:
        import bibtexparser
        from bibtexparser.bparser import BibTexParser
    except ImportError:
        raise ImportError("bibtexparser not installed")

    # Parse BibTeX content
    parser = BibTexParser()
    parser.ignore_nonstandard_types = False  # type: ignore[attr-defined]
    bib_database = bibtexparser.loads(content, parser=parser)

    # If no entries were found but the file contains characters suggesting
    # it's a BibTeX file, then parsing likely failed silently or partially.
    if not bib_database.entries and re.search(r"@[a-zA-Z]+", content):
        raise ValueError("Failed to parse BibTeX file: No valid entries found. Check for syntax errors.")

    tasks: List[Tuple[str, Optional[str], Optional[str]]] = []

    for entry in bib_database.entries:
        # Try to extract arXiv ID first
        arxiv_id = None
        doi = None
        title = entry.get("title", "").replace("{", "").replace("}", "")

        # Check for arXiv ID in various fields with improved field handling
        # Check primary fields first
        if "eprint" in entry:
            arxiv_id = entry["eprint"]
        # Check alternative field names (case-insensitive)
        elif "arxivId" in entry:
            arxiv_id = entry["arxivId"]
        elif "archivePrefix" in entry and "arxiv" in entry["archivePrefix"].lower():
            # For entries with archivePrefix=arXiv, the eprint field should have the ID
            if "eprint" in entry:
                arxiv_id = entry["eprint"]
        # Check for archiveprefix (lowercase) which is also common
        elif "archiveprefix" in entry and "arxiv" in entry["archiveprefix"].lower():
            if "eprint" in entry:
                arxiv_id = entry["eprint"]
        # Check for eprinttype field which may indicate arXiv
        elif "eprinttype" in entry and "arxiv" in entry["eprinttype"].lower():
            if "eprint" in entry:
                arxiv_id = entry["eprint"]
        # Check for journal field that might indicate arXiv
        elif "journal" in entry:
            journal = entry["journal"]
            if isinstance(journal, str) and "arxiv" in journal.lower():
                if "eprint" in entry:
                    arxiv_id = entry["eprint"]

        # Normalize arXiv ID format - remove arXiv prefix if present
        if arxiv_id and isinstance(arxiv_id, str):
            # Handle cases like "arXiv:1234.5678" or "arxiv:1234.5678"
            arxiv_id = re.sub(r"^(?:arxiv:?)?(?:org/abs/)?", "", arxiv_id.strip(), flags=re.IGNORECASE)

        # If no arXiv ID found, check for DOI
        if not arxiv_id and "doi" in entry:
            doi = entry["doi"]
            # Check if DOI contains arXiv ID
            if isinstance(doi, str) and "arxiv" in doi.lower():
                # Extract arXiv ID from DOI - handle various formats
                # Match formats like "10.48550/arXiv.1234.5678" or URLs in DOIs
                arxiv_match = re.search(
                    r"(?:arxiv[:.]org/abs/|arxiv[.:]|arxiv/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", doi, re.IGNORECASE
                )
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)

        # If still no arXiv ID, check URL field
        if not arxiv_id and "url" in entry:
            url = entry["url"]
            if isinstance(url, str) and "arxiv.org" in url.lower():
                # More robust URL extraction that handles various URL formats
                arxiv_match = re.search(
                    r"arxiv(?:\.org)?[\/:]?\/?(?:abs|pdf)[\/:]?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", url, re.IGNORECASE
                )
                if not arxiv_match:
                    # Fallback pattern for any arxiv.org URL
                    arxiv_match = re.search(r"arxiv\.org[^\s]*?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", url, re.IGNORECASE)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)

        # If we have an arXiv ID, create a task
        if arxiv_id:
            # Generate a name based on the BibTeX key or title
            name = entry.key if hasattr(entry, "key") else None
            if not name and title:
                # Create a slug from the title with better normalization
                # Remove LaTeX commands and normalize whitespace
                clean_title = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", title)
                clean_title = re.sub(r"[{}]", "", clean_title)
                # Create slug with better character handling
                name = re.sub(r"[^a-zA-Z0-9]+", "-", clean_title.lower()).strip("-")[:30] or None

            # Combine tags from the BibTeX entry with CLI tags
            tags = []
            if "keywords" in entry:
                keywords = entry["keywords"]
                if isinstance(keywords, str):
                    # Handle both comma-separated and other formats
                    tags.extend([tag.strip() for tag in keywords.split(",") if tag.strip()])
            if cli_tags:
                tags.extend(cli_tags.split(","))
            tags_str = ",".join(tags) if tags else None

            tasks.append((arxiv_id, name, tags_str))
        elif doi:
            # For now, we only support arXiv papers, so skip DOIs
            # In the future, we could add support for DOI-based paper fetching
            pass

    return tasks


def _is_semantic_scholar_id(identifier: str) -> bool:
    """Check if the identifier is a Semantic Scholar ID or URL."""
    if not isinstance(identifier, str):
        return False

    identifier = identifier.strip()

    # Check for official Semantic Scholar paper URLs
    if identifier.startswith("https://www.semanticscholar.org/paper/"):
        return True

    # Check for other Semantic Scholar URLs
    if "semanticscholar.org" in identifier:
        return True

    # Check for valid Semantic Scholar paper IDs (40-character hex strings)
    if len(identifier) == 40 and all(c in "0123456789abcdef" for c in identifier.lower()):
        return True

    # Check for longer strings that might be IDs (but not URLs)
    if len(identifier) > 40 and "/" not in identifier:
        # More precise check: should be mostly hex characters
        hex_chars = sum(1 for c in identifier.lower() if c in "0123456789abcdef")
        if hex_chars / len(identifier) > 0.8:  # At least 80% hex characters
            return True

    return False


def _extract_semantic_scholar_id(identifier: str) -> str:
    """Extract the Semantic Scholar paper ID from a URL or ID."""
    if not isinstance(identifier, str):
        return identifier

    identifier = identifier.strip()

    # Handle official Semantic Scholar paper URLs
    if identifier.startswith("https://www.semanticscholar.org/paper/"):
        # Extract ID from URL: https://www.semanticscholar.org/paper/{title}/{ID}
        parts = identifier.split("/")
        # The ID is the last part and should be a 40-character hex string
        for part in reversed(parts):
            if len(part) == 40 and all(c in "0123456789abcdef" for c in part.lower()):
                return part
        # Fallback: return the last part
        return parts[-1]

    # Handle other Semantic Scholar URLs
    elif "semanticscholar.org" in identifier:
        parts = identifier.split("/")
        # Look for the 40-character hex ID
        for part in parts:
            if len(part) == 40 and all(c in "0123456789abcdef" for c in part.lower()):
                return part
        # If no 40-char hex found, return the last non-empty part
        for part in reversed(parts):
            if part:
                return part

    # If it's already an ID (not a URL), return as-is
    return identifier


# Rate limiter for Semantic Scholar API calls
_semantic_scholar_rate_limiter = RateLimiter(min_interval=0.1)


def _fetch_semantic_scholar_metadata(s2_id: str) -> Optional[dict]:
    """Fetch paper metadata from Semantic Scholar API with rate limiting and retry logic."""
    import time

    import requests

    # Thread-safe rate limiting
    _semantic_scholar_rate_limiter.wait_if_needed()

    # Semantic Scholar API endpoint
    url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}"
    params = {"fields": "title,authors,abstract,year,venue,externalIds,url"}

    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    echo_warning(f"Rate limited by Semantic Scholar API. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    echo_error(f"Rate limit exceeded for Semantic Scholar API after {max_retries} attempts.")
                    return None

            response.raise_for_status()
            data = response.json()

            # Extract arXiv ID if available
            arxiv_id = None
            if "externalIds" in data:
                external_ids = data["externalIds"]
                if "ArXiv" in external_ids:
                    arxiv_id = external_ids["ArXiv"]

            # Extract DOI if available
            doi = None
            if "externalIds" in data:
                external_ids = data["externalIds"]
                if "DOI" in external_ids:
                    doi = external_ids["DOI"]

            # Extract authors
            authors = []
            if "authors" in data:
                authors = [author.get("name", "") for author in data["authors"]]

            # Extract other metadata
            metadata = {
                "title": data.get("title", ""),
                "authors": authors,
                "abstract": data.get("abstract", ""),
                "year": data.get("year"),
                "venue": data.get("venue", ""),
                "url": data.get("url", ""),
                "doi": doi,
                "arxiv_id": arxiv_id,
            }

            return metadata

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                echo_warning(
                    f"Timeout while fetching Semantic Scholar metadata for {s2_id}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                echo_error(
                    f"Timeout while fetching Semantic Scholar metadata for {s2_id} after {max_retries} attempts."
                )
                return None

        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                echo_warning(
                    f"Connection error while fetching Semantic Scholar metadata for {s2_id}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                echo_error(
                    f"Connection error while fetching Semantic Scholar metadata for {s2_id} after {max_retries} attempts. Check your internet connection."
                )
                return None

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "Unknown"
            if status_code == 404:
                echo_error(f"Paper not found in Semantic Scholar (404) for ID: {s2_id}")
                return None
            elif status_code == 500:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    echo_warning(
                        f"Semantic Scholar API server error (500) for ID: {s2_id}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    echo_error(f"Semantic Scholar API server error (500) for ID: {s2_id} after {max_retries} attempts.")
                    return None
            else:
                echo_error(f"HTTP error {status_code} while fetching Semantic Scholar metadata for {s2_id}: {e}")
                return None

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                echo_warning(
                    f"Request error while fetching Semantic Scholar metadata for {s2_id}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                echo_error(
                    f"Request error while fetching Semantic Scholar metadata for {s2_id} after {max_retries} attempts: {e}"
                )
                return None

        except ValueError as e:
            # JSON decode error
            echo_error(f"Invalid JSON response from Semantic Scholar API for {s2_id}: {e}")
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                echo_warning(
                    f"Unexpected error while fetching Semantic Scholar metadata for {s2_id}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                echo_error(
                    f"Unexpected error while fetching Semantic Scholar metadata for {s2_id} after {max_retries} attempts: {e}"
                )
                return None

    # This should never be reached, but just in case
    echo_error(f"Failed to fetch Semantic Scholar metadata for {s2_id} after all retries.")
    return None


# CLI Commands
# ============================================================================


def _cli_version() -> str:
    try:
        return package_version("paperpipe")
    except PackageNotFoundError:
        return "0+unknown"


@click.group()
@click.version_option(version=_cli_version())
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress messages.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug output.")
def cli(quiet: bool = False, verbose: bool = False):
    """paperpipe: Unified paper database for coding agents + PaperQA2."""
    set_quiet(quiet)
    if verbose:
        _setup_debug_logging()
    ensure_db()


@cli.command()
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
                # If we have metadata but no arXiv ID, we could potentially add support
                # for creating metadata-only entries in the future
                echo_warning(
                    f"Paper {identifier} does not have an arXiv ID. Currently only arXiv papers are supported."
                )
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
                    # JSON list of IDs?
                    for item in data:
                        if isinstance(item, str):
                            tasks.append((item, None, tags))
                        elif isinstance(item, dict) and "arxiv_id" in item:
                            tasks.append((item["arxiv_id"], item.get("name"), item.get("tags")))
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
        click.echo("No papers to add.")
        return

    index = load_index()
    existing_names = set(index.keys())
    base_to_names = _index_arxiv_base_to_names(index)

    added = 0
    updated = 0
    skipped = 0
    failures = 0

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

    # Print summary for multiple papers
    if len(tasks) > 1:
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


@cli.command()
@click.argument("papers", nargs=-1)
@click.option("--all", "regenerate_all", is_flag=True, help="Regenerate all papers")
@click.option("--no-llm", is_flag=True, help="Skip LLM-based regeneration")
@click.option(
    "--overwrite",
    "-o",
    default=None,
    help="Overwrite fields: 'all' or comma-separated list (summary,equations,tags,name)",
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
    """Regenerate summary/equations for existing papers (by name or arXiv ID).

    By default, only missing fields are generated. Use --overwrite to force regeneration:

    \b
      --overwrite all           Regenerate everything
      --overwrite name          Regenerate name only
      --overwrite tags,tldr     Regenerate tags and TL;DR

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


@cli.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_papers(tag: Optional[str], as_json: bool):
    """List all papers in the database."""
    index = load_index()

    if tag:
        index = {k: v for k, v in index.items() if tag in v.get("tags", [])}

    if as_json:
        click.echo(json.dumps(index, indent=2))
        return

    if not index:
        click.echo("No papers found.")
        return

    for name, info in sorted(index.items()):
        title = info.get("title", "Unknown")[:50]
        tags = ", ".join(info.get("tags", [])[:4])
        click.echo(name)
        click.echo(f"  {title}...")
        click.echo(f"  Tags: {tags}")
        click.echo()


@cli.command()
@click.argument("query")
@click.option(
    "--limit",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of results to show.",
)
@click.option(
    "--grep/--no-grep",
    "--rg/--no-rg",
    "use_grep",
    default=False,
    show_default=True,
    help="Use ripgrep/grep for fast exact-match search (shows file:line hits).",
)
@click.option(
    "--fixed-strings/--regex",
    "fixed_strings",
    default=False,
    show_default=True,
    help="In --grep mode, treat QUERY as a literal string instead of a regex.",
)
@click.option(
    "--context",
    "context_lines",
    type=int,
    default=2,
    show_default=True,
    help="In --grep mode, number of context lines around each match.",
)
@click.option(
    "--max-matches",
    type=int,
    default=200,
    show_default=True,
    help="In --grep mode, stop after this many matches (approx; tool-dependent; effectively per-file for grep).",
)
@click.option(
    "--ignore-case/--case-sensitive",
    default=False,
    show_default=True,
    help="In --grep mode, ignore case when matching QUERY.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="In --grep mode, output machine-readable JSON (forces --context 0).",
)
@click.option(
    "--fuzzy/--exact",
    default=True,
    show_default=True,
    help="Fall back to fuzzy matching only if no exact matches were found.",
)
@click.option(
    "--tex/--no-tex",
    default=False,
    show_default=True,
    help="Also search within LaTeX source (can be slower).",
)
@click.option(
    "--fts/--no-fts",
    "use_fts",
    default=True,
    show_default=True,
    help="Use SQLite FTS5 ranked search if `search.db` exists (falls back to scan). Use --no-fts to force scan.",
)
@click.option(
    "--hybrid/--no-hybrid",
    default=False,
    show_default=True,
    help="Hybrid search: FTS5 ranked search + grep signal boosting papers with exact matches.",
)
@click.option(
    "--show-grep-hits",
    is_flag=True,
    help="With --hybrid, show a few grep hit lines under each matching paper.",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    use_grep: bool,
    fixed_strings: bool,
    context_lines: int,
    max_matches: int,
    ignore_case: bool,
    as_json: bool,
    fuzzy: bool,
    tex: bool,
    use_fts: bool,
    hybrid: bool,
    show_grep_hits: bool,
):
    """Search papers by title, tags, metadata, and local content."""
    grep_only_params = ("fixed_strings", "context_lines", "max_matches", "ignore_case", "as_json")
    if not use_grep:
        flag_text = {
            "fixed_strings": "--fixed-strings/--regex",
            "context_lines": "--context",
            "max_matches": "--max-matches",
            "ignore_case": "--ignore-case/--case-sensitive",
            "as_json": "--json",
        }
        used = [p for p in grep_only_params if ctx.get_parameter_source(p) != click.core.ParameterSource.DEFAULT]
        if used:
            flags = ", ".join(flag_text.get(p, f"--{p.replace('_', '-')}") for p in used)
            raise click.UsageError(f"{flags} only apply with --grep/--rg.")

    if use_grep and ctx.get_parameter_source("use_fts") != click.core.ParameterSource.DEFAULT:
        raise click.UsageError("--fts/--no-fts do not apply with --grep/--rg.")

    if hybrid and use_grep:
        raise click.UsageError("--hybrid does not apply with --grep/--rg (use one or the other).")
    if hybrid and not use_fts:
        raise click.UsageError("--hybrid requires --fts (disable hybrid or drop --no-fts).")
    if show_grep_hits and not hybrid:
        raise click.UsageError("--show-grep-hits requires --hybrid.")

    # Default search mode (env/config) only applies when the user didn't explicitly choose.
    mode = default_search_mode()
    use_grep_source = ctx.get_parameter_source("use_grep")
    use_fts_source = ctx.get_parameter_source("use_fts")
    hybrid_source = ctx.get_parameter_source("hybrid")

    if (
        use_grep_source == click.core.ParameterSource.DEFAULT
        and use_fts_source == click.core.ParameterSource.DEFAULT
        and hybrid_source == click.core.ParameterSource.DEFAULT
    ):
        if mode == "scan":
            use_fts = False
            hybrid = False
        elif mode == "fts":
            use_fts = True
            hybrid = False
        elif mode == "hybrid":
            use_fts = True
            hybrid = True
        else:
            # auto: keep current behavior (fts if db exists, else scan)
            pass

    if use_grep:
        handled = _search_grep(
            query=query,
            fixed_strings=fixed_strings,
            context_lines=context_lines,
            max_matches=max_matches,
            ignore_case=ignore_case,
            as_json=as_json,
            include_tex=tex,
        )
        if handled:
            return
        use_fts = False
        hybrid = False

    if hybrid:
        db_path = _search_db_path()
        if not db_path.exists():
            # For configured default "hybrid", degrade to normal search rather than erroring.
            if (
                default_search_mode() == "hybrid"
                and ctx.get_parameter_source("hybrid") == click.core.ParameterSource.DEFAULT
            ):
                hybrid = False
            else:
                raise click.ClickException(
                    "Hybrid search requires `search.db`. Build it first: `papi index --backend search --search-rebuild`."
                )

        if hybrid:
            fts_results = _search_fts(query=query, limit=max(limit, 50))
            grep_matches = _collect_grep_matches(
                query=query,
                fixed_strings=True,
                max_matches=200,
                ignore_case=True,
                include_tex=tex,
            )
        else:
            fts_results = []
            grep_matches = []
        grep_by_paper: dict[str, list[dict[str, object]]] = {}
        for m in grep_matches:
            paper = str(m.get("paper") or "")
            if paper:
                grep_by_paper.setdefault(paper, []).append(m)

        fts_by_name = {str(r["name"]): r for r in fts_results}
        candidates = set(fts_by_name.keys()) | set(grep_by_paper.keys())

        def fts_score(name: str) -> float:
            raw = (fts_by_name.get(name) or {}).get("score")
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                try:
                    return float(raw)
                except ValueError:
                    return 0.0
            return 0.0

        def sort_key(name: str) -> tuple[int, float, int, str]:
            grep_count = len(grep_by_paper.get(name, []))
            score = fts_score(name)
            return (1 if grep_count > 0 else 0, score, grep_count, name)

        ranked = sorted(candidates, key=sort_key, reverse=True)[:limit]
        if not ranked:
            click.echo(f"No papers found matching '{query}'")
            return

        idx = load_index()
        for name in ranked:
            score = fts_score(name)
            grep_count = len(grep_by_paper.get(name, []))
            title = str((idx.get(name, {}) or {}).get("title") or fts_by_name.get(name, {}).get("title") or "Unknown")[
                :80
            ]
            if grep_count:
                click.echo(f"{name} (score: {score:.6g}, grep: {grep_count})")
            else:
                click.echo(f"{name} (score: {score:.6g})")
            if title:
                click.echo(f"  {title}...")
            if show_grep_hits and grep_count:
                for hit in grep_by_paper[name][:3]:
                    click.echo(f"  {hit.get('path')}:{hit.get('line')}: {str(hit.get('text') or '').strip()}")
            click.echo()
        return

    if use_fts and _search_db_path().exists():
        fts_results = _search_fts(query=query, limit=limit)

        if fts_results:
            for r in fts_results:
                click.echo(f"{r['name']} (score: {r['score']:.6g})")
                title = str(r.get("title") or "Unknown")[:80]
                if title:
                    click.echo(f"  {title}...")
                click.echo()
            return

    index = load_index()

    def collect_results(*, fuzzy_mode: bool) -> list[tuple[str, dict, int, list[str]]]:
        results: list[tuple[str, dict, int, list[str]]] = []
        for name, info in index.items():
            paper_dir = config.PAPERS_DIR / name
            meta_path = paper_dir / "meta.json"

            matched_fields: list[str] = []
            score = 0

            def add_field(field: str, text: str, weight: float) -> None:
                nonlocal score
                field_score = _fuzzy_text_score(query, text, fuzzy=fuzzy_mode)
                if field_score <= 0:
                    return
                score += int(100 * weight * field_score)
                matched_fields.append(field)

            add_field("name", name, 1.6)
            add_field("title", info.get("title", ""), 1.4)
            add_field("tags", " ".join(info.get("tags", [])), 1.2)
            add_field("arxiv_id", info.get("arxiv_id", ""), 1.0)

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    meta = {}
                add_field("authors", " ".join(meta.get("authors", []) or []), 0.6)
                add_field("abstract", meta.get("abstract", "") or "", 0.9)

            summary_path = paper_dir / "summary.md"
            if summary_path.exists():
                add_field("summary", _read_text_limited(summary_path, max_chars=80_000), 0.9)

            equations_path = paper_dir / "equations.md"
            if equations_path.exists():
                add_field("equations", _read_text_limited(equations_path, max_chars=80_000), 0.9)

            if tex:
                source_path = paper_dir / "source.tex"
                if source_path.exists():
                    add_field("source", _read_text_limited(source_path, max_chars=200_000), 0.5)

            if score > 0:
                results.append((name, info, score, matched_fields))

        results.sort(key=lambda x: (-x[2], x[0]))
        return results

    # Exact pass first; only fall back to fuzzy if enabled and no exact matches exist.
    results = collect_results(fuzzy_mode=False)
    if not results and fuzzy:
        results = collect_results(fuzzy_mode=True)

    if not results:
        click.echo(f"No papers found matching '{query}'")
        return

    for name, info, score, matched_fields in results[:limit]:
        click.echo(f"{name} (score: {score})")
        click.echo(f"  {info.get('title', 'Unknown')[:60]}...")
        if matched_fields:
            click.echo(f"  Matches: {', '.join(matched_fields[:6])}")
        click.echo()


@cli.command()
@click.argument("papers", nargs=-1)
@click.option("--all", "audit_all", is_flag=True, help="Audit all papers (default).")
@click.option("--limit", type=int, default=None, help="Audit only N random papers.")
@click.option("--seed", type=int, default=None, help="Random seed for --limit sampling.")
@click.option(
    "--interactive/--no-interactive", default=None, help="Prompt to regenerate flagged papers (default: auto)."
)
@click.option("--regenerate", "do_regenerate", is_flag=True, help="Regenerate all flagged papers.")
@click.option("--no-llm", is_flag=True, help="Use non-LLM regeneration when regenerating.")
@click.option(
    "--overwrite",
    "-o",
    default="summary,equations,tags",
    help="Overwrite fields when regenerating (all or list: summary,equations,tags,name).",
)
def audit(
    papers: tuple[str, ...],
    audit_all: bool,
    limit: Optional[int],
    seed: Optional[int],
    interactive: Optional[bool],
    do_regenerate: bool,
    no_llm: bool,
    overwrite: str,
):
    """Audit generated summaries/equations for obvious issues and optionally regenerate flagged papers."""
    index = load_index()
    overwrite_fields, overwrite_all = _parse_overwrite_option(overwrite)

    if audit_all and papers:
        raise click.UsageError("Use either paper(s)/arXiv id(s) OR `--all`, not both.")

    if not audit_all and not papers:
        audit_all = True

    if audit_all:
        names = sorted(index.keys())
    else:
        names = []
        for paper_ref in papers:
            name, error = _resolve_paper_name_from_ref(paper_ref, index)
            if not name:
                raise click.UsageError(error)
            names.append(name)

    if not names:
        click.echo("No papers found.")
        return

    if limit is not None:
        if limit <= 0:
            raise click.UsageError("--limit must be > 0")
        import random

        rng = random.Random(seed)
        if limit < len(names):
            names = rng.sample(names, k=limit)

    flagged: list[tuple[str, list[str]]] = []
    ok_count = 0
    for name in names:
        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            flagged.append((name, ["missing paper directory"]))
            continue
        reasons = _audit_paper_dir(paper_dir)
        if reasons:
            flagged.append((name, reasons))
        else:
            ok_count += 1

    click.echo(f"Audited {len(names)} paper(s): {ok_count} OK, {len(flagged)} flagged")
    if not flagged:
        return

    click.echo()
    for name, reasons in flagged:
        click.secho(f"{name}: FLAGGED", fg="yellow")
        for reason in reasons:
            click.echo(f"  - {reason}")

    auto_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    effective_interactive = interactive if interactive is not None else auto_interactive

    if do_regenerate:
        selected_names = [name for name, _ in flagged]
    elif effective_interactive:
        click.echo()
        if not click.confirm("Regenerate any flagged papers now?", default=False):
            return
        click.echo("Select papers by number (e.g. 1,3-5) or 'all':")
        for i, (name, _) in enumerate(flagged, 1):
            click.echo(f"  {i}. {name}")
        try:
            spec = click.prompt("Selection", default="all", show_default=True)
            picks = _parse_selection_spec(spec, max_index=len(flagged))
        except Exception as exc:
            raise click.ClickException(f"Invalid selection: {exc}") from exc
        selected_names = [flagged[i - 1][0] for i in picks]
    else:
        return

    if not selected_names:
        return

    reasons_by_name = {n: r for n, r in flagged}
    failures = 0
    click.echo()
    for i, name in enumerate(selected_names, 1):
        echo_progress(f"[{i}/{len(selected_names)}] {name}")
        success, _new_name = _regenerate_one_paper(
            name,
            index,
            no_llm=no_llm,
            overwrite_fields=overwrite_fields,
            overwrite_all=overwrite_all,
            audit_reasons=reasons_by_name.get(name),
        )
        if not success:
            failures += 1

    if failures:
        raise click.ClickException(f"{failures} paper(s) failed to regenerate.")


@cli.command()
@click.argument("papers", nargs=-1, required=True)
@click.option(
    "--level",
    "-l",
    type=click.Choice(["summary", "equations", "eq", "full"], case_sensitive=False),
    default="summary",
    help="What to export",
)
@click.option(
    "--to",
    "dest",
    type=click.Path(),
    help="Destination directory",
)
def export(papers: tuple[str, ...], level: str, dest: Optional[str]):
    """Export paper context for a coding session."""
    level_norm = (level or "").strip().lower()
    if level_norm == "eq":
        level_norm = "equations"

    index = load_index()

    if dest == "-":
        raise click.UsageError(
            "Use `papi show ... --level ...` to print to stdout; `export` only writes to a directory."
        )

    dest_path = Path(dest) if dest else Path.cwd() / "paper-context"
    dest_path.mkdir(exist_ok=True)

    if level_norm == "summary":
        src_name = "summary.md"
        out_suffix = "_summary.md"
        missing_msg = "No summary found"
    elif level_norm == "equations":
        src_name = "equations.md"
        out_suffix = "_equations.md"
        missing_msg = "No equations found"
    else:  # full
        src_name = "source.tex"
        out_suffix = ".tex"
        missing_msg = "No LaTeX source found"

    successes = 0
    failures = 0

    for paper_ref in papers:
        name, error = _resolve_paper_name_from_ref(paper_ref, index)
        if not name:
            echo_error(error)
            failures += 1
            continue

        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            echo_error(f"Paper not found: {paper_ref}")
            failures += 1
            continue

        src = paper_dir / src_name
        if not src.exists():
            echo_error(f"{missing_msg}: {name}")
            failures += 1
            continue

        dest_file = dest_path / f"{name}{out_suffix}"
        shutil.copy(src, dest_file)
        successes += 1

    if failures == 0:
        echo_success(f"Exported {successes} paper(s) to {dest_path}")
        return

    echo_warning(f"Exported {successes} paper(s), {failures} failed (see errors above).")
    raise SystemExit(1)


@cli.command("index", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "--backend",
    type=click.Choice(["pqa", "leann", "search"], case_sensitive=False),
    default="pqa",
    show_default=True,
    help="Which backend to index for: PaperQA2 (`pqa`), LEANN (`leann`), or SQLite FTS5 (`search`).",
)
@click.option(
    "--pqa-llm",
    default=None,
    show_default=False,
    help="PaperQA2 LLM model (LiteLLM id).",
)
@click.option(
    "--pqa-summary-llm",
    default=None,
    show_default=False,
    help="PaperQA2 summary LLM model (LiteLLM id).",
)
@click.option(
    "--pqa-embedding",
    default=None,
    show_default=False,
    help="PaperQA2 embedding model (LiteLLM id).",
)
@click.option("--pqa-temperature", type=float, default=None, show_default=False, help="PaperQA2 temperature.")
@click.option("--pqa-verbosity", type=int, default=None, show_default=False, help="PaperQA2 verbosity (0-3).")
@click.option("--pqa-concurrency", type=int, default=None, show_default=False, help="PaperQA2 indexing concurrency.")
@click.option("--pqa-rebuild-index", is_flag=True, help="Force PaperQA2 rebuild (equivalent to --agent.rebuild_index).")
@click.option("--pqa-retry-failed", is_flag=True, help="Clear PaperQA2 ERROR markers so failed PDFs retry.")
@click.option(
    "--pqa-raw",
    is_flag=True,
    help="Pass through PaperQA2 output without filtering (also enabled by global -v/--verbose).",
)
@click.option("--leann-index", default="papers", show_default=True, help="LEANN index name to build.")
@click.option("--leann-force", is_flag=True, help="Force LEANN rebuild (passes --force to `leann build`).")
@click.option(
    "--leann-backend-name",
    type=click.Choice(["hnsw", "diskann"], case_sensitive=False),
    default=None,
    help="LEANN backend name (maps to `leann build --backend-name`).",
)
@click.option(
    "--leann-embedding-model",
    default=None,
    help="LEANN embedding model (maps to `leann build --embedding-model`).",
)
@click.option(
    "--leann-embedding-mode",
    type=click.Choice(["sentence-transformers", "openai", "mlx", "ollama"], case_sensitive=False),
    default=None,
    help="LEANN embedding mode (maps to `leann build --embedding-mode`).",
)
@click.option(
    "--leann-embedding-host",
    default=None,
    help="LEANN embedding host (maps to `leann build --embedding-host`).",
)
@click.option(
    "--leann-embedding-api-base",
    default=None,
    help="LEANN embedding API base URL (maps to `leann build --embedding-api-base`).",
)
@click.option(
    "--leann-embedding-api-key",
    default=None,
    help="LEANN embedding API key (maps to `leann build --embedding-api-key`).",
)
@click.option(
    "--leann-graph-degree",
    type=int,
    default=None,
    help="LEANN build graph degree (maps to `leann build --graph-degree`).",
)
@click.option(
    "--leann-build-complexity",
    type=int,
    default=None,
    help="LEANN build complexity (maps to `leann build --complexity`).",
)
@click.option(
    "--leann-num-threads",
    type=int,
    default=None,
    help="LEANN build thread count (maps to `leann build --num-threads`).",
)
@click.option(
    "--leann-doc-chunk-size",
    type=int,
    default=None,
    help="LEANN PDF chunk size in TOKENS (maps to `leann build --doc-chunk-size`).",
)
@click.option(
    "--leann-doc-chunk-overlap",
    type=int,
    default=None,
    help="LEANN PDF chunk overlap in TOKENS (maps to `leann build --doc-chunk-overlap`).",
)
@click.option("--search-rebuild", is_flag=True, help="Rebuild the SQLite FTS search index from scratch.")
@click.option(
    "--search-include-tex/--search-no-include-tex",
    default=False,
    show_default=True,
    help="Index `source.tex` contents into the FTS index (larger DB; slower build).",
)
@click.pass_context
def index_cmd(
    ctx: click.Context,
    backend: str,
    pqa_llm: Optional[str],
    pqa_summary_llm: Optional[str],
    pqa_embedding: Optional[str],
    pqa_temperature: Optional[float],
    pqa_verbosity: Optional[int],
    pqa_concurrency: Optional[int],
    pqa_rebuild_index: bool,
    pqa_retry_failed: bool,
    pqa_raw: bool,
    leann_index: str,
    leann_force: bool,
    leann_backend_name: Optional[str],
    leann_embedding_model: Optional[str],
    leann_embedding_mode: Optional[str],
    leann_embedding_host: Optional[str],
    leann_embedding_api_base: Optional[str],
    leann_embedding_api_key: Optional[str],
    leann_graph_degree: Optional[int],
    leann_build_complexity: Optional[int],
    leann_num_threads: Optional[int],
    leann_doc_chunk_size: Optional[int],
    leann_doc_chunk_overlap: Optional[int],
    search_rebuild: bool,
    search_include_tex: bool,
) -> None:
    """Build/update the retrieval index for PaperQA2 (default), LEANN, or SQLite FTS5."""
    backend_norm = (backend or "pqa").strip().lower()
    if backend_norm == "leann":
        leann_index = _effective_leann_index_name(
            ctx=ctx,
            param_name="leann_index",
            raw_index_name=leann_index,
            embedding_mode=leann_embedding_mode,
            embedding_model=leann_embedding_model,
        )
        config.PAPER_DB.mkdir(parents=True, exist_ok=True)
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)

        if any(arg == "--docs" or arg.startswith("--docs=") for arg in ctx.args):
            raise click.UsageError("paperpipe controls LEANN --docs; do not pass --docs.")

        staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
        paperqa._refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)

        leann_extra_args: list[str] = []

        def _append_str_flag(flag: str, value: Optional[str]) -> None:
            if value is None:
                return
            value = value.strip()
            if value:
                leann_extra_args.extend([flag, value])

        def _append_int_flag(flag: str, value: Optional[int]) -> None:
            if value is None:
                return
            leann_extra_args.extend([flag, str(value)])

        _append_str_flag("--backend-name", leann_backend_name)
        _append_str_flag("--embedding-model", leann_embedding_model)
        _append_str_flag("--embedding-mode", leann_embedding_mode)
        _append_str_flag("--embedding-host", leann_embedding_host)
        _append_str_flag("--embedding-api-base", leann_embedding_api_base)
        _append_str_flag("--embedding-api-key", leann_embedding_api_key)
        _append_int_flag("--graph-degree", leann_graph_degree)
        _append_int_flag("--complexity", leann_build_complexity)
        _append_int_flag("--num-threads", leann_num_threads)
        _append_int_flag("--doc-chunk-size", leann_doc_chunk_size)
        _append_int_flag("--doc-chunk-overlap", leann_doc_chunk_overlap)

        leann_extra_args.extend(list(ctx.args))
        _leann_build_index(index_name=leann_index, docs_dir=staging_dir, force=leann_force, extra_args=leann_extra_args)
        echo_success(f"Built LEANN index {leann_index!r} under {config.PAPER_DB / '.leann' / 'indexes' / leann_index}")
        return

    if backend_norm == "search":
        if not search_rebuild and search_include_tex:
            raise click.UsageError("--search-include-tex only applies with --search-rebuild")

        db_path = _search_db_path()
        if search_rebuild or not db_path.exists():
            count = _search_index_rebuild(include_tex=search_include_tex)
            echo_success(f"Built search index for {count} paper(s) at {db_path}")
            return

        with _sqlite_connect(db_path) as conn:
            _ensure_search_index_schema(conn)
            idx = load_index()
            count = 0
            for name in sorted(idx.keys()):
                _search_index_upsert(conn, name=name, index=idx)
                count += 1
            echo_success(f"Updated search index for {count} paper(s) at {db_path}")
        return

    if backend_norm != "pqa":
        raise click.UsageError(f"Unknown --backend: {backend}")

    if pqa_concurrency is not None and pqa_concurrency < 1:
        raise click.UsageError("--pqa-concurrency must be >= 1")

    if not shutil.which("pqa"):
        echo_error("PaperQA2 not installed. Install with: pip install 'paperpipe[paperqa]' (Python 3.11+).")
        raise SystemExit(1)

    cmd = ["pqa"]

    has_settings_flag = any(arg in {"--settings", "-s"} or arg.startswith("--settings=") for arg in ctx.args)
    if not has_settings_flag:
        cmd.extend(["--settings", default_pqa_settings_name()])

    has_parsing_override = any(
        arg == "--parsing" or arg.startswith("--parsing.") or arg.startswith("--parsing=") for arg in ctx.args
    )
    if not has_parsing_override and not paperqa._pillow_available():
        cmd.extend(["--parsing.multimodal", "OFF"])

    pqa_llm_source = ctx.get_parameter_source("pqa_llm")
    pqa_embedding_source = ctx.get_parameter_source("pqa_embedding")
    pqa_summary_llm_source = ctx.get_parameter_source("pqa_summary_llm")
    pqa_temperature_source = ctx.get_parameter_source("pqa_temperature")
    pqa_verbosity_source = ctx.get_parameter_source("pqa_verbosity")

    llm_for_pqa: Optional[str] = None
    embedding_for_pqa: Optional[str] = None
    embedding_for_pqa_cmd: Optional[str] = None

    if pqa_llm_source != click.core.ParameterSource.DEFAULT:
        llm_for_pqa = pqa_llm
    elif not has_settings_flag:
        llm_for_pqa = default_pqa_llm_model()

    if pqa_embedding_source != click.core.ParameterSource.DEFAULT:
        embedding_for_pqa = pqa_embedding
    elif not has_settings_flag:
        embedding_for_pqa = default_pqa_embedding_model()

    embedding_for_pqa_cmd = embedding_for_pqa
    embedding_for_pqa_stripped = _strip_ollama_prefix(embedding_for_pqa)
    if embedding_for_pqa_stripped and embedding_for_pqa_stripped != embedding_for_pqa:
        # Work around a LiteLLM bug where `ollama/...` isn't stripped before calling Ollama's /api/embed.
        # PaperQA2/LMI calls litellm.aembedding(model=<embedding>), so we pass the bare model name and
        # force the provider via embedding_config kwargs.
        embedding_for_pqa_cmd = embedding_for_pqa_stripped

    if llm_for_pqa:
        cmd.extend(["--llm", llm_for_pqa])
    if embedding_for_pqa_cmd:
        cmd.extend(["--embedding", embedding_for_pqa_cmd])

    # Persist index under paper DB unless overridden
    has_index_dir_override = any(
        arg == "--agent.index.index_directory"
        or arg == "--agent.index.index-directory"
        or arg.startswith(("--agent.index.index_directory=", "--agent.index.index-directory="))
        for arg in ctx.args
    )
    if not has_index_dir_override:
        index_dir = default_pqa_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--agent.index.index_directory", str(index_dir)])

    # Stable index name based on embedding (unless overridden)
    embedding_for_name = embedding_for_pqa or (default_pqa_embedding_model() if not has_settings_flag else None)
    has_index_name_override = any(
        arg in {"--index", "-i", "--agent.index.name"} or arg.startswith(("--index=", "--agent.index.name="))
        for arg in ctx.args
    )
    if not has_index_name_override and embedding_for_name:
        # For the `pqa index` subcommand, the index name is controlled by the global `--index/-i` flag
        # (PaperQA2's `build_index()` overrides `agent.index.name` when --index is "default").
        cmd.extend(["--index", pqa_index_name_for_embedding(embedding_for_name)])

    # Determine effective index params to exclude ERROR-marked PDFs when staging.
    effective_args = cmd + list(ctx.args)
    idx_dir_val = paperqa._extract_flag_value(
        effective_args, names={"--agent.index.index_directory", "--agent.index.index-directory"}
    )
    idx_name_val = paperqa._extract_flag_value(effective_args, names={"--index", "-i", "--agent.index.name"})
    excluded_files: set[str] = set()
    if idx_dir_val and idx_name_val and not pqa_retry_failed:
        fp = paperqa._paperqa_index_files_path(index_directory=Path(idx_dir_val), index_name=idx_name_val)
        if fp.exists():
            m = paperqa._paperqa_load_index_files_map(fp)
            if m:
                excluded_files = {Path(k).name for k, v in m.items() if v == "ERROR"}

    # Paper directory (defaults to managed staging dir)
    paper_dir_override = paperqa._extract_flag_value(
        list(ctx.args),
        names={"--agent.index.paper_directory", "--agent.index.paper-directory"},
    )
    if paper_dir_override:
        paper_dir = Path(paper_dir_override).expanduser()
    else:
        paper_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
        paperqa._refresh_pqa_pdf_staging_dir(staging_dir=paper_dir, exclude_names=excluded_files)
        cmd.extend(["--agent.index.paper_directory", str(paper_dir)])

    has_sync_override = any(
        arg == "--agent.index.sync_with_paper_directory"
        or arg == "--agent.index.sync-with-paper-directory"
        or arg.startswith(
            (
                "--agent.index.sync_with_paper_directory=",
                "--agent.index.sync-with-paper-directory=",
            )
        )
        for arg in ctx.args
    )
    if not has_sync_override:
        cmd.extend(["--agent.index.sync_with_paper_directory", "true"])

    # summary_llm
    llm_effective = llm_for_pqa
    if pqa_summary_llm_source != click.core.ParameterSource.DEFAULT:
        if pqa_summary_llm:
            cmd.extend(["--summary_llm", pqa_summary_llm])
    else:
        summary_llm_default = default_pqa_summary_llm(llm_effective)
        if summary_llm_default:
            cmd.extend(["--summary_llm", summary_llm_default])

    # enrichment_llm: config/env default only (no first-class option)
    enrichment_llm_default = default_pqa_enrichment_llm(llm_effective)
    has_enrichment_llm_override = any(
        arg == "--parsing.enrichment_llm"
        or arg == "--parsing.enrichment-llm"
        or arg.startswith(("--parsing.enrichment_llm=", "--parsing.enrichment-llm="))
        for arg in ctx.args
    )
    if enrichment_llm_default and not has_enrichment_llm_override:
        cmd.extend(["--parsing.enrichment_llm", enrichment_llm_default])

    # temperature
    if pqa_temperature_source != click.core.ParameterSource.DEFAULT:
        if pqa_temperature is not None:
            cmd.extend(["--temperature", str(pqa_temperature)])
    else:
        temperature_default = default_pqa_temperature()
        if temperature_default is not None:
            cmd.extend(["--temperature", str(temperature_default)])

    # verbosity
    if pqa_verbosity_source != click.core.ParameterSource.DEFAULT:
        if pqa_verbosity is not None:
            cmd.extend(["--verbosity", str(pqa_verbosity)])
    else:
        verbosity_default = default_pqa_verbosity()
        if verbosity_default is not None:
            cmd.extend(["--verbosity", str(verbosity_default)])

    if pqa_concurrency is not None:
        cmd.extend(["--agent.index.concurrency", str(pqa_concurrency)])
    else:
        has_concurrency_passthrough = any(
            arg in {"--agent.index.concurrency"} or arg.startswith("--agent.index.concurrency=") for arg in ctx.args
        )
        if not has_concurrency_passthrough:
            cmd.extend(["--agent.index.concurrency", str(default_pqa_concurrency())])

    if pqa_rebuild_index and not any(
        arg in {"--agent.rebuild_index", "--agent.rebuild-index"}
        or arg.startswith(("--agent.rebuild_index=", "--agent.rebuild-index="))
        for arg in ctx.args
    ):
        cmd.extend(["--agent.rebuild_index", "true"])

    # If the embedding is an Ollama model id, inject an embedding_config that forces the provider
    # while keeping the user-friendly `ollama/...` id for index naming.
    #
    # Note: `--embedding_config` is a *global* pqa flag and must come before the `index` subcommand.
    if (
        embedding_for_pqa
        and _is_ollama_model_id(embedding_for_pqa)
        and not paperqa._pqa_has_flag(ctx.args, names={"--embedding_config", "--embedding-config"})
    ):
        ollama_timeout = default_pqa_ollama_timeout()
        cmd.extend(
            [
                "--embedding_config",
                json.dumps(
                    {"kwargs": {"custom_llm_provider": "ollama", "timeout": ollama_timeout}},
                    separators=(",", ":"),
                ),
            ]
        )

    cmd.extend(ctx.args)

    # Clear ERROR markers if requested (PaperQA2 won't retry failed docs otherwise)
    index_dir_raw = paperqa._extract_flag_value(
        cmd,
        names={"--agent.index.index_directory", "--agent.index.index-directory"},
    )
    index_name_raw = paperqa._extract_flag_value(cmd, names={"--agent.index.name"}) or paperqa._extract_flag_value(
        cmd, names={"--index", "-i"}
    )
    if pqa_retry_failed and index_dir_raw and index_name_raw:
        cleared, _ = paperqa._paperqa_clear_failed_documents(
            index_directory=Path(index_dir_raw), index_name=index_name_raw
        )
        if cleared:
            echo_progress(f"Cleared {cleared} failed PaperQA2 document(s) for retry.")

    cmd.extend(["index", str(paper_dir)])

    env = os.environ.copy()
    env.setdefault("PQA_LITELLM_MAX_CALLBACKS", "1000")
    env.setdefault("LMI_LITELLM_MAX_CALLBACKS", "1000")

    if _is_ollama_model_id(llm_for_pqa) or _is_ollama_model_id(embedding_for_pqa):
        config._prepare_ollama_env(env)
        err = config._ollama_reachability_error(api_base=env["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            raise SystemExit(1)

    proc = subprocess.Popen(
        cmd,
        cwd=config.PAPERS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    root_verbose = bool(ctx.find_root().params.get("verbose"))
    passthrough_verbose = any(arg in {"--verbose", "-v"} for arg in ctx.args)
    raw_output = bool(
        pqa_raw
        or root_verbose
        or passthrough_verbose
        or (pqa_verbosity_source != click.core.ParameterSource.DEFAULT and (pqa_verbosity or 0) > 0)
    )

    captured_output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured_output.append(line)
        if raw_output:
            click.echo(line, nl=False)
        elif not paperqa._pqa_is_noisy_index_line(line):
            click.echo(line, nl=False)
    returncode = proc.wait()
    if returncode != 0:
        if not raw_output:
            paperqa._pqa_print_filtered_index_output_on_failure(captured_output=captured_output)
        raise SystemExit(returncode)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("query")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "evidence-blocks"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format. Use 'evidence-blocks' for structured JSON with answer + cited evidence snippets.",
)
@click.option(
    "--pqa-llm",
    "llm",
    default=None,
    show_default=False,
    help=("LLM model for answer generation (LiteLLM id; e.g., gpt-4o, claude-sonnet-4-5, gemini/gemini-2.5-flash)."),
)
@click.option(
    "--pqa-summary-llm",
    "summary_llm",
    default=None,
    show_default=False,
    help="LLM for evidence summarization (often a cheaper/faster model than --pqa-llm).",
)
@click.option(
    "--pqa-embedding",
    "embedding",
    default=None,
    show_default=False,
    help="Embedding model for text chunks (e.g., text-embedding-3-small, voyage-3-lite).",
)
@click.option(
    "--pqa-temperature",
    "temperature",
    type=float,
    default=None,
    show_default=False,
    help="LLM temperature (0.0-1.0). Lower = more deterministic.",
)
@click.option(
    "--pqa-verbosity",
    "verbosity",
    type=int,
    default=None,
    show_default=False,
    help="Logging verbosity level (0-3). 3 = log all LLM/embedding calls.",
)
@click.option(
    "--pqa-agent-type",
    "agent_type",
    default=None,
    show_default=False,
    help="PaperQA2 agent type (e.g., 'fake' for deterministic low-token retrieval).",
)
@click.option(
    "--pqa-answer-length",
    "answer_length",
    default=None,
    show_default=False,
    help="Target answer length (e.g., 'about 200 words', 'short', '3 paragraphs').",
)
@click.option(
    "--pqa-evidence-k",
    "evidence_k",
    type=int,
    default=None,
    show_default=False,
    help="Number of evidence pieces to retrieve (default: 10).",
)
@click.option(
    "--pqa-max-sources",
    "max_sources",
    type=int,
    default=None,
    show_default=False,
    help="Maximum number of sources to cite in the answer (default: 5).",
)
@click.option(
    "--pqa-timeout",
    "timeout",
    type=float,
    default=None,
    show_default=False,
    help="Agent timeout in seconds (default: 500).",
)
@click.option(
    "--pqa-concurrency",
    "concurrency",
    type=int,
    default=None,
    show_default=False,
    help="Indexing concurrency (default: 1). Higher values speed up indexing but may cause rate limits.",
)
@click.option(
    "--pqa-rebuild-index",
    "rebuild_index",
    is_flag=True,
    default=False,
    help="Force a full rebuild of the paper index.",
)
@click.option(
    "--pqa-retry-failed",
    "retry_failed",
    is_flag=True,
    help="Retry docs previously marked failed (clears ERROR markers in the index).",
)
@click.option(
    "--pqa-raw",
    is_flag=True,
    help="Pass through PaperQA2 output without filtering (also enabled by global -v/--verbose).",
)
@click.option(
    "--backend",
    type=click.Choice(["pqa", "leann"], case_sensitive=False),
    default="pqa",
    show_default=True,
    help="Backend to use: PaperQA2 via `pqa` (default) or LEANN via `leann`.",
)
@click.option(
    "--leann-index",
    default="papers",
    show_default=True,
    help="LEANN index name (stored under <paper_db>/.leann/indexes).",
)
@click.option(
    "--leann-provider",
    type=click.Choice(["simulated", "ollama", "hf", "openai", "anthropic"], case_sensitive=False),
    default=None,
    show_default=False,
    help="LEANN LLM provider (maps to `leann ask --llm ...`).",
)
@click.option("--leann-model", default=None, show_default=False, help="LEANN model name (maps to `leann ask --model`).")
@click.option(
    "--leann-host",
    default=None,
    show_default=False,
    help="Override Ollama-compatible host (maps to `leann ask --host`).",
)
@click.option(
    "--leann-api-base",
    default=None,
    show_default=False,
    help="Base URL for OpenAI-compatible APIs (maps to `leann ask --api-base`).",
)
@click.option(
    "--leann-api-key",
    default=None,
    show_default=False,
    help="API key for cloud LLM providers (maps to `leann ask --api-key`).",
)
@click.option("--leann-top-k", type=int, default=None, show_default=False, help="LEANN retrieval count.")
@click.option("--leann-complexity", type=int, default=None, show_default=False, help="LEANN search complexity.")
@click.option("--leann-beam-width", type=int, default=None, show_default=False, help="LEANN search beam width.")
@click.option("--leann-prune-ratio", type=float, default=None, show_default=False, help="LEANN search prune ratio.")
@click.option(
    "--leann-recompute/--leann-no-recompute",
    default=True,
    show_default=True,
    help="Enable/disable embedding recomputation during LEANN ask.",
)
@click.option(
    "--leann-pruning-strategy",
    type=click.Choice(["global", "local", "proportional"], case_sensitive=False),
    default=None,
    show_default=False,
    help="LEANN pruning strategy.",
)
@click.option(
    "--leann-thinking-budget",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default=None,
    show_default=False,
    help="LEANN thinking budget for supported models.",
)
@click.option("--leann-interactive", is_flag=True, help="Run `leann ask --interactive` (terminal UI).")
@click.option(
    "--leann-auto-index/--leann-no-auto-index",
    default=True,
    show_default=True,
    help="Auto-build the LEANN index if missing when running `papi ask --backend leann`.",
)
@click.pass_context
def ask(
    ctx,
    query: str,
    output_format: str,
    llm: Optional[str],
    summary_llm: Optional[str],
    embedding: Optional[str],
    temperature: Optional[float],
    verbosity: Optional[int],
    agent_type: Optional[str],
    answer_length: Optional[str],
    evidence_k: Optional[int],
    max_sources: Optional[int],
    timeout: Optional[float],
    concurrency: Optional[int],
    rebuild_index: bool,
    retry_failed: bool,
    pqa_raw: bool,
    backend: str,
    leann_index: str,
    leann_provider: Optional[str],
    leann_model: Optional[str],
    leann_host: Optional[str],
    leann_api_base: Optional[str],
    leann_api_key: Optional[str],
    leann_top_k: Optional[int],
    leann_complexity: Optional[int],
    leann_beam_width: Optional[int],
    leann_prune_ratio: Optional[float],
    leann_recompute: bool,
    leann_pruning_strategy: Optional[str],
    leann_thinking_budget: Optional[str],
    leann_interactive: bool,
    leann_auto_index: bool,
):
    """
    Query papers using PaperQA2 (default) or LEANN.

    Common options are exposed as first-class flags. Any additional arguments
    are passed directly to PaperQA2 (e.g., --agent.search_count 10).
    """
    backend_norm = (backend or "pqa").strip().lower()
    output_format_norm = (output_format or "text").strip().lower()
    if backend_norm == "leann":
        leann_index = _effective_leann_index_name(ctx=ctx, param_name="leann_index", raw_index_name=leann_index)
        if output_format_norm != "text":
            raise click.UsageError("--format evidence-blocks is only supported with --backend pqa.")
        config.PAPER_DB.mkdir(parents=True, exist_ok=True)
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        if leann_auto_index:
            index_name = (leann_index or "").strip() or DEFAULT_LEANN_INDEX_NAME
            meta_path = _leann_index_meta_path(index_name)
            if not meta_path.exists():
                echo_progress(f"LEANN index {index_name!r} not found; building it now...")
                staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
                paperqa._refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)
                _leann_build_index(index_name=index_name, docs_dir=staging_dir, force=False, extra_args=[])
        _ask_leann(
            query=query,
            index_name=leann_index,
            provider=leann_provider,
            model=leann_model,
            host=leann_host,
            api_base=leann_api_base,
            api_key=leann_api_key,
            top_k=leann_top_k,
            complexity=leann_complexity,
            beam_width=leann_beam_width,
            prune_ratio=leann_prune_ratio,
            recompute_embeddings=leann_recompute,
            pruning_strategy=leann_pruning_strategy,
            thinking_budget=leann_thinking_budget,
            interactive=leann_interactive,
            extra_args=list(ctx.args),
        )
        return

    if not shutil.which("pqa"):
        if output_format_norm == "evidence-blocks":
            raise click.ClickException(
                "PaperQA2 is required for --format evidence-blocks. Install with: pip install 'paperpipe[paperqa]'"
            )
        echo_error("PaperQA2 not installed. Install with: pip install paper-qa")
        click.echo("\nFalling back to local search...")
        # Do a simple local search instead
        ctx_search = subprocess.run(["papi", "search", query], capture_output=True, text=True)
        click.echo(ctx_search.stdout.rstrip("\n"))
        return

    # Build pqa command
    # pqa [global_options] ask [ask_options] query
    cmd = ["pqa"]
    # PaperQA2 CLI defaults to `--settings high_quality`, which can be overridden by a user's
    # ~/.config/pqa/settings/high_quality.json. If that file is from an older PaperQA version,
    # pqa can crash on startup due to a schema mismatch. Use the special `default` settings
    # (which bypasses JSON config loading) unless the user explicitly passes `--settings/-s`.
    has_settings_flag = any(arg in {"--settings", "-s"} or arg.startswith("--settings=") for arg in ctx.args)
    if not has_settings_flag:
        cmd.extend(["--settings", default_pqa_settings_name()])

    # PaperQA2 can attempt PDF image extraction (multimodal parsing). If Pillow isn't installed,
    # PyPDF raises at import-time when accessing `page.images`. Disable multimodal parsing unless
    # the user explicitly provides parsing settings.
    has_parsing_override = any(
        arg == "--parsing" or arg.startswith("--parsing.") or arg.startswith("--parsing=") for arg in ctx.args
    )
    if not has_parsing_override and not paperqa._pillow_available():
        cmd.extend(["--parsing.multimodal", "OFF"])

    llm_for_pqa: Optional[str] = None
    embedding_for_pqa: Optional[str] = None

    llm_source = ctx.get_parameter_source("llm")
    embedding_source = ctx.get_parameter_source("embedding")

    if llm_source != click.core.ParameterSource.DEFAULT:
        llm_for_pqa = llm
    elif not has_settings_flag:
        llm_for_pqa = default_pqa_llm_model()

    if embedding_source != click.core.ParameterSource.DEFAULT:
        embedding_for_pqa = embedding
    elif not has_settings_flag:
        embedding_for_pqa = default_pqa_embedding_model()

    if llm_for_pqa:
        cmd.extend(["--llm", llm_for_pqa])

    embedding_for_pqa_cmd = embedding_for_pqa
    embedding_for_pqa_stripped = _strip_ollama_prefix(embedding_for_pqa)
    if embedding_for_pqa_stripped and embedding_for_pqa_stripped != embedding_for_pqa:
        # Work around a LiteLLM bug where `ollama/...` isn't stripped before calling Ollama's /api/embed.
        # PaperQA2/LMI calls litellm.aembedding(model=<embedding>), so we pass the bare model name and
        # force the provider via embedding_config kwargs.
        embedding_for_pqa_cmd = embedding_for_pqa_stripped

    if embedding_for_pqa_cmd:
        cmd.extend(["--embedding", embedding_for_pqa_cmd])

    # Persist the PaperQA index under the paper DB by default so repeated queries reuse embeddings.
    # Users can override via explicit pqa args.
    has_index_dir_override = any(
        arg == "--agent.index.index_directory"
        or arg == "--agent.index.index-directory"
        or arg.startswith(("--agent.index.index_directory=", "--agent.index.index-directory="))
        for arg in ctx.args
    )
    if not has_index_dir_override:
        index_dir = default_pqa_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--agent.index.index_directory", str(index_dir)])

    # Set an explicit index name based on the embedding model to ensure the same index is reused
    # across runs. PaperQA2 auto-generates a hash from all settings, which can vary due to
    # dynamic defaults, causing unnecessary re-indexing. Using an explicit name tied to the
    # embedding model ensures index reuse while still creating a new index when the embedding
    # model changes (since embeddings from different models are incompatible).
    has_index_name_override = any(
        arg in {"--index", "-i", "--agent.index.name"} or arg.startswith(("--index=", "--agent.index.name="))
        for arg in ctx.args
    )
    if not has_index_name_override and embedding_for_pqa:
        cmd.extend(["--agent.index.name", pqa_index_name_for_embedding(embedding_for_pqa)])

    # Determine effective index params to check for exclusions (files marked ERROR)
    # We need to look at both what we've built so far and what the user passed
    effective_args = cmd + ctx.args
    idx_dir_val = paperqa._extract_flag_value(
        effective_args, names={"--agent.index.index_directory", "--agent.index.index-directory"}
    )
    idx_name_val = paperqa._extract_flag_value(effective_args, names={"--index", "-i", "--agent.index.name"})

    excluded_files: set[str] = set()
    if idx_dir_val and idx_name_val and not retry_failed:
        # Load index and find errors
        fp = paperqa._paperqa_index_files_path(index_directory=Path(idx_dir_val), index_name=idx_name_val)
        if fp.exists():
            m = paperqa._paperqa_load_index_files_map(fp)
            if m:
                excluded_files = {Path(k).name for k, v in m.items() if v == "ERROR"}

    # PaperQA2 currently indexes Markdown by default; avoid indexing paperpipe's generated `summary.md`
    # / `equations.md` by staging only PDFs in a separate directory.
    has_paper_dir_override = any(
        arg == "--agent.index.paper_directory"
        or arg == "--agent.index.paper-directory"
        or arg.startswith(("--agent.index.paper_directory=", "--agent.index.paper-directory="))
        for arg in ctx.args
    )
    if not has_paper_dir_override:
        staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
        paperqa._refresh_pqa_pdf_staging_dir(staging_dir=staging_dir, exclude_names=excluded_files)
        cmd.extend(["--agent.index.paper_directory", str(staging_dir)])

    # Default to syncing the index with the paper directory so newly-added PDFs are indexed
    # automatically during `papi ask`. Users can override by passing the flag explicitly.
    has_sync_override = any(
        arg == "--agent.index.sync_with_paper_directory"
        or arg == "--agent.index.sync-with-paper-directory"
        or arg.startswith(
            (
                "--agent.index.sync_with_paper_directory=",
                "--agent.index.sync-with-paper-directory=",
            )
        )
        for arg in ctx.args
    )
    if not has_sync_override:
        cmd.extend(["--agent.index.sync_with_paper_directory", "true"])

    # --- Handle first-class options (with fallback to config/env defaults) ---

    summary_llm_for_pqa: Optional[str] = None
    enrichment_llm_for_pqa: Optional[str] = None

    # summary_llm: first-class option takes precedence, then config, then falls back to llm_for_pqa
    summary_llm_source = ctx.get_parameter_source("summary_llm")
    if summary_llm_source != click.core.ParameterSource.DEFAULT:
        # Explicit CLI --pqa-summary-llm takes precedence
        if summary_llm:
            cmd.extend(["--summary_llm", summary_llm])
            summary_llm_for_pqa = summary_llm
    else:
        summary_llm_default = default_pqa_summary_llm(llm_for_pqa)
        if summary_llm_default:
            cmd.extend(["--summary_llm", summary_llm_default])
            summary_llm_for_pqa = summary_llm_default

    # enrichment_llm: config/env default only (no first-class option)
    enrichment_llm_default = default_pqa_enrichment_llm(llm_for_pqa)
    has_enrichment_llm_override = any(
        arg == "--parsing.enrichment_llm"
        or arg == "--parsing.enrichment-llm"
        or arg.startswith(("--parsing.enrichment_llm=", "--parsing.enrichment-llm="))
        for arg in ctx.args
    )
    if enrichment_llm_default and not has_enrichment_llm_override:
        cmd.extend(["--parsing.enrichment_llm", enrichment_llm_default])
        enrichment_llm_for_pqa = enrichment_llm_default

    # Ollama can have long cold-start / first-token latency. If the user didn't provide explicit
    # per-provider configs, inject a larger LiteLLM router timeout to avoid spurious 60s timeouts.
    if (
        _is_ollama_model_id(llm_for_pqa)
        or _is_ollama_model_id(summary_llm_for_pqa)
        or _is_ollama_model_id(enrichment_llm_for_pqa)
        or _is_ollama_model_id(embedding_for_pqa)
    ):
        ollama_timeout = default_pqa_ollama_timeout()
        timeout_config = json.dumps({"router_kwargs": {"timeout": ollama_timeout}})

        if _is_ollama_model_id(llm_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args, names={"--llm_config", "--llm-config"}
        ):
            cmd.extend(["--llm_config", timeout_config])
        if _is_ollama_model_id(summary_llm_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args, names={"--summary_llm_config", "--summary-llm-config"}
        ):
            cmd.extend(["--summary_llm_config", timeout_config])
        if _is_ollama_model_id(enrichment_llm_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args,
            names={
                "--parsing.enrichment_llm_config",
                "--parsing.enrichment-llm-config",
            },
        ):
            cmd.extend(["--parsing.enrichment_llm_config", timeout_config])
        if _is_ollama_model_id(embedding_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args, names={"--embedding_config", "--embedding-config"}
        ):
            # For embeddings, LMI uses PassThroughRouter (no model_list), so router_kwargs doesn't help.
            # Also work around a LiteLLM bug where `ollama/...` isn't stripped before calling /api/embed.
            cmd.extend(
                [
                    "--embedding_config",
                    json.dumps({"kwargs": {"custom_llm_provider": "ollama", "timeout": ollama_timeout}}),
                ]
            )

    # temperature
    temperature_source = ctx.get_parameter_source("temperature")
    if temperature_source != click.core.ParameterSource.DEFAULT:
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
    else:
        temperature_default = default_pqa_temperature()
        if temperature_default is not None:
            cmd.extend(["--temperature", str(temperature_default)])

    # verbosity
    verbosity_source = ctx.get_parameter_source("verbosity")
    if verbosity_source != click.core.ParameterSource.DEFAULT:
        if verbosity is not None:
            cmd.extend(["--verbosity", str(verbosity)])
    else:
        verbosity_default = default_pqa_verbosity()
        if verbosity_default is not None:
            cmd.extend(["--verbosity", str(verbosity_default)])

    # agent_type -> --agent.agent_type
    agent_type_source = ctx.get_parameter_source("agent_type")
    has_agent_type_passthrough = any(
        arg in {"--agent.agent_type", "--agent.agent-type"}
        or arg.startswith(("--agent.agent_type=", "--agent.agent-type="))
        for arg in ctx.args
    )
    if agent_type_source != click.core.ParameterSource.DEFAULT:
        if agent_type:
            cmd.extend(["--agent.agent_type", agent_type])
    elif not has_agent_type_passthrough:
        # No default; only set when explicitly requested.
        pass

    # answer_length -> --answer.answer_length
    answer_length_source = ctx.get_parameter_source("answer_length")
    has_answer_length_passthrough = any(
        arg in {"--answer.answer_length", "--answer.answer-length"}
        or arg.startswith(("--answer.answer_length=", "--answer.answer-length="))
        for arg in ctx.args
    )
    if answer_length_source != click.core.ParameterSource.DEFAULT:
        if answer_length:
            cmd.extend(["--answer.answer_length", answer_length])
    elif not has_answer_length_passthrough:
        answer_length_default = default_pqa_answer_length()
        if answer_length_default:
            cmd.extend(["--answer.answer_length", answer_length_default])

    # evidence_k -> --answer.evidence_k
    evidence_k_source = ctx.get_parameter_source("evidence_k")
    has_evidence_k_passthrough = any(
        arg in {"--answer.evidence_k", "--answer.evidence-k"}
        or arg.startswith(("--answer.evidence_k=", "--answer.evidence-k="))
        for arg in ctx.args
    )
    if evidence_k_source != click.core.ParameterSource.DEFAULT:
        if evidence_k is not None:
            cmd.extend(["--answer.evidence_k", str(evidence_k)])
    elif not has_evidence_k_passthrough:
        evidence_k_default = default_pqa_evidence_k()
        if evidence_k_default is not None:
            cmd.extend(["--answer.evidence_k", str(evidence_k_default)])

    # max_sources -> --answer.answer_max_sources
    max_sources_source = ctx.get_parameter_source("max_sources")
    has_max_sources_passthrough = any(
        arg in {"--answer.answer_max_sources", "--answer.answer-max-sources"}
        or arg.startswith(("--answer.answer_max_sources=", "--answer.answer-max-sources="))
        for arg in ctx.args
    )
    if max_sources_source != click.core.ParameterSource.DEFAULT:
        if max_sources is not None:
            cmd.extend(["--answer.answer_max_sources", str(max_sources)])
    elif not has_max_sources_passthrough:
        max_sources_default = default_pqa_max_sources()
        if max_sources_default is not None:
            cmd.extend(["--answer.answer_max_sources", str(max_sources_default)])

    # timeout -> --agent.timeout
    timeout_source = ctx.get_parameter_source("timeout")
    has_timeout_passthrough = any(arg in {"--agent.timeout"} or arg.startswith("--agent.timeout=") for arg in ctx.args)
    if timeout_source != click.core.ParameterSource.DEFAULT:
        if timeout is not None:
            cmd.extend(["--agent.timeout", str(timeout)])
    elif not has_timeout_passthrough:
        timeout_default = default_pqa_timeout()
        if timeout_default is not None:
            cmd.extend(["--agent.timeout", str(timeout_default)])

    # concurrency -> --agent.index.concurrency
    concurrency_source = ctx.get_parameter_source("concurrency")
    has_concurrency_passthrough = any(
        arg in {"--agent.index.concurrency", "--agent.index.concurrency"}
        or arg.startswith(("--agent.index.concurrency=",))
        for arg in ctx.args
    )
    if concurrency_source != click.core.ParameterSource.DEFAULT:
        if concurrency is not None:
            cmd.extend(["--agent.index.concurrency", str(concurrency)])
    elif not has_concurrency_passthrough:
        concurrency_default = default_pqa_concurrency()
        cmd.extend(["--agent.index.concurrency", str(concurrency_default)])

    # rebuild_index -> --agent.rebuild_index
    has_rebuild_passthrough = any(
        arg in {"--agent.rebuild_index", "--agent.rebuild-index"}
        or arg.startswith(("--agent.rebuild_index=", "--agent.rebuild-index="))
        for arg in ctx.args
    )
    if rebuild_index and not has_rebuild_passthrough:
        cmd.extend(["--agent.rebuild_index", "true"])

    # Add any extra arguments passed after the known options
    cmd.extend(ctx.args)

    # If the index previously recorded failed documents, PaperQA2 will not retry them
    # (they are treated as already processed). Optionally clear those failure markers.
    index_dir_raw = paperqa._extract_flag_value(
        cmd,
        names={"--agent.index.index_directory", "--agent.index.index-directory"},
    )
    index_name_raw = paperqa._extract_flag_value(
        cmd,
        names={"--agent.index.name"},
    ) or paperqa._extract_flag_value(cmd, names={"--index", "-i"})

    if index_dir_raw and index_name_raw:
        files_path = paperqa._paperqa_index_files_path(index_directory=Path(index_dir_raw), index_name=index_name_raw)
        mapping = paperqa._paperqa_load_index_files_map(files_path) if files_path.exists() else None
        failed_count = sum(1 for v in (mapping or {}).values() if v == "ERROR")
        if failed_count and not retry_failed:
            echo_warning(
                f"PaperQA2 index '{index_name_raw}' has {failed_count} failed document(s) (marked ERROR); "
                "PaperQA2 will not retry them automatically. Re-run with --pqa-retry-failed "
                "or --pqa-rebuild-index to rebuild the whole index."
            )
        if retry_failed:
            cleared, cleared_files = paperqa._paperqa_clear_failed_documents(
                index_directory=Path(index_dir_raw),
                index_name=index_name_raw,
            )
            if cleared:
                echo_progress(f"Cleared {cleared} failed PaperQA2 document(s) for retry.")
                debug("Cleared failed PaperQA2 docs: %s", ", ".join(cleared_files[:50]))

    cmd.extend(["ask", query])

    if output_format_norm == "evidence-blocks":
        if ctx.args:
            raise click.UsageError(
                "--format evidence-blocks does not support extra passthrough args. "
                "Use the first-class --pqa-* options instead."
            )
        click.echo(json.dumps(paperqa._paperqa_ask_evidence_blocks(cmd=cmd, query=query), indent=2))
        return

    root_verbose = bool(ctx.find_root().params.get("verbose"))
    passthrough_verbose = any(arg in {"--verbose", "-v"} for arg in ctx.args)
    raw_output = bool(
        pqa_raw
        or root_verbose
        or passthrough_verbose
        or (verbosity_source != click.core.ParameterSource.DEFAULT and (verbosity or 0) > 0)
    )

    env = os.environ.copy()
    # PaperQA2/LiteLLM can register a lot of callbacks during long runs and emit noisy warnings
    # once it reaches the default cap (30). Allow raising this cap via env, and set a higher
    # default to avoid spam when users haven't configured anything.
    env.setdefault("PQA_LITELLM_MAX_CALLBACKS", "1000")
    env.setdefault("LMI_LITELLM_MAX_CALLBACKS", "1000")
    if _is_ollama_model_id(llm_for_pqa) or _is_ollama_model_id(embedding_for_pqa):
        config._prepare_ollama_env(env)
        err = config._ollama_reachability_error(api_base=env["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            raise SystemExit(1)

    # Run pqa while capturing output for crash detection.
    # We merge stderr into stdout so we can preserve ordering.
    proc = subprocess.Popen(
        cmd,
        cwd=config.PAPERS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_output: list[str] = []
    assert proc.stdout is not None  # for type checker
    for line in proc.stdout:
        captured_output.append(line)
        if raw_output:
            click.echo(line, nl=False)
    returncode = proc.wait()

    # Handle pqa failures gracefully
    if returncode != 0:
        if not raw_output:
            paperqa._pqa_print_filtered_output_on_failure(captured_output=captured_output)

        # Try to identify the crashing document from pqa's output
        # pqa prints "New file to index: <filename>..." before processing each file
        crashing_doc: Optional[str] = None
        for line in captured_output:
            if "New file to index:" in line:
                # Extract filename: "New file to index: nmr.pdf..."
                match = re.search(r"New file to index:\s*(\S+)", line)
                if match:
                    crashing_doc = match.group(1).rstrip(".")

        # If we identified the crashing document, mark only that one as ERROR
        if crashing_doc and index_dir_raw and index_name_raw:
            paper_dir = (
                paperqa._paperqa_effective_paper_directory(cmd, base_dir=config.PAPERS_DIR)
                or (config.PAPER_DB / ".pqa_papers").expanduser()
            )
            if paper_dir.exists():
                f = paperqa._paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc=crashing_doc)
                if f is not None:
                    count, _ = paperqa._paperqa_mark_failed_documents(
                        index_directory=Path(index_dir_raw),
                        index_name=index_name_raw,
                        staged_files={str(f)},
                    )
                    if count:
                        # Only remove files from paperpipe's managed staging directory.
                        # Never delete from a user-provided paper directory.
                        managed_staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
                        if paper_dir.resolve() == managed_staging_dir.resolve():
                            try:
                                f.unlink()
                                echo_warning(f"Removed '{crashing_doc}' from PaperQA2 staging to prevent re-indexing.")
                            except OSError:
                                echo_warning(f"Marked '{crashing_doc}' as ERROR to skip on retry.")
                        else:
                            echo_warning(f"Marked '{crashing_doc}' as ERROR to skip on retry.")

        # Show helpful error message
        if index_dir_raw and index_name_raw:
            mapping = paperqa._paperqa_load_index_files_map(
                paperqa._paperqa_index_files_path(index_directory=Path(index_dir_raw), index_name=index_name_raw)
            )
            failed_docs = sorted([k for k, v in (mapping or {}).items() if v == "ERROR"])
            if failed_docs:
                echo_warning(f"PaperQA2 failed. {len(failed_docs)} document(s) excluded from indexing.")
                echo_warning("This can happen with PDFs that have text extraction issues (e.g., surrogate characters).")
                echo_warning("Options:")
                echo_warning("  1. Remove problematic paper(s) entirely: papi remove <name>")
                echo_warning("  2. Re-run query (excluded docs will stay excluded): papi ask '...'")
                echo_warning("  3. Re-stage excluded docs for retry: papi ask '...' --pqa-retry-failed")
                echo_warning("  4. Rebuild index from scratch: papi ask '...' --pqa-rebuild-index")
                if len(failed_docs) <= 5:
                    echo_warning(f"Failed documents: {', '.join(Path(f).stem for f in failed_docs)}")
                raise SystemExit(1)
        # Generic failure message if we can't determine the cause
        echo_error("PaperQA2 failed. Re-run with --pqa-raw or 'papi -v ask ...' for full output.")
        raise SystemExit(returncode)

    if not raw_output:
        for line in captured_output:
            if not paperqa._pqa_is_noisy_stream_line(line):
                click.echo(line, nl=False)


@cli.command()
@click.argument(
    "preset_arg",
    required=False,
    type=click.Choice(["default", "latest", "last-gen", "all"], case_sensitive=False),
)
@click.option(
    "--kind",
    type=click.Choice(["completion", "embedding"], case_sensitive=False),
    multiple=True,
    default=("completion", "embedding"),
    show_default=True,
    help="Which API types to probe.",
)
@click.option(
    "--preset",
    type=click.Choice(["default", "latest", "last-gen", "all"], case_sensitive=False),
    default="latest",
    show_default=True,
    help="Which built-in model list to probe (ignored if you pass --model).",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    help=("Model id(s) to probe (LiteLLM ids). If omitted, probes a small curated set including paperpipe defaults."),
)
@click.option(
    "--timeout",
    type=float,
    default=15.0,
    show_default=True,
    help="Per-request timeout (seconds).",
)
@click.option(
    "--max-tokens",
    type=int,
    default=16,
    show_default=True,
    help="Max tokens for completion probes (minimizes cost).",
)
@click.option("--verbose", is_flag=True, help="Show provider debug output from LiteLLM.")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def models(
    preset_arg: Optional[str],
    kind: tuple[str, ...],
    preset: str,
    models: tuple[str, ...],
    timeout: float,
    max_tokens: int,
    verbose: bool,
    as_json: bool,
):
    """
    Probe which LLM/embedding models work with your currently configured API keys.

    This command makes small live API calls (may incur cost) and reports OK/FAIL.
    """
    try:
        from litellm import completion as llm_completion  # type: ignore[import-not-found]
        from litellm import embedding as llm_embedding  # type: ignore[import-not-found]
    except Exception as exc:
        raise click.ClickException(
            "LiteLLM is required for `papi models`. Install `paperpipe[paperqa]` (or `litellm`)."
        ) from exc

    requested_kinds = tuple(k.lower() for k in kind)
    embedding_timeout = max(1, int(math.ceil(timeout)))

    ctx = click.get_current_context()
    preset_source = ctx.get_parameter_source("preset")
    preset_explicit = preset_source != click.core.ParameterSource.DEFAULT or preset_arg is not None
    effective_preset = preset_arg or preset

    def provider_has_key(provider: str) -> bool:
        provider = provider.lower()
        if provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"))
        if provider == "gemini":
            return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        if provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if provider == "voyage":
            return bool(os.environ.get("VOYAGE_API_KEY"))
        if provider == "openrouter":
            return bool(os.environ.get("OPENROUTER_API_KEY"))
        return False

    def infer_provider(model: str) -> Optional[str]:
        if model.startswith("gemini/"):
            return "gemini"
        if model.startswith("voyage/"):
            return "voyage"
        if model.startswith("openrouter/"):
            return "openrouter"
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith("gpt-") or model.startswith("text-embedding-"):
            return "openai"
        return None

    enabled_providers = {p for p in ("openai", "gemini", "anthropic", "voyage", "openrouter") if provider_has_key(p)}

    def probe_one(kind_name: str, model: str):
        if _is_ollama_model_id(model):
            config._prepare_ollama_env(os.environ)
            err = config._ollama_reachability_error(api_base=os.environ["OLLAMA_API_BASE"])
            if err:
                raise RuntimeError(err)
        if kind_name == "completion":
            llm_completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word 'pong'."}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            llm_embedding(model=model, input=["ping"], timeout=embedding_timeout)

    def probe_group(kind_name: str, candidates: list[str]) -> paperqa._ModelProbeResult:
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                if verbose:
                    probe_one(kind_name, candidate)
                else:
                    with redirect_stdout(null_out), redirect_stderr(null_err):
                        probe_one(kind_name, candidate)
                return paperqa._ModelProbeResult(kind=kind_name, model=candidate, ok=True)
            except Exception as exc:
                last_exc = exc
                continue

        err = paperqa._first_line(str(last_exc)) if last_exc else "Unknown error"
        hint = paperqa._probe_hint(kind=kind_name, model=candidates[0], error_line=err)
        if hint:
            err = f"{err} ({hint})"
        return paperqa._ModelProbeResult(
            kind=kind_name,
            model=candidates[0],
            ok=False,
            error_type=type(last_exc).__name__ if last_exc else "Error",
            error=err,
        )

    completion_models: list[str]
    embedding_models: list[str]
    if models:
        completion_models = list(models)
        embedding_models = list(models)
    else:
        # If the user didn't explicitly request a preset, default to probing only one
        # "latest" model per configured provider (plus embeddings), rather than a full sweep.
        if effective_preset.lower() == "latest" and not preset_explicit:
            results: list[paperqa._ModelProbeResult] = []
            null_out = StringIO()
            null_err = StringIO()

            if "completion" in requested_kinds:
                completion_groups: list[tuple[str, list[str]]] = [
                    ("openai", ["gpt-5.2", "gpt-5.1"]),
                    ("gemini", ["gemini/gemini-3-flash-preview"]),
                    ("anthropic", ["claude-sonnet-4-5"]),
                ]
                for provider, candidates in completion_groups:
                    if provider not in enabled_providers:
                        continue
                    results.append(probe_group("completion", candidates))

            if "embedding" in requested_kinds:
                embedding_groups: list[tuple[str, list[str]]] = [
                    ("openai", ["text-embedding-3-large", "text-embedding-3-small"]),
                    ("gemini", ["gemini/gemini-embedding-001"]),
                    ("voyage", ["voyage/voyage-3-large"]),
                ]
                for provider, candidates in embedding_groups:
                    if provider not in enabled_providers:
                        continue
                    results.append(probe_group("embedding", candidates))

            if as_json:
                payload = [
                    {
                        "kind": r.kind,
                        "model": r.model,
                        "ok": r.ok,
                        "error_type": r.error_type,
                        "error": r.error,
                    }
                    for r in results
                ]
                click.echo(json.dumps(payload, indent=2))
                return

            ok_count = sum(1 for r in results if r.ok)
            fail_count = len(results) - ok_count
            click.echo(f"Probed {len(results)} combinations: {ok_count} OK, {fail_count} FAIL")
            for r in results:
                status = "OK" if r.ok else "FAIL"
                if r.ok:
                    click.secho(f"{status:4s}  {r.kind:10s}  {r.model}", fg="green")
                else:
                    err = r.error or ""
                    err_type = r.error_type or "Error"
                    click.secho(f"{status:4s}  {r.kind:10s}  {r.model}  ({err_type}: {err})", fg="red")
            return

        if effective_preset.lower() == "all":
            completion_models = [
                # OpenAI
                "gpt-5.2",
                "gpt-5.1",
                "gpt-4.1",
                "gpt-4o",
                "gpt-4o-mini",
                # Google
                "gemini/gemini-3-flash-preview",
                "gemini/gemini-3-pro-preview",
                "gemini/gemini-2.5-flash",
                "gemini/gemini-2.5-pro",
                # Anthropic
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-sonnet-4-20250514",
            ]
            embedding_models = [
                # OpenAI embeddings
                "text-embedding-3-large",
                "text-embedding-3-small",
                "text-embedding-ada-002",
                # Google + Voyage
                "gemini/gemini-embedding-001",
                "gemini/text-embedding-004",
                "voyage/voyage-3-large",
                "voyage/voyage-3-lite",
            ]
        elif effective_preset.lower() == "latest":
            completion_models = [
                # OpenAI (flagship)
                "gpt-5.2",
                "gpt-5.1",
                # Google (Gemini 3 series - preview ids)
                "gemini/gemini-3-flash-preview",
                "gemini/gemini-3-pro-preview",
                # Anthropic (Claude 4.5)
                "claude-sonnet-4-5",
                "claude-opus-4-5",
            ]
            embedding_models = [
                "text-embedding-3-large",
                "text-embedding-3-small",
                "gemini/gemini-embedding-001",
                "voyage/voyage-3-large",
            ]
        elif effective_preset.lower() == "last-gen":
            completion_models = [
                # OpenAI (GPT-4 generation)
                "gpt-4.1",
                "gpt-4o",
                # Google (Gemini 2.5 series - stable)
                "gemini/gemini-2.5-flash",
                "gemini/gemini-2.5-pro",
                # Anthropic (oldest commonly available Claude 4 family)
                "claude-sonnet-4-20250514",
            ]
            embedding_models = [
                # OpenAI embeddings
                "text-embedding-ada-002",
                "text-embedding-3-small",
                # Google + Voyage (include older/smaller options)
                "gemini/gemini-embedding-001",
                "gemini/text-embedding-004",
                "voyage/voyage-3-large",
                "voyage/voyage-3-lite",
            ]
        else:
            completion_models = [
                default_llm_model(),
                "gpt-4o",
                "claude-sonnet-4-20250514",
            ]
            embedding_models = [
                default_embedding_model(),
                "text-embedding-3-small",
                "voyage/voyage-3-large",
            ]

        # Only probe providers that are configured with an API key.
        completion_models = [
            m for m in completion_models if (infer_provider(m) is None) or (infer_provider(m) in enabled_providers)
        ]
        embedding_models = [
            m for m in embedding_models if (infer_provider(m) is None) or (infer_provider(m) in enabled_providers)
        ]

    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    completion_models = dedupe(completion_models)
    embedding_models = dedupe(embedding_models)

    results: list[paperqa._ModelProbeResult] = []
    null_out = StringIO()
    null_err = StringIO()
    for k in requested_kinds:
        probe_models = completion_models if k == "completion" else embedding_models
        for model in probe_models:
            if k == "completion":
                try:
                    if verbose:
                        llm_completion(
                            model=model,
                            messages=[{"role": "user", "content": "ping"}],
                            max_tokens=max_tokens,
                            timeout=timeout,
                        )
                    else:
                        with redirect_stdout(null_out), redirect_stderr(null_err):
                            llm_completion(
                                model=model,
                                messages=[{"role": "user", "content": "ping"}],
                                max_tokens=max_tokens,
                                timeout=timeout,
                            )
                    results.append(paperqa._ModelProbeResult(kind=k, model=model, ok=True))
                except Exception as exc:
                    err = paperqa._first_line(str(exc))
                    hint = paperqa._probe_hint(kind=k, model=model, error_line=err)
                    if hint:
                        err = f"{err} ({hint})"
                    results.append(
                        paperqa._ModelProbeResult(
                            kind=k,
                            model=model,
                            ok=False,
                            error_type=type(exc).__name__,
                            error=err,
                        )
                    )
            else:  # embedding
                try:
                    if verbose:
                        llm_embedding(model=model, input=["ping"], timeout=embedding_timeout)
                    else:
                        with redirect_stdout(null_out), redirect_stderr(null_err):
                            llm_embedding(model=model, input=["ping"], timeout=embedding_timeout)
                    results.append(paperqa._ModelProbeResult(kind=k, model=model, ok=True))
                except Exception as exc:
                    err = paperqa._first_line(str(exc))
                    hint = paperqa._probe_hint(kind=k, model=model, error_line=err)
                    if hint:
                        err = f"{err} ({hint})"
                    results.append(
                        paperqa._ModelProbeResult(
                            kind=k,
                            model=model,
                            ok=False,
                            error_type=type(exc).__name__,
                            error=err,
                        )
                    )

    if as_json:
        payload = [
            {
                "kind": r.kind,
                "model": r.model,
                "ok": r.ok,
                "error_type": r.error_type,
                "error": r.error,
            }
            for r in results
        ]
        click.echo(json.dumps(payload, indent=2))
        return

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    click.echo(f"Probed {len(results)} combinations: {ok_count} OK, {fail_count} FAIL")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        if r.ok:
            click.secho(f"{status:4s}  {r.kind:10s}  {r.model}", fg="green")
        else:
            err = r.error or ""
            err_type = r.error_type or "Error"
            click.secho(f"{status:4s}  {r.kind:10s}  {r.model}  ({err_type}: {err})", fg="red")


@cli.command()
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
            except Exception:
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
            except Exception:
                pass
            continue

        src = paper_dir / src_name
        if not src.exists():
            echo_error(f"{missing_msg}: {name}")
            continue

        click.echo(f"- Content: {level_norm}")
        click.echo()
        click.echo(src.read_text(errors="ignore").rstrip("\n"))


@cli.command()
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
            except Exception:
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


@cli.command()
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


@cli.command()
def tags():
    """List all tags in the database."""
    index = load_index()
    all_tags: dict[str, int] = {}

    for info in index.values():
        for tag in info.get("tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
        click.echo(f"{tag}: {count}")


@cli.command()
def path():
    """Print the paper database path."""
    click.echo(config.PAPER_DB)


@cli.command("mcp-server")
def mcp_server():
    """Run the PaperQA2 retrieval MCP server.

    This is used by MCP-enabled agents (Claude Code, Codex CLI, Gemini CLI).
    Normally invoked via MCP config, not directly.
    """
    try:
        from paperqa_mcp_server import main
    except ImportError as e:
        echo_error("PaperQA2 MCP server not available.")
        echo_error("Install with: pip install 'paperpipe[mcp]'")
        raise SystemExit(1) from e
    main()


@cli.command("leann-mcp-server")
def leann_mcp_server():
    """Run the LEANN MCP server from the paper database directory.

    This is used by MCP-enabled agents (Claude Code, Codex CLI, Gemini CLI).
    Normally invoked via MCP config, not directly.
    """
    config.PAPER_DB.mkdir(parents=True, exist_ok=True)
    if not shutil.which("leann_mcp"):
        echo_error("`leann_mcp` not found on PATH.")
        echo_error("Install with: pip install 'paperpipe[leann]'")
        raise SystemExit(1)

    os.chdir(config.PAPER_DB)
    os.execvp("leann_mcp", ["leann_mcp"])


@cli.command("install")
@click.argument("components", nargs=-1)
@click.option(
    "--claude",
    "targets",
    flag_value="claude",
    multiple=True,
    help="Install for Claude Code (~/.claude/...)",
)
@click.option(
    "--codex",
    "targets",
    flag_value="codex",
    multiple=True,
    help="Install for Codex CLI (~/.codex/...)",
)
@click.option(
    "--gemini",
    "targets",
    flag_value="gemini",
    multiple=True,
    help="Install for Gemini CLI (~/.gemini/...)",
)
@click.option(
    "--repo",
    "targets",
    flag_value="repo",
    multiple=True,
    help="Install repo-local MCP configs (.mcp.json + .gemini/settings.json) in the current directory (MCP only)",
)
@click.option("--force", is_flag=True, help="Overwrite existing installation")
@click.option("--copy", is_flag=True, help="Copy prompts instead of symlinking (prompts only).")
@click.option("--name", default="paperqa", show_default=True, help="PaperQA2 MCP server name (MCP only)")
@click.option("--leann-name", default="leann", show_default=True, help="LEANN MCP server name (MCP only)")
@click.option("--embedding", default=None, show_default=False, help="Embedding model id (MCP only)")
def install(
    components: tuple[str, ...],
    targets: tuple[str, ...],
    force: bool,
    copy: bool,
    name: str,
    leann_name: str,
    embedding: Optional[str],
) -> None:
    """Install papi integrations (skill, prompts, and/or MCP config).

    By default, installs everything: skill + prompts + mcp.

    Components can be selected by name and combined:
      - `papi install mcp prompts`
      - `papi install mcp,prompts`

    \b
    Examples:
        papi install                    # Install skill + prompts + mcp
        papi install skill              # Install skill only
        papi install prompts --copy     # Install prompts only, copy files
        papi install mcp --repo         # Install repo-local MCP configs
        papi install --codex            # Install everything for Codex only
        papi install mcp --embedding text-embedding-3-small
    """
    requested = _parse_components(components)
    if not requested:
        requested = ["skill", "prompts", "mcp"]

    allowed = {"skill", "prompts", "mcp"}
    unknown = sorted({c for c in requested if c not in allowed})
    if unknown:
        raise click.UsageError(f"Unknown component(s): {', '.join(unknown)} (choose from: skill, prompts, mcp)")

    want_skill = "skill" in requested
    want_prompts = "prompts" in requested
    want_mcp = "mcp" in requested

    if targets and "repo" in targets and not want_mcp:
        raise click.UsageError("--repo is only valid when installing mcp")
    if copy and not want_prompts:
        raise click.UsageError("--copy is only valid when installing prompts")
    if (name != "paperqa" or leann_name != "leann" or embedding is not None) and not want_mcp:
        raise click.UsageError("--name/--leann-name/--embedding are only valid when installing mcp")

    if want_skill:
        _install_skill(targets=tuple([t for t in targets if t != "repo"]), force=force)
    if want_prompts:
        _install_prompts(targets=tuple([t for t in targets if t != "repo"]), force=force, copy=copy)
    if want_mcp:
        _install_mcp(targets=targets, name=name, leann_name=leann_name, embedding=embedding, force=force)


@cli.command("uninstall")
@click.argument("components", nargs=-1)
@click.option(
    "--claude",
    "targets",
    flag_value="claude",
    multiple=True,
    help="Uninstall for Claude Code (~/.claude/...)",
)
@click.option(
    "--codex",
    "targets",
    flag_value="codex",
    multiple=True,
    help="Uninstall for Codex CLI (~/.codex/...)",
)
@click.option(
    "--gemini",
    "targets",
    flag_value="gemini",
    multiple=True,
    help="Uninstall for Gemini CLI (~/.gemini/...)",
)
@click.option(
    "--repo",
    "targets",
    flag_value="repo",
    multiple=True,
    help="Uninstall repo-local MCP configs (.mcp.json + .gemini/settings.json) in the current directory (MCP only)",
)
@click.option("--force", is_flag=True, help="Remove even if the install does not match exactly")
@click.option("--name", default="paperqa", show_default=True, help="PaperQA2 MCP server name (MCP only)")
@click.option("--leann-name", default="leann", show_default=True, help="LEANN MCP server name (MCP only)")
def uninstall(components: tuple[str, ...], targets: tuple[str, ...], force: bool, name: str, leann_name: str) -> None:
    """Uninstall papi integrations (skill, prompts, and/or MCP config).

    By default, uninstalls everything: mcp + prompts + skill.

    Components can be selected by name and combined:
      - `papi uninstall mcp prompts`
      - `papi uninstall mcp,prompts`

    \b
    Examples:
        papi uninstall                  # Uninstall skill + prompts + mcp
        papi uninstall skill            # Uninstall skill only
        papi uninstall prompts          # Uninstall prompts only
        papi uninstall mcp --repo       # Uninstall repo-local MCP configs
        papi uninstall --codex          # Uninstall everything for Codex only
        papi uninstall mcp --force      # Ignore remove failures / mismatches
    """
    requested = _parse_components(components)
    if not requested:
        requested = ["skill", "prompts", "mcp"]

    allowed = {"skill", "prompts", "mcp"}
    unknown = sorted({c for c in requested if c not in allowed})
    if unknown:
        raise click.UsageError(f"Unknown component(s): {', '.join(unknown)} (choose from: skill, prompts, mcp)")

    want_skill = "skill" in requested
    want_prompts = "prompts" in requested
    want_mcp = "mcp" in requested

    if targets and "repo" in targets and not want_mcp:
        raise click.UsageError("--repo is only valid when uninstalling mcp")
    if (name != "paperqa" or leann_name != "leann") and not want_mcp:
        raise click.UsageError("--name/--leann-name are only valid when uninstalling mcp")

    non_repo_targets = tuple([t for t in targets if t != "repo"])

    # Default uninstall order is reverse of install: mcp -> prompts -> skill.
    if want_mcp:
        _uninstall_mcp(targets=targets, name=name, leann_name=leann_name, force=force)
    if want_prompts:
        _uninstall_prompts(targets=non_repo_targets, force=force)
    if want_skill:
        _uninstall_skill(targets=non_repo_targets, force=force)
