"""Helper functions and utilities for CLI commands."""

from __future__ import annotations

import re
import threading
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import List, Optional, Tuple

from ..output import echo_error, echo_warning


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

        except (KeyboardInterrupt, SystemExit):
            # Don't retry on user interrupts or system exits
            raise

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


def _cli_version() -> str:
    try:
        return package_version("paperpipe")
    except PackageNotFoundError:
        return "0+unknown"
