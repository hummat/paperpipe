"""Paper name matching utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from enum import Enum
from typing import Optional

import click


class MatchType(Enum):
    """Type of paper name match."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    NOT_FOUND = "not_found"


@dataclass
class MatchResult:
    """Result of paper name matching."""

    match_type: MatchType
    matches: list[str]  # Empty for EXACT, 1+ for NORMALIZED/FUZZY
    query: str  # Original query
    normalized_query: str  # Normalized version


def normalize_paper_name(name: str) -> str:
    """Normalize paper name for fuzzy matching.

    Strips hyphens, underscores, and lowercases.

    Examples:
        "IF-Net" -> "ifnet"
        "my_paper" -> "mypaper"
    """
    raw = (name or "").strip()
    return raw.replace("-", "").replace("_", "").lower()


def find_paper_matches(
    query: str,
    index: dict,
    *,
    fuzzy_cutoff: float = 0.7,
) -> MatchResult:
    """Find paper name matches with exact, normalized, and fuzzy fallback.

    Args:
        query: User-provided paper name
        index: Paper index dict {name: metadata}
        fuzzy_cutoff: Minimum similarity for fuzzy matches (0.0-1.0)

    Returns:
        MatchResult with match type and candidate names
    """
    query_clean = query.strip()
    if not query_clean:
        return MatchResult(MatchType.NOT_FOUND, [], query, "")

    # Exact match
    if query_clean in index:
        return MatchResult(MatchType.EXACT, [query_clean], query, query_clean)

    # Build normalized -> list of original names (handles collisions)
    query_norm = normalize_paper_name(query_clean)
    index_normalized: dict[str, list[str]] = {}
    for k in index.keys():
        norm = normalize_paper_name(k)
        index_normalized.setdefault(norm, []).append(k)

    # Normalized match
    if query_norm in index_normalized:
        matches = index_normalized[query_norm]
        if len(matches) == 1:
            return MatchResult(MatchType.NORMALIZED, matches, query, query_norm)
        # Multiple papers normalize identically - treat as ambiguous
        return MatchResult(MatchType.FUZZY, matches, query, query_norm)

    # Fuzzy match
    candidates = list(index_normalized.keys())
    fuzzy_matches = get_close_matches(query_norm, candidates, n=5, cutoff=fuzzy_cutoff)

    if fuzzy_matches:
        # Flatten: each fuzzy match may map to multiple original names
        original_names = []
        for m in fuzzy_matches:
            original_names.extend(index_normalized[m])
        return MatchResult(MatchType.FUZZY, original_names, query, query_norm)

    return MatchResult(MatchType.NOT_FOUND, [], query, query_norm)


def select_paper_interactively(
    matches: list[str],
    query: str,
    index: dict,
) -> Optional[str]:
    """Prompt user to select from fuzzy matches.

    Args:
        matches: List of matching paper names (may be single low-confidence match)
        query: Original query for context
        index: Paper index for displaying titles

    Returns:
        Selected paper name or None if user cancels or non-interactive
    """
    if not matches:
        return None

    # Check if interactive - don't auto-select even single matches
    # (caller handles high-confidence auto-selection separately)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    click.secho(f"Multiple papers match '{query}':", fg="yellow")
    for i, name in enumerate(matches, 1):
        title = index.get(name, {}).get("title", "")
        if title:
            title_short = title[:50] + "..." if len(title) > 50 else title
            click.echo(f"  {i}. {name} - {title_short}")
        else:
            click.echo(f"  {i}. {name}")
    click.echo("  0. Cancel")

    try:
        choice = click.prompt("Select paper", type=int, default=0)
        if 1 <= choice <= len(matches):
            return matches[choice - 1]
    except (click.Abort, KeyboardInterrupt):
        pass

    return None


def get_best_fuzzy_similarity(query: str, match: str) -> float:
    """Get similarity ratio between normalized query and match."""
    query_norm = normalize_paper_name(query)
    match_norm = normalize_paper_name(match)
    return SequenceMatcher(None, query_norm, match_norm).ratio()
