"""Core utilities and index helpers."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click

from . import config
from .output import echo_warning


def _format_title_short(title: str, *, max_len: int = 60) -> str:
    t = (title or "").strip()
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "..."


def _slugify_title(title: str, *, max_len: int = 60) -> str:
    """Best-effort slug for local PDF ingestion (stable, human-readable)."""
    raw = (title or "").strip().lower()
    raw = raw.replace("’", "'")
    raw = re.sub(r"[\"']", "", raw)
    slug = re.sub(r"[^a-z0-9]+", "-", raw)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        return "paper"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "paper"


def _parse_authors(authors: Optional[str]) -> list[str]:
    """Parse authors from a CLI string.

    Conventions:
    - Prefer `;` as the separator (avoids splitting on commas inside "Last, First").
    - If no `;` is present, accept comma-separated values, but preserve a single "Last, First" author.
    """
    raw = (authors or "").strip()
    if not raw:
        return []
    # Prefer semicolons, since commas can appear in "Last, First" names.
    if ";" in raw:
        parts = [a.strip() for a in raw.split(";")]
        return [a for a in parts if a]

    # If there's exactly one comma, assume a single "Last, First" author.
    if raw.count(",") == 1 and ", " in raw:
        return [raw]

    parts = [a.strip() for a in raw.split(",")]
    return [a for a in parts if a]


def _looks_like_pdf(path: Path) -> bool:
    """Return True if the file likely is a PDF (best-effort magic header check)."""
    try:
        head = path.read_bytes()[:1024]
    except Exception:
        return False
    return b"%PDF-" in head


def ensure_notes_file(paper_dir: Path, meta: dict) -> Path:
    notes_path = paper_dir / "notes.md"
    if notes_path.exists():
        return notes_path

    title = str(meta.get("title") or "").strip()
    header = f"# Notes{': ' + title if title else ''}".rstrip()
    body = "\n".join(
        [
            header,
            "",
            "## Implementation Notes",
            "",
            "- Gotchas / pitfalls:",
            "- Hyperparameters / defaults:",
            "- Mapping to equations (e.g., eq. 7):",
            "",
            "## Code Snippets",
            "",
            "```",
            "# paste snippets here",
            "```",
            "",
        ]
    )
    notes_path.write_text(body)
    return notes_path


_ARXIV_NEW_STYLE_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", flags=re.IGNORECASE)
_ARXIV_OLD_STYLE_RE = re.compile(r"^[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?$", flags=re.IGNORECASE)
_ARXIV_ANY_RE = re.compile(
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?)",
    flags=re.IGNORECASE,
)

_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_ARXIV_VERSION_SUFFIX_RE = re.compile(r"v\d+$", flags=re.IGNORECASE)


def arxiv_base_id(arxiv_id: str) -> str:
    """Strip the version suffix from an arXiv ID: 1706.03762v2 -> 1706.03762."""
    return _ARXIV_VERSION_SUFFIX_RE.sub("", (arxiv_id or "").strip())


def _arxiv_base_from_any(value: object) -> str:
    """Best-effort arXiv base ID extraction from IDs/URLs/other strings."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return arxiv_base_id(normalize_arxiv_id(raw))
    except ValueError:
        return arxiv_base_id(raw)


def _index_arxiv_base_to_names(index: dict) -> dict[str, list[str]]:
    """Build a reverse index: arXiv base ID -> list of paper names."""
    base_to_names: dict[str, list[str]] = {}
    for name, info in index.items():
        if not isinstance(info, dict):
            continue
        entry_arxiv_id = info.get("arxiv_id")
        if not entry_arxiv_id:
            continue
        base = _arxiv_base_from_any(entry_arxiv_id)
        if not base:
            continue
        base_to_names.setdefault(base, []).append(name)
    for names in base_to_names.values():
        names.sort()
    return base_to_names


def normalize_arxiv_id(value: str) -> str:
    """
    Normalize an arXiv identifier from an ID or common arXiv URL.

    Examples:
      - 1706.03762
      - https://arxiv.org/abs/1706.03762
      - https://arxiv.org/pdf/1706.03762.pdf
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("missing arXiv id")

    # Handle arXiv URLs (including old-style IDs containing '/').
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        host = (parsed.netloc or "").lower()
        if host.endswith("arxiv.org"):
            path = (parsed.path or "").strip()
            for prefix in ("/abs/", "/pdf/", "/e-print/"):
                if path.startswith(prefix):
                    candidate = path[len(prefix) :].strip("/")
                    if candidate.lower().endswith(".pdf"):
                        candidate = candidate[:-4]
                    raw = candidate
                    break

    # Common paste formats like "arXiv:1706.03762" or "abs/1706.03762".
    raw = re.sub(r"^\s*arxiv:\s*", "", raw, flags=re.IGNORECASE).strip()
    for prefix in ("abs/", "/abs/", "pdf/", "/pdf/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :].strip()

    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]

    if _ARXIV_NEW_STYLE_RE.fullmatch(raw) or _ARXIV_OLD_STYLE_RE.fullmatch(raw):
        return raw

    embedded = _ARXIV_ANY_RE.search(raw)
    if embedded:
        return embedded.group(1)

    raise ValueError(f"could not parse arXiv id from: {value!r}")


def _is_safe_paper_name(name: str) -> bool:
    """
    Paper names are directory names under config.PAPERS_DIR.

    For safety, do not treat values containing path separators (or traversal) as a name.
    """
    raw = (name or "").strip()
    if not raw or raw in {".", ".."}:
        return False
    if "/" in raw or "\\" in raw:
        return False
    path = Path(raw)
    if path.is_absolute():
        return False
    if any(part == ".." for part in path.parts):
        return False
    return True


def _resolve_paper_name_from_ref(paper_or_arxiv: str, index: dict) -> tuple[Optional[str], str]:
    """
    Resolve a user-supplied reference into a paper name.

    Supports:
      - paper name (directory / index key)
      - arXiv ID
      - arXiv URL (abs/pdf/e-print)
    """
    raw = (paper_or_arxiv or "").strip()
    if not raw:
        return None, "Missing paper name or arXiv ID/URL."

    if raw in index:
        return raw, ""

    if _is_safe_paper_name(raw):
        paper_dir = config.PAPERS_DIR / raw
        if paper_dir.exists():
            return raw, ""

    try:
        arxiv_id = normalize_arxiv_id(raw)
    except ValueError:
        return None, f"Paper not found: {paper_or_arxiv}"

    arxiv_base = arxiv_base_id(arxiv_id)
    matches = [name for name, info in index.items() if _arxiv_base_from_any(info.get("arxiv_id", "")) == arxiv_base]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, f"Multiple papers match arXiv ID {arxiv_base}: {', '.join(sorted(matches))}"

    # Fallback: scan on-disk metadata if index is missing/out-of-date.
    matches = []
    if config.PAPERS_DIR.exists():
        for candidate in config.PAPERS_DIR.iterdir():
            if not candidate.is_dir():
                continue
            meta_path = candidate / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            if _arxiv_base_from_any(meta.get("arxiv_id", "")) == arxiv_base:
                matches.append(candidate.name)

    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, f"Multiple papers match arXiv ID {arxiv_base}: {', '.join(sorted(matches))}"

    return None, f"Paper not found: {paper_or_arxiv}"


def _normalize_for_search(text: str) -> str:
    return " ".join(_SEARCH_TOKEN_RE.findall((text or "").lower())).strip()


def _read_text_limited(path: Path, *, max_chars: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except FileNotFoundError:
        return ""
    except Exception:
        echo_warning(f"Could not read {path}: treating as empty for search.")
        return ""


def _best_line_ratio(query_norm: str, text: str, *, max_lines: int = 250) -> float:
    if not query_norm or not text:
        return 0.0
    best = 0.0
    for line in text.splitlines()[:max_lines]:
        line_norm = _normalize_for_search(line)
        if not line_norm:
            continue
        if query_norm in line_norm:
            return 1.0
        ratio = SequenceMatcher(None, query_norm, line_norm).ratio()
        if ratio > best:
            best = ratio
    return best


def _fuzzy_text_score(query: str, text: str, *, fuzzy: bool) -> float:
    """
    Return a [0.0, 1.0] score for how well `text` matches `query`.

    - exact mode: substring match only
    - fuzzy mode: token coverage + best line ratio
    """
    query_norm = _normalize_for_search(query)
    text_norm = _normalize_for_search(text)
    if not query_norm or not text_norm:
        return 0.0

    if query_norm in text_norm:
        return 1.0
    if not fuzzy:
        return 0.0

    q_tokens = query_norm.split()
    if not q_tokens:
        return 0.0

    t_tokens = set(text_norm.split())
    exact_hits = sum(1 for tok in q_tokens if tok in t_tokens)
    remaining = [tok for tok in q_tokens if tok not in t_tokens]

    fuzzy_hits = 0
    if remaining and t_tokens:
        candidates = sorted(t_tokens)
        if len(candidates) > 8000:
            candidates = candidates[:8000]
        for tok in remaining:
            if get_close_matches(tok, candidates, n=1, cutoff=0.88):
                fuzzy_hits += 1

    coverage = (exact_hits + 0.7 * fuzzy_hits) / len(q_tokens)
    line_ratio = _best_line_ratio(query_norm, text)

    return max(coverage, line_ratio)


def ensure_db():
    """Ensure the paper database directory structure exists."""
    config.PAPER_DB.mkdir(parents=True, exist_ok=True)
    config.PAPERS_DIR.mkdir(exist_ok=True)
    if not config.INDEX_FILE.exists():
        config.INDEX_FILE.write_text("{}")


def load_index() -> dict:
    """Load the paper index."""
    ensure_db()
    return json.loads(config.INDEX_FILE.read_text())


def save_index(index: dict):
    """Save the paper index."""
    config.INDEX_FILE.write_text(json.dumps(index, indent=2))


def categories_to_tags(categories: list[str]) -> list[str]:
    """Convert arXiv categories to human-readable tags."""
    tags: list[str] = []
    for cat in categories:
        if cat in config.CATEGORY_TAGS:
            tags.append(config.CATEGORY_TAGS[cat])
        else:
            # Use the category itself as a tag (e.g., cs.CV -> cs-cv)
            tags.append(cat.lower().replace(".", "-"))
    return config.normalize_tags(tags)


_VALID_REGENERATE_FIELDS = {"all", "summary", "equations", "tags", "name", "tldr", "figures"}


def _parse_overwrite_option(overwrite: Optional[str]) -> tuple[set[str], bool]:
    if overwrite is None:
        return set(), False
    overwrite_fields = {f.strip().lower() for f in overwrite.split(",") if f.strip()}
    invalid = overwrite_fields - _VALID_REGENERATE_FIELDS
    if invalid:
        raise click.UsageError(f"Invalid --overwrite fields: {', '.join(sorted(invalid))}")
    return overwrite_fields, "all" in overwrite_fields


def _is_arxiv_id_name(name: str) -> bool:
    """Check if name looks like an arXiv ID (e.g., 1706_03762 or hep-th_9901001)."""
    # New-style: 1706_03762 or 1706_03762v5
    if re.match(r"^\d{4}_\d{4,5}(v\d+)?$", name):
        return True
    # Old-style: hep-th_9901001
    if re.match(r"^[a-z-]+_\d{7}$", name):
        return True
    return False
