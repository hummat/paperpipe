#!/usr/bin/env python3
"""
paperpipe: Unified paper database for coding agents + PaperQA2.

A local paper database that:
- Downloads PDFs (for PaperQA2) and LaTeX source (for equation comparison)
- Auto-tags from arXiv categories + LLM-generated semantic tags
- Generates coding-context summaries and equation extractions
- Works with any CLI (Claude Code, Codex CLI, Gemini CLI)
"""

import json
import logging
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
from io import StringIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import click

# TOML config support (stdlib on 3.11+, tomli on 3.10)
try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]  # noqa: F401

# Simple debug logger (only used with --verbose)
_debug_logger = logging.getLogger("paperpipe")
_debug_logger.addHandler(logging.NullHandler())


def _setup_debug_logging() -> None:
    """Enable debug logging to stderr."""
    _debug_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    _debug_logger.addHandler(handler)


# Output helpers that respect --quiet mode
_quiet_mode = False


def set_quiet(quiet: bool) -> None:
    global _quiet_mode
    _quiet_mode = quiet


def echo(message: str = "", err: bool = False) -> None:
    """Print a message (respects --quiet for non-error messages)."""
    if _quiet_mode and not err:
        return
    click.echo(message, err=err)


def echo_success(message: str) -> None:
    """Print a success message in green."""
    click.secho(message, fg="green")


def echo_error(message: str) -> None:
    """Print an error message in red to stderr."""
    click.secho(message, fg="red", err=True)


def echo_warning(message: str) -> None:
    """Print a warning message in yellow to stderr."""
    click.secho(message, fg="yellow", err=True)


def echo_progress(message: str) -> None:
    """Print a progress message (suppressed in quiet mode)."""
    if not _quiet_mode:
        click.echo(message)


def debug(message: str, *args: object) -> None:
    """Log a debug message (only shown with --verbose)."""
    _debug_logger.debug(message, *args)


# Configuration
def _paper_db_root() -> Path:
    configured = os.environ.get("PAPER_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".paperpipe"


PAPER_DB = _paper_db_root()
PAPERS_DIR = PAPER_DB / "papers"
INDEX_FILE = PAPER_DB / "index.json"

DEFAULT_LLM_MODEL_FALLBACK = "gemini/gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL_FALLBACK = "gemini/gemini-embedding-001"
DEFAULT_LLM_TEMPERATURE_FALLBACK = 0.3

_CONFIG_CACHE: Optional[tuple[Path, Optional[float], dict[str, Any]]] = None


def _config_path() -> Path:
    configured = os.environ.get("PAPERPIPE_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser()
    return (PAPER_DB / "config.toml").expanduser()


def load_config() -> dict[str, Any]:
    """Load config from <paper_db>/config.toml (or PAPERPIPE_CONFIG_PATH).

    Returns an empty dict if missing or invalid.
    """
    path = _config_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = None

    global _CONFIG_CACHE
    if _CONFIG_CACHE and _CONFIG_CACHE[0] == path and _CONFIG_CACHE[1] == mtime:
        return _CONFIG_CACHE[2]

    if mtime is None:
        cfg: dict[str, Any] = {}
        _CONFIG_CACHE = (path, None, cfg)
        return cfg

    try:
        raw = path.read_bytes()
        cfg = tomllib.loads(raw.decode("utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception as e:
        debug("Failed to parse config.toml (%s) [%s]: %s", str(path), type(e).__name__, str(e))
        cfg = {}

    _CONFIG_CACHE = (path, mtime, cfg)
    return cfg


def _config_get(cfg: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _setting_str(*, env: str, keys: tuple[str, ...], default: str) -> str:
    val = os.environ.get(env)
    if val is not None and val.strip():
        return val.strip()
    cfg = load_config()
    raw = _config_get(cfg, keys)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default


def _setting_float(*, env: str, keys: tuple[str, ...], default: float) -> float:
    val = os.environ.get(env)
    if val is not None and val.strip():
        try:
            return float(val.strip())
        except Exception:
            return default
    cfg = load_config()
    raw = _config_get(cfg, keys)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except Exception:
            return default
    return default


def default_llm_model() -> str:
    return _setting_str(env="PAPERPIPE_LLM_MODEL", keys=("llm", "model"), default=DEFAULT_LLM_MODEL_FALLBACK)


def default_embedding_model() -> str:
    return _setting_str(
        env="PAPERPIPE_EMBEDDING_MODEL",
        keys=("embedding", "model"),
        default=DEFAULT_EMBEDDING_MODEL_FALLBACK,
    )


def default_llm_temperature() -> float:
    return _setting_float(
        env="PAPERPIPE_LLM_TEMPERATURE",
        keys=("llm", "temperature"),
        default=DEFAULT_LLM_TEMPERATURE_FALLBACK,
    )


def default_pqa_settings_name() -> str:
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "settings"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "default"


def default_pqa_llm_model() -> str:
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "llm"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default_llm_model()


def default_pqa_embedding_model() -> str:
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "embedding"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default_embedding_model()


def default_pqa_index_dir() -> Path:
    configured = os.environ.get("PAPERPIPE_PQA_INDEX_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "index_dir"))
    if isinstance(raw, str) and raw.strip():
        return Path(raw.strip()).expanduser()
    return (PAPER_DB / ".pqa_index").expanduser()


def default_pqa_summary_llm(fallback: Optional[str]) -> Optional[str]:
    configured = os.environ.get("PAPERPIPE_PQA_SUMMARY_LLM")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "summary_llm"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def default_pqa_enrichment_llm(fallback: Optional[str]) -> Optional[str]:
    configured = os.environ.get("PAPERPIPE_PQA_ENRICHMENT_LLM")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "enrichment_llm"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def default_pqa_temperature() -> Optional[float]:
    configured = os.environ.get("PAPERPIPE_PQA_TEMPERATURE")
    if configured and configured.strip():
        try:
            return float(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "temperature"))
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def default_pqa_verbosity() -> Optional[int]:
    configured = os.environ.get("PAPERPIPE_PQA_VERBOSITY")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "verbosity"))
    if isinstance(raw, int):
        return raw
    return None


def default_pqa_answer_length() -> Optional[str]:
    configured = os.environ.get("PAPERPIPE_PQA_ANSWER_LENGTH")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "answer_length"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def default_pqa_evidence_k() -> Optional[int]:
    configured = os.environ.get("PAPERPIPE_PQA_EVIDENCE_K")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "evidence_k"))
    if isinstance(raw, int):
        return raw
    return None


def default_pqa_max_sources() -> Optional[int]:
    configured = os.environ.get("PAPERPIPE_PQA_MAX_SOURCES")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "max_sources"))
    if isinstance(raw, int):
        return raw
    return None


def default_pqa_timeout() -> Optional[float]:
    configured = os.environ.get("PAPERPIPE_PQA_TIMEOUT")
    if configured and configured.strip():
        try:
            return float(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "timeout"))
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def default_pqa_concurrency() -> int:
    configured = os.environ.get("PAPERPIPE_PQA_CONCURRENCY")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "concurrency"))
    if isinstance(raw, int):
        return raw
    return 1  # Default to 1 for stability


def tag_aliases() -> dict[str, str]:
    cfg = load_config()
    raw = _config_get(cfg, ("tags", "aliases"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        k_norm = k.strip().lower()
        v_norm = v.strip().lower()
        if k_norm and v_norm:
            out[k_norm] = v_norm
    return out


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower().replace(" ", "-")
    t = re.sub(r"[^a-z0-9-]", "", t).strip("-")
    if not t:
        return ""
    aliases = tag_aliases()
    return aliases.get(t, t)


def normalize_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for t in tags:
        n = normalize_tag(t)
        if n:
            out.append(n)
    # Preserve a stable order for UX and deterministic tests
    return sorted(set(out))


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


def _generate_local_pdf_name(meta: dict, *, use_llm: bool) -> str:
    """Generate a base name for local PDF ingestion (no collision suffixing)."""
    title = str(meta.get("title") or "").strip()
    if not title:
        return "paper"

    name = _extract_name_from_title(title)
    if not name and use_llm and _litellm_available():
        name = _generate_name_with_llm(meta)
    if not name:
        name = _slugify_title(title)

    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9-]", "", name).strip("-")
    return name or "paper"


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


# arXiv category mappings for human-readable tags
CATEGORY_TAGS = {
    "cs.CV": "computer-vision",
    "cs.LG": "machine-learning",
    "cs.AI": "artificial-intelligence",
    "cs.CL": "nlp",
    "cs.GR": "graphics",
    "cs.RO": "robotics",
    "cs.NE": "neural-networks",
    "stat.ML": "machine-learning",
    "eess.IV": "image-processing",
    "physics.comp-ph": "computational-physics",
    "math.NA": "numerical-analysis",
}


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
    Paper names are directory names under PAPERS_DIR.

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
        paper_dir = PAPERS_DIR / raw
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
    if PAPERS_DIR.exists():
        for candidate in PAPERS_DIR.iterdir():
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
    except Exception:
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
    PAPER_DB.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("{}")


def _pillow_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("PIL") is not None


def _refresh_pqa_pdf_staging_dir(*, staging_dir: Path, exclude_names: Optional[set[str]] = None) -> int:
    """
    Create/update a flat directory containing only PDFs (one per paper) for PaperQA2 indexing.

    PaperQA2's default file filter includes Markdown. Since paperpipe stores generated `summary.md`
    and `equations.md` alongside each `paper.pdf`, we stage just PDFs to avoid indexing the generated
    artifacts.

    Returns the number of PDFs linked/copied into the staging directory.

    Note: This function preserves existing valid symlinks to maintain their modification times.
    PaperQA2 uses file modification times to track which files it has already indexed, so
    recreating symlinks would cause unnecessary re-indexing.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    exclude_names = exclude_names or set()

    # Build set of expected symlink names based on current papers.
    expected_names: set[str] = set()
    paper_sources: dict[str, Path] = {}  # symlink name -> source PDF path

    if PAPERS_DIR.exists():
        for paper_dir in PAPERS_DIR.iterdir():
            if not paper_dir.is_dir():
                continue
            pdf_src = paper_dir / "paper.pdf"
            if not pdf_src.exists():
                continue
            name = f"{paper_dir.name}.pdf"
            if name in exclude_names:
                continue
            expected_names.add(name)
            paper_sources[name] = pdf_src

    # Remove stale entries (papers that were removed or are now excluded) - best-effort cleanup.
    try:
        for child in staging_dir.iterdir():
            if child.name not in expected_names:
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except Exception:
                    debug("Failed cleaning pqa staging entry: %s", child)
    except Exception:
        debug("Failed listing pqa staging dir: %s", staging_dir)

    # Create/repair symlinks only where needed, preserving existing valid ones.
    count = 0
    for name, pdf_src in paper_sources.items():
        pdf_dest = staging_dir / name
        rel_target = os.path.relpath(pdf_src, start=pdf_dest.parent)

        # Check if existing symlink is valid and points to the right target.
        needs_update = True
        if pdf_dest.is_symlink():
            try:
                # Symlink exists - check if it points to the correct target and is valid.
                current_target = os.readlink(pdf_dest)
                if current_target == rel_target and pdf_dest.exists():
                    needs_update = False
            except Exception:
                pass  # Broken or unreadable symlink, will recreate.

        if needs_update:
            try:
                if pdf_dest.exists() or pdf_dest.is_symlink():
                    pdf_dest.unlink()
                pdf_dest.symlink_to(rel_target)
            except Exception:
                try:
                    shutil.copy2(pdf_src, pdf_dest)
                except Exception:
                    debug("Failed staging PDF for PaperQA2: %s", pdf_src)
                    continue

        count += 1

    return count


def _extract_flag_value(args: list[str], *, names: set[str]) -> Optional[str]:
    """
    Extract a value from argv-style args for flags like:
      --flag value
      --flag=value
    """
    for i, arg in enumerate(args):
        if arg in names:
            if i + 1 < len(args):
                return args[i + 1]
            return None
        for name in names:
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return None


def _paperqa_effective_paper_directory(args: list[str], *, base_dir: Path) -> Optional[Path]:
    raw = _extract_flag_value(args, names={"--agent.index.paper_directory", "--agent.index.paper-directory"})
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _paperqa_find_crashing_file(*, paper_directory: Path, crashing_doc: str) -> Optional[Path]:
    doc = (crashing_doc or "").strip().strip("\"'")
    doc = doc.rstrip(".…,:;")
    if not doc:
        return None

    doc_path = Path(doc)
    if doc_path.is_absolute():
        return doc_path if doc_path.exists() else None

    if ".." in doc_path.parts:
        doc_path = Path(doc_path.name)

    # Try the path as-is (relative to the paper directory).
    candidate = paper_directory / doc_path
    if candidate.exists():
        return candidate

    # Try matching by file name/stem (common when pqa prints just "foo.pdf" or "foo").
    name = doc_path.name
    expected_stem = Path(name).stem
    if expected_stem.lower().endswith(".pdf"):
        expected_stem = Path(expected_stem).stem

    try:
        for f in paper_directory.iterdir():
            if f.name == name or f.stem == expected_stem:
                return f
    except OSError:
        pass

    # As a last resort, search recursively by filename.
    try:
        for f in paper_directory.rglob(name):
            if f.name == name:
                return f
    except OSError:
        pass

    return None


def _paperqa_index_files_path(*, index_directory: Path, index_name: str) -> Path:
    return Path(index_directory) / index_name / "files.zip"


def _paperqa_load_index_files_map(path: Path) -> Optional[dict[str, str]]:
    try:
        raw = zlib.decompress(path.read_bytes())
        obj = pickle.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _paperqa_save_index_files_map(path: Path, mapping: dict[str, str]) -> bool:
    """Save the PaperQA2 index files map back to disk.

    Note: Uses pickle for compatibility with PaperQA2's existing index format.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = zlib.compress(pickle.dumps(mapping, protocol=pickle.HIGHEST_PROTOCOL))  # PaperQA2 format
        path.write_bytes(payload)
        return True
    except Exception:
        return False


def _paperqa_clear_failed_documents(*, index_directory: Path, index_name: str) -> tuple[int, list[str]]:
    """
    Clear PaperQA2's "ERROR" failure markers so it can retry indexing those docs.

    PaperQA2 records a per-file status in `<index>/files.zip` (zlib-compressed pickle).
    If a file is marked as ERROR, PaperQA2 treats it as already processed and won't retry
    unless you rebuild the entire index. Clearing those keys makes PaperQA2 treat them as new.
    """
    files_path = _paperqa_index_files_path(index_directory=index_directory, index_name=index_name)
    if not files_path.exists():
        return 0, []

    mapping = _paperqa_load_index_files_map(files_path)
    if mapping is None:
        return 0, []

    failed = sorted([k for k, v in mapping.items() if v == "ERROR"])
    if not failed:
        return 0, []

    for k in failed:
        mapping.pop(k, None)

    _paperqa_save_index_files_map(files_path, mapping)
    return len(failed), failed


def _paperqa_mark_failed_documents(
    *, index_directory: Path, index_name: str, staged_files: set[str]
) -> tuple[int, list[str]]:
    """
    Mark unprocessed staged files as ERROR in the PaperQA2 index.

    When pqa crashes with an unhandled exception, it doesn't mark the crashing document
    as ERROR. This function detects which staged files weren't processed and marks them
    as ERROR so pqa won't crash on them again (unless --retry-failed is used).

    Returns (count, list of newly marked files).
    """
    files_path = _paperqa_index_files_path(index_directory=index_directory, index_name=index_name)

    mapping = _paperqa_load_index_files_map(files_path) if files_path.exists() else {}
    if mapping is None:
        mapping = {}

    # Find staged files that have no status in the index (not processed)
    unprocessed = sorted([f for f in staged_files if f not in mapping])
    if not unprocessed:
        return 0, []

    for f in unprocessed:
        mapping[f] = "ERROR"

    if _paperqa_save_index_files_map(files_path, mapping):
        return len(unprocessed), unprocessed
    return 0, []


@dataclass(frozen=True)
class _ModelProbeResult:
    kind: str
    model: str
    ok: bool
    error_type: Optional[str] = None
    error: Optional[str] = None


def _first_line(text: str) -> str:
    return (text or "").splitlines()[0].strip()


def _probe_hint(kind: str, model: str, error_line: str) -> Optional[str]:
    low = (error_line or "").lower()
    if model == "gpt-5.2" and ("not supported" in low or "model_not_supported" in low):
        return "not enabled for this OpenAI key/project; try gpt-5.1"
    if model == "text-embedding-3-large" and ("not supported" in low or "model_not_supported" in low):
        return "not enabled for this OpenAI key/project; use text-embedding-3-small"
    if model.startswith("claude-3-5-sonnet") and ("not_found" in low or "model:" in low):
        return "Claude 3.5 appears retired; try claude-sonnet-4-5"
    if kind == "completion" and model.startswith("voyage/") and "does not support parameters" in low:
        return "Voyage is typically embedding-only; probe it under --kind embedding"
    return None


def load_index() -> dict:
    """Load the paper index."""
    ensure_db()
    return json.loads(INDEX_FILE.read_text())


def save_index(index: dict):
    """Save the paper index."""
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def categories_to_tags(categories: list[str]) -> list[str]:
    """Convert arXiv categories to human-readable tags."""
    tags: list[str] = []
    for cat in categories:
        if cat in CATEGORY_TAGS:
            tags.append(CATEGORY_TAGS[cat])
        else:
            # Use the category itself as a tag (e.g., cs.CV -> cs-cv)
            tags.append(cat.lower().replace(".", "-"))
    return normalize_tags(tags)


_VALID_REGENERATE_FIELDS = {"all", "summary", "equations", "tags", "name"}


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


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata from arXiv API."""
    import arxiv

    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(arxiv.Client().results(search))

    return {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "primary_category": paper.primary_category,
        "categories": paper.categories,
        "published": paper.published.isoformat(),
        "updated": paper.updated.isoformat(),
        "doi": paper.doi,
        "journal_ref": paper.journal_ref,
        "pdf_url": paper.pdf_url,
    }


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    """Download paper PDF."""
    import arxiv

    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(arxiv.Client().results(search))
    paper.download_pdf(filename=str(dest))
    return dest.exists()


def download_source(arxiv_id: str, paper_dir: Path) -> Optional[str]:
    """Download and extract LaTeX source from arXiv."""
    import requests

    source_url = f"https://arxiv.org/e-print/{arxiv_id}"

    try:
        response = requests.get(source_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        echo_warning(f"Could not download source: {e}")
        return None

    # Save and extract tarball
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        f.write(response.content)
        tar_path = Path(f.name)

    tex_content = None
    try:
        # Try to open as tar (most common)
        with tarfile.open(tar_path) as tar:
            tex_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".tex")]
            if tex_members:
                tex_by_name: dict[str, str] = {}
                for member in tex_members:
                    extracted = tar.extractfile(member)
                    if not extracted:
                        continue
                    tex_by_name[member.name] = extracted.read().decode("utf-8", errors="ignore")

                preferred_names = ("main.tex", "paper.tex")
                preferred = [n for n in tex_by_name if Path(n).name in preferred_names]
                if preferred:
                    main_name = preferred[0]
                else:
                    document_files = [n for n, c in tex_by_name.items() if "\\begin{document}" in c]
                    if document_files:
                        main_name = max(document_files, key=lambda n: len(tex_by_name[n]))
                    else:
                        main_name = max(tex_by_name, key=lambda n: len(tex_by_name[n]))

                main_content = tex_by_name[main_name]
                combined_parts: list[str] = [main_content]
                # Append other .tex files so equation extraction works even when main uses \input/\include.
                for name in sorted(tex_by_name):
                    if name == main_name:
                        continue
                    combined_parts.append(f"\n\n% --- file: {name} ---\n")
                    combined_parts.append(tex_by_name[name])
                tex_content = "".join(combined_parts)[:1_500_000]
    except tarfile.ReadError:
        # Might be a single gzipped file or plain tex
        import gzip

        try:
            with gzip.open(tar_path, "rt", encoding="utf-8", errors="ignore") as f:
                tex_content = f.read()
        except Exception:
            # Try as plain text
            tex_content = tar_path.read_text(errors="ignore")

    tar_path.unlink()

    if tex_content and "\\begin{document}" in tex_content:
        (paper_dir / "source.tex").write_text(tex_content)
        return tex_content

    return None


def extract_equations_simple(tex_content: str) -> str:
    """Extract equations from LaTeX source (simple regex-based extraction)."""
    equations = []

    # Find numbered equations
    eq_patterns = [
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        r"\\begin\{align\}(.*?)\\end\{align\}",
        r"\\begin\{align\*\}(.*?)\\end\{align\*\}",
        r"\\\[(.*?)\\\]",
    ]

    for pattern in eq_patterns:
        for match in re.finditer(pattern, tex_content, re.DOTALL):
            eq = match.group(1).strip()
            if eq and len(eq) > 5:  # Skip trivial equations
                equations.append(eq)

    if not equations:
        return "No equations extracted."

    md = "# Key Equations\n\n"
    for i, eq in enumerate(equations[:20], 1):  # Limit to first 20
        md += f"## Equation {i}\n```latex\n{eq}\n```\n\n"

    return md


_LATEX_SECTION_RE = re.compile(r"\\(sub)*section\*?\{([^}]*)\}", flags=re.IGNORECASE)


def _extract_section_headings(tex_content: str, *, max_items: int = 25) -> list[str]:
    headings: list[str] = []
    if not tex_content:
        return headings
    for match in _LATEX_SECTION_RE.finditer(tex_content):
        title = (match.group(2) or "").strip()
        if not title:
            continue
        title = re.sub(r"\s+", " ", title)
        if title and title not in headings:
            headings.append(title)
        if len(headings) >= max_items:
            break
    return headings


def _extract_equation_blocks(tex_content: str) -> list[str]:
    if not tex_content:
        return []

    blocks: list[str] = []
    patterns = [
        r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        r"\\begin\{align\*?\}.*?\\end\{align\*?\}",
        r"\\begin\{gather\*?\}.*?\\end\{gather\*?\}",
        r"\\\[[\s\S]*?\\\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, tex_content, flags=re.DOTALL):
            block = match.group(0).strip()
            if len(block) < 12:
                continue
            blocks.append(block)

    # de-dupe preserving order (common when files are concatenated)
    seen: set[str] = set()
    deduped: list[str] = []
    for b in blocks:
        key = b.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    return deduped


def _extract_name_from_title(title: str) -> Optional[str]:
    """Extract a short name from title prefix like 'NeRF: ...' → 'nerf'."""
    if ":" not in title:
        return None

    prefix = title.split(":")[0].strip()
    # Only use if it's short (1-3 words, under 30 chars)
    words = prefix.split()
    if len(words) <= 3 and len(prefix) <= 30:
        # Convert to lowercase, replace spaces with hyphens
        name = prefix.lower().replace(" ", "-")
        # Remove special chars except hyphens
        name = re.sub(r"[^a-z0-9-]", "", name)
        if name:
            return name
    return None


def _generate_name_with_llm(meta: dict) -> Optional[str]:
    """Ask LLM for a short memorable name."""
    prompt = f"""Given this paper title and abstract, suggest a single short name (1-2 words, lowercase, hyphenated if multi-word) that researchers commonly use to refer to this paper.

Examples:
- "Attention Is All You Need" → transformer
- "Deep Residual Learning for Image Recognition" → resnet
- "Generative Adversarial Networks" → gan
- "BERT: Pre-training of Deep Bidirectional Transformers" → bert

Return ONLY the name, nothing else. No quotes, no explanation.

Title: {meta["title"]}
Abstract: {meta["abstract"][:500]}"""

    result = _run_llm(prompt, purpose="name")
    if result:
        # Clean up the result - take first word/term only
        name = result.strip().lower().split()[0] if result.strip() else None
        if name:
            # Remove non-alphanumeric except hyphens, strip trailing hyphens
            name = re.sub(r"[^a-z0-9-]", "", name).strip("-")
            if name and 3 <= len(name) <= 30:
                return name
    return None


def generate_auto_name(meta: dict, existing_names: set[str], use_llm: bool = True) -> str:
    """Generate a short memorable name for a paper.

    Strategy:
    1. Extract from title prefix (e.g., "NeRF: ..." → "nerf")
    2. If LLM available, ask for a short name
    3. Fallback to arxiv ID
    4. Handle collisions by appending -2, -3, etc.
    """
    arxiv_id = meta.get("arxiv_id", "unknown")
    title = meta.get("title", "")

    # Try extracting from colon prefix
    name = _extract_name_from_title(title)

    # If no prefix name, try LLM
    if not name and use_llm and _litellm_available():
        name = _generate_name_with_llm(meta)

    # Fallback:
    # - arXiv ingest: arxiv ID
    # - local/meta-only ingest: slugified title
    if not name:
        if arxiv_id and arxiv_id != "unknown":
            name = str(arxiv_id).replace("/", "_").replace(".", "_")
        else:
            name = _slugify_title(str(title))

    # Handle collisions
    base_name = name
    counter = 2
    while name in existing_names:
        name = f"{base_name}-{counter}"
        counter += 1

    return name


def generate_llm_content(
    paper_dir: Path,
    meta: dict,
    tex_content: Optional[str],
    *,
    audit_reasons: Optional[list[str]] = None,
) -> tuple[str, str, list[str]]:
    """
    Generate summary, equations.md, and semantic tags using LLM.
    Returns (summary, equations_md, additional_tags)
    """
    if not _litellm_available():
        # Fallback: simple extraction without LLM
        summary = generate_simple_summary(meta, tex_content)
        equations = extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        return summary, equations, []

    try:
        return generate_with_litellm(meta, tex_content, audit_reasons=audit_reasons)
    except Exception as e:
        echo_warning(f"LLM generation failed: {e}")
        summary = generate_simple_summary(meta, tex_content)
        equations = extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        return summary, equations, []


def generate_simple_summary(meta: dict, tex_content: Optional[str] = None) -> str:
    """Generate a summary from metadata and optionally LaTeX structure (no LLM)."""
    title = meta.get("title") or "Untitled"
    arxiv_id = meta.get("arxiv_id")
    authors = meta.get("authors") or []
    published = meta.get("published")
    categories = meta.get("categories") or []
    abstract = meta.get("abstract") or ""

    lines: list[str] = [f"# {title}", ""]
    if arxiv_id:
        lines.append(f"**arXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
    if authors:
        shown_authors = ", ".join([str(a) for a in authors[:5]])
        lines.append(f"**Authors:** {shown_authors}{'...' if len(authors) > 5 else ''}")
    if published:
        lines.append(f"**Published:** {str(published)[:10]}")
    if categories:
        lines.append(f"**Categories:** {', '.join([str(c) for c in categories])}")

    lines.extend(["", "## Abstract", "", abstract])

    # Include section headings from LaTeX if available
    if tex_content:
        headings = _extract_section_headings(tex_content)
        if headings:
            lines.extend(["", "## Paper Structure", ""])
            for h in headings:
                lines.append(f"- {h}")

    lines.extend(["", "---"])
    regen_target = arxiv_id if arxiv_id else "<paper-name-or-arxiv-id>"
    lines.append(
        "*Summary auto-generated from metadata. Configure an LLM and run "
        f"`papi regenerate {regen_target}` for a richer summary.*"
    )
    lines.append("")
    return "\n".join(lines)


def _litellm_available() -> bool:
    """Check if LiteLLM is available."""
    try:
        import litellm  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def _run_llm(prompt: str, *, purpose: str) -> Optional[str]:
    """Run a prompt through LiteLLM. Returns None on any failure."""
    try:
        import litellm  # type: ignore[import-not-found]

        litellm.suppress_debug_info = True
    except ImportError:
        echo_error("LiteLLM not installed. Install with: pip install litellm")
        return None

    model = default_llm_model()
    echo_progress(f"  LLM ({model}): generating {purpose}...")

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=default_llm_temperature(),
        )
        out = response.choices[0].message.content  # type: ignore[union-attr]
        if out:
            out = out.strip()
        echo_progress(f"  LLM ({model}): {purpose} ok")
        return out or None
    except Exception as e:
        err_msg = str(e).split("\n")[0][:100]
        echo_error(f"LLM ({model}): {purpose} failed: {err_msg}")
        return None


def generate_with_litellm(
    meta: dict,
    tex_content: Optional[str],
    *,
    audit_reasons: Optional[list[str]] = None,
) -> tuple[str, str, list[str]]:
    """Generate summary, equations, and tags using LiteLLM.

    Simple approach: send title/abstract/raw LaTeX to the LLM and let it decide what's important.
    """
    title = str(meta.get("title") or "")
    authors = meta.get("authors") or []
    abstract = str(meta.get("abstract") or "")

    # Build context: metadata + raw LaTeX (will be truncated by _run_llm if needed)
    context_parts = [
        f"Paper: {title}",
        f"Authors: {', '.join([str(a) for a in authors[:10]])}",
        f"Abstract: {abstract}",
    ]
    if audit_reasons:
        context_parts.append("\nPrevious issues to address:")
        context_parts.extend([f"- {r}" for r in audit_reasons[:8]])
    if tex_content:
        context_parts.append("\nLaTeX source:")
        context_parts.append(tex_content)

    context = "\n".join(context_parts)

    # Generate summary
    summary_prompt = f"""Write a technical summary of this paper for a developer implementing the methods.

Include:
- Core contribution (1-2 sentences)
- Key methods/architecture
- Important implementation details

Keep it under 400 words. Use markdown. Only include information from the provided context.

{context}"""

    try:
        llm_summary = _run_llm(summary_prompt, purpose="summary")
        summary = llm_summary if llm_summary else generate_simple_summary(meta, tex_content)
    except Exception:
        summary = generate_simple_summary(meta, tex_content)

    # Generate equations.md
    if tex_content:
        eq_prompt = f"""Extract the key equations from this paper's LaTeX source.

For each important equation:
1. Show the LaTeX
2. Briefly explain what it represents
3. Note key variables

Focus on: definitions, loss functions, main results. Skip trivial math.
Use markdown with ```latex blocks.

{context}"""

        try:
            llm_equations = _run_llm(eq_prompt, purpose="equations")
            equations = llm_equations if llm_equations else extract_equations_simple(tex_content)
        except Exception:
            equations = extract_equations_simple(tex_content)
    else:
        equations = "No LaTeX source available."

    # Generate semantic tags
    tag_prompt = f"""Suggest 3-5 technical tags for this paper (lowercase, hyphenated).
Focus on methods, domains, techniques.
Return ONLY tags, one per line.

Title: {title}
Abstract: {abstract[:800]}"""

    additional_tags = []
    try:
        llm_tags_text = _run_llm(tag_prompt, purpose="tags")
        if llm_tags_text:
            additional_tags = [
                t.strip().lower().replace(" ", "-")
                for t in llm_tags_text.split("\n")
                if t.strip() and len(t.strip()) < 30
            ][:5]
    except Exception:
        pass

    return summary, equations, additional_tags


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
@click.version_option(version="0.3.0")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress messages.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug output.")
def cli(quiet: bool = False, verbose: bool = False):
    """paperpipe: Unified paper database for coding agents + PaperQA2."""
    set_quiet(quiet)
    if verbose:
        _setup_debug_logging()
    ensure_db()


def _add_single_paper(
    arxiv_id: str,
    name: Optional[str],
    tags: Optional[str],
    no_llm: bool,
    duplicate: bool,
    update: bool,
    index: dict,
    existing_names: set[str],
    base_to_names: dict[str, list[str]],
) -> tuple[bool, Optional[str], str]:
    """Add a single paper to the database.

    Returns (success, paper_name, action) tuple.
    """
    # Normalize arXiv ID / URL
    try:
        arxiv_id = normalize_arxiv_id(arxiv_id)
    except ValueError as e:
        echo_error(f"Invalid arXiv ID: {e}")
        return False, None, "failed"

    base = arxiv_base_id(arxiv_id)
    existing_for_arxiv = base_to_names.get(base, [])

    if existing_for_arxiv and not duplicate:
        # Idempotent by default: re-adding the same paper is a no-op.
        if not update:
            if name and name not in existing_for_arxiv:
                echo_error(
                    f"arXiv {base} already added as {', '.join(existing_for_arxiv)}; use --update or --duplicate."
                )
                return False, None, "failed"
            echo_warning(f"Already added (arXiv {base}): {', '.join(existing_for_arxiv)} (skipping)")
            return True, existing_for_arxiv[0], "skipped"

        # Update mode: refresh an existing entry in-place.
        if name:
            if name not in existing_for_arxiv:
                echo_error(f"arXiv {base} already added as {', '.join(existing_for_arxiv)}; cannot update '{name}'.")
                return False, None, "failed"
            target = name
        else:
            if len(existing_for_arxiv) > 1:
                echo_error(
                    f"Multiple papers match arXiv {base}: {', '.join(existing_for_arxiv)}. "
                    "Re-run with --name to pick one, or use --duplicate to add another copy."
                )
                return False, None, "failed"
            target = existing_for_arxiv[0]

        success, paper_name = _update_existing_paper(
            arxiv_id=arxiv_id,
            name=target,
            tags=tags,
            no_llm=no_llm,
            index=index,
            base_to_names=base_to_names,
        )
        return success, paper_name, "updated" if success else "failed"

    if name:
        if not _is_safe_paper_name(name):
            echo_error(f"Invalid paper name: {name!r}")
            return False, None, "failed"
        paper_dir = PAPERS_DIR / name
        if paper_dir.exists():
            echo_error(f"Paper '{name}' already exists. Use --name to specify a different name.")
            return False, None, "failed"
        if name in existing_names:
            echo_error(f"Paper '{name}' already in index. Use --name to specify a different name.")
            return False, None, "failed"

    # 1. Fetch metadata (needed for auto-name generation)
    echo_progress("  Fetching metadata...")
    try:
        meta = fetch_arxiv_metadata(arxiv_id)
    except Exception as e:
        echo_error(f"Error fetching metadata: {e}")
        return False, None, "failed"

    # 2. Generate name from title if not provided
    if not name:
        name = generate_auto_name(meta, existing_names, use_llm=not no_llm)
        echo_progress(f"  Auto-generated name: {name}")

    paper_dir = PAPERS_DIR / name

    if paper_dir.exists():
        echo_error(f"Paper '{name}' already exists. Use --name to specify a different name.")
        return False, None, "failed"

    if name in existing_names:
        echo_error(f"Paper '{name}' already in index. Use --name to specify a different name.")
        return False, None, "failed"

    paper_dir.mkdir(parents=True)

    # 3. Download PDF (for PaperQA2)
    echo_progress("  Downloading PDF...")
    pdf_path = paper_dir / "paper.pdf"
    try:
        download_pdf(arxiv_id, pdf_path)
    except Exception as e:
        echo_warning(f"Could not download PDF: {e}")

    # 4. Download LaTeX source
    echo_progress("  Downloading LaTeX source...")
    tex_content = download_source(arxiv_id, paper_dir)
    if tex_content:
        echo_progress(f"  Found LaTeX source ({len(tex_content) // 1000}k chars)")
    else:
        echo_progress("  No LaTeX source available (PDF-only submission)")

    # 5. Generate tags
    auto_tags = categories_to_tags(meta["categories"])
    user_tags = [t.strip() for t in tags.split(",")] if tags else []

    # 6. Generate summary and equations
    echo_progress("  Generating summary and equations...")
    if no_llm:
        summary = generate_simple_summary(meta, tex_content)
        equations = extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        llm_tags: list[str] = []
    else:
        summary, equations, llm_tags = generate_llm_content(paper_dir, meta, tex_content)

    # Combine all tags
    all_tags = normalize_tags([*auto_tags, *user_tags, *llm_tags])

    # 7. Save files
    (paper_dir / "summary.md").write_text(summary)
    (paper_dir / "equations.md").write_text(equations)

    # Save metadata
    paper_meta = {
        "arxiv_id": meta["arxiv_id"],
        "title": meta["title"],
        "authors": meta["authors"],
        "abstract": meta["abstract"],
        "categories": meta["categories"],
        "tags": all_tags,
        "published": meta["published"],
        "added": datetime.now().isoformat(),
        "has_source": tex_content is not None,
        "has_pdf": pdf_path.exists(),
    }
    (paper_dir / "meta.json").write_text(json.dumps(paper_meta, indent=2))
    ensure_notes_file(paper_dir, paper_meta)

    # 8. Update index
    index[name] = {
        "arxiv_id": meta["arxiv_id"],
        "title": meta["title"],
        "tags": all_tags,
        "added": paper_meta["added"],
    }
    save_index(index)

    # Update existing_names for subsequent papers in batch
    existing_names.add(name)
    base_to_names.setdefault(base, []).append(name)
    base_to_names[base].sort()

    echo_success(f"Added: {name}")
    click.echo(f"  Title: {_format_title_short(str(meta['title']))}")
    click.echo(f"  Tags: {', '.join(all_tags)}")
    click.echo(f"  Location: {paper_dir}")

    return True, name, "added"


def _add_local_pdf(
    *,
    pdf: Path,
    title: str,
    name: Optional[str],
    tags: Optional[str],
    authors: Optional[str],
    abstract: Optional[str],
    year: Optional[int],
    venue: Optional[str],
    doi: Optional[str],
    url: Optional[str],
    no_llm: bool,
) -> tuple[bool, Optional[str]]:
    """Add a local PDF as a first-class paper entry."""
    if not pdf.exists() or not pdf.is_file():
        echo_error(f"PDF not found: {pdf}")
        return False, None
    if not _looks_like_pdf(pdf):
        echo_error(f"File does not look like a PDF (missing %PDF- header): {pdf}")
        return False, None

    title = (title or "").strip()
    if not title:
        echo_error("Missing title for local PDF ingestion.")
        return False, None

    abstract_text = (abstract or "").strip()
    if not abstract_text:
        abstract_text = "No abstract available (local PDF)."

    if year is not None and not (1000 <= year <= 3000):
        echo_error("Invalid --year (expected YYYY)")
        return False, None

    index = load_index()
    existing_names = set(index.keys())

    if name:
        if not _is_safe_paper_name(name):
            echo_error(f"Invalid paper name: {name!r}")
            return False, None
        if name in existing_names or (PAPERS_DIR / name).exists():
            echo_error(f"Paper '{name}' already exists. Use --name to specify a different name.")
            return False, None
    else:
        candidate = _generate_local_pdf_name({"title": title, "abstract": ""}, use_llm=not no_llm)
        if candidate in existing_names or (PAPERS_DIR / candidate).exists():
            echo_error(
                f"Name conflict for local PDF '{title}': '{candidate}' already exists. "
                "Re-run with --name to pick a different name."
            )
            return False, None
        name = candidate
        echo_progress(f"  Auto-generated name: {name}")

    if not name:
        echo_error("Failed to determine a paper name (use --name to set one explicitly).")
        return False, None
    paper_dir = PAPERS_DIR / name
    paper_dir.mkdir(parents=True)

    echo_progress("  Copying PDF...")
    dest_pdf = paper_dir / "paper.pdf"
    shutil.copy2(pdf, dest_pdf)

    user_tags = [t.strip() for t in (tags or "").split(",") if t.strip()]
    all_tags = normalize_tags(user_tags)

    meta: dict[str, Any] = {
        "arxiv_id": None,
        "title": title,
        "authors": _parse_authors(authors),
        "abstract": abstract_text,
        "categories": [],
        "tags": all_tags,
        "published": None,
        "year": year,
        "venue": (venue or "").strip() or None,
        "doi": (doi or "").strip() or None,
        "url": (url or "").strip() or None,
        "added": datetime.now().isoformat(),
        "has_source": False,
        "has_pdf": dest_pdf.exists(),
    }

    # Best-effort artifacts (no PDF parsing in MVP)
    summary = generate_simple_summary(meta, None)
    equations = "No LaTeX source available."

    (paper_dir / "summary.md").write_text(summary)
    (paper_dir / "equations.md").write_text(equations)
    (paper_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    ensure_notes_file(paper_dir, meta)

    index[name] = {"arxiv_id": None, "title": title, "tags": all_tags, "added": meta["added"]}
    save_index(index)

    echo_success(f"Added: {name}")
    click.echo(f"  Title: {_format_title_short(title)}")
    click.echo(f"  Tags: {', '.join(all_tags)}")
    click.echo(f"  Location: {paper_dir}")
    return True, name


def _update_existing_paper(
    *,
    arxiv_id: str,
    name: str,
    tags: Optional[str],
    no_llm: bool,
    index: dict,
    base_to_names: dict[str, list[str]],
) -> tuple[bool, Optional[str]]:
    """Refresh an existing paper in-place (PDF/source/meta + generated content)."""
    paper_dir = PAPERS_DIR / name
    paper_dir.mkdir(parents=True, exist_ok=True)

    meta_path = paper_dir / "meta.json"
    prior_meta: dict = {}
    if meta_path.exists():
        try:
            prior_meta = json.loads(meta_path.read_text())
        except Exception:
            prior_meta = {}

    echo_progress(f"Updating existing paper: {name}")

    echo_progress("  Fetching metadata...")
    try:
        meta = fetch_arxiv_metadata(arxiv_id)
    except Exception as e:
        echo_error(f"Error fetching metadata: {e}")
        return False, None

    # Download PDF (overwrite if present)
    echo_progress("  Downloading PDF...")
    pdf_path = paper_dir / "paper.pdf"
    try:
        download_pdf(arxiv_id, pdf_path)
    except Exception as e:
        echo_warning(f"Could not download PDF: {e}")

    # Download LaTeX source (only overwrites if source is valid)
    echo_progress("  Downloading LaTeX source...")
    tex_content = download_source(arxiv_id, paper_dir)
    if not tex_content:
        source_path = paper_dir / "source.tex"
        if source_path.exists():
            tex_content = source_path.read_text(errors="ignore")

    # Tags: merge prior tags + new auto tags + optional user tags (+ LLM tags if used)
    auto_tags = categories_to_tags(meta.get("categories", []))
    prior_tags_raw = prior_meta.get("tags")
    prior_tags = prior_tags_raw if isinstance(prior_tags_raw, list) else []
    user_tags = [t.strip() for t in tags.split(",")] if tags else []

    echo_progress("  Generating summary and equations...")
    if no_llm:
        summary = generate_simple_summary(meta, tex_content)
        equations = extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        llm_tags: list[str] = []
    else:
        summary, equations, llm_tags = generate_llm_content(paper_dir, meta, tex_content)

    all_tags = normalize_tags([*auto_tags, *prior_tags, *user_tags, *llm_tags])

    (paper_dir / "summary.md").write_text(summary)
    (paper_dir / "equations.md").write_text(equations)

    paper_meta = {
        "arxiv_id": meta.get("arxiv_id"),
        "title": meta.get("title"),
        "authors": meta.get("authors", []),
        "abstract": meta.get("abstract", ""),
        "categories": meta.get("categories", []),
        "tags": all_tags,
        "published": meta.get("published"),
        "added": prior_meta.get("added") or datetime.now().isoformat(),
        "has_source": tex_content is not None,
        "has_pdf": pdf_path.exists(),
    }
    meta_path.write_text(json.dumps(paper_meta, indent=2))
    ensure_notes_file(paper_dir, paper_meta)

    index[name] = {
        "arxiv_id": meta.get("arxiv_id"),
        "title": meta.get("title"),
        "tags": all_tags,
        "added": paper_meta["added"],
    }
    save_index(index)

    base = arxiv_base_id(str(meta.get("arxiv_id") or arxiv_id))
    base_to_names.setdefault(base, [])
    if name not in base_to_names[base]:
        base_to_names[base].append(name)
        base_to_names[base].sort()

    echo_success(f"Updated: {name}")
    click.echo(f"  Title: {_format_title_short(str(meta.get('title', '')))}")
    click.echo(f"  Tags: {', '.join(all_tags)}")
    click.echo(f"  Location: {paper_dir}")
    return True, name


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
@click.option(
    "--duplicate",
    is_flag=True,
    help="Allow adding a second copy even if this arXiv ID already exists (creates a new name like -2/-3).",
)
@click.option(
    "--update", is_flag=True, help="If this arXiv ID already exists, refresh it in-place instead of skipping."
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
    duplicate: bool,
    update: bool,
):
    """Add one or more papers to the database."""
    if pdf:
        if arxiv_ids:
            raise click.UsageError("Use either arXiv IDs/URLs OR `--pdf`, not both.")
        if not title or not title.strip():
            raise click.UsageError("Missing required option: --title (required with --pdf).")
        if duplicate or update:
            raise click.UsageError("--duplicate/--update are only supported for arXiv ingestion.")
        success, _ = _add_local_pdf(
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
        )
        if not success:
            raise SystemExit(1)
        return

    if not arxiv_ids:
        raise click.UsageError("Missing arXiv ID/URL argument(s) (or pass `--pdf`).")

    if name and len(arxiv_ids) > 1:
        raise click.UsageError("--name can only be used when adding a single paper.")
    if duplicate and update:
        raise click.UsageError("Use either --duplicate or --update, not both.")

    index = load_index()
    existing_names = set(index.keys())
    base_to_names = _index_arxiv_base_to_names(index)

    added = 0
    updated = 0
    skipped = 0
    failures = 0

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        if len(arxiv_ids) > 1:
            echo_progress(f"[{i}/{len(arxiv_ids)}] Adding {arxiv_id}...")
        else:
            echo_progress(f"Adding paper: {arxiv_id}")

        success, _, action = _add_single_paper(
            arxiv_id,
            name,
            tags,
            no_llm,
            duplicate,
            update,
            index,
            existing_names,
            base_to_names,
        )
        if success:
            if action == "added":
                added += 1
            elif action == "updated":
                updated += 1
            elif action == "skipped":
                skipped += 1
        else:
            failures += 1

    # Print summary for multiple papers
    if len(arxiv_ids) > 1:
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


def _regenerate_one_paper(
    name: str,
    index: dict,
    *,
    no_llm: bool,
    overwrite_fields: set[str],
    overwrite_all: bool,
    audit_reasons: Optional[list[str]] = None,
) -> tuple[bool, Optional[str]]:
    """Regenerate fields for a paper. Returns (success, new_name or None)."""
    paper_dir = PAPERS_DIR / name
    meta_path = paper_dir / "meta.json"
    if not meta_path.exists():
        echo_error(f"Missing metadata for: {name} ({meta_path})")
        return False, None

    meta = json.loads(meta_path.read_text())
    tex_content = None
    source_path = paper_dir / "source.tex"
    if source_path.exists():
        tex_content = source_path.read_text(errors="ignore")

    summary_path = paper_dir / "summary.md"
    equations_path = paper_dir / "equations.md"

    # Determine what needs regeneration
    if overwrite_all:
        do_summary = True
        do_equations = True
        do_tags = True
        do_name = True
    elif overwrite_fields:
        do_summary = "summary" in overwrite_fields
        do_equations = "equations" in overwrite_fields
        do_tags = "tags" in overwrite_fields
        do_name = "name" in overwrite_fields
    else:
        do_summary = not summary_path.exists() or summary_path.stat().st_size == 0
        do_equations = not equations_path.exists() or equations_path.stat().st_size == 0
        do_tags = not meta.get("tags")
        do_name = _is_arxiv_id_name(name)

    if not (do_summary or do_equations or do_tags or do_name):
        echo_progress(f"  {name}: nothing to regenerate")
        return True, None

    actions: list[str] = []
    if do_summary:
        actions.append("summary")
    if do_equations:
        actions.append("equations")
    if do_tags:
        actions.append("tags")
    if do_name:
        actions.append("name")
    echo_progress(f"Regenerating {name}: {', '.join(actions)}")

    new_name: Optional[str] = None
    updated_meta = False

    # Regenerate name if requested
    if do_name:
        existing_names = set(index.keys()) - {name}
        candidate = generate_auto_name(meta, existing_names, use_llm=not no_llm)
        if candidate != name:
            new_dir = PAPERS_DIR / candidate
            if new_dir.exists():
                echo_warning(f"Cannot rename to '{candidate}' (already exists)")
            else:
                paper_dir.rename(new_dir)
                paper_dir = new_dir
                summary_path = paper_dir / "summary.md"
                equations_path = paper_dir / "equations.md"
                meta_path = paper_dir / "meta.json"
                new_name = candidate
                echo_progress(f"  Renamed: {name} → {candidate}")

    # Generate content based on what's needed
    summary: Optional[str] = None
    equations: Optional[str] = None
    llm_tags: list[str] = []

    if do_summary or do_equations or do_tags:
        if no_llm:
            if do_summary:
                summary = generate_simple_summary(meta, tex_content)
            if do_equations:
                equations = extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        else:
            llm_summary, llm_equations, llm_tags = generate_llm_content(
                paper_dir,
                meta,
                tex_content,
                audit_reasons=audit_reasons,
            )
            if do_summary:
                summary = llm_summary
            if do_equations:
                equations = llm_equations
            if not do_tags:
                llm_tags = []

    if summary is not None:
        summary_path.write_text(summary)

    if equations is not None:
        equations_path.write_text(equations)

    if llm_tags:
        meta["tags"] = normalize_tags([*meta.get("tags", []), *llm_tags])
        updated_meta = True

    if updated_meta:
        meta_path.write_text(json.dumps(meta, indent=2))

    ensure_notes_file(paper_dir, meta)

    # Update index
    current_name = new_name if new_name else name
    if new_name:
        if name in index:
            del index[name]
        index[current_name] = {
            "arxiv_id": meta.get("arxiv_id"),
            "title": meta.get("title"),
            "tags": meta.get("tags", []),
            "added": meta.get("added"),
        }
        save_index(index)
    elif updated_meta:
        index_entry = index.get(current_name, {})
        index_entry["tags"] = meta.get("tags", [])
        index[current_name] = index_entry
        save_index(index)

    echo_success("  Done")
    return True, new_name


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
      --overwrite tags,summary  Regenerate tags and summary

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

        paper_dir = PAPERS_DIR / name
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
                new_dir = PAPERS_DIR / set_name
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
def search(query: str, limit: int, fuzzy: bool, tex: bool):
    """Search papers by title, tags, metadata, and local content."""
    index = load_index()

    def collect_results(*, fuzzy_mode: bool) -> list[tuple[str, dict, int, list[str]]]:
        results: list[tuple[str, dict, int, list[str]]] = []
        for name, info in index.items():
            paper_dir = PAPERS_DIR / name
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


_AUDIT_EQUATIONS_TITLE_RE = re.compile(r'paper\s+\*\*["“](.+?)["”]\*\*', flags=re.IGNORECASE)
_AUDIT_BOLD_RE = re.compile(r"\*\*([^*\n]{3,80})\*\*")
_AUDIT_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,9}\b")

_AUDIT_IGNORED_WORDS = {
    # Section/document terms
    "core",
    "contribution",
    "key",
    "overview",
    "summary",
    "equations",
    "equation",
    "notes",
    "details",
    "discussion",
    "background",
    "related",
    "work",
    "results",
    "paper",
    # Generic ML/technical terms
    "method",
    "methods",
    "architecture",
    "important",
    "implementation",
    "loss",
    "losses",
    "functions",
    "training",
    "objectives",
    "variables",
    "representation",
    "standard",
    "total",
    "approach",
    "model",
    "models",
    # Common technical vocabulary (often used in summaries but not always in abstracts)
    "optimization",
    "regularization",
    "extraction",
    "refinement",
    "distillation",
    "supervision",
    "efficiency",
    "handling",
    "flexibility",
    "robustness",
    "strategy",
    "schedule",
    "scheduler",
    "processing",
    "calculation",
    "masking",
    "residuals",
    "hyperparameters",
    "hyperparameter",
    "awareness",
    "hardware",
    "specs",
    "normalization",
    "initialization",
    "convergence",
    "inference",
    "prediction",
    "interpolation",
    "extrapolation",
    "aggregation",
    "sampling",
    "weighting",
    "management",
    "configuration",
    "integration",
}

# Common acronyms that shouldn't trigger hallucination warnings.
# Keep this broad and domain-agnostic (general computing, math, common paper terms).
_AUDIT_ACRONYM_ALLOWLIST = {
    # General computing/tech
    "API",
    "CPU",
    "GPU",
    "TPU",
    "RAM",
    "SSD",
    "HTTP",
    "JSON",
    "XML",
    "SQL",
    "PDF",
    "URL",
    "IEEE",
    "ACM",
    "CUDA",
    "FPS",
    "RGB",
    "RGBA",
    "HDR",
    # Math/stats
    "IID",
    "ODE",
    "PDE",
    "SVD",
    "PCA",
    "KKT",
    "CDF",
    "MSE",
    "MAE",
    "RMSE",
    "PSNR",
    "SSIM",
    "LPIPS",
    "IOU",
    # Common ML architectures and techniques
    "AI",
    "ML",
    "DL",
    "RL",
    "NLP",
    "LLM",
    "CNN",
    "RNN",
    "MLP",
    "LSTM",
    "GRU",
    "GAN",
    "VAE",
    "BERT",
    "GPT",
    "VIT",
    "CLIP",
    # Optimizers/training
    "SGD",
    "ADAM",
    "LBFGS",
    "BCE",
    # Graphics/vision
    "SDF",
    "BRDF",
    "BSDF",
    "HDR",
    "LOD",
    "FOV",
    # Norms/metrics
    "L1",
    "L2",
}

# LLM boilerplate phrases that indicate prompt leakage or missing content.
# Only flag phrases that are actual problems, not normal academic writing style.
_AUDIT_BOILERPLATE_PHRASES = [
    # Prompt leakage (LLM responding to instructions rather than generating content)
    "based on the provided",
    "based on the given",
    "from the provided",
    "from the given",
    "i cannot",
    "i can't",
    "i don't have access",
    "i do not have access",
    # Missing content indicators
    "no latex source available",
    "no equations available",
    "no source available",
    "not available in the",
]


def _extract_referenced_title_from_equations(text: str) -> Optional[str]:
    match = _AUDIT_EQUATIONS_TITLE_RE.search(text or "")
    if not match:
        return None
    title = match.group(1).strip()
    return title or None


def _extract_suspicious_tokens_from_summary(summary_text: str) -> list[str]:
    """
    Extract a small set of tokens that are likely to be groundable in source.tex/abstract.

    Heuristics:
    - bold phrases followed by ':' often name specific components ("Eikonal Regularization")
    - acronyms (ROS, ONNX, CUDA)
    """
    tokens: list[str] = []

    for match in _AUDIT_BOLD_RE.finditer(summary_text or ""):
        phrase = match.group(1).strip()
        next_char = (summary_text or "")[match.end() : match.end() + 1]
        if not (phrase.endswith(":") or next_char == ":"):
            continue
        phrase = phrase.rstrip(":").strip()
        for token in re.findall(r"[A-Za-z]{5,}", phrase):
            if token.lower() in _AUDIT_IGNORED_WORDS:
                continue
            tokens.append(token)

    for token in _AUDIT_ACRONYM_RE.findall(summary_text or ""):
        if token in _AUDIT_ACRONYM_ALLOWLIST:
            continue
        tokens.append(token)

    # Dedupe preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(token)
    return ordered[:20]


def _extract_summary_title(summary_text: str) -> Optional[str]:
    """Extract title from summary heading (e.g., '# Paper Title' or '## Paper Title')."""
    if not summary_text:
        return None
    for line in summary_text.split("\n")[:5]:
        line = line.strip()
        if line.startswith("#"):
            # Remove markdown heading markers
            title = line.lstrip("#").strip()
            if title:
                return title
    return None


def _check_boilerplate(text: str) -> list[str]:
    """Return list of boilerplate phrases found in text."""
    if not text:
        return []
    low = text.lower()
    found = []
    for phrase in _AUDIT_BOILERPLATE_PHRASES:
        if phrase in low:
            found.append(phrase)
    return found[:3]  # Limit to avoid noisy output


def _audit_paper_dir(paper_dir: Path) -> list[str]:
    reasons: list[str] = []
    meta_path = paper_dir / "meta.json"
    summary_path = paper_dir / "summary.md"
    equations_path = paper_dir / "equations.md"
    source_path = paper_dir / "source.tex"

    if not meta_path.exists():
        return ["missing meta.json"]

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return ["invalid meta.json"]

    title = (meta.get("title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()
    if not title:
        reasons.append("meta.json missing title")

    if not summary_path.exists() or summary_path.stat().st_size == 0:
        reasons.append("missing summary.md")
    if not equations_path.exists() or equations_path.stat().st_size == 0:
        reasons.append("missing equations.md")

    equations_text = _read_text_limited(equations_path, max_chars=120_000) if equations_path.exists() else ""
    summary_text = _read_text_limited(summary_path, max_chars=160_000) if summary_path.exists() else ""

    # Check for title mismatch in equations.md
    referenced_title = _extract_referenced_title_from_equations(equations_text)
    if referenced_title and title:
        ratio = SequenceMatcher(None, referenced_title.lower(), title.lower()).ratio()
        if ratio < 0.8:
            reasons.append(f"equations.md references different title: {referenced_title!r}")

    # Check for title mismatch in summary.md heading
    # Instead of similarity matching, check if paper's short name/acronym appears in heading
    # Allow generic section headings that don't claim to be about a specific paper
    _GENERIC_HEADING_PREFIXES = {
        "core contribution",
        "key methods",
        "key contribution",
        "technical summary",
        "summary",
        "overview",
        "main contribution",
        "architecture",
        "methods",
    }
    summary_title = _extract_summary_title(summary_text)
    if summary_title and title:
        summary_lower = summary_title.lower()
        # Skip check for generic headings (they don't claim to be about a specific paper)
        is_generic = any(
            summary_lower.startswith(prefix) or summary_lower == prefix for prefix in _GENERIC_HEADING_PREFIXES
        )
        if not is_generic:
            title_lower = title.lower()
            # Extract short name (before colon) and acronyms from title
            short_name = title.split(":")[0].strip() if ":" in title else None
            acronyms = re.findall(r"\b[A-Z][A-Za-z]*[A-Z]+[A-Za-z]*\b|\b[A-Z]{2,}\b", title)
            # Check if short name or any acronym appears in heading
            found_match = False
            if short_name and short_name.lower() in summary_lower:
                found_match = True
            for acr in acronyms:
                if acr.lower() in summary_lower:
                    found_match = True
                    break
            # Also accept if significant title words appear
            if not found_match:
                title_words = [
                    w
                    for w in re.findall(r"[A-Za-z]{4,}", title_lower)
                    if w
                    not in {
                        "with",
                        "from",
                        "this",
                        "that",
                        "using",
                        "based",
                        "neural",
                        "learning",
                        "network",
                        "networks",
                    }
                ]
                for word in title_words[:5]:
                    if word in summary_lower:
                        found_match = True
                        break
            if not found_match:
                reasons.append(f"summary.md heading doesn't reference paper: {summary_title!r}")

    # Check for incomplete context markers
    if "provided latex snippet ends" in equations_text.lower():
        reasons.append("equations.md indicates incomplete LaTeX context")

    # Check for LLM boilerplate in summary
    boilerplate_in_summary = _check_boilerplate(summary_text)
    if boilerplate_in_summary:
        reasons.append(f"summary.md contains boilerplate: {', '.join(repr(p) for p in boilerplate_in_summary)}")

    # Check for LLM boilerplate in equations
    boilerplate_in_equations = _check_boilerplate(equations_text)
    if boilerplate_in_equations:
        reasons.append(f"equations.md contains boilerplate: {', '.join(repr(p) for p in boilerplate_in_equations)}")

    # Check for ungrounded terms in summary (domain-agnostic: extracts specific terms and checks source)
    evidence_parts: list[str] = [abstract]
    if source_path.exists():
        evidence_parts.append(_read_text_limited(source_path, max_chars=800_000))
    evidence = "\n".join(evidence_parts)
    evidence_lower = evidence.lower()

    missing_tokens: list[str] = []
    for token in _extract_suspicious_tokens_from_summary(summary_text):
        if token.lower() in evidence_lower:
            continue
        missing_tokens.append(token)
        if len(missing_tokens) >= 5:
            break
    if missing_tokens:
        reasons.append(f"summary.md contains terms not found in source/abstract: {', '.join(missing_tokens)}")

    return reasons


def _parse_selection_spec(spec: str, *, max_index: int) -> list[int]:
    raw = (spec or "").strip().lower()
    if not raw:
        return []
    if raw in {"a", "all", "*"}:
        return list(range(1, max_index + 1))

    selected: set[int] = set()
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if "-" in part:
            lo_s, hi_s = [p.strip() for p in part.split("-", 1)]
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                selected.add(i)
        else:
            selected.add(int(part))

    return sorted(i for i in selected if 1 <= i <= max_index)


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
        paper_dir = PAPERS_DIR / name
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
    type=click.Choice(["summary", "equations", "full"], case_sensitive=False),
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

        paper_dir = PAPERS_DIR / name
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


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("query")
@click.option(
    "--llm",
    default=None,
    show_default=False,
    help=("LLM model for answer generation (LiteLLM id; e.g., gpt-4o, claude-sonnet-4-5, gemini/gemini-2.5-flash)."),
)
@click.option(
    "--summary-llm",
    default=None,
    show_default=False,
    help="LLM for evidence summarization (often a cheaper/faster model than --llm).",
)
@click.option(
    "--embedding",
    default=None,
    show_default=False,
    help="Embedding model for text chunks (e.g., text-embedding-3-small, voyage-3-lite).",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=None,
    show_default=False,
    help="LLM temperature (0.0-1.0). Lower = more deterministic.",
)
@click.option(
    "-v",
    "--verbosity",
    type=int,
    default=None,
    show_default=False,
    help="Logging verbosity level (0-3). 3 = log all LLM/embedding calls.",
)
@click.option(
    "--answer-length",
    default=None,
    show_default=False,
    help="Target answer length (e.g., 'about 200 words', 'short', '3 paragraphs').",
)
@click.option(
    "--evidence-k",
    type=int,
    default=None,
    show_default=False,
    help="Number of evidence pieces to retrieve (default: 10).",
)
@click.option(
    "--max-sources",
    type=int,
    default=None,
    show_default=False,
    help="Maximum number of sources to cite in the answer (default: 5).",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    show_default=False,
    help="Agent timeout in seconds (default: 500).",
)
@click.option(
    "--concurrency",
    type=int,
    default=None,
    show_default=False,
    help="Indexing concurrency (default: 1). Higher values speed up indexing but may cause rate limits.",
)
@click.option(
    "--rebuild-index",
    is_flag=True,
    default=False,
    help="Force a full rebuild of the paper index.",
)
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Retry docs previously marked failed (clears ERROR markers in the index).",
)
@click.pass_context
def ask(
    ctx,
    query: str,
    llm: Optional[str],
    summary_llm: Optional[str],
    embedding: Optional[str],
    temperature: Optional[float],
    verbosity: Optional[int],
    answer_length: Optional[str],
    evidence_k: Optional[int],
    max_sources: Optional[int],
    timeout: Optional[float],
    concurrency: Optional[int],
    rebuild_index: bool,
    retry_failed: bool,
):
    """
    Query papers using PaperQA2 (if installed).

    Common options are exposed as first-class flags. Any additional arguments
    are passed directly to PaperQA2 (e.g., --agent.search_count 10).
    """
    if not shutil.which("pqa"):
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
    if not has_parsing_override and not _pillow_available():
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
    if embedding_for_pqa:
        cmd.extend(["--embedding", embedding_for_pqa])

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
        # Create a stable index name from the embedding model.
        # Replace special chars that might cause filesystem issues.
        safe_name = embedding_for_pqa.replace("/", "_").replace(":", "_")
        cmd.extend(["--agent.index.name", f"paperpipe_{safe_name}"])

    # Determine effective index params to check for exclusions (files marked ERROR)
    # We need to look at both what we've built so far and what the user passed
    effective_args = cmd + ctx.args
    idx_dir_val = _extract_flag_value(
        effective_args, names={"--agent.index.index_directory", "--agent.index.index-directory"}
    )
    idx_name_val = _extract_flag_value(effective_args, names={"--index", "-i", "--agent.index.name"})

    excluded_files: set[str] = set()
    if idx_dir_val and idx_name_val and not retry_failed:
        # Load index and find errors
        fp = _paperqa_index_files_path(index_directory=Path(idx_dir_val), index_name=idx_name_val)
        if fp.exists():
            m = _paperqa_load_index_files_map(fp)
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
        staging_dir = (PAPER_DB / ".pqa_papers").expanduser()
        _refresh_pqa_pdf_staging_dir(staging_dir=staging_dir, exclude_names=excluded_files)
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

    # summary_llm: first-class option takes precedence, then config, then falls back to llm_for_pqa
    summary_llm_source = ctx.get_parameter_source("summary_llm")
    has_summary_llm_passthrough = any(
        arg in {"--summary_llm", "--summary-llm"} or arg.startswith(("--summary_llm=", "--summary-llm="))
        for arg in ctx.args
    )
    if summary_llm_source != click.core.ParameterSource.DEFAULT:
        # Explicit CLI --summary-llm takes precedence
        if summary_llm:
            cmd.extend(["--summary_llm", summary_llm])
    elif not has_summary_llm_passthrough:
        summary_llm_default = default_pqa_summary_llm(llm_for_pqa)
        if summary_llm_default:
            cmd.extend(["--summary_llm", summary_llm_default])

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

    # temperature
    temperature_source = ctx.get_parameter_source("temperature")
    has_temperature_passthrough = any(arg in {"--temperature"} or arg.startswith("--temperature=") for arg in ctx.args)
    if temperature_source != click.core.ParameterSource.DEFAULT:
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
    elif not has_temperature_passthrough:
        temperature_default = default_pqa_temperature()
        if temperature_default is not None:
            cmd.extend(["--temperature", str(temperature_default)])

    # verbosity
    verbosity_source = ctx.get_parameter_source("verbosity")
    has_verbosity_passthrough = any(arg in {"--verbosity", "-v"} or arg.startswith("--verbosity=") for arg in ctx.args)
    if verbosity_source != click.core.ParameterSource.DEFAULT:
        if verbosity is not None:
            cmd.extend(["--verbosity", str(verbosity)])
    elif not has_verbosity_passthrough:
        verbosity_default = default_pqa_verbosity()
        if verbosity_default is not None:
            cmd.extend(["--verbosity", str(verbosity_default)])

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
    index_dir_raw = _extract_flag_value(
        cmd,
        names={"--agent.index.index_directory", "--agent.index.index-directory"},
    )
    index_name_raw = _extract_flag_value(
        cmd,
        names={"--agent.index.name"},
    ) or _extract_flag_value(cmd, names={"--index", "-i"})

    if index_dir_raw and index_name_raw:
        files_path = _paperqa_index_files_path(index_directory=Path(index_dir_raw), index_name=index_name_raw)
        mapping = _paperqa_load_index_files_map(files_path) if files_path.exists() else None
        failed_count = sum(1 for v in (mapping or {}).values() if v == "ERROR")
        if failed_count and not retry_failed:
            echo_warning(
                f"PaperQA2 index '{index_name_raw}' has {failed_count} failed document(s) (marked ERROR); "
                "PaperQA2 will not retry them automatically. Re-run with --retry-failed "
                "or --rebuild-index to rebuild the whole index."
            )
        if retry_failed:
            cleared, cleared_files = _paperqa_clear_failed_documents(
                index_directory=Path(index_dir_raw),
                index_name=index_name_raw,
            )
            if cleared:
                echo_progress(f"Cleared {cleared} failed PaperQA2 document(s) for retry.")
                debug("Cleared failed PaperQA2 docs: %s", ", ".join(cleared_files[:50]))

    cmd.extend(["ask", query])

    # Run pqa with real-time output streaming while capturing for crash detection
    # We merge stderr into stdout so we can stream everything in order
    proc = subprocess.Popen(cmd, cwd=PAPERS_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    captured_output: list[str] = []
    assert proc.stdout is not None  # for type checker
    for line in proc.stdout:
        click.echo(line, nl=False)
        captured_output.append(line)
    returncode = proc.wait()

    # Handle pqa failures gracefully
    if returncode != 0:
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
                _paperqa_effective_paper_directory(cmd, base_dir=PAPERS_DIR) or (PAPER_DB / ".pqa_papers").expanduser()
            )
            if paper_dir.exists():
                f = _paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc=crashing_doc)
                if f is not None:
                    count, _ = _paperqa_mark_failed_documents(
                        index_directory=Path(index_dir_raw),
                        index_name=index_name_raw,
                        staged_files={str(f)},
                    )
                    if count:
                        # Only remove files from paperpipe's managed staging directory.
                        # Never delete from a user-provided paper directory.
                        managed_staging_dir = (PAPER_DB / ".pqa_papers").expanduser()
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
            mapping = _paperqa_load_index_files_map(
                _paperqa_index_files_path(index_directory=Path(index_dir_raw), index_name=index_name_raw)
            )
            failed_docs = sorted([k for k, v in (mapping or {}).items() if v == "ERROR"])
            if failed_docs:
                echo_warning(f"PaperQA2 failed. {len(failed_docs)} document(s) excluded from indexing.")
                echo_warning("This can happen with PDFs that have text extraction issues (e.g., surrogate characters).")
                echo_warning("Options:")
                echo_warning("  1. Remove problematic paper(s) entirely: papi remove <name>")
                echo_warning("  2. Re-run query (excluded docs will stay excluded): papi ask '...'")
                echo_warning("  3. Re-stage excluded docs for retry: papi ask '...' --retry-failed")
                echo_warning("  4. Rebuild index from scratch: papi ask '...' --rebuild-index")
                if len(failed_docs) <= 5:
                    echo_warning(f"Failed documents: {', '.join(Path(f).stem for f in failed_docs)}")
                raise SystemExit(1)
        # Generic failure message if we can't determine the cause
        echo_error("PaperQA2 failed. Check the output above for details.")
        raise SystemExit(returncode)


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
        if kind_name == "completion":
            llm_completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word 'pong'."}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            llm_embedding(model=model, input=["ping"], timeout=embedding_timeout)

    def probe_group(kind_name: str, candidates: list[str]) -> _ModelProbeResult:
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                if verbose:
                    probe_one(kind_name, candidate)
                else:
                    with redirect_stdout(null_out), redirect_stderr(null_err):
                        probe_one(kind_name, candidate)
                return _ModelProbeResult(kind=kind_name, model=candidate, ok=True)
            except Exception as exc:
                last_exc = exc
                continue

        err = _first_line(str(last_exc)) if last_exc else "Unknown error"
        hint = _probe_hint(kind=kind_name, model=candidates[0], error_line=err)
        if hint:
            err = f"{err} ({hint})"
        return _ModelProbeResult(
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
            results: list[_ModelProbeResult] = []
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
                # OpenAI embeddings (current + legacy)
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

    results: list[_ModelProbeResult] = []
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
                    results.append(_ModelProbeResult(kind=k, model=model, ok=True))
                except Exception as exc:
                    err = _first_line(str(exc))
                    hint = _probe_hint(kind=k, model=model, error_line=err)
                    if hint:
                        err = f"{err} ({hint})"
                    results.append(
                        _ModelProbeResult(
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
                    results.append(_ModelProbeResult(kind=k, model=model, ok=True))
                except Exception as exc:
                    err = _first_line(str(exc))
                    hint = _probe_hint(kind=k, model=model, error_line=err)
                    if hint:
                        err = f"{err} ({hint})"
                    results.append(
                        _ModelProbeResult(
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
    type=click.Choice(["meta", "summary", "equations", "eq", "tex", "latex", "full"], case_sensitive=False),
    default="meta",
    show_default=True,
    help="What to show (prints to stdout).",
)
def show(papers: tuple[str, ...], level: str):
    """Show paper details or print saved content (summary/equations/LaTeX)."""
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
    else:
        src_name = ""
        missing_msg = ""

    first_output = True
    for paper_ref in papers:
        name, error = _resolve_paper_name_from_ref(paper_ref, index)
        if not name:
            echo_error(error)
            continue

        paper_dir = PAPERS_DIR / name
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

        paper_dir = PAPERS_DIR / name
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

        paper_dir = PAPERS_DIR / name
        if not paper_dir.exists():
            echo_error(f"Paper not found: {paper_ref}")
            failures += 1
            continue

        shutil.rmtree(paper_dir)

        if name in index:
            del index[name]
            save_index(index)

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
    click.echo(PAPER_DB)


@cli.command("install-skill")
@click.option(
    "--claude", "targets", flag_value="claude", multiple=True, help="Install for Claude Code (~/.claude/skills)"
)
@click.option("--codex", "targets", flag_value="codex", multiple=True, help="Install for Codex CLI (~/.codex/skills)")
@click.option("--force", is_flag=True, help="Overwrite existing skill installation")
def install_skill(targets: tuple[str, ...], force: bool):
    """Install the papi skill for Claude Code and/or Codex CLI.

    Creates a symlink from the skill directory to the appropriate location.

    \b
    Examples:
        papi install-skill              # Install for both Claude Code and Codex CLI
        papi install-skill --claude     # Install for Claude Code only
        papi install-skill --codex      # Install for Codex CLI only
        papi install-skill --force      # Overwrite existing installation
    """
    # Find the skill directory relative to this module
    module_dir = Path(__file__).parent
    skill_source = module_dir / "skill"

    if not skill_source.exists():
        echo_error(f"Skill directory not found at {skill_source}")
        echo_error("This may happen if paperpipe was installed without the skill files.")
        raise SystemExit(1)

    # Default to both if no specific target given
    install_targets = set(targets) if targets else {"claude", "codex"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "skills",
        "codex": Path.home() / ".codex" / "skills",
    }

    installed = []
    for target in sorted(install_targets):
        skills_dir = target_dirs[target]
        dest = skills_dir / "papi"

        # Check if already installed
        if dest.exists() or dest.is_symlink():
            if not force:
                if dest.is_symlink() and dest.resolve() == skill_source.resolve():
                    echo(f"{target}: already installed at {dest}")
                    continue
                echo_warning(f"{target}: {dest} already exists (use --force to overwrite)")
                continue
            # Remove existing
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)

        # Create parent directory if needed
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink
        dest.symlink_to(skill_source)
        installed.append((target, dest))
        echo_success(f"{target}: installed at {dest} -> {skill_source}")

    if installed:
        echo()
        echo("Restart your CLI to activate the skill.")


@cli.command("install-prompts")
@click.option(
    "--claude",
    "targets",
    flag_value="claude",
    multiple=True,
    help="Install for Claude Code (~/.claude/commands)",
)
@click.option(
    "--codex",
    "targets",
    flag_value="codex",
    multiple=True,
    help="Install for Codex CLI (~/.codex/prompts)",
)
@click.option("--force", is_flag=True, help="Overwrite existing prompt installation")
@click.option("--copy", is_flag=True, help="Copy files instead of symlinking (useful if symlinks are unavailable).")
def install_prompts(targets: tuple[str, ...], force: bool, copy: bool):
    """Install shared paperpipe prompts for Claude Code and/or Codex CLI.

    These are lightweight "prompt templates" (not the papi skill). By default, this command creates symlinks
    from the packaged `prompts/` directory into the target prompt directories.

    \b
    Examples:
        papi install-prompts             # Install for both Claude Code and Codex CLI
        papi install-prompts --claude    # Install for Claude Code only
        papi install-prompts --codex     # Install for Codex CLI only
        papi install-prompts --force     # Overwrite existing installation
        papi install-prompts --copy      # Copy files (no symlinks)
    """
    module_dir = Path(__file__).parent

    prompt_root = module_dir / "prompts"
    if not prompt_root.exists():
        echo_error(f"Prompts directory not found at {prompt_root}")
        echo_error("This may happen if paperpipe was installed without the prompt files.")
        raise SystemExit(1)

    install_targets = set(targets) if targets else {"claude", "codex"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "commands",
        "codex": Path.home() / ".codex" / "prompts",
    }

    source_dirs = {
        "claude": prompt_root / "claude",
        "codex": prompt_root / "codex",
    }

    installed: list[tuple[str, Path]] = []
    for target in sorted(install_targets):
        prompt_source = source_dirs.get(target, prompt_root)
        if not prompt_source.exists():
            echo_error(f"{target}: prompts directory not found at {prompt_source}")
            raise SystemExit(1)

        prompt_files = sorted([p for p in prompt_source.glob("*.md") if p.is_file()])
        if not prompt_files:
            echo_error(f"{target}: no prompts found in {prompt_source}")
            raise SystemExit(1)

        dest_dir = target_dirs[target]
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src in prompt_files:
            dest = dest_dir / src.name

            if dest.exists() or dest.is_symlink():
                if not force:
                    if dest.is_symlink() and dest.resolve() == src.resolve():
                        echo(f"{target}: already installed: {dest.name}")
                        continue
                    echo_warning(f"{target}: {dest} already exists (use --force to overwrite)")
                    continue
                if dest.is_symlink() or dest.is_file():
                    dest.unlink()
                elif dest.is_dir():
                    shutil.rmtree(dest)

            try:
                if copy:
                    shutil.copy2(src, dest)
                else:
                    dest.symlink_to(src)
            except OSError as e:
                echo_error(f"{target}: failed to install {src.name}: {e}")
                if not copy:
                    echo_error("If your filesystem does not support symlinks, re-run with --copy.")
                raise SystemExit(1)

            installed.append((target, dest))

        mode = "copied" if copy else "linked"
        echo_success(f"{target}: {mode} {len(prompt_files)} prompt(s) into {dest_dir}")

    if installed:
        echo()
        echo("Restart your CLI to pick up new prompts/commands.")


if __name__ == "__main__":
    cli()
