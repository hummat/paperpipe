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
import sqlite3
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
from typing import Any, MutableMapping, Optional, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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

DEFAULT_LEANN_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LEANN_EMBEDDING_MODE = "ollama"
DEFAULT_LEANN_LLM_PROVIDER = "ollama"
DEFAULT_LEANN_LLM_MODEL = "olmo-3:7b"


_CONFIG_CACHE: Optional[tuple[Path, Optional[float], dict[str, Any]]] = None
_SEARCH_DB_SCHEMA_VERSION = "1"


def _is_ollama_model_id(model_id: Optional[str]) -> bool:
    return bool(model_id) and model_id.strip().lower().startswith("ollama/")


def _normalize_ollama_base_url(raw: str) -> str:
    base = (raw or "").strip()
    if not base:
        return "http://localhost:11434"
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _prepare_ollama_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    raw = (
        env.get("OLLAMA_API_BASE")
        or env.get("OLLAMA_API_BASE_URL")
        or env.get("OLLAMA_BASE_URL")
        or env.get("OLLAMA_HOST")
    )
    base = _normalize_ollama_base_url(raw or "http://localhost:11434")
    env["OLLAMA_API_BASE"] = base
    env["OLLAMA_HOST"] = base
    return env


def _ollama_reachability_error(*, api_base: str, timeout_sec: float = 1.5) -> Optional[str]:
    api_base = _normalize_ollama_base_url(api_base)
    url = f"{api_base}/api/version"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout_sec) as resp:
            if 200 <= int(getattr(resp, "status", 200)) < 400:
                return None
        return f"Ollama returned non-OK status when probing {url!r}."
    except Exception as e:
        msg = str(e).split("\n")[0][:160]
        return f"Ollama not reachable at {api_base!r} ({type(e).__name__}: {msg})."


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
    configured = os.environ.get("PAPERQA_EMBEDDING")
    if configured and configured.strip():
        return configured.strip()
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


def pqa_index_name_for_embedding(embedding_model: str) -> str:
    """Return the stable PaperQA2 index name used by paperpipe for a given embedding model."""
    safe_name = (embedding_model or "").replace("/", "_").replace(":", "_")
    safe_name = safe_name.strip() or "default"
    return f"paperpipe_{safe_name}"


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


def default_search_mode() -> str:
    """Default `papi search` mode.

    Values:
    - auto: current behavior (use FTS if `search.db` exists; else scan)
    - fts: prefer FTS (still falls back to scan if `search.db` missing)
    - scan: force in-memory scan
    - hybrid: FTS + grep signal (falls back to non-hybrid if prerequisites missing)
    """
    mode = _setting_str(env="PAPERPIPE_SEARCH_MODE", keys=("search", "mode"), default="auto")
    mode = mode.strip().lower()
    if mode in {"auto", "fts", "scan", "hybrid"}:
        return mode
    return "auto"


def default_leann_embedding_model() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_EMBEDDING_MODEL",
        keys=("leann", "embedding_model"),
        default=DEFAULT_LEANN_EMBEDDING_MODEL,
    )


def default_leann_embedding_mode() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_EMBEDDING_MODE",
        keys=("leann", "embedding_mode"),
        default=DEFAULT_LEANN_EMBEDDING_MODE,
    )


def default_leann_llm_provider() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_LLM_PROVIDER",
        keys=("leann", "llm_provider"),
        default=DEFAULT_LEANN_LLM_PROVIDER,
    )


def default_leann_llm_model() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_LLM_MODEL",
        keys=("leann", "llm_model"),
        default=DEFAULT_LEANN_LLM_MODEL,
    )


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

    Note: Uses pickle to match PaperQA2's on-disk index format.
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
    as ERROR so pqa won't crash on them again (unless --pqa-retry-failed is used).

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
    if _is_ollama_model_id(model):
        _prepare_ollama_env(os.environ)
        err = _ollama_reachability_error(api_base=os.environ["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            return None
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
@click.version_option(version="0.5.2")
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
        )
        if not success:
            raise SystemExit(1)
        if paper_name:
            _maybe_update_search_index(name=paper_name)
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

        success, paper_name, action = _add_single_paper(
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

    _maybe_update_search_index(name=current_name, old_name=name if new_name else None)

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
        _search_grep(
            query=query,
            fixed_strings=fixed_strings,
            context_lines=context_lines,
            max_matches=max_matches,
            ignore_case=ignore_case,
            as_json=as_json,
            include_tex=tex,
        )
        return

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
                    "Hybrid search requires `search.db`. Build it first: `papi search-index --rebuild`."
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


@cli.command("search-index")
@click.option("--rebuild", is_flag=True, help="Rebuild the SQLite FTS search index from scratch.")
@click.option(
    "--include-tex/--no-include-tex",
    default=False,
    show_default=True,
    help="Index `source.tex` contents into the FTS index (larger DB; slower build).",
)
def search_index(rebuild: bool, include_tex: bool) -> None:
    """Build/update the local SQLite FTS5 search index (no LLM required)."""
    if not rebuild and include_tex:
        raise click.UsageError("--include-tex only applies with --rebuild")

    db_path = _search_db_path()
    if rebuild or not db_path.exists():
        count = _search_index_rebuild(include_tex=include_tex)
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


def _search_grep(
    *,
    query: str,
    fixed_strings: bool,
    context_lines: int,
    max_matches: int,
    ignore_case: bool,
    as_json: bool,
    include_tex: bool,
) -> None:
    """Search using ripgrep/grep for exact hits + line numbers + context."""
    if context_lines < 0:
        raise click.UsageError("--context must be >= 0")
    if max_matches < 1:
        raise click.UsageError("--max-matches must be >= 1")

    include_globs = ["*/summary.md", "*/equations.md", "*/notes.md", "*/meta.json"]
    if include_tex:
        include_globs.append("*/source.tex")

    if not PAPERS_DIR.exists():
        click.echo("No papers directory found.")
        return

    effective_context_lines = 0 if as_json else context_lines

    rg = shutil.which("rg")
    if rg:
        cmd = [
            rg,
            "--color=never",
            "--no-heading",
            "--with-filename",
            "--line-number",
            "--context",
            str(effective_context_lines),
            "--max-count",
            str(max_matches),
        ]
        if fixed_strings:
            cmd.append("--fixed-strings")
        if ignore_case:
            cmd.append("--ignore-case")
        if as_json:
            cmd.append("--no-context-separator")
        for glob_pat in include_globs:
            cmd.extend(["--glob", glob_pat])
        cmd.append(query)
        cmd.append(str(PAPERS_DIR))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=PAPERS_DIR)
            if as_json:
                click.echo(json.dumps(_parse_grep_matches(out), indent=2))
            else:
                click.echo(out.rstrip("\n"))
            return
        if proc.returncode == 1:
            if as_json:
                click.echo("[]")
            else:
                click.echo(f"No matches for '{query}'")
            return
        raise click.ClickException((proc.stderr or proc.stdout or "").strip() or "ripgrep failed")

    grep = shutil.which("grep")
    if grep:
        cmd = [
            grep,
            "-RIn",
            "--color=never",
            f"-C{effective_context_lines}",
            "--binary-files=without-match",
            "-m",
            str(max_matches),
        ]
        if fixed_strings:
            cmd.append("-F")
        if ignore_case:
            cmd.append("-i")
        for glob_pat in include_globs:
            cmd.append(f"--include={Path(glob_pat).name}")
        cmd.extend([query, str(PAPERS_DIR)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=PAPERS_DIR)
            if as_json:
                click.echo(json.dumps(_parse_grep_matches(out), indent=2))
            else:
                click.echo(out.rstrip("\n"))
            return
        if proc.returncode == 1:
            if as_json:
                click.echo("[]")
            else:
                click.echo(f"No matches for '{query}'")
            return
        raise click.ClickException((proc.stderr or proc.stdout or "").strip() or "grep failed")

    echo_warning("Neither `rg` nor `grep` found; falling back to in-memory scan.")
    ctx = click.get_current_context()
    ctx.invoke(
        search,
        query=query,
        limit=5,
        use_grep=False,
        fixed_strings=False,
        context_lines=2,
        max_matches=200,
        ignore_case=False,
        as_json=False,
        fuzzy=True,
        tex=include_tex,
        use_fts=False,
    )


def _relativize_grep_output(text: str, *, root_dir: Path) -> str:
    root = str(root_dir.resolve())
    prefix = root + os.sep
    out_lines: list[str] = []
    for line in (text or "").splitlines(keepends=True):
        if line.startswith(prefix):
            out_lines.append(line[len(prefix) :])
        else:
            out_lines.append(line)
    return "".join(out_lines)


def _parse_grep_matches(text: str) -> list[dict[str, object]]:
    """Parse grep-style lines like: paper/file:line:... (context lines ignored)."""
    matches: list[dict[str, object]] = []
    for raw in (text or "").splitlines():
        if raw == "--":
            continue
        if ":" not in raw:
            continue
        parts = raw.split(":", 2)
        if len(parts) < 3:
            continue
        path_part, line_part, content = parts
        if not line_part.isdigit():
            continue
        rel_path = path_part.strip()
        paper = rel_path.split("/", 1)[0] if rel_path else ""
        matches.append(
            {
                "paper": paper,
                "path": rel_path,
                "line": int(line_part),
                "text": content,
            }
        )
    return matches


def _collect_grep_matches(
    *,
    query: str,
    fixed_strings: bool,
    max_matches: int,
    ignore_case: bool,
    include_tex: bool,
) -> list[dict[str, object]]:
    include_globs = ["*/summary.md", "*/equations.md", "*/notes.md", "*/meta.json"]
    if include_tex:
        include_globs.append("*/source.tex")

    if not PAPERS_DIR.exists():
        return []

    rg = shutil.which("rg")
    if rg:
        cmd = [
            rg,
            "--color=never",
            "--no-heading",
            "--with-filename",
            "--line-number",
            "--no-context-separator",
            "--context",
            "0",
            "--max-count",
            str(max_matches),
        ]
        if fixed_strings:
            cmd.append("--fixed-strings")
        if ignore_case:
            cmd.append("--ignore-case")
        for glob_pat in include_globs:
            cmd.extend(["--glob", glob_pat])
        cmd.append(query)
        cmd.append(str(PAPERS_DIR))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=PAPERS_DIR)
            return _parse_grep_matches(out)
        if proc.returncode == 1:
            return []
        raise click.ClickException((proc.stderr or proc.stdout or "").strip() or "ripgrep failed")

    grep = shutil.which("grep")
    if grep:
        cmd = [
            grep,
            "-RIn",
            "--color=never",
            "-C0",
            "--binary-files=without-match",
            "-m",
            str(max_matches),
        ]
        if fixed_strings:
            cmd.append("-F")
        if ignore_case:
            cmd.append("-i")
        for glob_pat in include_globs:
            cmd.append(f"--include={Path(glob_pat).name}")
        cmd.extend([query, str(PAPERS_DIR)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=PAPERS_DIR)
            return _parse_grep_matches(out)
        if proc.returncode == 1:
            return []
        raise click.ClickException((proc.stderr or proc.stdout or "").strip() or "grep failed")

    return []


def _search_db_path() -> Path:
    return PAPER_DB / "search.db"


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_test USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_test")
        return True
    except sqlite3.OperationalError:
        return False


def _ensure_search_index_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS search_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM search_meta WHERE key='schema_version'").fetchone()
    if row is not None:
        existing = str(row["value"])
        if existing != _SEARCH_DB_SCHEMA_VERSION:
            raise click.ClickException(
                f"Search index schema version mismatch (have {existing}, need {_SEARCH_DB_SCHEMA_VERSION}). "
                "Run `papi search-index --rebuild` (or delete `search.db`)."
            )
    else:
        conn.execute("INSERT INTO search_meta(key, value) VALUES ('schema_version', ?)", (_SEARCH_DB_SCHEMA_VERSION,))

    if not _sqlite_fts5_available(conn):
        raise click.ClickException("SQLite FTS5 not available in this Python/SQLite build.")

    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
          name UNINDEXED,
          title,
          authors,
          tags,
          abstract,
          summary,
          equations,
          notes,
          tex,
          tokenize='porter'
        )
        """
    )
    conn.commit()


def _set_search_index_meta(conn: sqlite3.Connection, *, include_tex: bool) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO search_meta(key, value) VALUES ('include_tex', ?)",
        ("1" if include_tex else "0",),
    )
    conn.commit()


def _get_search_index_include_tex(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM search_meta WHERE key='include_tex'").fetchone()
    return bool(row and str(row["value"]).strip() == "1")


def _search_index_delete(conn: sqlite3.Connection, *, name: str) -> None:
    conn.execute("DELETE FROM papers_fts WHERE name = ?", (name,))
    conn.commit()


def _search_index_upsert(conn: sqlite3.Connection, *, name: str, index: Optional[dict[str, Any]] = None) -> None:
    paper_dir = PAPERS_DIR / name
    meta_path = paper_dir / "meta.json"
    if not paper_dir.exists() or not meta_path.exists():
        return

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}

    info = (index or {}).get(name, {})

    title = str(meta.get("title") or info.get("title") or "")

    authors_list = meta.get("authors") or []
    if isinstance(authors_list, list):
        authors = " ".join(str(a) for a in authors_list)
    else:
        authors = str(authors_list)

    tags_list = meta.get("tags") or info.get("tags") or []
    if isinstance(tags_list, list):
        tags = " ".join(str(t) for t in tags_list)
    else:
        tags = str(tags_list)

    abstract = str(meta.get("abstract") or "")
    summary = (
        _read_text_limited(paper_dir / "summary.md", max_chars=200_000) if (paper_dir / "summary.md").exists() else ""
    )
    equations = (
        _read_text_limited(paper_dir / "equations.md", max_chars=200_000)
        if (paper_dir / "equations.md").exists()
        else ""
    )
    notes = _read_text_limited(paper_dir / "notes.md", max_chars=200_000) if (paper_dir / "notes.md").exists() else ""

    include_tex = _get_search_index_include_tex(conn)
    tex = ""
    if include_tex and (paper_dir / "source.tex").exists():
        tex = _read_text_limited(paper_dir / "source.tex", max_chars=400_000)

    conn.execute("DELETE FROM papers_fts WHERE name = ?", (name,))
    conn.execute(
        """
        INSERT INTO papers_fts(name, title, authors, tags, abstract, summary, equations, notes, tex)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, title, authors, tags, abstract, summary, equations, notes, tex),
    )
    conn.commit()


def _search_index_rebuild(*, include_tex: bool) -> int:
    db_path = _search_db_path()
    with _sqlite_connect(db_path) as conn:
        _ensure_search_index_schema(conn)
        _set_search_index_meta(conn, include_tex=include_tex)
        conn.execute("DELETE FROM papers_fts")
        idx = load_index()
        count = 0
        for name in sorted(idx.keys()):
            _search_index_upsert(conn, name=name, index=idx)
            count += 1
        return count


def _search_fts(*, query: str, limit: int) -> list[dict[str, object]]:
    db_path = _search_db_path()
    if not db_path.exists():
        return []

    with _sqlite_connect(db_path) as conn:
        _ensure_search_index_schema(conn)

        def run(match_query: str) -> list[sqlite3.Row]:
            return conn.execute(
                """
                SELECT
                  name,
                  title,
                  bm25(papers_fts, 0.0, 10.0, 3.0, 5.0, 2.0, 1.0, 1.0, 0.5, 0.2) AS bm25
                FROM papers_fts
                WHERE papers_fts MATCH ?
                ORDER BY bm25
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()

        try:
            rows = run(query)
        except sqlite3.OperationalError:
            # If the user query contains FTS5 special syntax characters, retry with a quoted literal.
            quoted = _fts5_quote_literal(query)
            try:
                rows = run(quoted)
            except sqlite3.OperationalError as exc:
                raise click.ClickException(
                    f"FTS query failed. Try a simpler query or use `papi search --grep --fixed-strings ...`. ({exc})"
                ) from exc

        results: list[dict[str, object]] = []
        for r in rows:
            raw = float(r["bm25"])
            # SQLite FTS5 bm25() returns "more negative = better". Display a positive score for UX.
            results.append({"name": r["name"], "title": r["title"], "score": -raw})
        return results


def _fts5_quote_literal(query: str) -> str:
    return '"' + (query or "").replace('"', '""') + '"'


def _maybe_update_search_index(*, name: str, old_name: Optional[str] = None) -> None:
    db_path = _search_db_path()
    if not db_path.exists():
        return
    try:
        with _sqlite_connect(db_path) as conn:
            _ensure_search_index_schema(conn)
            if old_name and old_name != name:
                _search_index_delete(conn, name=old_name)
            _search_index_upsert(conn, name=name, index=load_index())
    except Exception as exc:
        debug("Search index update failed for %s: %s", name, str(exc))


def _maybe_delete_from_search_index(*, name: str) -> None:
    db_path = _search_db_path()
    if not db_path.exists():
        return
    try:
        with _sqlite_connect(db_path) as conn:
            _ensure_search_index_schema(conn)
            _search_index_delete(conn, name=name)
    except Exception as exc:
        debug("Search index delete failed for %s: %s", name, str(exc))


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


def _leann_index_meta_path(index_name: str) -> Path:
    return PAPER_DB / ".leann" / "indexes" / index_name / "documents.leann.meta.json"


def _leann_build_index(*, index_name: str, docs_dir: Path, force: bool, extra_args: list[str]) -> None:
    if not shutil.which("leann"):
        echo_error("LEANN not installed. Install with: pip install 'paperpipe[leann]'")
        raise SystemExit(1)

    index_name = (index_name or "").strip()
    if not index_name:
        raise click.UsageError("index name must be non-empty")

    if any(arg == "--file-types" or arg.startswith("--file-types=") for arg in extra_args):
        raise click.UsageError("LEANN indexing in paperpipe is PDF-only; do not pass --file-types.")

    has_embedding_model_override = any(
        arg == "--embedding-model" or arg.startswith("--embedding-model=") for arg in extra_args
    )
    has_embedding_mode_override = any(
        arg == "--embedding-mode" or arg.startswith("--embedding-mode=") for arg in extra_args
    )

    cmd = ["leann", "build", index_name, "--docs", str(docs_dir), "--file-types", ".pdf"]
    if force:
        cmd.append("--force")

    if not has_embedding_model_override:
        cmd.extend(["--embedding-model", default_leann_embedding_model()])
    if not has_embedding_mode_override:
        cmd.extend(["--embedding-mode", default_leann_embedding_mode()])

    cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=PAPER_DB)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _ask_leann(
    *,
    query: str,
    index_name: str,
    provider: Optional[str],
    model: Optional[str],
    host: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
    top_k: Optional[int],
    complexity: Optional[int],
    beam_width: Optional[int],
    prune_ratio: Optional[float],
    recompute_embeddings: bool,
    pruning_strategy: Optional[str],
    thinking_budget: Optional[str],
    interactive: bool,
    extra_args: list[str],
) -> None:
    if not shutil.which("leann"):
        echo_error("LEANN not installed. Install with: pip install 'paperpipe[leann]'")
        raise SystemExit(1)

    provider = (provider or "").strip() or default_leann_llm_provider()
    model = (model or "").strip() or default_leann_llm_model()

    index_name = (index_name or "").strip() or "papers"
    meta_path = _leann_index_meta_path(index_name)
    if not meta_path.exists():
        echo_error(f"LEANN index {index_name!r} not found at {meta_path}")
        echo_error("Build it first: papi leann-index (or: papi index --backend leann)")
        raise SystemExit(1)

    cmd: list[str] = ["leann", "ask", index_name, query]
    cmd.extend(["--llm", provider])
    cmd.extend(["--model", model])
    if host:
        cmd.extend(["--host", host])
    if api_base:
        cmd.extend(["--api-base", api_base])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if interactive:
        cmd.append("--interactive")
    if top_k is not None:
        cmd.extend(["--top-k", str(top_k)])
    if complexity is not None:
        cmd.extend(["--complexity", str(complexity)])
    if beam_width is not None:
        cmd.extend(["--beam-width", str(beam_width)])
    if prune_ratio is not None:
        cmd.extend(["--prune-ratio", str(prune_ratio)])
    if not recompute_embeddings:
        cmd.append("--no-recompute")
    if pruning_strategy:
        cmd.extend(["--pruning-strategy", pruning_strategy])
    if thinking_budget:
        cmd.extend(["--thinking-budget", thinking_budget])

    cmd.extend(extra_args)

    if interactive:
        proc = subprocess.run(cmd, cwd=PAPER_DB)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        return

    proc = subprocess.Popen(cmd, cwd=PAPER_DB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        click.echo(line, nl=False)
    returncode = proc.wait()
    if returncode != 0:
        raise SystemExit(returncode)


@cli.command("leann-index", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--index", "index_name", default="papers", show_default=True, help="LEANN index name.")
@click.option("--force", is_flag=True, help="Force rebuild existing LEANN index.")
@click.pass_context
def leann_index(ctx: click.Context, index_name: str, force: bool) -> None:
    """Build/update a LEANN index over your paper PDFs (PDF-only)."""
    index_name = (index_name or "").strip()
    if not index_name:
        raise click.UsageError("--index must be non-empty")

    PAPER_DB.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    staging_dir = (PAPER_DB / ".pqa_papers").expanduser()
    _refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)
    _leann_build_index(index_name=index_name, docs_dir=staging_dir, force=force, extra_args=list(ctx.args))

    echo_success(f"Built LEANN index {index_name!r} under {PAPER_DB / '.leann' / 'indexes' / index_name}")


@cli.command("index", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "--backend",
    type=click.Choice(["pqa", "leann"], case_sensitive=False),
    default="pqa",
    show_default=True,
    help="Which backend to index for: PaperQA2 (`pqa`) or LEANN (`leann`).",
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
@click.option("--leann-index", default="papers", show_default=True, help="LEANN index name to build.")
@click.option("--leann-force", is_flag=True, help="Force LEANN rebuild (passes --force to `leann build`).")
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
    leann_index: str,
    leann_force: bool,
) -> None:
    """Build/update the retrieval index for PaperQA2 (default) or LEANN."""
    backend_norm = (backend or "pqa").strip().lower()
    if backend_norm == "leann":
        PAPER_DB.mkdir(parents=True, exist_ok=True)
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)

        if any(arg == "--docs" or arg.startswith("--docs=") for arg in ctx.args):
            raise click.UsageError("paperpipe controls LEANN --docs; do not pass --docs.")
        if any(arg.startswith(("--agent.", "--answer.", "--parsing.")) for arg in ctx.args):
            raise click.UsageError("PaperQA2 passthrough args are not supported with --backend leann.")

        staging_dir = (PAPER_DB / ".pqa_papers").expanduser()
        _refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)
        _leann_build_index(index_name=leann_index, docs_dir=staging_dir, force=leann_force, extra_args=list(ctx.args))
        echo_success(f"Built LEANN index {leann_index!r} under {PAPER_DB / '.leann' / 'indexes' / leann_index}")
        return

    if backend_norm != "pqa":
        raise click.UsageError(f"Unknown --backend: {backend}")

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
    if not has_parsing_override and not _pillow_available():
        cmd.extend(["--parsing.multimodal", "OFF"])

    pqa_llm_source = ctx.get_parameter_source("pqa_llm")
    pqa_embedding_source = ctx.get_parameter_source("pqa_embedding")
    pqa_summary_llm_source = ctx.get_parameter_source("pqa_summary_llm")
    pqa_temperature_source = ctx.get_parameter_source("pqa_temperature")
    pqa_verbosity_source = ctx.get_parameter_source("pqa_verbosity")

    llm_for_pqa: Optional[str] = None
    embedding_for_pqa: Optional[str] = None

    if pqa_llm_source != click.core.ParameterSource.DEFAULT:
        llm_for_pqa = pqa_llm
    elif not has_settings_flag:
        llm_for_pqa = default_pqa_llm_model()

    if pqa_embedding_source != click.core.ParameterSource.DEFAULT:
        embedding_for_pqa = pqa_embedding
    elif not has_settings_flag:
        embedding_for_pqa = default_pqa_embedding_model()

    if llm_for_pqa:
        cmd.extend(["--llm", llm_for_pqa])
    if embedding_for_pqa:
        cmd.extend(["--embedding", embedding_for_pqa])

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
    idx_dir_val = _extract_flag_value(
        effective_args, names={"--agent.index.index_directory", "--agent.index.index-directory"}
    )
    idx_name_val = _extract_flag_value(effective_args, names={"--index", "-i", "--agent.index.name"})
    excluded_files: set[str] = set()
    if idx_dir_val and idx_name_val and not pqa_retry_failed:
        fp = _paperqa_index_files_path(index_directory=Path(idx_dir_val), index_name=idx_name_val)
        if fp.exists():
            m = _paperqa_load_index_files_map(fp)
            if m:
                excluded_files = {Path(k).name for k, v in m.items() if v == "ERROR"}

    # Paper directory (defaults to managed staging dir)
    paper_dir_override = _extract_flag_value(
        list(ctx.args),
        names={"--agent.index.paper_directory", "--agent.index.paper-directory"},
    )
    if paper_dir_override:
        paper_dir = Path(paper_dir_override).expanduser()
    else:
        paper_dir = (PAPER_DB / ".pqa_papers").expanduser()
        _refresh_pqa_pdf_staging_dir(staging_dir=paper_dir, exclude_names=excluded_files)
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

    cmd.extend(ctx.args)

    # Clear ERROR markers if requested (PaperQA2 won't retry failed docs otherwise)
    index_dir_raw = _extract_flag_value(
        cmd,
        names={"--agent.index.index_directory", "--agent.index.index-directory"},
    )
    index_name_raw = _extract_flag_value(cmd, names={"--agent.index.name"}) or _extract_flag_value(
        cmd, names={"--index", "-i"}
    )
    if pqa_retry_failed and index_dir_raw and index_name_raw:
        cleared, _ = _paperqa_clear_failed_documents(index_directory=Path(index_dir_raw), index_name=index_name_raw)
        if cleared:
            echo_progress(f"Cleared {cleared} failed PaperQA2 document(s) for retry.")

    cmd.extend(["index", str(paper_dir)])

    env = os.environ.copy()
    if _is_ollama_model_id(llm_for_pqa) or _is_ollama_model_id(embedding_for_pqa):
        _prepare_ollama_env(env)
        err = _ollama_reachability_error(api_base=env["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            raise SystemExit(1)

    proc = subprocess.Popen(
        cmd,
        cwd=PAPERS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        click.echo(line, nl=False)
    returncode = proc.wait()
    if returncode != 0:
        raise SystemExit(returncode)


_PQA_NOISY_STREAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^New file to index:\s+"),
    re.compile(r"^Indexing\b"),
    re.compile(r"^Building\b"),
    re.compile(r"^Loading\b"),
    re.compile(r"^Using settings\b"),
    re.compile(r"^\d{2}:\d{2}:\d{2}\s+\[(DEBUG|INFO|WARNING|ERROR)\]\s+"),
    re.compile(r"^\[(DEBUG|INFO|WARNING|ERROR)\]\s+"),
    re.compile(r"^(DEBUG|INFO|WARNING|ERROR)\s*[:\\-]\s+"),
)


def _pqa_is_noisy_stream_line(line: str) -> bool:
    s = (line or "").rstrip("\n")
    if not s:
        return False
    return any(p.search(s) for p in _PQA_NOISY_STREAM_PATTERNS)


def _paperqa_ask_evidence_blocks(*, cmd: list[str], query: str) -> dict[str, Any]:
    try:
        import paperqa as paperqa_mod
        from paperqa import Settings
    except Exception as e:
        raise click.ClickException(
            "PaperQA2 Python package is required for --format evidence-blocks. "
            "Install with: pip install 'paperpipe[paperqa]'"
        ) from e

    def _bool_flag(names: set[str]) -> bool:
        return any(arg in names or any(arg.startswith(f"{n}=") for n in names) for arg in cmd)

    def _extract(names: set[str]) -> Optional[str]:
        return _extract_flag_value(cmd, names=names)

    def _as_int(s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _as_float(s: Optional[str]) -> Optional[float]:
        if s is None:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    settings_kwargs: dict[str, Any] = {}

    llm = _extract({"--llm"})
    if llm:
        settings_kwargs["llm"] = llm

    embedding = _extract({"--embedding"})
    if embedding:
        settings_kwargs["embedding"] = embedding

    summary_llm = _extract({"--summary_llm"})
    if summary_llm:
        settings_kwargs["summary_llm"] = summary_llm

    temperature = _as_float(_extract({"--temperature"}))
    if temperature is not None:
        settings_kwargs["temperature"] = temperature

    verbosity = _as_int(_extract({"--verbosity"}))
    if verbosity is not None:
        settings_kwargs["verbosity"] = verbosity

    parsing_multimodal = _extract({"--parsing.multimodal"})
    if parsing_multimodal:
        settings_kwargs["parsing"] = {"multimodal": parsing_multimodal}

    answer: dict[str, Any] = {}
    answer_length = _extract({"--answer.answer_length", "--answer.answer-length"})
    if answer_length:
        answer["answer_length"] = answer_length
    evidence_k = _as_int(_extract({"--answer.evidence_k", "--answer.evidence-k"}))
    if evidence_k is not None:
        answer["evidence_k"] = evidence_k
    answer_max_sources = _as_int(_extract({"--answer.answer_max_sources", "--answer.answer-max-sources"}))
    if answer_max_sources is not None:
        answer["answer_max_sources"] = answer_max_sources
    if answer:
        settings_kwargs["answer"] = answer

    agent: dict[str, Any] = {}
    agent_type = _extract({"--agent.agent_type", "--agent.agent-type"})
    if agent_type:
        agent["agent_type"] = agent_type
    timeout = _as_float(_extract({"--agent.timeout"}))
    if timeout is not None:
        agent["timeout"] = timeout
    if _bool_flag({"--agent.rebuild_index", "--agent.rebuild-index"}):
        agent["rebuild_index"] = True

    idx: dict[str, Any] = {}
    paper_directory = _extract({"--agent.index.paper_directory", "--agent.index.paper-directory"})
    if paper_directory:
        idx["paper_directory"] = paper_directory
    index_directory = _extract({"--agent.index.index_directory", "--agent.index.index-directory"})
    if index_directory:
        idx["index_directory"] = index_directory
    index_name = _extract({"--agent.index.name"}) or _extract({"--index", "-i"})
    if index_name:
        idx["name"] = index_name
    sync_with = _extract({"--agent.index.sync_with_paper_directory", "--agent.index.sync-with-paper-directory"})
    if sync_with is not None:
        idx["sync_with_paper_directory"] = (sync_with or "").strip().lower() == "true"
    concurrency = _as_int(_extract({"--agent.index.concurrency"}))
    if concurrency is not None:
        idx["concurrency"] = concurrency
    if idx:
        agent["index"] = idx
    if agent:
        settings_kwargs["agent"] = agent

    settings = Settings(**cast(Any, settings_kwargs))
    response = paperqa_mod.ask(query, settings=settings)

    answer_text: str = getattr(response, "answer", "") or ""
    session = getattr(response, "session", None)
    contexts = getattr(session, "contexts", []) if session is not None else []

    evidence: list[dict[str, Any]] = []
    for ctx in contexts or []:
        text_obj = getattr(ctx, "text", None)
        paper = getattr(text_obj, "name", None) if text_obj is not None else None
        snippet = getattr(ctx, "context", None) or getattr(ctx, "snippet", None) or ""
        pages = (
            getattr(ctx, "pages", None)
            or (getattr(text_obj, "pages", None) if text_obj is not None else None)
            or getattr(ctx, "page", None)
        )
        section = getattr(ctx, "section", None) or (
            getattr(text_obj, "section", None) if text_obj is not None else None
        )

        item: dict[str, Any] = {"paper": paper, "snippet": snippet}
        if pages is not None:
            item["page"] = pages
        if section is not None:
            item["section"] = section
        evidence.append(item)

    return {"backend": "pqa", "question": query, "answer": answer_text, "evidence": evidence}


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
        if output_format_norm != "text":
            raise click.UsageError("--format evidence-blocks is only supported with --backend pqa.")
        if ctx.args:
            raise click.UsageError(
                "Extra passthrough args are only supported for PaperQA2. For LEANN, use the supported --leann-* flags."
            )
        PAPER_DB.mkdir(parents=True, exist_ok=True)
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        if leann_auto_index:
            index_name = (leann_index or "").strip() or "papers"
            meta_path = _leann_index_meta_path(index_name)
            if not meta_path.exists():
                echo_progress(f"LEANN index {index_name!r} not found; building it now...")
                staging_dir = (PAPER_DB / ".pqa_papers").expanduser()
                _refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)
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
        cmd.extend(["--agent.index.name", pqa_index_name_for_embedding(embedding_for_pqa)])

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
    if summary_llm_source != click.core.ParameterSource.DEFAULT:
        # Explicit CLI --pqa-summary-llm takes precedence
        if summary_llm:
            cmd.extend(["--summary_llm", summary_llm])
    else:
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
                "PaperQA2 will not retry them automatically. Re-run with --pqa-retry-failed "
                "or --pqa-rebuild-index to rebuild the whole index."
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

    if output_format_norm == "evidence-blocks":
        if ctx.args:
            raise click.UsageError(
                "--format evidence-blocks does not support extra passthrough args. "
                "Use the first-class --pqa-* options instead."
            )
        click.echo(json.dumps(_paperqa_ask_evidence_blocks(cmd=cmd, query=query), indent=2))
        return

    root_verbose = bool(ctx.find_root().params.get("verbose"))
    raw_output = bool(
        pqa_raw or root_verbose or (verbosity_source != click.core.ParameterSource.DEFAULT and (verbosity or 0) > 0)
    )

    env = os.environ.copy()
    if _is_ollama_model_id(llm_for_pqa) or _is_ollama_model_id(embedding_for_pqa):
        _prepare_ollama_env(env)
        err = _ollama_reachability_error(api_base=env["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            raise SystemExit(1)

    # Run pqa while capturing output for crash detection.
    # We merge stderr into stdout so we can preserve ordering.
    proc = subprocess.Popen(
        cmd,
        cwd=PAPERS_DIR,
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
                echo_warning("  3. Re-stage excluded docs for retry: papi ask '...' --pqa-retry-failed")
                echo_warning("  4. Rebuild index from scratch: papi ask '...' --pqa-rebuild-index")
                if len(failed_docs) <= 5:
                    echo_warning(f"Failed documents: {', '.join(Path(f).stem for f in failed_docs)}")
                raise SystemExit(1)
        # Generic failure message if we can't determine the cause
        echo_error("PaperQA2 failed. Check the output above for details.")
        raise SystemExit(returncode)

    if not raw_output:
        for line in captured_output:
            if not _pqa_is_noisy_stream_line(line):
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
            _prepare_ollama_env(os.environ)
            err = _ollama_reachability_error(api_base=os.environ["OLLAMA_API_BASE"])
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
    click.echo(PAPER_DB)


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
    PAPER_DB.mkdir(parents=True, exist_ok=True)
    if not shutil.which("leann_mcp"):
        echo_error("`leann_mcp` not found on PATH.")
        echo_error("Install with: pip install 'paperpipe[leann]'")
        raise SystemExit(1)

    os.chdir(PAPER_DB)
    os.execvp("leann_mcp", ["leann_mcp"])


def _install_skill(*, targets: tuple[str, ...], force: bool) -> None:
    # Find the skill directory relative to this module
    module_dir = Path(__file__).parent
    skill_source = module_dir / "skill"

    if not skill_source.exists():
        echo_error(f"Skill directory not found at {skill_source}")
        echo_error("This may happen if paperpipe was installed without the skill files.")
        raise SystemExit(1)

    # Default to all if no specific target given
    install_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "skills",
        "codex": Path.home() / ".codex" / "skills",
        "gemini": Path.home() / ".gemini" / "skills",
    }

    installed = []
    for target in sorted(install_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown install target: {target}")
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

        if target == "gemini":
            settings_path = Path.home() / ".gemini" / "settings.json"
            enabled = False
            if settings_path.exists():
                try:
                    obj = json.loads(settings_path.read_text())
                    experimental = obj.get("experimental")
                    enabled = isinstance(experimental, dict) and experimental.get("skills") is True
                except Exception:
                    enabled = False
            if not enabled:
                echo_warning("gemini: skills are experimental; enable them in ~/.gemini/settings.json:")
                echo('  {"experimental": {"skills": true}}')

    if installed:
        echo()
        echo("Restart your CLI to activate the skill.")


def _install_prompts(*, targets: tuple[str, ...], force: bool, copy: bool) -> None:
    module_dir = Path(__file__).parent

    prompt_root = module_dir / "prompts"
    if not prompt_root.exists():
        echo_error(f"Prompts directory not found at {prompt_root}")
        echo_error("This may happen if paperpipe was installed without the prompt files.")
        raise SystemExit(1)

    install_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "commands",
        "codex": Path.home() / ".codex" / "prompts",
        "gemini": Path.home() / ".gemini" / "commands",
    }

    source_dirs = {
        "claude": prompt_root / "claude",
        "codex": prompt_root / "codex",
        "gemini": prompt_root / "gemini",
    }

    installed: list[tuple[str, Path]] = []
    for target in sorted(install_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown install target: {target}")
        prompt_source = source_dirs.get(target, prompt_root)
        if not prompt_source.exists():
            echo_error(f"{target}: prompts directory not found at {prompt_source}")
            raise SystemExit(1)

        suffix = ".toml" if target == "gemini" else ".md"
        prompt_files = sorted([p for p in prompt_source.glob(f"*{suffix}") if p.is_file()])
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


def _install_mcp(
    *, targets: tuple[str, ...], name: str, leann_name: str, embedding: Optional[str], force: bool
) -> None:
    @dataclass(frozen=True)
    class McpServerSpec:
        name: str
        command: str
        args: tuple[str, ...]
        env: dict[str, str]
        description: str

    def _paperqa_mcp_is_available() -> bool:
        if sys.version_info < (3, 11):
            return False
        import importlib.util

        return (
            importlib.util.find_spec("mcp.server.fastmcp") is not None
            and importlib.util.find_spec("paperqa") is not None
        )

    def _leann_mcp_is_available() -> bool:
        return shutil.which("leann_mcp") is not None

    paperqa_name = (name or "").strip()
    leann_server_name = (leann_name or "").strip()

    embedding_model = (embedding or "").strip() if embedding else default_pqa_embedding_model()

    servers: list[McpServerSpec] = []
    if _paperqa_mcp_is_available():
        if not paperqa_name:
            raise click.UsageError("--name must be non-empty")
        servers.append(
            McpServerSpec(
                name=paperqa_name,
                command="papi",
                args=("mcp-server",),
                env={"PAPERQA_EMBEDDING": embedding_model},
                description="PaperQA2 retrieval-only search",
            )
        )

    if _leann_mcp_is_available():
        if not leann_server_name:
            raise click.UsageError("--leann-name must be non-empty")
        servers.append(
            McpServerSpec(
                name=leann_server_name,
                command="papi",
                args=("leann-mcp-server",),
                env={},
                description="LEANN semantic search",
            )
        )

    if not servers:
        echo_error("No MCP servers available to install in this environment.")
        echo_error("Install one of:")
        echo_error("  pip install 'paperpipe[mcp]'    # PaperQA2 MCP server (Python 3.11+)")
        echo_error("  pip install 'paperpipe[leann]'  # LEANN MCP server (compiled)")
        raise SystemExit(1)

    install_targets = set(targets) if targets else {"claude", "codex", "gemini"}
    successes: list[str] = []

    def _read_json_object(path: Path, *, where: str) -> Optional[dict[str, Any]]:
        if path.exists():
            try:
                obj = json.loads(path.read_text())
            except Exception:
                echo_error(f"{where}: failed to parse JSON at {path}")
                return None
            if not isinstance(obj, dict):
                echo_error(f"{where}: expected a JSON object at {path}")
                return None
            return obj
        return {}

    def _write_json_object(path: Path, obj: dict[str, Any], *, where: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2) + "\n")
        except OSError as e:
            echo_error(f"{where}: failed to write {path}: {e}")
            return False
        return True

    def upsert_mcp_servers(path: Path, *, where: str, server_key: str, entry: dict[str, Any]) -> str:
        obj = _read_json_object(path, where=where)
        if obj is None:
            return "error"

        mcp_servers = obj.get("mcpServers")
        if mcp_servers is None:
            mcp_servers = {}
            obj["mcpServers"] = mcp_servers
        if not isinstance(mcp_servers, dict):
            echo_error(f"{where}: expected 'mcpServers' to be an object in {path}")
            return "error"

        existing = mcp_servers.get(server_key)
        if existing is not None and existing != entry and not force:
            echo_warning(f"{where}: {server_key!r} already configured in {path} (use --force to overwrite)")
            return "skipped"
        if existing is not None and existing == entry:
            return "unchanged"

        mcp_servers[server_key] = entry
        return "written" if _write_json_object(path, obj, where=where) else "error"

    for target in sorted(install_targets):
        if target == "claude":
            if not shutil.which("claude"):
                echo_warning("claude: `claude` not found on PATH; install project config instead:")
                echo("  papi install mcp --repo")
                continue

            for spec in servers:
                if force:
                    subprocess.run(["claude", "mcp", "remove", spec.name], capture_output=True, text=True)

                cmd = ["claude", "mcp", "add", "--transport", "stdio"]
                for k, v in spec.env.items():
                    cmd.extend(["--env", f"{k}={v}"])
                cmd.extend(["--scope", "user", spec.name, "--", spec.command, *spec.args])

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    echo_warning(f"claude: failed to install {spec.name!r} via `claude mcp add`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    echo_warning("You can install a project-scoped config file instead:")
                    echo("  papi install mcp --repo")
                    continue

                echo_success(f"claude: installed {spec.name!r}")
                successes.append(f"claude:{spec.name}")
            continue

        if target == "repo":
            claude_dest = Path.cwd() / ".mcp.json"
            gemini_dest = Path.cwd() / ".gemini" / "settings.json"
            for spec in servers:
                entry: dict[str, Any] = {"command": spec.command, "args": list(spec.args), "env": dict(spec.env)}
                claude_status = upsert_mcp_servers(claude_dest, where="repo/claude", server_key=spec.name, entry=entry)
                if claude_status == "written":
                    echo_success(f"repo: wrote {claude_dest} ({spec.name!r})")
                    successes.append(f"repo/claude:{spec.name}")
                elif claude_status == "unchanged":
                    echo(f"repo: already configured {spec.name!r} in {claude_dest}")
                    successes.append(f"repo/claude:{spec.name}")
                elif claude_status == "skipped":
                    successes.append(f"repo/claude:{spec.name}")

                gemini_status = upsert_mcp_servers(gemini_dest, where="repo/gemini", server_key=spec.name, entry=entry)
                if gemini_status == "written":
                    echo_success(f"repo: wrote {gemini_dest} ({spec.name!r})")
                    successes.append(f"repo/gemini:{spec.name}")
                elif gemini_status == "unchanged":
                    echo(f"repo: already configured {spec.name!r} in {gemini_dest}")
                    successes.append(f"repo/gemini:{spec.name}")
                elif gemini_status == "skipped":
                    successes.append(f"repo/gemini:{spec.name}")
            continue

        if target == "codex":
            if not shutil.which("codex"):
                echo_warning("codex: `codex` not found on PATH; run this manually:")
                for spec in servers:
                    env_flags = " ".join([f"--env {k}={v}" for k, v in spec.env.items()])
                    env_flags = f" {env_flags}" if env_flags else ""
                    echo(f"  codex mcp add {spec.name}{env_flags} -- {spec.command}")
                continue

            # Let Codex manage its own config. Prefer replacing only when requested.
            for spec in servers:
                if force:
                    subprocess.run(["codex", "mcp", "remove", spec.name], capture_output=True, text=True)

                cmd = ["codex", "mcp", "add", spec.name]
                for k, v in spec.env.items():
                    cmd.extend(["--env", f"{k}={v}"])
                cmd.extend(["--", spec.command, *spec.args])

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    echo_warning(f"codex: failed to install {spec.name!r} via `codex mcp add`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    echo("Try re-running with --force, or run manually:")
                    env_flags = " ".join([f"--env {k}={v}" for k, v in spec.env.items()])
                    env_flags = f" {env_flags}" if env_flags else ""
                    echo(f"  codex mcp add {spec.name}{env_flags} -- {spec.command}")
                    continue

                echo_success(f"codex: installed {spec.name!r}")
                successes.append(f"codex:{spec.name}")
            continue

        if target == "gemini":
            install_via_cli_ok = False
            if shutil.which("gemini"):
                install_via_cli_ok = True
                for spec in servers:
                    if force:
                        subprocess.run(
                            ["gemini", "mcp", "remove", "--scope", "user", spec.name],
                            capture_output=True,
                            text=True,
                        )

                    cmd = [
                        "gemini",
                        "mcp",
                        "add",
                        "--scope",
                        "user",
                        "--transport",
                        "stdio",
                    ]
                    for k, v in spec.env.items():
                        cmd.extend(["--env", f"{k}={v}"])
                    cmd.extend(["--description", spec.description, spec.name, spec.command, *spec.args])

                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        echo_success(f"gemini: installed {spec.name!r}")
                        successes.append(f"gemini:{spec.name}")
                        continue

                    install_via_cli_ok = False
                    echo_warning(
                        f"gemini: failed to install {spec.name!r} via `gemini mcp add`; "
                        "falling back to ~/.gemini/settings.json"
                    )
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)

            if not install_via_cli_ok:
                dest = Path.home() / ".gemini" / "settings.json"
                for spec in servers:
                    gemini_entry: dict[str, Any] = {
                        "command": spec.command,
                        "args": list(spec.args),
                        "env": dict(spec.env),
                    }
                    status = upsert_mcp_servers(dest, where="gemini", server_key=spec.name, entry=gemini_entry)
                    if status == "written":
                        echo_success(f"gemini: configured {spec.name!r} in {dest}")
                        successes.append(f"gemini:{spec.name}")
                    elif status == "unchanged":
                        echo(f"gemini: already configured {spec.name!r} in {dest}")
                        successes.append(f"gemini:{spec.name}")
                    elif status == "skipped":
                        successes.append(f"gemini:{spec.name}")
            continue

        raise click.UsageError(f"Unknown target: {target}")

    if not successes:
        raise SystemExit(1)

    echo()
    echo("Restart your CLI to pick up the new MCP server.")


def _parse_components(args: tuple[str, ...]) -> list[str]:
    raw: list[str] = []
    for item in args:
        for part in item.split(","):
            part = part.strip().lower()
            if part:
                raw.append(part)
    return raw


def _uninstall_skill(*, targets: tuple[str, ...], force: bool) -> None:
    module_dir = Path(__file__).parent
    skill_source = module_dir / "skill"

    if not skill_source.exists() and not force:
        echo_error(f"Skill directory not found at {skill_source}")
        echo_error("Re-run with --force to remove install locations without validating the source.")
        raise SystemExit(1)

    uninstall_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "skills",
        "codex": Path.home() / ".codex" / "skills",
        "gemini": Path.home() / ".gemini" / "skills",
    }

    removed = 0
    skipped = 0
    for target in sorted(uninstall_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown uninstall target: {target}")
        skills_dir = target_dirs[target]
        dest = skills_dir / "papi"

        if not dest.exists() and not dest.is_symlink():
            echo(f"{target}: not installed")
            continue

        if dest.is_symlink() and skill_source.exists() and dest.resolve() == skill_source.resolve():
            dest.unlink()
            echo_success(f"{target}: removed {dest}")
            removed += 1
            continue

        if force:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink(missing_ok=True)
            echo_success(f"{target}: removed {dest}")
            removed += 1
            continue

        echo_warning(f"{target}: {dest} exists but does not point to this install (use --force to remove)")
        skipped += 1

    if removed:
        echo()
        echo("Restart your CLI to unload the skill.")
    if skipped:
        raise SystemExit(1)


def _uninstall_prompts(*, targets: tuple[str, ...], force: bool) -> None:
    module_dir = Path(__file__).parent
    prompt_root = module_dir / "prompts"
    if not prompt_root.exists():
        echo_error(f"Prompts directory not found at {prompt_root}")
        echo_error("This may happen if paperpipe was installed without the prompt files.")
        raise SystemExit(1)

    uninstall_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "commands",
        "codex": Path.home() / ".codex" / "prompts",
        "gemini": Path.home() / ".gemini" / "commands",
    }

    source_dirs = {
        "claude": prompt_root / "claude",
        "codex": prompt_root / "codex",
        "gemini": prompt_root / "gemini",
    }

    removed = 0
    skipped = 0
    for target in sorted(uninstall_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown uninstall target: {target}")

        dest_dir = target_dirs[target]
        if not dest_dir.exists():
            echo(f"{target}: no prompt directory at {dest_dir}")
            continue

        suffix = ".toml" if target == "gemini" else ".md"
        prompt_source = source_dirs.get(target, prompt_root)
        if not prompt_source.exists():
            echo_error(f"{target}: prompts directory not found at {prompt_source}")
            raise SystemExit(1)

        source_files = sorted([p for p in prompt_source.glob(f"*{suffix}") if p.is_file()])
        for src in source_files:
            dest = dest_dir / src.name
            if not dest.exists() and not dest.is_symlink():
                continue

            if dest.is_symlink():
                if dest.resolve() == src.resolve() or force:
                    dest.unlink()
                    removed += 1
                else:
                    echo_warning(f"{target}: {dest} points elsewhere (use --force to remove)")
                    skipped += 1
                continue

            # Copied file case: remove only if identical unless forced.
            if dest.is_file():
                if force:
                    dest.unlink()
                    removed += 1
                    continue
                try:
                    if dest.read_bytes() == src.read_bytes():
                        dest.unlink()
                        removed += 1
                    else:
                        echo_warning(f"{target}: {dest} differs from packaged prompt (use --force to remove)")
                        skipped += 1
                except OSError as e:
                    echo_warning(f"{target}: failed to read {dest}: {e}")
                    skipped += 1
                continue

            if dest.is_dir():
                if force:
                    shutil.rmtree(dest)
                    removed += 1
                else:
                    echo_warning(f"{target}: {dest} is a directory (use --force to remove)")
                    skipped += 1

        if source_files:
            echo_success(f"{target}: removed prompts from {dest_dir}")

    if removed:
        echo()
        echo("Restart your CLI to unload prompts/commands.")
    if skipped:
        raise SystemExit(1)


def _uninstall_mcp(*, targets: tuple[str, ...], name: str, leann_name: str, force: bool) -> None:
    paperqa_name = (name or "").strip()
    leann_server_name = (leann_name or "").strip()
    if not paperqa_name:
        raise click.UsageError("--name must be non-empty")
    if not leann_server_name:
        raise click.UsageError("--leann-name must be non-empty")

    server_keys = [paperqa_name, leann_server_name]

    uninstall_targets = set(targets) if targets else {"claude", "codex", "gemini"}
    successes: list[str] = []
    failures: list[str] = []

    def _read_json_object(path: Path, *, where: str) -> Optional[dict[str, Any]]:
        if path.exists():
            try:
                obj = json.loads(path.read_text())
            except Exception:
                echo_error(f"{where}: failed to parse JSON at {path}")
                return None
            if not isinstance(obj, dict):
                echo_error(f"{where}: expected a JSON object at {path}")
                return None
            return obj
        return {}

    def _write_json_object(path: Path, obj: dict[str, Any], *, where: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2) + "\n")
        except OSError as e:
            echo_error(f"{where}: failed to write {path}: {e}")
            return False
        return True

    def remove_mcp_servers(path: Path, *, where: str, keys: list[str]) -> bool:
        obj = _read_json_object(path, where=where)
        if obj is None:
            return False

        mcp_servers = obj.get("mcpServers")
        if mcp_servers is None:
            return True
        if not isinstance(mcp_servers, dict):
            echo_error(f"{where}: expected 'mcpServers' to be an object in {path}")
            return False

        changed = False
        for key in keys:
            if key in mcp_servers:
                del mcp_servers[key]
                changed = True

        if not changed:
            return True
        return _write_json_object(path, obj, where=where)

    for target in sorted(uninstall_targets):
        if target == "claude":
            if not shutil.which("claude"):
                echo_warning("claude: `claude` not found on PATH; remove manually or use repo-local config:")
                echo("  papi uninstall mcp --repo")
                continue

            for key in server_keys:
                proc = subprocess.run(["claude", "mcp", "remove", key], capture_output=True, text=True)
                if proc.returncode != 0 and not force:
                    echo_warning(f"claude: failed to remove {key!r} via `claude mcp remove`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    failures.append(f"claude:{key}")
                    continue
                echo_success(f"claude: removed {key!r}")
                successes.append(f"claude:{key}")
            continue

        if target == "repo":
            claude_dest = Path.cwd() / ".mcp.json"
            gemini_dest = Path.cwd() / ".gemini" / "settings.json"
            if remove_mcp_servers(claude_dest, where="repo/claude", keys=server_keys):
                echo_success(f"repo: updated {claude_dest}")
                successes.extend([f"repo/claude:{k}" for k in server_keys])
            else:
                failures.extend([f"repo/claude:{k}" for k in server_keys])
            if remove_mcp_servers(gemini_dest, where="repo/gemini", keys=server_keys):
                echo_success(f"repo: updated {gemini_dest}")
                successes.extend([f"repo/gemini:{k}" for k in server_keys])
            else:
                failures.extend([f"repo/gemini:{k}" for k in server_keys])
            continue

        if target == "codex":
            if not shutil.which("codex"):
                echo_warning("codex: `codex` not found on PATH; run this manually:")
                for key in server_keys:
                    echo(f"  codex mcp remove {key}")
                continue

            for key in server_keys:
                proc = subprocess.run(["codex", "mcp", "remove", key], capture_output=True, text=True)
                if proc.returncode != 0 and not force:
                    echo_warning(f"codex: failed to remove {key!r} via `codex mcp remove`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    failures.append(f"codex:{key}")
                    continue
                echo_success(f"codex: removed {key!r}")
                successes.append(f"codex:{key}")
            continue

        if target == "gemini":
            if shutil.which("gemini"):
                for key in server_keys:
                    proc = subprocess.run(
                        ["gemini", "mcp", "remove", "--scope", "user", key],
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0 and not force:
                        echo_warning(f"gemini: failed to remove {key!r} via `gemini mcp remove`")
                        if proc.stdout.strip():
                            echo(proc.stdout.rstrip("\n"))
                        if proc.stderr.strip():
                            echo(proc.stderr.rstrip("\n"), err=True)
                        failures.append(f"gemini:{key}")
                        continue
                    echo_success(f"gemini: removed {key!r}")
                    successes.append(f"gemini:{key}")

            dest = Path.home() / ".gemini" / "settings.json"
            if remove_mcp_servers(dest, where="gemini", keys=server_keys):
                echo_success(f"gemini: updated {dest}")
                successes.extend([f"gemini:{k}" for k in server_keys])
            else:
                failures.extend([f"gemini:{k}" for k in server_keys])
            continue

        raise click.UsageError(f"Unknown target: {target}")

    if failures and not force:
        raise SystemExit(1)
    if successes:
        echo()
        echo("Restart your CLI to unload the MCP server config.")


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


if __name__ == "__main__":
    cli()
