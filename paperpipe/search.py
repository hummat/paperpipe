"""Search helpers, FTS5 indexing, and audit utilities."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator, Optional

import click

from . import config
from .core import _read_text_limited, load_index
from .output import debug, echo_warning


def _search_grep(
    *,
    query: str,
    fixed_strings: bool,
    context_lines: int,
    max_matches: int,
    ignore_case: bool,
    as_json: bool,
    include_tex: bool,
) -> bool:
    """Search using ripgrep/grep for exact hits + line numbers + context."""
    if context_lines < 0:
        raise click.UsageError("--context must be >= 0")
    if max_matches < 1:
        raise click.UsageError("--max-matches must be >= 1")

    include_globs = ["**/summary.md", "**/equations.md", "**/notes.md", "**/meta.json"]
    if include_tex:
        include_globs.append("**/source.tex")

    if not config.PAPERS_DIR.exists():
        click.echo("No papers directory found.")
        return True

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
        cmd.append(str(config.PAPERS_DIR))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=config.PAPERS_DIR)
            if as_json:
                click.echo(json.dumps(_parse_grep_matches(out), indent=2))
            else:
                click.echo(out.rstrip("\n"))
            return True
        if proc.returncode == 1:
            if as_json:
                click.echo("[]")
            else:
                click.echo(f"No matches for '{query}'")
            return True
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
        cmd.extend([query, str(config.PAPERS_DIR)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=config.PAPERS_DIR)
            if as_json:
                click.echo(json.dumps(_parse_grep_matches(out), indent=2))
            else:
                click.echo(out.rstrip("\n"))
            return True
        if proc.returncode == 1:
            if as_json:
                click.echo("[]")
            else:
                click.echo(f"No matches for '{query}'")
            return True
        raise click.ClickException((proc.stderr or proc.stdout or "").strip() or "grep failed")

    echo_warning("Neither `rg` nor `grep` found; falling back to in-memory scan.")
    return False


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
    include_globs = ["**/summary.md", "**/equations.md", "**/notes.md", "**/meta.json"]
    if include_tex:
        include_globs.append("**/source.tex")

    if not config.PAPERS_DIR.exists():
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
        cmd.append(str(config.PAPERS_DIR))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=config.PAPERS_DIR)
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
        cmd.extend([query, str(config.PAPERS_DIR)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = _relativize_grep_output(proc.stdout, root_dir=config.PAPERS_DIR)
            return _parse_grep_matches(out)
        if proc.returncode == 1:
            return []
        raise click.ClickException((proc.stderr or proc.stdout or "").strip() or "grep failed")

    return []


def _search_db_path() -> Path:
    return config.PAPER_DB / "search.db"


@contextmanager
def _sqlite_connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
        if existing != config._SEARCH_DB_SCHEMA_VERSION:
            raise click.ClickException(
                f"Search index schema version mismatch (have {existing}, need {config._SEARCH_DB_SCHEMA_VERSION}). "
                "Run `papi index --backend search --search-rebuild` (or delete `search.db`)."
            )
    else:
        conn.execute(
            "INSERT INTO search_meta(key, value) VALUES ('schema_version', ?)", (config._SEARCH_DB_SCHEMA_VERSION,)
        )

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
    paper_dir = config.PAPERS_DIR / name
    meta_path = paper_dir / "meta.json"
    if not paper_dir.exists() or not meta_path.exists():
        return

    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        echo_warning(f"Failed to parse metadata for {name}: {exc}. Search results may be incomplete.")
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
        echo_warning(
            f"Search index update failed for {name}: {exc}. Rebuild with `papi index --backend search --search-rebuild`."
        )


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
        echo_warning(
            f"Search index delete failed for {name}: {exc}. Rebuild with `papi index --backend search --search-rebuild`."
        )


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
