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
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click


# Configuration
def _paper_db_root() -> Path:
    configured = os.environ.get("PAPER_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".paperpipe"


PAPER_DB = _paper_db_root()
PAPERS_DIR = PAPER_DB / "papers"
INDEX_FILE = PAPER_DB / "index.json"

# LLM defaults (overridable via env vars) - uses LiteLLM model identifiers
DEFAULT_LLM_MODEL = os.environ.get("PAPERPIPE_LLM_MODEL", "gemini/gemini-3-flash-preview")
DEFAULT_EMBEDDING_MODEL = os.environ.get("PAPERPIPE_EMBEDDING_MODEL", "gemini/gemini-embedding-001")

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
_ARXIV_OLD_STYLE_RE = re.compile(
    r"^[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?$", flags=re.IGNORECASE
)
_ARXIV_ANY_RE = re.compile(
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?)",
    flags=re.IGNORECASE,
)


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


def ensure_db():
    """Ensure the paper database directory structure exists."""
    PAPER_DB.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("{}")


def _pillow_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("PIL") is not None


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
    if model == "text-embedding-3-large" and (
        "not supported" in low or "model_not_supported" in low
    ):
        return "not enabled for this OpenAI key/project; use text-embedding-3-small"
    if model.startswith("claude-3-5-sonnet") and ("not_found" in low or "model:" in low):
        return "Claude 3.5 appears retired; try claude-sonnet-4-5"
    if (
        kind == "completion"
        and model.startswith("voyage/")
        and "does not support parameters" in low
    ):
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
    tags = []
    for cat in categories:
        if cat in CATEGORY_TAGS:
            tags.append(CATEGORY_TAGS[cat])
        else:
            # Use the category itself as a tag (e.g., cs.CV -> cs-cv)
            tags.append(cat.lower().replace(".", "-"))
    return list(set(tags))  # Deduplicate


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
        click.echo(f"  Warning: Could not download source: {e}", err=True)
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
            name = re.sub(r"[^a-z0-9-]", "", name)
            if name and len(name) <= 30:
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

    # Fallback to arxiv ID
    if not name:
        name = arxiv_id.replace("/", "_").replace(".", "_")

    # Handle collisions
    base_name = name
    counter = 2
    while name in existing_names:
        name = f"{base_name}-{counter}"
        counter += 1

    return name


def generate_llm_content(
    paper_dir: Path, meta: dict, tex_content: Optional[str]
) -> tuple[str, str, list[str]]:
    """
    Generate summary, equations.md, and semantic tags using LLM.
    Returns (summary, equations_md, additional_tags)
    """
    if not _litellm_available():
        # Fallback: simple extraction without LLM
        summary = generate_simple_summary(meta)
        equations = (
            extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        )
        return summary, equations, []

    try:
        return generate_with_litellm(meta, tex_content)
    except Exception as e:
        click.echo(f"  Warning: LLM generation failed: {e}", err=True)
        summary = generate_simple_summary(meta)
        equations = (
            extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        )
        return summary, equations, []


def generate_simple_summary(meta: dict) -> str:
    """Generate a simple summary from metadata (no LLM)."""
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

    lines.extend(["", "## Abstract", "", abstract, "", "---"])
    regen_target = arxiv_id if arxiv_id else "<paper-name-or-arxiv-id>"
    lines.append(
        "*Summary auto-generated from metadata. Configure an LLM and run "
        f"`papi regenerate {regen_target}` for a richer summary and equation explanations.*"
    )
    lines.append("")
    return "\n".join(lines)


def _litellm_available() -> bool:
    """Check if LiteLLM is available."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def _run_llm(prompt: str, *, purpose: str) -> Optional[str]:
    """Run a prompt through LiteLLM."""
    try:
        import litellm

        litellm.suppress_debug_info = True
    except ImportError:
        click.echo("  LiteLLM not installed. Install with: pip install litellm", err=True)
        return None

    model = DEFAULT_LLM_MODEL
    click.echo(f"  LLM ({model}): generating {purpose}...")

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        out = response.choices[0].message.content  # type: ignore[union-attr]
        if out:
            out = out.strip()
        click.echo(f"  LLM ({model}): {purpose} ok")
        return out or None
    except Exception as e:
        err_msg = str(e).split("\n")[0][:100]
        click.echo(f"  LLM ({model}): {purpose} failed: {err_msg}", err=True)
        return None


def generate_with_litellm(meta: dict, tex_content: Optional[str]) -> tuple[str, str, list[str]]:
    """Generate content using LiteLLM."""

    # Prepare context
    context = f"""Paper: {meta["title"]}
Authors: {", ".join(meta["authors"][:10])}
Abstract: {meta["abstract"]}
"""
    if tex_content:
        # Include first ~8000 chars of tex for context
        context += f"\n\nLaTeX source (truncated):\n{tex_content[:8000]}"

    # Generate summary
    summary_prompt = f"""Based on this paper, create a concise technical summary for a software developer implementing the methods. Focus on:
1. Core contribution (1-2 sentences)
2. Key method/architecture (bullet points)
3. Important implementation details
4. Loss functions or training objectives

Keep it under 500 words. Use markdown formatting.

{context}"""

    try:
        llm_summary = _run_llm(summary_prompt, purpose="summary")
        summary = llm_summary if llm_summary else generate_simple_summary(meta)
    except Exception:
        summary = generate_simple_summary(meta)

    # Generate equations.md
    if tex_content:
        eq_prompt = f"""Extract the key equations from this LaTeX paper. For each equation:
1. Show the LaTeX code
2. Briefly explain what it represents (1 sentence)
3. Note any important variables

Focus on: loss functions, core formulas, key derivations.
Format as markdown with ```latex blocks.

LaTeX content:
{tex_content[:12000]}"""

        try:
            llm_equations = _run_llm(eq_prompt, purpose="equations")
            equations = llm_equations if llm_equations else extract_equations_simple(tex_content)
        except Exception:
            equations = extract_equations_simple(tex_content)
    else:
        equations = "No LaTeX source available for equation extraction."

    # Generate semantic tags
    tag_prompt = f"""Given this paper title and abstract, suggest 3-5 specific technical tags (single words or hyphenated terms) that would help a researcher find this paper.

Focus on: methods, architectures, problem domains, key techniques.
Return ONLY the tags, one per line, lowercase, hyphenated.

Title: {meta["title"]}
Abstract: {meta["abstract"][:1000]}"""

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
@click.version_option(version="0.1.0")
def cli():
    """paperpipe: Unified paper database for coding agents + PaperQA2."""
    ensure_db()


@cli.command()
@click.argument("arxiv_id")
@click.option("--name", "-n", help="Short name for the paper (default: auto-generated from title)")
@click.option("--tags", "-t", help="Additional comma-separated tags")
@click.option("--no-llm", is_flag=True, help="Skip LLM-based generation")
def add(arxiv_id: str, name: Optional[str], tags: Optional[str], no_llm: bool):
    """Add a paper to the database."""
    # Normalize arXiv ID / URL
    try:
        arxiv_id = normalize_arxiv_id(arxiv_id)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    click.echo(f"Adding paper: {arxiv_id}")

    # 1. Fetch metadata (needed for auto-name generation)
    click.echo("  Fetching metadata...")
    try:
        meta = fetch_arxiv_metadata(arxiv_id)
    except Exception as e:
        click.echo(f"Error fetching metadata: {e}", err=True)
        return

    # 2. Generate name from title if not provided
    index = load_index()
    existing_names = set(index.keys())
    if not name:
        name = generate_auto_name(meta, existing_names, use_llm=not no_llm)
        click.echo(f"  Auto-generated name: {name}")

    paper_dir = PAPERS_DIR / name

    if paper_dir.exists():
        click.echo(f"Paper '{name}' already exists. Use --name to specify a different name.")
        return

    if name in existing_names:
        click.echo(f"Paper '{name}' already in index. Use --name to specify a different name.")
        return

    paper_dir.mkdir(parents=True)

    # 3. Download PDF (for PaperQA2)
    click.echo("  Downloading PDF...")
    pdf_path = paper_dir / "paper.pdf"
    try:
        download_pdf(arxiv_id, pdf_path)
    except Exception as e:
        click.echo(f"  Warning: Could not download PDF: {e}", err=True)

    # 4. Download LaTeX source
    click.echo("  Downloading LaTeX source...")
    tex_content = download_source(arxiv_id, paper_dir)
    if tex_content:
        click.echo(f"  Found LaTeX source ({len(tex_content) // 1000}k chars)")
    else:
        click.echo("  No LaTeX source available (PDF-only submission)")

    # 5. Generate tags
    auto_tags = categories_to_tags(meta["categories"])
    user_tags = [t.strip() for t in tags.split(",")] if tags else []

    # 6. Generate summary and equations
    click.echo("  Generating summary and equations...")
    if no_llm:
        summary = generate_simple_summary(meta)
        equations = (
            extract_equations_simple(tex_content) if tex_content else "No LaTeX source available."
        )
        llm_tags = []
    else:
        summary, equations, llm_tags = generate_llm_content(paper_dir, meta, tex_content)

    # Combine all tags
    all_tags = list(set(auto_tags + user_tags + llm_tags))

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

    # 8. Update index
    index = load_index()
    index[name] = {
        "arxiv_id": meta["arxiv_id"],
        "title": meta["title"],
        "tags": all_tags,
        "added": paper_meta["added"],
    }
    save_index(index)

    click.echo(f"\n✓ Added: {name}")
    click.echo(f"  Title: {meta['title'][:60]}...")
    click.echo(f"  Tags: {', '.join(all_tags)}")
    click.echo(f"  Location: {paper_dir}")


@cli.command()
@click.argument("paper_or_arxiv", required=False)
@click.option("--all", "regenerate_all", is_flag=True, help="Regenerate all papers")
@click.option("--no-llm", is_flag=True, help="Skip LLM-based regeneration")
@click.option(
    "--overwrite",
    "-o",
    default=None,
    help="Overwrite fields: 'all' or comma-separated list (summary,equations,tags,name)",
)
def regenerate(
    paper_or_arxiv: Optional[str],
    regenerate_all: bool,
    no_llm: bool,
    overwrite: Optional[str],
):
    """Regenerate summary/equations for an existing paper (by name or arXiv ID).

    By default, only missing fields are generated. Use --overwrite to force regeneration:

    \b
      --overwrite all           Regenerate everything
      --overwrite name          Regenerate name only
      --overwrite tags,summary  Regenerate tags and summary
    """
    index = load_index()

    # Parse overwrite option
    valid_fields = {"all", "summary", "equations", "tags", "name"}
    if overwrite is not None:
        overwrite_fields = {f.strip().lower() for f in overwrite.split(",") if f.strip()}
        invalid = overwrite_fields - valid_fields
        if invalid:
            raise click.UsageError(f"Invalid --overwrite fields: {', '.join(sorted(invalid))}")
        overwrite_all = "all" in overwrite_fields
    else:
        overwrite_fields = set()
        overwrite_all = False

    if regenerate_all and paper_or_arxiv:
        raise click.UsageError("Use either a paper/arXiv id OR `--all`, not both.")

    def resolve_name(target: str) -> Optional[str]:
        if target in index:
            return target
        try:
            normalized = normalize_arxiv_id(target)
        except ValueError:
            normalized = target

        matches = [n for n, info in index.items() if info.get("arxiv_id") == normalized]
        if not matches:
            return None
        if len(matches) > 1:
            click.echo(
                f"Multiple papers match arXiv ID {normalized}: {', '.join(sorted(matches))}",
                err=True,
            )
            return None
        return matches[0]

    def _is_arxiv_id_name(name: str) -> bool:
        """Check if name looks like an arXiv ID (e.g., 1706_03762 or hep-th_9901001)."""
        # New-style: 1706_03762 or 1706_03762v5
        if re.match(r"^\d{4}_\d{4,5}(v\d+)?$", name):
            return True
        # Old-style: hep-th_9901001
        if re.match(r"^[a-z-]+_\d{7}$", name):
            return True
        return False

    def regenerate_one(name: str) -> tuple[bool, Optional[str]]:
        """Regenerate fields for a paper. Returns (success, new_name or None)."""
        paper_dir = PAPERS_DIR / name
        meta_path = paper_dir / "meta.json"
        if not meta_path.exists():
            click.echo(f"Missing metadata for: {name} ({meta_path})", err=True)
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
            # Overwrite everything
            do_summary = True
            do_equations = True
            do_tags = True
            do_name = True
        elif overwrite_fields:
            # User specified specific fields
            do_summary = "summary" in overwrite_fields
            do_equations = "equations" in overwrite_fields
            do_tags = "tags" in overwrite_fields
            do_name = "name" in overwrite_fields
        else:
            # Only fill missing fields
            do_summary = not summary_path.exists() or summary_path.stat().st_size == 0
            do_equations = not equations_path.exists() or equations_path.stat().st_size == 0
            do_tags = not meta.get("tags")
            do_name = _is_arxiv_id_name(name)

        if not (do_summary or do_equations or do_tags or do_name):
            click.echo(f"  {name}: nothing to regenerate")
            return True, None

        actions = []
        if do_summary:
            actions.append("summary")
        if do_equations:
            actions.append("equations")
        if do_tags:
            actions.append("tags")
        if do_name:
            actions.append("name")
        click.echo(f"Regenerating {name}: {', '.join(actions)}")

        new_name: Optional[str] = None
        updated_meta = False

        # Regenerate name if requested
        if do_name:
            existing_names = set(index.keys()) - {name}
            candidate = generate_auto_name(meta, existing_names, use_llm=not no_llm)
            if candidate != name:
                new_dir = PAPERS_DIR / candidate
                if new_dir.exists():
                    click.echo(f"  Warning: Cannot rename to '{candidate}' (already exists)", err=True)
                else:
                    paper_dir.rename(new_dir)
                    paper_dir = new_dir
                    summary_path = paper_dir / "summary.md"
                    equations_path = paper_dir / "equations.md"
                    meta_path = paper_dir / "meta.json"
                    new_name = candidate
                    click.echo(f"  Renamed: {name} → {candidate}")

        # Generate content based on what's needed
        summary: Optional[str] = None
        equations: Optional[str] = None
        llm_tags: list[str] = []

        if do_summary or do_equations or do_tags:
            if no_llm:
                if do_summary:
                    summary = generate_simple_summary(meta)
                if do_equations:
                    equations = (
                        extract_equations_simple(tex_content)
                        if tex_content
                        else "No LaTeX source available."
                    )
            else:
                # LLM generates all three together, but we only use what we need
                llm_summary, llm_equations, llm_tags = generate_llm_content(
                    paper_dir, meta, tex_content
                )
                if do_summary:
                    summary = llm_summary
                if do_equations:
                    equations = llm_equations
                if not do_tags:
                    llm_tags = []  # Discard if not requested

        # Write files
        if summary is not None:
            summary_path.write_text(summary)

        if equations is not None:
            equations_path.write_text(equations)

        if llm_tags:
            meta["tags"] = list(set(meta.get("tags", []) + llm_tags))
            updated_meta = True

        if updated_meta:
            meta_path.write_text(json.dumps(meta, indent=2))

        # Update index
        current_name = new_name if new_name else name
        if new_name:
            # Remove old entry, add new
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

        click.echo("  ✓ Done")
        return True, new_name

    if regenerate_all or (paper_or_arxiv == "all" and "all" not in index):
        names = sorted(index.keys())
        if not names:
            click.echo("No papers found.")
            return

        failures = 0
        renames: list[tuple[str, str]] = []
        for i, name in enumerate(names, 1):
            click.echo(f"[{i}/{len(names)}] {name}")
            success, new_name = regenerate_one(name)
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

    if not paper_or_arxiv:
        raise click.UsageError("Missing PAPER_OR_ARXIV argument (or pass `--all`).")

    name = resolve_name(paper_or_arxiv)
    if not name:
        click.echo(f"Paper not found: {paper_or_arxiv}", err=True)
        return

    success, new_name = regenerate_one(name)
    if new_name:
        click.echo(f"Paper renamed: {name} → {new_name}")


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
        click.echo(f"{name}")
        click.echo(f"  {title}...")
        click.echo(f"  Tags: {tags}")
        click.echo()


@cli.command()
@click.argument("query")
def search(query: str):
    """Search papers by title, tag, or content."""
    index = load_index()
    query_lower = query.lower()

    results = []
    for name, info in index.items():
        score = 0
        # Check title
        if query_lower in info.get("title", "").lower():
            score += 10
        # Check tags
        for tag in info.get("tags", []):
            if query_lower in tag:
                score += 5
        # Check arxiv_id
        if query_lower in info.get("arxiv_id", ""):
            score += 3

        if score > 0:
            results.append((name, info, score))

    results.sort(key=lambda x: -x[2])

    if not results:
        click.echo(f"No papers found matching '{query}'")
        return

    for name, info, score in results[:10]:
        click.echo(f"{name} (score: {score})")
        click.echo(f"  {info.get('title', 'Unknown')[:60]}...")
        click.echo()


@cli.command()
@click.argument("papers", nargs=-1, required=True)
@click.option(
    "--level",
    "-l",
    type=click.Choice(["summary", "equations", "full"]),
    default="summary",
    help="What to export",
)
@click.option("--to", "dest", type=click.Path(), help="Destination directory")
def export(papers: tuple, level: str, dest: Optional[str]):
    """Export paper context for a coding session."""
    dest_path = Path(dest) if dest else Path.cwd() / "paper-context"
    dest_path.mkdir(exist_ok=True)

    for paper in papers:
        paper_dir = PAPERS_DIR / paper
        if not paper_dir.exists():
            click.echo(f"Paper not found: {paper}", err=True)
            continue

        if level == "summary":
            src = paper_dir / "summary.md"
            if src.exists():
                shutil.copy(src, dest_path / f"{paper}_summary.md")
        elif level == "equations":
            src = paper_dir / "equations.md"
            if src.exists():
                shutil.copy(src, dest_path / f"{paper}_equations.md")
        else:  # full
            src = paper_dir / "source.tex"
            if src.exists():
                shutil.copy(src, dest_path / f"{paper}.tex")
            else:
                click.echo(f"  No LaTeX source for {paper}", err=True)

    click.echo(f"Exported {len(papers)} paper(s) to {dest_path}")


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("query")
@click.option("--papers", "-p", help="Limit to specific papers (comma-separated)")
@click.option(
    "--llm",
    default=DEFAULT_LLM_MODEL,
    show_default=True,
    help=(
        "LLM model to use (PaperQA/LiteLLM id; e.g., gpt-5.2, claude-sonnet-4-5, "
        "gemini/gemini-2.5-flash)"
    ),
)
@click.option(
    "--embedding",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Embedding model to use",
)
@click.pass_context
def ask(ctx, query: str, papers: Optional[str], llm: Optional[str], embedding: Optional[str]):
    """
    Query papers using PaperQA2 (if installed).

    Any additional arguments are passed directly to PaperQA2.
    Example: papi ask "query" --summary_llm gpt-4o-mini --temperature 0.5
    """
    if not shutil.which("pqa"):
        click.echo("PaperQA2 not installed. Install with: pip install paper-qa")
        click.echo("\nFalling back to local search...")
        # Do a simple local search instead
        ctx_search = subprocess.run(["papi", "search", query], capture_output=True, text=True)
        click.echo(ctx_search.stdout)
        return

    # Build pqa command
    # pqa [global_options] ask [ask_options] query
    cmd = ["pqa"]
    # PaperQA2 CLI defaults to `--settings high_quality`, which can be overridden by a user's
    # ~/.config/pqa/settings/high_quality.json. If that file is from an older PaperQA version,
    # pqa can crash on startup due to a schema mismatch. Use the special `default` settings
    # (which bypasses JSON config loading) unless the user explicitly passes `--settings/-s`.
    has_settings_flag = any(
        arg in {"--settings", "-s"} or arg.startswith("--settings=") for arg in ctx.args
    )
    if not has_settings_flag:
        cmd.extend(["--settings", "default"])

    # PaperQA2 can attempt PDF image extraction (multimodal parsing). If Pillow isn't installed,
    # PyPDF raises at import-time when accessing `page.images`. Disable multimodal parsing unless
    # the user explicitly provides parsing settings.
    has_parsing_override = any(
        arg == "--parsing" or arg.startswith("--parsing.") or arg.startswith("--parsing=")
        for arg in ctx.args
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
        llm_for_pqa = llm

    if embedding_source != click.core.ParameterSource.DEFAULT:
        embedding_for_pqa = embedding
    elif not has_settings_flag:
        embedding_for_pqa = embedding

    if llm_for_pqa:
        cmd.extend(["--llm", llm_for_pqa])
    if embedding_for_pqa:
        cmd.extend(["--embedding", embedding_for_pqa])

    summary_llm_default = os.environ.get("PAPERPIPE_PQA_SUMMARY_LLM") or llm_for_pqa
    enrichment_llm_default = os.environ.get("PAPERPIPE_PQA_ENRICHMENT_LLM") or llm_for_pqa

    has_summary_llm_override = any(
        arg in {"--summary_llm", "--summary-llm"}
        or arg.startswith(("--summary_llm=", "--summary-llm="))
        for arg in ctx.args
    )
    if summary_llm_default and not has_summary_llm_override:
        cmd.extend(["--summary_llm", summary_llm_default])

    has_enrichment_llm_override = any(
        arg == "--parsing.enrichment_llm"
        or arg == "--parsing.enrichment-llm"
        or arg.startswith(("--parsing.enrichment_llm=", "--parsing.enrichment-llm="))
        for arg in ctx.args
    )
    if enrichment_llm_default and not has_enrichment_llm_override:
        cmd.extend(["--parsing.enrichment_llm", enrichment_llm_default])

    # Add any extra arguments passed after the known options
    cmd.extend(ctx.args)

    cmd.extend(["ask", query])

    # Run PaperQA2 on the papers directory
    if papers:
        # Create temp dir with just the specified papers' PDFs
        with tempfile.TemporaryDirectory() as tmpdir:
            for p in papers.split(","):
                pdf = PAPERS_DIR / p.strip() / "paper.pdf"
                if pdf.exists():
                    shutil.copy(pdf, Path(tmpdir) / f"{p.strip()}.pdf")
            subprocess.run(cmd, cwd=tmpdir)
    else:
        # Create a directory of all PDFs for pqa
        with tempfile.TemporaryDirectory() as tmpdir:
            for paper_dir in PAPERS_DIR.iterdir():
                if paper_dir.is_dir():
                    pdf = paper_dir / "paper.pdf"
                    if pdf.exists():
                        shutil.copy(pdf, Path(tmpdir) / f"{paper_dir.name}.pdf")
            subprocess.run(cmd, cwd=tmpdir)


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
    help=(
        "Model id(s) to probe (LiteLLM ids). If omitted, probes a small curated set "
        "including paperpipe defaults."
    ),
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
        from litellm import completion as llm_completion
        from litellm import embedding as llm_embedding
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
        return False

    def infer_provider(model: str) -> Optional[str]:
        if model.startswith("gemini/"):
            return "gemini"
        if model.startswith("voyage/"):
            return "voyage"
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith("gpt-") or model.startswith("text-embedding-"):
            return "openai"
        return None

    enabled_providers = {
        p for p in ("openai", "gemini", "anthropic", "voyage") if provider_has_key(p)
    }

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
                    click.echo(f"{status:4s}  {r.kind:10s}  {r.model}")
                else:
                    err = r.error or ""
                    err_type = r.error_type or "Error"
                    click.echo(f"{status:4s}  {r.kind:10s}  {r.model}  ({err_type}: {err})")
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
                DEFAULT_LLM_MODEL,
                "gpt-4o",
                "claude-sonnet-4-20250514",
            ]
            embedding_models = [
                DEFAULT_EMBEDDING_MODEL,
                "text-embedding-3-small",
                "voyage/voyage-3-large",
            ]

        # Only probe providers that are configured with an API key.
        completion_models = [
            m
            for m in completion_models
            if (infer_provider(m) is None) or (infer_provider(m) in enabled_providers)
        ]
        embedding_models = [
            m
            for m in embedding_models
            if (infer_provider(m) is None) or (infer_provider(m) in enabled_providers)
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
            click.echo(f"{status:4s}  {r.kind:10s}  {r.model}")
        else:
            err = r.error or ""
            err_type = r.error_type or "Error"
            click.echo(f"{status:4s}  {r.kind:10s}  {r.model}  ({err_type}: {err})")


@cli.command()
@click.argument("paper")
def show(paper: str):
    """Show details of a paper."""
    paper_dir = PAPERS_DIR / paper
    if not paper_dir.exists():
        click.echo(f"Paper not found: {paper}", err=True)
        return

    meta_file = paper_dir / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        click.echo(f"Title: {meta['title']}")
        click.echo(f"arXiv: {meta['arxiv_id']}")
        click.echo(f"Authors: {', '.join(meta['authors'][:5])}")
        click.echo(f"Tags: {', '.join(meta.get('tags', []))}")
        click.echo(f"Has PDF: {meta.get('has_pdf', False)}")
        click.echo(f"Has LaTeX: {meta.get('has_source', False)}")
        click.echo(f"\nFiles: {', '.join(f.name for f in paper_dir.iterdir())}")
        click.echo(f"Location: {paper_dir}")


@cli.command()
@click.argument("paper")
@click.confirmation_option(prompt="Are you sure you want to remove this paper?")
def remove(paper: str):
    """Remove a paper from the database."""
    paper_dir = PAPERS_DIR / paper
    if not paper_dir.exists():
        click.echo(f"Paper not found: {paper}", err=True)
        return

    shutil.rmtree(paper_dir)

    index = load_index()
    if paper in index:
        del index[paper]
        save_index(index)

    click.echo(f"Removed: {paper}")


@cli.command()
def tags():
    """List all tags in the database."""
    index = load_index()
    all_tags = {}

    for info in index.values():
        for tag in info.get("tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
        click.echo(f"{tag}: {count}")


@cli.command()
def path():
    """Print the paper database path."""
    click.echo(PAPER_DB)


if __name__ == "__main__":
    cli()
