"""Paper add/update/regenerate helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click

from . import config
from .config import (
    _is_ollama_model_id,
    _ollama_reachability_error,
    _prepare_ollama_env,
    default_llm_model,
    default_llm_temperature,
    normalize_tags,
)
from .core import (
    _format_title_short,
    _is_arxiv_id_name,
    _is_safe_paper_name,
    _looks_like_pdf,
    _parse_authors,
    _slugify_title,
    arxiv_base_id,
    categories_to_tags,
    ensure_notes_file,
    load_index,
    normalize_arxiv_id,
    save_index,
)
from .output import debug, echo_error, echo_progress, echo_success, echo_warning
from .search import _maybe_update_search_index


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
    except requests.Timeout:
        echo_warning(f"Timed out downloading source for {arxiv_id}. Try again later.")
        return None
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        status_msg = f"HTTP {status}" if status else "HTTP error"
        echo_warning(f"{status_msg} downloading source for {arxiv_id}: {e}")
        return None
    except requests.RequestException as e:
        echo_warning(f"Could not download source for {arxiv_id}: {e}")
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
                    tex_by_name[member.name] = extracted.read().decode("utf-8", errors="replace")

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
            with gzip.open(tar_path, "rt", encoding="utf-8", errors="replace") as f:
                tex_content = f.read()
        except gzip.BadGzipFile as e:
            echo_warning(f"Source archive for {arxiv_id} is not gzip/tar ({type(e).__name__}). Trying plain text.")
            try:
                tex_content = tar_path.read_text(errors="replace")
            except OSError as read_error:
                echo_warning(f"Could not read source as text for {arxiv_id}: {read_error}")
        except OSError as e:
            echo_warning(f"Failed to extract gzip source for {arxiv_id}: {e}")
            try:
                tex_content = tar_path.read_text(errors="replace")
            except OSError as read_error:
                echo_warning(f"Could not read source as text for {arxiv_id}: {read_error}")

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
    except ImportError as e:
        echo_warning(f"LiteLLM not installed; falling back to non-LLM generation ({e}).")
        debug("LiteLLM import failed:\n%s", traceback.format_exc())
    except Exception as e:
        echo_warning(f"LLM generation failed ({type(e).__name__}: {e}). Falling back to simple extraction.")
        debug("LLM generation failed:\n%s", traceback.format_exc())
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
        paper_dir = config.PAPERS_DIR / name
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

    paper_dir = config.PAPERS_DIR / name

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
        if name in existing_names or (config.PAPERS_DIR / name).exists():
            echo_error(f"Paper '{name}' already exists. Use --name to specify a different name.")
            return False, None
    else:
        candidate = _generate_local_pdf_name({"title": title, "abstract": ""}, use_llm=not no_llm)
        if candidate in existing_names or (config.PAPERS_DIR / candidate).exists():
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
    paper_dir = config.PAPERS_DIR / name
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
    paper_dir = config.PAPERS_DIR / name
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
    paper_dir = config.PAPERS_DIR / name
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
            new_dir = config.PAPERS_DIR / candidate
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
