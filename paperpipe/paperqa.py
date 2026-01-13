"""PaperQA2 helpers and integration utilities."""

from __future__ import annotations

import os
import pickle
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import click

from . import config
from .output import debug


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

    if config.PAPERS_DIR.exists():
        for paper_dir in config.PAPERS_DIR.iterdir():
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


_PQA_NOISY_STREAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^New file to index:\s+"),
    re.compile(r"^Indexing\b"),
    re.compile(r"^Building\b"),
    re.compile(r"^Loading\b"),
    re.compile(r"^Using settings\b"),
    re.compile(r"^Cannot add callback - would exceed MAX_CALLBACKS limit of\b"),
    re.compile(r"^/.*pydantic/main\.py:\d+:\s+UserWarning:\s+Pydantic serializer warnings:\s*$"),
    re.compile(r"^\s+PydanticSerializationUnexpectedValue\("),
    re.compile(r"^\s+return self\.__pydantic_serializer__\.to_python\("),
    re.compile(r"^\d{2}:\d{2}:\d{2}\s+\[(DEBUG|INFO|WARNING|ERROR)\]\s+"),
    re.compile(r"^\[(DEBUG|INFO|WARNING|ERROR)\]\s+"),
    re.compile(r"^(DEBUG|INFO|WARNING|ERROR)\s*[:\\-]\s+"),
)


_PQA_INDEX_NOISY_STREAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Cannot add callback - would exceed MAX_CALLBACKS limit of\b"),
    re.compile(r"^/.*pydantic/main\.py:\d+:\s+UserWarning:\s+Pydantic serializer warnings:\s*$"),
    re.compile(r"^\s+PydanticSerializationUnexpectedValue\("),
    re.compile(r"^\s+return self\.__pydantic_serializer__\.to_python\("),
)


def _pqa_is_noisy_stream_line(line: str) -> bool:
    s = (line or "").rstrip("\n")
    if not s:
        return False
    return any(p.search(s) for p in _PQA_NOISY_STREAM_PATTERNS)


def _pqa_is_noisy_index_line(line: str) -> bool:
    s = (line or "").rstrip("\n")
    if not s:
        return False
    return any(p.search(s) for p in _PQA_INDEX_NOISY_STREAM_PATTERNS)


def _pqa_has_flag(args: Sequence[str], *, names: set[str]) -> bool:
    for arg in args:
        if arg in names:
            return True
        for n in names:
            if arg.startswith(f"{n}="):
                return True
    return False


def _pqa_print_filtered_output_on_failure(
    captured_output: list[str],
    *,
    max_lines: int = 200,
) -> None:
    filtered = [line for line in captured_output if not _pqa_is_noisy_stream_line(line)]
    if not filtered:
        tail = captured_output[-max_lines:] if max_lines > 0 else captured_output
        if not tail:
            return
        click.echo(f"[paperpipe] Showing last {len(tail)} raw PaperQA2 output lines:", err=False)
        for line in tail:
            click.echo(line, nl=False, err=False)
        return
    tail = filtered[-max_lines:] if max_lines > 0 else filtered
    if len(filtered) > len(tail):
        click.echo(f"[paperpipe] Showing last {len(tail)} filtered PaperQA2 output lines:", err=False)
    for line in tail:
        click.echo(line, nl=False, err=False)


def _pqa_print_filtered_index_output_on_failure(
    captured_output: list[str],
    *,
    max_lines: int = 200,
) -> None:
    filtered = [line for line in captured_output if not _pqa_is_noisy_index_line(line)]
    if not filtered:
        tail = captured_output[-max_lines:] if max_lines > 0 else captured_output
        if not tail:
            return
        click.echo(f"[paperpipe] Showing last {len(tail)} raw PaperQA2 output lines:", err=False)
        for line in tail:
            click.echo(line, nl=False, err=False)
        return
    tail = filtered[-max_lines:] if max_lines > 0 else filtered
    if len(filtered) > len(tail):
        click.echo(f"[paperpipe] Showing last {len(tail)} filtered PaperQA2 output lines:", err=False)
    for line in tail:
        click.echo(line, nl=False, err=False)


def _paperqa_ask_evidence_blocks(*, cmd: list[str], query: str) -> dict[str, Any]:
    try:
        import importlib

        paperqa_mod = importlib.import_module("paperqa")
        Settings = getattr(paperqa_mod, "Settings", None)
        ask = getattr(paperqa_mod, "ask", None)
        if Settings is None or ask is None:
            raise AttributeError("paperqa.Settings or paperqa.ask not available")
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
    response = ask(query, settings=settings)

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
