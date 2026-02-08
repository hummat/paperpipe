"""LEANN indexing helpers and MCP server runner."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

import click

from . import config
from .config import (
    DEFAULT_LEANN_INDEX_NAME,
    GEMINI_OPENAI_COMPAT_BASE_URL,
    _gemini_api_key,
    default_leann_embedding_mode,
    default_leann_embedding_model,
    default_leann_llm_model,
    default_leann_llm_provider,
)
from .output import debug, echo_error, echo_progress, echo_warning
from .paperqa import _validate_index_name

# -----------------------------------------------------------------------------
# Manifest types and I/O for incremental indexing
# -----------------------------------------------------------------------------

MANIFEST_VERSION = 1
LEANN_DEFAULT_GRAPH_DEGREE = 32
LEANN_DEFAULT_BUILD_COMPLEXITY = 64
LEANN_DEFAULT_NUM_THREADS = 1
LEANN_DEFAULT_RECOMPUTE = True


class FileEntry(TypedDict):
    """Tracking info for a single indexed file."""

    mtime: float  # File modification time at indexing
    indexed_at: str  # ISO timestamp when indexed
    status: str  # "ok" or "error"


class LeannManifest(TypedDict):
    """Manifest tracking indexed files for incremental updates."""

    version: int
    is_compact: bool
    embedding_mode: str
    embedding_model: str
    created_at: str
    updated_at: str
    files: dict[str, FileEntry]


@dataclass
class IndexDelta:
    """Result of comparing current PDFs against the manifest."""

    new_files: list[Path]  # Not in manifest
    changed_files: list[Path]  # mtime differs
    removed_files: list[str]  # In manifest but not on disk
    unchanged_count: int  # Files that don't need re-indexing


def _leann_manifest_path(index_name: str) -> Path:
    """Path to paperpipe's incremental indexing manifest."""
    return config.PAPER_DB / ".leann" / "indexes" / _validate_index_name(index_name) / "paperpipe_manifest.json"


def _load_leann_manifest(index_name: str) -> Optional[LeannManifest]:
    """Load manifest for an index. Returns None if missing or corrupt."""
    path = _leann_manifest_path(index_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("version") != MANIFEST_VERSION:
            debug("Manifest version mismatch (expected %d, got %s)", MANIFEST_VERSION, data.get("version"))
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        debug("Failed to load manifest: %s", e)
        return None


def _save_leann_manifest(index_name: str, manifest: LeannManifest) -> bool:
    """Save manifest. Returns True on success."""
    path = _leann_manifest_path(index_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2))
        return True
    except (OSError, PermissionError) as e:
        debug("Failed to save manifest: %s", e)
        return False


def _create_initial_manifest(
    *,
    index_name: str,
    docs_dir: Path,
    is_compact: bool,
    embedding_mode: str,
    embedding_model: str,
) -> LeannManifest:
    """Create manifest after a full index build, recording all current PDFs."""
    now_iso = datetime.now(timezone.utc).isoformat()
    files: dict[str, FileEntry] = {}

    for pdf in docs_dir.glob("*.pdf"):
        try:
            files[str(pdf.resolve())] = {
                "mtime": pdf.stat().st_mtime,
                "indexed_at": now_iso,
                "status": "ok",
            }
        except OSError:
            continue

    manifest: LeannManifest = {
        "version": MANIFEST_VERSION,
        "is_compact": is_compact,
        "embedding_mode": embedding_mode or "",
        "embedding_model": embedding_model or "",
        "created_at": now_iso,
        "updated_at": now_iso,
        "files": files,
    }

    _save_leann_manifest(index_name, manifest)
    return manifest


def _compute_index_delta(docs_dir: Path, manifest: Optional[LeannManifest]) -> IndexDelta:
    """Compare current PDFs against manifest to find new/changed/removed files."""
    indexed = manifest.get("files", {}) if manifest else {}
    current_pdfs = list(docs_dir.glob("*.pdf"))
    current_paths = {str(p.resolve()) for p in current_pdfs}

    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged_count = 0

    for pdf in current_pdfs:
        pdf_str = str(pdf.resolve())
        if pdf_str not in indexed:
            new_files.append(pdf)
        elif indexed[pdf_str].get("status") == "error":
            # Skip previously failed files (require --leann-force to retry)
            unchanged_count += 1
        else:
            try:
                current_mtime = pdf.stat().st_mtime
                indexed_mtime = indexed[pdf_str]["mtime"]
                # Allow small float tolerance for mtime comparison
                if abs(current_mtime - indexed_mtime) > 0.001:
                    changed_files.append(pdf)
                else:
                    unchanged_count += 1
            except OSError:
                # Can't stat file, skip it
                continue

    removed_files = [p for p in indexed if p not in current_paths]

    return IndexDelta(
        new_files=new_files,
        changed_files=changed_files,
        removed_files=removed_files,
        unchanged_count=unchanged_count,
    )


# -----------------------------------------------------------------------------
# Incremental update via LEANN Python API
# -----------------------------------------------------------------------------


class IncrementalUpdateError(Exception):
    """Raised when incremental update fails and a full rebuild is needed."""

    pass


def _leann_incremental_update(
    *,
    index_name: str,
    docs_dir: Path,
    backend_name: Optional[str] = None,
    embedding_mode: str,
    embedding_model: str,
    embedding_host: Optional[str] = None,
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    doc_chunk_size: Optional[int] = None,
    doc_chunk_overlap: Optional[int] = None,
    graph_degree: Optional[int] = None,
    build_complexity: Optional[int] = None,
    num_threads: Optional[int] = None,
    is_recompute: Optional[bool] = None,
) -> tuple[int, int, int]:
    """
    Incrementally update LEANN index with new/changed files using Python API.

    Returns: (added_count, unchanged_count, error_count)
    Raises: IncrementalUpdateError if incremental update not possible
    """
    manifest = _load_leann_manifest(index_name)
    if manifest is None:
        raise IncrementalUpdateError("No manifest found; run full build first")
    if manifest.get("is_compact", True):
        raise IncrementalUpdateError("Index is compact; incremental updates not supported")

    meta_backend_name = _load_leann_backend_name(index_name)
    meta_backend_kwargs = _load_leann_backend_kwargs(index_name)
    if backend_name and meta_backend_name and backend_name != meta_backend_name:
        raise IncrementalUpdateError(f"Backend mismatch: index={meta_backend_name}, requested={backend_name}")

    # Validate embedding settings match
    if embedding_mode and manifest["embedding_mode"] and manifest["embedding_mode"] != embedding_mode:
        raise IncrementalUpdateError(
            f"Embedding mode mismatch: index={manifest['embedding_mode']}, requested={embedding_mode}"
        )
    if embedding_model and manifest["embedding_model"] and manifest["embedding_model"] != embedding_model:
        raise IncrementalUpdateError(
            f"Embedding model mismatch: index={manifest['embedding_model']}, requested={embedding_model}"
        )

    delta = _compute_index_delta(docs_dir, manifest)
    if delta.removed_files:
        raise IncrementalUpdateError("Removed files detected; full rebuild required")
    files_to_add = delta.new_files + delta.changed_files

    if not files_to_add:
        # Clean up removed files from manifest
        if delta.removed_files:
            for removed in delta.removed_files:
                manifest["files"].pop(removed, None)
            _save_leann_manifest(index_name, manifest)
        return 0, delta.unchanged_count, 0

    try:
        from leann.api import LeannBuilder
    except ImportError as e:
        raise IncrementalUpdateError(f"LEANN Python API not available: {e}") from e

    index_dir = config.PAPER_DB / ".leann" / "indexes" / _validate_index_name(index_name)
    index_path = index_dir / "documents.leann"

    index_file = index_dir / f"{index_path.stem}.index"
    if not index_file.exists():
        raise IncrementalUpdateError(f"Index file not found: {index_file}")

    # Build kwargs for LeannBuilder
    builder_kwargs: dict = {
        "is_compact": False,
    }
    # Use manifest settings as defaults, override with explicit params
    effective_mode = embedding_mode or manifest.get("embedding_mode") or ""
    effective_model = embedding_model or manifest.get("embedding_model") or ""
    effective_backend = (backend_name or "").strip() or (meta_backend_name or "").strip() or "hnsw"
    effective_graph_degree = (
        graph_degree
        if graph_degree is not None
        else meta_backend_kwargs.get("graph_degree", LEANN_DEFAULT_GRAPH_DEGREE)
    )
    effective_complexity = (
        build_complexity
        if build_complexity is not None
        else meta_backend_kwargs.get("complexity", LEANN_DEFAULT_BUILD_COMPLEXITY)
    )
    effective_num_threads = (
        num_threads if num_threads is not None else meta_backend_kwargs.get("num_threads", LEANN_DEFAULT_NUM_THREADS)
    )
    effective_recompute = (
        is_recompute if is_recompute is not None else meta_backend_kwargs.get("is_recompute", LEANN_DEFAULT_RECOMPUTE)
    )

    if effective_mode:
        builder_kwargs["embedding_mode"] = effective_mode
    if effective_model:
        builder_kwargs["embedding_model"] = effective_model
    if effective_backend:
        builder_kwargs["backend_name"] = effective_backend
    if effective_graph_degree is not None:
        builder_kwargs["graph_degree"] = int(effective_graph_degree)
    if effective_complexity is not None:
        builder_kwargs["complexity"] = int(effective_complexity)
    if effective_num_threads is not None:
        builder_kwargs["num_threads"] = int(effective_num_threads)
    if effective_recompute is not None:
        builder_kwargs["is_recompute"] = bool(effective_recompute)
    embedding_options: dict[str, str] = {}
    try:
        from leann.settings import resolve_ollama_host, resolve_openai_api_key, resolve_openai_base_url

        if effective_mode.lower() == "ollama":
            embedding_options["host"] = resolve_ollama_host(embedding_host)
        elif "openai" in effective_mode.lower():
            embedding_options["base_url"] = resolve_openai_base_url(embedding_api_base)
            resolved_key = resolve_openai_api_key(embedding_api_key)
            if resolved_key:
                embedding_options["api_key"] = resolved_key
    except Exception:
        if effective_mode.lower() == "ollama":
            if embedding_host:
                embedding_options["host"] = embedding_host
        elif "openai" in effective_mode.lower():
            if embedding_api_base:
                embedding_options["base_url"] = embedding_api_base
            if embedding_api_key:
                embedding_options["api_key"] = embedding_api_key
    if embedding_options:
        builder_kwargs["embedding_options"] = embedding_options
    if doc_chunk_size:
        builder_kwargs["doc_chunk_size"] = doc_chunk_size
    if doc_chunk_overlap:
        builder_kwargs["doc_chunk_overlap"] = doc_chunk_overlap

    try:
        builder = LeannBuilder(**builder_kwargs)
    except Exception as e:
        raise IncrementalUpdateError(f"Failed to create LeannBuilder: {e}") from e

    # Check for document adding capability - LEANN API may vary by version
    add_method = None
    for method_name in ("add_document", "add_file"):
        if hasattr(builder, method_name):
            add_method = getattr(builder, method_name)
            break

    use_add_text = add_method is None and hasattr(builder, "add_text")
    if add_method is None and not use_add_text:
        raise IncrementalUpdateError(
            "LEANN Python API does not support incremental document updates. "
            "LeannBuilder lacks add_document/add_file/add_text methods."
        )

    if not hasattr(builder, "update_index"):
        raise IncrementalUpdateError("LEANN Python API does not support update_index method")

    added = 0
    errors = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    if use_add_text:
        try:
            from leann.cli import LeannCLI
            from llama_index.core.node_parser import SentenceSplitter
        except Exception as e:
            raise IncrementalUpdateError(f"LEANN chunking helpers unavailable: {e}") from e

        leann_cli = LeannCLI()
        effective_doc_chunk_size = doc_chunk_size or leann_cli.node_parser.chunk_size
        effective_doc_chunk_overlap = doc_chunk_overlap or leann_cli.node_parser.chunk_overlap
        if effective_doc_chunk_overlap >= effective_doc_chunk_size:
            effective_doc_chunk_overlap = max(0, effective_doc_chunk_size - 1)
        leann_cli.node_parser = SentenceSplitter(
            chunk_size=effective_doc_chunk_size,
            chunk_overlap=effective_doc_chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
        )

        args: dict[str, Any] = {
            "use_ast_chunking": False,
            "ast_chunk_size": 768,
            "ast_chunk_overlap": 96,
            "ast_fallback_traditional": True,
        }

        for pdf in files_to_add:
            try:
                chunks = leann_cli.load_documents(
                    [str(pdf)],
                    custom_file_types=".pdf",
                    include_hidden=True,
                    args=args,
                )
                if not chunks:
                    raise ValueError("No chunks extracted")
                for i, chunk in enumerate(chunks):
                    metadata = dict(chunk.get("metadata") or {})
                    if "id" not in metadata:
                        metadata["id"] = f"{pdf.resolve()}::{i}"
                    # Sanitize text: PDF extractors can produce lone surrogates
                    # (e.g. \ud835 from math symbols) that are invalid in UTF-8.
                    text = chunk.get("text", "").encode("utf-8", errors="replace").decode("utf-8")
                    metadata = {
                        k: v.encode("utf-8", errors="replace").decode("utf-8") if isinstance(v, str) else v
                        for k, v in metadata.items()
                    }
                    builder.add_text(text, metadata=metadata)
                manifest["files"][str(pdf.resolve())] = {
                    "mtime": pdf.stat().st_mtime,
                    "indexed_at": now_iso,
                    "status": "ok",
                }
                added += 1
            except Exception as e:
                debug("Failed to index %s: %s", pdf, e)
                errors += 1
    else:
        assert add_method is not None
        for pdf in files_to_add:
            try:
                add_method(str(pdf))
                manifest["files"][str(pdf.resolve())] = {
                    "mtime": pdf.stat().st_mtime,
                    "indexed_at": now_iso,
                    "status": "ok",
                }
                added += 1
            except Exception as e:
                debug("Failed to index %s: %s", pdf, e)
                errors += 1

    if added > 0:
        try:
            builder.update_index(str(index_path))
        except Exception as e:
            raise IncrementalUpdateError(f"Failed to update index: {e}") from e

    # Clean up removed files from manifest
    for removed in delta.removed_files:
        manifest["files"].pop(removed, None)

    _save_leann_manifest(index_name, manifest)

    return added, delta.unchanged_count, errors


_REDACT_FLAGS = {"--api-key", "--embedding-api-key"}


def _redact_cmd(cmd: list[str]) -> str:
    """Return a shell-safe string representation of cmd with API keys redacted."""
    redacted = list(cmd)
    i = 0
    while i < len(redacted):
        if redacted[i] in _REDACT_FLAGS and i + 1 < len(redacted):
            redacted[i + 1] = "***"
            i += 2
        elif any(redacted[i].startswith(f"{flag}=") for flag in _REDACT_FLAGS):
            flag = redacted[i].split("=", 1)[0]
            redacted[i] = f"{flag}=***"
            i += 1
        else:
            i += 1
    return shlex.join(redacted)


def _extract_arg_value(args: list[str], flag: str) -> Optional[str]:
    """Extract the value for a CLI flag from args list."""
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


# -----------------------------------------------------------------------------
# Index path helpers
# -----------------------------------------------------------------------------


def _leann_index_meta_path(index_name: str) -> Path:
    return config.PAPER_DB / ".leann" / "indexes" / _validate_index_name(index_name) / "documents.leann.meta.json"


def _load_leann_backend_name(index_name: str) -> Optional[str]:
    meta_path = _leann_index_meta_path(index_name)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    backend_name = meta.get("backend_name")
    return backend_name if isinstance(backend_name, str) and backend_name.strip() else None


def _load_leann_backend_kwargs(index_name: str) -> dict:
    meta_path = _leann_index_meta_path(index_name)
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    backend_kwargs = meta.get("backend_kwargs")
    return backend_kwargs if isinstance(backend_kwargs, dict) else {}


def _leann_build_index(
    *, index_name: str, docs_dir: Path, force: bool, no_compact: bool, extra_args: list[str]
) -> None:
    """Build LEANN index. Tries incremental update first if possible, falls back to full rebuild."""
    if not shutil.which("leann"):
        echo_error("LEANN not installed. Install with: pip install 'paperpipe[leann]'")
        raise SystemExit(1)

    index_name = (index_name or "").strip()
    if not index_name:
        raise click.UsageError("index name must be non-empty")

    if any(arg == "--file-types" or arg.startswith("--file-types=") for arg in extra_args):
        raise click.UsageError("LEANN indexing in paperpipe is PDF-only; do not pass --file-types.")

    # Try incremental update if manifest exists and not forcing full rebuild
    manifest = _load_leann_manifest(index_name)
    can_incremental = manifest is not None and not manifest.get("is_compact", True) and not force and no_compact

    if can_incremental:
        # Extract embedding settings from extra_args
        embedding_mode = _extract_arg_value(extra_args, "--embedding-mode") or default_leann_embedding_mode()
        embedding_model = _extract_arg_value(extra_args, "--embedding-model") or default_leann_embedding_model()
        backend_name = _extract_arg_value(extra_args, "--backend-name")
        embedding_host = _extract_arg_value(extra_args, "--embedding-host")
        embedding_api_base = _extract_arg_value(extra_args, "--embedding-api-base")
        embedding_api_key = _extract_arg_value(extra_args, "--embedding-api-key")
        doc_chunk_size_str = _extract_arg_value(extra_args, "--doc-chunk-size")
        doc_chunk_overlap_str = _extract_arg_value(extra_args, "--doc-chunk-overlap")
        graph_degree_str = _extract_arg_value(extra_args, "--graph-degree")
        build_complexity_str = _extract_arg_value(extra_args, "--complexity")
        num_threads_str = _extract_arg_value(extra_args, "--num-threads")
        is_recompute: Optional[bool]
        if "--no-recompute" in extra_args:
            is_recompute = False
        elif "--recompute" in extra_args:
            is_recompute = True
        else:
            is_recompute = None

        try:
            added, unchanged, errors = _leann_incremental_update(
                index_name=index_name,
                docs_dir=docs_dir,
                backend_name=backend_name,
                embedding_mode=embedding_mode or "",
                embedding_model=embedding_model or "",
                embedding_host=embedding_host,
                embedding_api_base=embedding_api_base,
                embedding_api_key=embedding_api_key,
                doc_chunk_size=int(doc_chunk_size_str) if doc_chunk_size_str else None,
                doc_chunk_overlap=int(doc_chunk_overlap_str) if doc_chunk_overlap_str else None,
                graph_degree=int(graph_degree_str) if graph_degree_str else None,
                build_complexity=int(build_complexity_str) if build_complexity_str else None,
                num_threads=int(num_threads_str) if num_threads_str else None,
                is_recompute=is_recompute,
            )
            if added > 0 or errors > 0:
                echo_progress(f"Incremental update: {added} added, {unchanged} unchanged, {errors} errors")
            else:
                echo_progress(f"Index up to date ({unchanged} files)")
            return
        except IncrementalUpdateError as e:
            echo_warning(f"Incremental update not possible ({e}); performing full rebuild")
            # Fall through to full rebuild

    has_embedding_model_override = any(
        arg == "--embedding-model" or arg.startswith("--embedding-model=") for arg in extra_args
    )
    has_embedding_mode_override = any(
        arg == "--embedding-mode" or arg.startswith("--embedding-mode=") for arg in extra_args
    )

    cmd = ["leann", "build", index_name, "--docs", str(docs_dir), "--file-types", ".pdf"]
    if force:
        cmd.append("--force")
    if no_compact:
        cmd.append("--no-compact")

    # Extract explicit overrides from extra_args first to avoid spurious fallback logs
    embedding_model_override: Optional[str] = None
    embedding_mode_override: Optional[str] = None
    for i, arg in enumerate(extra_args):
        if arg == "--embedding-model":
            if i + 1 >= len(extra_args):
                raise click.UsageError("--embedding-model flag requires a value")
            embedding_model_override = extra_args[i + 1]
            if not embedding_model_override.strip():
                raise click.UsageError("--embedding-model flag requires a non-empty value")
        elif arg.startswith("--embedding-model="):
            embedding_model_override = arg.split("=", 1)[1]
            if not embedding_model_override.strip():
                raise click.UsageError("--embedding-model flag requires a non-empty value")
        elif arg == "--embedding-mode":
            if i + 1 >= len(extra_args):
                raise click.UsageError("--embedding-mode flag requires a value")
            embedding_mode_override = extra_args[i + 1]
            if not embedding_mode_override.strip():
                raise click.UsageError("--embedding-mode flag requires a non-empty value")
        elif arg.startswith("--embedding-mode="):
            embedding_mode_override = arg.split("=", 1)[1]
            if not embedding_mode_override.strip():
                raise click.UsageError("--embedding-mode flag requires a non-empty value")

    # Add defaults to command only if user didn't provide explicit overrides
    if not has_embedding_model_override:
        embedding_model_default = default_leann_embedding_model()
        if embedding_model_default:
            cmd.extend(["--embedding-model", embedding_model_default])
    if not has_embedding_mode_override:
        embedding_mode_default = default_leann_embedding_mode()
        if embedding_mode_default:
            cmd.extend(["--embedding-mode", embedding_mode_default])

    # Track effective embedding settings for metadata (explicit or default)
    embedding_model_for_meta = embedding_model_override or default_leann_embedding_model()
    embedding_mode_for_meta = embedding_mode_override or default_leann_embedding_mode()

    cmd.extend(extra_args)
    debug("Running LEANN: %s", _redact_cmd(cmd))
    proc = subprocess.run(cmd, cwd=config.PAPER_DB)
    if proc.returncode != 0:
        echo_error(f"LEANN command failed (exit code {proc.returncode})")
        echo_error(f"Command: {_redact_cmd(cmd)}")
        raise SystemExit(proc.returncode)

    # Write metadata on success
    try:
        from paperpipe.paperqa_mcp_server import _write_leann_metadata

        _write_leann_metadata(
            index_name=index_name,
            embedding_mode=embedding_mode_for_meta,
            embedding_model=embedding_model_for_meta,
        )
    except (ImportError, ModuleNotFoundError) as e:
        echo_warning(f"MCP server not available; skipping metadata write: {e}")
    except (PermissionError, OSError) as e:
        echo_error(f"Failed to write LEANN index metadata due to filesystem error: {e}")
        echo_error("Index was built successfully but metadata is incomplete.")
        raise SystemExit(1)
    except Exception as e:
        # Log unexpected errors but don't fail the build
        echo_warning(f"Unexpected error writing LEANN index metadata: {e}")
        debug("Metadata write failed:\n%s", traceback.format_exc())

    # Create manifest for incremental indexing
    _create_initial_manifest(
        index_name=index_name,
        docs_dir=docs_dir,
        is_compact=not no_compact,
        embedding_mode=embedding_mode_for_meta or "",
        embedding_model=embedding_model_for_meta or "",
    )


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

    index_name = (index_name or "").strip() or DEFAULT_LEANN_INDEX_NAME
    meta_path = _leann_index_meta_path(index_name)
    if not meta_path.exists():
        echo_error(f"LEANN index {index_name!r} not found at {meta_path}")
        echo_error("Build it first: papi index --backend leann")
        raise SystemExit(1)

    cmd: list[str] = ["leann", "ask", index_name, query]
    cmd.extend(["--llm", provider])
    cmd.extend(["--model", model])
    if not api_base and provider.lower() == "openai" and model.lower().startswith("gemini-"):
        api_base = GEMINI_OPENAI_COMPAT_BASE_URL
    if not api_key and provider.lower() == "openai" and model.lower().startswith("gemini-"):
        api_key = _gemini_api_key()
        if not api_key:
            echo_warning(
                "LEANN is configured for Gemini via OpenAI-compatible endpoint but GEMINI_API_KEY/GOOGLE_API_KEY "
                "is not set; the request will likely fail."
            )
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
    debug("Running LEANN: %s", _redact_cmd(cmd))

    if interactive:
        proc = subprocess.run(cmd, cwd=config.PAPER_DB)
        if proc.returncode != 0:
            echo_error(f"LEANN command failed (exit code {proc.returncode}).")
            raise SystemExit(proc.returncode)
        return

    proc = subprocess.Popen(
        cmd, cwd=config.PAPER_DB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        click.echo(line, nl=False)
    returncode = proc.wait()
    if returncode != 0:
        echo_error(f"LEANN command failed (exit code {returncode}).")
        raise SystemExit(returncode)
