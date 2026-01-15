"""LEANN indexing helpers and MCP server runner."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional

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
from .output import debug, echo_error, echo_warning


def _leann_index_meta_path(index_name: str) -> Path:
    return config.PAPER_DB / ".leann" / "indexes" / index_name / "documents.leann.meta.json"


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
    debug("Running LEANN: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=config.PAPER_DB)
    if proc.returncode != 0:
        echo_error(f"LEANN command failed (exit code {proc.returncode})")
        echo_error(f"Command: {shlex.join(cmd)}")
        raise SystemExit(proc.returncode)

    # Write metadata on success
    try:
        from paperpipe.mcp_server import _write_leann_metadata

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
    debug("Running LEANN: %s", shlex.join(cmd))

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
