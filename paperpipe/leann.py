"""LEANN indexing helpers and MCP server runner."""

from __future__ import annotations

import shlex
import shutil
import subprocess
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

    if not has_embedding_model_override:
        cmd.extend(["--embedding-model", default_leann_embedding_model()])
    if not has_embedding_mode_override:
        cmd.extend(["--embedding-mode", default_leann_embedding_mode()])

    cmd.extend(extra_args)
    debug("Running LEANN: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=config.PAPER_DB)
    if proc.returncode != 0:
        echo_error(f"LEANN command failed (exit code {proc.returncode}).")
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

    index_name = (index_name or "").strip() or DEFAULT_LEANN_INDEX_NAME
    meta_path = _leann_index_meta_path(index_name)
    if not meta_path.exists():
        echo_error(f"LEANN index {index_name!r} not found at {meta_path}")
        echo_error("Build it first: papi leann-index (or: papi index --backend leann)")
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
