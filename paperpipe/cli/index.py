"""Index command for PaperQA2/LEANN/SQLite."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click

from .. import config, paperqa
from ..config import (
    _effective_leann_index_name,
    _is_ollama_model_id,
    _strip_ollama_prefix,
    default_pqa_concurrency,
    default_pqa_embedding_model,
    default_pqa_enrichment_llm,
    default_pqa_index_dir,
    default_pqa_llm_model,
    default_pqa_ollama_timeout,
    default_pqa_settings_name,
    default_pqa_summary_llm,
    default_pqa_temperature,
    default_pqa_verbosity,
    pqa_index_name_for_embedding,
)
from ..core import load_index
from ..leann import _leann_build_index
from ..output import echo_error, echo_progress, echo_success
from ..search import (
    _ensure_search_index_schema,
    _search_db_path,
    _search_index_rebuild,
    _search_index_upsert,
    _sqlite_connect,
)


@click.command("index", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "--backend",
    type=click.Choice(["pqa", "leann", "search"], case_sensitive=False),
    default="pqa",
    show_default=True,
    help="Which backend to index for: PaperQA2 (`pqa`), LEANN (`leann`), or SQLite FTS5 (`search`).",
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
@click.option(
    "--pqa-raw",
    is_flag=True,
    help="Pass through PaperQA2 output without filtering (also enabled by global -v/--verbose).",
)
@click.option("--leann-index", default="papers", show_default=True, help="LEANN index name to build.")
@click.option("--leann-force", is_flag=True, help="Force LEANN rebuild (passes --force to `leann build`).")
@click.option(
    "--leann-backend-name",
    type=click.Choice(["hnsw", "diskann"], case_sensitive=False),
    default=None,
    help="LEANN backend name (maps to `leann build --backend-name`).",
)
@click.option(
    "--leann-embedding-model",
    default=None,
    help="LEANN embedding model (maps to `leann build --embedding-model`).",
)
@click.option(
    "--leann-embedding-mode",
    type=click.Choice(["sentence-transformers", "openai", "mlx", "ollama"], case_sensitive=False),
    default=None,
    help="LEANN embedding mode (maps to `leann build --embedding-mode`).",
)
@click.option(
    "--leann-embedding-host",
    default=None,
    help="LEANN embedding host (maps to `leann build --embedding-host`).",
)
@click.option(
    "--leann-embedding-api-base",
    default=None,
    help="LEANN embedding API base URL (maps to `leann build --embedding-api-base`).",
)
@click.option(
    "--leann-embedding-api-key",
    default=None,
    help="LEANN embedding API key (maps to `leann build --embedding-api-key`).",
)
@click.option(
    "--leann-graph-degree",
    type=int,
    default=None,
    help="LEANN build graph degree (maps to `leann build --graph-degree`).",
)
@click.option(
    "--leann-build-complexity",
    type=int,
    default=None,
    help="LEANN build complexity (maps to `leann build --complexity`).",
)
@click.option(
    "--leann-num-threads",
    type=int,
    default=None,
    help="LEANN build thread count (maps to `leann build --num-threads`).",
)
@click.option(
    "--leann-doc-chunk-size",
    type=int,
    default=None,
    help="LEANN PDF chunk size in TOKENS (maps to `leann build --doc-chunk-size`).",
)
@click.option(
    "--leann-doc-chunk-overlap",
    type=int,
    default=None,
    help="LEANN PDF chunk overlap in TOKENS (maps to `leann build --doc-chunk-overlap`).",
)
@click.option(
    "--leann-no-recompute",
    is_flag=True,
    help="Disable embedding recomputation during LEANN build.",
)
@click.option("--search-rebuild", is_flag=True, help="Rebuild the SQLite FTS search index from scratch.")
@click.option(
    "--search-include-tex",
    is_flag=True,
    help="Index `source.tex` contents into the FTS index (larger DB; slower build).",
)
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
    pqa_raw: bool,
    leann_index: str,
    leann_force: bool,
    leann_backend_name: Optional[str],
    leann_embedding_model: Optional[str],
    leann_embedding_mode: Optional[str],
    leann_embedding_host: Optional[str],
    leann_embedding_api_base: Optional[str],
    leann_embedding_api_key: Optional[str],
    leann_graph_degree: Optional[int],
    leann_build_complexity: Optional[int],
    leann_num_threads: Optional[int],
    leann_doc_chunk_size: Optional[int],
    leann_doc_chunk_overlap: Optional[int],
    leann_no_recompute: bool,
    search_rebuild: bool,
    search_include_tex: bool,
) -> None:
    """Build/update the retrieval index for PaperQA2 (default), LEANN, or SQLite FTS5."""
    backend_norm = (backend or "pqa").strip().lower()
    if backend_norm == "leann":
        leann_index = _effective_leann_index_name(
            ctx=ctx,
            param_name="leann_index",
            raw_index_name=leann_index,
            embedding_mode=leann_embedding_mode,
            embedding_model=leann_embedding_model,
        )
        config.PAPER_DB.mkdir(parents=True, exist_ok=True)
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)

        if any(arg == "--docs" or arg.startswith("--docs=") for arg in ctx.args):
            raise click.UsageError("paperpipe controls LEANN --docs; do not pass --docs.")

        staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
        paperqa._refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)

        leann_extra_args: list[str] = []

        def _append_str_flag(flag: str, value: Optional[str]) -> None:
            if value is None:
                return
            value = value.strip()
            if value:
                leann_extra_args.extend([flag, value])

        def _append_int_flag(flag: str, value: Optional[int]) -> None:
            if value is None:
                return
            leann_extra_args.extend([flag, str(value)])

        _append_str_flag("--backend-name", leann_backend_name)
        _append_str_flag("--embedding-model", leann_embedding_model)
        _append_str_flag("--embedding-mode", leann_embedding_mode)
        _append_str_flag("--embedding-host", leann_embedding_host)
        _append_str_flag("--embedding-api-base", leann_embedding_api_base)
        _append_str_flag("--embedding-api-key", leann_embedding_api_key)
        _append_int_flag("--graph-degree", leann_graph_degree)
        _append_int_flag("--complexity", leann_build_complexity)
        _append_int_flag("--num-threads", leann_num_threads)
        _append_int_flag("--doc-chunk-size", leann_doc_chunk_size)
        _append_int_flag("--doc-chunk-overlap", leann_doc_chunk_overlap)

        if leann_no_recompute:
            leann_extra_args.append("--no-recompute")

        leann_extra_args.extend(list(ctx.args))
        _leann_build_index(index_name=leann_index, docs_dir=staging_dir, force=leann_force, extra_args=leann_extra_args)
        echo_success(f"Built LEANN index {leann_index!r} under {config.PAPER_DB / '.leann' / 'indexes' / leann_index}")
        return

    if backend_norm == "search":
        if not search_rebuild and search_include_tex:
            raise click.UsageError("--search-include-tex only applies with --search-rebuild")

        db_path = _search_db_path()
        if search_rebuild or not db_path.exists():
            count = _search_index_rebuild(include_tex=search_include_tex)
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
        return

    if backend_norm != "pqa":
        raise click.UsageError(f"Unknown --backend: {backend}")

    if pqa_concurrency is not None and pqa_concurrency < 1:
        raise click.UsageError("--pqa-concurrency must be >= 1")

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
    if not has_parsing_override and not paperqa._pillow_available():
        cmd.extend(["--parsing.multimodal", "OFF"])

    pqa_llm_source = ctx.get_parameter_source("pqa_llm")
    pqa_embedding_source = ctx.get_parameter_source("pqa_embedding")
    pqa_summary_llm_source = ctx.get_parameter_source("pqa_summary_llm")
    pqa_temperature_source = ctx.get_parameter_source("pqa_temperature")
    pqa_verbosity_source = ctx.get_parameter_source("pqa_verbosity")

    llm_for_pqa: Optional[str] = None
    embedding_for_pqa: Optional[str] = None
    embedding_for_pqa_cmd: Optional[str] = None

    if pqa_llm_source != click.core.ParameterSource.DEFAULT:
        llm_for_pqa = pqa_llm
    elif not has_settings_flag:
        llm_for_pqa = default_pqa_llm_model()

    if pqa_embedding_source != click.core.ParameterSource.DEFAULT:
        embedding_for_pqa = pqa_embedding
    elif not has_settings_flag:
        embedding_for_pqa = default_pqa_embedding_model()

    embedding_for_pqa_cmd = embedding_for_pqa
    embedding_for_pqa_stripped = _strip_ollama_prefix(embedding_for_pqa)
    if embedding_for_pqa_stripped and embedding_for_pqa_stripped != embedding_for_pqa:
        # Work around a LiteLLM bug where `ollama/...` isn't stripped before calling Ollama's /api/embed.
        # PaperQA2/LMI calls litellm.aembedding(model=<embedding>), so we pass the bare model name and
        # force the provider via embedding_config kwargs.
        embedding_for_pqa_cmd = embedding_for_pqa_stripped

    if llm_for_pqa:
        cmd.extend(["--llm", llm_for_pqa])
    if embedding_for_pqa_cmd:
        cmd.extend(["--embedding", embedding_for_pqa_cmd])

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
    idx_dir_val = paperqa._extract_flag_value(
        effective_args, names={"--agent.index.index_directory", "--agent.index.index-directory"}
    )
    idx_name_val = paperqa._extract_flag_value(effective_args, names={"--index", "-i", "--agent.index.name"})
    excluded_files: set[str] = set()
    if idx_dir_val and idx_name_val and not pqa_retry_failed:
        fp = paperqa._paperqa_index_files_path(index_directory=Path(idx_dir_val), index_name=idx_name_val)
        if fp.exists():
            m = paperqa._paperqa_load_index_files_map(fp)
            if m:
                excluded_files = {Path(k).name for k, v in m.items() if v == "ERROR"}

    # Paper directory (defaults to managed staging dir)
    paper_dir_override = paperqa._extract_flag_value(
        list(ctx.args),
        names={"--agent.index.paper_directory", "--agent.index.paper-directory"},
    )
    if paper_dir_override:
        paper_dir = Path(paper_dir_override).expanduser()
    else:
        paper_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
        paperqa._refresh_pqa_pdf_staging_dir(staging_dir=paper_dir, exclude_names=excluded_files)
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

    # If the embedding is an Ollama model id, inject an embedding_config that forces the provider
    # while keeping the user-friendly `ollama/...` id for index naming.
    #
    # Note: `--embedding_config` is a *global* pqa flag and must come before the `index` subcommand.
    if (
        embedding_for_pqa
        and _is_ollama_model_id(embedding_for_pqa)
        and not paperqa._pqa_has_flag(ctx.args, names={"--embedding_config", "--embedding-config"})
    ):
        ollama_timeout = default_pqa_ollama_timeout()
        cmd.extend(
            [
                "--embedding_config",
                json.dumps(
                    {"kwargs": {"custom_llm_provider": "ollama", "timeout": ollama_timeout}},
                    separators=(",", ":"),
                ),
            ]
        )

    cmd.extend(ctx.args)

    # Clear ERROR markers if requested (PaperQA2 won't retry failed docs otherwise)
    index_dir_raw = paperqa._extract_flag_value(
        cmd,
        names={"--agent.index.index_directory", "--agent.index.index-directory"},
    )
    index_name_raw = paperqa._extract_flag_value(cmd, names={"--agent.index.name"}) or paperqa._extract_flag_value(
        cmd, names={"--index", "-i"}
    )
    if pqa_retry_failed and index_dir_raw and index_name_raw:
        cleared, _ = paperqa._paperqa_clear_failed_documents(
            index_directory=Path(index_dir_raw), index_name=index_name_raw
        )
        if cleared:
            echo_progress(f"Cleared {cleared} failed PaperQA2 document(s) for retry.")

    cmd.extend(["index", str(paper_dir)])

    env = os.environ.copy()
    env.setdefault("PQA_LITELLM_MAX_CALLBACKS", "1000")
    env.setdefault("LMI_LITELLM_MAX_CALLBACKS", "1000")

    if _is_ollama_model_id(llm_for_pqa) or _is_ollama_model_id(embedding_for_pqa):
        config._prepare_ollama_env(env)
        err = config._ollama_reachability_error(api_base=env["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            raise SystemExit(1)

    proc = subprocess.Popen(
        cmd,
        cwd=config.PAPERS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    root_verbose = bool(ctx.find_root().params.get("verbose"))
    passthrough_verbose = any(arg in {"--verbose", "-v"} for arg in ctx.args)
    raw_output = bool(
        pqa_raw
        or root_verbose
        or passthrough_verbose
        or (pqa_verbosity_source != click.core.ParameterSource.DEFAULT and (pqa_verbosity or 0) > 0)
    )

    captured_output: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured_output.append(line)
        if raw_output:
            click.echo(line, nl=False)
        elif not paperqa._pqa_is_noisy_index_line(line):
            click.echo(line, nl=False)
    returncode = proc.wait()

    # Write metadata on success (before error check)
    if returncode == 0 and backend == "pqa":
        try:
            # Import here to avoid issues if MCP isn't installed
            from paperpipe.mcp_server import _write_index_metadata

            # Extract values used for indexing
            index_dir_for_meta = Path(index_dir_raw) if index_dir_raw else default_pqa_index_dir()
            index_name_for_meta = index_name_raw or pqa_index_name_for_embedding(embedding_for_pqa or "")

            _write_index_metadata(
                index_root=index_dir_for_meta,
                index_name=index_name_for_meta,
                embedding_model=embedding_for_pqa or "",
            )
        except Exception as e:
            # Non-fatal: log warning but don't fail the index operation
            echo_progress(f"Warning: Failed to write index metadata: {e}")

    if returncode != 0:
        if not raw_output:
            paperqa._pqa_print_filtered_index_output_on_failure(captured_output=captured_output)
        raise SystemExit(returncode)
