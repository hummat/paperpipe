"""Ask command for querying papers via PaperQA2/LEANN."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click

from .. import config, paperqa
from ..config import (
    DEFAULT_LEANN_INDEX_NAME,
    _effective_leann_index_name,
    _is_ollama_model_id,
    _strip_ollama_prefix,
    default_pqa_answer_length,
    default_pqa_concurrency,
    default_pqa_embedding_model,
    default_pqa_enrichment_llm,
    default_pqa_evidence_k,
    default_pqa_index_dir,
    default_pqa_llm_model,
    default_pqa_max_sources,
    default_pqa_ollama_timeout,
    default_pqa_settings_name,
    default_pqa_summary_llm,
    default_pqa_temperature,
    default_pqa_timeout,
    default_pqa_verbosity,
    pqa_index_name_for_embedding,
)
from ..leann import _ask_leann, _leann_build_index, _leann_index_meta_path
from ..output import debug, echo_error, echo_progress, echo_warning


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
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
    "--leann-no-recompute",
    is_flag=True,
    help="Disable embedding recomputation during LEANN ask.",
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
    "--leann-no-auto-index",
    is_flag=True,
    help="Disable auto-build of the LEANN index when missing.",
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
    leann_no_recompute: bool,
    leann_pruning_strategy: Optional[str],
    leann_thinking_budget: Optional[str],
    leann_interactive: bool,
    leann_no_auto_index: bool,
):
    """
    Query papers using PaperQA2 (default) or LEANN.

    Common options are exposed as first-class flags. Any additional arguments
    are passed directly to PaperQA2 (e.g., --agent.search_count 10).
    """
    backend_norm = (backend or "pqa").strip().lower()
    output_format_norm = (output_format or "text").strip().lower()
    if backend_norm == "leann":
        leann_index = _effective_leann_index_name(ctx=ctx, param_name="leann_index", raw_index_name=leann_index)
        if output_format_norm != "text":
            raise click.UsageError("--format evidence-blocks is only supported with --backend pqa.")
        config.PAPER_DB.mkdir(parents=True, exist_ok=True)
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        if not leann_no_auto_index:
            index_name = (leann_index or "").strip() or DEFAULT_LEANN_INDEX_NAME
            meta_path = _leann_index_meta_path(index_name)
            if not meta_path.exists():
                echo_progress(f"LEANN index {index_name!r} not found; building it now...")
                staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
                paperqa._refresh_pqa_pdf_staging_dir(staging_dir=staging_dir)
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
            recompute_embeddings=not leann_no_recompute,
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
    if not has_parsing_override and not paperqa._pillow_available():
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

    embedding_for_pqa_cmd = embedding_for_pqa
    embedding_for_pqa_stripped = _strip_ollama_prefix(embedding_for_pqa)
    if embedding_for_pqa_stripped and embedding_for_pqa_stripped != embedding_for_pqa:
        # Work around a LiteLLM bug where `ollama/...` isn't stripped before calling Ollama's /api/embed.
        # PaperQA2/LMI calls litellm.aembedding(model=<embedding>), so we pass the bare model name and
        # force the provider via embedding_config kwargs.
        embedding_for_pqa_cmd = embedding_for_pqa_stripped

    if embedding_for_pqa_cmd:
        cmd.extend(["--embedding", embedding_for_pqa_cmd])

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
    idx_dir_val = paperqa._extract_flag_value(
        effective_args, names={"--agent.index.index_directory", "--agent.index.index-directory"}
    )
    idx_name_val = paperqa._extract_flag_value(effective_args, names={"--index", "-i", "--agent.index.name"})

    excluded_files: set[str] = set()
    if idx_dir_val and idx_name_val and not retry_failed:
        # Load index and find errors
        fp = paperqa._paperqa_index_files_path(index_directory=Path(idx_dir_val), index_name=idx_name_val)
        if fp.exists():
            m = paperqa._paperqa_load_index_files_map(fp)
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
        staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
        paperqa._refresh_pqa_pdf_staging_dir(staging_dir=staging_dir, exclude_names=excluded_files)
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

    summary_llm_for_pqa: Optional[str] = None
    enrichment_llm_for_pqa: Optional[str] = None

    # summary_llm: first-class option takes precedence, then config, then falls back to llm_for_pqa
    summary_llm_source = ctx.get_parameter_source("summary_llm")
    if summary_llm_source != click.core.ParameterSource.DEFAULT:
        # Explicit CLI --pqa-summary-llm takes precedence
        if summary_llm:
            cmd.extend(["--summary_llm", summary_llm])
            summary_llm_for_pqa = summary_llm
    else:
        summary_llm_default = default_pqa_summary_llm(llm_for_pqa)
        if summary_llm_default:
            cmd.extend(["--summary_llm", summary_llm_default])
            summary_llm_for_pqa = summary_llm_default

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
        enrichment_llm_for_pqa = enrichment_llm_default

    # Ollama can have long cold-start / first-token latency. If the user didn't provide explicit
    # per-provider configs, inject a larger LiteLLM router timeout to avoid spurious 60s timeouts.
    if (
        _is_ollama_model_id(llm_for_pqa)
        or _is_ollama_model_id(summary_llm_for_pqa)
        or _is_ollama_model_id(enrichment_llm_for_pqa)
        or _is_ollama_model_id(embedding_for_pqa)
    ):
        ollama_timeout = default_pqa_ollama_timeout()
        timeout_config = json.dumps({"router_kwargs": {"timeout": ollama_timeout}})

        if _is_ollama_model_id(llm_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args, names={"--llm_config", "--llm-config"}
        ):
            cmd.extend(["--llm_config", timeout_config])
        if _is_ollama_model_id(summary_llm_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args, names={"--summary_llm_config", "--summary-llm-config"}
        ):
            cmd.extend(["--summary_llm_config", timeout_config])
        if _is_ollama_model_id(enrichment_llm_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args,
            names={
                "--parsing.enrichment_llm_config",
                "--parsing.enrichment-llm-config",
            },
        ):
            cmd.extend(["--parsing.enrichment_llm_config", timeout_config])
        if _is_ollama_model_id(embedding_for_pqa) and not paperqa._pqa_has_flag(
            ctx.args, names={"--embedding_config", "--embedding-config"}
        ):
            # For embeddings, LMI uses PassThroughRouter (no model_list), so router_kwargs doesn't help.
            # Also work around a LiteLLM bug where `ollama/...` isn't stripped before calling /api/embed.
            cmd.extend(
                [
                    "--embedding_config",
                    json.dumps({"kwargs": {"custom_llm_provider": "ollama", "timeout": ollama_timeout}}),
                ]
            )

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
        arg in {"--agent.index.concurrency"} or arg.startswith(("--agent.index.concurrency=",)) for arg in ctx.args
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
    index_dir_raw = paperqa._extract_flag_value(
        cmd,
        names={"--agent.index.index_directory", "--agent.index.index-directory"},
    )
    index_name_raw = paperqa._extract_flag_value(
        cmd,
        names={"--agent.index.name"},
    ) or paperqa._extract_flag_value(cmd, names={"--index", "-i"})

    if index_dir_raw and index_name_raw:
        files_path = paperqa._paperqa_index_files_path(index_directory=Path(index_dir_raw), index_name=index_name_raw)
        mapping = paperqa._paperqa_load_index_files_map(files_path) if files_path.exists() else None
        failed_count = sum(1 for v in (mapping or {}).values() if v == "ERROR")
        if failed_count and not retry_failed:
            echo_warning(
                f"PaperQA2 index '{index_name_raw}' has {failed_count} failed document(s) (marked ERROR); "
                "PaperQA2 will not retry them automatically. Re-run with --pqa-retry-failed "
                "or --pqa-rebuild-index to rebuild the whole index."
            )
        if retry_failed:
            cleared, cleared_files = paperqa._paperqa_clear_failed_documents(
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
        click.echo(json.dumps(paperqa._paperqa_ask_evidence_blocks(cmd=cmd, query=query), indent=2))
        return

    root_verbose = bool(ctx.find_root().params.get("verbose"))
    passthrough_verbose = any(arg in {"--verbose", "-v"} for arg in ctx.args)
    raw_output = bool(
        pqa_raw
        or root_verbose
        or passthrough_verbose
        or (verbosity_source != click.core.ParameterSource.DEFAULT and (verbosity or 0) > 0)
    )

    env = os.environ.copy()
    # PaperQA2/LiteLLM can register a lot of callbacks during long runs and emit noisy warnings
    # once it reaches the default cap (30). Allow raising this cap via env, and set a higher
    # default to avoid spam when users haven't configured anything.
    env.setdefault("PQA_LITELLM_MAX_CALLBACKS", "1000")
    env.setdefault("LMI_LITELLM_MAX_CALLBACKS", "1000")
    if _is_ollama_model_id(llm_for_pqa) or _is_ollama_model_id(embedding_for_pqa):
        config._prepare_ollama_env(env)
        err = config._ollama_reachability_error(api_base=env["OLLAMA_API_BASE"])
        if err:
            echo_error(err)
            echo_error("Start Ollama (`ollama serve`) or set OLLAMA_HOST / OLLAMA_API_BASE to a reachable server.")
            raise SystemExit(1)

    # Run pqa while capturing output for crash detection.
    # We merge stderr into stdout so we can preserve ordering.
    proc = subprocess.Popen(
        cmd,
        cwd=config.PAPERS_DIR,
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
        if not raw_output:
            paperqa._pqa_print_filtered_output_on_failure(captured_output=captured_output)

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
                paperqa._paperqa_effective_paper_directory(cmd, base_dir=config.PAPERS_DIR)
                or (config.PAPER_DB / ".pqa_papers").expanduser()
            )
            if paper_dir.exists():
                f = paperqa._paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc=crashing_doc)
                if f is not None:
                    count, _ = paperqa._paperqa_mark_failed_documents(
                        index_directory=Path(index_dir_raw),
                        index_name=index_name_raw,
                        staged_files={str(f)},
                    )
                    if count:
                        # Only remove files from paperpipe's managed staging directory.
                        # Never delete from a user-provided paper directory.
                        managed_staging_dir = (config.PAPER_DB / ".pqa_papers").expanduser()
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
            mapping = paperqa._paperqa_load_index_files_map(
                paperqa._paperqa_index_files_path(index_directory=Path(index_dir_raw), index_name=index_name_raw)
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
        echo_error("PaperQA2 failed. Re-run with --pqa-raw or 'papi -v ask ...' for full output.")
        raise SystemExit(returncode)

    if not raw_output:
        for line in captured_output:
            if not paperqa._pqa_is_noisy_stream_line(line):
                click.echo(line, nl=False)
