"""Model probing command."""

from __future__ import annotations

import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Optional

import click

from .. import config, paperqa
from ..config import (
    _is_ollama_model_id,
    default_embedding_model,
    default_llm_model,
)


@click.command()
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
            config._prepare_ollama_env(os.environ)
            err = config._ollama_reachability_error(api_base=os.environ["OLLAMA_API_BASE"])
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

    def probe_group(kind_name: str, candidates: list[str]) -> paperqa._ModelProbeResult:
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                if verbose:
                    probe_one(kind_name, candidate)
                else:
                    with redirect_stdout(null_out), redirect_stderr(null_err):
                        probe_one(kind_name, candidate)
                return paperqa._ModelProbeResult(kind=kind_name, model=candidate, ok=True)
            except Exception as exc:
                last_exc = exc
                continue

        err = paperqa._first_line(str(last_exc)) if last_exc else "Unknown error"
        hint = paperqa._probe_hint(kind=kind_name, model=candidates[0], error_line=err)
        if hint:
            err = f"{err} ({hint})"
        return paperqa._ModelProbeResult(
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
            results: list[paperqa._ModelProbeResult] = []
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

    results: list[paperqa._ModelProbeResult] = []
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
                    results.append(paperqa._ModelProbeResult(kind=k, model=model, ok=True))
                except Exception as exc:
                    err = paperqa._first_line(str(exc))
                    hint = paperqa._probe_hint(kind=k, model=model, error_line=err)
                    if hint:
                        err = f"{err} ({hint})"
                    results.append(
                        paperqa._ModelProbeResult(
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
                    results.append(paperqa._ModelProbeResult(kind=k, model=model, ok=True))
                except Exception as exc:
                    err = paperqa._first_line(str(exc))
                    hint = paperqa._probe_hint(kind=k, model=model, error_line=err)
                    if hint:
                        err = f"{err} ({hint})"
                    results.append(
                        paperqa._ModelProbeResult(
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
