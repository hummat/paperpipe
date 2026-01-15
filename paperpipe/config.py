"""Configuration and defaults."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, MutableMapping, Optional
from urllib.request import Request, urlopen

import click

# TOML config support (stdlib on 3.11+, tomli on 3.10)
try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]  # noqa: F401

from .output import debug, echo_warning


# Configuration
def _paper_db_root() -> Path:
    configured = os.environ.get("PAPER_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".paperpipe"


PAPER_DB = _paper_db_root()
PAPERS_DIR = PAPER_DB / "papers"
INDEX_FILE = PAPER_DB / "index.json"

DEFAULT_LLM_MODEL_FALLBACK = "gemini/gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL_FALLBACK = "gemini/gemini-embedding-001"
DEFAULT_LLM_TEMPERATURE_FALLBACK = 0.3

DEFAULT_LEANN_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LEANN_EMBEDDING_MODE = "ollama"
DEFAULT_LEANN_LLM_PROVIDER = "ollama"
DEFAULT_LEANN_LLM_MODEL = "olmo-3:7b"
DEFAULT_LEANN_INDEX_NAME = "papers"
GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


_CONFIG_CACHE: Optional[tuple[Path, Optional[float], dict[str, Any]]] = None
_WARNED_INVALID_SETTINGS: set[str] = set()
_SEARCH_DB_SCHEMA_VERSION = "1"


def _is_ollama_model_id(model_id: Optional[str]) -> bool:
    return bool(model_id) and model_id.strip().lower().startswith("ollama/")


def _normalize_ollama_base_url(raw: str) -> str:
    base = (raw or "").strip()
    if not base:
        return "http://localhost:11434"
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _prepare_ollama_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    raw = (
        env.get("OLLAMA_API_BASE")
        or env.get("OLLAMA_API_BASE_URL")
        or env.get("OLLAMA_BASE_URL")
        or env.get("OLLAMA_HOST")
    )
    base = _normalize_ollama_base_url(raw or "http://localhost:11434")
    env["OLLAMA_API_BASE"] = base
    env["OLLAMA_HOST"] = base
    return env


def _ollama_reachability_error(*, api_base: str, timeout_sec: float = 1.5) -> Optional[str]:
    api_base = _normalize_ollama_base_url(api_base)
    url = f"{api_base}/api/version"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout_sec) as resp:
            if 200 <= int(getattr(resp, "status", 200)) < 400:
                return None
        return f"Ollama returned non-OK status when probing {url!r}."
    except Exception as e:
        msg = str(e).split("\n")[0][:160]
        return f"Ollama not reachable at {api_base!r} ({type(e).__name__}: {msg})."


def _config_path() -> Path:
    configured = os.environ.get("PAPERPIPE_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser()
    return (PAPER_DB / "config.toml").expanduser()


def load_config() -> dict[str, Any]:
    """Load config from <paper_db>/config.toml (or PAPERPIPE_CONFIG_PATH).

    Returns an empty dict if missing or invalid.
    """
    path = _config_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = None

    global _CONFIG_CACHE
    if _CONFIG_CACHE and _CONFIG_CACHE[0] == path and _CONFIG_CACHE[1] == mtime:
        return _CONFIG_CACHE[2]

    if mtime is None:
        cfg: dict[str, Any] = {}
        _CONFIG_CACHE = (path, None, cfg)
        return cfg

    try:
        raw = path.read_bytes()
        cfg = tomllib.loads(raw.decode("utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception as e:
        debug("Failed to parse config.toml (%s) [%s]: %s", str(path), type(e).__name__, str(e))
        cfg = {}

    _CONFIG_CACHE = (path, mtime, cfg)
    return cfg


def _config_get(cfg: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def _setting_str(*, env: str, keys: tuple[str, ...], default: str) -> str:
    val = os.environ.get(env)
    if val is not None and val.strip():
        return val.strip()
    cfg = load_config()
    raw = _config_get(cfg, keys)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default


def _setting_float(*, env: str, keys: tuple[str, ...], default: float) -> float:
    def warn_once(source: str, raw_value: object) -> None:
        key = f"{source}:{env}:{'.'.join(keys)}"
        if key in _WARNED_INVALID_SETTINGS:
            return
        _WARNED_INVALID_SETTINGS.add(key)
        setting_label = env
        if keys:
            setting_label = f"{env} ({'.'.join(keys)})"
        echo_warning(f"Ignoring invalid {source} value for {setting_label}: {raw_value!r}")

    val = os.environ.get(env)
    if val is not None and val.strip():
        try:
            return float(val.strip())
        except Exception:
            warn_once("env", val)
            return default
    cfg = load_config()
    raw = _config_get(cfg, keys)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except Exception:
            warn_once("config", raw)
            return default
    return default


def default_llm_model() -> str:
    return _setting_str(env="PAPERPIPE_LLM_MODEL", keys=("llm", "model"), default=DEFAULT_LLM_MODEL_FALLBACK)


def default_embedding_model() -> str:
    return _setting_str(
        env="PAPERPIPE_EMBEDDING_MODEL",
        keys=("embedding", "model"),
        default=DEFAULT_EMBEDDING_MODEL_FALLBACK,
    )


def default_llm_temperature() -> float:
    return _setting_float(
        env="PAPERPIPE_LLM_TEMPERATURE",
        keys=("llm", "temperature"),
        default=DEFAULT_LLM_TEMPERATURE_FALLBACK,
    )


def default_pqa_settings_name() -> str:
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "settings"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "default"


def default_pqa_llm_model() -> str:
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "llm"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default_llm_model()


def default_pqa_embedding_model() -> str:
    configured = os.environ.get("PAPERQA_EMBEDDING")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "embedding"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default_embedding_model()


def default_pqa_index_dir() -> Path:
    configured = os.environ.get("PAPERPIPE_PQA_INDEX_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "index_dir"))
    if isinstance(raw, str) and raw.strip():
        return Path(raw.strip()).expanduser()
    return (PAPER_DB / ".pqa_index").expanduser()


def pqa_index_name_for_embedding(embedding_model: str) -> str:
    """Return the stable PaperQA2 index name used by paperpipe for a given embedding model."""
    safe_name = (embedding_model or "").replace("/", "_").replace(":", "_")
    safe_name = safe_name.strip() or "default"
    return f"paperpipe_{safe_name}"


def embedding_model_from_index_name(index_name: str) -> str | None:
    """
    Reverse-engineer embedding model from paperpipe index name.

    Returns None if name doesn't follow paperpipe naming convention.
    Best-effort for ambiguous cases (multiple underscores).

    Examples:
        paperpipe_voyage_voyage-3.5 → voyage/voyage-3.5
        paperpipe_openai_text-embedding-3-small → openai/text-embedding-3-small
        paperpipe_gemini_gemini-embedding-001 → gemini/gemini-embedding-001
        paperpipe_default → ""
        custom_name → None
    """
    if not index_name.startswith("paperpipe_"):
        return None

    suffix = index_name[len("paperpipe_") :]

    if suffix == "default":
        return ""

    # Try known provider patterns (with slashes in original model IDs)
    # These are common LiteLLM provider prefixes
    known_providers = [
        "openai/",
        "voyage/",
        "ollama/",
        "openrouter/",
        "anthropic/",
        "google/",
        "gemini/",
        "cohere/",
        "huggingface/",
    ]

    for provider in known_providers:
        safe_provider = provider.replace("/", "_")
        if suffix.startswith(safe_provider):
            # Found a match - reconstruct by replacing first underscore with slash
            # e.g., "voyage_voyage-3.5" → "voyage/voyage-3.5"
            return suffix.replace("_", "/", 1)

    # Generic fallback for models without provider prefix
    # This is ambiguous (foo_bar_baz could be foo/bar/baz or foo_bar/baz)
    # but better than nothing for backward compatibility
    # Most embedding models follow provider/model pattern, so replace first underscore
    if "_" in suffix:
        return suffix.replace("_", "/", 1)

    # No underscores left, return as-is
    return suffix


def default_pqa_ollama_timeout() -> float:
    """
    Per-request timeout (seconds) for PaperQA2 LiteLLM router calls when using `ollama/...`.

    Ollama can take a while to load a model / produce first token. LiteLLM's default timeout
    is often too low (commonly 60s), causing spurious failures.
    """

    configured = os.environ.get("PAPERPIPE_PQA_OLLAMA_TIMEOUT")
    if configured and configured.strip():
        try:
            v = float(configured.strip())
            if v > 0:
                return v
            debug(
                "Ignoring PAPERPIPE_PQA_OLLAMA_TIMEOUT=%r (must be > 0); falling back to config/default.",
                configured,
            )
        except ValueError:
            debug(
                "Ignoring PAPERPIPE_PQA_OLLAMA_TIMEOUT=%r (must be a number); falling back to config/default.",
                configured,
            )
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "ollama_timeout"))
    if isinstance(raw, (int, float)) and float(raw) > 0:
        return float(raw)
    return 300.0


def _strip_ollama_prefix(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    model = model.strip()
    if model.lower().startswith("ollama/"):
        return model.split("/", 1)[1].strip() or None
    return model or None


def default_pqa_summary_llm(fallback: Optional[str]) -> Optional[str]:
    configured = os.environ.get("PAPERPIPE_PQA_SUMMARY_LLM")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "summary_llm"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def default_pqa_enrichment_llm(fallback: Optional[str]) -> Optional[str]:
    configured = os.environ.get("PAPERPIPE_PQA_ENRICHMENT_LLM")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "enrichment_llm"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def default_pqa_temperature() -> Optional[float]:
    configured = os.environ.get("PAPERPIPE_PQA_TEMPERATURE")
    if configured and configured.strip():
        try:
            return float(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "temperature"))
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def default_pqa_verbosity() -> Optional[int]:
    configured = os.environ.get("PAPERPIPE_PQA_VERBOSITY")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "verbosity"))
    if isinstance(raw, int):
        return raw
    return None


def default_pqa_answer_length() -> Optional[str]:
    configured = os.environ.get("PAPERPIPE_PQA_ANSWER_LENGTH")
    if configured and configured.strip():
        return configured.strip()
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "answer_length"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def default_pqa_evidence_k() -> Optional[int]:
    configured = os.environ.get("PAPERPIPE_PQA_EVIDENCE_K")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "evidence_k"))
    if isinstance(raw, int):
        return raw
    return None


def default_pqa_max_sources() -> Optional[int]:
    configured = os.environ.get("PAPERPIPE_PQA_MAX_SOURCES")
    if configured and configured.strip():
        try:
            return int(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "max_sources"))
    if isinstance(raw, int):
        return raw
    return None


def default_pqa_timeout() -> Optional[float]:
    configured = os.environ.get("PAPERPIPE_PQA_TIMEOUT")
    if configured and configured.strip():
        try:
            return float(configured.strip())
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "timeout"))
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def default_pqa_concurrency() -> int:
    configured = os.environ.get("PAPERPIPE_PQA_CONCURRENCY")
    if configured and configured.strip():
        try:
            return max(1, int(configured.strip()))
        except ValueError:
            pass
    cfg = load_config()
    raw = _config_get(cfg, ("paperqa", "concurrency"))
    if isinstance(raw, int):
        return max(1, raw)
    return 1  # Default to 1 for stability


def default_search_mode() -> str:
    """Default `papi search` mode.

    Values:
    - auto: current behavior (use FTS if `search.db` exists; else scan)
    - fts: prefer FTS (still falls back to scan if `search.db` missing)
    - scan: force in-memory scan
    - hybrid: FTS + grep signal (falls back to non-hybrid if prerequisites missing)
    """
    mode = _setting_str(env="PAPERPIPE_SEARCH_MODE", keys=("search", "mode"), default="auto")
    mode = mode.strip().lower()
    if mode in {"auto", "fts", "scan", "hybrid"}:
        return mode
    return "auto"


def _gemini_api_key() -> Optional[str]:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _split_model_id(model_id: str) -> tuple[Optional[str], str]:
    model_id = (model_id or "").strip()
    if not model_id:
        return None, ""
    if "/" not in model_id:
        return None, model_id
    prefix, rest = model_id.split("/", 1)
    prefix = prefix.strip().lower()
    rest = rest.strip()
    if not prefix or not rest:
        return None, model_id
    return prefix, rest


def _infer_leann_llm_provider_from_litellm_id(model_id: str) -> Optional[str]:
    provider, model = _split_model_id(model_id)
    if provider == "gemini":
        # Gemini is supported by LEANN via OpenAI-compatible endpoint.
        return "openai"
    if provider in {"ollama", "openai"}:
        return provider
    if provider is not None:
        return None
    if model.startswith("gpt-") or model.startswith("text-embedding-"):
        return "openai"
    return None


def _infer_leann_embedding_mode_from_litellm_id(model_id: str) -> Optional[str]:
    provider, model = _split_model_id(model_id)
    if provider in {"ollama", "openai"}:
        return provider
    if provider is not None:
        # Gemini embeddings via OpenAI-compat currently break with LEANN CLI due to batch-size limits.
        return None
    if model.startswith("text-embedding-"):
        return "openai"
    return None


def _default_leann_llm_provider_fallback() -> str:
    model_id = default_llm_model()
    inferred = _infer_leann_llm_provider_from_litellm_id(model_id)
    if inferred is None:
        debug(
            "Could not infer LEANN LLM provider from default_llm_model=%r; falling back to %r.",
            model_id,
            DEFAULT_LEANN_LLM_PROVIDER,
        )
    return inferred or DEFAULT_LEANN_LLM_PROVIDER


def _default_leann_llm_model_fallback() -> str:
    model_id = default_llm_model()
    _, model = _split_model_id(model_id)
    if _infer_leann_llm_provider_from_litellm_id(model_id) is not None:
        return model
    debug(
        "Could not infer LEANN LLM model from default_llm_model=%r; falling back to %r.",
        model_id,
        DEFAULT_LEANN_LLM_MODEL,
    )
    return DEFAULT_LEANN_LLM_MODEL


def _default_leann_embedding_mode_fallback() -> str:
    model_id = default_embedding_model()
    inferred = _infer_leann_embedding_mode_from_litellm_id(model_id)
    if inferred is None:
        debug(
            "Could not infer LEANN embedding mode from default_embedding_model=%r; falling back to %r.",
            model_id,
            DEFAULT_LEANN_EMBEDDING_MODE,
        )
    return inferred or DEFAULT_LEANN_EMBEDDING_MODE


def _default_leann_embedding_model_fallback() -> str:
    model_id = default_embedding_model()
    _, model = _split_model_id(model_id)
    if _infer_leann_embedding_mode_from_litellm_id(model_id) is not None:
        return model
    debug(
        "Could not infer LEANN embedding model from default_embedding_model=%r; falling back to %r.",
        model_id,
        DEFAULT_LEANN_EMBEDDING_MODEL,
    )
    return DEFAULT_LEANN_EMBEDDING_MODEL


def default_leann_embedding_model() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_EMBEDDING_MODEL",
        keys=("leann", "embedding_model"),
        default=_default_leann_embedding_model_fallback(),
    )


def default_leann_embedding_mode() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_EMBEDDING_MODE",
        keys=("leann", "embedding_mode"),
        default=_default_leann_embedding_mode_fallback(),
    )


def default_leann_llm_provider() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_LLM_PROVIDER",
        keys=("leann", "llm_provider"),
        default=_default_leann_llm_provider_fallback(),
    )


def default_leann_llm_model() -> str:
    return _setting_str(
        env="PAPERPIPE_LEANN_LLM_MODEL",
        keys=("leann", "llm_model"),
        default=_default_leann_llm_model_fallback(),
    )


def leann_index_by_embedding() -> bool:
    """Whether to derive the default LEANN index name from the embedding model/mode."""
    configured = (os.environ.get("PAPERPIPE_LEANN_INDEX_BY_EMBEDDING") or "").strip().lower()
    if configured in {"1", "true", "yes", "on"}:
        return True
    if configured in {"0", "false", "no", "off"}:
        return False
    cfg = load_config()
    raw = _config_get(cfg, ("leann", "index_by_embedding"))
    if isinstance(raw, bool):
        return raw
    return True


def leann_index_name_for_embedding(*, embedding_mode: str, embedding_model: str) -> str:
    mode = (embedding_mode or "").strip().lower()
    model = (embedding_model or "").strip()
    safe_mode = mode.replace("/", "_").replace(":", "_").strip() or "default"
    safe_model = model.replace("/", "_").replace(":", "_").strip() or "default"
    return f"{DEFAULT_LEANN_INDEX_NAME}_{safe_mode}_{safe_model}"


def default_leann_index_name(*, embedding_mode: Optional[str] = None, embedding_model: Optional[str] = None) -> str:
    if not leann_index_by_embedding():
        return DEFAULT_LEANN_INDEX_NAME
    effective_mode = (embedding_mode or "").strip() or default_leann_embedding_mode()
    effective_model = (embedding_model or "").strip() or default_leann_embedding_model()
    return leann_index_name_for_embedding(embedding_mode=effective_mode, embedding_model=effective_model)


def _effective_leann_index_name(
    *,
    ctx: click.Context,
    param_name: str,
    raw_index_name: str,
    embedding_mode: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> str:
    # Honor explicit index overrides, even if they match the default string.
    if ctx.get_parameter_source(param_name) != click.core.ParameterSource.DEFAULT:
        name = (raw_index_name or "").strip()
        return name or DEFAULT_LEANN_INDEX_NAME

    name = (raw_index_name or "").strip() or DEFAULT_LEANN_INDEX_NAME
    if name != DEFAULT_LEANN_INDEX_NAME:
        return name

    return default_leann_index_name(embedding_mode=embedding_mode, embedding_model=embedding_model)


def tag_aliases() -> dict[str, str]:
    cfg = load_config()
    raw = _config_get(cfg, ("tags", "aliases"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        k_norm = k.strip().lower()
        v_norm = v.strip().lower()
        if k_norm and v_norm:
            out[k_norm] = v_norm
    return out


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower().replace(" ", "-")
    t = re.sub(r"[^a-z0-9-]", "", t).strip("-")
    if not t:
        return ""
    aliases = tag_aliases()
    return aliases.get(t, t)


def normalize_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for t in tags:
        n = normalize_tag(t)
        if n:
            out.append(n)
    # Preserve a stable order for UX and deterministic tests
    return sorted(set(out))


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
