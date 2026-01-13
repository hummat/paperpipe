from __future__ import annotations

from pathlib import Path

import pytest

import paperpipe
import paperpipe.config as config


class TestConfigToml:
    def test_config_toml_sets_defaults(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PAPERPIPE_LLM_MODEL", raising=False)
        monkeypatch.delenv("PAPERPIPE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("PAPERPIPE_LLM_TEMPERATURE", raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "gpt-4o-mini"',
                    "temperature = 0.42",
                    "",
                    "[embedding]",
                    'model = "text-embedding-3-small"',
                    "",
                    "[tags.aliases]",
                    'cv = "computer-vision"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        assert paperpipe.default_llm_model() == "gpt-4o-mini"
        assert paperpipe.default_embedding_model() == "text-embedding-3-small"
        assert paperpipe.default_llm_temperature() == 0.42
        assert paperpipe.normalize_tag("cv") == "computer-vision"


class TestConfigPrecedence:
    def test_env_overrides_config(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "config-model"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_LLM_MODEL", "env-model")
        assert paperpipe.default_llm_model() == "env-model"

    def test_pqa_config_from_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_PQA_TEMPERATURE",
            "PAPERPIPE_PQA_VERBOSITY",
            "PAPERPIPE_PQA_ANSWER_LENGTH",
            "PAPERPIPE_PQA_EVIDENCE_K",
            "PAPERPIPE_PQA_MAX_SOURCES",
            "PAPERPIPE_PQA_TIMEOUT",
            "PAPERPIPE_PQA_CONCURRENCY",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[paperqa]",
                    "temperature = 0.5",
                    "verbosity = 2",
                    'answer_length = "about 100 words"',
                    "evidence_k = 15",
                    "max_sources = 8",
                    "timeout = 300.0",
                    "concurrency = 4",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        assert paperpipe.default_pqa_temperature() == 0.5
        assert paperpipe.default_pqa_verbosity() == 2
        assert paperpipe.default_pqa_answer_length() == "about 100 words"
        assert paperpipe.default_pqa_evidence_k() == 15
        assert paperpipe.default_pqa_max_sources() == 8
        assert paperpipe.default_pqa_timeout() == 300.0
        assert paperpipe.default_pqa_concurrency() == 4

    def test_leann_config_from_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_LEANN_LLM_PROVIDER",
            "PAPERPIPE_LEANN_LLM_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODE",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[leann]",
                    'llm_provider = "ollama"',
                    'llm_model = "qwen3:8b"',
                    'embedding_model = "nomic-embed-text"',
                    'embedding_mode = "ollama"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        assert paperpipe.default_leann_llm_provider() == "ollama"
        assert paperpipe.default_leann_llm_model() == "qwen3:8b"
        assert paperpipe.default_leann_embedding_model() == "nomic-embed-text"
        assert paperpipe.default_leann_embedding_mode() == "ollama"

    def test_leann_config_env_overrides_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[leann]",
                    'llm_model = "config-model"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_LEANN_LLM_MODEL", "env-model")

        assert paperpipe.default_leann_llm_model() == "env-model"

    def test_leann_defaults_derive_from_gemini_models(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_LLM_MODEL",
            "PAPERPIPE_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_LLM_PROVIDER",
            "PAPERPIPE_LEANN_LLM_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODE",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "gemini/gemini-3-flash-preview"',
                    "",
                    "[embedding]",
                    'model = "gemini/gemini-embedding-001"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        # LEANN supports Gemini via OpenAI-compatible endpoint (so provider/mode are "openai").
        assert paperpipe.default_leann_llm_provider() == "openai"
        assert paperpipe.default_leann_llm_model() == "gemini-3-flash-preview"
        # Gemini embeddings via OpenAI-compat currently break with LEANN CLI due to batch-size limits (max 100).
        # Keep LEANN embedding defaults unless explicitly overridden.
        assert paperpipe.default_leann_embedding_mode() == "ollama"
        assert paperpipe.default_leann_embedding_model() == "nomic-embed-text"

    def test_leann_defaults_derive_from_project_models(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_LLM_MODEL",
            "PAPERPIPE_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_LLM_PROVIDER",
            "PAPERPIPE_LEANN_LLM_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODE",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "gpt-4o-mini"',
                    "",
                    "[embedding]",
                    'model = "text-embedding-3-small"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        assert paperpipe.default_leann_llm_provider() == "openai"
        assert paperpipe.default_leann_llm_model() == "gpt-4o-mini"
        assert paperpipe.default_leann_embedding_mode() == "openai"
        assert paperpipe.default_leann_embedding_model() == "text-embedding-3-small"

    def test_leann_defaults_derive_from_ollama_prefixed_models(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_LLM_MODEL",
            "PAPERPIPE_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_LLM_PROVIDER",
            "PAPERPIPE_LEANN_LLM_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODEL",
            "PAPERPIPE_LEANN_EMBEDDING_MODE",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[llm]",
                    'model = "ollama/qwen3:8b"',
                    "",
                    "[embedding]",
                    'model = "ollama/nomic-embed-text"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        assert paperpipe.default_leann_llm_provider() == "ollama"
        assert paperpipe.default_leann_llm_model() == "qwen3:8b"
        assert paperpipe.default_leann_embedding_mode() == "ollama"
        assert paperpipe.default_leann_embedding_model() == "nomic-embed-text"

    def test_pqa_config_env_overrides_toml(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[paperqa]",
                    "temperature = 0.5",
                    "concurrency = 4",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_PQA_TEMPERATURE", "0.9")
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "8")

        assert paperpipe.default_pqa_temperature() == 0.9
        assert paperpipe.default_pqa_concurrency() == 8

    def test_pqa_config_defaults_when_unset(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        for env_var in [
            "PAPERPIPE_PQA_TEMPERATURE",
            "PAPERPIPE_PQA_VERBOSITY",
            "PAPERPIPE_PQA_ANSWER_LENGTH",
            "PAPERPIPE_PQA_EVIDENCE_K",
            "PAPERPIPE_PQA_MAX_SOURCES",
            "PAPERPIPE_PQA_TIMEOUT",
            "PAPERPIPE_PQA_CONCURRENCY",
        ]:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        # These return None when unset (no hardcoded default)
        assert paperpipe.default_pqa_temperature() is None
        assert paperpipe.default_pqa_verbosity() is None
        assert paperpipe.default_pqa_answer_length() is None
        assert paperpipe.default_pqa_evidence_k() is None
        assert paperpipe.default_pqa_max_sources() is None
        assert paperpipe.default_pqa_timeout() is None
        # concurrency defaults to 1
        assert paperpipe.default_pqa_concurrency() == 1

    def test_pqa_config_invalid_env_values_fallback(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        """Invalid env var values should fall back to config/defaults."""
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        # Set invalid (non-numeric) values for numeric env vars
        monkeypatch.setenv("PAPERPIPE_PQA_TEMPERATURE", "not-a-number")
        monkeypatch.setenv("PAPERPIPE_PQA_VERBOSITY", "high")
        monkeypatch.setenv("PAPERPIPE_PQA_EVIDENCE_K", "many")
        monkeypatch.setenv("PAPERPIPE_PQA_MAX_SOURCES", "all")
        monkeypatch.setenv("PAPERPIPE_PQA_TIMEOUT", "forever")
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "max")
        monkeypatch.setenv("PAPERPIPE_PQA_OLLAMA_TIMEOUT", "slow")

        # Should fall back to None (no config) or default
        assert paperpipe.default_pqa_temperature() is None
        assert paperpipe.default_pqa_verbosity() is None
        assert paperpipe.default_pqa_evidence_k() is None
        assert paperpipe.default_pqa_max_sources() is None
        assert paperpipe.default_pqa_timeout() is None
        assert paperpipe.default_pqa_concurrency() == 1  # Falls back to default
        assert paperpipe.default_pqa_ollama_timeout() == 300.0

    def test_pqa_ollama_timeout_env_zero_falls_back(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_PQA_OLLAMA_TIMEOUT", "0")
        assert paperpipe.default_pqa_ollama_timeout() == 300.0

    def test_pqa_concurrency_env_zero_clamps_to_one(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "0")
        assert paperpipe.default_pqa_concurrency() == 1

    def test_leann_index_by_embedding_default_on(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.delenv("PAPERPIPE_LEANN_INDEX_BY_EMBEDDING", raising=False)
        assert paperpipe.default_leann_index_name() == "papers_ollama_nomic-embed-text"

    def test_leann_index_by_embedding_env_off_uses_plain_index(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_LEANN_INDEX_BY_EMBEDDING", "0")
        assert paperpipe.default_leann_index_name() == "papers"

    def test_leann_index_by_embedding_uses_mode_and_model(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        (temp_db / "config.toml").write_text(
            "\n".join(
                [
                    "[leann]",
                    "index_by_embedding = true",
                    'embedding_mode = "openai"',
                    'embedding_model = "text-embedding-3-small"',
                    "",
                ]
            )
        )
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.delenv("PAPERPIPE_LEANN_INDEX_BY_EMBEDDING", raising=False)

        assert paperpipe.default_leann_index_name() == "papers_openai_text-embedding-3-small"

    def test_pqa_config_env_vars_direct(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        """Test env vars are read correctly without config file."""
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERPIPE_PQA_TEMPERATURE", "0.7")
        monkeypatch.setenv("PAPERPIPE_PQA_VERBOSITY", "3")
        monkeypatch.setenv("PAPERPIPE_PQA_ANSWER_LENGTH", "short")
        monkeypatch.setenv("PAPERPIPE_PQA_EVIDENCE_K", "20")
        monkeypatch.setenv("PAPERPIPE_PQA_MAX_SOURCES", "10")
        monkeypatch.setenv("PAPERPIPE_PQA_TIMEOUT", "600.5")
        monkeypatch.setenv("PAPERPIPE_PQA_CONCURRENCY", "2")

        assert paperpipe.default_pqa_temperature() == 0.7
        assert paperpipe.default_pqa_verbosity() == 3
        assert paperpipe.default_pqa_answer_length() == "short"
        assert paperpipe.default_pqa_evidence_k() == 20
        assert paperpipe.default_pqa_max_sources() == 10
        assert paperpipe.default_pqa_timeout() == 600.5
        assert paperpipe.default_pqa_concurrency() == 2


class TestOllamaHelpers:
    def test_strip_ollama_prefix_handles_none_and_whitespace(self) -> None:
        assert config._strip_ollama_prefix(None) is None
        assert config._strip_ollama_prefix("") is None
        assert config._strip_ollama_prefix("   ") is None
        assert config._strip_ollama_prefix("ollama/") is None

    def test_strip_ollama_prefix_strips_prefix(self) -> None:
        assert config._strip_ollama_prefix("ollama/nomic-embed-text") == "nomic-embed-text"
        assert config._strip_ollama_prefix("  OLLAMA/nomic-embed-text  ") == "nomic-embed-text"
        assert config._strip_ollama_prefix("nomic-embed-text") == "nomic-embed-text"

    def test_normalize_ollama_base_url_adds_scheme_and_strips_v1(self) -> None:
        assert config._normalize_ollama_base_url("localhost:11434") == "http://localhost:11434"
        assert config._normalize_ollama_base_url("http://localhost:11434/") == "http://localhost:11434"
        assert config._normalize_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434"

    def test_prepare_ollama_env_sets_both_vars(self) -> None:
        env: dict[str, str] = {}
        config._prepare_ollama_env(env)
        assert env["OLLAMA_API_BASE"] == "http://localhost:11434"
        assert env["OLLAMA_HOST"] == "http://localhost:11434"

        env2: dict[str, str] = {"OLLAMA_HOST": "localhost:11434"}
        config._prepare_ollama_env(env2)
        assert env2["OLLAMA_API_BASE"] == "http://localhost:11434"
        assert env2["OLLAMA_HOST"] == "http://localhost:11434"

    def test_ollama_reachability_error_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(config, "urlopen", lambda *args, **kwargs: _Resp())
        assert config._ollama_reachability_error(api_base="http://localhost:11434") is None


class TestModelIdHelpers:
    def test_split_model_id_edge_cases(self) -> None:
        assert config._split_model_id("") == (None, "")
        assert config._split_model_id("   ") == (None, "")
        assert config._split_model_id("model") == (None, "model")
        assert config._split_model_id("/model") == (None, "/model")
        assert config._split_model_id("provider/") == (None, "provider/")
        assert config._split_model_id("provider/model") == ("provider", "model")
