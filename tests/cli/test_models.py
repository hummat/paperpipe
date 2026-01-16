"""Tests for paperpipe/cli/models.py (models command)."""

from __future__ import annotations

import json
import sys
import types

from click.testing import CliRunner

from .conftest import cli_mod


class TestModelsCommand:
    def test_models_reports_ok_and_fail(self, monkeypatch):
        calls = []

        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            calls.append(("completion", model))
            if model == "bad-model":
                raise RuntimeError("missing key")
            return {"ok": True}

        def embedding(*, model, input, timeout):
            calls.append(("embedding", model))
            if model == "bad-embed":
                raise RuntimeError("not authorized")
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "models",
                "--kind",
                "completion",
                "--model",
                "ok-model",
                "--model",
                "bad-model",
            ],
        )

        assert result.exit_code == 0
        assert "OK" in result.output
        assert "FAIL" in result.output
        assert "ok-model" in result.output
        assert "bad-model" in result.output
        assert ("completion", "ok-model") in calls
        assert ("completion", "bad-model") in calls

    def test_models_json_output(self, monkeypatch):
        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            return {"ok": True}

        def embedding(*, model, input, timeout):
            raise RuntimeError("no key")

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        runner = CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            [
                "models",
                "--kind",
                "completion",
                "--kind",
                "embedding",
                "--model",
                "ok-model",
                "--model",
                "bad-embed",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert any(r["ok"] is True and r["model"] == "ok-model" and r["kind"] == "completion" for r in payload)
        assert any(r["ok"] is False and r["model"] == "bad-embed" and r["kind"] == "embedding" for r in payload)

    def test_models_requires_litellm(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace())
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["models", "--kind", "completion", "--model", "ok-model"])
        assert result.exit_code != 0
        assert "LiteLLM is required" in result.output

    def test_models_accepts_presets(self, monkeypatch):
        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            return {"ok": True}

        def embedding(*, model, input, timeout, **kwargs):
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["models", "--preset", "latest", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "latest", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "--preset", "last-gen", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "last-gen", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "--preset", "all", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        result = runner.invoke(cli_mod.cli, ["models", "all", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

    def test_models_default_probes_only_configured_combo(self, monkeypatch):
        calls = []

        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            calls.append(("completion", model))
            return {"ok": True}

        def embedding(*, model, input, timeout, **kwargs):
            calls.append(("embedding", model))
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        # Pretend all provider keys are set; the default should still probe only the
        # configured paperpipe model+embedding combo (mapped to "latest" for that provider).
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("VOYAGE_API_KEY", "x")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["models", "--json"])
        assert result.exit_code == 0

        # Default run probes one "latest" completion + embedding per configured provider.
        assert ("completion", "gpt-5.2") in calls
        assert ("completion", "gemini/gemini-3-flash-preview") in calls
        assert ("completion", "claude-sonnet-4-5") in calls
        assert ("embedding", "text-embedding-3-large") in calls
        assert ("embedding", "gemini/gemini-embedding-001") in calls
        assert ("embedding", "voyage/voyage-3-large") in calls

    def test_models_positional_preset_is_explicit(self, monkeypatch):
        calls = []

        def completion(*, model, messages, max_tokens, timeout, **kwargs):
            calls.append(("completion", model))
            return {"ok": True}

        def embedding(*, model, input, timeout, **kwargs):
            calls.append(("embedding", model))
            return {"data": []}

        fake_litellm = types.SimpleNamespace(completion=completion, embedding=embedding)
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("VOYAGE_API_KEY", "x")

        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["models", "latest", "--kind", "completion", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)

        # Explicit "latest" probes the full preset list (includes Pro/Opus).
        assert ("completion", "gemini/gemini-3-pro-preview") in calls
        assert ("completion", "claude-opus-4-5") in calls
