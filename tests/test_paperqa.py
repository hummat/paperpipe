from __future__ import annotations

import pytest

import paperpipe
import paperpipe.config as config
import paperpipe.paperqa as paperqa


class TestPqaOutputFiltering:
    def test_pqa_failure_output_falls_back_to_raw_when_all_noisy(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Ensure we print something useful even if every line is filtered as "noisy".
        captured_output = [
            "/home/user/site-packages/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:\n",
            "  PydanticSerializationUnexpectedValue(Expected 10 fields but got 7: ...)\n",
            "  return self.__pydantic_serializer__.to_python(\n",
        ]
        paperqa._pqa_print_filtered_output_on_failure(captured_output=captured_output, max_lines=200)
        out = capsys.readouterr().out
        assert "raw PaperQA2 output" in out
        assert "Pydantic serializer warnings" in out

    def test_ollama_reachability_error_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(config, "urlopen", boom)
        err = config._ollama_reachability_error(api_base="http://localhost:11434")
        assert err is not None and "not reachable" in err


class TestPqaIndexNaming:
    def test_pqa_index_name_for_embedding_is_stable_and_safe(self):
        assert paperpipe.pqa_index_name_for_embedding("text-embedding-3-small") == "paperpipe_text-embedding-3-small"
        assert paperpipe.pqa_index_name_for_embedding("foo/bar:baz") == "paperpipe_foo_bar_baz"
        assert paperpipe.pqa_index_name_for_embedding("") == "paperpipe_default"


class TestProbeHint:
    """Tests for _probe_hint helper."""

    def test_hint_for_gpt52_not_supported(self):
        hint = paperqa._probe_hint("completion", "gpt-5.2", "model_not_supported error")
        assert hint is not None
        assert "gpt-5.1" in hint

    def test_hint_for_embedding_not_supported(self):
        hint = paperqa._probe_hint("embedding", "text-embedding-3-large", "not supported")
        assert hint is not None
        assert "text-embedding-3-small" in hint

    def test_hint_for_claude_35_retired(self):
        hint = paperqa._probe_hint("completion", "claude-3-5-sonnet", "not_found")
        assert hint is not None
        assert "claude-sonnet-4-5" in hint

    def test_hint_for_voyage_completion(self):
        hint = paperqa._probe_hint("completion", "voyage/voyage-3", "does not support parameters")
        assert hint is not None
        assert "embedding" in hint

    def test_no_hint_for_unknown_error(self):
        hint = paperqa._probe_hint("completion", "some-model", "random error")
        assert hint is None


class TestPillowAvailable:
    """Tests for _pillow_available helper."""

    def test_returns_bool(self):
        # Just verify it returns a boolean without crashing
        result = paperqa._pillow_available()
        assert isinstance(result, bool)


class TestFirstLine:
    """Tests for _first_line helper."""

    def test_extracts_first_line(self):
        assert paperqa._first_line("first\nsecond\nthird") == "first"

    def test_strips_whitespace(self):
        assert paperqa._first_line("  hello  \nworld") == "hello"

    def test_single_line(self):
        assert paperqa._first_line("just one line") == "just one line"
