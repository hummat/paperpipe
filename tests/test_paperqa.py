from __future__ import annotations

import pickle
import zlib

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


class TestSafeUnpickle:
    """Tests for _safe_unpickle (Fix #1: restricted pickle deserialization)."""

    def test_allows_valid_dict(self):
        data = pickle.dumps({"key": "value", "num": 42})
        result = paperqa._safe_unpickle(data)
        assert result == {"key": "value", "num": 42}

    def test_allows_empty_dict(self):
        data = pickle.dumps({})
        result = paperqa._safe_unpickle(data)
        assert result == {}

    def test_rejects_os_system(self):
        # Build a payload using reduce that references os.system
        import os as _os

        class _OsSystem:
            def __reduce__(self):
                return (_os.system, ("echo pwned",))

        payload = pickle.dumps(_OsSystem())
        with pytest.raises(pickle.UnpicklingError, match="Restricted"):
            paperqa._safe_unpickle(payload)

    def test_rejects_subprocess(self):
        # Build a pickle payload that tries to import subprocess.call
        class Evil:
            def __reduce__(self):
                import subprocess

                return (subprocess.call, (["echo", "pwned"],))

        payload = pickle.dumps(Evil())
        with pytest.raises(pickle.UnpicklingError, match="Restricted"):
            paperqa._safe_unpickle(payload)

    def test_integration_with_load_index_files_map(self, tmp_path):
        """Integration: _paperqa_load_index_files_map uses safe unpickle."""
        mapping = {"file1.pdf": "ok", "file2.pdf": "ERROR"}
        compressed = zlib.compress(pickle.dumps(mapping))
        files_zip = tmp_path / "test_index" / "files.zip"
        files_zip.parent.mkdir(parents=True)
        files_zip.write_bytes(compressed)
        result = paperqa._paperqa_load_index_files_map(files_zip)
        assert result == mapping


class TestValidateIndexName:
    """Tests for _validate_index_name (Fix #2: path traversal prevention)."""

    def test_valid_name(self):
        assert paperqa._validate_index_name("paperpipe_text-embedding-3-small") == "paperpipe_text-embedding-3-small"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            paperqa._validate_index_name("")

    def test_rejects_dot_dot(self):
        with pytest.raises(ValueError, match="\\.\\."):
            paperqa._validate_index_name("..secret")

    def test_rejects_slash(self):
        with pytest.raises(ValueError, match="path separator"):
            paperqa._validate_index_name("foo/bar")

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="path separator"):
            paperqa._validate_index_name("foo\\bar")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            paperqa._validate_index_name("foo\x00bar")


class TestFindCrashingFileSecurity:
    """Tests for _paperqa_find_crashing_file boundary checks (Fix #3)."""

    def test_absolute_outside_rejected(self, tmp_path):
        paper_dir = tmp_path / "papers"
        paper_dir.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        result = paperqa._paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc=str(outside))
        assert result is None

    def test_absolute_inside_allowed(self, tmp_path):
        paper_dir = tmp_path / "papers"
        paper_dir.mkdir()
        inside = paper_dir / "paper.pdf"
        inside.write_text("pdf")
        result = paperqa._paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc=str(inside))
        assert result is not None

    def test_dotdot_traversal_rejected(self, tmp_path):
        paper_dir = tmp_path / "papers"
        paper_dir.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        result = paperqa._paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc="../secret.txt")
        assert result is None

    def test_relative_inside_allowed(self, tmp_path):
        paper_dir = tmp_path / "papers"
        paper_dir.mkdir()
        (paper_dir / "test.pdf").write_text("pdf")
        result = paperqa._paperqa_find_crashing_file(paper_directory=paper_dir, crashing_doc="test.pdf")
        assert result is not None


class TestSafeZlibDecompress:
    """Tests for _safe_zlib_decompress (Fix #5a: bounded decompression)."""

    def test_normal_data(self):
        original = b"hello world" * 100
        compressed = zlib.compress(original)
        result = paperqa._safe_zlib_decompress(compressed)
        assert result == original

    def test_oversized_rejection(self):
        # Create data larger than a 1KB limit
        original = b"x" * 2048
        compressed = zlib.compress(original)
        with pytest.raises(ValueError, match="exceeds"):
            paperqa._safe_zlib_decompress(compressed, max_size=1024)

    def test_empty_data(self):
        compressed = zlib.compress(b"")
        result = paperqa._safe_zlib_decompress(compressed)
        assert result == b""
