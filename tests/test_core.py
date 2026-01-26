from __future__ import annotations

import ast
import json
from importlib import util
from pathlib import Path
from typing import Optional

import pytest

import paperpipe
import paperpipe.core as core


class TestEnsureDb:
    def test_creates_directories(self, temp_db: Path):
        paperpipe.ensure_db()
        assert temp_db.exists()
        assert (temp_db / "papers").exists()
        assert (temp_db / "index.json").exists()

    def test_creates_empty_index(self, temp_db: Path):
        paperpipe.ensure_db()
        index = json.loads((temp_db / "index.json").read_text())
        assert index == {}

    def test_idempotent(self, temp_db: Path):
        paperpipe.ensure_db()
        paperpipe.ensure_db()
        assert temp_db.exists()


class TestPublicApi:
    def test_internal_imports_have_no_cycles(self) -> None:
        modules = [
            "paperpipe.cli",
            "paperpipe.config",
            "paperpipe.core",
            "paperpipe.install",
            "paperpipe.leann",
            "paperpipe.output",
            "paperpipe.paper",
            "paperpipe.paperqa",
            "paperpipe.search",
        ]

        graph = {name: set() for name in modules}
        for name in modules:
            spec = util.find_spec(name)
            assert spec is not None and spec.origin, f"Could not resolve module {name}"
            if not spec.origin.endswith(".py"):
                continue
            tree = ast.parse(Path(spec.origin).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    if node.module:
                        target = f"paperpipe.{node.module.split('.')[0]}"
                        if target in graph:
                            graph[name].add(target)
                    else:
                        for alias in node.names:
                            target = f"paperpipe.{alias.name.split('.')[0]}"
                            if target in graph:
                                graph[name].add(target)

        visiting: set[str] = set()
        visited: set[str] = set()

        def find_cycle(node: str, stack: list[str]) -> Optional[list[str]]:
            visiting.add(node)
            for child in graph[node]:
                if child in visiting:
                    return stack + [child]
                if child not in visited:
                    cycle = find_cycle(child, stack + [child])
                    if cycle:
                        return cycle
            visiting.remove(node)
            visited.add(node)
            return None

        cycle = None
        for node in modules:
            if node not in visited:
                cycle = find_cycle(node, [node])
                if cycle:
                    break

        assert not cycle, f"Import cycle detected: {' -> '.join(cycle or [])}"


class TestIndex:
    def test_load_empty_index(self, temp_db: Path):
        index = paperpipe.load_index()
        assert index == {}

    def test_save_and_load_index(self, temp_db: Path):
        test_data = {"paper1": {"title": "Test Paper", "tags": ["ml"]}}
        paperpipe.save_index(test_data)
        loaded = paperpipe.load_index()
        assert loaded == test_data


class TestIsSafePaperName:
    """Tests for _is_safe_paper_name helper."""

    def test_rejects_empty_string(self):
        assert core._is_safe_paper_name("") is False

    def test_rejects_dot(self):
        assert core._is_safe_paper_name(".") is False

    def test_rejects_dotdot(self):
        assert core._is_safe_paper_name("..") is False

    def test_rejects_forward_slash(self):
        assert core._is_safe_paper_name("foo/bar") is False

    def test_rejects_backslash(self):
        assert core._is_safe_paper_name("foo\\bar") is False

    def test_rejects_absolute_path(self):
        assert core._is_safe_paper_name("/etc/passwd") is False

    def test_accepts_valid_name(self):
        assert core._is_safe_paper_name("nerf-2020") is True

    def test_accepts_name_with_dots(self):
        assert core._is_safe_paper_name("paper.v2") is True


class TestResolvePaperNameFromRef:
    """Tests for _resolve_paper_name_from_ref helper."""

    def test_returns_error_for_empty_input(self, temp_db: Path):
        name, error = core._resolve_paper_name_from_ref("", {})
        assert name is None
        assert "Missing" in error

    def test_finds_paper_in_index(self, temp_db: Path):
        index = {"my-paper": {"arxiv_id": "1234.5678", "title": "Test"}}
        name, error = core._resolve_paper_name_from_ref("my-paper", index)
        assert name == "my-paper"
        assert error == ""

    def test_finds_paper_on_disk_not_in_index(self, temp_db: Path):
        # Paper exists on disk but not in index
        paper_dir = temp_db / "papers" / "disk-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text('{"arxiv_id": "1234.5678"}')

        name, error = core._resolve_paper_name_from_ref("disk-paper", {})
        assert name == "disk-paper"
        assert error == ""

    def test_returns_error_for_invalid_arxiv_id(self, temp_db: Path):
        name, error = core._resolve_paper_name_from_ref("not-a-paper-or-id", {})
        assert name is None
        assert "not found" in error.lower()

    def test_fallback_scan_finds_paper_by_arxiv_id(self, temp_db: Path):
        # Paper on disk with arxiv_id, but not in index
        paper_dir = temp_db / "papers" / "some-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        # Empty index, but valid arxiv ID should trigger fallback scan
        name, error = core._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "some-paper"
        assert error == ""

    def test_fallback_scan_reports_multiple_matches(self, temp_db: Path):
        # Two papers with same arxiv_id on disk
        for pname in ["paper-a", "paper-b"]:
            paper_dir = temp_db / "papers" / pname
            paper_dir.mkdir(parents=True)
            (paper_dir / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = core._resolve_paper_name_from_ref("2301.00001", {})
        assert name is None
        assert "Multiple papers match" in error

    def test_fallback_scan_skips_non_directories(self, temp_db: Path):
        # Create a file (not directory) in papers dir
        (temp_db / "papers" / "not-a-dir.txt").write_text("just a file")
        # And a valid paper
        paper_dir = temp_db / "papers" / "real-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = core._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "real-paper"

    def test_fallback_scan_skips_invalid_json(self, temp_db: Path):
        # Paper with invalid JSON
        bad_paper = temp_db / "papers" / "bad-paper"
        bad_paper.mkdir(parents=True)
        (bad_paper / "meta.json").write_text("not valid json {{{")
        # And a valid paper
        good_paper = temp_db / "papers" / "good-paper"
        good_paper.mkdir(parents=True)
        (good_paper / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = core._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "good-paper"

    def test_fallback_scan_skips_missing_meta(self, temp_db: Path):
        # Paper directory without meta.json
        no_meta = temp_db / "papers" / "no-meta"
        no_meta.mkdir(parents=True)
        # And a valid paper
        valid = temp_db / "papers" / "valid"
        valid.mkdir(parents=True)
        (valid / "meta.json").write_text('{"arxiv_id": "2301.00001"}')

        name, error = core._resolve_paper_name_from_ref("2301.00001", {})
        assert name == "valid"

    def test_fallback_scan_returns_not_found(self, temp_db: Path):
        # Papers exist but none match the arxiv_id
        paper = temp_db / "papers" / "other-paper"
        paper.mkdir(parents=True)
        (paper / "meta.json").write_text('{"arxiv_id": "9999.99999"}')

        name, error = core._resolve_paper_name_from_ref("2301.00001", {})
        assert name is None
        assert "not found" in error.lower()

    # Fuzzy matching tests

    def test_fuzzy_match_normalized_hyphen(self, temp_db: Path):
        """Normalized match: ifnet -> if-net"""
        index = {"if-net": {"title": "IF-Net"}}
        name, error = core._resolve_paper_name_from_ref("ifnet", index)
        assert name == "if-net"
        assert error == ""

    def test_fuzzy_match_normalized_case(self, temp_db: Path):
        """Normalized match: nerf -> NeRF"""
        index = {"NeRF": {"title": "NeRF"}}
        name, error = core._resolve_paper_name_from_ref("nerf", index)
        assert name == "NeRF"
        assert error == ""

    def test_fuzzy_match_high_confidence_auto(self, temp_db: Path):
        """High similarity (>= 0.85) single match auto-selects"""
        index = {"nerf-2020": {"title": "NeRF"}}
        # nerf2020 vs nerf-2020 has high similarity after normalization
        name, error = core._resolve_paper_name_from_ref("nerf2020", index)
        assert name == "nerf-2020"
        assert error == ""

    def test_fuzzy_match_multiple_non_interactive(self, temp_db: Path, monkeypatch):
        """Multiple fuzzy matches in non-interactive mode returns error with suggestions"""
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        # Use similar-length names so they all match with get_close_matches
        index = {"nerfa": {}, "nerfb": {}, "nerfc": {}}
        name, error = core._resolve_paper_name_from_ref("nerf", index)
        assert name is None
        assert "Did you mean" in error or "Multiple papers match" in error

    def test_fuzzy_match_interactive_selection(self, temp_db: Path, monkeypatch):
        """Interactive selection from multiple fuzzy matches"""
        import sys

        import click

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: 1)

        # Use similar-length names so they all match with get_close_matches
        index = {"nerfa": {}, "nerfb": {}}
        name, error = core._resolve_paper_name_from_ref("nerf", index)
        assert name is not None
        assert error == ""

    def test_exact_match_takes_priority(self, temp_db: Path):
        """Exact match should be preferred over fuzzy"""
        index = {"nerf": {}, "nerf-2020": {}}
        name, error = core._resolve_paper_name_from_ref("nerf", index)
        assert name == "nerf"
        assert error == ""


class TestCategoriesToTags:
    def test_known_categories(self):
        tags = paperpipe.categories_to_tags(["cs.CV", "cs.LG"])
        assert "computer-vision" in tags
        assert "machine-learning" in tags

    def test_unknown_category(self):
        tags = paperpipe.categories_to_tags(["cs.XX"])
        assert "cs-xx" in tags

    def test_deduplication(self):
        tags = paperpipe.categories_to_tags(["cs.LG", "stat.ML"])
        assert tags.count("machine-learning") == 1

    def test_empty_input(self):
        tags = paperpipe.categories_to_tags([])
        assert tags == []


class TestNormalizeArxivId:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1706.03762", "1706.03762"),
            ("1706.03762v5", "1706.03762v5"),
            ("https://arxiv.org/abs/1706.03762", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762v5.pdf", "1706.03762v5"),
            ("http://arxiv.org/abs/1706.03762", "1706.03762"),
            ("arXiv:1706.03762", "1706.03762"),
            ("abs/1706.03762", "1706.03762"),
            ("hep-th/9901001", "hep-th/9901001"),
            ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
            ("https://arxiv.org/pdf/hep-th/9901001.pdf", "hep-th/9901001"),
        ],
    )
    def test_normalizes_common_inputs(self, raw: str, expected: str):
        assert paperpipe.normalize_arxiv_id(raw) == expected

    def test_rejects_unparseable(self):
        with pytest.raises(ValueError):
            paperpipe.normalize_arxiv_id("not-an-arxiv-id")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="missing"):
            paperpipe.normalize_arxiv_id("")

    def test_handles_plain_id_with_pdf_extension(self):
        # Plain ID ending with .pdf (not a URL)
        assert paperpipe.normalize_arxiv_id("1706.03762.pdf") == "1706.03762"

    def test_extracts_embedded_arxiv_id(self):
        # Arxiv ID embedded in surrounding text
        result = paperpipe.normalize_arxiv_id("See paper 1706.03762 for details")
        assert result == "1706.03762"
