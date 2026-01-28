"""Tests for paper name matching utilities."""

from __future__ import annotations

import pytest

from paperpipe.matching import (
    MatchResult,
    MatchType,
    find_paper_matches,
    get_best_fuzzy_similarity,
    normalize_paper_name,
    select_arxiv_result_interactively,
    select_paper_interactively,
)


class TestNormalizePaperName:
    def test_strips_hyphens(self):
        assert normalize_paper_name("if-net") == "ifnet"

    def test_strips_underscores(self):
        assert normalize_paper_name("my_paper") == "mypaper"

    def test_lowercases(self):
        assert normalize_paper_name("IF-Net") == "ifnet"

    def test_combined(self):
        assert normalize_paper_name("My_Super-Paper") == "mysuperpaper"

    def test_empty_string(self):
        assert normalize_paper_name("") == ""

    def test_whitespace_stripped(self):
        assert normalize_paper_name("  paper  ") == "paper"

    def test_preserves_numbers(self):
        assert normalize_paper_name("nerf-2020") == "nerf2020"


class TestFindPaperMatches:
    def test_exact_match(self):
        index = {"nerf-2020": {}, "if-net": {}}
        result = find_paper_matches("nerf-2020", index)
        assert result.match_type == MatchType.EXACT
        assert result.matches == ["nerf-2020"]

    def test_normalized_match_hyphen(self):
        index = {"if-net": {}, "nerf-2020": {}}
        result = find_paper_matches("ifnet", index)
        assert result.match_type == MatchType.NORMALIZED
        assert result.matches == ["if-net"]

    def test_normalized_match_underscore(self):
        index = {"my_paper": {}, "other": {}}
        result = find_paper_matches("mypaper", index)
        assert result.match_type == MatchType.NORMALIZED
        assert result.matches == ["my_paper"]

    def test_normalized_match_case_insensitive(self):
        index = {"NeRF": {}, "other": {}}
        result = find_paper_matches("nerf", index)
        assert result.match_type == MatchType.NORMALIZED
        assert result.matches == ["NeRF"]

    def test_fuzzy_match_typo(self):
        index = {"neural-radiance-fields": {}, "other-paper": {}}
        result = find_paper_matches("nueral-radiance-fields", index, fuzzy_cutoff=0.7)
        assert result.match_type == MatchType.FUZZY
        assert "neural-radiance-fields" in result.matches

    def test_fuzzy_match_multiple(self):
        index = {"nerf-2020": {}, "nerf-2021": {}, "nerf-w": {}}
        result = find_paper_matches("nerf", index, fuzzy_cutoff=0.5)
        assert result.match_type == MatchType.FUZZY
        assert len(result.matches) >= 2

    def test_not_found(self):
        index = {"paper-a": {}, "paper-b": {}}
        result = find_paper_matches("completely-different", index)
        assert result.match_type == MatchType.NOT_FOUND
        assert result.matches == []

    def test_empty_query(self):
        result = find_paper_matches("", {"paper": {}})
        assert result.match_type == MatchType.NOT_FOUND

    def test_empty_index(self):
        result = find_paper_matches("paper", {})
        assert result.match_type == MatchType.NOT_FOUND

    def test_cutoff_threshold_high(self):
        index = {"very-specific-name": {}}
        # Low similarity should not match with high cutoff
        result = find_paper_matches("abc", index, fuzzy_cutoff=0.9)
        assert result.match_type == MatchType.NOT_FOUND

    def test_preserves_original_query(self):
        index = {"if-net": {}}
        result = find_paper_matches("  ifnet  ", index)
        assert result.query == "  ifnet  "
        assert result.normalized_query == "ifnet"

    def test_normalized_collision_returns_fuzzy(self):
        """Multiple papers normalizing to same value should be FUZZY, not NORMALIZED"""
        index = {"IF-Net": {}, "if_net": {}}  # Both normalize to "ifnet"
        result = find_paper_matches("ifnet", index)
        assert result.match_type == MatchType.FUZZY
        assert len(result.matches) == 2
        assert "IF-Net" in result.matches
        assert "if_net" in result.matches


class TestSelectPaperInteractively:
    def test_returns_none_for_empty_list(self):
        result = select_paper_interactively([], "query", {})
        assert result is None

    def test_single_match_returns_none_non_tty(self, monkeypatch):
        """Single match should NOT auto-select - requires TTY confirmation"""
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        result = select_paper_interactively(["paper-a"], "query", {})
        assert result is None

    def test_returns_none_for_non_tty(self, monkeypatch):
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        result = select_paper_interactively(["a", "b"], "query", {})
        assert result is None

    @pytest.mark.parametrize(
        "choice,expected",
        [
            (1, "paper-a"),
            (2, "paper-b"),
            (0, None),
        ],
    )
    def test_interactive_selection(self, monkeypatch, choice, expected):
        import sys

        import click

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: choice)

        result = select_paper_interactively(["paper-a", "paper-b"], "query", {})
        assert result == expected

    def test_handles_abort(self, monkeypatch):
        import sys

        import click

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        def raise_abort(*args, **kwargs):
            raise click.Abort()

        monkeypatch.setattr(click, "prompt", raise_abort)

        result = select_paper_interactively(["paper-a", "paper-b"], "query", {})
        assert result is None


class TestGetBestFuzzySimilarity:
    def test_identical_names(self):
        assert get_best_fuzzy_similarity("paper", "paper") == 1.0

    def test_normalized_identical(self):
        assert get_best_fuzzy_similarity("if-net", "ifnet") == 1.0

    def test_different_names(self):
        sim = get_best_fuzzy_similarity("paper", "completely-different")
        assert sim < 0.5

    def test_similar_names(self):
        sim = get_best_fuzzy_similarity("nerf2020", "nerf-2020")
        assert sim == 1.0  # Normalized forms are identical


class TestMatchResult:
    def test_dataclass_creation(self):
        result = MatchResult(
            match_type=MatchType.EXACT,
            matches=["paper"],
            query="query",
            normalized_query="query",
        )
        assert result.match_type == MatchType.EXACT
        assert result.matches == ["paper"]

    def test_match_type_values(self):
        assert MatchType.EXACT.value == "exact"
        assert MatchType.NORMALIZED.value == "normalized"
        assert MatchType.FUZZY.value == "fuzzy"
        assert MatchType.NOT_FOUND.value == "not_found"


class TestSelectArxivResultInteractively:
    """Tests for select_arxiv_result_interactively function."""

    def test_returns_none_for_empty_list(self):
        result = select_arxiv_result_interactively([], "query")
        assert result is None

    def test_returns_none_for_non_tty(self, monkeypatch):
        import sys

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        results = [
            {"arxiv_id": "1234.56789", "title": "Test", "authors": [], "published": "2023-01-01", "similarity": 0.8}
        ]
        result = select_arxiv_result_interactively(results, "query")
        assert result is None

    @pytest.mark.parametrize(
        "choice,expected",
        [
            (1, "1234.56789"),
            (2, "5678.12345"),
            (0, None),
        ],
    )
    def test_interactive_selection(self, monkeypatch, choice, expected):
        import sys

        import click

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: choice)

        results = [
            {
                "arxiv_id": "1234.56789",
                "title": "Paper A",
                "authors": ["Author"],
                "published": "2023-01-01",
                "similarity": 0.9,
            },
            {
                "arxiv_id": "5678.12345",
                "title": "Paper B",
                "authors": ["Other"],
                "published": "2022-06-15",
                "similarity": 0.7,
            },
        ]
        result = select_arxiv_result_interactively(results, "query")
        assert result == expected

    def test_handles_abort(self, monkeypatch):
        import sys

        import click

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        def raise_abort(*args, **kwargs):
            raise click.Abort()

        monkeypatch.setattr(click, "prompt", raise_abort)

        results = [
            {"arxiv_id": "1234.56789", "title": "Test", "authors": [], "published": "2023-01-01", "similarity": 0.8}
        ]
        result = select_arxiv_result_interactively(results, "query")
        assert result is None

    def test_handles_out_of_range_selection(self, monkeypatch):
        import sys

        import click

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: 99)  # Out of range

        results = [
            {"arxiv_id": "1234.56789", "title": "Test", "authors": [], "published": "2023-01-01", "similarity": 0.8}
        ]
        result = select_arxiv_result_interactively(results, "query")
        assert result is None
