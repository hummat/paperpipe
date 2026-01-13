from __future__ import annotations

import shutil
import subprocess
import types
from pathlib import Path

import pytest

import paperpipe.search as search_mod


class TestGrepCollection:
    def test_collect_grep_matches_uses_rg_and_parses(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (temp_db / "papers" / "p").mkdir(parents=True)

        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="p/summary.md:2:hit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        matches = search_mod._collect_grep_matches(
            query="hit",
            fixed_strings=True,
            max_matches=10,
            ignore_case=True,
            include_tex=False,
        )
        assert matches and matches[0]["paper"] == "p"
        assert matches[0]["path"] == "p/summary.md"
        assert matches[0]["line"] == 2
        assert "--context" in calls[0] and "0" in calls[0]
        assert "--fixed-strings" in calls[0]
        assert "--ignore-case" in calls[0]

    def test_collect_grep_matches_rg_no_hits_returns_empty(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (temp_db / "papers" / "p").mkdir(parents=True)
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/rg" if cmd == "rg" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        matches = search_mod._collect_grep_matches(
            query="nope",
            fixed_strings=True,
            max_matches=10,
            ignore_case=False,
            include_tex=False,
        )
        assert matches == []

    def test_collect_grep_matches_falls_back_to_grep(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (temp_db / "papers" / "p").mkdir(parents=True)
        monkeypatch.setattr(shutil, "which", lambda cmd: "/bin/grep" if cmd == "grep" else None)

        def fake_run(args: list[str], **kwargs):
            return types.SimpleNamespace(returncode=0, stdout="p/notes.md:5:hit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        matches = search_mod._collect_grep_matches(
            query="hit",
            fixed_strings=False,
            max_matches=10,
            ignore_case=True,
            include_tex=False,
        )
        assert matches and matches[0]["path"] == "p/notes.md"

    def test_collect_grep_matches_no_tools_returns_empty(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (temp_db / "papers").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        matches = search_mod._collect_grep_matches(
            query="x",
            fixed_strings=True,
            max_matches=10,
            ignore_case=False,
            include_tex=False,
        )
        assert matches == []
