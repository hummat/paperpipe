"""Tests for paperpipe/cli/system.py (install, uninstall, path commands)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import paperpipe.config as config

from .conftest import REPO_ROOT, cli_mod


class TestInstallSkillCommand:
    def test_install_skill_creates_symlink_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "skills" / "papi"
        assert dest.is_symlink()
        assert dest.resolve() == (REPO_ROOT / "skill").resolve()

    def test_install_skill_existing_dest_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.mkdir(parents=True)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0
        assert "use --force" in result.output.lower()

    def test_install_skill_force_overwrites_existing_file(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex", "--force"])
        assert result.exit_code == 0, result.output

        assert dest.is_symlink()

    def test_install_skill_creates_symlink_for_gemini(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--gemini"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".gemini" / "skills" / "papi"
        assert dest.is_symlink()
        assert dest.resolve() == (REPO_ROOT / "skill").resolve()


class TestInstallPromptsCommand:
    def test_install_prompts_creates_symlinks_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0, result.output

        prompt_dir = tmp_path / ".codex" / "prompts"
        assert prompt_dir.exists()

        expected = [
            "compare-papers.md",
            "curate-paper-note.md",
            "ground-with-paper.md",
            "papi.md",
            "verify-with-paper.md",
        ]
        for filename in expected:
            dest = prompt_dir / filename
            assert dest.is_symlink()
            assert dest.resolve() == (REPO_ROOT / "prompts" / "codex" / filename).resolve()

    def test_install_prompts_existing_dest_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "prompts" / "ground-with-paper.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0
        assert "use --force" in result.output.lower()

    def test_install_prompts_copy_mode_copies_files(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex", "--copy"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "curate-paper-note.md"
        assert dest.exists()
        assert not dest.is_symlink()

    def test_install_prompts_creates_symlinks_for_gemini(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--gemini"])
        assert result.exit_code == 0, result.output

        prompt_dir = tmp_path / ".gemini" / "commands"
        assert prompt_dir.exists()

        expected = [
            "compare-papers.toml",
            "curate-paper-note.toml",
            "ground-with-paper.toml",
            "papi-run.toml",
            "papi.toml",
            "papi-list.toml",
            "papi-path.toml",
            "papi-search.toml",
            "papi-show-eq.toml",
            "papi-show-summary.toml",
            "papi-show-tex.toml",
            "papi-tags.toml",
            "verify-with-paper.toml",
        ]
        for filename in expected:
            dest = prompt_dir / filename
            assert dest.is_symlink()
            assert dest.resolve() == (REPO_ROOT / "prompts" / "gemini" / filename).resolve()


class TestInstallMcpCommand:
    @pytest.fixture(autouse=True)
    def _pretend_paperqa_mcp_is_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """install mcp now installs only MCP servers available in the environment.

        Tests run without optional deps installed, so we pretend PaperQA2 MCP deps exist.
        """
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, package: str | None = None):  # type: ignore[override]
            if name in {"mcp.server.fastmcp", "paperqa"}:
                return importlib.machinery.ModuleSpec(name, loader=None)
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    def test_install_mcp_claude_runs_claude_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--claude"])
        assert result.exit_code == 0, result.output

        assert any(
            c[:3] == ["claude", "mcp", "add"]
            and "--transport" in c
            and "stdio" in c
            and "--env" in c
            and any(s.startswith("PAPERQA_EMBEDDING=") for s in c)
            and "--scope" in c
            and "user" in c
            and "paperqa" in c
            and "--" in c
            and "paperqa_mcp" in c
            for c in calls
        )

    def test_install_mcp_repo_writes_mcp_json(self, temp_db: Path):
        runner = pytest.importorskip("click.testing").CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["install", "mcp", "--repo", "--embedding", "text-embedding-3-small"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["command"] == "paperqa_mcp"
            assert cfg["mcpServers"]["paperqa"]["args"] == []
            cfg2 = json.loads((Path(".gemini") / "settings.json").read_text())
            assert cfg2["mcpServers"]["paperqa"]["command"] == "paperqa_mcp"
            assert cfg2["mcpServers"]["paperqa"]["args"] == []

    def test_install_mcp_repo_writes_config(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = pytest.importorskip("click.testing").CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["install", "mcp", "--repo", "--embedding", "text-embedding-3-small"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["command"] == "paperqa_mcp"
            assert cfg["mcpServers"]["paperqa"]["args"] == []
            # leann_mcp is installed as a separate server if available
            if shutil.which("leann_mcp"):
                assert cfg["mcpServers"]["leann"]["command"] == "leann_mcp"

    def test_install_mcp_repo_uses_paperqa_embedding_env_override(
        self, temp_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (temp_db / "config.toml").write_text("\n".join(["[paperqa]", 'embedding = "config-embedding"', ""]))
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)
        monkeypatch.setenv("PAPERQA_EMBEDDING", "env-embedding")

        runner = pytest.importorskip("click.testing").CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli_mod.cli, ["install", "mcp", "--repo"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert cfg["mcpServers"]["paperqa"]["env"]["PAPERQA_EMBEDDING"] == "env-embedding"

    def test_install_mcp_gemini_writes_settings_json(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--gemini", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output

        cfg = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert cfg["mcpServers"]["paperqa"]["command"] == "paperqa_mcp"
        assert cfg["mcpServers"]["paperqa"]["args"] == []
        assert cfg["mcpServers"]["paperqa"]["env"]["PAPERQA_EMBEDDING"] == "text-embedding-3-small"

    def test_install_mcp_gemini_runs_gemini_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--gemini", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output

        assert any(
            c[:3] == ["gemini", "mcp", "add"]
            and "--scope" in c
            and "user" in c
            and "--transport" in c
            and "stdio" in c
            and "--env" in c
            and "PAPERQA_EMBEDDING=text-embedding-3-small" in c
            and "paperqa" in c
            and "paperqa_mcp" in c
            for c in calls
        )

    def test_install_mcp_gemini_does_not_write_settings_json_when_cli_succeeds(
        self, temp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        settings_path = tmp_path / ".gemini" / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        before = json.dumps({"mcpServers": {"paperqa": {"some": "other-shape"}}}, indent=2) + "\n"
        settings_path.write_text(before)

        def fake_run(_args: list[str], **_kwargs):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output
        assert "already configured" not in result.output.lower()
        assert settings_path.read_text() == before

    def test_install_mcp_codex_runs_codex_mcp_add(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/codex" if cmd == "codex" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "mcp", "--codex", "--embedding", "text-embedding-3-small"])
        assert result.exit_code == 0, result.output
        assert any(
            c[:4] == ["codex", "mcp", "add", "paperqa"]
            and "paperqa_mcp" in c
            and "PAPERQA_EMBEDDING=text-embedding-3-small" in " ".join(c)
            for c in calls
        )

    def test_install_mcp_codex_force_removes_then_adds(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/codex" if cmd == "codex" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(
            cli_mod.cli,
            ["install", "mcp", "--codex", "--force", "--embedding", "text-embedding-3-small"],
        )
        assert result.exit_code == 0, result.output
        assert calls[0][:4] == ["codex", "mcp", "remove", "paperqa"]
        assert calls[1][:4] == ["codex", "mcp", "add", "paperqa"]


class TestUninstallSkillCommand:
    def test_uninstall_skill_removes_symlink_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "skill", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "skills" / "papi"
        assert dest.is_symlink()

        result2 = runner.invoke(cli_mod.cli, ["uninstall", "skill", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_skill_mismatch_requires_force(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        dest = tmp_path / ".codex" / "skills" / "papi"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("not a symlink")

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "skill", "--codex"])
        assert result.exit_code == 1
        assert "use --force" in result.output.lower()


class TestUninstallPromptsCommand:
    def test_uninstall_prompts_removes_symlinks_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "papi.md"
        assert dest.is_symlink()

        result2 = runner.invoke(cli_mod.cli, ["uninstall", "prompts", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_prompts_removes_copied_files_for_codex(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["install", "prompts", "--codex", "--copy"])
        assert result.exit_code == 0, result.output

        dest = tmp_path / ".codex" / "prompts" / "curate-paper-note.md"
        assert dest.exists()
        assert not dest.is_symlink()

        result2 = runner.invoke(cli_mod.cli, ["uninstall", "prompts", "--codex"])
        assert result2.exit_code == 0, result2.output
        assert not dest.exists()

    def test_uninstall_parses_commas_for_components(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp,prompts", "--codex", "--force"])
        assert result.exit_code == 0, result.output


class TestUninstallMcpCommand:
    def test_uninstall_mcp_claude_runs_claude_mcp_remove(self, temp_db: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude" if cmd == "claude" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--claude"])
        assert result.exit_code == 0, result.output

        assert ["claude", "mcp", "remove", "paperqa"] in calls
        # No separate leann server anymore

    def test_uninstall_mcp_repo_removes_server_keys(self, temp_db: Path):
        runner = pytest.importorskip("click.testing").CliRunner()
        with runner.isolated_filesystem():
            Path(".mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "paperqa": {"command": "papi", "args": [], "env": {}},
                            "other": {"command": "x", "args": [], "env": {}},
                        }
                    }
                )
                + "\n"
            )
            (Path(".gemini") / "settings.json").parent.mkdir(parents=True, exist_ok=True)
            (Path(".gemini") / "settings.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "paperqa": {"command": "papi", "args": [], "env": {}},
                            "other": {"command": "x", "args": [], "env": {}},
                        }
                    }
                )
                + "\n"
            )

            result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--repo"])
            assert result.exit_code == 0, result.output

            cfg = json.loads(Path(".mcp.json").read_text())
            assert "paperqa" not in cfg["mcpServers"]
            # No separate leann server anymore
            assert "other" in cfg["mcpServers"]

            cfg2 = json.loads((Path(".gemini") / "settings.json").read_text())
            assert "paperqa" not in cfg2["mcpServers"]
            # No separate leann server anymore
            assert "other" in cfg2["mcpServers"]

    def test_uninstall_mcp_gemini_writes_settings_json(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)

        (tmp_path / ".gemini").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".gemini" / "settings.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "paperqa": {"command": "papi", "args": [], "env": {}},
                    }
                }
            )
            + "\n"
        )

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output

        cfg = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert "paperqa" not in cfg.get("mcpServers", {})
        # No separate leann server anymore

    def test_uninstall_mcp_gemini_runs_gemini_mcp_remove(self, temp_db: Path, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/gemini" if cmd == "gemini" else None)

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs):
            calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "mcp", "--gemini"])
        assert result.exit_code == 0, result.output

        assert ["gemini", "mcp", "remove", "--scope", "user", "paperqa"] in calls
        # No separate leann server anymore


class TestUninstallValidation:
    def test_uninstall_repo_requires_mcp_component(self, temp_db: Path):
        runner = pytest.importorskip("click.testing").CliRunner()
        result = runner.invoke(cli_mod.cli, ["uninstall", "skill", "--repo"])
        assert result.exit_code != 0
        assert "--repo is only valid" in result.output
