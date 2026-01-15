"""Install/uninstall helpers for skills, prompts, and MCP servers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import click

from .config import default_pqa_embedding_model
from .output import echo, echo_error, echo_success, echo_warning


def _install_skill(*, targets: tuple[str, ...], force: bool) -> None:
    # Find the skill directory relative to this module
    module_dir = Path(__file__).resolve().parent
    root_dir = module_dir.parent
    skill_source = module_dir / "skill"
    if not skill_source.exists():
        skill_source = root_dir / "skill"

    if not skill_source.exists():
        echo_error(f"Skill directory not found at {skill_source}")
        echo_error("This may happen if paperpipe was installed without the skill files.")
        raise SystemExit(1)

    # Default to all if no specific target given
    install_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "skills",
        "codex": Path.home() / ".codex" / "skills",
        "gemini": Path.home() / ".gemini" / "skills",
    }

    installed = []
    for target in sorted(install_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown install target: {target}")
        skills_dir = target_dirs[target]
        dest = skills_dir / "papi"

        # Check if already installed
        if dest.exists() or dest.is_symlink():
            if not force:
                if dest.is_symlink() and dest.resolve() == skill_source.resolve():
                    echo(f"{target}: already installed at {dest}")
                    continue
                echo_warning(f"{target}: {dest} already exists (use --force to overwrite)")
                continue
            # Remove existing
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)

        # Create parent directory if needed
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink
        dest.symlink_to(skill_source)
        installed.append((target, dest))
        echo_success(f"{target}: installed at {dest} -> {skill_source}")

        if target == "gemini":
            settings_path = Path.home() / ".gemini" / "settings.json"
            enabled = False
            if settings_path.exists():
                try:
                    obj = json.loads(settings_path.read_text())
                    experimental = obj.get("experimental")
                    enabled = isinstance(experimental, dict) and experimental.get("skills") is True
                except Exception:
                    enabled = False
            if not enabled:
                echo_warning("gemini: skills are experimental; enable them in ~/.gemini/settings.json:")
                echo('  {"experimental": {"skills": true}}')

    if installed:
        echo()
        echo("Restart your CLI to activate the skill.")


def _install_prompts(*, targets: tuple[str, ...], force: bool, copy: bool) -> None:
    module_dir = Path(__file__).resolve().parent
    root_dir = module_dir.parent

    prompt_root = module_dir / "prompts"
    if not prompt_root.exists():
        prompt_root = root_dir / "prompts"
    if not prompt_root.exists():
        echo_error(f"Prompts directory not found at {prompt_root}")
        echo_error("This may happen if paperpipe was installed without the prompt files.")
        raise SystemExit(1)

    install_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "commands",
        "codex": Path.home() / ".codex" / "prompts",
        "gemini": Path.home() / ".gemini" / "commands",
    }

    source_dirs = {
        "claude": prompt_root / "claude",
        "codex": prompt_root / "codex",
        "gemini": prompt_root / "gemini",
    }

    installed: list[tuple[str, Path]] = []
    for target in sorted(install_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown install target: {target}")
        prompt_source = source_dirs.get(target, prompt_root)
        if not prompt_source.exists():
            echo_error(f"{target}: prompts directory not found at {prompt_source}")
            raise SystemExit(1)

        suffix = ".toml" if target == "gemini" else ".md"
        prompt_files = sorted([p for p in prompt_source.glob(f"*{suffix}") if p.is_file()])
        if not prompt_files:
            echo_error(f"{target}: no prompts found in {prompt_source}")
            raise SystemExit(1)

        dest_dir = target_dirs[target]
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src in prompt_files:
            dest = dest_dir / src.name

            if dest.exists() or dest.is_symlink():
                if not force:
                    if dest.is_symlink() and dest.resolve() == src.resolve():
                        echo(f"{target}: already installed: {dest.name}")
                        continue
                    echo_warning(f"{target}: {dest} already exists (use --force to overwrite)")
                    continue
                if dest.is_symlink() or dest.is_file():
                    dest.unlink()
                elif dest.is_dir():
                    shutil.rmtree(dest)

            try:
                if copy:
                    shutil.copy2(src, dest)
                else:
                    dest.symlink_to(src)
            except OSError as e:
                echo_error(f"{target}: failed to install {src.name}: {e}")
                if not copy:
                    echo_error("If your filesystem does not support symlinks, re-run with --copy.")
                raise SystemExit(1)

            installed.append((target, dest))

        mode = "copied" if copy else "linked"
        echo_success(f"{target}: {mode} {len(prompt_files)} prompt(s) into {dest_dir}")

    if installed:
        echo()
        echo("Restart your CLI to pick up new prompts/commands.")


def _install_mcp(*, targets: tuple[str, ...], name: str, embedding: Optional[str], force: bool) -> None:
    @dataclass(frozen=True)
    class McpServerSpec:
        name: str
        command: str
        args: tuple[str, ...]
        env: dict[str, str]
        description: str
        cwd: Optional[str] = None

    def _paperqa_mcp_is_available() -> bool:
        if sys.version_info < (3, 11):
            return False
        import importlib.util

        return (
            importlib.util.find_spec("mcp.server.fastmcp") is not None
            and importlib.util.find_spec("paperqa") is not None
        )

    paperqa_name = (name or "").strip()

    embedding_model = (embedding or "").strip() if embedding else default_pqa_embedding_model()

    servers: list[McpServerSpec] = []
    if _paperqa_mcp_is_available():
        if not paperqa_name:
            raise click.UsageError("--name must be non-empty")
        servers.append(
            McpServerSpec(
                name=paperqa_name,
                command="paperqa_mcp_server",
                args=(),
                env={"PAPERQA_EMBEDDING": embedding_model},
                description="PaperQA2 and LEANN retrieval search",
            )
        )

    if not servers:
        echo_error("No MCP servers available to install in this environment.")
        echo_error("Install: pip install 'paperpipe[mcp]' (Python 3.11+)")
        raise SystemExit(1)

    install_targets = set(targets) if targets else {"claude", "codex", "gemini"}
    successes: list[str] = []

    def _read_json_object(path: Path, *, where: str) -> Optional[dict[str, Any]]:
        if path.exists():
            try:
                obj = json.loads(path.read_text())
            except Exception:
                echo_error(f"{where}: failed to parse JSON at {path}")
                return None
            if not isinstance(obj, dict):
                echo_error(f"{where}: expected a JSON object at {path}")
                return None
            return obj
        return {}

    def _write_json_object(path: Path, obj: dict[str, Any], *, where: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2) + "\n")
        except OSError as e:
            echo_error(f"{where}: failed to write {path}: {e}")
            return False
        return True

    def upsert_mcp_servers(path: Path, *, where: str, server_key: str, entry: dict[str, Any]) -> str:
        obj = _read_json_object(path, where=where)
        if obj is None:
            return "error"

        mcp_servers = obj.get("mcpServers")
        if mcp_servers is None:
            mcp_servers = {}
            obj["mcpServers"] = mcp_servers
        if not isinstance(mcp_servers, dict):
            echo_error(f"{where}: expected 'mcpServers' to be an object in {path}")
            return "error"

        existing = mcp_servers.get(server_key)
        if existing is not None and existing != entry and not force:
            echo_warning(f"{where}: {server_key!r} already configured in {path} (use --force to overwrite)")
            return "skipped"
        if existing is not None and existing == entry:
            return "unchanged"

        mcp_servers[server_key] = entry
        return "written" if _write_json_object(path, obj, where=where) else "error"

    for target in sorted(install_targets):
        if target == "claude":
            if not shutil.which("claude"):
                echo_warning("claude: `claude` not found on PATH; install project config instead:")
                echo("  papi install mcp --repo")
                continue

            for spec in servers:
                if force:
                    subprocess.run(["claude", "mcp", "remove", spec.name], capture_output=True, text=True)

                cmd = ["claude", "mcp", "add", "--transport", "stdio"]
                for k, v in spec.env.items():
                    cmd.extend(["--env", f"{k}={v}"])
                cmd.extend(["--scope", "user", spec.name, "--", spec.command, *spec.args])

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    echo_warning(f"claude: failed to install {spec.name!r} via `claude mcp add`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    echo_warning("You can install a project-scoped config file instead:")
                    echo("  papi install mcp --repo")
                    continue

                echo_success(f"claude: installed {spec.name!r}")
                successes.append(f"claude:{spec.name}")
            continue

        if target == "repo":
            claude_dest = Path.cwd() / ".mcp.json"
            gemini_dest = Path.cwd() / ".gemini" / "settings.json"
            for spec in servers:
                entry: dict[str, Any] = {"command": spec.command, "args": list(spec.args), "env": dict(spec.env)}
                if spec.cwd is not None:
                    entry["cwd"] = spec.cwd
                claude_status = upsert_mcp_servers(claude_dest, where="repo/claude", server_key=spec.name, entry=entry)
                if claude_status == "written":
                    echo_success(f"repo: wrote {claude_dest} ({spec.name!r})")
                    successes.append(f"repo/claude:{spec.name}")
                elif claude_status == "unchanged":
                    echo(f"repo: already configured {spec.name!r} in {claude_dest}")
                    successes.append(f"repo/claude:{spec.name}")
                elif claude_status == "skipped":
                    successes.append(f"repo/claude:{spec.name}")

                gemini_status = upsert_mcp_servers(gemini_dest, where="repo/gemini", server_key=spec.name, entry=entry)
                if gemini_status == "written":
                    echo_success(f"repo: wrote {gemini_dest} ({spec.name!r})")
                    successes.append(f"repo/gemini:{spec.name}")
                elif gemini_status == "unchanged":
                    echo(f"repo: already configured {spec.name!r} in {gemini_dest}")
                    successes.append(f"repo/gemini:{spec.name}")
                elif gemini_status == "skipped":
                    successes.append(f"repo/gemini:{spec.name}")
            continue

        if target == "codex":
            if not shutil.which("codex"):
                echo_warning("codex: `codex` not found on PATH; run this manually:")
                for spec in servers:
                    env_flags = " ".join([f"--env {k}={v}" for k, v in spec.env.items()])
                    env_flags = f" {env_flags}" if env_flags else ""
                    echo(f"  codex mcp add {spec.name}{env_flags} -- {spec.command}")
                continue

            # Let Codex manage its own config. Prefer replacing only when requested.
            for spec in servers:
                if force:
                    subprocess.run(["codex", "mcp", "remove", spec.name], capture_output=True, text=True)

                cmd = ["codex", "mcp", "add", spec.name]
                for k, v in spec.env.items():
                    cmd.extend(["--env", f"{k}={v}"])
                cmd.extend(["--", spec.command, *spec.args])

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    echo_warning(f"codex: failed to install {spec.name!r} via `codex mcp add`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    echo("Try re-running with --force, or run manually:")
                    env_flags = " ".join([f"--env {k}={v}" for k, v in spec.env.items()])
                    env_flags = f" {env_flags}" if env_flags else ""
                    echo(f"  codex mcp add {spec.name}{env_flags} -- {spec.command}")
                    continue

                echo_success(f"codex: installed {spec.name!r}")
                successes.append(f"codex:{spec.name}")
            continue

        if target == "gemini":
            install_via_cli_ok = False
            if shutil.which("gemini"):
                install_via_cli_ok = True
                for spec in servers:
                    if force:
                        subprocess.run(
                            ["gemini", "mcp", "remove", "--scope", "user", spec.name],
                            capture_output=True,
                            text=True,
                        )

                    cmd = [
                        "gemini",
                        "mcp",
                        "add",
                        "--scope",
                        "user",
                        "--transport",
                        "stdio",
                    ]
                    for k, v in spec.env.items():
                        cmd.extend(["--env", f"{k}={v}"])
                    cmd.extend(["--description", spec.description, spec.name, spec.command, *spec.args])

                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        echo_success(f"gemini: installed {spec.name!r}")
                        successes.append(f"gemini:{spec.name}")
                        continue

                    install_via_cli_ok = False
                    echo_warning(
                        f"gemini: failed to install {spec.name!r} via `gemini mcp add`; "
                        "falling back to ~/.gemini/settings.json"
                    )
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)

            if not install_via_cli_ok:
                dest = Path.home() / ".gemini" / "settings.json"
                for spec in servers:
                    gemini_entry: dict[str, Any] = {
                        "command": spec.command,
                        "args": list(spec.args),
                        "env": dict(spec.env),
                    }
                    if spec.cwd is not None:
                        gemini_entry["cwd"] = spec.cwd
                    status = upsert_mcp_servers(dest, where="gemini", server_key=spec.name, entry=gemini_entry)
                    if status == "written":
                        echo_success(f"gemini: configured {spec.name!r} in {dest}")
                        successes.append(f"gemini:{spec.name}")
                    elif status == "unchanged":
                        echo(f"gemini: already configured {spec.name!r} in {dest}")
                        successes.append(f"gemini:{spec.name}")
                    elif status == "skipped":
                        successes.append(f"gemini:{spec.name}")
            continue

        raise click.UsageError(f"Unknown target: {target}")

    if not successes:
        raise SystemExit(1)

    echo()
    echo("Restart your CLI to pick up the new MCP server.")


def _parse_components(args: tuple[str, ...]) -> list[str]:
    raw: list[str] = []
    for item in args:
        for part in item.split(","):
            part = part.strip().lower()
            if part:
                raw.append(part)
    return raw


def _uninstall_skill(*, targets: tuple[str, ...], force: bool) -> None:
    module_dir = Path(__file__).resolve().parent
    root_dir = module_dir.parent
    skill_source = module_dir / "skill"
    if not skill_source.exists():
        skill_source = root_dir / "skill"

    if not skill_source.exists() and not force:
        echo_error(f"Skill directory not found at {skill_source}")
        echo_error("Re-run with --force to remove install locations without validating the source.")
        raise SystemExit(1)

    uninstall_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "skills",
        "codex": Path.home() / ".codex" / "skills",
        "gemini": Path.home() / ".gemini" / "skills",
    }

    removed = 0
    skipped = 0
    for target in sorted(uninstall_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown uninstall target: {target}")
        skills_dir = target_dirs[target]
        dest = skills_dir / "papi"

        if not dest.exists() and not dest.is_symlink():
            echo(f"{target}: not installed")
            continue

        if dest.is_symlink() and skill_source.exists() and dest.resolve() == skill_source.resolve():
            dest.unlink()
            echo_success(f"{target}: removed {dest}")
            removed += 1
            continue

        if force:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink(missing_ok=True)
            echo_success(f"{target}: removed {dest}")
            removed += 1
            continue

        echo_warning(f"{target}: {dest} exists but does not point to this install (use --force to remove)")
        skipped += 1

    if removed:
        echo()
        echo("Restart your CLI to unload the skill.")
    if skipped:
        raise SystemExit(1)


def _uninstall_prompts(*, targets: tuple[str, ...], force: bool) -> None:
    module_dir = Path(__file__).resolve().parent
    root_dir = module_dir.parent
    prompt_root = module_dir / "prompts"
    if not prompt_root.exists():
        prompt_root = root_dir / "prompts"
    if not prompt_root.exists():
        echo_error(f"Prompts directory not found at {prompt_root}")
        echo_error("This may happen if paperpipe was installed without the prompt files.")
        raise SystemExit(1)

    uninstall_targets = set(targets) if targets else {"claude", "codex", "gemini"}

    target_dirs = {
        "claude": Path.home() / ".claude" / "commands",
        "codex": Path.home() / ".codex" / "prompts",
        "gemini": Path.home() / ".gemini" / "commands",
    }

    source_dirs = {
        "claude": prompt_root / "claude",
        "codex": prompt_root / "codex",
        "gemini": prompt_root / "gemini",
    }

    removed = 0
    skipped = 0
    for target in sorted(uninstall_targets):
        if target not in target_dirs:
            raise click.UsageError(f"Unknown uninstall target: {target}")

        dest_dir = target_dirs[target]
        if not dest_dir.exists():
            echo(f"{target}: no prompt directory at {dest_dir}")
            continue

        suffix = ".toml" if target == "gemini" else ".md"
        prompt_source = source_dirs.get(target, prompt_root)
        if not prompt_source.exists():
            echo_error(f"{target}: prompts directory not found at {prompt_source}")
            raise SystemExit(1)

        source_files = sorted([p for p in prompt_source.glob(f"*{suffix}") if p.is_file()])
        for src in source_files:
            dest = dest_dir / src.name
            if not dest.exists() and not dest.is_symlink():
                continue

            if dest.is_symlink():
                if dest.resolve() == src.resolve() or force:
                    dest.unlink()
                    removed += 1
                else:
                    echo_warning(f"{target}: {dest} points elsewhere (use --force to remove)")
                    skipped += 1
                continue

            # Copied file case: remove only if identical unless forced.
            if dest.is_file():
                if force:
                    dest.unlink()
                    removed += 1
                    continue
                try:
                    if dest.read_bytes() == src.read_bytes():
                        dest.unlink()
                        removed += 1
                    else:
                        echo_warning(f"{target}: {dest} differs from packaged prompt (use --force to remove)")
                        skipped += 1
                except OSError as e:
                    echo_warning(f"{target}: failed to read {dest}: {e}")
                    skipped += 1
                continue

            if dest.is_dir():
                if force:
                    shutil.rmtree(dest)
                    removed += 1
                else:
                    echo_warning(f"{target}: {dest} is a directory (use --force to remove)")
                    skipped += 1

        if source_files:
            echo_success(f"{target}: removed prompts from {dest_dir}")

    if removed:
        echo()
        echo("Restart your CLI to unload prompts/commands.")
    if skipped:
        raise SystemExit(1)


def _uninstall_mcp(*, targets: tuple[str, ...], name: str, force: bool) -> None:
    paperqa_name = (name or "").strip()
    if not paperqa_name:
        raise click.UsageError("--name must be non-empty")

    server_keys = [paperqa_name]

    uninstall_targets = set(targets) if targets else {"claude", "codex", "gemini"}
    successes: list[str] = []
    failures: list[str] = []

    def _read_json_object(path: Path, *, where: str) -> Optional[dict[str, Any]]:
        if path.exists():
            try:
                obj = json.loads(path.read_text())
            except Exception:
                echo_error(f"{where}: failed to parse JSON at {path}")
                return None
            if not isinstance(obj, dict):
                echo_error(f"{where}: expected a JSON object at {path}")
                return None
            return obj
        return {}

    def _write_json_object(path: Path, obj: dict[str, Any], *, where: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2) + "\n")
        except OSError as e:
            echo_error(f"{where}: failed to write {path}: {e}")
            return False
        return True

    def remove_mcp_servers(path: Path, *, where: str, keys: list[str]) -> bool:
        obj = _read_json_object(path, where=where)
        if obj is None:
            return False

        mcp_servers = obj.get("mcpServers")
        if mcp_servers is None:
            return True
        if not isinstance(mcp_servers, dict):
            echo_error(f"{where}: expected 'mcpServers' to be an object in {path}")
            return False

        changed = False
        for key in keys:
            if key in mcp_servers:
                del mcp_servers[key]
                changed = True

        if not changed:
            return True
        return _write_json_object(path, obj, where=where)

    for target in sorted(uninstall_targets):
        if target == "claude":
            if not shutil.which("claude"):
                echo_warning("claude: `claude` not found on PATH; remove manually or use repo-local config:")
                echo("  papi uninstall mcp --repo")
                continue

            for key in server_keys:
                proc = subprocess.run(["claude", "mcp", "remove", key], capture_output=True, text=True)
                if proc.returncode != 0 and not force:
                    echo_warning(f"claude: failed to remove {key!r} via `claude mcp remove`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    failures.append(f"claude:{key}")
                    continue
                echo_success(f"claude: removed {key!r}")
                successes.append(f"claude:{key}")
            continue

        if target == "repo":
            claude_dest = Path.cwd() / ".mcp.json"
            gemini_dest = Path.cwd() / ".gemini" / "settings.json"
            if remove_mcp_servers(claude_dest, where="repo/claude", keys=server_keys):
                echo_success(f"repo: updated {claude_dest}")
                successes.extend([f"repo/claude:{k}" for k in server_keys])
            else:
                failures.extend([f"repo/claude:{k}" for k in server_keys])
            if remove_mcp_servers(gemini_dest, where="repo/gemini", keys=server_keys):
                echo_success(f"repo: updated {gemini_dest}")
                successes.extend([f"repo/gemini:{k}" for k in server_keys])
            else:
                failures.extend([f"repo/gemini:{k}" for k in server_keys])
            continue

        if target == "codex":
            if not shutil.which("codex"):
                echo_warning("codex: `codex` not found on PATH; run this manually:")
                for key in server_keys:
                    echo(f"  codex mcp remove {key}")
                continue

            for key in server_keys:
                proc = subprocess.run(["codex", "mcp", "remove", key], capture_output=True, text=True)
                if proc.returncode != 0 and not force:
                    echo_warning(f"codex: failed to remove {key!r} via `codex mcp remove`")
                    if proc.stdout.strip():
                        echo(proc.stdout.rstrip("\n"))
                    if proc.stderr.strip():
                        echo(proc.stderr.rstrip("\n"), err=True)
                    failures.append(f"codex:{key}")
                    continue
                echo_success(f"codex: removed {key!r}")
                successes.append(f"codex:{key}")
            continue

        if target == "gemini":
            if shutil.which("gemini"):
                for key in server_keys:
                    proc = subprocess.run(
                        ["gemini", "mcp", "remove", "--scope", "user", key],
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0 and not force:
                        echo_warning(f"gemini: failed to remove {key!r} via `gemini mcp remove`")
                        if proc.stdout.strip():
                            echo(proc.stdout.rstrip("\n"))
                        if proc.stderr.strip():
                            echo(proc.stderr.rstrip("\n"), err=True)
                        failures.append(f"gemini:{key}")
                        continue
                    echo_success(f"gemini: removed {key!r}")
                    successes.append(f"gemini:{key}")

            dest = Path.home() / ".gemini" / "settings.json"
            if remove_mcp_servers(dest, where="gemini", keys=server_keys):
                echo_success(f"gemini: updated {dest}")
                successes.extend([f"gemini:{k}" for k in server_keys])
            else:
                failures.extend([f"gemini:{k}" for k in server_keys])
            continue

        raise click.UsageError(f"Unknown target: {target}")

    if failures and not force:
        raise SystemExit(1)
    if successes:
        echo()
        echo("Restart your CLI to unload the MCP server config.")
