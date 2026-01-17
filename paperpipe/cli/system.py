"""System commands: install, uninstall, path, and docs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .. import config
from ..install import (
    _install_mcp,
    _install_prompts,
    _install_skill,
    _parse_components,
    _uninstall_mcp,
    _uninstall_prompts,
    _uninstall_skill,
)


@click.command()
def path():
    """Print the paper database path."""
    click.echo(config.PAPER_DB)


@click.command("install")
@click.argument("components", nargs=-1)
@click.option(
    "--claude",
    "targets",
    flag_value="claude",
    multiple=True,
    help="Install for Claude Code (~/.claude/...)",
)
@click.option(
    "--codex",
    "targets",
    flag_value="codex",
    multiple=True,
    help="Install for Codex CLI (~/.codex/...)",
)
@click.option(
    "--gemini",
    "targets",
    flag_value="gemini",
    multiple=True,
    help="Install for Gemini CLI (~/.gemini/...)",
)
@click.option(
    "--repo",
    "targets",
    flag_value="repo",
    multiple=True,
    help="Install repo-local MCP configs (.mcp.json + .gemini/settings.json) in the current directory (MCP only)",
)
@click.option("--force", is_flag=True, help="Overwrite existing installation")
@click.option("--copy", is_flag=True, help="Copy prompts instead of symlinking (prompts only).")
@click.option("--name", default="paperqa", show_default=True, help="MCP server name (MCP only)")
@click.option("--embedding", default=None, show_default=False, help="Embedding model id (MCP only)")
def install(
    components: tuple[str, ...],
    targets: tuple[str, ...],
    force: bool,
    copy: bool,
    name: str,
    embedding: Optional[str],
) -> None:
    """Install papi integrations (skill, prompts, and/or MCP config).

    By default, installs everything: skill + prompts + mcp.

    Components can be selected by name and combined:
      - `papi install mcp prompts`
      - `papi install mcp,prompts`

    \b
    Examples:
        papi install                    # Install skill + prompts + mcp
        papi install skill              # Install skill only
        papi install prompts --copy     # Install prompts only, copy files
        papi install mcp --repo         # Install repo-local MCP configs
        papi install --codex            # Install everything for Codex only
        papi install mcp --embedding text-embedding-3-small
    """
    requested = _parse_components(components)
    if not requested:
        requested = ["skill", "prompts", "mcp"]

    allowed = {"skill", "prompts", "mcp"}
    unknown = sorted({c for c in requested if c not in allowed})
    if unknown:
        raise click.UsageError(f"Unknown component(s): {', '.join(unknown)} (choose from: skill, prompts, mcp)")

    want_skill = "skill" in requested
    want_prompts = "prompts" in requested
    want_mcp = "mcp" in requested

    if targets and "repo" in targets and not want_mcp:
        raise click.UsageError("--repo is only valid when installing mcp")
    if copy and not want_prompts:
        raise click.UsageError("--copy is only valid when installing prompts")
    if (name != "paperqa" or embedding is not None) and not want_mcp:
        raise click.UsageError("--name/--embedding are only valid when installing mcp")

    if want_skill:
        _install_skill(targets=tuple([t for t in targets if t != "repo"]), force=force)
    if want_prompts:
        _install_prompts(targets=tuple([t for t in targets if t != "repo"]), force=force, copy=copy)
    if want_mcp:
        _install_mcp(targets=targets, name=name, embedding=embedding, force=force)


@click.command("uninstall")
@click.argument("components", nargs=-1)
@click.option(
    "--claude",
    "targets",
    flag_value="claude",
    multiple=True,
    help="Uninstall for Claude Code (~/.claude/...)",
)
@click.option(
    "--codex",
    "targets",
    flag_value="codex",
    multiple=True,
    help="Uninstall for Codex CLI (~/.codex/...)",
)
@click.option(
    "--gemini",
    "targets",
    flag_value="gemini",
    multiple=True,
    help="Uninstall for Gemini CLI (~/.gemini/...)",
)
@click.option(
    "--repo",
    "targets",
    flag_value="repo",
    multiple=True,
    help="Uninstall repo-local MCP configs (.mcp.json + .gemini/settings.json) in the current directory (MCP only)",
)
@click.option("--force", is_flag=True, help="Remove even if the install does not match exactly")
@click.option("--name", default="paperqa", show_default=True, help="MCP server name (MCP only)")
def uninstall(components: tuple[str, ...], targets: tuple[str, ...], force: bool, name: str) -> None:
    """Uninstall papi integrations (skill, prompts, and/or MCP config).

    By default, uninstalls everything: mcp + prompts + skill.

    Components can be selected by name and combined:
      - `papi uninstall mcp prompts`
      - `papi uninstall mcp,prompts`

    \b
    Examples:
        papi uninstall                  # Uninstall skill + prompts + mcp
        papi uninstall skill            # Uninstall skill only
        papi uninstall prompts          # Uninstall prompts only
        papi uninstall mcp --repo       # Uninstall repo-local MCP configs
        papi uninstall --codex          # Uninstall everything for Codex only
        papi uninstall mcp --force      # Ignore remove failures / mismatches
    """
    requested = _parse_components(components)
    if not requested:
        requested = ["skill", "prompts", "mcp"]

    allowed = {"skill", "prompts", "mcp"}
    unknown = sorted({c for c in requested if c not in allowed})
    if unknown:
        raise click.UsageError(f"Unknown component(s): {', '.join(unknown)} (choose from: skill, prompts, mcp)")

    want_skill = "skill" in requested
    want_prompts = "prompts" in requested
    want_mcp = "mcp" in requested

    if targets and "repo" in targets and not want_mcp:
        raise click.UsageError("--repo is only valid when uninstalling mcp")
    if name != "paperqa" and not want_mcp:
        raise click.UsageError("--name is only valid when uninstalling mcp")

    non_repo_targets = tuple([t for t in targets if t != "repo"])

    # Default uninstall order is reverse of install: mcp -> prompts -> skill.
    if want_mcp:
        _uninstall_mcp(targets=targets, name=name, force=force)
    if want_prompts:
        _uninstall_prompts(targets=non_repo_targets, force=force)
    if want_skill:
        _uninstall_skill(targets=non_repo_targets, force=force)


@click.command("docs")
def docs() -> None:
    """Print the agent integration snippet (AGENT_INTEGRATION.md content).

    Use this to get the current agent integration snippet for your project's
    CLAUDE.md, AGENTS.md, or similar agent instructions file.

    \b
    Examples:
        papi docs                       # Print snippet to stdout
        papi docs > ./AGENTS.md         # Write to file
    """
    module_dir = Path(__file__).resolve().parent.parent  # paperpipe/
    root_dir = module_dir.parent
    docs_path = root_dir / "AGENT_INTEGRATION.md"

    if not docs_path.exists():
        raise click.ClickException(f"AGENT_INTEGRATION.md not found at {docs_path}")

    click.echo(docs_path.read_text())
