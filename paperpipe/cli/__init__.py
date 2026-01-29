"""CLI entry point and command registration."""

import click

from ..core import ensure_db
from ..output import _setup_debug_logging, set_quiet
from .ask import ask
from .export import audit, export
from .helpers import _cli_version
from .index import index_cmd
from .models import models
from .papers import add, notes, regenerate, remove, show
from .rebuild import rebuild_index
from .search_cli import list_papers, search, tags
from .system import docs, install, path, uninstall

# Aliases: maps alias -> canonical name
_ALIASES: dict[str, str] = {
    "rm": "remove",
    "ls": "list",
    "regen": "regenerate",
    "s": "search",
    "idx": "index",
}


class AliasGroup(click.Group):
    """Click Group that supports hidden command aliases."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # Resolve alias to canonical name
        canonical = _ALIASES.get(cmd_name, cmd_name)
        return super().get_command(ctx, canonical)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Show aliases inline with their canonical commands
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            # Find aliases for this command
            aliases = [alias for alias, canon in _ALIASES.items() if canon == subcommand]
            if aliases:
                name_col = ", ".join([subcommand, *sorted(aliases)])
            else:
                name_col = subcommand
            commands.append((name_col, cmd))

        if commands:
            limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
            rows = []
            for name_col, cmd in commands:
                help_text = cmd.get_short_help_str(limit=limit)
                rows.append((name_col, help_text))

            if rows:
                with formatter.section("Commands"):
                    formatter.write_dl(rows)


@click.group(cls=AliasGroup)
@click.version_option(version=_cli_version())
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress messages.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug output.")
def cli(quiet: bool = False, verbose: bool = False):
    """paperpipe: Unified paper database for coding agents + PaperQA2."""
    set_quiet(quiet)
    if verbose:
        _setup_debug_logging()
    ensure_db()


# Register all commands (aliases resolved via AliasGroup.get_command)
cli.add_command(add)
cli.add_command(regenerate)
cli.add_command(list_papers, name="list")
cli.add_command(search)
cli.add_command(audit)
cli.add_command(export)
cli.add_command(index_cmd, name="index")
cli.add_command(ask)
cli.add_command(models)
cli.add_command(show)
cli.add_command(notes)
cli.add_command(remove)
cli.add_command(tags)
cli.add_command(path)
cli.add_command(install)
cli.add_command(uninstall)
cli.add_command(docs)
cli.add_command(rebuild_index)
