"""Output helpers and debug logging."""

from __future__ import annotations

import logging
import sys

import click

# Simple debug logger (only used with --verbose)
_debug_logger = logging.getLogger("paperpipe")
_debug_logger.addHandler(logging.NullHandler())


def _setup_debug_logging() -> None:
    """Enable debug logging to stderr."""
    _debug_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    _debug_logger.addHandler(handler)


# Output helpers that respect --quiet mode
_quiet_mode = False


def set_quiet(quiet: bool) -> None:
    global _quiet_mode
    _quiet_mode = quiet


def echo(message: str = "", err: bool = False) -> None:
    """Print a message (respects --quiet for non-error messages)."""
    if _quiet_mode and not err:
        return
    click.echo(message, err=err)


def echo_success(message: str) -> None:
    """Print a success message in green."""
    click.secho(message, fg="green")


def echo_error(message: str) -> None:
    """Print an error message in red to stderr."""
    click.secho(message, fg="red", err=True)


def echo_warning(message: str) -> None:
    """Print a warning message in yellow to stderr."""
    click.secho(message, fg="yellow", err=True)


def echo_progress(message: str) -> None:
    """Print a progress message (suppressed in quiet mode)."""
    if not _quiet_mode:
        click.echo(message)


def debug(message: str, *args: object) -> None:
    """Log a debug message (only shown with --verbose)."""
    _debug_logger.debug(message, *args)
