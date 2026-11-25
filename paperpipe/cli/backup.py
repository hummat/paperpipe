"""Backup command: write the database to a tar archive."""

from __future__ import annotations

from pathlib import Path

import click

from .. import config
from ..backup import LEVELS, write_backup
from ..output import echo_success, echo_warning


@click.command()
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--level",
    type=click.Choice(LEVELS, case_sensitive=False),
    default="off",
    show_default=True,
    help="Shrink PDF images: high (JPEG q95), medium (max 300 dpi, JPEG q85), text (drop raster images).",
)
def backup(out: Path, level: str) -> None:
    """Write the database to the tar archive OUT.

    Archive paths are relative to the database root. Embedding indexes are included, so a restored
    database does not need re-embedding; search.db and index.json backups are left out.
    """
    out = out.expanduser().resolve()
    level = level.lower()
    if out.exists():
        raise click.UsageError(f"{out} already exists.")
    if not out.parent.is_dir():
        raise click.UsageError(f"Directory does not exist: {out.parent}")
    if out.is_relative_to(config.PAPER_DB.resolve()):
        raise click.UsageError("OUT must be outside the database directory.")
    if level != "off":
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            raise click.ClickException("Compression needs PyMuPDF: pip install 'paperpipe[figures]'") from None

    stats = write_backup(out, level=level)

    for failure in stats.failed:
        echo_warning(f"Archived unchanged (could not compress): {failure}")
    if level != "off":
        echo_success(
            f"PDFs: {stats.pdf_bytes_before / 2**20:,.0f} MB -> {stats.pdf_bytes_after / 2**20:,.0f} MB "
            f"({stats.pdfs_compressed} compressed)"
        )
    echo_success(f"Wrote {stats.files} files to {out} ({out.stat().st_size / 2**20:,.0f} MB)")
