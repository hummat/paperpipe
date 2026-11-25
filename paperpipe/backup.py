"""Database backup: a portable tar archive of the paper DB, with optional lossy PDF compression."""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Optional

from . import config
from .leann import _migrate_manifest_key
from .output import echo_progress

# Compression levels for papers/*/paper.pdf. Only raster images change; text and vector graphics stay intact.
# Measured on 16 PDFs spread over a 1336-paper DB (archived size vs. original): high 58%, medium 38%, text 13%.
# Lossless repacking saved ~1%, so there is no lossless level.
LEVELS = ("off", "high", "medium", "text")

# Root-level files left out of the archive: search.db (plus SQLite WAL sidecars) is rebuilt for free by
# `papi index --backend search`; index.json.backup.* are snapshots left behind by rebuild-index and tag edits.
_EXCLUDED_ROOT_PREFIXES = ("search.db", "index.json.backup.")


@dataclass
class BackupStats:
    files: int = 0
    pdfs_compressed: int = 0
    pdf_bytes_before: int = 0
    pdf_bytes_after: int = 0
    failed: list[str] = field(default_factory=list)  # "<paper> (<error>)" for PDFs MuPDF could not rewrite


def compress_pdf(pdf: Path, level: str) -> bytes:
    """Return ``pdf`` rewritten at ``level`` (one of ``high``, ``medium``, ``text``)."""
    import pymupdf
    from pymupdf.mupdf import PDF_REDACT_IMAGE_REMOVE, PDF_REDACT_LINE_ART_NONE, PDF_REDACT_TEXT_NONE

    with pymupdf.open(pdf) as doc:
        if level == "high":
            doc.rewrite_images(quality=95)
        elif level == "medium":
            doc.rewrite_images(dpi_threshold=330, dpi_target=300, quality=85)
        elif level == "text":
            for page in doc:
                page.add_redact_annot(page.rect)
                page.apply_redactions(
                    images=PDF_REDACT_IMAGE_REMOVE,
                    graphics=PDF_REDACT_LINE_ART_NONE,
                    text=PDF_REDACT_TEXT_NONE,
                )
        else:
            raise ValueError(f"Unknown compression level: {level}")
        return doc.tobytes(garbage=3, deflate=True, use_objstms=True)


def _compress_job(pdf: Path, level: str) -> tuple[Path, Optional[bytes], Optional[str]]:
    # MuPDF raises a variety of exception types on malformed PDFs; any of them means "archive the original".
    try:
        return pdf, compress_pdf(pdf, level), None
    except Exception as e:
        return pdf, None, f"{type(e).__name__}: {e}"


def _compressed(pdfs: list[Path], level: str, workers: int) -> Iterator[tuple[Path, Optional[bytes], Optional[str]]]:
    """Yield compression results as they finish, keeping at most 2 * workers PDFs in flight to bound memory."""
    jobs = iter(pdfs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: set[Future] = {pool.submit(_compress_job, pdf, level) for pdf in islice(jobs, 2 * workers)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                next_pdf = next(jobs, None)
                if next_pdf is not None:
                    pending.add(pool.submit(_compress_job, next_pdf, level))


def _backup_files(root: Path) -> list[Path]:
    """All files and symlinks under ``root`` that belong in a backup."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        names = filenames + [d for d in dirnames if (base / d).is_symlink()]
        for name in names:
            if base == root and name.startswith(_EXCLUDED_ROOT_PREFIXES):
                continue
            files.append(base / name)
    return sorted(files)


def _patched_leann_manifest(path: Path, new_sizes: dict[str, int]) -> bytes:
    """Record compressed PDF sizes so LEANN's incremental indexing does not re-embed unchanged text."""
    manifest = json.loads(path.read_text())
    for key, entry in manifest.get("files", {}).items():
        size = new_sizes.get(_migrate_manifest_key(key))
        if size is not None:
            entry["size"] = size
    return json.dumps(manifest, indent=2).encode()


def _add_bytes(tar: tarfile.TarFile, source: Path, arcname: str, data: bytes) -> None:
    info = tar.gettarinfo(source, arcname=arcname)  # keeps mode and mtime of the original file
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_backup(out: Path, *, level: str, workers: Optional[int] = None) -> BackupStats:
    """Write the database directory to an uncompressed tar archive at ``out``.

    Paths in the archive are relative to the database root. PDFs that do not shrink, or that MuPDF cannot
    rewrite, are archived unchanged. The archive is written to ``<out>.partial`` and renamed on success.
    """
    root = config.PAPER_DB
    leann_indexes = root / ".leann" / "indexes"
    files = _backup_files(root)
    pdfs = (
        [p for p in files if p.name == "paper.pdf" and p.parent.parent == config.PAPERS_DIR] if level != "off" else []
    )
    pdf_set = set(pdfs)
    stats = BackupStats()
    new_sizes: dict[str, int] = {}  # staging name ({paper}.pdf, the LEANN manifest key) -> compressed size

    partial = out.with_name(out.name + ".partial")
    try:
        with tarfile.open(partial, "w") as tar:
            if pdfs:
                workers = max(1, min(workers or os.cpu_count() or 1, len(pdfs)))
                echo_progress(f"Compressing {len(pdfs)} PDF(s) at level '{level}' with {workers} worker(s)...")
                step = max(1, len(pdfs) // 20)
                for done, (pdf, data, error) in enumerate(_compressed(pdfs, level, workers), start=1):
                    arcname = pdf.relative_to(root).as_posix()
                    size = pdf.stat().st_size
                    stats.pdf_bytes_before += size
                    if error is not None:
                        stats.failed.append(f"{pdf.parent.name} ({error})")
                    if data is not None and len(data) < size:
                        _add_bytes(tar, pdf, arcname, data)
                        new_sizes[f"{pdf.parent.name}.pdf"] = len(data)
                        stats.pdfs_compressed += 1
                        stats.pdf_bytes_after += len(data)
                    else:
                        tar.add(pdf, arcname=arcname, recursive=False)
                        stats.pdf_bytes_after += size
                    stats.files += 1
                    if done % step == 0 or done == len(pdfs):
                        echo_progress(f"  {done}/{len(pdfs)} PDFs")

            for path in files:
                if path in pdf_set:
                    continue
                arcname = path.relative_to(root).as_posix()
                if new_sizes and path.name == "paperpipe_manifest.json" and path.parent.parent == leann_indexes:
                    _add_bytes(tar, path, arcname, _patched_leann_manifest(path, new_sizes))
                else:
                    tar.add(path, arcname=arcname, recursive=False)
                stats.files += 1
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)
    return stats
