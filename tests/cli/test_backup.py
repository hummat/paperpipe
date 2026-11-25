"""Tests for paperpipe/backup.py and the backup command."""

from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from paperpipe.backup import compress_pdf

from .conftest import cli_mod

pymupdf = pytest.importorskip("pymupdf")

TEXT = "Neural radiance fields for view synthesis"


def _write_pdf_with_photo(path: Path) -> None:
    """One page of text plus a 1200x1200 noise image drawn 2 inches wide (600 dpi), stored losslessly."""
    side = 1200
    samples = random.Random(0).randbytes(side * side * 3)
    pix = pymupdf.Pixmap(pymupdf.csRGB, side, side, samples, False)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), TEXT)
    page.insert_image(pymupdf.Rect(72, 100, 216, 244), pixmap=pix)
    doc.save(path)


@pytest.mark.parametrize("level", ["high", "medium", "text"])
def test_compress_pdf_shrinks_images_and_keeps_text(tmp_path: Path, level: str):
    pdf = tmp_path / "paper.pdf"
    _write_pdf_with_photo(pdf)

    data = compress_pdf(pdf, level)

    assert len(data) < pdf.stat().st_size, "compressed PDF should be smaller than the lossless original"
    with pymupdf.open(stream=data) as doc:
        assert TEXT in doc[0].get_text(), "text must survive compression"
        if level == "text":
            assert doc[0].get_images() == [], "text level drops raster images"


def test_backup_archives_db_with_compressed_pdfs(temp_db: Path, tmp_path: Path, runner: CliRunner):
    photo = temp_db / "papers" / "photo"
    photo.mkdir(parents=True)
    _write_pdf_with_photo(photo / "paper.pdf")
    (photo / "meta.json").write_text(json.dumps({"title": "Photo"}))
    broken = temp_db / "papers" / "broken"
    broken.mkdir(parents=True)
    (broken / "paper.pdf").write_bytes(b"not a pdf")
    original_size = (photo / "paper.pdf").stat().st_size

    (temp_db / "search.db").write_bytes(b"sqlite")
    (temp_db / "index.json.backup.20260101_000000").write_text("{}")
    (temp_db / ".pqa_papers").mkdir()
    (temp_db / ".pqa_papers" / "photo.pdf").symlink_to("../papers/photo/paper.pdf")
    manifest_path = temp_db / ".leann" / "indexes" / "papers" / "paperpipe_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "version": 1,
        "files": {
            "photo.pdf": {"mtime": 1.0, "size": original_size, "indexed_at": "", "status": "ok"},
            "broken.pdf": {"mtime": 1.0, "size": 9, "indexed_at": "", "status": "ok"},
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "backup.tar"

    result = runner.invoke(cli_mod.cli, ["backup", str(out), "--level", "medium"])

    assert result.exit_code == 0, result.output
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
        photo_bytes = tar.extractfile("papers/photo/paper.pdf").read()  # type: ignore[union-attr]
        broken_bytes = tar.extractfile("papers/broken/paper.pdf").read()  # type: ignore[union-attr]
        archived_manifest = json.load(tar.extractfile(".leann/indexes/papers/paperpipe_manifest.json"))  # type: ignore[arg-type]
        assert tar.getmember(".pqa_papers/photo.pdf").issym(), "staging symlinks stay symlinks"

    assert {"index.json", "papers/photo/meta.json"} <= names
    assert "search.db" not in names and "index.json.backup.20260101_000000" not in names
    assert len(photo_bytes) < original_size
    assert broken_bytes == b"not a pdf", "an unreadable PDF is archived unchanged"
    assert archived_manifest["files"]["photo.pdf"]["size"] == len(photo_bytes), (
        "LEANN manifest must match the compressed size, or the next index run re-embeds the paper"
    )
    assert archived_manifest["files"]["broken.pdf"]["size"] == 9
    assert not out.with_name("backup.tar.partial").exists()
