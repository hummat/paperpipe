#!/usr/bin/env python3
"""
MCP server for retrieval-only search over PaperQA2's on-disk index.

This server is optimized for "fast RAG": it returns raw retrieved chunks (with
citations) so the client model can do synthesis, avoiding PaperQA2's full agent
loop (LLM summarization + answer generation).

It reuses the same PaperQA2 index that `papi ask` / `papi index` maintain under
`~/.paperpipe/.pqa_index/` by default (no duplicate embeddings).

Usage:
    # Install with MCP support
    pip install "paperpipe[mcp]"

    # Run directly (for testing)
    python paperqa_mcp_server.py

    # Add to your agent (recommended)
    papi install mcp

Configuration via environment variables:
    PAPERPIPE_PQA_INDEX_DIR   - Root dir containing PaperQA2 indices (default: ~/.paperpipe/.pqa_index)
    PAPERPIPE_PQA_INDEX_NAME  - Index name to query (default: paperpipe_<embedding_model>)
    PAPERQA_EMBEDDING         - Embedding model used to embed the query (default: from paperpipe config)
    PAPERQA_LLM               - Unused for retrieval-only tools

Requires Python 3.11+ (paper-qa requirement).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP  # type: ignore[reportMissingImports]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports to defer heavy dependencies
_mcp: "FastMCP | None" = None


def _get_mcp() -> "FastMCP":
    """Lazily import and create MCP server."""
    global _mcp
    if _mcp is None:
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[reportMissingImports]
        except ImportError as e:
            print(
                "Error: MCP not installed. Install with: pip install 'paperpipe[mcp]'",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        _mcp = FastMCP("paperqa", instructions="Retrieval-only search over PaperQA2 index (fast RAG)")
    return _mcp


def _require_python311() -> None:
    if sys.version_info < (3, 11):
        print("Error: paper-qa MCP server requires Python 3.11+", file=sys.stderr)
        raise SystemExit(1)


def _paperpipe_module() -> Any | None:
    try:
        import paperpipe
    except Exception:
        return None
    return paperpipe


def _default_index_root() -> Path:
    configured = os.getenv("PAPERPIPE_PQA_INDEX_DIR") or os.getenv("PAPERQA_INDEX_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    if (pp := _paperpipe_module()) is not None:
        try:
            return pp.default_pqa_index_dir()
        except Exception:
            pass
    return Path("~/.paperpipe/.pqa_index").expanduser()


def _default_embedding_model() -> str:
    configured = os.getenv("PAPERQA_EMBEDDING")
    if configured and configured.strip():
        return configured.strip()
    if (pp := _paperpipe_module()) is not None:
        try:
            return pp.default_pqa_embedding_model()
        except Exception:
            pass
    configured2 = os.getenv("PAPERPIPE_EMBEDDING_MODEL")
    if configured2 and configured2.strip():
        return configured2.strip()
    return "text-embedding-3-small"


def _index_name_for_embedding(embedding_model: str) -> str:
    if (pp := _paperpipe_module()) is not None:
        try:
            return pp.pqa_index_name_for_embedding(embedding_model)
        except Exception:
            pass
    safe_name = (embedding_model or "").replace("/", "_").replace(":", "_").strip() or "default"
    return f"paperpipe_{safe_name}"


def _default_index_name(embedding_model: str) -> str:
    configured = os.getenv("PAPERPIPE_PQA_INDEX_NAME") or os.getenv("PAPERQA_INDEX_NAME")
    if configured and configured.strip():
        return configured.strip()
    return _index_name_for_embedding(embedding_model)


def _index_meta_exists(index_root: Path, index_name: str) -> bool:
    return (index_root / index_name / "index" / "meta.json").exists()


def _load_files_zip_map(files_zip: Path) -> dict[str, str] | None:
    """Load PaperQA2's index file map (zlib-compressed pickle)."""
    try:
        import pickle

        raw = zlib.decompress(files_zip.read_bytes())
        obj = pickle.loads(raw)  # noqa: S301 - PaperQA2 format
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _load_embedding_from_metadata(index_root: Path, index_name: str) -> str | None:
    """
    Load embedding model from paperpipe_meta.json if exists.

    Returns None if file doesn't exist or is invalid.
    """
    meta_path = index_root / index_name / "paperpipe_meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        return data.get("embedding_model") or None
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in %s: %s", meta_path, e)
        return None
    except (OSError, IOError) as e:
        logger.warning("Could not read %s: %s", meta_path, e)
        return None


def _write_index_metadata(index_root: Path, index_name: str, embedding_model: str) -> None:
    """
    Write paperpipe metadata after index creation.

    Creates paperpipe_meta.json with embedding model, timestamp, and version.
    Callers should wrap in try/except to ensure index operations succeed regardless of metadata write failures.
    """
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        version = get_version("paperpipe")
    except (ImportError, PackageNotFoundError):
        version = "unknown"

    from datetime import datetime, timezone

    metadata = {
        "embedding_model": embedding_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paperpipe_version": version,
    }
    meta_path = index_root / index_name / "paperpipe_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2))


def _write_leann_metadata(index_name: str, embedding_mode: str, embedding_model: str) -> None:
    """
    Write paperpipe metadata for LEANN index.

    Creates paperpipe_leann_meta.json alongside LEANN's documents.leann.meta.json.
    """
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        version = get_version("paperpipe")
    except (ImportError, PackageNotFoundError):
        version = "unknown"

    from datetime import datetime, timezone

    if (pp := _paperpipe_module()) is None:
        logger.warning("Cannot write LEANN metadata: paperpipe module not available")
        return

    index_dir = pp.PAPER_DB / ".leann" / "indexes" / index_name
    metadata = {
        "embedding_mode": embedding_mode,
        "embedding_model": embedding_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paperpipe_version": version,
    }
    meta_path = index_dir / "paperpipe_leann_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2))


def _register_tools() -> None:
    """Register all MCP tools. Must be called after _get_mcp()."""
    mcp = _get_mcp()

    @mcp.tool()
    async def retrieve_chunks(
        query: str,
        k: int = 10,
        search_k: int = 20,
        embedding_model: str | None = None,
        index_dir: str | None = None,
        index_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve raw chunks from the PaperQA2 index (no LLM summarization/answering).

        Args:
            query: Search query.
            k: Maximum number of chunks to return.
            search_k: Number of candidate papers to pull from the index first (lexical prefilter).
            embedding_model: LiteLLM embedding id; optional, auto-inferred from index metadata if not provided.
            index_dir: Override PaperQA2 index root directory (contains index subfolders).
            index_name: Override PaperQA2 index name (subfolder under index_dir).

        Returns:
            JSON payload with retrieved chunks and citations.
        """
        _require_python311()

        index_root = Path(index_dir).expanduser() if index_dir else _default_index_root()

        # If index_name provided, try to infer embedding from it
        if (index_name or "").strip():
            index_name = (index_name or "").strip()

            # Try metadata first
            inferred_embedding = _load_embedding_from_metadata(index_root, index_name)

            # Fall back to name parsing
            if not inferred_embedding:
                if (pp := _paperpipe_module()) is not None:
                    try:
                        inferred_embedding = pp.embedding_model_from_index_name(index_name)
                    except Exception as e:
                        logger.debug("Could not infer embedding from index name %r: %s", index_name, e)

            # Use inferred if no explicit parameter
            if not (embedding_model or "").strip() and inferred_embedding:
                embedding_model = inferred_embedding
            else:
                embedding_model = (embedding_model or "").strip() or _default_embedding_model()
        else:
            # Standard flow: embedding determines index name
            embedding_model = (embedding_model or "").strip() or _default_embedding_model()
            index_name = _default_index_name(embedding_model)

        index_path = index_root / index_name
        files_zip = index_path / "files.zip"

        if not _index_meta_exists(index_root, index_name) or not files_zip.exists():
            return {
                "ok": False,
                "error": "PaperQA2 index not found. Run `papi index` (or `papi ask ...`) to build it, or set index_dir/index_name.",
                "index": {"directory": str(index_root), "name": index_name, "embedding_model": embedding_model},
            }

        try:
            from paperqa.agents.search import SearchIndex  # type: ignore[reportMissingImports]
            from paperqa.docs import Docs  # type: ignore[reportMissingImports]
            from paperqa.settings import Settings  # type: ignore[reportMissingImports]
        except Exception as e:
            return {"ok": False, "error": f"paper-qa import failed: {type(e).__name__}: {e}"}

        # LLM fields are irrelevant for retrieval-only, but Settings requires them.
        llm_model = (os.getenv("PAPERQA_LLM") or "gpt-4o-mini").strip()
        # Ensure embedding_model is always a string (should never be None after inference logic above)
        embedding_model_str = embedding_model or _default_embedding_model()
        settings = Settings(llm=llm_model, summary_llm=llm_model, embedding=embedding_model_str)
        embedding = settings.get_embedding_model()

        try:
            search_index = SearchIndex(
                fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
                index_name=index_name,
                index_directory=str(index_root),
            )

            # 1) Lexical prefilter at the paper level
            results = await search_index.query(
                query,
                top_n=max(1, min(int(search_k), 200)),
                field_subset=[f for f in search_index.fields if f != "year"],
            )

            # 2) Combine candidate papers into a single in-memory Docs for vector retrieval
            combined = Docs()
            for r in results:
                try:
                    doc = next(iter(r.docs.values()))
                except Exception:
                    continue
                # aadd_texts preserves existing embeddings when present
                await combined.aadd_texts(texts=r.texts, doc=doc, settings=settings, embedding_model=embedding)

            # 3) Vector retrieval (MMR) for the user query
            matches = await combined.retrieve_texts(
                query, k=max(1, min(int(k), 50)), settings=settings, embedding_model=embedding
            )
            chunks: list[dict[str, Any]] = []
            for i, t in enumerate(matches, 1):
                doc = t.doc
                citation = getattr(doc, "formatted_citation", None) or getattr(doc, "citation", None) or ""
                chunks.append(
                    {
                        "rank": i,
                        "text": t.text,
                        "chunk_name": t.name,
                        "docname": getattr(doc, "docname", None) or "",
                        "citation": citation,
                    }
                )

            mapping = _load_files_zip_map(files_zip) or {}
            failed = sum(1 for v in mapping.values() if v == "ERROR")
            return {
                "ok": True,
                "query": query,
                "index": {"directory": str(index_root), "name": index_name, "embedding_model": embedding_model},
                "candidate_papers": len(results),
                "chunks": chunks,
                "index_files": {"total": len(mapping), "failed": failed},
            }
        except Exception as e:
            logger.exception("retrieve_chunks failed")
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "index": {"directory": str(index_root), "name": index_name, "embedding_model": embedding_model},
            }

    @mcp.tool()
    async def list_pqa_indexes(index_dir: str | None = None) -> list[dict[str, Any]]:
        """
        List available PaperQA2 indexes with metadata.

        Returns a list of dicts with index information including embedding model.
        """
        index_root = Path(index_dir).expanduser() if index_dir else _default_index_root()
        try:
            if not index_root.exists():
                return []
            out: list[dict[str, Any]] = []
            for child in sorted(index_root.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "index" / "meta.json").exists():
                    # Try to load embedding model from metadata
                    embedding = _load_embedding_from_metadata(index_root, child.name)

                    # Fall back to name parsing if metadata doesn't exist
                    if not embedding:
                        if (pp := _paperpipe_module()) is not None:
                            try:
                                embedding = pp.embedding_model_from_index_name(child.name)
                            except Exception:
                                pass

                    out.append(
                        {
                            "name": child.name,
                            "embedding_model": embedding,
                            "has_metadata": (child / "paperpipe_meta.json").exists(),
                        }
                    )
            return out
        except PermissionError as e:
            logger.warning("Cannot access index directory %s: %s", index_root, e)
            return []
        except Exception:
            logger.exception("Unexpected error listing PaperQA2 indexes")
            return []

    @mcp.tool()
    async def get_pqa_index_status(
        index_dir: str | None = None, index_name: str | None = None, embedding_model: str | None = None
    ) -> dict[str, Any]:
        """Return basic status info about the PaperQA2 index (no heavy imports)."""
        embedding_model = (embedding_model or "").strip() or _default_embedding_model()
        index_root = Path(index_dir).expanduser() if index_dir else _default_index_root()
        index_name = (index_name or "").strip() or _default_index_name(embedding_model)
        index_path = index_root / index_name
        files_zip = index_path / "files.zip"
        mapping = _load_files_zip_map(files_zip) if files_zip.exists() else None
        return {
            "index": {"directory": str(index_root), "name": index_name, "embedding_model": embedding_model},
            "exists": _index_meta_exists(index_root, index_name),
            "files_zip_exists": files_zip.exists(),
            "index_files_total": len(mapping or {}),
            "index_files_failed": sum(1 for v in (mapping or {}).values() if v == "ERROR"),
        }

    @mcp.tool()
    async def leann_search(
        index_name: str,
        query: str,
        top_k: int = 5,
        complexity: int = 32,
        show_metadata: bool = False,
    ) -> dict[str, Any]:
        """
        Semantic search over papers using LEANN.

        Faster and simpler than PaperQA2 retrieve_chunks.

        Args:
            index_name: Name of the LEANN index to search.
            query: Search query (natural language or technical terms).
            top_k: Number of results (1-20).
            complexity: Search complexity level (16-128, default 32).
            show_metadata: Include file paths and metadata.

        Returns:
            Search results from LEANN.
        """
        import shutil

        if not shutil.which("leann"):
            return {"ok": False, "error": "LEANN not installed. Run: pip install 'paperpipe[leann]'"}

        if not index_name or not query:
            return {"ok": False, "error": "Both index_name and query are required"}

        cmd = [
            "leann",
            "search",
            index_name,
            query,
            f"--top-k={max(1, min(int(top_k), 20))}",
            f"--complexity={max(16, min(int(complexity), 128))}",
            "--non-interactive",
        ]
        if show_metadata:
            cmd.append("--show-metadata")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {
                "ok": result.returncode == 0,
                "output": result.stdout if result.returncode == 0 else result.stderr,
                "index_name": index_name,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "LEANN search timed out (30s)"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @mcp.tool()
    async def leann_list() -> dict[str, Any]:
        """
        List available LEANN indexes with metadata.

        Returns list of dicts with index information including embedding model.
        """
        import shutil

        if not shutil.which("leann"):
            return {"ok": False, "error": "LEANN not installed. Run: pip install 'paperpipe[leann]'"}

        if (pp := _paperpipe_module()) is None:
            return {"ok": False, "error": "Could not load paperpipe config"}

        index_root = pp.PAPER_DB / ".leann" / "indexes"
        if not index_root.exists():
            return {"ok": True, "indexes": []}

        indexes: list[dict[str, Any]] = []
        try:
            for child in sorted(index_root.iterdir()):
                if not child.is_dir():
                    continue
                if not (child / "documents.leann.meta.json").exists():
                    continue

                # Try to load embedding info from paperpipe metadata
                meta_path = child / "paperpipe_leann_meta.json"
                embedding_mode = None
                embedding_model = None
                has_metadata = False

                if meta_path.exists():
                    try:
                        data = json.loads(meta_path.read_text())
                        embedding_mode = data.get("embedding_mode")
                        embedding_model = data.get("embedding_model")
                        has_metadata = True
                    except json.JSONDecodeError as e:
                        logger.warning("Invalid JSON in %s: %s", meta_path, e)
                        has_metadata = False
                    except (OSError, IOError) as e:
                        logger.warning("Could not read %s: %s", meta_path, e)
                        has_metadata = False

                indexes.append(
                    {
                        "name": child.name,
                        "embedding_mode": embedding_mode,
                        "embedding_model": embedding_model,
                        "has_metadata": has_metadata,
                    }
                )

            return {"ok": True, "indexes": indexes}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    """Run the MCP server."""
    _require_python311()

    embedding_model = _default_embedding_model()
    index_root = _default_index_root()
    index_name = _default_index_name(embedding_model)

    logger.info("Starting PaperQA2 retrieval MCP server")
    logger.info("Index root: %s", index_root)
    logger.info("Index name: %s", index_name)
    logger.info("Embedding: %s", embedding_model)

    _register_tools()
    _get_mcp().run()


if __name__ == "__main__":
    main()
