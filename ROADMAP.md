# Roadmap

This file tracks planned features and intended CLI surface area for paperpipe (`papi`).
It is not a commitment to specific timelines.

## Principles

- Prefer one mental model: `papi add` adds papers (arXiv or local files).
- Keep the local database format stable and easy to inspect/edit.
- Avoid API-heavy features unless they are clearly optional and cached.
- Precedence for configuration: **CLI flags > env vars > config.toml > defaults**.

## Planned (next)

### 1) `papi attach` (upgrade/attach files)

Goal: let users fix missing/low-quality assets after initial ingest.

- **Command**
  - `papi attach PAPER --pdf /path/to/better.pdf`
  - `papi attach PAPER --source /path/to/main.tex`
- **Behavior**
  - Replace/attach the specified file(s)
  - Update `meta.json` (`has_pdf` / `has_source`)
  - Optionally regenerate dependent artifacts (e.g., `equations.md` when source changes)
- **Options (TBD)**
  - `--regen equations|summary|tags|all`
  - `--backup` (keep `paper.pdf.bak`, `source.tex.bak`, etc.)

### 2) `papi bibtex` (export)

Goal: easy citation export that integrates with LaTeX workflows.

- **Command**
  - `papi bibtex PAPER...`
- **Output**
  - Prints BibTeX entries derived from stored metadata.
- **Options (TBD)**
  - `--to library.bib` (write/append)
  - `--key-style name|doi|arxiv|slug`

### 3) `papi import-bib` (bulk ingest)

Goal: bootstrap a library from an existing BibTeX file.

- **Command**
  - `papi import-bib /path/to/library.bib`
- **Dependency**
  - Use `bibtexparser` (BibTeX is irregular; avoid hand-rolled parsing).
- **Behavior (MVP)**
  - Create metadata-only paper entries (PDF can be attached later with `papi attach`)
  - Dedup/match order: `doi` > `arxiv_id` > bibtex key
- **Options (TBD)**
  - `--dry-run`
  - `--update-existing`
  - `--tag TAG` (apply to all imported)
  - `--name-from key|slug(title)`

## Later (after the above stabilizes)

- `papi rename OLD NEW` (safe rename + index/meta updates)
- `papi rebuild-index` (recover `index.json` from on-disk state)
- `papi stats` (tags over time, has_pdf/has_source, storage usage)
- arXiv version tracking + update checks (`papi check-updates`, `papi update`)
- `papi diff` (start as text diff; avoid semantic parsing in MVP)

## Out of scope for now (high scope creep)

- Citation graph / related paper discovery across multiple APIs
- Semantic embedding search with a dedicated local vector index
- Watch/notifications for new papers
- Zotero/Mendeley integration

## Completed

### Non-arXiv ingestion via `papi add --pdf` (MVP)

Implemented (see `README.md` → “Non-arXiv Papers” for usage and examples).
