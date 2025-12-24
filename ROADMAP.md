# Roadmap

This file tracks planned features and intended CLI surface area for paperpipe (`papi`).
It is not a commitment to specific timelines.

## Principles

- Prefer one mental model: `papi add` adds papers (arXiv or local files).
- Keep the local database format stable and easy to inspect/edit.
- Avoid API-heavy features unless they are clearly optional and cached.
- Precedence for configuration: **CLI flags > env vars > config.toml > defaults**.

## Planned (next)

### 1) Non-arXiv ingestion via `papi add --pdf`

Goal: treat local PDFs as first-class papers (list/search/export/ask) without requiring arXiv.

- **Command**
  - `papi add --pdf /path/to/paper.pdf --title "Some Paper" [options]`
- **Required**
  - `--pdf PATH`
  - `--title TEXT` (used to auto-generate a stable paper name, consistent with arXiv adds)
- **Name generation**
  - Default: slugify title (optionally LLM-shortened, like arXiv auto-names)
  - Override: `--name NAME`
- **Optional metadata**
  - `--authors "A; B; C"` (or a repeatable flag; TBD)
  - `--year YYYY`
  - `--venue TEXT`
  - `--doi DOI`
  - `--url URL`
  - `--tags t1,t2`
- **Storage**
  - Create/ensure `<paper_db>/papers/<name>/`
  - Copy PDF to `paper.pdf`
  - Write `meta.json` with `arxiv_id: null` and `has_pdf: true`
  - Create `notes.md` if missing
  - Best-effort `summary.md` / `equations.md`:
    - If a LaTeX source is not available, equations extraction may be minimal.
- **Constraints / non-goals (MVP)**
  - No DOI/OpenReview/Crossref/Semantic Scholar lookup in the MVP.
  - Name conflicts must fail fast with a clear error.

### 2) `papi attach` (upgrade/attach files)

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

### 3) `papi bibtex` (export)

Goal: easy citation export that integrates with LaTeX workflows.

- **Command**
  - `papi bibtex PAPER...`
- **Output**
  - Prints BibTeX entries derived from stored metadata.
- **Options (TBD)**
  - `--to library.bib` (write/append)
  - `--key-style name|doi|arxiv|slug`

### 4) `papi import-bib` (bulk ingest)

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
