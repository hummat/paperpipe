---
description: Turn papi (summary/equations/tex) excerpts into a project note with citations.
argument-hint: [project-context...]
---
You are given paper excerpts (attach files with `@...`, or paste `papi show ... --level ...` output above).

Project context (optional): $ARGUMENTS

Create a project-focused note in markdown.

Include:
- Core claims (bulleted)
- Method sketch (key equations/pseudocode; keep symbol names exact)
- Evaluation (datasets, metrics, main numbers, compute)
- Limitations and failure modes
- Implementation checklist for the project (only if project context is provided)
- Canonical quote snippets (<= 15 words each) with citations:
  (paper: <name>, arXiv: <id if present>, source: summary|equations|tex|notes, ref: section/eq/table/figure if present)

If multiple papers are included, structure sections per paper and add a short cross-paper synthesis.

Return the complete markdown only.
