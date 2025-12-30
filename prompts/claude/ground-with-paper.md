---
description: Ground responses strictly in papi-provided paper excerpts; cite with paper name + arXiv ID.
---
Ground the response strictly in the following excerpts from the local paper database (produced by `papi show` / `papi export`):
{{refs}}

Rules:
- If a claim is not supported by the provided excerpts, say: "Not supported by provided excerpts."
- For supported claims, include a short quote snippet (<= 15 words) and cite as:
  (paper: <name>, arXiv: <id if present>, source: summary|equations|tex|notes, ref: section/eq/table/figure if present)
- Call out assumptions, dataset/compute constraints, and symbol definitions.
- Prefer equations/LaTeX source over summaries when there’s a conflict.

End with:
Cited papers: <comma-separated paper names and/or arXiv IDs>
