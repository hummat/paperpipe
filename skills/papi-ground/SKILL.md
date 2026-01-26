---
name: papi-ground
description: Ground responses in paper excerpts with citations. Use when user wants cited claims, quotes, or needs to avoid hallucination about paper content.
---

# Ground Responses in Paper Excerpts

You are given paper excerpts (paste `papi show ... --level ...` output above, or reference exported files).

## Rules

- If a claim is not supported by the provided excerpts, say: "Not supported by provided excerpts."
- For supported claims, include a short quote snippet (<= 15 words) and cite as:
  (paper: <name>, arXiv: <id if present>, source: summary|equations|tex|notes, ref: section/eq/table/figure if present)
- Call out assumptions, dataset/compute constraints, and symbol definitions.
- Prefer equations/LaTeX source over summaries when there's a conflict.

End with:
**Cited papers**: <comma-separated paper names and/or arXiv IDs>
