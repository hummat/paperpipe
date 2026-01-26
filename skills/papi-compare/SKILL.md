---
name: papi-compare
description: Compare papers for a decision. Use when user asks "which paper should I use", "compare approaches", or needs to choose between methods/algorithms.
---

# Compare Papers for Decision

You are given paper excerpts (paste `papi show ... --level ...` output above, or reference exported files).

Project context (optional): $ARGUMENTS

## Task

Compare the papers for a decision in this project context.

**Axes**: objective, assumptions/data regime, compute, latency, robustness, eval metrics, reproducibility, implementation risk.

## Output

1) **Decision matrix** (table)
2) **Recommended choice** + assumptions + risks
3) **Missing evidence** / what to fetch next (e.g., ask for `papi show <paper> --level tex`)

## Citations

For each non-trivial claim, include a short quote snippet (<= 15 words) and cite as:
(paper: <name>, arXiv: <id if present>, source: summary|equations|tex|notes, ref: section/eq/table/figure if present)

End with:
**Cited papers**: <comma-separated paper names and/or arXiv IDs>
