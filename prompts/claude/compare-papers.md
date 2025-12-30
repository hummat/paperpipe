---
description: Compare multiple papers for a decision (prefer using papi exports as input).
---
You are given excerpts exported from paperpipe via `papi show` / `papi export`:
{{refs}}

Task:
- Compare the papers for a decision in this project context: {{context}}
- Axes: objective, assumptions/data regime, compute, latency, robustness, eval metrics, reproducibility, implementation risk.

Output:
1) Decision matrix (table)
2) Recommended choice + assumptions + risks
3) What evidence is missing / what to fetch next (e.g., ask for `papi show <paper> --level tex`)

Citations:
- For each non-trivial claim, include a short quote snippet (<= 15 words) and cite as:
  (paper: <name>, arXiv: <id if present>, source: summary|equations|tex|notes, ref: section/eq/table/figure if present)

End with:
Cited papers: <comma-separated paper names and/or arXiv IDs>
