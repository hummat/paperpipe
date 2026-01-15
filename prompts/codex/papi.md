---
description: Route a question to the cheapest exact `papi ...` command(s), then suggest the next prompt to use.
argument-hint: [question/context...]
---
Task: Given the user's question/context: $ARGUMENTS

Output the exact shell command(s) to run next, choosing the cheapest thing that can answer.

Rules:
- Prefer: `papi search` (uses ranked BM25 by default if indexed; use `--grep --fixed-strings` for exact hits only; use `--hybrid` for ranked+exact)
  / `papi list` / `papi show ... --level eq|tex|summary` / `papi notes` / `papi export`.
- Recommend `papi ask` only if the user explicitly asked for RAG/synthesis OR `search/show` cannot answer.
- If you need machine-readable citations/snippets, use `papi ask --format evidence-blocks` (JSON output).
- If network/LLM should be avoided, include `--no-llm` where relevant and avoid `papi ask`.
- After commands, output: `Next prompt:` one of `/prompts:verify-with-paper` or `/prompts:ground-with-paper` or
  `/prompts:compare-papers` or `/prompts:curate-paper-note`.

Format:
Commands:
- <exact command 1>
- <exact command 2 (if needed)>
Rationale: <one line>
Next prompt: <one of the above>
