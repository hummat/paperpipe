# Code Conventions

## Python
- Use `typing` module for all type hints (`from typing import ...`)
- Use `collections.abc` for abstract container types
- Prefer pure functions over stateful classes
- Explicit error handling; no bare `except:`
- Standard library over new dependencies unless clear win

## Style
- KISS principle; Zen of Python
- Boring, readable solutions over clever ones
- Clear names; avoid abbreviations
- Docstrings only when they add real clarity (no boilerplate)

## Changes
- Minimal diffs; don't reformat unrelated code
- Match existing patterns and style in the file
- Add/update tests when behavior changes
- For multi-file edits: label filenames clearly, keep changes localized

## Metaprogramming
- Avoid unless the repo already uses it heavily
