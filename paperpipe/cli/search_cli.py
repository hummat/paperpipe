"""Search commands: list, search, tags."""

from __future__ import annotations

import json
from typing import Optional

import click

from .. import config
from ..config import default_search_mode
from ..core import (
    _fuzzy_text_score,
    _read_text_limited,
    load_index,
)
from ..search import (
    _collect_grep_matches,
    _search_db_path,
    _search_fts,
    _search_grep,
)


@click.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_papers(tag: Optional[str], as_json: bool):
    """List all papers in the database."""
    index = load_index()

    if tag:
        index = {k: v for k, v in index.items() if tag in v.get("tags", [])}

    if as_json:
        click.echo(json.dumps(index, indent=2))
        return

    if not index:
        click.echo("No papers found.")
        return

    for name, info in sorted(index.items()):
        title = info.get("title", "Unknown")[:50]
        tags = ", ".join(info.get("tags", [])[:4])
        click.echo(name)
        click.echo(f"  {title}...")
        click.echo(f"  Tags: {tags}")
        click.echo()


@click.command()
@click.argument("query")
@click.option(
    "--limit",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of results to show.",
)
@click.option(
    "--grep",
    "--rg",
    "use_grep",
    is_flag=True,
    help="Use ripgrep/grep for fast exact-match search (shows file:line hits).",
)
@click.option(
    "--fixed-strings/--regex",
    "fixed_strings",
    default=True,
    show_default=True,
    help="In --grep mode, treat QUERY as a literal string instead of a regex.",
)
@click.option(
    "--context",
    "context_lines",
    type=int,
    default=2,
    show_default=True,
    help="In --grep mode, number of context lines around each match.",
)
@click.option(
    "--max-matches",
    type=int,
    default=200,
    show_default=True,
    help="In --grep mode, stop after this many matches (approx; tool-dependent; effectively per-file for grep).",
)
@click.option(
    "--ignore-case/--case-sensitive",
    "ignore_case",
    default=True,
    show_default=True,
    help="In --grep mode, ignore case when matching QUERY.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="In --grep mode, output machine-readable JSON (forces --context 0).",
)
@click.option(
    "--fuzzy/--exact",
    default=True,
    show_default=True,
    help="Fall back to fuzzy matching only if no exact matches were found.",
)
@click.option(
    "--tex",
    is_flag=True,
    help="Also search within LaTeX source (can be slower).",
)
@click.option(
    "--fts/--no-fts",
    "use_fts",
    default=True,
    show_default=True,
    help="Use SQLite FTS5 ranked search if `search.db` exists (falls back to scan). Use --no-fts to force scan.",
)
@click.option(
    "--hybrid",
    is_flag=True,
    help="Hybrid search: FTS5 ranked search + grep signal boosting papers with exact matches.",
)
@click.option(
    "--show-grep-hits",
    is_flag=True,
    help="With --hybrid, show a few grep hit lines under each matching paper.",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    use_grep: bool,
    fixed_strings: bool,
    context_lines: int,
    max_matches: int,
    ignore_case: bool,
    as_json: bool,
    fuzzy: bool,
    tex: bool,
    use_fts: bool,
    hybrid: bool,
    show_grep_hits: bool,
):
    """Search papers by title, tags, metadata, and local content."""
    grep_only_params = ("fixed_strings", "context_lines", "max_matches", "ignore_case", "as_json")
    if not use_grep:
        flag_text = {
            "fixed_strings": "--fixed-strings/--regex",
            "context_lines": "--context",
            "max_matches": "--max-matches",
            "ignore_case": "--ignore-case/--case-sensitive",
            "as_json": "--json",
        }
        used = [p for p in grep_only_params if ctx.get_parameter_source(p) != click.core.ParameterSource.DEFAULT]
        if used:
            flags = ", ".join(flag_text.get(p, f"--{p.replace('_', '-')}") for p in used)
            raise click.UsageError(f"{flags} only apply with --grep/--rg.")

    if use_grep and ctx.get_parameter_source("use_fts") != click.core.ParameterSource.DEFAULT:
        raise click.UsageError("--fts/--no-fts do not apply with --grep/--rg.")

    if hybrid and use_grep:
        raise click.UsageError("--hybrid does not apply with --grep/--rg (use one or the other).")
    if hybrid and not use_fts:
        raise click.UsageError("--hybrid requires --fts (disable hybrid or drop --no-fts).")
    if show_grep_hits and not hybrid:
        raise click.UsageError("--show-grep-hits requires --hybrid.")

    # Default search mode (env/config) only applies when the user didn't explicitly choose.
    mode = default_search_mode()
    use_grep_source = ctx.get_parameter_source("use_grep")
    use_fts_source = ctx.get_parameter_source("use_fts")
    hybrid_source = ctx.get_parameter_source("hybrid")

    if (
        use_grep_source == click.core.ParameterSource.DEFAULT
        and use_fts_source == click.core.ParameterSource.DEFAULT
        and hybrid_source == click.core.ParameterSource.DEFAULT
    ):
        if mode == "scan":
            use_fts = False
            hybrid = False
        elif mode == "fts":
            use_fts = True
            hybrid = False
        elif mode == "hybrid":
            use_fts = True
            hybrid = True
        else:
            # auto: keep current behavior (fts if db exists, else scan)
            pass

    if use_grep:
        handled = _search_grep(
            query=query,
            fixed_strings=fixed_strings,
            context_lines=context_lines,
            max_matches=max_matches,
            ignore_case=ignore_case,
            as_json=as_json,
            include_tex=tex,
        )
        if handled:
            return
        use_fts = False
        hybrid = False

    if hybrid:
        db_path = _search_db_path()
        if not db_path.exists():
            # For configured default "hybrid", degrade to normal search rather than erroring.
            if (
                default_search_mode() == "hybrid"
                and ctx.get_parameter_source("hybrid") == click.core.ParameterSource.DEFAULT
            ):
                hybrid = False
            else:
                raise click.ClickException(
                    "Hybrid search requires `search.db`. Build it first: `papi index --backend search --search-rebuild`."
                )

        if hybrid:
            fts_results = _search_fts(query=query, limit=max(limit, 50))
            grep_matches = _collect_grep_matches(
                query=query,
                fixed_strings=True,
                max_matches=200,
                ignore_case=True,
                include_tex=tex,
            )
        else:
            fts_results = []
            grep_matches = []
        grep_by_paper: dict[str, list[dict[str, object]]] = {}
        for m in grep_matches:
            paper = str(m.get("paper") or "")
            if paper:
                grep_by_paper.setdefault(paper, []).append(m)

        fts_by_name = {str(r["name"]): r for r in fts_results}
        candidates = set(fts_by_name.keys()) | set(grep_by_paper.keys())

        def fts_score(name: str) -> float:
            raw = (fts_by_name.get(name) or {}).get("score")
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                try:
                    return float(raw)
                except ValueError:
                    return 0.0
            return 0.0

        def sort_key(name: str) -> tuple[int, float, int, str]:
            grep_count = len(grep_by_paper.get(name, []))
            score = fts_score(name)
            return (1 if grep_count > 0 else 0, score, grep_count, name)

        ranked = sorted(candidates, key=sort_key, reverse=True)[:limit]
        if not ranked:
            click.echo(f"No papers found matching '{query}'")
            return

        idx = load_index()
        for name in ranked:
            score = fts_score(name)
            grep_count = len(grep_by_paper.get(name, []))
            title = str((idx.get(name, {}) or {}).get("title") or fts_by_name.get(name, {}).get("title") or "Unknown")[
                :80
            ]
            if grep_count:
                click.echo(f"{name} (score: {score:.6g}, grep: {grep_count})")
            else:
                click.echo(f"{name} (score: {score:.6g})")
            if title:
                click.echo(f"  {title}...")
            if show_grep_hits and grep_count:
                for hit in grep_by_paper[name][:3]:
                    click.echo(f"  {hit.get('path')}:{hit.get('line')}: {str(hit.get('text') or '').strip()}")
            click.echo()
        return

    if use_fts and _search_db_path().exists():
        fts_results = _search_fts(query=query, limit=limit)

        if fts_results:
            for r in fts_results:
                click.echo(f"{r['name']} (score: {r['score']:.6g})")
                title = str(r.get("title") or "Unknown")[:80]
                if title:
                    click.echo(f"  {title}...")
                click.echo()
            return

    index = load_index()

    def collect_results(*, fuzzy_mode: bool) -> list[tuple[str, dict, int, list[str]]]:
        results: list[tuple[str, dict, int, list[str]]] = []
        for name, info in index.items():
            paper_dir = config.PAPERS_DIR / name
            meta_path = paper_dir / "meta.json"

            matched_fields: list[str] = []
            score = 0

            def add_field(field: str, text: str, weight: float) -> None:
                nonlocal score
                field_score = _fuzzy_text_score(query, text, fuzzy=fuzzy_mode)
                if field_score <= 0:
                    return
                score += int(100 * weight * field_score)
                matched_fields.append(field)

            add_field("name", name, 1.6)
            add_field("title", info.get("title", ""), 1.4)
            add_field("tags", " ".join(info.get("tags", [])), 1.2)
            add_field("arxiv_id", info.get("arxiv_id", ""), 1.0)

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    # Silent fallback during search - don't spam warnings for every paper
                    meta = {}
                add_field("authors", " ".join(meta.get("authors", []) or []), 0.6)
                add_field("abstract", meta.get("abstract", "") or "", 0.9)

            summary_path = paper_dir / "summary.md"
            if summary_path.exists():
                add_field("summary", _read_text_limited(summary_path, max_chars=80_000), 0.9)

            equations_path = paper_dir / "equations.md"
            if equations_path.exists():
                add_field("equations", _read_text_limited(equations_path, max_chars=80_000), 0.9)

            if tex:
                source_path = paper_dir / "source.tex"
                if source_path.exists():
                    add_field("source", _read_text_limited(source_path, max_chars=200_000), 0.5)

            if score > 0:
                results.append((name, info, score, matched_fields))

        results.sort(key=lambda x: (-x[2], x[0]))
        return results

    # Exact pass first; only fall back to fuzzy if enabled and no exact matches exist.
    results = collect_results(fuzzy_mode=False)
    if not results and fuzzy:
        results = collect_results(fuzzy_mode=True)

    if not results:
        click.echo(f"No papers found matching '{query}'")
        return

    for name, info, score, matched_fields in results[:limit]:
        click.echo(f"{name} (score: {score})")
        click.echo(f"  {info.get('title', 'Unknown')[:60]}...")
        if matched_fields:
            click.echo(f"  Matches: {', '.join(matched_fields[:6])}")
        click.echo()


@click.command()
def tags():
    """List all tags in the database."""
    index = load_index()
    all_tags: dict[str, int] = {}

    for info in index.values():
        for tag in info.get("tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
        click.echo(f"{tag}: {count}")
