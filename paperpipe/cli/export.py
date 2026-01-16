"""Export and audit commands."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import click

from .. import config
from ..core import (
    _parse_overwrite_option,
    _resolve_paper_name_from_ref,
    load_index,
)
from ..output import echo_error, echo_progress, echo_success, echo_warning
from ..paper import _regenerate_one_paper
from ..search import _audit_paper_dir, _parse_selection_spec


@click.command()
@click.argument("papers", nargs=-1)
@click.option("--all", "audit_all", is_flag=True, help="Audit all papers (default).")
@click.option("--limit", type=int, default=None, help="Audit only N random papers.")
@click.option("--seed", type=int, default=None, help="Random seed for --limit sampling.")
@click.option(
    "--interactive/--no-interactive", default=None, help="Prompt to regenerate flagged papers (default: auto)."
)
@click.option("--regenerate", "do_regenerate", is_flag=True, help="Regenerate all flagged papers.")
@click.option("--no-llm", is_flag=True, help="Use non-LLM regeneration when regenerating.")
@click.option(
    "--overwrite",
    "-o",
    default="summary,equations,tags",
    help="Overwrite fields when regenerating (all or list: summary,equations,tags,name).",
)
def audit(
    papers: tuple[str, ...],
    audit_all: bool,
    limit: Optional[int],
    seed: Optional[int],
    interactive: Optional[bool],
    do_regenerate: bool,
    no_llm: bool,
    overwrite: str,
):
    """Audit generated summaries/equations for obvious issues and optionally regenerate flagged papers."""
    index = load_index()
    overwrite_fields, overwrite_all = _parse_overwrite_option(overwrite)

    if audit_all and papers:
        raise click.UsageError("Use either paper(s)/arXiv id(s) OR `--all`, not both.")

    if not audit_all and not papers:
        audit_all = True

    if audit_all:
        names = sorted(index.keys())
    else:
        names = []
        for paper_ref in papers:
            name, error = _resolve_paper_name_from_ref(paper_ref, index)
            if not name:
                raise click.UsageError(error)
            names.append(name)

    if not names:
        click.echo("No papers found.")
        return

    if limit is not None:
        if limit <= 0:
            raise click.UsageError("--limit must be > 0")
        import random

        rng = random.Random(seed)
        if limit < len(names):
            names = rng.sample(names, k=limit)

    flagged: list[tuple[str, list[str]]] = []
    ok_count = 0
    for name in names:
        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            flagged.append((name, ["missing paper directory"]))
            continue
        reasons = _audit_paper_dir(paper_dir)
        if reasons:
            flagged.append((name, reasons))
        else:
            ok_count += 1

    click.echo(f"Audited {len(names)} paper(s): {ok_count} OK, {len(flagged)} flagged")
    if not flagged:
        return

    click.echo()
    for name, reasons in flagged:
        click.secho(f"{name}: FLAGGED", fg="yellow")
        for reason in reasons:
            click.echo(f"  - {reason}")

    auto_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    effective_interactive = interactive if interactive is not None else auto_interactive

    if do_regenerate:
        selected_names = [name for name, _ in flagged]
    elif effective_interactive:
        click.echo()
        if not click.confirm("Regenerate any flagged papers now?", default=False):
            return
        click.echo("Select papers by number (e.g. 1,3-5) or 'all':")
        for i, (name, _) in enumerate(flagged, 1):
            click.echo(f"  {i}. {name}")
        try:
            spec = click.prompt("Selection", default="all", show_default=True)
            picks = _parse_selection_spec(spec, max_index=len(flagged))
        except Exception as exc:
            raise click.ClickException(f"Invalid selection: {exc}") from exc
        selected_names = [flagged[i - 1][0] for i in picks]
    else:
        return

    if not selected_names:
        return

    reasons_by_name = {n: r for n, r in flagged}
    failures = 0
    click.echo()
    for i, name in enumerate(selected_names, 1):
        echo_progress(f"[{i}/{len(selected_names)}] {name}")
        success, _new_name = _regenerate_one_paper(
            name,
            index,
            no_llm=no_llm,
            overwrite_fields=overwrite_fields,
            overwrite_all=overwrite_all,
            audit_reasons=reasons_by_name.get(name),
        )
        if not success:
            failures += 1

    if failures:
        raise click.ClickException(f"{failures} paper(s) failed to regenerate.")


@click.command()
@click.argument("papers", nargs=-1, required=True)
@click.option(
    "--level",
    "-l",
    type=click.Choice(["summary", "equations", "eq", "full"], case_sensitive=False),
    default="summary",
    help="What to export",
)
@click.option(
    "--to",
    "dest",
    type=click.Path(),
    help="Destination directory",
)
@click.option(
    "--figures",
    is_flag=True,
    help="Also export figures directory if it exists",
)
def export(papers: tuple[str, ...], level: str, dest: Optional[str], figures: bool):
    """Export paper context for a coding session."""
    level_norm = (level or "").strip().lower()
    if level_norm == "eq":
        level_norm = "equations"

    index = load_index()

    if dest == "-":
        raise click.UsageError(
            "Use `papi show ... --level ...` to print to stdout; `export` only writes to a directory."
        )

    dest_path = Path(dest) if dest else Path.cwd() / "paper-context"
    dest_path.mkdir(exist_ok=True)

    if level_norm == "summary":
        src_name = "summary.md"
        out_suffix = "_summary.md"
        missing_msg = "No summary found"
    elif level_norm == "equations":
        src_name = "equations.md"
        out_suffix = "_equations.md"
        missing_msg = "No equations found"
    else:  # full
        src_name = "source.tex"
        out_suffix = ".tex"
        missing_msg = "No LaTeX source found"

    successes = 0
    failures = 0

    for paper_ref in papers:
        name, error = _resolve_paper_name_from_ref(paper_ref, index)
        if not name:
            echo_error(error)
            failures += 1
            continue

        paper_dir = config.PAPERS_DIR / name
        if not paper_dir.exists():
            echo_error(f"Paper not found: {paper_ref}")
            failures += 1
            continue

        src = paper_dir / src_name
        if not src.exists():
            echo_error(f"{missing_msg}: {name}")
            failures += 1
            continue

        dest_file = dest_path / f"{name}{out_suffix}"
        shutil.copy(src, dest_file)

        # Export figures if requested
        if figures:
            figures_dir = paper_dir / "figures"
            if figures_dir.exists() and figures_dir.is_dir():
                dest_figures_dir = dest_path / f"{name}_figures"
                try:
                    if dest_figures_dir.exists():
                        shutil.rmtree(dest_figures_dir)
                    shutil.copytree(figures_dir, dest_figures_dir)
                except OSError as e:
                    # Handles PermissionError, shutil.Error (both OSError subclasses)
                    echo_warning(f"  Failed to export figures for {name}: {e}")
                    # Don't count as failure since main content was exported

        successes += 1

    if failures == 0:
        echo_success(f"Exported {successes} paper(s) to {dest_path}")
        return

    if successes > 0:
        click.echo()
        echo_success(f"Exported {successes} paper(s) to {dest_path}")
        echo_error(f"{failures} paper(s) failed")
        raise SystemExit(1)
    else:
        raise click.ClickException("All exports failed")
