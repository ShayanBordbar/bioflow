"""Command-line entry point for BioFlow."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

import typer
from rich.console import Console
from rich.table import Table

from bioflow import __version__ as _fallback_version
from bioflow.core.doctor import CheckResult, CheckStatus, has_missing, run_checks

app = typer.Typer(
    name="bioflow",
    help="Provenance-first orchestration for computational biology.",
    no_args_is_help=True,
)

# Sub-apps are registered now so the command surface is visible from the start; their
# commands arrive with the capabilities they belong to.
_SUBAPPS: tuple[tuple[str, str], ...] = (
    ("run", "Execute a capability. (For past runs, see `bioflow runs`.)"),
    ("runs", "Inspect past runs. (To start one, see `bioflow run`.)"),
    ("sequence", "Sequence-model operations."),
    ("model", "Model registry and checkpoint management."),
    ("design", "In-silico sequence design."),
    ("mpra", "MPRA analysis workflows."),
    ("rnaseq", "RNA-seq analysis workflows."),
    ("eval", "Benchmarks and evaluation."),
    ("demo", "End-to-end demonstrations."),
)

_registered: dict[str, typer.Typer] = {}
for _name, _help in _SUBAPPS:
    _sub = typer.Typer(name=_name, help=_help, no_args_is_help=True)
    _registered[_name] = _sub
    app.add_typer(_sub, name=_name)

runs_app = _registered["runs"]

console = Console()
err_console = Console(stderr=True)

_STATUS_STYLE = {
    CheckStatus.OK: "green",
    CheckStatus.DEGRADED: "yellow",
    CheckStatus.MISSING: "red",
}


def get_version() -> str:
    """Return the installed distribution version, or the in-tree literal."""
    try:
        return _package_version("bioflow")
    except PackageNotFoundError:
        return _fallback_version


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(get_version())
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def cli(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the BioFlow version and exit.",
    ),
) -> None:
    """BioFlow root command."""


@app.command()
def version() -> None:
    """Print the BioFlow version."""
    typer.echo(get_version())


def _render_table(results: list[CheckResult]) -> Table:
    table = Table(title="bioflow doctor", header_style="bold")
    table.add_column("check", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold")
    for result in results:
        table.add_row(
            result.name,
            f"[{_STATUS_STYLE[result.status]}]{result.status.value}[/]",
            result.detail,
        )
    return table


@app.command()
def doctor(
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if any check reports MISSING."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit results as JSON instead of a table."),
) -> None:
    """Diagnose the local environment.

    Runs on a machine with nothing installed: absent tooling is reported, never raised.
    The exit code is 0 unless ``--strict`` is given and something is MISSING.
    """
    results = run_checks()

    if as_json:
        typer.echo(json.dumps([result.to_json_dict() for result in results], indent=2))
    else:
        console.print(_render_table(results))

    if strict and has_missing(results):
        missing = [r.name for r in results if r.status is CheckStatus.MISSING]
        names = ", ".join(missing)
        err_console.print(f"[red]--strict: {len(missing)} check(s) MISSING: {names}[/]")
        raise typer.Exit(code=1)


@runs_app.command("list")
def runs_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of runs to show."),
) -> None:
    """List past runs, newest first."""
    from bioflow.config import get_settings
    from bioflow.core.runstore import RunStore

    settings = get_settings()
    with RunStore(settings.data_dir) as store:
        records = store.list_runs(limit=limit)

    if not records:
        console.print(f"No runs recorded in {settings.data_dir}.")
        return

    table = Table(title=f"runs in {settings.data_dir}", header_style="bold")
    table.add_column("run_id", no_wrap=True)
    table.add_column("created", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("bioflow", no_wrap=True)
    for record in records:
        table.add_row(
            record.run_id,
            record.created_at.isoformat(timespec="seconds"),
            record.status.value,
            record.bioflow_version,
        )
    console.print(table)


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument(..., help="The run to describe."),
    events: int = typer.Option(10, "--events", "-e", help="Provenance events to show."),
) -> None:
    """Describe one run: its header, its artifacts, and its recent provenance."""
    from bioflow.config import get_settings
    from bioflow.core.models import BioflowError
    from bioflow.core.runstore import RunStore

    settings = get_settings()
    with RunStore(settings.data_dir) as store:
        try:
            manifest = store.get_manifest(run_id)
        except BioflowError as exc:
            err_console.print(f"[red]{exc.error.message}[/]")
            raise typer.Exit(code=1) from exc
        provenance = store.list_provenance(run_id)

    header = Table(show_header=False, box=None)
    header.add_row("run_id", manifest.run_id)
    header.add_row("created", manifest.created_at.isoformat(timespec="seconds"))
    header.add_row("status", manifest.status.value)
    header.add_row("bioflow", manifest.bioflow_version)
    header.add_row("python", manifest.python_version)
    header.add_row("git commit", manifest.git_commit or "-")
    header.add_row("artifacts", str(len(manifest.artifacts)))
    header.add_row("evidence", str(len(manifest.evidence)))
    console.print(header)

    if manifest.artifacts:
        artifacts = Table(title="artifacts", header_style="bold")
        artifacts.add_column("summary", overflow="fold")
        for ref in manifest.artifacts:
            # short() deliberately omits the URI, so this stays path-free.
            artifacts.add_row(ref.short())
        console.print(artifacts)

    if provenance:
        recent = provenance[-events:]
        title = f"provenance (last {len(recent)} of {len(provenance)})"
        table = Table(title=title, header_style="bold")
        table.add_column("time", no_wrap=True)
        table.add_column("actor", no_wrap=True)
        table.add_column("action", no_wrap=True)
        table.add_column("status", no_wrap=True)
        table.add_column("ms", justify="right", no_wrap=True)
        for event in recent:
            table.add_row(
                event.timestamp.isoformat(timespec="seconds"),
                event.actor.value,
                event.action,
                event.status.value,
                "-" if event.duration_ms is None else str(event.duration_ms),
            )
        console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
