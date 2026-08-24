"""Command-line entry point for BioFlow."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

import typer

from bioflow import __version__ as _fallback_version

app = typer.Typer(
    name="bioflow",
    help="Provenance-first orchestration for computational biology.",
    no_args_is_help=True,
)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
