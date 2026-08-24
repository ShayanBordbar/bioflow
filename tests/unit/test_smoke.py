"""Scaffold smoke test: the package imports and the CLI reports its version."""

import pytest
from typer.testing import CliRunner

import bioflow
from bioflow.cli import app

pytestmark = pytest.mark.unit


def test_package_exposes_version() -> None:
    assert isinstance(bioflow.__version__, str)
    assert bioflow.__version__


def test_cli_version_flag() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert bioflow.__version__ in result.stdout
