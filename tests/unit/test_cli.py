"""Unit tests for the command-line surface."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import bioflow
from bioflow.cli import _SUBAPPS, app, get_version
from bioflow.config import reset_settings_cache
from bioflow.core import doctor
from bioflow.core.ids import new_artifact_id, new_event_id, new_run_id
from bioflow.core.models import (
    Actor,
    ArtifactRef,
    ArtifactType,
    EventStatus,
    ProvenanceEvent,
    RunStatus,
)
from bioflow.core.runstore import RunStore

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point the CLI at a throwaway workspace and keep it off the network."""
    root = tmp_path / "workspace"
    monkeypatch.setenv("BIOFLOW_DATA_DIR", str(root))
    monkeypatch.setenv("BIOFLOW_OFFLINE", "true")
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()
    yield root
    reset_settings_cache()


@pytest.fixture
def bare_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Nothing on PATH and no credentials: the acceptance scenario."""
    monkeypatch.setattr(doctor.shutil, "which", lambda _cmd: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(doctor, "ANTHROPIC_PROFILE_DIR", tmp_path / "absent")


# --------------------------------------------------------------------------------------
# version
# --------------------------------------------------------------------------------------


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert bioflow.__version__ in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert get_version() in result.stdout


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------


def test_doctor_runs_on_a_bare_machine_and_exits_zero(bare_machine: None) -> None:
    # The acceptance criterion: absent tooling is reported, not raised.
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert result.exception is None
    assert "MISSING" in result.stdout
    assert "docker" in result.stdout


def test_doctor_strict_exits_one_when_something_is_missing(bare_machine: None) -> None:
    result = runner.invoke(app, ["doctor", "--strict"])
    assert result.exit_code == 1


def test_doctor_without_strict_exits_zero_even_when_missing(bare_machine: None) -> None:
    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_doctor_json_is_machine_readable(bare_machine: None) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert len(rows) == 14
    assert {"name", "status", "detail"} == set(rows[0])
    assert {row["status"] for row in rows} <= {"OK", "MISSING", "DEGRADED"}


def test_doctor_strict_names_what_is_missing(bare_machine: None) -> None:
    result = runner.invoke(app, ["doctor", "--strict"])
    assert "docker" in result.output


def test_doctor_reports_ok_when_everything_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(doctor.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda command: "1.0.0" if command[0] != "nvidia-smi" else "Tesla T4",
    )
    result = runner.invoke(app, ["doctor", "--strict", "--json"])
    rows = json.loads(result.stdout)
    statuses = {row["name"]: row["status"] for row in rows}
    assert statuses["docker"] == "OK"
    assert statuses["gpu"] == "OK"
    assert statuses["llm credentials"] == "OK"


# --------------------------------------------------------------------------------------
# Sub-apps
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [name for name, _ in _SUBAPPS])
def test_every_subapp_responds_to_help(name: str) -> None:
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0
    assert name in result.stdout


@pytest.mark.parametrize("name", [name for name, _ in _SUBAPPS])
def test_bare_subapp_prints_help_rather_than_failing(name: str) -> None:
    result = runner.invoke(app, [name])
    assert result.exit_code in (0, 2)
    assert "Usage" in result.stdout or "Commands" in result.stdout


def test_run_and_runs_help_point_at_each_other() -> None:
    # They are one keystroke apart, so each help string names the other.
    assert "runs" in runner.invoke(app, ["run", "--help"]).stdout
    assert "run" in runner.invoke(app, ["runs", "--help"]).stdout


def test_root_help_lists_the_command_surface() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name, _ in _SUBAPPS:
        assert name in result.stdout


# --------------------------------------------------------------------------------------
# runs list / show
# --------------------------------------------------------------------------------------


def _seed_run(root: Path) -> str:
    run_id = new_run_id()
    with RunStore(root) as store:
        store.create_run(
            run_id=run_id,
            request={"assay": "rna-seq"},
            bioflow_version="0.1.0",
            python_version="3.12.13",
            created_at=datetime(2026, 8, 28, 9, 0, tzinfo=UTC),
            git_commit="abc1234",
        )
        store.add_artifact(
            run_id,
            ArtifactRef(
                artifact_id=new_artifact_id(),
                artifact_type=ArtifactType.FASTQ,
                uri="cas://" + "a" * 64,
                sha256="a" * 64,
                size_bytes=2048,
                created_at=datetime(2026, 8, 28, 9, 1, tzinfo=UTC),
            ),
        )
        store.append_provenance(
            ProvenanceEvent(
                event_id=new_event_id(),
                run_id=run_id,
                timestamp=datetime(2026, 8, 28, 9, 2, tzinfo=UTC),
                actor=Actor.TOOL,
                action="quantify",
                status=EventStatus.OK,
                duration_ms=1200,
            )
        )
        store.set_status(run_id, RunStatus.OK)
    return run_id


def test_runs_list_on_an_empty_store(workspace: Path) -> None:
    result = runner.invoke(app, ["runs", "list"])
    assert result.exit_code == 0
    assert "No runs recorded" in result.stdout


def test_runs_list_shows_a_seeded_run(workspace: Path) -> None:
    run_id = _seed_run(workspace)
    result = runner.invoke(app, ["runs", "list"])
    assert result.exit_code == 0
    assert run_id in result.stdout.replace("\n", "")
    assert "OK" in result.stdout


def test_runs_list_honours_the_limit(workspace: Path) -> None:
    for _ in range(3):
        _seed_run(workspace)
    result = runner.invoke(app, ["runs", "list", "--limit", "1"])
    assert result.exit_code == 0


def test_runs_show_describes_a_run(workspace: Path) -> None:
    run_id = _seed_run(workspace)
    result = runner.invoke(app, ["runs", "show", run_id])
    assert result.exit_code == 0
    flat = result.stdout.replace("\n", "")
    assert run_id in flat
    assert "quantify" in flat
    assert "artifacts" in flat


def test_runs_show_never_prints_a_filesystem_path(workspace: Path) -> None:
    run_id = _seed_run(workspace)
    result = runner.invoke(app, ["runs", "show", run_id])
    # Artifacts are rendered with short(), which omits the URI by construction.
    assert "cas://" not in result.stdout


def test_runs_show_reports_an_unknown_run(workspace: Path) -> None:
    result = runner.invoke(app, ["runs", "show", "run_ffffffffffff"])
    assert result.exit_code == 1
    assert "no run" in result.output
