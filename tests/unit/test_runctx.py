"""Unit tests for RunContext and its timed, always-recorded step."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from bioflow.core.ids import new_evidence_id
from bioflow.core.models import (
    Actor,
    ArtifactType,
    BioflowError,
    ErrorCode,
    EventStatus,
    Evidence,
    EvidenceType,
    RunStatus,
    StructuredError,
)
from bioflow.core.runctx import RUN_SUBDIRS, RunContext

pytestmark = pytest.mark.unit


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[RunContext]:
    with RunContext.create(tmp_path / "workspace", {"assay": "rna-seq"}) as context:
        yield context


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def test_create_builds_the_run_directory(ctx: RunContext) -> None:
    for name in RUN_SUBDIRS:
        assert (ctx.run_dir / name).is_dir()
    assert json.loads((ctx.run_dir / "request.json").read_text()) == {"assay": "rna-seq"}


def test_create_registers_the_run(ctx: RunContext) -> None:
    record = ctx.runstore.get_run(ctx.run_id)
    assert record is not None
    assert record.status is RunStatus.PENDING
    assert record.request == {"assay": "rna-seq"}
    assert record.python_version  # stamped from the interpreter that ran the run
    assert record.bioflow_version


def test_create_mints_distinct_run_ids(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    with RunContext.create(root) as first, RunContext.create(root) as second:
        assert first.run_id != second.run_id
        assert first.run_dir != second.run_dir


def test_open_reattaches_to_an_existing_run(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    with RunContext.create(root, {"assay": "mpra"}) as created:
        run_id = created.run_id
        created.set_status(RunStatus.RUNNING)

    with RunContext.open(root, run_id) as reopened:
        assert reopened.run_id == run_id
        assert reopened.run_dir == created.run_dir
        assert reopened.manifest().status is RunStatus.RUNNING


def test_open_rejects_an_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(BioflowError) as excinfo:
        RunContext.open(tmp_path / "workspace", "run_ffffffffffff")
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND


# --------------------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------------------


def test_record_puts_an_artifact_in_the_manifest(ctx: RunContext, tmp_path: Path) -> None:
    source = tmp_path / "reads.fastq"
    source.write_bytes(b"@r1\nACGT\n+\nIIII\n")
    ref = ctx.record(ctx.store.put(source, ArtifactType.FASTQ, run_id=ctx.run_id))

    manifest = ctx.manifest()
    assert manifest.artifacts == [ref]
    assert ctx.store.verify(ref) is True


def test_record_evidence_puts_evidence_in_the_manifest(ctx: RunContext) -> None:
    evidence = ctx.record_evidence(
        Evidence(
            evidence_id=new_evidence_id(),
            evidence_type=EvidenceType.MEASURED,
            statement="Median TIN is 8.4.",
            value=8.4,
            produced_by="bioflow.capabilities.qc",
        )
    )
    assert ctx.manifest().evidence == [evidence]


def test_set_status_is_visible_in_the_manifest(ctx: RunContext) -> None:
    ctx.set_status(RunStatus.NEEDS_REVIEW)
    assert ctx.manifest().status is RunStatus.NEEDS_REVIEW


# --------------------------------------------------------------------------------------
# step()
# --------------------------------------------------------------------------------------


def test_step_emits_one_ok_event(ctx: RunContext) -> None:
    with ctx.step("quantify", tool="kallisto", tool_version="0.50.1"):
        pass

    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.action == "quantify"
    assert event.status is EventStatus.OK
    assert event.tool == "kallisto"
    assert event.tool_version == "0.50.1"
    assert event.duration_ms is not None
    assert event.duration_ms >= 0
    assert event.error is None


def test_step_links_inputs_to_outputs(ctx: RunContext, tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_bytes(b"input")
    incoming = ctx.record(ctx.store.put(source, ArtifactType.OTHER))

    with ctx.step("transform") as step:
        step.add_input(incoming)
        produced = ctx.record(ctx.store.put_bytes(b"output", ArtifactType.OTHER))
        step.add_output(produced)

    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.inputs == (incoming.artifact_id,)
    assert event.outputs == (produced.artifact_id,)


def test_step_records_the_failure_and_reraises(ctx: RunContext) -> None:
    with pytest.raises(ZeroDivisionError), ctx.step("explode"):
        _ = 1 / 0

    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.status is EventStatus.ERROR
    assert event.error is not None
    # An unexpected exception type is a bug, so it is reported as INTERNAL.
    assert event.error.error_code is ErrorCode.INTERNAL
    assert "ZeroDivisionError" in event.error.message


def test_step_preserves_a_structured_error(ctx: RunContext) -> None:
    original = StructuredError(
        error_code=ErrorCode.EXTERNAL_TOOL_FAILED,
        message="kallisto exited 1",
        remediation="Check the index.",
    )
    with pytest.raises(BioflowError), ctx.step("quantify", tool="kallisto"):
        raise BioflowError(original)

    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.status is EventStatus.ERROR
    # The diagnosed payload survives intact rather than being flattened to INTERNAL.
    assert event.error == original


def test_step_records_partial_outputs_on_failure(ctx: RunContext) -> None:
    with pytest.raises(RuntimeError), ctx.step("half-done") as step:
        step.add_output(ctx.store.put_bytes(b"partial", ArtifactType.OTHER))
        raise RuntimeError("stopped halfway")

    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert len(event.outputs) == 1
    assert event.status is EventStatus.ERROR


def test_step_accepts_a_custom_actor(ctx: RunContext) -> None:
    with ctx.step("triage", actor=Actor.AGENT):
        pass
    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.actor is Actor.AGENT


def test_steps_are_recorded_in_order(ctx: RunContext) -> None:
    for name in ("first", "second", "third"):
        with ctx.step(name):
            pass
    assert [e.action for e in ctx.runstore.list_provenance(ctx.run_id)] == [
        "first",
        "second",
        "third",
    ]


def test_steps_reach_the_jsonl_mirror(ctx: RunContext) -> None:
    with ctx.step("quantify"):
        pass
    lines = ctx.runstore.provenance_path(ctx.run_id).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["action"] for line in lines] == ["quantify"]
