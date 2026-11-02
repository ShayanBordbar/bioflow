"""Unit tests for the SQLite run store and the provenance mirror."""

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bioflow.core.ids import new_artifact_id, new_event_id, new_evidence_id
from bioflow.core.models import (
    Actor,
    ArtifactRef,
    ArtifactType,
    BioflowError,
    ErrorCode,
    EventStatus,
    Evidence,
    EvidenceType,
    ProvenanceEvent,
    RunStatus,
)
from bioflow.core.runstore import GLOBAL_SCOPE, SCHEMA_VERSION, RunStore

pytestmark = pytest.mark.unit

RUN_ID = "run_0123456789ab"
NOW = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture
def store(root: Path) -> Iterator[RunStore]:
    with RunStore(root) as opened:
        yield opened


def _create(store: RunStore, run_id: str = RUN_ID, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "request": {"assay": "rna-seq", "samples": 6},
        "bioflow_version": "0.1.0",
        "python_version": "3.12.13",
        "created_at": NOW,
        "git_commit": "e89f70d",
    }
    kwargs.update(overrides)
    store.create_run(run_id=run_id, **kwargs)  # type: ignore[arg-type]
    return run_id


def _artifact(**overrides: object) -> ArtifactRef:
    kwargs: dict[str, object] = {
        "artifact_id": new_artifact_id(),
        "artifact_type": ArtifactType.FASTQ,
        "uri": "cas://" + "a" * 64,
        "sha256": "a" * 64,
        "size_bytes": 1024,
        "created_at": NOW,
    }
    kwargs.update(overrides)
    return ArtifactRef(**kwargs)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> Evidence:
    kwargs: dict[str, object] = {
        "evidence_id": new_evidence_id(),
        "evidence_type": EvidenceType.MEASURED,
        "statement": "Median TIN is 8.4.",
        "value": 8.4,
        "produced_by": "bioflow.capabilities.qc",
    }
    kwargs.update(overrides)
    return Evidence(**kwargs)  # type: ignore[arg-type]


def _event(action: str, *, run_id: str = RUN_ID, timestamp: datetime = NOW) -> ProvenanceEvent:
    return ProvenanceEvent(
        event_id=new_event_id(),
        run_id=run_id,
        timestamp=timestamp,
        actor=Actor.ADAPTER,
        action=action,
    )


# --------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------


def test_create_and_get_run(store: RunStore) -> None:
    _create(store)
    record = store.get_run(RUN_ID)

    assert record is not None
    assert record.run_id == RUN_ID
    assert record.status is RunStatus.PENDING
    assert record.request == {"assay": "rna-seq", "samples": 6}
    assert record.created_at == NOW
    assert record.python_version == "3.12.13"
    assert record.git_commit == "e89f70d"


def test_get_run_returns_none_for_an_unknown_run(store: RunStore) -> None:
    assert store.get_run("run_ffffffffffff") is None


def test_run_record_repr_identifies_the_run(store: RunStore) -> None:
    _create(store)
    assert repr(store.get_run(RUN_ID)) == f"RunRecord(run_id={RUN_ID!r}, status='PENDING')"


def test_list_runs_is_newest_first(store: RunStore) -> None:
    _create(store, "run_00000000000a", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    _create(store, "run_00000000000b", created_at=datetime(2026, 6, 1, tzinfo=UTC))
    assert [r.run_id for r in store.list_runs()] == ["run_00000000000b", "run_00000000000a"]
    assert [r.run_id for r in store.list_runs(limit=1)] == ["run_00000000000b"]


def test_set_status_moves_the_run(store: RunStore) -> None:
    _create(store)
    store.set_status(RUN_ID, RunStatus.NEEDS_REVIEW)
    record = store.get_run(RUN_ID)
    assert record is not None
    assert record.status is RunStatus.NEEDS_REVIEW


def test_set_status_on_an_unknown_run_is_an_error(store: RunStore) -> None:
    with pytest.raises(BioflowError) as excinfo:
        store.set_status("run_ffffffffffff", RunStatus.OK)
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND


def test_duplicate_run_id_is_rejected(store: RunStore) -> None:
    _create(store)
    with pytest.raises(sqlite3.IntegrityError):
        _create(store)


def test_artifact_for_an_unknown_run_is_rejected(store: RunStore) -> None:
    # foreign_keys=ON is what makes this fail rather than silently orphaning a row.
    with pytest.raises(sqlite3.IntegrityError):
        store.add_artifact("run_ffffffffffff", _artifact())


# --------------------------------------------------------------------------------------
# Artifacts, evidence, manifest
# --------------------------------------------------------------------------------------


def test_artifacts_round_trip(store: RunStore) -> None:
    _create(store)
    ref = _artifact(metadata={"lanes": [1, 2]})
    store.add_artifact(RUN_ID, ref)
    assert store.list_artifacts(RUN_ID) == [ref]


def test_evidence_round_trips(store: RunStore) -> None:
    _create(store)
    evidence = _evidence(caveats=("Subsampled.",))
    store.add_evidence(RUN_ID, evidence)
    assert store.list_evidence(RUN_ID) == [evidence]


def test_get_manifest_reassembles_the_whole_run(store: RunStore) -> None:
    _create(store)
    ref = _artifact()
    evidence = _evidence()
    store.add_artifact(RUN_ID, ref)
    store.add_evidence(RUN_ID, evidence)
    store.set_tool_versions(RUN_ID, {"kallisto": "0.50.1"})
    store.set_llm_usage(RUN_ID, {"model": "claude-opus-5", "n_calls": 2})
    store.set_status(RUN_ID, RunStatus.OK)

    manifest = store.get_manifest(RUN_ID)
    assert manifest.run_id == RUN_ID
    assert manifest.status is RunStatus.OK
    assert manifest.artifacts == [ref]
    assert manifest.evidence == [evidence]
    assert manifest.tool_versions == {"kallisto": "0.50.1"}
    assert manifest.llm_usage == {"model": "claude-opus-5", "n_calls": 2}
    assert manifest.python_version == "3.12.13"


def test_manifest_defaults_when_nothing_was_recorded(store: RunStore) -> None:
    _create(store)
    manifest = store.get_manifest(RUN_ID)
    assert manifest.artifacts == []
    assert manifest.evidence == []
    assert manifest.tool_versions == {}
    assert manifest.llm_usage == {}


def test_get_manifest_for_an_unknown_run_is_an_error(store: RunStore) -> None:
    with pytest.raises(BioflowError) as excinfo:
        store.get_manifest("run_ffffffffffff")
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND


def test_kv_round_trips_json(store: RunStore) -> None:
    _create(store)
    store.set_kv(RUN_ID, "plan_pointer", {"steps": [1, 2, 3]})
    assert store.get_kv(RUN_ID, "plan_pointer") == {"steps": [1, 2, 3]}
    assert store.get_kv(RUN_ID, "absent") is None


def test_kv_overwrites_in_place(store: RunStore) -> None:
    _create(store)
    store.set_kv(RUN_ID, "k", "first")
    store.set_kv(RUN_ID, "k", "second")
    assert store.get_kv(RUN_ID, "k") == "second"


# --------------------------------------------------------------------------------------
# Provenance ordering and the JSONL mirror
# --------------------------------------------------------------------------------------


def test_provenance_keeps_emission_order_within_one_timestamp(store: RunStore) -> None:
    _create(store)
    actions = [f"step-{i}" for i in range(10)]
    for action in actions:
        # Every event shares NOW: ordering by ts alone could not distinguish them, so
        # this is really a test that reads order by seq.
        store.append_provenance(_event(action, timestamp=NOW))

    assert [event.action for event in store.list_provenance(RUN_ID)] == actions


def test_provenance_is_scoped_per_run(store: RunStore) -> None:
    _create(store, "run_00000000000a")
    _create(store, "run_00000000000b")
    store.append_provenance(_event("a-only", run_id="run_00000000000a"))
    store.append_provenance(_event("b-only", run_id="run_00000000000b"))

    assert [e.action for e in store.list_provenance("run_00000000000a")] == ["a-only"]
    assert [e.action for e in store.list_provenance("run_00000000000b")] == ["b-only"]


def test_provenance_mirror_is_appended_line_by_line(store: RunStore) -> None:
    _create(store)
    events = [_event(f"step-{i}") for i in range(5)]
    for event in events:
        store.append_provenance(event)

    lines = store.provenance_path(RUN_ID).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]
    assert [p["action"] for p in parsed] == [e.action for e in events]
    # The mirror must agree with the source of truth, not drift from it.
    assert [p["event_id"] for p in parsed] == [e.event_id for e in store.list_provenance(RUN_ID)]


def test_provenance_mirror_survives_a_reopened_store(root: Path) -> None:
    with RunStore(root) as first:
        _create(first)
        first.append_provenance(_event("before-close"))
    with RunStore(root) as second:
        second.append_provenance(_event("after-reopen"))

    lines = (root / "runs" / RUN_ID / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["action"] for line in lines] == ["before-close", "after-reopen"]


def test_provenance_records_an_error_payload(store: RunStore) -> None:
    _create(store)
    failed = _event("quantify").model_copy(update={"status": EventStatus.ERROR})
    store.append_provenance(failed)
    restored = store.list_provenance(RUN_ID)[0]
    assert restored.status is EventStatus.ERROR


def test_duplicate_event_id_is_rejected(store: RunStore) -> None:
    _create(store)
    event = _event("once")
    store.append_provenance(event)
    with pytest.raises(sqlite3.IntegrityError):
        store.append_provenance(event)


# --------------------------------------------------------------------------------------
# Durability and the migration guard
# --------------------------------------------------------------------------------------


def test_state_survives_close_and_reopen(root: Path) -> None:
    ref = _artifact()
    evidence = _evidence()
    with RunStore(root) as first:
        _create(first)
        first.add_artifact(RUN_ID, ref)
        first.add_evidence(RUN_ID, evidence)
        first.append_provenance(_event("quantify"))
        first.set_status(RUN_ID, RunStatus.OK)

    with RunStore(root) as second:
        manifest = second.get_manifest(RUN_ID)
        assert manifest.status is RunStatus.OK
        assert manifest.artifacts == [ref]
        assert manifest.evidence == [evidence]
        assert [e.action for e in second.list_provenance(RUN_ID)] == ["quantify"]


def test_reopening_does_not_reset_the_schema(root: Path) -> None:
    with RunStore(root) as first:
        _create(first)
    with RunStore(root) as second:
        assert second.get_run(RUN_ID) is not None
        assert second.get_kv(GLOBAL_SCOPE, "schema_version") == SCHEMA_VERSION


def test_a_newer_database_is_refused(root: Path) -> None:
    with RunStore(root) as first:
        _create(first)
        first.set_kv(GLOBAL_SCOPE, "schema_version", SCHEMA_VERSION + 1)

    with pytest.raises(BioflowError) as excinfo:
        RunStore(root)
    assert excinfo.value.error_code is ErrorCode.SCHEMA_MISMATCH
    assert excinfo.value.error.details["found"] == SCHEMA_VERSION + 1
    assert excinfo.value.error.remediation is not None
