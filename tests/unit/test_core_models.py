"""Unit tests for the core domain models.

Four properties are load-bearing and get explicit coverage: identifier formats are
validated on the way in, frozen records cannot be mutated, every model survives a JSON
round trip unchanged, and ``ErrorCode`` is a closed set.
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from bioflow.core.models import (
    Actor,
    ArtifactRef,
    ArtifactType,
    BioflowError,
    CostClass,
    ErrorCode,
    EventStatus,
    Evidence,
    EvidenceType,
    ProvenanceEvent,
    RunManifest,
    RunStatus,
    SideEffect,
    StructuredError,
    ToolSpec,
)

pytestmark = pytest.mark.unit

SHA = "a" * 64
NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Sample builders
# --------------------------------------------------------------------------------------


def _artifact(**overrides: Any) -> ArtifactRef:
    kwargs: dict[str, Any] = {
        "artifact_id": "art_0123456789ab",
        "artifact_type": ArtifactType.FASTQ,
        "uri": "/runs/abc/reads.fastq.gz",
        "sha256": SHA,
        "size_bytes": 12_900_000,
        "media_type": "application/gzip",
        "produced_by": "kallisto@0.50.1",
        "run_id": "run_0123456789ab",
        "created_at": NOW,
        "metadata": {"read_length": 150, "paired": True, "lanes": [1, 2]},
    }
    return ArtifactRef(**(kwargs | overrides))


def _error(**overrides: Any) -> StructuredError:
    kwargs: dict[str, Any] = {
        "error_code": ErrorCode.EXTERNAL_TOOL_FAILED,
        "message": "kallisto quant exited with code 1",
        "remediation": "Check that the index matches the read length.",
        "details": {"exit_code": 1, "stderr_tail": ["Error: index not found"]},
        "retryable": False,
        "raised_by": "bioflow.adapters.kallisto",
    }
    return StructuredError(**(kwargs | overrides))


def _evidence(**overrides: Any) -> Evidence:
    kwargs: dict[str, Any] = {
        "evidence_id": "Eabc123",
        "evidence_type": EvidenceType.MEASURED,
        "statement": "Median transcript integrity number across samples is 8.4.",
        "value": 8.4,
        "units": "TIN",
        "artifacts": ["art_0123456789ab"],
        "produced_by": "bioflow.capabilities.qc",
        "provenance_refs": ["ev_0123456789ab"],
        "caveats": ["Computed on a 10% read subsample."],
    }
    return Evidence(**(kwargs | overrides))


def _event(**overrides: Any) -> ProvenanceEvent:
    kwargs: dict[str, Any] = {
        "event_id": "ev_0123456789ab",
        "run_id": "run_0123456789ab",
        "timestamp": NOW,
        "actor": Actor.ADAPTER,
        "action": "quantify",
        "tool": "kallisto",
        "tool_version": "0.50.1",
        "inputs": ["art_0123456789ab"],
        "outputs": ["art_ba9876543210"],
        "params_sha256": SHA,
        "duration_ms": 41_233,
        "status": EventStatus.OK,
        "error": None,
    }
    return ProvenanceEvent(**(kwargs | overrides))


def _tool_spec(**overrides: Any) -> ToolSpec:
    kwargs: dict[str, Any] = {
        "name": "quantify_rnaseq",
        "version": "1.2.0",
        "summary": "Quantify transcript abundance from paired FASTQ reads.",
        "input_schema": {"type": "object", "required": ["reads"]},
        "output_schema": {"type": "object"},
        "side_effect": SideEffect.EXECUTE,
        "cost_class": CostClass.EXPENSIVE,
        "requires_confirmation": True,
        "tags": ["rna-seq", "quantification"],
    }
    return ToolSpec(**(kwargs | overrides))


def _manifest(**overrides: Any) -> RunManifest:
    kwargs: dict[str, Any] = {
        "run_id": "run_0123456789ab",
        "created_at": NOW,
        "request": {"assay": "rna-seq", "samples": 6},
        "bioflow_version": "0.1.0",
        "python_version": "3.12.13",
        "git_commit": "e89f70d",
        "artifacts": [_artifact()],
        "evidence": [_evidence()],
        "status": RunStatus.OK,
        "tool_versions": {"kallisto": "0.50.1"},
        "llm_usage": {
            "model": "claude-opus-5",
            "prompt_version": "v1",
            "input_tokens": 1200,
            "output_tokens": 340,
            "cost_usd": 0.031,
            "n_calls": 2,
        },
    }
    return RunManifest(**(kwargs | overrides))


ALL_SAMPLES = [_artifact, _error, _evidence, _event, _tool_spec, _manifest]
FROZEN_SAMPLES = [_artifact, _error, _evidence, _event, _tool_spec]


# --------------------------------------------------------------------------------------
# Identifier and value format validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "artifact_id",
    [
        "art_0123456789ab",
        "art_ffffffffffff",
    ],
)
def test_artifact_id_accepts_well_formed_ids(artifact_id: str) -> None:
    assert _artifact(artifact_id=artifact_id).artifact_id == artifact_id


@pytest.mark.parametrize(
    "artifact_id",
    [
        "",
        "0123456789ab",  # missing prefix
        "art_0123456789",  # 10 hex digits
        "art_0123456789abc",  # 13 hex digits
        "art_0123456789AB",  # uppercase hex
        "art_0123456789ag",  # non-hex character
        "ART_0123456789ab",  # uppercase prefix
        " art_0123456789ab",  # leading whitespace
    ],
)
def test_artifact_id_rejects_malformed_ids(artifact_id: str) -> None:
    with pytest.raises(ValidationError):
        _artifact(artifact_id=artifact_id)


@pytest.mark.parametrize("evidence_id", ["Eabc123", "E000000", "Effffff"])
def test_evidence_id_accepts_well_formed_ids(evidence_id: str) -> None:
    assert _evidence(evidence_id=evidence_id).evidence_id == evidence_id


@pytest.mark.parametrize(
    "evidence_id",
    ["", "abc123", "E12345", "E1234567", "EABC123", "E-12345"],
)
def test_evidence_id_rejects_malformed_ids(evidence_id: str) -> None:
    with pytest.raises(ValidationError):
        _evidence(evidence_id=evidence_id)


@pytest.mark.parametrize(
    "run_id",
    ["", "run_0123456789", "0123456789ab", "art_0123456789ab", "run_0123456789AB"],
)
def test_run_id_rejects_malformed_ids(run_id: str) -> None:
    with pytest.raises(ValidationError):
        _manifest(run_id=run_id)


@pytest.mark.parametrize(
    "event_id",
    ["", "ev_0123456789", "event_0123456789ab", "ev_0123456789abc"],
)
def test_event_id_rejects_malformed_ids(event_id: str) -> None:
    with pytest.raises(ValidationError):
        _event(event_id=event_id)


@pytest.mark.parametrize("sha256", ["a" * 63, "a" * 65, "A" * 64, "z" * 64, ""])
def test_sha256_rejects_non_hex_digests(sha256: str) -> None:
    with pytest.raises(ValidationError):
        _artifact(sha256=sha256)


def test_sha256_is_optional() -> None:
    assert _artifact(sha256=None).sha256 is None


@pytest.mark.parametrize("version", ["1.2.0", "0.0.1", "1.0.0-rc.1", "1.0.0-rc.1+build.5"])
def test_tool_version_accepts_semver(version: str) -> None:
    assert _tool_spec(version=version).version == version


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0.0", "01.0.0", "latest", ""])
def test_tool_version_rejects_non_semver(version: str) -> None:
    with pytest.raises(ValidationError):
        _tool_spec(version=version)


@pytest.mark.parametrize("size_bytes", [-1, -1024])
def test_size_bytes_must_be_non_negative(size_bytes: int) -> None:
    with pytest.raises(ValidationError):
        _artifact(size_bytes=size_bytes)


def test_duration_ms_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        _event(duration_ms=-1)


def test_uri_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        _artifact(uri="")


def test_artifact_type_is_closed() -> None:
    with pytest.raises(ValidationError):
        _artifact(artifact_type="bam")


# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _artifact(created_at=datetime(2026, 8, 23, 12, 0))


def test_offset_datetime_is_normalized_to_utc() -> None:
    kolkata = timezone(timedelta(hours=5, minutes=30))
    artifact = _artifact(created_at=datetime(2026, 8, 23, 17, 30, tzinfo=kolkata))
    assert artifact.created_at.tzinfo is UTC
    assert artifact.created_at == NOW


def test_event_timestamp_is_also_tz_checked() -> None:
    with pytest.raises(ValidationError):
        _event(timestamp=datetime(2026, 8, 23, 12, 0))


# --------------------------------------------------------------------------------------
# Frozen-ness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", FROZEN_SAMPLES, ids=lambda b: b.__name__)
def test_frozen_models_reject_assignment(build: Any) -> None:
    model = build()
    field = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError):
        setattr(model, field, None)


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (_evidence(), "artifacts"),
        (_evidence(), "provenance_refs"),
        (_evidence(), "caveats"),
        (_event(), "inputs"),
        (_event(), "outputs"),
        (_tool_spec(), "tags"),
    ],
    ids=lambda x: x if isinstance(x, str) else type(x).__name__,
)
def test_frozen_sequences_are_tuples(model: BaseModel, field: str) -> None:
    # Tuples are what stop a caller from silently appending to a provenance list.
    assert isinstance(getattr(model, field), tuple)


def test_run_manifest_is_mutable() -> None:
    # A run accumulates artifacts as it proceeds, so the manifest is deliberately open.
    manifest = _manifest(artifacts=[], status=RunStatus.RUNNING)
    manifest.artifacts.append(_artifact())
    manifest.status = RunStatus.OK
    assert len(manifest.artifacts) == 1
    assert manifest.status is RunStatus.OK


@pytest.mark.parametrize("build", ALL_SAMPLES, ids=lambda b: b.__name__)
def test_unknown_fields_are_rejected(build: Any) -> None:
    with pytest.raises(ValidationError):
        build(definitely_not_a_field="x")


# --------------------------------------------------------------------------------------
# JSON round trips
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", ALL_SAMPLES, ids=lambda b: b.__name__)
def test_json_round_trip_preserves_equality(build: Any) -> None:
    model = build()
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_round_trip_of_event_carrying_an_error() -> None:
    event = _event(status=EventStatus.ERROR, error=_error())
    restored = ProvenanceEvent.model_validate_json(event.model_dump_json())
    assert restored == event
    assert restored.error is not None
    assert restored.error.error_code is ErrorCode.EXTERNAL_TOOL_FAILED
    assert restored.error.details["exit_code"] == 1


def test_round_trip_of_manifest_preserves_nested_records() -> None:
    manifest = _manifest()
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.artifacts[0].artifact_id == "art_0123456789ab"
    assert restored.evidence[0].evidence_type is EvidenceType.MEASURED
    assert restored.llm_usage["n_calls"] == 2


def test_enums_serialize_as_their_string_values() -> None:
    dumped = _event(status=EventStatus.SKIPPED).model_dump(mode="json")
    assert dumped["actor"] == "ADAPTER"
    assert dumped["status"] == "SKIPPED"
    assert _artifact().model_dump(mode="json")["artifact_type"] == "fastq"


def test_created_at_serializes_as_utc_iso8601() -> None:
    dumped = _artifact().model_dump(mode="json")
    assert dumped["created_at"].startswith("2026-08-23T12:00:00")
    assert dumped["created_at"].endswith("Z") or dumped["created_at"].endswith("+00:00")


def test_sequences_serialize_as_json_arrays() -> None:
    assert _evidence().model_dump(mode="json")["artifacts"] == ["art_0123456789ab"]


def test_nested_json_metadata_survives_the_round_trip() -> None:
    nested = {"a": {"b": [1, 2.5, "three", None, True]}}
    artifact = _artifact(metadata=nested)
    assert ArtifactRef.model_validate_json(artifact.model_dump_json()).metadata == nested


# --------------------------------------------------------------------------------------
# ErrorCode is a closed enum
# --------------------------------------------------------------------------------------


def test_error_code_membership_is_exactly_the_documented_set() -> None:
    assert {code.value for code in ErrorCode} == {
        "VALIDATION_FAILED",
        "MISSING_INPUT",
        "AMBIGUOUS_INPUT",
        "UNSUPPORTED_ASSAY",
        "OUT_OF_DOMAIN",
        "EXTERNAL_TOOL_FAILED",
        "TIMEOUT",
        "RESOURCE_LIMIT",
        "NOT_FOUND",
        "RATE_LIMITED",
        "SCHEMA_MISMATCH",
        "INTERNAL",
    }


def test_error_code_rejects_unknown_members() -> None:
    with pytest.raises(ValueError):
        ErrorCode("NOPE")


def test_structured_error_rejects_an_unknown_code() -> None:
    with pytest.raises(ValidationError):
        _error(error_code="NOPE")


def test_error_code_compares_equal_to_its_string() -> None:
    assert ErrorCode.TIMEOUT == "TIMEOUT"


def test_structured_error_defaults_are_conservative() -> None:
    error = StructuredError(error_code=ErrorCode.INTERNAL, message="boom")
    assert error.retryable is False
    assert error.details == {}
    assert error.remediation is None
    assert error.raised_by is None


# --------------------------------------------------------------------------------------
# BioflowError
# --------------------------------------------------------------------------------------


def test_bioflow_error_carries_its_payload() -> None:
    error = _error()
    with pytest.raises(BioflowError) as excinfo:
        raise BioflowError(error)
    assert excinfo.value.error == error
    assert excinfo.value.error_code is ErrorCode.EXTERNAL_TOOL_FAILED
    assert str(excinfo.value) == error.message


# --------------------------------------------------------------------------------------
# ArtifactRef.short()
# --------------------------------------------------------------------------------------


def test_short_summarizes_on_one_line_without_leaking_the_uri() -> None:
    artifact = _artifact()
    summary = artifact.short()
    assert "\n" not in summary
    assert artifact.artifact_id in summary
    # Invariant: the LLM handles artifact IDs only and must never see a path.
    assert artifact.uri not in summary
    assert "reads.fastq.gz" not in summary
    assert "[fastq]" in summary
    assert "12.3 MB" in summary
    assert "sha256:aaaaaaaa" in summary
    assert "via kallisto@0.50.1" in summary


def test_short_drops_optional_parts_when_unset() -> None:
    summary = _artifact(sha256=None, size_bytes=None, produced_by=None).short()
    assert summary == "art_0123456789ab [fastq]"


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (5 * 1024**2, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
        (2 * 1024**4, "2.0 TB"),
        (9 * 1024**5, "9216.0 TB"),
    ],
)
def test_short_renders_sizes_in_human_units(size_bytes: int, expected: str) -> None:
    assert expected in _artifact(size_bytes=size_bytes).short()


# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------


def test_run_manifest_defaults_are_empty() -> None:
    manifest = RunManifest(
        run_id="run_0123456789ab",
        created_at=NOW,
        bioflow_version="0.1.0",
        python_version="3.12.13",
    )
    assert manifest.status is RunStatus.PENDING
    assert manifest.artifacts == []
    assert manifest.evidence == []
    assert manifest.request == {}
    assert manifest.tool_versions == {}
    assert manifest.llm_usage == {}
    assert manifest.git_commit is None


def test_artifact_schema_version_defaults_to_one() -> None:
    assert _artifact().schema_version == "1"


def test_default_collections_are_not_shared_between_instances() -> None:
    first = StructuredError(error_code=ErrorCode.INTERNAL, message="a")
    second = StructuredError(error_code=ErrorCode.INTERNAL, message="b")
    assert first.details is not second.details


def test_tool_spec_defaults_are_least_privilege() -> None:
    spec = ToolSpec(name="noop", version="0.1.0", summary="Does nothing.")
    assert spec.side_effect is SideEffect.NONE
    assert spec.requires_confirmation is False
    assert spec.tags == ()
