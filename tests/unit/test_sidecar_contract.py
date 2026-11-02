"""Unit tests for the sidecar wire contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bioflow.adapters.sidecar.contract import (
    CONTRACT_VERSION,
    INPUT_IDENTITY_FIELDS,
    JOB_SCHEMA_FILENAME,
    REGENERATE_COMMAND,
    RESULT_SCHEMA_FILENAME,
    SCHEMA_DIR,
    SidecarJob,
    SidecarResult,
    SidecarStatus,
    canonical_job,
    canonical_json,
    export_schemas,
    job_schema,
    render_schema,
    result_schema,
)
from bioflow.core.ids import new_artifact_id
from bioflow.core.models import ArtifactRef, ArtifactType, ErrorCode, StructuredError

pytestmark = pytest.mark.unit


def make_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=new_artifact_id(),
        artifact_type=ArtifactType.FASTA,
        uri=f"cas://{'a' * 64}",
        sha256="a" * 64,
        size_bytes=99,
        created_at=datetime.now(UTC),
    )


# --------------------------------------------------------------------------------------
# The models
# --------------------------------------------------------------------------------------


def test_a_job_defaults_to_the_current_contract_version() -> None:
    assert SidecarJob(op="ping").contract_version == CONTRACT_VERSION == "1"


def test_a_job_with_another_contract_version_is_rejected() -> None:
    # Pinned to a literal so a v2 message cannot be validated as v1 by accident.
    with pytest.raises(ValidationError):
        SidecarJob(op="ping", contract_version="2")  # type: ignore[arg-type]


def test_an_empty_op_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SidecarJob(op="")


def test_unknown_fields_are_refused_on_both_messages() -> None:
    # A field this version does not know about is a version mismatch; dropping it
    # silently would let two incompatible implementations appear to agree.
    with pytest.raises(ValidationError):
        SidecarJob(op="ping", gpu=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SidecarResult(status=SidecarStatus.OK, extra=1)  # type: ignore[call-arg]


def test_both_messages_are_frozen() -> None:
    job = SidecarJob(op="ping")
    with pytest.raises(ValidationError):
        job.op = "other"  # type: ignore[misc]


def test_a_result_reports_success_plainly() -> None:
    assert SidecarResult(status=SidecarStatus.OK).ok
    assert not SidecarResult(status=SidecarStatus.ERROR).ok


def test_a_result_carries_a_structured_error() -> None:
    result = SidecarResult(
        status=SidecarStatus.ERROR,
        error=StructuredError(error_code=ErrorCode.OUT_OF_DOMAIN, message="sequence too long"),
    )
    assert result.error is not None
    assert result.error.error_code is ErrorCode.OUT_OF_DOMAIN


def test_observation_fields_default_to_empty() -> None:
    # Filled in by the runner, never by the sidecar: a process should not be asked to
    # tail its own logs or time itself.
    result = SidecarResult(status=SidecarStatus.OK)
    assert result.stdout_tail == ""
    assert result.stderr_tail == ""
    assert result.duration_ms == 0


def test_a_negative_duration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SidecarResult(status=SidecarStatus.OK, duration_ms=-1)


def test_a_result_round_trips_through_json() -> None:
    original = SidecarResult(
        status=SidecarStatus.OK,
        outputs=(make_ref(),),
        metrics={"auprc": 0.81},
        versions={"torch": "2.4.0"},
    )
    assert SidecarResult.model_validate(json.loads(original.model_dump_json())) == original


# --------------------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------------------


def test_the_canonical_form_keeps_only_content_identity_for_inputs() -> None:
    canonical = canonical_job(SidecarJob(op="ping", inputs=(make_ref(),)))
    entry = canonical["inputs"][0]  # type: ignore[index]
    assert set(entry) == set(INPUT_IDENTITY_FIELDS)  # type: ignore[arg-type]
    # The three that would break fixture reuse, and the one that staging rewrites.
    for dropped in ("artifact_id", "run_id", "created_at", "uri"):
        assert dropped not in entry  # type: ignore[operator]


def test_the_canonical_json_is_deterministic() -> None:
    job = SidecarJob(op="predict", params={"z": 1, "a": 2})
    assert canonical_json(job) == canonical_json(job)
    # Sorted keys, tight separators -- the form another language has to reproduce.
    assert '"a":2' in canonical_json(job)
    assert canonical_json(job).index('"a"') < canonical_json(job).index('"z"')


def test_nested_params_are_sorted_too() -> None:
    left = SidecarJob(op="p", params={"outer": {"b": 1, "a": 2}})
    right = SidecarJob(op="p", params={"outer": {"a": 2, "b": 1}})
    assert canonical_json(left) == canonical_json(right)


def test_the_canonical_form_records_the_contract_version() -> None:
    # A v1 job and a hypothetical v2 job asking the same thing are not the same job.
    assert canonical_job(SidecarJob(op="ping"))["contract_version"] == "1"


# --------------------------------------------------------------------------------------
# Exported schemas
# --------------------------------------------------------------------------------------


def test_the_checked_in_schemas_match_the_models() -> None:
    """The drift guard.

    Without it the checked-in schemas become decoration within one refactor, and a
    second-language implementation would be built against a contract Python no longer
    speaks.
    """
    for filename, schema in (
        (JOB_SCHEMA_FILENAME, job_schema()),
        (RESULT_SCHEMA_FILENAME, result_schema()),
    ):
        path = SCHEMA_DIR / filename
        assert path.is_file(), f"{filename} is missing; regenerate with:\n  {REGENERATE_COMMAND}"
        assert path.read_text(encoding="utf-8") == render_schema(schema), (
            f"{filename} is out of date with the model. Regenerate with:\n  {REGENERATE_COMMAND}"
        )


def test_the_schemas_are_named_for_the_contract_version() -> None:
    assert JOB_SCHEMA_FILENAME == "sidecar_job.v1.json"
    assert RESULT_SCHEMA_FILENAME == "sidecar_result.v1.json"


def test_the_job_schema_describes_the_documented_fields() -> None:
    properties = job_schema()["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"contract_version", "op", "params", "inputs", "seed"}


def test_the_result_schema_describes_the_documented_fields() -> None:
    properties = result_schema()["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {
        "contract_version",
        "status",
        "outputs",
        "metrics",
        "versions",
        "stdout_tail",
        "stderr_tail",
        "error",
        "duration_ms",
    }


def test_the_schemas_inline_the_artifact_ref_definition() -> None:
    # A second-language implementation reads these files and nothing else, so a dangling
    # reference to a Python class would make them unusable.
    schema = job_schema()
    assert "$defs" in schema
    defs = schema["$defs"]
    assert isinstance(defs, dict)
    assert "ArtifactRef" in defs


def test_export_writes_both_files(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert {path.name for path in written} == {JOB_SCHEMA_FILENAME, RESULT_SCHEMA_FILENAME}
    for path in written:
        json.loads(path.read_text(encoding="utf-8"))  # must be valid JSON


def test_export_is_idempotent(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    first = (tmp_path / JOB_SCHEMA_FILENAME).read_text()
    export_schemas(tmp_path)
    assert (tmp_path / JOB_SCHEMA_FILENAME).read_text() == first
