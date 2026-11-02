"""The wire contract between BioFlow and a heavy runtime.

Invariant #9 says torch, Polygraph, and gReLU are never imported into ``bioflow.core`` or
``bioflow.capabilities``. This module is what makes that structural rather than
aspirational: a heavy runtime is reached by writing JSON to a file, running a process, and
reading JSON back. Nothing else crosses.

The contract is JSON, not Python. These Pydantic models are BioFlow's *view* of it —
a sidecar written in another language, or in the same language without pydantic
installed, is a first-class implementation. That is why the JSON Schemas are exported to
``schemas/`` and checked in: they, not this file, are the thing a second implementation
reads.

``contract_version`` is on both messages and is pinned to a literal. A sidecar built
against v1 that receives a v2 job must fail loudly rather than guess which fields it can
ignore.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bioflow.core.models import ArtifactRef, JSONValue, StructuredError

# Typed as the literal, not as str, so the model fields below can default to it without
# widening: a Literal["1"] field defaulting to a plain str would type-check as a mismatch.
CONTRACT_VERSION: Literal["1"] = "1"

SCHEMA_DIR = Path(__file__).resolve().parents[4] / "schemas"
JOB_SCHEMA_FILENAME = f"sidecar_job.v{CONTRACT_VERSION}.json"
RESULT_SCHEMA_FILENAME = f"sidecar_result.v{CONTRACT_VERSION}.json"

REGENERATE_COMMAND = (
    'uv run python -c "'
    'from bioflow.adapters.sidecar.contract import export_schemas; export_schemas()"'
)

# The only ArtifactRef fields that identify an input's *content*. Everything else --
# artifact_id, run_id, created_at, produced_by, metadata -- varies between two runs that
# are doing the same work, and a fixture keyed on them would miss every time.
#
# `uri` is excluded for a sharper reason: staging rewrites it to a workdir-relative name
# before the sidecar sees it. Including it would make one job hash differently before and
# after staging, which is the single thing this hash must never do.
INPUT_IDENTITY_FIELDS: tuple[str, ...] = ("artifact_type", "sha256", "size_bytes", "media_type")


class SidecarStatus(StrEnum):
    """Whether the sidecar completed its op."""

    OK = "ok"
    ERROR = "error"


class _ContractModel(BaseModel):
    """Base for both messages.

    ``extra="forbid"`` is doing real work here: a sidecar that sends a field this version
    does not know about is a version mismatch, and silently dropping it would let two
    incompatible implementations appear to agree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class SidecarJob(_ContractModel):
    """One unit of work handed to a heavy runtime.

    ``seed`` is separate from ``params`` rather than buried in it because determinism is
    a property of the whole job, and a caller reproducing a run needs to find it without
    knowing the op's parameter names.
    """

    contract_version: Literal["1"] = CONTRACT_VERSION
    op: str = Field(min_length=1)
    params: dict[str, JSONValue] = Field(default_factory=dict)
    inputs: tuple[ArtifactRef, ...] = ()
    seed: int | None = None


class SidecarResult(_ContractModel):
    """What a heavy runtime reports back.

    ``stdout_tail``, ``stderr_tail``, and ``duration_ms`` are filled in by the *runner*,
    not by the sidecar. A process should not be asked to tail its own logs or time
    itself: those are the caller's observations, and a sidecar that reported them could
    misreport them — including by omission, when it is the hang that needs explaining.
    """

    contract_version: Literal["1"] = CONTRACT_VERSION
    status: SidecarStatus
    outputs: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, JSONValue] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: StructuredError | None = None
    duration_ms: int = Field(default=0, ge=0)

    @property
    def ok(self) -> bool:
        """Whether the op succeeded."""
        return self.status is SidecarStatus.OK


# --------------------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------------------


def canonical_job(job: SidecarJob) -> dict[str, JSONValue]:
    """Reduce a job to the fields that define *what work was asked for*.

    Written as an explicit allowlist rather than a heuristic. The LLM cassette
    canonicalizer strips anything matching ``.*_id$``, which is fine for a key private to
    one module; this hash is part of a cross-language contract, so the rule has to be
    something a sidecar written in another language can follow exactly.
    """
    return {
        "contract_version": job.contract_version,
        "op": job.op,
        "params": json.loads(json.dumps(job.params, sort_keys=True, default=str)),
        "seed": job.seed,
        "inputs": [
            {field: getattr(ref, field) for field in INPUT_IDENTITY_FIELDS} for ref in job.inputs
        ],
    }


def canonical_json(job: SidecarJob) -> str:
    """Render a job's canonical form. Sorted keys, tight separators, no whitespace."""
    return json.dumps(canonical_job(job), sort_keys=True, separators=(",", ":"), default=str)


def job_sha256(job: SidecarJob) -> str:
    """The fixture key for a job. Stable across dict ordering and across run identity."""
    return hashlib.sha256(canonical_json(job).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Schema export
# --------------------------------------------------------------------------------------


def job_schema() -> dict[str, JSONValue]:
    """The JSON Schema for a job message."""
    schema: dict[str, JSONValue] = SidecarJob.model_json_schema()
    return schema


def result_schema() -> dict[str, JSONValue]:
    """The JSON Schema for a result message."""
    schema: dict[str, JSONValue] = SidecarResult.model_json_schema()
    return schema


def render_schema(schema: dict[str, JSONValue]) -> str:
    """Render a schema exactly as it is written to disk, so drift can be compared."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def export_schemas(root: Path | None = None) -> list[Path]:
    """Write both schemas to ``schemas/``, returning the paths written.

    Checked in rather than generated at build time: a second implementation in another
    language needs to read them from the repository, and a file nobody can see until
    after a Python build is not a contract.
    """
    directory = Path(root) if root is not None else SCHEMA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, schema in (
        (JOB_SCHEMA_FILENAME, job_schema()),
        (RESULT_SCHEMA_FILENAME, result_schema()),
    ):
        path = directory / filename
        path.write_text(render_schema(schema), encoding="utf-8")
        written.append(path)
    return written
