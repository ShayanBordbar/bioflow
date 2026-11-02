"""The reference tool: look up an artifact and describe it.

``echo_artifact`` exists to exercise the registry and the golden harness from day one,
but it is a real demonstration of the path-free contract rather than a stub. It takes an
artifact ID, returns :meth:`ArtifactRef.short`, and never reveals where the bytes live --
which is invariant #2 shown rather than asserted.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bioflow.core.models import (
    ArtifactId,
    BioflowError,
    ErrorCode,
    SideEffect,
    StructuredError,
)
from bioflow.core.registry import tool
from bioflow.core.runctx import RunContext


class EchoArtifactIn(BaseModel):
    """Arguments for ``echo_artifact``."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: ArtifactId = Field(description="Identifier of the artifact to describe.")


class EchoArtifactOut(BaseModel):
    """A one-line description of an artifact, safe to put back into a prompt."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: ArtifactId = Field(description="The artifact that was described.")
    artifact_type: str = Field(description="The artifact's type, such as 'fastq'.")
    summary: str = Field(description="One-line human summary. Contains no filesystem path.")


@tool(
    name="echo_artifact",
    version="1.0.0",
    summary="Describe a stored artifact in one line, given its identifier.",
    side_effect=SideEffect.NONE,
    tags=("reference", "introspection"),
)
def echo_artifact(params: EchoArtifactIn, ctx: RunContext) -> EchoArtifactOut:
    """Return a one-line summary of an artifact recorded against this run."""
    for ref in ctx.runstore.list_artifacts(ctx.run_id):
        if ref.artifact_id == params.artifact_id:
            return EchoArtifactOut(
                artifact_id=ref.artifact_id,
                artifact_type=ref.artifact_type.value,
                summary=ref.short(),
            )
    raise BioflowError(
        StructuredError(
            error_code=ErrorCode.NOT_FOUND,
            message=f"no artifact {params.artifact_id} in this run",
            remediation="Check the identifier, or list the run's artifacts first.",
            details={"artifact_id": params.artifact_id},
            raised_by="echo_artifact",
        )
    )
