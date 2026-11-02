"""The two tools the model can use to stop.

The agent loop advances the happy path in code -- the model never names its next state.
These are the exceptions, and they are deliberately the only ones: the model can ask to
*leave*, never to go somewhere specific. That asymmetry is what keeps invariant #10
("ambiguous input produces NEEDS_REVIEW") reachable from inside the model's judgment
rather than something code has to infer on its behalf.

They are ordinary registered tools, not a side channel. Their arguments are validated,
their calls are recorded as provenance, and they carry goldens like everything else --
because an escape hatch that skips the enforcement path is not an escape hatch, it is a
hole.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bioflow.core.models import CostClass, SideEffect
from bioflow.core.registry import tool
from bioflow.core.runctx import RunContext


class RequestReviewIn(BaseModel):
    """Arguments for ``request_review``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        min_length=1,
        description="Why this run cannot proceed without a human. Be specific.",
    )
    blocking_question: str | None = Field(
        default=None,
        description="The single question whose answer would unblock the run, if there is one.",
    )


class RequestReviewOut(BaseModel):
    """Confirmation that a review request was recorded."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(description="The reason as recorded.")
    blocking_question: str | None = Field(description="The blocking question, if one was given.")


class HandoffIn(BaseModel):
    """Arguments for ``handoff``."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        min_length=1,
        description="Why the remaining work is outside what BioFlow can do.",
    )
    to: str = Field(
        min_length=1,
        description="Who or what should pick the work up, in plain words.",
    )


class HandoffOut(BaseModel):
    """Confirmation that a handoff was recorded."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(description="The reason as recorded.")
    to: str = Field(description="The intended recipient, as recorded.")


@tool(
    name="request_review",
    version="1.0.0",
    summary=(
        "Stop and ask for human review. Use when the input is ambiguous, when two "
        "readings are equally supported, or when proceeding would require a guess."
    ),
    side_effect=SideEffect.NONE,
    cost_class=CostClass.FREE,
    tags=("control", "escape"),
)
def request_review(params: RequestReviewIn, ctx: RunContext) -> RequestReviewOut:
    """Record a request for human review and end the run in ``NEEDS_REVIEW``."""
    return RequestReviewOut(reason=params.reason, blocking_question=params.blocking_question)


@tool(
    name="handoff",
    version="1.0.0",
    summary=(
        "Stop and hand the remaining work to someone else. Use when the work is "
        "well-defined but outside what BioFlow can carry out."
    ),
    side_effect=SideEffect.NONE,
    cost_class=CostClass.FREE,
    tags=("control", "escape"),
)
def handoff(params: HandoffIn, ctx: RunContext) -> HandoffOut:
    """Record a handoff and end the run in ``HANDOFF``."""
    return HandoffOut(reason=params.reason, to=params.to)
