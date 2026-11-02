"""Who gets to say yes to a tool that asked for confirmation.

A tool declaring ``requires_confirmation`` is saying it wants a decision from outside the
loop. That decision cannot be made by the model -- it is the thing the model is being
checked on -- and it cannot be made by ``orchestration`` importing the CLI, because
imports only ever point downward. So it arrives as an injected policy.

The default is :class:`DenyAllPolicy`. A run nobody is watching declines, which turns an
unattended agent's most expensive mistake into a ``NEEDS_REVIEW`` instead of an action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bioflow.core.models import CostClass

if TYPE_CHECKING:
    from bioflow.core.models import JSONValue, ToolSpec

# CostClass is a StrEnum, so it carries no ordering of its own. This is the only place
# that needs one; it moves into core/models.py when a second consumer appears.
COST_RANK: dict[CostClass, int] = {
    CostClass.FREE: 0,
    CostClass.CHEAP: 1,
    CostClass.MODERATE: 2,
    CostClass.EXPENSIVE: 3,
}


@runtime_checkable
class ConfirmationPolicy(Protocol):
    """Decides whether a confirmation-requiring tool call may proceed."""

    def confirm(self, spec: ToolSpec, arguments: dict[str, JSONValue]) -> bool:
        """Return ``True`` to allow the call."""
        ...


class DenyAllPolicy:
    """Declines every confirmation request.

    The default, and the correct behaviour for any non-interactive run: nobody is there
    to approve, so nothing that asked for approval runs.
    """

    def confirm(self, spec: ToolSpec, arguments: dict[str, JSONValue]) -> bool:
        """Decline."""
        return False


class AutoApproveBelow:
    """Approves calls at or below a cost threshold, declining the rest.

    This backs ``--yes``. It is deliberately not "approve everything": the point of
    ``settings.require_confirmation_above`` is that unattended approval stops somewhere,
    and an unknown cost class stops there too.
    """

    def __init__(self, threshold: CostClass = CostClass.MODERATE) -> None:
        self.threshold = threshold

    def confirm(self, spec: ToolSpec, arguments: dict[str, JSONValue]) -> bool:
        """Approve when the tool's cost class does not exceed the threshold."""
        limit = COST_RANK.get(self.threshold)
        cost = COST_RANK.get(spec.cost_class)
        if limit is None or cost is None:  # pragma: no cover - CostClass is closed
            return False
        return cost <= limit


class ReplayedPolicy:
    """Replays confirmation decisions recorded in a trace.

    Without this, a run that approved an expensive tool could never be replayed: the
    replay would decline where the original approved and diverge for a reason that has
    nothing to do with the model. Running past the recorded decisions declines, so a
    replay that asks more questions than the original stops rather than inventing answers.
    """

    def __init__(self, decisions: list[bool] | None = None) -> None:
        self.decisions = list(decisions or [])
        self.index = 0

    def confirm(self, spec: ToolSpec, arguments: dict[str, JSONValue]) -> bool:
        """Return the next recorded decision, or decline if there are none left."""
        if self.index >= len(self.decisions):
            return False
        decision = self.decisions[self.index]
        self.index += 1
        return decision
