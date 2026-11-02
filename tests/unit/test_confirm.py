"""Unit tests for the confirmation policies."""

import pytest

from bioflow.core.models import CostClass, SideEffect, ToolSpec
from bioflow.orchestration.confirm import (
    AutoApproveBelow,
    ConfirmationPolicy,
    DenyAllPolicy,
    ReplayedPolicy,
)

pytestmark = pytest.mark.unit


def spec(cost: CostClass) -> ToolSpec:
    return ToolSpec(
        name="run_pipeline",
        version="1.0.0",
        summary="Run something that costs money.",
        side_effect=SideEffect.EXECUTE,
        cost_class=cost,
        requires_confirmation=True,
    )


@pytest.mark.parametrize("policy", [DenyAllPolicy(), AutoApproveBelow(), ReplayedPolicy([True])])
def test_every_policy_satisfies_the_protocol(policy: object) -> None:
    assert isinstance(policy, ConfirmationPolicy)


# --------------------------------------------------------------------------------------
# DenyAll
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("cost", list(CostClass))
def test_deny_all_declines_regardless_of_cost(cost: CostClass) -> None:
    # The default for an unattended run: nobody is there to approve, so nothing runs.
    assert DenyAllPolicy().confirm(spec(cost), {}) is False


# --------------------------------------------------------------------------------------
# AutoApproveBelow
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "cost", "expected"),
    [
        (CostClass.MODERATE, CostClass.FREE, True),
        (CostClass.MODERATE, CostClass.CHEAP, True),
        (CostClass.MODERATE, CostClass.MODERATE, True),  # "at or below"
        (CostClass.MODERATE, CostClass.EXPENSIVE, False),
        (CostClass.FREE, CostClass.FREE, True),
        (CostClass.FREE, CostClass.CHEAP, False),
        (CostClass.EXPENSIVE, CostClass.EXPENSIVE, True),
    ],
)
def test_auto_approve_respects_the_threshold(
    threshold: CostClass, cost: CostClass, expected: bool
) -> None:
    assert AutoApproveBelow(threshold).confirm(spec(cost), {}) is expected


def test_auto_approve_defaults_to_the_configured_confirmation_threshold() -> None:
    # Matches Settings.require_confirmation_above, so --yes and the config agree.
    assert AutoApproveBelow().threshold is CostClass.MODERATE


def test_auto_approve_is_not_approve_everything() -> None:
    # The point of a threshold is that unattended approval stops somewhere.
    assert AutoApproveBelow(CostClass.MODERATE).confirm(spec(CostClass.EXPENSIVE), {}) is False


# --------------------------------------------------------------------------------------
# ReplayedPolicy
# --------------------------------------------------------------------------------------


def test_replayed_decisions_are_served_in_order() -> None:
    policy = ReplayedPolicy([True, False, True])
    assert [policy.confirm(spec(CostClass.EXPENSIVE), {}) for _ in range(3)] == [
        True,
        False,
        True,
    ]


def test_replay_declines_once_the_recorded_decisions_run_out() -> None:
    # A replay that asks more questions than the original stops rather than inventing
    # an answer -- an invented approval would be an approval nobody ever gave.
    policy = ReplayedPolicy([True])
    assert policy.confirm(spec(CostClass.EXPENSIVE), {}) is True
    assert policy.confirm(spec(CostClass.EXPENSIVE), {}) is False


def test_an_empty_replay_policy_declines() -> None:
    assert ReplayedPolicy().confirm(spec(CostClass.FREE), {}) is False
