"""The state machine an agent run moves through.

The transition set is declared, not derived. A language model asked to pick its own next
action will eventually pick a destructive one, so the set of moves that exist at all is
written down here in one readable block, and anything outside it raises rather than
proceeding. That is the difference between a guard and a convention.

Two things live here and nowhere else:

* ``TRANSITIONS`` -- which moves are legal at all.
* ``STATE_TOOLS`` -- which tools the model is even shown in a given state. It cannot
  call an ``EXECUTE``-class tool from ``TRIAGE`` because it is never told one exists,
  and :class:`~bioflow.orchestration.agent.Agent` rejects the call a second time if it
  invents the name anyway.

The module is deliberately free of I/O. It imports no run context and writes no
provenance, so the transition graph can be tested as a graph -- reachability, terminal
closure, and the absence of dead ends are properties of this file alone.
"""

from __future__ import annotations

from enum import StrEnum

from bioflow.core.models import BioflowError, ErrorCode, StructuredError

_RAISED_BY = "bioflow.orchestration.fsm"


class State(StrEnum):
    """Where a run currently is.

    ``NEEDS_REVIEW`` and ``HANDOFF`` are terminal *successes*, not failures: invariant
    #10 says refusing to proceed on ambiguous input is a correct answer. They are
    distinct because they ask for different things -- ``NEEDS_REVIEW`` wants a human to
    resolve an ambiguity BioFlow surfaced, ``HANDOFF`` wants work done somewhere BioFlow
    does not reach.
    """

    INTAKE = "INTAKE"
    TRIAGE = "TRIAGE"
    PLAN = "PLAN"
    AWAIT_CONFIRM = "AWAIT_CONFIRM"
    EXECUTE = "EXECUTE"
    PARSE = "PARSE"
    DIAGNOSE = "DIAGNOSE"
    INTERPRET = "INTERPRET"
    REPORT = "REPORT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    HANDOFF = "HANDOFF"
    DONE = "DONE"
    FAILED = "FAILED"


TERMINAL: frozenset[State] = frozenset(
    {State.DONE, State.FAILED, State.NEEDS_REVIEW, State.HANDOFF}
)
"""States a run stops in. Nothing legally leaves one."""

WORKING: frozenset[State] = frozenset(State) - TERMINAL
"""States a run can still move out of."""


# The declared graph, minus the two edges every working state has. Written as a mapping
# so the shape is readable; flattened into a frozenset of pairs below, which is what the
# membership test actually uses.
_FORWARD: dict[State, tuple[State, ...]] = {
    State.INTAKE: (State.TRIAGE,),
    State.TRIAGE: (State.PLAN,),
    # A plan whose every step is unconfirmed-and-harmless skips the gate entirely.
    State.PLAN: (State.AWAIT_CONFIRM, State.EXECUTE),
    State.AWAIT_CONFIRM: (State.EXECUTE,),
    # Execution either produced output to parse, or failed in a way worth diagnosing.
    State.EXECUTE: (State.PARSE, State.DIAGNOSE),
    State.PARSE: (State.INTERPRET, State.DIAGNOSE),
    # A diagnosis is only useful if it can lead to a repaired plan; the alternative is
    # to give up honestly, which the universal escapes below already allow.
    State.DIAGNOSE: (State.PLAN,),
    State.INTERPRET: (State.REPORT,),
    State.REPORT: (State.DONE,),
}

# Every working state can escape to review or handoff -- that is what the model's two
# control tools request -- and can fail, because a cap or an internal error can strike
# anywhere. Adding them here rather than to each row above keeps the intent visible:
# these edges are universal by design, not by accident of enumeration.
_UNIVERSAL: tuple[State, ...] = (State.NEEDS_REVIEW, State.HANDOFF, State.FAILED)

TRANSITIONS: frozenset[tuple[State, State]] = frozenset(
    [(frm, to) for frm, targets in _FORWARD.items() for to in targets]
    + [(frm, to) for frm in WORKING for to in _UNIVERSAL]
)
"""Every legal ``(from, to)`` move. Membership is the whole authorization check."""


# Which tools the model is offered in each state. Names are matched against the registry
# at run time, so a state may name a tool a later step will add; `tests/unit/test_fsm.py`
# asserts the reverse -- that every name here is registered -- so a typo fails the build
# rather than silently leaving the model with nothing to call.
STATE_TOOLS: dict[State, frozenset[str]] = {
    State.INTAKE: frozenset({"echo_artifact", "request_review", "handoff"}),
    State.TRIAGE: frozenset({"echo_artifact", "request_review", "handoff"}),
    State.PLAN: frozenset({"echo_artifact", "request_review", "handoff"}),
    # The gate itself takes no model input: a human or a policy answers, not the LLM.
    State.AWAIT_CONFIRM: frozenset(),
    # Neither does execution. The plan runs as deterministic code -- that is the whole
    # point of separating judgment from computation -- so the model is not in the loop
    # here and is offered nothing. The plan executor arrives with the capabilities.
    State.EXECUTE: frozenset(),
    State.PARSE: frozenset({"echo_artifact", "request_review", "handoff"}),
    State.DIAGNOSE: frozenset({"echo_artifact", "request_review", "handoff"}),
    State.INTERPRET: frozenset({"echo_artifact", "request_review", "handoff"}),
    State.REPORT: frozenset({"echo_artifact", "request_review", "handoff"}),
    State.NEEDS_REVIEW: frozenset(),
    State.HANDOFF: frozenset(),
    State.DONE: frozenset(),
    State.FAILED: frozenset(),
}

# Which state each control tool asks for. The model names a tool, never a state, so the
# only escapes it can request are the ones listed here.
CONTROL_TOOL_TARGETS: dict[str, State] = {
    "request_review": State.NEEDS_REVIEW,
    "handoff": State.HANDOFF,
}


class IllegalTransition(BioflowError):
    """Raised when a move outside :data:`TRANSITIONS` is attempted."""

    def __init__(self, frm: State, to: State) -> None:
        allowed = ", ".join(sorted(state.value for state in allowed_targets(frm))) or "(none)"
        super().__init__(
            StructuredError(
                error_code=ErrorCode.VALIDATION_FAILED,
                message=f"illegal state transition {frm.value} -> {to.value}",
                remediation=f"Legal targets from {frm.value}: {allowed}.",
                details={"from": frm.value, "to": to.value},
                raised_by=_RAISED_BY,
            )
        )
        self.frm = frm
        self.to = to


def is_legal(frm: State, to: State) -> bool:
    """Report whether a move is declared."""
    return (frm, to) in TRANSITIONS


def allowed_targets(frm: State) -> frozenset[State]:
    """Return every state legally reachable in one move from ``frm``."""
    return frozenset(to for source, to in TRANSITIONS if source == frm)


def allowed_tools(state: State) -> frozenset[str]:
    """Return the tool names the model may call in ``state``."""
    return STATE_TOOLS.get(state, frozenset())


class StateMachine:
    """A run's position in the graph, plus the record of how it got there.

    ``illegal_attempts`` is counted rather than merely raised. A model asking to escape
    from a state that forbids it is a fact about the model worth carrying into the run
    manifest, not just a traceback that vanishes when the process exits.
    """

    def __init__(self, initial: State = State.INTAKE) -> None:
        self.current = initial
        self.history: list[State] = [initial]
        self.illegal_attempts = 0

    @property
    def is_terminal(self) -> bool:
        """Whether the run has stopped."""
        return self.current in TERMINAL

    def can(self, to: State) -> bool:
        """Whether ``to`` is reachable in one move from here."""
        return is_legal(self.current, to)

    def transition(self, to: State) -> tuple[State, State]:
        """Move to ``to``, returning the accepted ``(from, to)`` pair.

        Raises :class:`IllegalTransition` on anything undeclared, after counting it.
        """
        frm = self.current
        if not is_legal(frm, to):
            self.illegal_attempts += 1
            raise IllegalTransition(frm, to)
        self.current = to
        self.history.append(to)
        return frm, to
