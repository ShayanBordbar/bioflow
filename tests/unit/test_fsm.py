"""Unit tests for the agent state machine.

The transition set is a security boundary, so it is tested as a graph rather than by
spot-checking a few moves: reachability, terminal closure, and the absence of dead ends
are properties that must hold for every state, including ones added later.
"""

from collections import deque

import pytest

import bioflow.tools  # noqa: F401  (registers the built-in tools)
from bioflow.core.models import ErrorCode
from bioflow.core.registry import registry
from bioflow.orchestration.fsm import (
    CONTROL_TOOL_TARGETS,
    STATE_TOOLS,
    TERMINAL,
    TRANSITIONS,
    WORKING,
    IllegalTransition,
    State,
    StateMachine,
    allowed_targets,
    allowed_tools,
    is_legal,
)

pytestmark = pytest.mark.unit


def _reachable_from(start: State) -> set[State]:
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for target in allowed_targets(current):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


# --------------------------------------------------------------------------------------
# The graph as a graph
# --------------------------------------------------------------------------------------


def test_every_state_is_reachable_from_intake() -> None:
    # A state nothing can reach is dead code that looks like a feature.
    assert _reachable_from(State.INTAKE) == set(State)


def test_terminal_states_have_no_exits() -> None:
    for state in TERMINAL:
        assert allowed_targets(state) == frozenset(), state


def test_working_and_terminal_partition_the_states() -> None:
    assert set(State) == WORKING | TERMINAL
    assert not (WORKING & TERMINAL)


def test_every_working_state_can_reach_a_terminal_state() -> None:
    for state in WORKING:
        assert _reachable_from(state) & TERMINAL, state


@pytest.mark.parametrize("escape", [State.NEEDS_REVIEW, State.HANDOFF, State.FAILED])
def test_every_working_state_can_escape_in_one_move(escape: State) -> None:
    # Refusing to proceed, and failing a cap, must always be one move away. A state that
    # could not stop would be a state that has to guess.
    for state in WORKING:
        assert is_legal(state, escape), f"{state} cannot reach {escape}"


def test_no_transition_leaves_a_terminal_state() -> None:
    assert not [pair for pair in TRANSITIONS if pair[0] in TERMINAL]


def test_the_forward_path_is_declared() -> None:
    for frm, to in (
        (State.INTAKE, State.TRIAGE),
        (State.TRIAGE, State.PLAN),
        (State.PLAN, State.EXECUTE),
        (State.PLAN, State.AWAIT_CONFIRM),
        (State.AWAIT_CONFIRM, State.EXECUTE),
        (State.EXECUTE, State.PARSE),
        (State.PARSE, State.INTERPRET),
        (State.INTERPRET, State.REPORT),
        (State.REPORT, State.DONE),
    ):
        assert is_legal(frm, to), f"{frm} -> {to}"


@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (State.TRIAGE, State.EXECUTE),  # skipping the plan entirely
        (State.INTAKE, State.DONE),  # concluding without doing anything
        (State.DONE, State.INTAKE),  # restarting a finished run
        (State.NEEDS_REVIEW, State.PLAN),  # continuing past a review request
        (State.EXECUTE, State.PLAN),  # replanning without diagnosing
    ],
)
def test_undeclared_moves_are_illegal(frm: State, to: State) -> None:
    assert not is_legal(frm, to)


# --------------------------------------------------------------------------------------
# The whitelist
# --------------------------------------------------------------------------------------


def test_every_state_has_a_whitelist_entry() -> None:
    # A missing entry would read as "no tools" and silently strand the state.
    assert set(STATE_TOOLS) == set(State)


def test_every_whitelisted_tool_is_registered() -> None:
    # This is the typo guard. A name here that no tool answers to would leave the model
    # with fewer options than intended, and nothing else would notice.
    for state, names in STATE_TOOLS.items():
        for name in names:
            assert name in registry, f"{state.value} whitelists unregistered tool {name!r}"


def test_terminal_states_offer_no_tools() -> None:
    for state in TERMINAL:
        assert allowed_tools(state) == frozenset(), state


def test_execution_does_not_consult_the_model() -> None:
    # Computation is deterministic and happens without the LLM; offering it a tool here
    # would be offering it a say in something it must not have a say in.
    assert allowed_tools(State.EXECUTE) == frozenset()
    assert allowed_tools(State.AWAIT_CONFIRM) == frozenset()


def test_confirmation_tools_only_appear_where_the_gate_is_reachable() -> None:
    # A confirmation-requiring tool offered in a state with no edge to AWAIT_CONFIRM
    # would be a tool the model can pick and the machine cannot process.
    for state, names in STATE_TOOLS.items():
        gated = [n for n in names if (e := registry.get(n)) and e.spec.requires_confirmation]
        if gated:
            assert is_legal(state, State.AWAIT_CONFIRM), f"{state.value} offers {gated}"


# --------------------------------------------------------------------------------------
# Control tools
# --------------------------------------------------------------------------------------


def test_control_tools_are_registered_and_terminal() -> None:
    for name, target in CONTROL_TOOL_TARGETS.items():
        assert name in registry, name
        assert target in TERMINAL, target


def test_control_tools_are_offered_wherever_the_model_is_asked() -> None:
    # The model must always be able to decline. A state that asks for judgment without
    # offering a way out is a state that forces a guess.
    for state in WORKING:
        offered = allowed_tools(state)
        if offered:
            assert set(CONTROL_TOOL_TARGETS) <= offered, state


# --------------------------------------------------------------------------------------
# StateMachine
# --------------------------------------------------------------------------------------


def test_a_new_machine_starts_at_intake() -> None:
    machine = StateMachine()
    assert machine.current is State.INTAKE
    assert machine.history == [State.INTAKE]
    assert not machine.is_terminal


def test_a_legal_move_returns_the_pair_and_records_history() -> None:
    machine = StateMachine()
    assert machine.transition(State.TRIAGE) == (State.INTAKE, State.TRIAGE)
    assert machine.current is State.TRIAGE
    assert machine.history == [State.INTAKE, State.TRIAGE]


def test_an_illegal_move_raises_and_leaves_the_state_untouched() -> None:
    machine = StateMachine()
    with pytest.raises(IllegalTransition) as caught:
        machine.transition(State.DONE)
    assert machine.current is State.INTAKE
    assert machine.history == [State.INTAKE]
    assert caught.value.error.error_code is ErrorCode.VALIDATION_FAILED


def test_illegal_attempts_are_counted_not_merely_raised() -> None:
    # The count is what reaches the run manifest; a traceback vanishes with the process.
    machine = StateMachine()
    for target in (State.DONE, State.REPORT, State.PARSE):
        with pytest.raises(IllegalTransition):
            machine.transition(target)
    assert machine.illegal_attempts == 3


def test_the_illegal_move_error_names_what_was_allowed() -> None:
    machine = StateMachine(State.TRIAGE)
    with pytest.raises(IllegalTransition) as caught:
        machine.transition(State.DONE)
    remediation = caught.value.error.remediation or ""
    assert "PLAN" in remediation
    assert "NEEDS_REVIEW" in remediation
    assert caught.value.error.details == {"from": "TRIAGE", "to": "DONE"}


def test_can_reports_reachability_without_moving() -> None:
    machine = StateMachine()
    assert machine.can(State.TRIAGE)
    assert not machine.can(State.DONE)
    assert machine.current is State.INTAKE


def test_is_terminal_becomes_true_at_the_end() -> None:
    machine = StateMachine(State.REPORT)
    machine.transition(State.DONE)
    assert machine.is_terminal
