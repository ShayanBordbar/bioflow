"""The agent loop.

This is the project's central claim in executable form: a language model performing
judgment inside a system where nothing it says is trusted. Four mechanisms do the work,
and they are all small on purpose.

**The state machine.** Every move is checked against a declared transition set
(:mod:`bioflow.orchestration.fsm`). Code advances the happy path; the model may only
*escape*, by calling ``request_review`` or ``handoff``, and even those targets are
validated. The model never names a state.

**The per-state whitelist.** In each state the model is shown only the tools legal
there, and a call to anything else is rejected before it reaches
:func:`~bioflow.core.registry.invoke`. It cannot call an ``EXECUTE``-class tool from
``TRIAGE`` because it is never told one exists, and because saying the name anyway does
not work.

**Repair.** A malformed call comes back as the specific fields that were wrong, up to
``llm_max_repair_attempts`` times. Argument validation is checked with
:func:`~bioflow.core.registry.validate_args` *before* invoking, so an argument the model
can fix is never confused with a tool that failed on its own terms -- retrying the
second kind can never succeed.

**Caps.** Steps, tool calls per state, and wall clock. Every one of them ends the run in
``FAILED`` with ``RESOURCE_LIMIT`` naming which cap was hit and what it was set to.

The conversation is kept as a plain text transcript rather than native tool-use blocks.
Those blocks are provider-shaped, and translating them here would pull provider knowledge
into ``orchestration`` -- which ADR-0001's amendment puts in the adapters. The cost is
that tool results are re-stated as text; the benefit is that this module never learns
what an Anthropic content block looks like.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from bioflow.adapters.llm.base import LLMResponse
from bioflow.core.ids import new_event_id
from bioflow.core.models import (
    Actor,
    BioflowError,
    ErrorCode,
    EventStatus,
    JSONValue,
    ProvenanceEvent,
    RunStatus,
    StructuredError,
    ToolSpec,
)
from bioflow.core.registry import Registry, invoke, validate_args
from bioflow.core.registry import registry as default_registry
from bioflow.orchestration.confirm import ConfirmationPolicy, DenyAllPolicy
from bioflow.orchestration.context import BuiltContext, ContextBuilder, Turn
from bioflow.orchestration.fsm import (
    CONTROL_TOOL_TARGETS,
    IllegalTransition,
    State,
    StateMachine,
    allowed_tools,
)
from bioflow.orchestration.trace import (
    TRACE_FILENAME,
    TraceWriter,
    arguments_digest,
    messages_digest,
)
from bioflow.prompts.loader import DEFAULT_VERSION

if TYPE_CHECKING:
    from bioflow.adapters.llm.base import LLMClient
    from bioflow.config import Settings
    from bioflow.core.runctx import RunContext

_RAISED_BY = "bioflow.orchestration.agent"

# Where each state goes when its work succeeds. The model has no say in this: it chooses
# a tool, and the tool succeeding is what advances the run.
_SUCCESSOR: dict[State, State] = {
    State.INTAKE: State.TRIAGE,
    State.TRIAGE: State.PLAN,
    State.PLAN: State.EXECUTE,
    State.AWAIT_CONFIRM: State.EXECUTE,
    State.EXECUTE: State.PARSE,
    State.PARSE: State.INTERPRET,
    State.DIAGNOSE: State.PLAN,
    State.INTERPRET: State.REPORT,
    State.REPORT: State.DONE,
}

# Where a state goes when a tool fails at run time rather than being malformed. Only the
# two states that actually run scientific work have somewhere useful to go; everywhere
# else a repeated failure is a failure.
_ON_FAILURE: dict[State, State] = {
    State.EXECUTE: State.DIAGNOSE,
    State.PARSE: State.DIAGNOSE,
}

# How a final state is reported as a run outcome. HANDOFF maps to NEEDS_REVIEW because
# both mean "a human has to pick this up", and RunStatus deliberately has no third word
# for that.
_RUN_STATUS: dict[State, RunStatus] = {
    State.DONE: RunStatus.OK,
    State.FAILED: RunStatus.ERROR,
    State.NEEDS_REVIEW: RunStatus.NEEDS_REVIEW,
    State.HANDOFF: RunStatus.NEEDS_REVIEW,
}


@dataclass(frozen=True)
class AgentLimits:
    """The hard caps on one agent run."""

    max_steps: int = 25
    max_tool_calls_per_state: int = 8
    max_wall_clock_seconds: float = 7200.0
    max_repair_attempts: int = 2

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentLimits:
        """Read the caps from configuration.

        The wall clock comes from ``max_runtime_minutes`` rather than a setting of its
        own: the agent loop is part of the run, not something with a separate allowance.
        """
        return cls(
            max_steps=settings.agent_max_steps,
            max_tool_calls_per_state=settings.agent_max_tool_calls_per_state,
            max_wall_clock_seconds=float(settings.max_runtime_minutes) * 60.0,
            max_repair_attempts=settings.llm_max_repair_attempts,
        )

    def as_dict(self) -> dict[str, JSONValue]:
        """Render the caps for the trace header."""
        return {
            "max_steps": self.max_steps,
            "max_tool_calls_per_state": self.max_tool_calls_per_state,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "max_repair_attempts": self.max_repair_attempts,
        }


@dataclass
class AgentMetrics:
    """How the run behaved, as opposed to what it cost.

    A "call slot" is one attempt by a state to obtain a usable tool call. It may take
    several LLM calls, if the first ones are malformed. The two rates are computed over
    slots, not over LLM calls, because that is the unit a human means by "did it get it
    right first time".
    """

    n_llm_calls: int = 0
    n_call_slots: int = 0
    n_first_attempt_valid: int = 0
    n_slots_repaired: int = 0
    n_repairs_succeeded: int = 0
    n_out_of_state_calls: int = 0
    n_state_transitions: int = 0
    illegal_transition_attempts: int = 0
    n_tool_invocations: int = 0

    def as_dict(self) -> dict[str, JSONValue]:
        """Render the counters and the two derived rates.

        The rates are ``None`` rather than ``0.0`` when nothing was attempted. A zero
        would read as "everything failed", which is a very different claim from
        "nothing happened".
        """
        return {
            "n_llm_calls": self.n_llm_calls,
            "n_call_slots": self.n_call_slots,
            "n_first_attempt_valid": self.n_first_attempt_valid,
            "n_slots_repaired": self.n_slots_repaired,
            "n_repairs_succeeded": self.n_repairs_succeeded,
            "n_out_of_state_calls": self.n_out_of_state_calls,
            "n_state_transitions": self.n_state_transitions,
            "illegal_transition_attempts": self.illegal_transition_attempts,
            "n_tool_invocations": self.n_tool_invocations,
            "first_attempt_valid_call_rate": _rate(self.n_first_attempt_valid, self.n_call_slots),
            "repair_success_rate": _rate(self.n_repairs_succeeded, self.n_slots_repaired),
        }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


@dataclass(frozen=True)
class AgentResult:
    """What one agent run produced."""

    final_state: State
    error: StructuredError | None
    metrics: AgentMetrics
    tool_calls: tuple[str, ...]
    states: tuple[State, ...]

    @property
    def ok(self) -> bool:
        """Whether the run reached ``DONE``."""
        return self.final_state is State.DONE


@dataclass(frozen=True)
class _Chosen:
    """A tool call the model asked for that survived every pre-invocation check."""

    name: str
    arguments: dict[str, JSONValue]
    spec: ToolSpec


@dataclass(frozen=True)
class _Outcome:
    """Where a state decided the run should go next."""

    target: State
    error: StructuredError | None = None


@dataclass
class _Slot:
    """Bookkeeping for one attempt to obtain a usable tool call."""

    chosen: _Chosen | None = None
    error: StructuredError | None = None
    problems: list[JSONValue] = field(default_factory=list)


class Agent:
    """Drives a run through the state machine, one validated tool call at a time."""

    def __init__(
        self,
        llm: LLMClient,
        ctx: RunContext,
        settings: Settings,
        *,
        tools: Registry | None = None,
        limits: AgentLimits | None = None,
        confirm: ConfirmationPolicy | None = None,
        cassette: str | None = None,
        prompt_version: str = DEFAULT_VERSION,
        initial_state: State = State.INTAKE,
        context: ContextBuilder | None = None,
    ) -> None:
        self.llm = llm
        self.ctx = ctx
        self.settings = settings
        # An empty Registry is falsy, so this must test against None.
        self.registry = tools if tools is not None else default_registry
        self.limits = limits if limits is not None else AgentLimits.from_settings(settings)
        # Fails closed: with nobody to ask, a tool that wanted confirmation does not run.
        self.confirm = confirm if confirm is not None else DenyAllPolicy()
        self.cassette = cassette
        self.prompt_version = prompt_version

        self.machine = StateMachine(initial_state)
        self.metrics = AgentMetrics()
        self.tool_calls: list[str] = []
        self.confirmations: list[bool] = []
        self.trace = TraceWriter(ctx.run_dir / TRACE_FILENAME)
        self._started = time.monotonic()

        record = ctx.runstore.get_run(ctx.run_id)
        # Context assembly lives entirely in the builder. The agent decides *what* to
        # ask; the builder decides how much of it fits.
        self.context = (
            context
            if context is not None
            else ContextBuilder(
                settings.agent_token_budget,
                store=ctx.store,
                registry=self.registry,
                request=record.request if record is not None else {},
                window_turns=settings.agent_context_window_turns,
                prompt_version=prompt_version,
            )
        )
        self.system = self.context.system

        # Written by _ask / _invoke / build and read by the caller immediately afterwards.
        self._last_latency_ms = 0
        self._last_error: StructuredError | None = None
        self._messages: list[dict[str, JSONValue]] = []

    # -- the loop ---------------------------------------------------------------------

    def run(self) -> AgentResult:
        """Drive the machine to a terminal state and return what happened."""
        self._started = time.monotonic()
        record = self.ctx.runstore.get_run(self.ctx.run_id)
        self.trace.header(
            run_id=self.ctx.run_id,
            request=record.request if record is not None else {},
            cassette=self.cassette,
            model=self.settings.llm_model,
            prompt_version=self.prompt_version,
            limits=self.limits.as_dict(),
            root=str(self.ctx.runstore.root),
        )

        error: StructuredError | None = None
        steps = 0

        while not self.machine.is_terminal:
            cap = self._cap_exceeded(steps)
            if cap is not None:
                error = cap
                self._force_failed()
                break
            steps += 1
            try:
                outcome = self._run_state(self.machine.current)
                if outcome.error is not None:
                    error = outcome.error
                self._move(outcome.target)
            except IllegalTransition as exc:
                # Already counted by the machine. Recorded here so a run manifest shows
                # the model asked for something the graph forbids.
                error = exc.error
                self._force_failed()
                break

        final = self.machine.current
        self.ctx.flush_usage()
        self.ctx.runstore.set_metrics(self.ctx.run_id, self.metrics.as_dict())
        self.ctx.set_status(_RUN_STATUS.get(final, RunStatus.ERROR))
        self.trace.footer(
            final_state=final.value,
            tool_calls=self.tool_calls,
            metrics=self.metrics.as_dict(),
            error=None if error is None else error.model_dump(mode="json"),
        )
        return AgentResult(
            final_state=final,
            error=error,
            metrics=self.metrics,
            tool_calls=tuple(self.tool_calls),
            states=tuple(self.machine.history),
        )

    def _cap_exceeded(self, steps: int) -> StructuredError | None:
        """Return the error for whichever hard cap has been reached, if any."""
        if steps >= self.limits.max_steps:
            return self._limit_error("max_steps", self.limits.max_steps, steps)
        elapsed = time.monotonic() - self._started
        if elapsed > self.limits.max_wall_clock_seconds:
            return self._limit_error(
                "max_wall_clock_seconds", self.limits.max_wall_clock_seconds, round(elapsed, 3)
            )
        return None

    def _limit_error(self, name: str, limit: float | int, reached: float | int) -> StructuredError:
        return StructuredError(
            error_code=ErrorCode.RESOURCE_LIMIT,
            message=f"agent run hit {name}={limit}",
            remediation=f"Raise {name}, or narrow the request so it needs fewer steps.",
            details={"limit_name": name, "limit": limit, "reached": reached},
            raised_by=_RAISED_BY,
        )

    def _run_state(self, state: State) -> _Outcome:
        """Work one state until it produces a transition target."""
        if not allowed_tools(state):
            # A state that offers the model nothing does not ask it anything. EXECUTE is
            # the real case: the plan runs as deterministic code, so there is no judgment
            # to solicit and the run simply advances. The plan executor lands with the
            # capabilities in Step 2; until then this is a pass-through.
            return _Outcome(_SUCCESSOR.get(state, State.FAILED))

        calls = 0
        last_error: StructuredError | None = None

        while calls < self.limits.max_tool_calls_per_state:
            slot = self._obtain_call(state)
            if slot.chosen is None:
                return _Outcome(State.FAILED, slot.error)

            chosen = slot.chosen
            calls += 1

            if chosen.spec.requires_confirmation:
                return self._confirm_and_invoke(chosen)

            result = self._invoke(chosen, confirmed=False)
            if result is not None:
                return result
            last_error = self._last_error

        # Every cap reports RESOURCE_LIMIT, so "we ran out of budget" always reads the
        # same way. The tool failure that consumed the budget is carried in details
        # rather than being reported instead -- it is the cause, not the outcome.
        cap = self._limit_error(
            "max_tool_calls_per_state", self.limits.max_tool_calls_per_state, calls
        )
        if last_error is not None:
            cap = cap.model_copy(
                update={"details": {**cap.details, "last_tool_error": last_error.message}}
            )
        return _Outcome(_ON_FAILURE.get(state, State.FAILED), cap)

    def _confirm_and_invoke(self, chosen: _Chosen) -> _Outcome:
        """Take a confirmation-requiring call through the gate.

        The run moves to ``AWAIT_CONFIRM`` first, so the pause is visible in the
        provenance record rather than happening inside a function call. The model is
        offered no tools there: the decision belongs to the policy, and the call it is
        deciding about was already chosen and validated.
        """
        self._move(State.AWAIT_CONFIRM)
        approved = self.confirm.confirm(chosen.spec, chosen.arguments)
        self.confirmations.append(approved)
        self.trace.call(
            state=State.AWAIT_CONFIRM.value,
            prompt_version=self.prompt_version,
            messages_sha256=messages_digest(self._messages),
            tool_call={"name": chosen.name, "arguments": dict(chosen.arguments)},
            validation_errors=[],
            repair_attempt=0,
            confirmed=approved,
            usage={},
            latency_ms=0,
            model=self.settings.llm_model,
            outcome="confirmed" if approved else "declined",
        )
        if not approved:
            # Declining is a correct answer, not a failure. Invariant #10.
            self._record_turn(
                chosen, "declined", "not confirmed, so it did not run; stopping for review"
            )
            return _Outcome(State.NEEDS_REVIEW)

        result = self._invoke(chosen, confirmed=True)
        if result is not None:
            return result
        return _Outcome(State.FAILED, self._last_error)

    def _invoke(self, chosen: _Chosen, *, confirmed: bool) -> _Outcome | None:
        """Run a validated call. Returns ``None`` when the state should try again."""
        result = invoke(
            chosen.name,
            chosen.arguments,
            self.ctx,
            confirmed=confirmed,
            into=self.registry,
        )
        self.metrics.n_tool_invocations += 1

        if not result.ok:
            # The tool itself failed. That is information for the model, not a malformed
            # call, so it is fed back rather than repaired.
            self._last_error = result.error
            message = "the tool failed" if result.error is None else result.error.message
            self._record_turn(chosen, "failed", message)
            return None

        self.tool_calls.append(chosen.name)
        summary = "" if result.output is None else result.output.model_dump_json()
        self._record_turn(chosen, "ok", summary)

        escape = CONTROL_TOOL_TARGETS.get(chosen.name)
        if escape is not None:
            return _Outcome(escape)
        return _Outcome(_SUCCESSOR.get(self.machine.current, State.FAILED))

    def _record_turn(self, chosen: _Chosen, outcome: str, detail: str) -> None:
        """Add a completed exchange to the context builder's history."""
        self.context.record_turn(
            Turn(
                state=self.machine.current.value,
                tool=chosen.name,
                arguments=dict(chosen.arguments),
                outcome=outcome,
                detail=detail,
            )
        )

    # -- obtaining one usable call ----------------------------------------------------

    def _obtain_call(self, state: State) -> _Slot:
        """Ask the model for a tool call, repairing malformed ones up to the cap."""
        self.metrics.n_call_slots += 1

        repairs = 0
        attempt = 0
        slot = _Slot()
        correction: str | None = None

        while True:
            try:
                # Rebuilt every attempt: the correction has to reach the model, and the
                # budget has to be re-satisfied now that it is there.
                built = self._build(state, correction)
                response = self._ask(built)
            except BioflowError as exc:
                # The client itself failed -- a cassette miss, an exhausted rate limit,
                # an unreachable endpoint. Not something the model can correct, so the
                # repair loop is skipped and the state ends here.
                self._trace_llm_failure(state, attempt, exc.error)
                slot.error = exc.error
                return slot

            problem = self._inspect(response, state, attempt)

            if isinstance(problem, _Chosen):
                if attempt == 0:
                    self.metrics.n_first_attempt_valid += 1
                else:
                    self.metrics.n_repairs_succeeded += 1
                slot.chosen = problem
                return slot

            slot.error = problem
            slot.problems.append(problem.model_dump(mode="json"))
            if repairs >= self.limits.max_repair_attempts:
                return slot

            if repairs == 0:
                self.metrics.n_slots_repaired += 1
            repairs += 1
            attempt += 1
            correction = (
                f"## Correction needed\n\nThat call was rejected: {problem.message} "
                f"{problem.remediation or 'Correct it and call again.'}"
            )

    def _build(self, state: State, correction: str | None) -> BuiltContext:
        """Assemble the request for one attempt, within the token budget."""
        built = self.context.build(
            state,
            artifacts=self.ctx.runstore.list_artifacts(self.ctx.run_id),
            correction=correction,
        )
        self._messages = built.messages
        return built

    def _ask(self, built: BuiltContext) -> LLMResponse:
        """Make one LLM call, folding its usage into the run."""
        started = time.monotonic()
        response = self.llm.complete(
            built.messages,
            tools=built.tools,
            # "any" forces a tool call: a state exists to produce one, and prose is not
            # an outcome the machine can act on.
            tool_choice={"type": "any"},
            system=built.system,
        )
        self.metrics.n_llm_calls += 1
        self.ctx.usage.record(response)
        self._last_latency_ms = int((time.monotonic() - started) * 1000) or response.latency_ms
        return response

    def _trace_llm_failure(self, state: State, attempt: int, error: StructuredError) -> None:
        """Record a call that never produced a response."""
        self.trace.call(
            state=state.value,
            prompt_version=self.prompt_version,
            messages_sha256=messages_digest(self._messages),
            tool_call=None,
            validation_errors=[error.model_dump(mode="json")],
            repair_attempt=attempt,
            confirmed=None,
            usage={},
            latency_ms=self._last_latency_ms,
            model=self.settings.llm_model,
            outcome="llm_error",
        )

    def _inspect(
        self, response: LLMResponse, state: State, attempt: int
    ) -> _Chosen | StructuredError:
        """Check one response, tracing it, and return either a usable call or the fault."""
        allowed = allowed_tools(state)
        call = response.tool_calls[0] if response.tool_calls else None
        problem: StructuredError | None = None
        outcome = "invoked"

        if response.refused:
            problem = StructuredError(
                error_code=ErrorCode.VALIDATION_FAILED,
                message="the model declined to answer",
                remediation="Rephrase the request, or stop the run.",
                raised_by=_RAISED_BY,
            )
            outcome = "refused"
        elif call is None:
            names = ", ".join(sorted(allowed)) or "(none)"
            problem = StructuredError(
                error_code=ErrorCode.VALIDATION_FAILED,
                message="no tool call in the response; exactly one is required",
                remediation=f"Call one of: {names}.",
                details={"state": state.value},
                raised_by=_RAISED_BY,
            )
            outcome = "no_tool_call"
        elif call.name not in allowed:
            # The model invented a name it was never shown. Rejected before invoke() so
            # an out-of-state tool cannot run even if it exists in the registry.
            self.metrics.n_out_of_state_calls += 1
            names = ", ".join(sorted(allowed)) or "(none)"
            problem = StructuredError(
                error_code=ErrorCode.VALIDATION_FAILED,
                message=f"tool {call.name!r} is not available in state {state.value}",
                remediation=f"Call one of: {names}.",
                details={"state": state.value, "requested": call.name, "allowed": sorted(allowed)},
                raised_by=_RAISED_BY,
            )
            outcome = "out_of_state"
        else:
            problem = validate_args(call.name, dict(call.arguments), into=self.registry)
            if problem is not None:
                outcome = "invalid_arguments"

        self.trace.call(
            state=state.value,
            prompt_version=self.prompt_version,
            messages_sha256=messages_digest(self._messages),
            tool_call=None
            if call is None
            else {
                "name": call.name,
                "arguments": dict(call.arguments),
                "params_sha256": arguments_digest(dict(call.arguments)),
            },
            validation_errors=[] if problem is None else [problem.model_dump(mode="json")],
            repair_attempt=attempt,
            confirmed=None,
            usage=response.usage.model_dump(mode="json"),
            latency_ms=self._last_latency_ms,
            model=response.model or self.settings.llm_model,
            outcome=outcome,
        )

        if problem is not None:
            return problem

        assert call is not None  # every branch above that leaves call None sets problem
        entry = self.registry.get(call.name)
        assert entry is not None  # validate_args returned None, so the tool is registered
        return _Chosen(name=call.name, arguments=dict(call.arguments), spec=entry.spec)

    # -- transitions ------------------------------------------------------------------

    def _move(self, to: State) -> None:
        """Take a transition and record it, or let the illegal-move error propagate."""
        frm, target = self.machine.transition(to)
        self.metrics.n_state_transitions += 1
        self.metrics.illegal_transition_attempts = self.machine.illegal_attempts
        self.ctx.emit(
            ProvenanceEvent(
                event_id=new_event_id(),
                run_id=self.ctx.run_id,
                timestamp=datetime.now(UTC),
                actor=Actor.AGENT,
                action=f"transition {frm.value} -> {target.value}",
                status=EventStatus.OK,
            )
        )

    def _force_failed(self) -> None:
        """End the run in ``FAILED``, keeping the illegal-attempt count current."""
        self.metrics.illegal_transition_attempts = self.machine.illegal_attempts
        if not self.machine.is_terminal:
            self._move(State.FAILED)
