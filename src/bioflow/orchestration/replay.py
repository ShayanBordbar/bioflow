"""Deterministic re-execution of a recorded agent run.

This is what turns "we logged it" into "we can prove it". A trace plus its cassette is a
complete record of a run's judgment, and :func:`replay` re-runs the agent against both,
asserting it makes the same decisions. Used by evals and by CI, where a prompt change
that silently alters behaviour is exactly the regression nobody notices.

Three things have to be reconstructed or the comparison is meaningless:

* **The model's answers** come from the cassette, keyed by the canonicalized request. No
  network, no credentials.
* **The confirmation decisions** come from the trace. Without them a replay would decline
  where the original approved, and diverge for a reason that has nothing to do with the
  model.
* **The artifacts the run could see** are copied from the original run into the replay
  run. Tools look artifacts up by run, so an unseeded replay would return ``NOT_FOUND``
  everywhere and report a divergence that is entirely our own doing.

The replay is itself a run, recorded like any other with ``replay_of`` in its request.
Hiding it would make the provenance record a place where some executions are logged and
others are not, which is the property the whole system exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bioflow.adapters.llm.cassette import DEFAULT_CASSETTE_ROOT, CassetteClient
from bioflow.core.models import BioflowError, ErrorCode, StructuredError
from bioflow.core.runctx import RunContext
from bioflow.orchestration.agent import Agent, AgentLimits
from bioflow.orchestration.confirm import ReplayedPolicy
from bioflow.orchestration.context import ContextBuilder
from bioflow.orchestration.fsm import State
from bioflow.orchestration.trace import (
    parse_header,
    read_trace,
    recorded_confirmations,
    recorded_tool_calls,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from bioflow.config import Settings
    from bioflow.core.registry import Registry

_RAISED_BY = "bioflow.orchestration.replay"


@dataclass(frozen=True)
class ReplayResult:
    """The comparison between a recorded run and its re-execution."""

    ok: bool
    expected: tuple[str, ...]
    observed: tuple[str, ...]
    final_state: State
    expected_final_state: State | None
    first_divergence: int | None
    run_id: str

    def diff(self) -> str:
        """Render the divergence in a form worth reading in CI output.

        A bare "sequences differ" tells you nothing at 3am. This says which index, what
        was recorded, and what happened instead.
        """
        if self.ok:
            return (
                f"replay matched: {len(self.expected)} tool call(s), ended {self.final_state.value}"
            )

        lines = [
            f"replay diverged (run {self.run_id})",
            f"  expected: {list(self.expected)}",
            f"  observed: {list(self.observed)}",
        ]
        if self.first_divergence is not None:
            index = self.first_divergence
            was = self.expected[index] if index < len(self.expected) else "(nothing)"
            now = self.observed[index] if index < len(self.observed) else "(nothing)"
            lines.append(f"  first difference at index {index}: expected {was!r}, got {now!r}")
        if (
            self.expected_final_state is not None
            and self.expected_final_state is not self.final_state
        ):
            lines.append(
                f"  final state: expected {self.expected_final_state.value}, "
                f"got {self.final_state.value}"
            )
        return "\n".join(lines)

    def raise_for_mismatch(self) -> None:
        """Raise when the replay diverged. For callers that want an exception."""
        if self.ok:
            return
        raise BioflowError(
            StructuredError(
                error_code=ErrorCode.VALIDATION_FAILED,
                message="replay did not reproduce the recorded tool-call sequence",
                remediation=(
                    "Check whether a prompt, tool schema, or the state machine changed "
                    "since the trace was recorded."
                ),
                details={
                    "expected": list(self.expected),
                    "observed": list(self.observed),
                    "first_divergence": self.first_divergence,
                },
                raised_by=_RAISED_BY,
            )
        )


def replay(
    trace_path: Path | str,
    settings: Settings,
    *,
    cassette_root: Path = DEFAULT_CASSETTE_ROOT,
    root: Path | None = None,
    tools: Registry | None = None,
) -> ReplayResult:
    """Re-execute a recorded run against its cassette and compare the decisions."""
    trace_path = Path(trace_path)
    records = read_trace(trace_path)
    header = parse_header(records)
    expected = recorded_tool_calls(records)
    expected_final = _recorded_final_state(records)

    if header.cassette is None:
        raise BioflowError(
            StructuredError(
                error_code=ErrorCode.MISSING_INPUT,
                message=f"trace {trace_path.name} names no cassette, so it cannot be replayed",
                remediation=(
                    "Re-run with BIOFLOW_CASSETTE_MODE=record and a cassette name, which "
                    "is what records the model's answers."
                ),
                details={"trace": str(trace_path), "run_id": header.run_id},
                raised_by=_RAISED_BY,
            )
        )

    workspace = Path(root) if root is not None else _workspace_from(header, trace_path)

    with RunContext.create(workspace, {**header.request, "replay_of": header.run_id}) as ctx:
        _seed_artifacts(ctx, header.run_id)
        client = CassetteClient(None, name=header.cassette, mode="replay", root=cassette_root)
        version = header.prompt_version or "v1"
        agent = Agent(
            client,
            ctx,
            settings,
            tools=tools,
            limits=_limits_from(header.limits, settings),
            confirm=ReplayedPolicy(recorded_confirmations(records)),
            cassette=header.cassette,
            prompt_version=version,
            # Built from the *original* request, not this run's. The replay run records
            # `replay_of` for provenance, and that marker reaching the prompt would
            # change every cassette key and make replay diverge from its own bookkeeping.
            context=ContextBuilder(
                settings.agent_token_budget,
                store=ctx.store,
                registry=tools,
                request=header.request,
                window_turns=settings.agent_context_window_turns,
                prompt_version=version,
            ),
        )
        result = agent.run()
        run_id = ctx.run_id

    observed = list(result.tool_calls)
    divergence = _first_divergence(expected, observed)
    states_match = expected_final is None or expected_final is result.final_state

    return ReplayResult(
        ok=divergence is None and states_match,
        expected=tuple(expected),
        observed=tuple(observed),
        final_state=result.final_state,
        expected_final_state=expected_final,
        first_divergence=divergence,
        run_id=run_id,
    )


def _workspace_from(header: object, trace_path: Path) -> Path:
    """Locate the data root the original run lived in.

    Prefers the root recorded in the header; falls back to walking up from the trace
    (``<root>/runs/<run_id>/trace.jsonl``), which is how a trace copied without its
    header's absolute path still replays.
    """
    recorded_root = getattr(header, "root", None)
    if isinstance(recorded_root, str) and Path(recorded_root).is_dir():
        return Path(recorded_root)
    return trace_path.parent.parent.parent


def _seed_artifacts(ctx: RunContext, original_run_id: str) -> None:
    """Give the replay run the same artifacts the original could see."""
    if not original_run_id:
        return
    for ref in ctx.runstore.list_artifacts(original_run_id):
        ctx.runstore.add_artifact(ctx.run_id, ref)


def _limits_from(recorded: Mapping[str, object], settings: Settings) -> AgentLimits:
    """Rebuild the caps the original run used, falling back to configuration."""
    fallback = AgentLimits.from_settings(settings)
    return AgentLimits(
        max_steps=_as_int(recorded.get("max_steps"), fallback.max_steps),
        max_tool_calls_per_state=_as_int(
            recorded.get("max_tool_calls_per_state"), fallback.max_tool_calls_per_state
        ),
        max_wall_clock_seconds=_as_float(
            recorded.get("max_wall_clock_seconds"), fallback.max_wall_clock_seconds
        ),
        max_repair_attempts=_as_int(
            recorded.get("max_repair_attempts"), fallback.max_repair_attempts
        ),
    )


def _as_int(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _as_float(value: object, fallback: float) -> float:
    return (
        float(value) if isinstance(value, int | float) and not isinstance(value, bool) else fallback
    )


def _recorded_final_state(records: Sequence[Mapping[str, object]]) -> State | None:
    """Return the final state a trace's footer recorded, if it has one."""
    for record in records:
        if record.get("kind") == "footer":
            try:
                return State(str(record.get("final_state")))
            except ValueError:  # pragma: no cover - a footer with an unknown state
                return None
    return None


def _first_divergence(expected: Sequence[str], observed: Sequence[str]) -> int | None:
    """Return the first index at which two sequences differ, or ``None``."""
    for index in range(max(len(expected), len(observed))):
        left = expected[index] if index < len(expected) else None
        right = observed[index] if index < len(observed) else None
        if left != right:
            return index
    return None
