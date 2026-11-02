"""Unit tests for the agent loop.

Everything here runs against :class:`FakeLLMClient`. No network, no credentials, and no
real model -- which is the point: the behaviour being tested is the loop's, and a real
model would make every assertion probabilistic.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

import bioflow.tools  # noqa: F401  (registers the built-in tools)
from bioflow.adapters.llm.base import LLMResponse, StopReason, ToolCall, Usage
from bioflow.adapters.llm.cassette import CassetteClient
from bioflow.adapters.llm.fake_client import FakeLLMClient
from bioflow.config import Settings
from bioflow.core.models import (
    ArtifactType,
    CostClass,
    ErrorCode,
    JSONValue,
    RunStatus,
    SideEffect,
)
from bioflow.core.registry import Registry, tool
from bioflow.core.runctx import RunContext
from bioflow.orchestration import fsm
from bioflow.orchestration.agent import Agent, AgentLimits
from bioflow.orchestration.confirm import AutoApproveBelow, DenyAllPolicy
from bioflow.orchestration.fsm import State
from bioflow.orchestration.replay import replay
from bioflow.orchestration.trace import TRACE_FILENAME, read_trace

pytestmark = pytest.mark.unit

MISSING_ARTIFACT = "art_000000000000"


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(data_dir=workspace)


@pytest.fixture
def ctx(workspace: Path) -> Iterator[RunContext]:
    with RunContext.create(workspace, {"ask": "triage this dataset"}) as context:
        yield context


@pytest.fixture
def artifact_id(ctx: RunContext) -> str:
    ref = ctx.record(
        ctx.store.put_bytes(
            b">seq1\nACGT\n", ArtifactType.FASTA, produced_by="bioflow@0.1.0", run_id=ctx.run_id
        )
    )
    return ref.artifact_id


def tool_call(name: str, **arguments: JSONValue) -> LLMResponse:
    """One scripted response asking for a single tool call."""
    return LLMResponse(
        tool_calls=(ToolCall(name=name, arguments=arguments, call_id="call_1"),),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(input_tokens=100, output_tokens=20),
        model="fake-model",
    )


def prose(text: str = "I think we should proceed.") -> LLMResponse:
    """A response with no tool call at all."""
    return LLMResponse(text=text, stop_reason=StopReason.END_TURN, model="fake-model")


def refusal() -> LLMResponse:
    return LLMResponse(stop_reason=StopReason.REFUSAL, model="fake-model")


def only(**whitelist: frozenset[str]) -> dict[State, frozenset[str]]:
    """Build a STATE_TOOLS override where every unnamed state offers nothing.

    States offering nothing are pass-throughs, so a test can isolate one state's
    behaviour without scripting responses for the whole pipeline.
    """
    table = dict.fromkeys(State, frozenset[str]())
    for name, names in whitelist.items():
        table[State[name]] = names
    return table


# The number of states that ask the model something on the unmodified happy path.
HAPPY_PATH_CALLS = 6


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------


def test_the_happy_path_reaches_done(ctx: RunContext, settings: Settings, artifact_id: str) -> None:
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    result = Agent(llm, ctx, settings).run()

    assert result.final_state is State.DONE
    assert result.ok
    assert result.error is None
    assert result.states == (
        State.INTAKE,
        State.TRIAGE,
        State.PLAN,
        State.EXECUTE,
        State.PARSE,
        State.INTERPRET,
        State.REPORT,
        State.DONE,
    )
    assert result.tool_calls == ("echo_artifact",) * HAPPY_PATH_CALLS


def test_execute_advances_without_asking_the_model(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    # Computation is deterministic. The loop passes through EXECUTE without spending a
    # call, which is why exactly six responses drive seven states.
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    result = Agent(llm, ctx, settings).run()
    assert State.EXECUTE in result.states
    assert result.metrics.n_llm_calls == HAPPY_PATH_CALLS
    assert len(llm.calls) == HAPPY_PATH_CALLS


def test_the_run_status_and_metrics_land_in_the_manifest(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    Agent(llm, ctx, settings).run()

    manifest = ctx.manifest()
    assert manifest.status is RunStatus.OK
    assert manifest.metrics["n_llm_calls"] == HAPPY_PATH_CALLS
    assert manifest.metrics["first_attempt_valid_call_rate"] == 1.0
    # Nothing needed repair, so the rate is None rather than 0.0 -- "nothing happened",
    # not "everything failed".
    assert manifest.metrics["repair_success_rate"] is None
    assert manifest.llm_usage["input_tokens"] == 100 * HAPPY_PATH_CALLS
    assert manifest.llm_usage["n_calls"] == HAPPY_PATH_CALLS


def test_each_transition_leaves_a_provenance_event(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    result = Agent(llm, ctx, settings).run()

    events = ctx.runstore.list_provenance(ctx.run_id)
    transitions = [e for e in events if e.action.startswith("transition ")]
    assert len(transitions) == result.metrics.n_state_transitions == len(result.states) - 1
    assert transitions[0].action == "transition INTAKE -> TRIAGE"
    # Tool invocations are recorded too, by invoke(), so both layers are visible.
    assert [e for e in events if e.tool == "echo_artifact"]


def test_the_model_is_never_shown_a_filesystem_path(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    # Invariant #2, asserted against what actually went over the wire.
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    Agent(llm, ctx, settings).run()

    sent = repr(llm.calls)
    assert str(ctx.run_dir) not in sent
    assert "cas://" not in sent
    assert artifact_id in sent  # it does see the identifier


# --------------------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------------------


def test_invalid_arguments_are_repaired_on_the_second_attempt(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [
            tool_call("echo_artifact", artifact_id="not-an-id"),
            tool_call("echo_artifact", artifact_id=artifact_id),
            *[tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(5)],
        ]
    )
    result = Agent(llm, ctx, settings).run()

    assert result.final_state is State.DONE
    assert result.metrics.n_slots_repaired == 1
    assert result.metrics.n_repairs_succeeded == 1
    assert result.metrics.as_dict()["repair_success_rate"] == 1.0
    assert result.metrics.n_llm_calls == HAPPY_PATH_CALLS + 1


def test_the_repair_message_names_the_offending_field(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    # The model can only fix what it is told about; a bare "invalid" is not actionable.
    llm = FakeLLMClient(
        [
            tool_call("echo_artifact", artifact_id="not-an-id"),
            *[tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(6)],
        ]
    )
    Agent(llm, ctx, settings).run()

    second_prompt = repr(llm.calls[1]["messages"])
    assert "artifact_id" in second_prompt
    assert "rejected" in second_prompt


def test_three_invalid_calls_fail_the_run_with_validation_failed(
    ctx: RunContext, settings: Settings
) -> None:
    # Two repairs allowed by default, so the third bad call exhausts the budget.
    assert settings.llm_max_repair_attempts == 2
    llm = FakeLLMClient([tool_call("echo_artifact", artifact_id="not-an-id") for _ in range(3)])
    result = Agent(llm, ctx, settings).run()

    assert result.final_state is State.FAILED
    assert result.error is not None
    assert result.error.error_code is ErrorCode.VALIDATION_FAILED
    assert result.metrics.n_llm_calls == 3
    assert result.metrics.n_repairs_succeeded == 0
    assert result.metrics.as_dict()["repair_success_rate"] == 0.0
    assert ctx.manifest().status is RunStatus.ERROR


def test_a_response_with_no_tool_call_is_repaired(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [prose(), *[tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(6)]]
    )
    result = Agent(llm, ctx, settings).run()
    assert result.final_state is State.DONE
    assert result.metrics.n_slots_repaired == 1


def test_a_refusal_is_treated_as_a_malformed_turn(ctx: RunContext, settings: Settings) -> None:
    # A refusal arrives as a perfectly ordinary success, so it has to be caught here or
    # it reads as an empty answer.
    llm = FakeLLMClient([refusal() for _ in range(3)])
    result = Agent(llm, ctx, settings).run()
    assert result.final_state is State.FAILED
    assert result.error is not None
    assert "declined" in result.error.message


# --------------------------------------------------------------------------------------
# The per-state whitelist
# --------------------------------------------------------------------------------------


def test_an_out_of_state_tool_call_is_rejected_and_recorded(
    ctx: RunContext, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # echo_artifact is a real, registered tool -- it is simply not offered in this state.
    # Naming it anyway must not run it.
    monkeypatch.setattr(fsm, "STATE_TOOLS", only(INTAKE=frozenset({"request_review"})))
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=MISSING_ARTIFACT) for _ in range(3)]
    )
    result = Agent(llm, ctx, settings).run()

    assert result.final_state is State.FAILED
    assert result.metrics.n_out_of_state_calls == 3
    assert result.metrics.n_tool_invocations == 0  # never reached invoke()
    assert result.error is not None
    assert "not available in state INTAKE" in result.error.message

    outcomes = [r["outcome"] for r in read_trace(ctx.run_dir / TRACE_FILENAME) if "outcome" in r]
    assert outcomes == ["out_of_state"] * 3


def test_the_model_is_only_shown_the_tools_legal_in_the_state(
    ctx: RunContext, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsm, "STATE_TOOLS", only(INTAKE=frozenset({"request_review"})))
    llm = FakeLLMClient([tool_call("request_review", reason="unclear")])
    Agent(llm, ctx, settings).run()

    offered = llm.calls[0]["tools"]
    assert isinstance(offered, list)
    assert [entry["name"] for entry in offered] == ["request_review"]  # type: ignore[index]


def test_an_out_of_state_call_can_still_be_repaired(
    ctx: RunContext, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsm, "STATE_TOOLS", only(INTAKE=frozenset({"request_review"})))
    llm = FakeLLMClient(
        [
            tool_call("echo_artifact", artifact_id=MISSING_ARTIFACT),
            tool_call("request_review", reason="the assay is ambiguous"),
        ]
    )
    result = Agent(llm, ctx, settings).run()
    assert result.final_state is State.NEEDS_REVIEW
    assert result.metrics.n_out_of_state_calls == 1


# --------------------------------------------------------------------------------------
# Escapes
# --------------------------------------------------------------------------------------


def test_request_review_ends_the_run_in_needs_review(ctx: RunContext, settings: Settings) -> None:
    llm = FakeLLMClient([tool_call("request_review", reason="two assays fit the metadata")])
    result = Agent(llm, ctx, settings).run()

    assert result.final_state is State.NEEDS_REVIEW
    assert result.error is None  # refusing to proceed is not a failure
    assert result.tool_calls == ("request_review",)
    assert ctx.manifest().status is RunStatus.NEEDS_REVIEW


def test_handoff_ends_the_run_in_handoff(ctx: RunContext, settings: Settings) -> None:
    llm = FakeLLMClient([tool_call("handoff", reason="needs wet-lab work", to="the core facility")])
    result = Agent(llm, ctx, settings).run()

    assert result.final_state is State.HANDOFF
    assert result.error is None
    assert ctx.manifest().status is RunStatus.NEEDS_REVIEW


def test_an_escape_is_available_from_a_late_state(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [
            *[tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(4)],
            tool_call("request_review", reason="the results contradict each other"),
        ]
    )
    result = Agent(llm, ctx, settings).run()
    assert result.final_state is State.NEEDS_REVIEW
    assert result.states[-2] is State.INTERPRET


# --------------------------------------------------------------------------------------
# Caps
# --------------------------------------------------------------------------------------


def test_the_step_cap_fails_the_run(ctx: RunContext, settings: Settings, artifact_id: str) -> None:
    llm = FakeLLMClient([tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(10)])
    result = Agent(llm, ctx, settings, limits=AgentLimits(max_steps=2)).run()

    assert result.final_state is State.FAILED
    assert result.error is not None
    assert result.error.error_code is ErrorCode.RESOURCE_LIMIT
    assert result.error.details["limit_name"] == "max_steps"
    assert result.error.details["limit"] == 2
    assert "max_steps=2" in result.error.message


def test_the_per_state_tool_call_cap_fails_the_run(ctx: RunContext, settings: Settings) -> None:
    # A well-formed call to a tool that keeps failing: repairs never fire, so the only
    # thing that stops the state is the per-state cap.
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=MISSING_ARTIFACT) for _ in range(9)]
    )
    result = Agent(llm, ctx, settings, limits=AgentLimits(max_tool_calls_per_state=3)).run()

    assert result.final_state is State.FAILED
    assert result.error is not None
    assert result.error.error_code is ErrorCode.RESOURCE_LIMIT
    assert result.error.details["limit_name"] == "max_tool_calls_per_state"
    # The tool failure that consumed the budget is the cause, kept alongside the outcome.
    assert "no artifact" in str(result.error.details["last_tool_error"])
    assert result.metrics.n_tool_invocations == 3


def test_the_wall_clock_cap_fails_the_run(ctx: RunContext, settings: Settings) -> None:
    llm = FakeLLMClient([tool_call("request_review", reason="x")])
    result = Agent(llm, ctx, settings, limits=AgentLimits(max_wall_clock_seconds=-1.0)).run()

    assert result.final_state is State.FAILED
    assert result.error is not None
    assert result.error.details["limit_name"] == "max_wall_clock_seconds"
    assert result.metrics.n_llm_calls == 0  # refused before asking anything


def test_limits_come_from_settings_by_default() -> None:
    configured = Settings(
        agent_max_steps=7,
        agent_max_tool_calls_per_state=3,
        max_runtime_minutes=5,
        llm_max_repair_attempts=1,
    )
    limits = AgentLimits.from_settings(configured)
    assert limits.max_steps == 7
    assert limits.max_tool_calls_per_state == 3
    # The run's time budget is the agent's time budget; there is no separate setting.
    assert limits.max_wall_clock_seconds == 300.0
    assert limits.max_repair_attempts == 1


# --------------------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------------------


class GatedIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str


class GatedOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> Registry:
    """A registry holding one expensive, confirmation-requiring tool."""
    isolated = Registry()

    @tool(
        name="run_pipeline",
        version="1.0.0",
        summary="Run an expensive pipeline.",
        side_effect=SideEffect.EXECUTE,
        cost_class=CostClass.EXPENSIVE,
        requires_confirmation=True,
        into=isolated,
    )
    def run_pipeline(params: GatedIn, ctx: RunContext) -> GatedOut:
        return GatedOut(note=params.note)

    monkeypatch.setattr(fsm, "STATE_TOOLS", only(PLAN=frozenset({"run_pipeline"})))
    return isolated


def test_a_confirmation_requiring_tool_pauses_at_await_confirm(
    ctx: RunContext, settings: Settings, gated: Registry
) -> None:
    llm = FakeLLMClient([tool_call("run_pipeline", note="align the reads")])
    result = Agent(
        llm, ctx, settings, tools=gated, confirm=DenyAllPolicy(), initial_state=State.PLAN
    ).run()

    assert State.AWAIT_CONFIRM in result.states
    # Declining is a correct answer, not a failure.
    assert result.final_state is State.NEEDS_REVIEW
    assert result.error is None
    assert result.tool_calls == ()
    assert result.metrics.n_tool_invocations == 0


def test_an_approved_call_runs_and_the_pipeline_continues(
    ctx: RunContext, settings: Settings, gated: Registry
) -> None:
    llm = FakeLLMClient([tool_call("run_pipeline", note="align the reads")])
    result = Agent(
        llm,
        ctx,
        settings,
        tools=gated,
        confirm=AutoApproveBelow(CostClass.EXPENSIVE),
        initial_state=State.PLAN,
    ).run()

    assert result.final_state is State.DONE
    assert result.tool_calls == ("run_pipeline",)
    assert result.states[:3] == (State.PLAN, State.AWAIT_CONFIRM, State.EXECUTE)


def test_a_threshold_below_the_tool_declines(
    ctx: RunContext, settings: Settings, gated: Registry
) -> None:
    llm = FakeLLMClient([tool_call("run_pipeline", note="align the reads")])
    result = Agent(
        llm,
        ctx,
        settings,
        tools=gated,
        confirm=AutoApproveBelow(CostClass.MODERATE),
        initial_state=State.PLAN,
    ).run()
    assert result.final_state is State.NEEDS_REVIEW


def test_the_default_policy_declines(ctx: RunContext, settings: Settings, gated: Registry) -> None:
    # No policy passed at all: an unattended run must not approve its own spending.
    llm = FakeLLMClient([tool_call("run_pipeline", note="align the reads")])
    result = Agent(llm, ctx, settings, tools=gated, initial_state=State.PLAN).run()
    assert result.final_state is State.NEEDS_REVIEW


def test_a_gate_the_graph_cannot_reach_is_counted_as_an_illegal_transition(
    ctx: RunContext, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing that makes ``illegal_transition_attempts`` a real number.

    With the shipped whitelist the counter stays at zero: every working state can reach
    every escape, so the model cannot ask for a move the graph forbids. It becomes
    non-zero when the whitelist and the graph disagree -- here, a confirmation-requiring
    tool offered in a state with no edge to ``AWAIT_CONFIRM``. That is a configuration
    bug, and this is how it surfaces in the manifest instead of as a traceback.
    """
    isolated = Registry()

    @tool(
        name="run_pipeline",
        version="1.0.0",
        summary="Run an expensive pipeline.",
        side_effect=SideEffect.EXECUTE,
        cost_class=CostClass.CHEAP,
        requires_confirmation=True,
        into=isolated,
    )
    def run_pipeline(params: GatedIn, ctx: RunContext) -> GatedOut:  # pragma: no cover
        return GatedOut(note=params.note)

    assert not fsm.is_legal(State.INTAKE, State.AWAIT_CONFIRM)
    monkeypatch.setattr(fsm, "STATE_TOOLS", only(INTAKE=frozenset({"run_pipeline"})))

    llm = FakeLLMClient([tool_call("run_pipeline", note="go")])
    result = Agent(
        llm, ctx, settings, tools=isolated, confirm=AutoApproveBelow(CostClass.EXPENSIVE)
    ).run()

    assert result.final_state is State.FAILED
    assert result.metrics.illegal_transition_attempts == 1
    assert ctx.manifest().metrics["illegal_transition_attempts"] == 1
    assert result.error is not None
    assert "illegal state transition INTAKE -> AWAIT_CONFIRM" in result.error.message
    assert result.metrics.n_tool_invocations == 0  # the gated tool never ran


def test_the_confirmation_decision_is_traced(
    ctx: RunContext, settings: Settings, gated: Registry
) -> None:
    llm = FakeLLMClient([tool_call("run_pipeline", note="align the reads")])
    Agent(
        llm,
        ctx,
        settings,
        tools=gated,
        confirm=AutoApproveBelow(CostClass.EXPENSIVE),
        initial_state=State.PLAN,
    ).run()

    records = read_trace(ctx.run_dir / TRACE_FILENAME)
    gate = [r for r in records if r.get("state") == "AWAIT_CONFIRM"]
    assert len(gate) == 1
    assert gate[0]["confirmed"] is True
    assert gate[0]["outcome"] == "confirmed"


# --------------------------------------------------------------------------------------
# The trace
# --------------------------------------------------------------------------------------


def test_the_trace_opens_with_a_header_and_closes_with_a_footer(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    result = Agent(llm, ctx, settings, cassette="demo").run()

    records = read_trace(ctx.run_dir / TRACE_FILENAME)
    assert records[0]["kind"] == "header"
    assert records[0]["cassette"] == "demo"
    assert records[0]["request"] == {"ask": "triage this dataset"}
    assert records[-1]["kind"] == "footer"
    assert records[-1]["final_state"] == "DONE"
    assert records[-1]["tool_calls"] == list(result.tool_calls)


def test_each_call_record_carries_the_state_prompt_and_cost(
    ctx: RunContext, settings: Settings, artifact_id: str
) -> None:
    llm = FakeLLMClient(
        [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
    )
    Agent(llm, ctx, settings).run()

    calls = [r for r in read_trace(ctx.run_dir / TRACE_FILENAME) if r["kind"] == "call"]
    assert [r["state"] for r in calls] == [
        "INTAKE",
        "TRIAGE",
        "PLAN",
        "PARSE",
        "INTERPRET",
        "REPORT",
    ]
    first = calls[0]
    assert first["prompt_version"] == "v1"
    assert len(str(first["messages_sha256"])) == 64
    assert first["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": 0.0,
        "cost_is_estimated": False,
    }
    assert first["tool_call"] == {
        "name": "echo_artifact",
        "arguments": {"artifact_id": artifact_id},
        "params_sha256": first["tool_call"]["params_sha256"],  # type: ignore[index]
    }


def test_a_rejected_call_records_why(ctx: RunContext, settings: Settings) -> None:
    llm = FakeLLMClient([tool_call("echo_artifact", artifact_id="not-an-id") for _ in range(3)])
    Agent(llm, ctx, settings).run()

    calls = [r for r in read_trace(ctx.run_dir / TRACE_FILENAME) if r["kind"] == "call"]
    assert [r["repair_attempt"] for r in calls] == [0, 1, 2]
    assert all(r["outcome"] == "invalid_arguments" for r in calls)
    errors = calls[0]["validation_errors"]
    assert isinstance(errors, list)
    assert errors[0]["error_code"] == "VALIDATION_FAILED"  # type: ignore[index]


# --------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------


def test_a_trace_round_trips_through_replay(
    ctx: RunContext, settings: Settings, artifact_id: str, tmp_path: Path
) -> None:
    cassettes = tmp_path / "cassettes"
    recorder = CassetteClient(
        FakeLLMClient(
            [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
        ),
        name="agent_happy",
        mode="record",
        root=cassettes,
    )
    original = Agent(recorder, ctx, settings, cassette="agent_happy").run()
    trace_path = ctx.run_dir / TRACE_FILENAME
    assert original.final_state is State.DONE

    result = replay(trace_path, settings, cassette_root=cassettes)

    assert result.ok, result.diff()
    assert result.observed == original.tool_calls
    assert result.final_state is State.DONE
    assert "replay matched" in result.diff()
    result.raise_for_mismatch()  # must not raise


def test_a_replay_is_itself_a_recorded_run(
    ctx: RunContext, settings: Settings, artifact_id: str, tmp_path: Path
) -> None:
    cassettes = tmp_path / "cassettes"
    recorder = CassetteClient(
        FakeLLMClient(
            [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
        ),
        name="agent_happy",
        mode="record",
        root=cassettes,
    )
    Agent(recorder, ctx, settings, cassette="agent_happy").run()
    original_run_id = ctx.run_id

    result = replay(ctx.run_dir / TRACE_FILENAME, settings, cassette_root=cassettes)

    replayed = ctx.runstore.get_run(result.run_id)
    assert replayed is not None
    assert replayed.request["replay_of"] == original_run_id
    # Seeded with the original's artifacts, or every tool call would miss.
    assert [r.artifact_id for r in ctx.runstore.list_artifacts(result.run_id)] == [artifact_id]


def test_replay_reports_a_divergence_rather_than_hiding_it(
    ctx: RunContext, settings: Settings, artifact_id: str, tmp_path: Path
) -> None:
    cassettes = tmp_path / "cassettes"
    recorder = CassetteClient(
        FakeLLMClient(
            [tool_call("echo_artifact", artifact_id=artifact_id) for _ in range(HAPPY_PATH_CALLS)]
        ),
        name="agent_happy",
        mode="record",
        root=cassettes,
    )
    Agent(recorder, ctx, settings, cassette="agent_happy").run()

    # Delete a recording: the replay now misses partway through, which is exactly the
    # shape of a prompt change that alters behaviour.
    recordings = sorted((cassettes / "agent_happy").glob("*.json"))
    recordings[-1].unlink()

    result = replay(ctx.run_dir / TRACE_FILENAME, settings, cassette_root=cassettes)
    assert not result.ok
    assert result.first_divergence is not None
    assert "diverged" in result.diff()
    with pytest.raises(Exception, match="did not reproduce"):
        result.raise_for_mismatch()


def test_a_failing_llm_client_fails_the_run_rather_than_escaping(
    ctx: RunContext, settings: Settings, tmp_path: Path
) -> None:
    # A cassette miss stands in for any client-level failure. The loop must end in
    # FAILED with a structured error, the same discipline invoke() follows -- a
    # traceback out of run() would leave the manifest describing a run that never ended.
    client = CassetteClient(None, name="empty", mode="replay", root=tmp_path / "cassettes")
    result = Agent(client, ctx, settings).run()

    assert result.final_state is State.FAILED
    assert result.error is not None
    assert result.error.error_code is ErrorCode.NOT_FOUND
    assert "cassette miss" in result.error.message
    # Not repaired: retrying a miss just misses again.
    assert result.metrics.n_slots_repaired == 0
    outcomes = [r["outcome"] for r in read_trace(ctx.run_dir / TRACE_FILENAME) if "outcome" in r]
    assert outcomes == ["llm_error"]
    assert ctx.manifest().status is RunStatus.ERROR


def test_a_trace_without_a_cassette_cannot_be_replayed(
    ctx: RunContext, settings: Settings, tmp_path: Path
) -> None:
    llm = FakeLLMClient([tool_call("request_review", reason="unclear")])
    Agent(llm, ctx, settings).run()  # no cassette name

    with pytest.raises(Exception, match="names no cassette"):
        replay(ctx.run_dir / TRACE_FILENAME, settings, cassette_root=tmp_path)
