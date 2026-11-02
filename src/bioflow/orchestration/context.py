"""What the model is allowed to see, and how much of it.

Before this module the agent appended to a message list that only grew: a state prompt, a
tool result, a repair note, another state prompt, for as many steps as the run took. The
first thing that breaks is not cost. It is that the instruction which mattered ends up
buried in the middle of a transcript, where models attend to it least.

The shape here is a **bounded window**. The last few turns stay verbatim, because recency
is where detail earns its cost. Everything older folds into one line per turn, so the
run's history stays present without staying expensive.

The budget is enforced by construction rather than hoped for. A floor -- the system
prompt, the state's objective, the tool schemas, the request, and any correction the model
must read to fix its last call -- is never truncated; if the floor alone exceeds the
budget, that is a `RESOURCE_LIMIT` failure naming the parts, not a request sent hopefully
at a provider. What remains is divided among the three elastic sections, each of which
truncates itself with an explicit marker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from bioflow.core.models import (
    ArtifactRef,
    BioflowError,
    ErrorCode,
    JSONValue,
    StructuredError,
    ToolSpec,
)
from bioflow.orchestration.fsm import State, allowed_tools
from bioflow.orchestration.summarize import StoreSource, redact_paths, summarize
from bioflow.orchestration.tokens import count_tokens, fit, truncation_marker
from bioflow.prompts.loader import DEFAULT_VERSION, load_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioflow.core.registry import Registry
    from bioflow.core.store import ArtifactStore

SYSTEM_PROMPT = "agent_system"
STATE_PROMPT = "agent_state"
CONTEXT_PROMPT = "agent_context"

DEFAULT_TOKEN_BUDGET = 12000
DEFAULT_WINDOW_TURNS = 4

# How the elastic remainder is divided. Artifacts first: without a valid identifier and
# some idea of an artifact's shape, the model cannot produce a usable call at all, which
# makes the other two sections moot.
ARTIFACT_SHARE = 0.45
WINDOW_SHARE = 0.35
SUMMARY_SHARE = 0.20

# Enough for one dense line of description in an index row.
ROW_SUMMARY_TOKENS = 32
# How many of the most recent artifacts get a full description rather than a row.
DETAILED_ARTIFACTS = 3
# How much of a turn's result survives into its one-line history entry. Without a cap
# the rolling summary grows with every step, which is the thing this module exists to
# stop -- a compacted turn has to actually be smaller than the turn.
COMPACT_DETAIL_CHARS = 80

_RAISED_BY = "bioflow.orchestration.context"


@dataclass(frozen=True)
class Turn:
    """One completed exchange: what the model called and what came back."""

    state: str
    tool: str | None
    arguments: dict[str, JSONValue] = field(default_factory=dict)
    outcome: str = "ok"
    detail: str = ""

    def call_text(self) -> str:
        """How the model's side of the exchange is replayed to it."""
        if self.tool is None:
            return f"[{self.state}] I did not call a tool."
        return f"[{self.state}] I called {self.tool} with {json.dumps(self.arguments, default=str)}"

    def result_text(self) -> str:
        """How the outcome is replayed to the model."""
        return f"{self.outcome}: {self.detail}" if self.detail else self.outcome

    def compact(self) -> str:
        """One line, for a turn that has aged out of the verbatim window.

        The detail is clipped to a fixed width with an ellipsis. This is the one place a
        cut is marked by "…" rather than a full ``[truncated: ...]`` marker: a compacted
        line is already an abbreviation of the turn, and stamping every history line with
        a marker would bury the section it is meant to clarify. The section as a whole
        still carries a marker when whole lines are dropped.
        """
        name = self.tool or "(no tool)"
        clipped = self.detail
        if len(clipped) > COMPACT_DETAIL_CHARS:
            clipped = clipped[:COMPACT_DETAIL_CHARS].rstrip() + "…"
        detail = f" — {clipped}" if clipped else ""
        return f"{self.state}: {name} -> {self.outcome}{detail}"


@dataclass(frozen=True)
class BuiltContext:
    """One assembled request, with the accounting that produced it."""

    system: str
    messages: list[dict[str, JSONValue]]
    tools: list[dict[str, JSONValue]]
    tokens: int
    floor_tokens: int
    budget: int
    sections: dict[str, int]

    def text(self) -> str:
        """Everything the model will read, for measuring and for tests."""
        return self.system + "\n" + "\n".join(str(m["content"]) for m in self.messages)


class ArtifactIndex:
    """Renders a run's artifacts as identifiers, never as locations.

    The table is the spec's contract -- ``artifact_id | type | one-line summary | size``
    -- and the most recent few artifacts additionally get a full description, because
    those are the ones the current state is actually about.

    Summaries are cached on ``(artifact_id, budget)``. Artifacts are immutable and
    content-addressed, so re-reading one on every turn of a twenty-step run would be pure
    waste.
    """

    def __init__(self, store: ArtifactStore | None) -> None:
        self.store = store
        self._cache: dict[tuple[str, int], str] = {}

    def summary_for(self, ref: ArtifactRef, budget_tokens: int) -> str:
        """Summarize one artifact, memoized."""
        key = (ref.artifact_id, budget_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.store is None:
            rendered = f"{ref.artifact_type.value} artifact"
        else:
            rendered = summarize(ref, StoreSource(self.store, ref), budget_tokens)
        self._cache[key] = rendered
        return rendered

    def render(self, refs: Sequence[ArtifactRef], budget_tokens: int) -> str:
        """Render the index within a budget, truncating rows before descriptions."""
        if not refs:
            return "No artifacts in this run yet."
        if budget_tokens <= 0:
            return truncation_marker(0, len(refs), "artifacts")

        rows = ["artifact_id | type | summary | size"]
        for ref in refs:
            one_line = self.summary_for(ref, ROW_SUMMARY_TOKENS).splitlines()[0]
            size = "unknown" if ref.size_bytes is None else f"{ref.size_bytes} B"
            rows.append(f"{ref.artifact_id} | {ref.artifact_type.value} | {one_line} | {size}")

        table = fit("\n".join(rows), budget_tokens, "artifacts")
        spare = budget_tokens - count_tokens(table)
        if spare <= 0 or len(refs) == 0:
            return table

        # Whatever the table did not spend goes to describing the newest artifacts.
        recent = list(refs)[-DETAILED_ARTIFACTS:]
        each = max(spare // len(recent), 0)
        if each < ROW_SUMMARY_TOKENS:
            return table

        blocks = [f"### {ref.artifact_id}\n{self.summary_for(ref, each)}" for ref in recent]
        return table + "\n\n" + fit("\n\n".join(blocks), spare, "artifact detail")


class ContextBuilder:
    """Assembles the prompt for one LLM call, within a token budget."""

    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        *,
        store: ArtifactStore | None = None,
        registry: Registry | None = None,
        request: dict[str, JSONValue] | None = None,
        window_turns: int = DEFAULT_WINDOW_TURNS,
        prompt_version: str = DEFAULT_VERSION,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget
        self.window_turns = window_turns
        self.prompt_version = prompt_version
        self.request = dict(request or {})
        self.index = ArtifactIndex(store)
        self.turns: list[Turn] = []

        from bioflow.core.registry import registry as default_registry

        # An empty Registry is falsy, so this must test against None.
        self.registry = registry if registry is not None else default_registry
        self.system = load_prompt(SYSTEM_PROMPT, prompt_version).text

    # -- history ----------------------------------------------------------------------

    def record_turn(self, turn: Turn) -> None:
        """Add a completed exchange to the history."""
        self.turns.append(turn)

    @property
    def window(self) -> list[Turn]:
        """The turns still shown verbatim."""
        return self.turns[-self.window_turns :] if self.window_turns > 0 else []

    @property
    def older(self) -> list[Turn]:
        """The turns that have aged into the rolling summary."""
        return self.turns[: -self.window_turns] if self.window_turns > 0 else list(self.turns)

    # -- tools ------------------------------------------------------------------------

    def specs_for(self, state: State) -> list[ToolSpec]:
        """The tool specs legal in a state, in registry order."""
        allowed = allowed_tools(state)
        return [spec for spec in self.registry.list_specs() if spec.name in allowed]

    def schemas_for(self, state: State) -> list[dict[str, JSONValue]]:
        """The neutral tool schemas the model may see in a state."""
        allowed = allowed_tools(state)
        return [entry for entry in self.registry.to_llm_schema() if entry.get("name") in allowed]

    # -- assembly ---------------------------------------------------------------------

    def build(
        self,
        state: State,
        *,
        artifacts: Sequence[ArtifactRef] = (),
        correction: str | None = None,
    ) -> BuiltContext:
        """Assemble the messages for one call, or explain why the budget cannot hold them."""
        objective = load_prompt(STATE_PROMPT, self.prompt_version, state=state.value).text
        specs = self.specs_for(state)
        schemas = self.schemas_for(state)

        # Redacted like summarizer output. The request is the other channel through
        # which a path can reach the model -- a caller passing {"input": "/data/x.fastq"}
        # would otherwise walk straight past invariant #2. The correct way to hand the
        # agent a file is to ingest it and refer to the artifact by identifier.
        request_text = (
            redact_paths(json.dumps(self.request, indent=2, default=str))
            if self.request
            else "(none)"
        )
        tools_text = (
            "\n".join(f"- `{spec.name}` — {spec.summary}" for spec in specs)
            or "No tools are available in this state."
        )

        # The floor is *measured*, not estimated: the block is rendered once with its
        # elastic sections empty. Estimating it from the objective and request alone
        # missed the template's own headings and standing instructions, and the sections
        # then spent a budget that was never really there.
        skeleton = self._render_block(state, objective, request_text, tools_text, "", "")
        floor_tokens = (
            count_tokens(self.system)
            + count_tokens(skeleton)
            + _schema_tokens(schemas)
            + count_tokens(correction or "")
        )
        remaining = self.token_budget - floor_tokens
        if remaining < 0:
            raise self._over_budget(floor_tokens, objective, request_text, schemas)

        # Spill forward: each section takes its share of what is still unspent, so an
        # artifact index that came in under allocation funds a longer history.
        artifact_budget = int(remaining * ARTIFACT_SHARE)
        artifacts_text = self.index.render(artifacts, artifact_budget)
        remaining -= count_tokens(artifacts_text)

        window_share = WINDOW_SHARE / (WINDOW_SHARE + SUMMARY_SHARE)
        window_budget = max(int(remaining * window_share), 0)
        window_messages, window_tokens, shown = self._window_messages(window_budget)
        remaining -= window_tokens

        # Everything not shown verbatim is summarized, including turns the window was
        # too small to afford. Nothing falls between the two.
        summary_text = self._rolling_summary(max(remaining, 0), shown)

        block = self._render_block(
            state, objective, request_text, tools_text, artifacts_text, summary_text
        )

        messages: list[dict[str, JSONValue]] = [{"role": "user", "content": block}]
        messages.extend(window_messages)
        if correction:
            messages.append({"role": "user", "content": correction})

        built = BuiltContext(
            system=self.system,
            messages=messages,
            tools=schemas,
            tokens=0,
            floor_tokens=floor_tokens,
            budget=self.token_budget,
            # Disjoint by construction, so they sum to `tokens`. Reporting the whole
            # rendered block here instead would double-count the elastic sections it
            # contains and make the accounting useless for finding what overflowed.
            sections={
                "system": count_tokens(self.system),
                "skeleton": count_tokens(skeleton),
                "tools": _schema_tokens(schemas),
                "correction": count_tokens(correction or ""),
                "artifacts": count_tokens(artifacts_text),
                "window": window_tokens,
                "summary": count_tokens(summary_text),
            },
        )
        # tokens is derived from the assembled text, so it is filled in once here rather
        # than recomputed by every caller. Schemas count too: they go over the wire.
        return replace(built, tokens=count_tokens(built.text()) + _schema_tokens(schemas))

    # -- sections ---------------------------------------------------------------------

    def _render_block(
        self,
        state: State,
        objective: str,
        request_text: str,
        tools_text: str,
        artifacts_text: str,
        history_text: str,
    ) -> str:
        """Render the assembled block. Called twice: once empty to measure the floor."""
        return load_prompt(
            CONTEXT_PROMPT,
            self.prompt_version,
            state=state.value,
            objective=objective,
            request=request_text,
            tools=tools_text,
            artifacts=artifacts_text,
            history=history_text,
        ).text

    def _window_messages(self, budget: int) -> tuple[list[dict[str, JSONValue]], int, int]:
        """Render the verbatim window, shedding oldest turns until it fits.

        Returns the messages, their cost, and how many turns they cover -- the caller
        needs the count so the rolling summary can pick up exactly where this left off.
        """
        turns = self.window
        while turns:
            messages: list[dict[str, JSONValue]] = []
            for turn in turns:
                messages.append({"role": "assistant", "content": turn.call_text()})
                messages.append({"role": "user", "content": turn.result_text()})
            used = sum(count_tokens(str(m["content"])) for m in messages)
            if used <= budget:
                return messages, used, len(turns)
            # An over-budget window sheds its oldest turn, which then shows up in the
            # rolling summary instead of vanishing.
            turns = turns[1:]
        return [], 0, 0

    def _rolling_summary(self, budget: int, shown_verbatim: int) -> str:
        """One line per turn not shown verbatim, truncated with a marker when it will not fit."""
        older = self.turns[: len(self.turns) - shown_verbatim] if shown_verbatim else self.turns
        if not older:
            return "Nothing has happened yet." if not self.turns else "(all steps shown below)"
        return fit("\n".join(turn.compact() for turn in older), budget, "earlier steps")

    def _over_budget(
        self,
        floor_tokens: int,
        objective: str,
        request_text: str,
        schemas: Sequence[dict[str, JSONValue]],
    ) -> BioflowError:
        """Explain which fixed part made the budget unreachable."""
        return BioflowError(
            StructuredError(
                error_code=ErrorCode.RESOURCE_LIMIT,
                message=(
                    f"context floor is {floor_tokens} tokens but the budget is {self.token_budget}"
                ),
                remediation=(
                    "Raise BIOFLOW_AGENT_TOKEN_BUDGET, or narrow the state's tool "
                    "whitelist so fewer schemas are sent."
                ),
                details={
                    "floor_tokens": floor_tokens,
                    "budget": self.token_budget,
                    "system": count_tokens(self.system),
                    "objective": count_tokens(objective),
                    "request": count_tokens(request_text),
                    "tool_schemas": _schema_tokens(schemas),
                    "n_tools": len(schemas),
                },
                raised_by=_RAISED_BY,
            )
        )


def _schema_tokens(schemas: Sequence[dict[str, JSONValue]]) -> int:
    """Estimate what the tool schemas cost once serialized for a provider."""
    return count_tokens(json.dumps(list(schemas), default=str))
