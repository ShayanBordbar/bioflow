# ADR-0002: Code advances the agent's state; the model may only stop

## Status
Accepted — 2026-08-28.

## Context

Step 1.6 built the orchestration loop: the code that decides what to ask the model next.
The step's specification declared thirteen states, an explicit legal-transition set, a
per-state tool whitelist, and hard caps — but it did not say *who chooses the next state*.
That gap is the whole design.

The risk being managed is specific and well documented. A language model given an
open-ended loop and a set of tools will, over enough steps, select an action nobody
intended: an expensive pipeline during triage, a repair cycle on an argument it cannot
fix, or an unbounded sequence of calls. BioFlow's premise is that the LLM performs
judgment while all computation stays deterministic and reproducible without it. A loop
where the model picks its own control flow quietly moves control flow into the
non-reproducible half of that split.

Three arrangements were considered.

**Pure code-driven.** A deterministic `next_state(state, outcome)`; the model's only lever
is which whitelisted tool it calls. Simplest and most defensible. Its problem is that
invariant #10 — "ambiguous input produces NEEDS_REVIEW; refusing to run is a correct
answer" — becomes something code has to *infer* on the model's behalf. The system would
have to guess that the model is uncertain, which is exactly the inference it is worst at.

**Fully LLM-driven.** An `advance_to(state)` tool; the model names its next state every
step and the machine rejects illegal targets. Maximum flexibility, and it makes the
transition counters meaningful, but it puts the happy path in the model's hands. That
directly contradicts the project's central claim.

**Hybrid.** Code advances the happy path; the model may only *escape*.

## Decision

Transitions along the forward path are chosen by code, in `_SUCCESSOR`
(`orchestration/agent.py`). The model never names a state.

The model's only influence over control flow is two registered tools —
`request_review` and `handoff` — whose target states are fixed in `CONTROL_TOOL_TARGETS`
(`orchestration/fsm.py`). It requests *leaving*, never *going somewhere specific*, and
even the two destinations it can request are validated against the transition set like
any other move.

These are ordinary registered tools, not a side channel. Their arguments are validated by
`registry.validate_args`, their calls are recorded as provenance by `registry.invoke`, and
they carry golden fixtures like every other tool. An escape hatch that bypassed the
enforcement path would not be an escape hatch; it would be a hole.

Two supporting decisions follow from this one:

- **`AWAIT_CONFIRM` and `EXECUTE` offer the model nothing.** The confirmation decision
  belongs to an injected `ConfirmationPolicy`, and plan execution is deterministic code.
  Offering the model a tool in either state would give it a say in something it must not
  have a say in.
- **Argument validation is split out of `invoke()`.** `validate_args` runs first, so the
  repair loop is unambiguously about arguments the model can fix. Branching on the error
  code returned by `invoke()` would confuse a malformed call with a tool that legitimately
  raised `VALIDATION_FAILED` from inside its own body, and retrying the second kind can
  never succeed.

## Consequences

**Easy.** The happy path is deterministic and inspectable: reading `_SUCCESSOR` tells you
exactly what a run will do. `replay()` can assert an identical tool-call sequence because
there is no model-chosen control flow to diverge. Invariant #10 is reachable from inside
the model's judgment rather than by inference.

**Hard.** Adding a state means editing three places — `State`, `_FORWARD`, and
`_SUCCESSOR` — and the whitelist besides. This is deliberate friction on a security
boundary, but it is friction.

**Ruled out.** Dynamic replanning, where the model decides mid-run to revisit an earlier
state. `DIAGNOSE -> PLAN` is the one loop in the graph, and code chooses it.

**A correction to the reasoning that led here.** The hybrid arrangement was argued for
partly on the grounds that it gives `illegal_transition_attempts` a real source. Building
it showed that argument is weaker than it looked: because every working state can reach
`NEEDS_REVIEW`, `HANDOFF`, and `FAILED` in one move, a model calling a control tool can
never actually request an illegal transition. With the shipped whitelist the counter
reads zero.

It is not dead, and it was kept rather than removed. Its real source is a *disagreement
between the whitelist and the graph* — a confirmation-requiring tool offered in a state
with no edge to `AWAIT_CONFIRM` produces exactly one illegal attempt, and the run fails
with the count in its manifest instead of a traceback. That is a configuration bug rather
than model misbehaviour, which is a less interesting thing to count but still worth
seeing. `tests/unit/test_fsm.py` guards against it statically; the counter catches it if
the guard is ever weakened.

## Evidence

- `tests/unit/test_fsm.py` tests the transition set as a graph: every state reachable from
  `INTAKE`, every working state one move from each escape, terminal closure, no dead ends,
  and every whitelisted tool name actually registered. Verified to fail when a whitelist
  entry is misspelled.
- `tests/unit/test_agent_loop.py::test_a_gate_the_graph_cannot_reach_is_counted_as_an_illegal_transition`
  demonstrates the one path that makes `illegal_transition_attempts` non-zero.
- `tests/contract/test_all_tools_have_goldens.py` verified to fail when the control tools'
  goldens are removed, confirming they are on the enforcement path rather than beside it.
- Replay round-trip verified end to end: a recorded run reproduces its six-call sequence
  exactly, and deleting one cassette recording produces a reported divergence
  (`ok=False`, first difference at index 0, final state `FAILED` vs `DONE`) rather than a
  crash.
