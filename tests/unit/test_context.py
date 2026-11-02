"""Unit tests for context assembly.

The budget is the point. A ``ContextBuilder`` that assembles a good prompt but does not
respect its allocation is the same as no builder at all, so most of what follows measures
rather than inspects.
"""

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

import bioflow.tools  # noqa: F401  (registers the built-in tools)
from bioflow.core.ids import new_artifact_id
from bioflow.core.models import ArtifactRef, ArtifactType, BioflowError, ErrorCode
from bioflow.core.runctx import RunContext
from bioflow.core.store import ArtifactStore
from bioflow.orchestration.context import (
    ARTIFACT_SHARE,
    ArtifactIndex,
    BuiltContext,
    ContextBuilder,
    Turn,
)
from bioflow.orchestration.fsm import State
from bioflow.orchestration.summarize import REDACTION
from bioflow.orchestration.tokens import count_tokens

pytestmark = pytest.mark.unit

# The tolerance the specification allows.
BUDGET_TOLERANCE = 0.10

PATH_LIKE = re.compile(r"/(?:[\w.+@-]+/)+|[A-Za-z]:\\|cas://|file://")


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "ws")


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[RunContext]:
    with RunContext.create(tmp_path / "ws", {"ask": "triage GSE000000"}) as context:
        yield context


def seed(store: ArtifactStore, count: int = 3) -> list[ArtifactRef]:
    """Store a few real artifacts of assorted types."""
    refs = []
    rows = "\n".join(f"ENSG{index:08d}\t{index * 3}" for index in range(1, 40))
    payloads = [
        (ArtifactType.TSV, f"gene_id\tcount\n{rows}\n".encode()),
        (ArtifactType.FASTA, b">seq1 example\n" + b"ACGT" * 40 + b"\n"),
        (ArtifactType.LOG, b"\n".join(b"INFO step %d" % i for i in range(80)) + b"\n"),
        (ArtifactType.JSON, b'{"assay": "rna-seq", "n": 12}'),
        (ArtifactType.CSV, b"a,b\n1,2\n3,4\n"),
    ]
    for index in range(count):
        artifact_type, data = payloads[index % len(payloads)]
        refs.append(store.put_bytes(data, artifact_type, produced_by="bioflow@0.1.0"))
    return refs


def builder(store: ArtifactStore, budget: int = 12000, **kwargs: object) -> ContextBuilder:
    return ContextBuilder(
        budget,
        store=store,
        request={"ask": "triage GSE000000"},
        **kwargs,  # type: ignore[arg-type]
    )


def ref_without_summary() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=new_artifact_id(),
        artifact_type=ArtifactType.PARQUET,
        uri=f"cas://{'b' * 64}",
        size_bytes=4096,
        created_at=datetime.now(UTC),
    )


# --------------------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [1500, 3000, 6000, 12000, 40000])
@pytest.mark.parametrize("n_artifacts", [0, 1, 5])
def test_the_budget_is_respected(store: ArtifactStore, budget: int, n_artifacts: int) -> None:
    """The specification's headline requirement, across budgets and artifact counts."""
    context = builder(store, budget)
    refs = seed(store, n_artifacts)
    for index in range(12):
        context.record_turn(
            Turn(
                state="TRIAGE",
                tool="echo_artifact",
                arguments={"artifact_id": f"art_{index:012x}"},
                detail="a reasonably wordy result " * 8,
            )
        )

    built = context.build(State.TRIAGE, artifacts=refs)
    assert built.tokens <= budget * (1 + BUDGET_TOLERANCE), built.sections


def test_a_budget_that_cannot_hold_the_floor_is_a_resource_limit(store: ArtifactStore) -> None:
    # A silently over-budget request is not visible to anyone until a provider rejects
    # it, so the failure is raised here with the parts named.
    with pytest.raises(BioflowError) as caught:
        builder(store, 50).build(State.TRIAGE, artifacts=[])

    error = caught.value.error
    assert error.error_code is ErrorCode.RESOURCE_LIMIT
    assert "context floor" in error.message
    assert error.details["budget"] == 50
    # Every part of the floor is itemized, so it is obvious which one to shrink.
    for part in ("system", "objective", "request", "tool_schemas", "n_tools"):
        assert part in error.details


def test_the_floor_is_never_truncated(store: ArtifactStore) -> None:
    built = builder(store, 1500).build(State.TRIAGE, artifacts=seed(store, 5))
    text = built.text()
    # The state objective and the request survive intact however tight the budget is.
    assert "Decide which assay this is" in text
    assert "GSE000000" in text


def test_a_correction_is_part_of_the_floor(store: ArtifactStore) -> None:
    # Truncating the specific thing the model must read to fix its call would guarantee
    # the next attempt fails too.
    correction = "## Correction needed\n\nThat call was rejected: artifact_id must match ..."
    built = builder(store, 1500).build(
        State.TRIAGE, artifacts=seed(store, 5), correction=correction
    )
    assert correction in built.text()
    assert built.floor_tokens >= count_tokens(correction)


def test_a_nonpositive_budget_is_rejected_at_construction(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ContextBuilder(0, store=store)


def test_sections_are_accounted_for(store: ArtifactStore) -> None:
    built = builder(store, 12000).build(State.TRIAGE, artifacts=seed(store, 3))
    assert set(built.sections) == {
        "system",
        "skeleton",
        "tools",
        "correction",
        "artifacts",
        "window",
        "summary",
    }
    assert built.sections["artifacts"] > 0
    assert built.budget == 12000


def test_the_sections_are_disjoint_and_sum_to_the_total(store: ArtifactStore) -> None:
    # Reporting the whole rendered block as a section would double-count the elastic
    # parts inside it, which is exactly what made the floor wrong in the first place.
    context = builder(store, 12000, window_turns=2)
    for index in range(6):
        context.record_turn(Turn(state="PARSE", tool="echo_artifact", detail=f"row {index}"))
    built = context.build(State.TRIAGE, artifacts=seed(store, 3))

    total = sum(built.sections.values())
    assert abs(total - built.tokens) <= built.tokens * 0.05, built.sections


# --------------------------------------------------------------------------------------
# Truncation markers
# --------------------------------------------------------------------------------------


def test_a_squeezed_history_carries_a_marker(store: ArtifactStore) -> None:
    context = builder(store, 2000, window_turns=1)
    for index in range(60):
        context.record_turn(
            Turn(state="PARSE", tool="echo_artifact", detail=f"result number {index} " * 12)
        )
    built = context.build(State.TRIAGE, artifacts=[])
    assert "[truncated:" in built.text()


def test_a_squeezed_artifact_index_carries_a_marker(store: ArtifactStore) -> None:
    refs = seed(store, 5)
    index = ArtifactIndex(store)
    rendered = index.render(refs, 40)
    assert "[truncated:" in rendered


def test_nothing_is_dropped_without_saying_so(store: ArtifactStore) -> None:
    """Across a range of tight budgets, any loss of content is announced."""
    refs = seed(store, 5)
    context = builder(store, 12000, window_turns=2)
    for index in range(30):
        context.record_turn(Turn(state="PARSE", tool="echo_artifact", detail=f"row {index} " * 20))

    for budget in (1500, 2000, 4000):
        context.token_budget = budget
        text = context.build(State.TRIAGE, artifacts=refs).text()
        # Either everything fit, or the reader is told it did not.
        full = context.index.render(refs, 100_000)
        if full not in text:
            assert "[truncated:" in text, budget


# --------------------------------------------------------------------------------------
# The artifact index
# --------------------------------------------------------------------------------------


def test_the_index_renders_the_documented_columns(store: ArtifactStore) -> None:
    refs = seed(store, 3)
    rendered = ArtifactIndex(store).render(refs, 4000)
    assert "artifact_id | type | summary | size" in rendered
    for ref in refs:
        assert ref.artifact_id in rendered
    # One row per artifact, each a single line.
    rows = [line for line in rendered.splitlines() if line.startswith("art_")]
    assert len(rows) == 3
    assert all(line.count(" | ") == 3 for line in rows)


def test_the_index_never_shows_a_path(store: ArtifactStore, tmp_path: Path) -> None:
    """Invariant #2, asserted on the thing the model actually receives."""
    refs = seed(store, 5)
    for budget in (30, 120, 800, 8000):
        rendered = ArtifactIndex(store).render(refs, budget)
        assert not PATH_LIKE.search(rendered), rendered
        assert str(tmp_path) not in rendered


def test_an_empty_run_says_so(store: ArtifactStore) -> None:
    assert ArtifactIndex(store).render([], 500) == "No artifacts in this run yet."


def test_recent_artifacts_get_a_detailed_block(store: ArtifactStore) -> None:
    refs = seed(store, 3)
    rendered = ArtifactIndex(store).render(refs, 4000)
    # The table plus a description of the newest few, which is what the state is about.
    assert f"### {refs[-1].artifact_id}" in rendered
    assert "columns:" in rendered or "records" in rendered or "last" in rendered


def test_summaries_are_cached_per_artifact_and_budget(store: ArtifactStore) -> None:
    # A twenty-step run would otherwise re-read every artifact on every turn.
    refs = seed(store, 2)
    index = ArtifactIndex(store)
    index.render(refs, 4000)
    before = dict(index._cache)
    index.render(refs, 4000)
    assert index._cache == before
    assert len(before) >= len(refs)


def test_an_index_without_a_store_still_renders(store: ArtifactStore) -> None:
    # Nothing to read from, so no summary -- but the identifiers still reach the model.
    ref = ref_without_summary()
    rendered = ArtifactIndex(None).render([ref], 500)
    assert ref.artifact_id in rendered
    assert "parquet" in rendered


# --------------------------------------------------------------------------------------
# The bounded window
# --------------------------------------------------------------------------------------


def test_recent_turns_are_kept_verbatim(store: ArtifactStore) -> None:
    context = builder(store, 12000, window_turns=2)
    for index in range(5):
        context.record_turn(
            Turn(state="PARSE", tool="echo_artifact", detail=f"unique-detail-{index}")
        )
    built = context.build(State.TRIAGE, artifacts=[])
    text = built.text()
    assert "unique-detail-4" in text
    assert "unique-detail-3" in text


def test_older_turns_fold_into_one_line_each(store: ArtifactStore) -> None:
    context = builder(store, 12000, window_turns=2)
    for index in range(5):
        context.record_turn(
            Turn(state="PARSE", tool="echo_artifact", detail=f"unique-detail-{index}")
        )
    built = context.build(State.TRIAGE, artifacts=[])
    text = built.text()
    # Present, but compacted: the compact form is "STATE: tool -> outcome — detail".
    assert "PARSE: echo_artifact -> ok" in text


def test_the_window_shows_the_most_recent_turns(store: ArtifactStore) -> None:
    context = builder(store, 12000, window_turns=3)
    for index in range(10):
        context.record_turn(Turn(state="PARSE", tool=f"tool_{index}"))
    assert [turn.tool for turn in context.window] == ["tool_7", "tool_8", "tool_9"]
    assert len(context.turns) == 10


def test_a_zero_length_window_summarizes_everything(store: ArtifactStore) -> None:
    context = builder(store, 12000, window_turns=0)
    context.record_turn(Turn(state="PARSE", tool="echo_artifact", detail="only-detail"))
    built = context.build(State.TRIAGE, artifacts=[])
    assert context.window == []
    assert "PARSE: echo_artifact -> ok" in built.text()


def test_an_aged_out_turn_costs_less_than_a_verbatim_one(store: ArtifactStore) -> None:
    """Compaction has to actually compact, or the window bounds nothing."""
    turn = Turn(state="PARSE", tool="echo_artifact", detail="a wordy result " * 20)
    verbatim = count_tokens(turn.call_text()) + count_tokens(turn.result_text())
    assert count_tokens(turn.compact()) < verbatim / 2


def test_history_growth_is_bounded_by_compaction(store: ArtifactStore) -> None:
    """The property the whole module exists for.

    Asserted as marginal cost per step rather than as a ratio: under the old unbounded
    transcript every turn added its full cost forever, so the claim worth pinning is that
    a step now costs about a compacted line and never a verbatim exchange.
    """
    context = builder(store, 40000, window_turns=4)
    sizes = []
    sample = Turn(state="PARSE", tool="echo_artifact", detail="result 0 " * 15)
    for index in range(40):
        context.record_turn(
            Turn(state="PARSE", tool="echo_artifact", detail=f"result {index} " * 15)
        )
        built = context.build(State.TRIAGE, artifacts=[])
        sizes.append(built.tokens)
        assert built.tokens <= 40000

    verbatim_cost = count_tokens(sample.call_text()) + count_tokens(sample.result_text())
    marginal = (sizes[-1] - sizes[12]) / (len(sizes) - 13)
    assert marginal < verbatim_cost, f"{marginal:.1f} tokens/step vs {verbatim_cost} verbatim"
    # And it really is the compacted line that is being paid for, not something larger.
    assert marginal <= count_tokens(sample.compact()) + 2


def test_nothing_has_happened_yet_is_stated_plainly(store: ArtifactStore) -> None:
    built = builder(store, 12000).build(State.INTAKE, artifacts=[])
    assert "Nothing has happened yet." in built.text()


# --------------------------------------------------------------------------------------
# Tools and state
# --------------------------------------------------------------------------------------


def test_only_the_states_tools_are_offered(store: ArtifactStore) -> None:
    context = builder(store)
    names = {str(schema["name"]) for schema in context.schemas_for(State.TRIAGE)}
    assert names == {"echo_artifact", "request_review", "handoff"}
    # EXECUTE runs deterministically; the model is offered nothing there.
    assert context.schemas_for(State.EXECUTE) == []


def test_a_toolless_state_says_so_rather_than_leaving_a_gap(store: ArtifactStore) -> None:
    built = builder(store).build(State.EXECUTE, artifacts=[])
    assert "No tools are available in this state." in built.text()
    assert built.tools == []


def test_the_state_objective_comes_from_the_prompt_tree(store: ArtifactStore) -> None:
    # Prompts are versioned files, never inlined in code.
    intake = builder(store).build(State.INTAKE, artifacts=[]).text()
    triage = builder(store).build(State.TRIAGE, artifacts=[]).text()
    assert "Establish what you have been given" in intake
    assert "Decide which assay this is" in triage


def test_the_built_context_carries_the_system_prompt_separately(store: ArtifactStore) -> None:
    built = builder(store).build(State.TRIAGE, artifacts=[])
    assert isinstance(built, BuiltContext)
    assert "You are the judgment layer" in built.system
    # The system prompt is not duplicated into the messages.
    assert all("judgment layer" not in str(m["content"]) for m in built.messages)


def test_the_assembled_block_warns_about_truncation_markers(store: ArtifactStore) -> None:
    # The model has to know what a marker means, or it will describe a sample as a whole.
    built = builder(store).build(State.TRIAGE, artifacts=seed(store, 2))
    assert "[truncated: ...]" in built.text()


# --------------------------------------------------------------------------------------
# End to end, through a real run
# --------------------------------------------------------------------------------------


def test_no_path_reaches_the_model_from_a_real_run(ctx: RunContext, tmp_path: Path) -> None:
    """The whole assembly, not just one summarizer, checked against a real store."""
    for artifact_type, data in (
        (ArtifactType.LOG, b"ERROR could not open /Users/someone/data/run.bam\n"),
        (ArtifactType.TSV, b"name\tfile\nx\t/opt/data/x.bam\n"),
    ):
        ctx.record(ctx.store.put_bytes(data, artifact_type, run_id=ctx.run_id))

    context = ContextBuilder(8000, store=ctx.store, request={"ask": "diagnose"})
    text = context.build(State.DIAGNOSE, artifacts=ctx.runstore.list_artifacts(ctx.run_id)).text()

    assert not PATH_LIKE.search(text), text
    assert str(ctx.run_dir) not in text
    assert REDACTION in text  # the paths were there and were removed


def test_a_path_in_the_request_is_redacted_too(store: ArtifactStore) -> None:
    # The other channel a path can arrive through. A caller passing a filename would
    # otherwise walk straight past invariant #2, since the request is never truncated
    # and goes into the prompt verbatim.
    context = ContextBuilder(
        8000, store=store, request={"input": "/data/reads/sample1.fastq.gz", "ask": "triage"}
    )
    text = context.build(State.TRIAGE, artifacts=[]).text()
    assert "/data/reads" not in text
    assert REDACTION in text
    assert "triage" in text  # the rest of the request survives


def test_the_artifact_share_is_the_largest(store: ArtifactStore) -> None:
    # Without a usable identifier and some idea of shape, the model cannot produce a
    # valid call at all, which makes the other sections moot.
    assert ARTIFACT_SHARE > 0.4
