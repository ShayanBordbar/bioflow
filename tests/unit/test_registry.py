"""Unit tests for the tool registry, invoke(), and the golden harness."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from bioflow.core.contract_tests import assert_tool_case, assert_tool_contract, case_names
from bioflow.core.models import (
    Actor,
    ArtifactId,
    ArtifactType,
    BioflowError,
    CostClass,
    ErrorCode,
    EventStatus,
    SideEffect,
    StructuredError,
)
from bioflow.core.registry import Registry, ToolResult, invoke, tool
from bioflow.core.runctx import RunContext

pytestmark = pytest.mark.unit


class SampleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Anything at all.")
    artifact_id: ArtifactId | None = None


class SampleOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shouted: str
    produced_id: ArtifactId | None = None


@pytest.fixture
def bench() -> Registry:
    """An isolated registry, so tests never touch the module-level one."""
    return Registry()


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[RunContext]:
    with RunContext.create(tmp_path / "workspace", {"case": "registry"}) as context:
        yield context


@pytest.fixture
def shout(bench: Registry) -> Registry:
    @tool(name="shout", version="1.0.0", summary="Uppercase some text.", into=bench)
    def _shout(params: SampleIn, ctx: RunContext) -> SampleOut:
        return SampleOut(shouted=params.text.upper())

    return bench


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def test_tool_builds_a_spec_from_the_models(shout: Registry) -> None:
    (spec,) = shout.list_specs()
    assert spec.name == "shout"
    assert spec.version == "1.0.0"
    assert spec.side_effect is SideEffect.NONE
    assert spec.cost_class is CostClass.CHEAP
    assert spec.input_schema["type"] == "object"
    assert "text" in spec.input_schema["properties"]  # type: ignore[operator,index]
    assert spec.output_schema["properties"]["shouted"]["type"] == "string"  # type: ignore[index]


def test_tool_records_declared_metadata(bench: Registry) -> None:
    @tool(
        name="risky",
        version="2.1.0",
        summary="Does something expensive.",
        side_effect=SideEffect.EXECUTE,
        cost_class=CostClass.EXPENSIVE,
        requires_confirmation=True,
        tags=("danger",),
        into=bench,
    )
    def _risky(params: SampleIn, ctx: RunContext) -> SampleOut:
        return SampleOut(shouted=params.text)

    (spec,) = bench.list_specs()
    assert spec.side_effect is SideEffect.EXECUTE
    assert spec.cost_class is CostClass.EXPENSIVE
    assert spec.requires_confirmation is True
    assert spec.tags == ("danger",)


def test_duplicate_registration_is_refused(shout: Registry) -> None:
    with pytest.raises(ValueError, match="already registered"):

        @tool(name="shout", version="1.0.0", summary="A clashing tool.", into=shout)
        def _other(params: SampleIn, ctx: RunContext) -> SampleOut:
            return SampleOut(shouted="")


def test_missing_annotations_are_refused_at_decoration(bench: Registry) -> None:
    # A malformed tool should break the build, not wait to surprise someone mid-run.
    with pytest.raises(TypeError, match="missing annotations"):

        @tool(name="bad", version="1.0.0", summary="No annotations.", into=bench)
        def _bad(params, ctx):  # type: ignore[no-untyped-def]
            return SampleOut(shouted="")

    assert len(bench) == 0


@pytest.mark.parametrize("annotation", ["params", "return"])
def test_non_model_annotations_are_refused(bench: Registry, annotation: str) -> None:
    with pytest.raises(TypeError, match="pydantic BaseModel subclass"):
        if annotation == "params":

            @tool(name="bad", version="1.0.0", summary="Bad params.", into=bench)
            def _bad_params(params: str, ctx: RunContext) -> SampleOut:  # type: ignore[arg-type]
                return SampleOut(shouted="")

        else:

            @tool(name="bad", version="1.0.0", summary="Bad return.", into=bench)
            def _bad_return(params: SampleIn, ctx: RunContext) -> str:  # type: ignore[return-value]
                return ""


def test_registry_lookup_and_listing(shout: Registry) -> None:
    assert shout.get("shout") is not None
    assert shout.get("absent") is None
    assert "shout" in shout
    assert len(shout) == 1
    shout.clear()
    assert len(shout) == 0


def test_list_specs_is_sorted_by_name(bench: Registry) -> None:
    for name in ("zebra", "alpha", "middle"):

        @tool(name=name, version="1.0.0", summary=f"The {name} tool.", into=bench)
        def _t(params: SampleIn, ctx: RunContext) -> SampleOut:
            return SampleOut(shouted="")

    assert [spec.name for spec in bench.list_specs()] == ["alpha", "middle", "zebra"]


def test_to_llm_schema_is_provider_agnostic(shout: Registry) -> None:
    (entry,) = shout.to_llm_schema()
    assert set(entry) == {"name", "description", "input_schema"}
    assert entry["name"] == "shout"
    assert entry["description"] == "Uppercase some text."
    # No filesystem paths, no python types -- just JSON Schema.
    assert json.loads(json.dumps(entry)) == entry


# --------------------------------------------------------------------------------------
# invoke()
# --------------------------------------------------------------------------------------


def test_invoke_runs_a_tool(shout: Registry, ctx: RunContext) -> None:
    result = invoke("shout", {"text": "hello"}, ctx, into=shout)
    assert result.ok is True
    assert isinstance(result.output, SampleOut)
    assert result.output.shouted == "HELLO"
    assert result.error is None
    assert result.duration_ms >= 0


def test_invoke_reports_an_unknown_tool_without_raising(shout: Registry, ctx: RunContext) -> None:
    # An LLM will invent a tool name eventually; it needs something it can read.
    result = invoke("teleport", {}, ctx, into=shout)
    assert result.ok is False
    assert result.error is not None
    assert result.error.error_code is ErrorCode.NOT_FOUND
    assert result.error.remediation is not None
    assert "shout" in result.error.remediation


def test_invoke_reports_invalid_arguments_compactly(shout: Registry, ctx: RunContext) -> None:
    result = invoke("shout", {"text": 5, "artifact_id": "nope"}, ctx, into=shout)
    assert result.ok is False
    assert result.error is not None
    assert result.error.error_code is ErrorCode.VALIDATION_FAILED
    # One readable line per bad field is what gets fed back into the next prompt.
    assert "artifact_id" in result.error.message
    errors = result.error.details["errors"]
    assert isinstance(errors, list)
    assert {e["loc"] for e in errors if isinstance(e, dict)} == {"text", "artifact_id"}


def test_invoke_reports_a_missing_required_field(shout: Registry, ctx: RunContext) -> None:
    result = invoke("shout", {}, ctx, into=shout)
    assert result.ok is False
    assert result.error is not None
    assert "text" in result.error.message


def test_invoke_preserves_a_structured_error_from_a_tool(bench: Registry, ctx: RunContext) -> None:
    payload = StructuredError(
        error_code=ErrorCode.OUT_OF_DOMAIN,
        message="sequence is outside the training distribution",
    )

    @tool(name="predict", version="1.0.0", summary="Predict something.", into=bench)
    def _predict(params: SampleIn, ctx: RunContext) -> SampleOut:
        raise BioflowError(payload)

    result = invoke("predict", {"text": "x"}, ctx, into=bench)
    assert result.ok is False
    # A tool that diagnosed its own failure keeps the diagnosis.
    assert result.error == payload


def test_invoke_wraps_an_unexpected_exception(bench: Registry, ctx: RunContext) -> None:
    @tool(name="explode", version="1.0.0", summary="Fails badly.", into=bench)
    def _explode(params: SampleIn, ctx: RunContext) -> SampleOut:
        raise ZeroDivisionError("nope")

    result = invoke("explode", {"text": "x"}, ctx, into=bench)
    assert result.ok is False
    assert result.error is not None
    assert result.error.error_code is ErrorCode.INTERNAL
    assert "ZeroDivisionError" in result.error.message
    assert result.error.raised_by == "explode"


# --------------------------------------------------------------------------------------
# Confirmation gating
# --------------------------------------------------------------------------------------


@pytest.fixture
def guarded(bench: Registry) -> Registry:
    @tool(
        name="guarded",
        version="1.0.0",
        summary="Needs a human.",
        side_effect=SideEffect.WRITE,
        requires_confirmation=True,
        into=bench,
    )
    def _guarded(params: SampleIn, ctx: RunContext) -> SampleOut:
        return SampleOut(shouted=params.text.upper())

    return bench


def test_confirmation_required_tool_is_refused_by_default(
    guarded: Registry, ctx: RunContext
) -> None:
    result = invoke("guarded", {"text": "x"}, ctx, into=guarded)
    assert result.ok is False
    assert result.error is not None
    assert result.error.error_code is ErrorCode.VALIDATION_FAILED
    assert "confirmation" in result.error.message


def test_confirmation_required_tool_runs_when_confirmed(guarded: Registry, ctx: RunContext) -> None:
    result = invoke("guarded", {"text": "x"}, ctx, confirmed=True, into=guarded)
    assert result.ok is True


def test_unconfirmed_refusal_happens_before_the_tool_body(
    guarded: Registry, ctx: RunContext
) -> None:
    # Refusal must precede execution, or the gate is decorative.
    invoke("guarded", {"text": "x"}, ctx, into=guarded)
    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.status is EventStatus.ERROR


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def test_invoke_emits_exactly_one_event_on_success(shout: Registry, ctx: RunContext) -> None:
    invoke("shout", {"text": "hello"}, ctx, into=shout)
    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.action == "shout"
    assert event.tool == "shout"
    assert event.tool_version == "1.0.0"
    assert event.actor is Actor.TOOL
    assert event.status is EventStatus.OK
    assert event.error is None
    assert event.duration_ms is not None


def test_invoke_emits_exactly_one_event_on_failure(shout: Registry, ctx: RunContext) -> None:
    invoke("shout", {"text": 5}, ctx, into=shout)
    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.status is EventStatus.ERROR
    assert event.error is not None
    assert event.error.error_code is ErrorCode.VALIDATION_FAILED


def test_events_reach_the_jsonl_mirror(shout: Registry, ctx: RunContext) -> None:
    invoke("shout", {"text": "hello"}, ctx, into=shout)
    lines = ctx.runstore.provenance_path(ctx.run_id).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["action"] for line in lines] == ["shout"]


def test_params_digest_is_canonical(shout: Registry, ctx: RunContext) -> None:
    invoke("shout", {"text": "a", "artifact_id": None}, ctx, into=shout)
    invoke("shout", {"artifact_id": None, "text": "a"}, ctx, into=shout)
    first, second = ctx.runstore.list_provenance(ctx.run_id)
    # Key order is not part of the call's identity.
    assert first.params_sha256 == second.params_sha256


def test_params_digest_changes_with_the_arguments(shout: Registry, ctx: RunContext) -> None:
    invoke("shout", {"text": "a"}, ctx, into=shout)
    invoke("shout", {"text": "b"}, ctx, into=shout)
    first, second = ctx.runstore.list_provenance(ctx.run_id)
    assert first.params_sha256 != second.params_sha256


def test_artifact_ids_are_scanned_out_of_params_and_output(
    bench: Registry, ctx: RunContext
) -> None:
    incoming = ctx.record(ctx.store.put_bytes(b"in", ArtifactType.OTHER))
    produced = ctx.record(ctx.store.put_bytes(b"out", ArtifactType.OTHER))

    @tool(name="transform", version="1.0.0", summary="Transforms.", into=bench)
    def _transform(params: SampleIn, ctx: RunContext) -> SampleOut:
        return SampleOut(shouted="done", produced_id=produced.artifact_id)

    invoke("transform", {"text": "x", "artifact_id": incoming.artifact_id}, ctx, into=bench)
    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    # The ID format is closed, so a matching string is an artifact reference.
    assert event.inputs == (incoming.artifact_id,)
    assert event.outputs == (produced.artifact_id,)


def test_ids_nested_inside_structures_are_found(bench: Registry, ctx: RunContext) -> None:
    class NestedIn(BaseModel):
        payload: dict[str, list[str]]

    @tool(name="nested", version="1.0.0", summary="Nested inputs.", into=bench)
    def _nested(params: NestedIn, ctx: RunContext) -> SampleOut:
        return SampleOut(shouted="ok")

    invoke(
        "nested",
        {"payload": {"refs": ["art_0123456789ab", "not-an-id", "art_0123456789ab"]}},
        ctx,
        into=bench,
    )
    (event,) = ctx.runstore.list_provenance(ctx.run_id)
    assert event.inputs == ("art_0123456789ab",)  # deduplicated, order preserved


def test_tool_result_json_dict_shape(shout: Registry, ctx: RunContext) -> None:
    result = invoke("shout", {"text": "hi"}, ctx, into=shout)
    assert result.to_json_dict() == {
        "ok": True,
        "output": {"shouted": "HI", "produced_id": None},
        "error": None,
    }


def test_tool_result_is_frozen(shout: Registry, ctx: RunContext) -> None:
    result = invoke("shout", {"text": "hi"}, ctx, into=shout)
    with pytest.raises(AttributeError):
        result.ok = False  # type: ignore[misc]
    assert isinstance(result, ToolResult)


# --------------------------------------------------------------------------------------
# The golden harness itself
# --------------------------------------------------------------------------------------


def _write_case(directory: Path, name: str, files: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, payload in files.items():
        (directory / f"{name}{suffix}").write_text(json.dumps(payload), encoding="utf-8")


def test_harness_passes_a_matching_case(shout: Registry, tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "ok",
        {
            ".in.json": {"text": "hi"},
            ".out.json": {
                "ok": True,
                "output": {"shouted": "HI", "produced_id": None},
                "error": None,
            },
        },
    )
    assert_tool_contract("shout", tmp_path, into=shout)


def test_harness_reports_a_mismatch_with_a_diff(shout: Registry, tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "wrong",
        {
            ".in.json": {"text": "hi"},
            ".out.json": {
                "ok": True,
                "output": {"shouted": "SOMETHING ELSE", "produced_id": None},
                "error": None,
            },
        },
    )
    with pytest.raises(AssertionError) as excinfo:
        assert_tool_contract("shout", tmp_path, into=shout)
    message = str(excinfo.value)
    assert "does not match" in message
    assert "SOMETHING ELSE" in message  # the diff shows what was expected
    assert "HI" in message


def test_harness_honours_the_ignore_file(shout: Registry, tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "ignored",
        {
            ".in.json": {"text": "hi"},
            ".out.json": {
                "ok": True,
                "output": {"shouted": "NOT WHAT HAPPENS", "produced_id": None},
                "error": None,
            },
            ".ignore.json": ["output.shouted"],
        },
    )
    assert_tool_contract("shout", tmp_path, into=shout)


def test_harness_ignore_supports_wildcards(bench: Registry, tmp_path: Path) -> None:
    class ListOut(BaseModel):
        items: list[dict[str, str]]

    @tool(name="lister", version="1.0.0", summary="Lists things.", into=bench)
    def _lister(params: SampleIn, ctx: RunContext) -> ListOut:
        return ListOut(items=[{"id": "varies", "kind": "stable"}])

    _write_case(
        tmp_path,
        "wild",
        {
            ".in.json": {"text": "x"},
            ".out.json": {
                "ok": True,
                "output": {"items": [{"id": "anything", "kind": "stable"}]},
                "error": None,
            },
            ".ignore.json": ["output.items.*.id"],
        },
    )
    assert_tool_contract("lister", tmp_path, into=bench)


def test_harness_seeds_artifacts_and_substitutes_ids(bench: Registry, tmp_path: Path) -> None:
    @tool(name="reflect", version="1.0.0", summary="Reflects an ID.", into=bench)
    def _reflect(params: SampleIn, ctx: RunContext) -> SampleOut:
        return SampleOut(shouted=params.text, produced_id=params.artifact_id)

    _write_case(
        tmp_path,
        "seeded",
        {
            ".setup.json": {
                "artifacts": [{"as": "seq", "content": "ACGT", "artifact_type": "fasta"}]
            },
            ".in.json": {"text": "x", "artifact_id": "${seq}"},
            ".out.json": {
                "ok": True,
                "output": {"shouted": "x", "produced_id": "${seq}"},
                "error": None,
            },
        },
    )
    # The placeholder is substituted on both sides, so the golden pins the real ID
    # rather than just ignoring it.
    assert_tool_contract("reflect", tmp_path, into=bench)


def test_harness_requires_at_least_one_case(shout: Registry, tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no golden cases"):
        assert_tool_contract("shout", tmp_path, into=shout)


def test_harness_requires_a_matching_out_file(shout: Registry, tmp_path: Path) -> None:
    _write_case(tmp_path, "lonely", {".in.json": {"text": "hi"}})
    with pytest.raises(AssertionError, match=r"has no lonely\.out\.json"):
        assert_tool_case("shout", tmp_path, "lonely", into=shout)


def test_case_names_are_discovered_and_sorted(shout: Registry, tmp_path: Path) -> None:
    for name in ("zebra", "alpha"):
        _write_case(tmp_path, name, {".in.json": {"text": "x"}})
    assert case_names(tmp_path) == ["alpha", "zebra"]


@pytest.mark.parametrize(
    ("files", "message"),
    [
        pytest.param(
            {".in.json": ["not", "an", "object"], ".out.json": {}},
            r"must contain a JSON object",
            id="in-file-not-an-object",
        ),
        pytest.param(
            {".in.json": {"text": "x"}, ".out.json": {}, ".setup.json": ["nope"]},
            r"must contain a JSON object",
            id="setup-not-an-object",
        ),
        pytest.param(
            {".in.json": {"text": "x"}, ".out.json": {}, ".setup.json": {"artifacts": "nope"}},
            r"'artifacts' must be a list",
            id="artifacts-not-a-list",
        ),
        pytest.param(
            {".in.json": {"text": "x"}, ".out.json": {}, ".setup.json": {"artifacts": ["nope"]}},
            r"artifact 0 must be a JSON object",
            id="artifact-not-an-object",
        ),
        pytest.param(
            {".in.json": {"text": "x"}, ".out.json": {}, ".ignore.json": {"not": "a list"}},
            r"must contain a JSON list",
            id="ignore-not-a-list",
        ),
    ],
)
def test_harness_rejects_a_malformed_golden(
    shout: Registry, tmp_path: Path, files: dict[str, object], message: str
) -> None:
    # These messages are read by whoever is writing a golden case, so they are part of
    # the harness's usability rather than incidental defensive code.
    _write_case(tmp_path, "broken", files)
    with pytest.raises(AssertionError, match=message):
        assert_tool_contract("shout", tmp_path, into=shout)


def test_harness_reports_a_missing_in_file(shout: Registry, tmp_path: Path) -> None:
    _write_case(tmp_path, "orphan", {".out.json": {}})
    with pytest.raises(AssertionError, match="missing golden file"):
        assert_tool_case("shout", tmp_path, "orphan", into=shout)


def test_ignoring_a_whole_list_and_the_root(shout: Registry, tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "blunt",
        {
            ".in.json": {"text": "hi"},
            ".out.json": {"ok": True, "output": {"anything": "at all"}, "error": None},
            # "output" drops the subtree; "output.*" would drop each value under it.
            ".ignore.json": ["output"],
        },
    )
    assert_tool_contract("shout", tmp_path, into=shout)
