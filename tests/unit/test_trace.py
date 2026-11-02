"""Unit tests for the agent trace file."""

import json
from pathlib import Path

import pytest

from bioflow.core.models import BioflowError, ErrorCode
from bioflow.orchestration.trace import (
    KIND_CALL,
    KIND_FOOTER,
    KIND_HEADER,
    TraceWriter,
    arguments_digest,
    messages_digest,
    parse_header,
    read_trace,
    recorded_confirmations,
    recorded_tool_calls,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def writer(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "logs" / "trace.jsonl")


def write_call(writer: TraceWriter, **overrides: object) -> None:
    payload: dict[str, object] = {
        "state": "TRIAGE",
        "prompt_version": "v1",
        "messages_sha256": "0" * 64,
        "tool_call": {"name": "echo_artifact", "arguments": {}},
        "validation_errors": [],
        "repair_attempt": 0,
        "confirmed": None,
        "usage": {"input_tokens": 1},
        "latency_ms": 12,
        "model": "fake",
        "outcome": "invoked",
    }
    payload.update(overrides)
    writer.call(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------------------


def test_message_digest_ignores_key_order() -> None:
    left = messages_digest([{"role": "user", "content": "hi"}])
    right = messages_digest([{"content": "hi", "role": "user"}])
    assert left == right


def test_message_digest_changes_with_content() -> None:
    assert messages_digest([{"role": "user", "content": "a"}]) != messages_digest(
        [{"role": "user", "content": "b"}]
    )


def test_argument_digest_ignores_key_order() -> None:
    assert arguments_digest({"a": 1, "b": 2}) == arguments_digest({"b": 2, "a": 1})


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def test_the_writer_creates_its_parent_directory(writer: TraceWriter) -> None:
    writer.header(
        run_id="run_000000000000",
        request={},
        cassette=None,
        model="m",
        prompt_version="v1",
        limits={},
    )
    assert writer.path.is_file()


def test_records_are_sequenced_and_timestamped(writer: TraceWriter) -> None:
    writer.header(
        run_id="run_000000000000",
        request={"a": 1},
        cassette="c",
        model="m",
        prompt_version="v1",
        limits={"max_steps": 3},
    )
    write_call(writer)
    writer.footer(final_state="DONE", tool_calls=["echo_artifact"], metrics={})

    records = read_trace(writer.path)
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert [r["kind"] for r in records] == [KIND_HEADER, KIND_CALL, KIND_FOOTER]
    assert all(isinstance(r["ts"], str) for r in records)


def test_each_record_is_one_line_of_json(writer: TraceWriter) -> None:
    writer.header(
        run_id="run_000000000000",
        request={},
        cassette=None,
        model="m",
        prompt_version="v1",
        limits={},
    )
    write_call(writer)
    lines = writer.path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must not raise


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def test_reading_a_missing_trace_says_so(tmp_path: Path) -> None:
    with pytest.raises(BioflowError) as caught:
        read_trace(tmp_path / "nope.jsonl")
    assert caught.value.error_code is ErrorCode.NOT_FOUND


def test_a_corrupt_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"kind": "header"}\nnot json\n')
    with pytest.raises(BioflowError) as caught:
        read_trace(path)
    assert caught.value.error_code is ErrorCode.SCHEMA_MISMATCH
    assert "line 2" in caught.value.error.message


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"kind": "header"}\n\n\n{"kind": "footer"}\n')
    assert len(read_trace(path)) == 2


def test_the_header_carries_what_replay_needs(writer: TraceWriter) -> None:
    writer.header(
        run_id="run_abcdef123456",
        request={"ask": "triage"},
        cassette="agent_happy",
        model="claude-opus-5",
        prompt_version="v1",
        limits={"max_steps": 25},
        root="/tmp/ws",
    )
    header = parse_header(read_trace(writer.path))
    assert header.run_id == "run_abcdef123456"
    assert header.request == {"ask": "triage"}
    assert header.cassette == "agent_happy"
    assert header.model == "claude-opus-5"
    assert header.limits == {"max_steps": 25}
    assert header.root == "/tmp/ws"


def test_a_trace_without_a_header_is_not_replayable(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"kind": "call"}\n')
    with pytest.raises(BioflowError) as caught:
        parse_header(read_trace(path))
    assert caught.value.error_code is ErrorCode.SCHEMA_MISMATCH


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------


def test_the_footer_supplies_the_tool_call_sequence(writer: TraceWriter) -> None:
    write_call(writer)
    writer.footer(final_state="DONE", tool_calls=["a", "b"], metrics={})
    assert recorded_tool_calls(read_trace(writer.path)) == ["a", "b"]


def test_a_footerless_trace_still_yields_its_sequence(writer: TraceWriter) -> None:
    # A run that died before writing a footer is still worth comparing against.
    write_call(writer, tool_call={"name": "first", "arguments": {}})
    write_call(writer, tool_call={"name": "rejected", "arguments": {}}, outcome="out_of_state")
    write_call(writer, tool_call={"name": "second", "arguments": {}})
    assert recorded_tool_calls(read_trace(writer.path)) == ["first", "second"]


def test_confirmations_are_recovered_in_order(writer: TraceWriter) -> None:
    write_call(writer, confirmed=None)
    write_call(writer, confirmed=True)
    write_call(writer, confirmed=False)
    assert recorded_confirmations(read_trace(writer.path)) == [True, False]
