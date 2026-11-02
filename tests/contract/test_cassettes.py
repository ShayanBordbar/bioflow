"""Contract tests for LLM cassettes.

The property that matters: a call recorded once replays forever, in CI, with no network
and no credentials. Everything here runs offline by construction.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from bioflow.adapters.llm.anthropic_client import API_URL, AnthropicClient
from bioflow.adapters.llm.base import LLMResponse, StopReason, ToolCall, Usage
from bioflow.adapters.llm.cassette import (
    CassetteClient,
    canonicalize,
    request_hash,
)
from bioflow.adapters.llm.fake_client import FakeLLMClient
from bioflow.adapters.llm.openai_client import OpenAICompatibleClient
from bioflow.config import Settings
from bioflow.core.models import BioflowError, ErrorCode

pytestmark = pytest.mark.contract

MESSAGES = [{"role": "user", "content": "classify this dataset"}]


def _api_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_01",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": "rna-seq"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 120, "output_tokens": 8},
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def _live_client() -> AnthropicClient:
    return AnthropicClient(Settings(llm_model="claude-opus-5", offline=False))


# --------------------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------------------


def test_key_order_does_not_change_the_hash() -> None:
    first = {"model": "claude-opus-5", "messages": MESSAGES, "max_tokens": 100}
    second = {"max_tokens": 100, "messages": MESSAGES, "model": "claude-opus-5"}
    assert request_hash(first) == request_hash(second)


def test_volatile_fields_are_stripped_before_hashing() -> None:
    base = {"model": "claude-opus-5", "messages": MESSAGES}
    with_noise = {
        **base,
        "metadata": {"user_id": "u-1"},
        "request_id": "req-abc",
        "timestamp": "2026-08-28T09:00:00Z",
        "trace_id": "trace-1",
    }
    # A cassette recorded on Tuesday has to hit on Wednesday.
    assert request_hash(base) == request_hash(with_noise)


def test_volatile_stripping_reaches_into_nested_structures() -> None:
    a = {"messages": [{"role": "user", "content": "x", "message_id": "m1"}]}
    b = {"messages": [{"role": "user", "content": "x", "message_id": "m2"}]}
    assert request_hash(a) == request_hash(b)


def test_a_real_difference_changes_the_hash() -> None:
    base = {"model": "claude-opus-5", "messages": MESSAGES}
    assert request_hash(base) != request_hash({**base, "model": "claude-sonnet-5"})
    assert request_hash(base) != request_hash(
        {"model": "claude-opus-5", "messages": [{"role": "user", "content": "different"}]}
    )


def test_canonical_form_is_stable_json() -> None:
    canonical = canonicalize({"b": 1, "a": {"d": 2, "c": 3}})
    assert canonical == '{"a":{"c":3,"d":2},"b":1}'
    assert json.loads(canonical) == {"a": {"c": 3, "d": 2}, "b": 1}


# --------------------------------------------------------------------------------------
# Record -> replay
# --------------------------------------------------------------------------------------


@respx.mock
def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))

    recorder = CassetteClient(_live_client(), name="triage", mode="record", root=tmp_path)
    recorded = recorder.complete(MESSAGES)
    assert route.call_count == 1
    assert recorded.text == "rna-seq"

    # Replay with no live client at all: nothing can reach the network from here.
    player = CassetteClient(
        None, name="triage", mode="replay", root=tmp_path, request_builder=_live_client()
    )
    replayed = player.complete(MESSAGES)

    assert route.call_count == 1  # still one; replay made no request
    assert replayed.text == recorded.text
    assert replayed.usage.input_tokens == recorded.usage.input_tokens
    assert replayed.model == recorded.model


@respx.mock
def test_recorded_file_is_readable_and_named_by_hash(tmp_path: Path) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))
    client = _live_client()
    CassetteClient(client, name="triage", mode="record", root=tmp_path).complete(MESSAGES)

    (recorded,) = list((tmp_path / "triage").glob("*.json"))
    payload = json.loads(recorded.read_text(encoding="utf-8"))
    assert recorded.stem == payload["request_sha256"]
    assert recorded.stem == request_hash(client.build_request(MESSAGES))
    # The stored request is for a human reading the diff; only the hash is looked up.
    assert payload["request"]["model"] == "claude-opus-5"
    assert payload["response"]["text"] == "rna-seq"


@respx.mock
def test_tool_calls_survive_the_round_trip(tmp_path: Path) -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=_api_body(
                stop_reason="tool_use",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "echo_artifact",
                        "input": {"artifact_id": "art_0123456789ab"},
                    }
                ],
            ),
        )
    )
    CassetteClient(_live_client(), name="tools", mode="record", root=tmp_path).complete(MESSAGES)

    player = CassetteClient(
        None, name="tools", mode="replay", root=tmp_path, request_builder=_live_client()
    )
    replayed = player.complete(MESSAGES)
    assert replayed.stop_reason is StopReason.TOOL_USE
    (call,) = replayed.tool_calls
    assert call.name == "echo_artifact"
    assert call.arguments == {"artifact_id": "art_0123456789ab"}


@respx.mock
def test_a_refusal_replays_as_a_refusal(tmp_path: Path) -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(200, json=_api_body(stop_reason="refusal", content=[]))
    )
    CassetteClient(_live_client(), name="refusal", mode="record", root=tmp_path).complete(MESSAGES)
    player = CassetteClient(
        None, name="refusal", mode="replay", root=tmp_path, request_builder=_live_client()
    )
    assert player.complete(MESSAGES).refused is True


def test_replay_works_with_no_http_transport_configured(tmp_path: Path) -> None:
    # No respx.mock decorator here at all. If replay touched the network this would
    # fail, which is the whole point of the cassette in CI.
    directory = tmp_path / "offline"
    directory.mkdir()
    builder = _live_client()
    request = builder.build_request(MESSAGES)
    response = LLMResponse(
        text="from disk",
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.5),
        model="claude-opus-5",
    )
    (directory / f"{request_hash(request)}.json").write_text(
        json.dumps({"response": response.model_dump(mode="json")}), encoding="utf-8"
    )

    player = CassetteClient(
        None, name="offline", mode="replay", root=tmp_path, request_builder=builder
    )
    assert player.complete(MESSAGES).text == "from disk"


def test_replay_works_while_offline_mode_is_on(tmp_path: Path) -> None:
    # BIOFLOW_OFFLINE blocks live calls; replay must still work, or the offline guard
    # would make the test suite unrunnable.
    directory = tmp_path / "guarded"
    directory.mkdir()
    builder = AnthropicClient(Settings(llm_model="claude-opus-5", offline=True))
    request = builder.build_request(MESSAGES)
    (directory / f"{request_hash(request)}.json").write_text(
        json.dumps({"response": LLMResponse(text="cached").model_dump(mode="json")}),
        encoding="utf-8",
    )
    player = CassetteClient(
        None, name="guarded", mode="replay", root=tmp_path, request_builder=builder
    )
    assert player.complete(MESSAGES).text == "cached"


# --------------------------------------------------------------------------------------
# Misses
# --------------------------------------------------------------------------------------


def test_a_miss_says_which_cassette_which_hash_and_how_to_fix_it(tmp_path: Path) -> None:
    builder = _live_client()
    player = CassetteClient(
        None, name="absent", mode="replay", root=tmp_path, request_builder=builder
    )
    with pytest.raises(BioflowError) as excinfo:
        player.complete(MESSAGES)

    error = excinfo.value.error
    expected = request_hash(builder.build_request(MESSAGES))
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND
    assert "absent" in error.message
    assert expected in error.message
    assert "BIOFLOW_CASSETTE_MODE=record" in (error.remediation or "")
    assert error.details["expected_sha256"] == expected


@respx.mock
def test_a_changed_request_misses_rather_than_serving_the_wrong_recording(
    tmp_path: Path,
) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))
    CassetteClient(_live_client(), name="drift", mode="record", root=tmp_path).complete(MESSAGES)

    player = CassetteClient(
        None, name="drift", mode="replay", root=tmp_path, request_builder=_live_client()
    )
    with pytest.raises(BioflowError) as excinfo:
        player.complete([{"role": "user", "content": "a different question"}])
    # The directory has a recording, just not this one -- the message says so.
    assert excinfo.value.error.details["available"]


def test_a_miss_lists_what_the_directory_does_hold(tmp_path: Path) -> None:
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / ("a" * 64 + ".json")).write_text("{}", encoding="utf-8")
    player = CassetteClient(
        None, name="partial", mode="replay", root=tmp_path, request_builder=_live_client()
    )
    with pytest.raises(BioflowError) as excinfo:
        player.complete(MESSAGES)
    assert excinfo.value.error.details["available"] == ["a" * 64]
    assert "1 recording(s)" in excinfo.value.error.message


# --------------------------------------------------------------------------------------
# Modes and wrapping
# --------------------------------------------------------------------------------------


def test_off_mode_passes_straight_through(tmp_path: Path) -> None:
    inner = FakeLLMClient([LLMResponse(text="passed through")])
    client = CassetteClient(inner, name="unused", mode="off", root=tmp_path)
    assert client.complete(MESSAGES).text == "passed through"
    assert not (tmp_path / "unused").exists()


def test_a_fake_client_can_be_recorded_and_replayed(tmp_path: Path) -> None:
    # No build_request on a fake, so the wrapper keys on the call arguments instead.
    inner = FakeLLMClient([LLMResponse(text="scripted", model="claude-opus-5")])
    CassetteClient(inner, name="fake", mode="record", root=tmp_path).complete(MESSAGES)

    player = CassetteClient(None, name="fake", mode="replay", root=tmp_path)
    assert player.complete(MESSAGES).text == "scripted"


def test_replay_without_a_client_never_needs_one(tmp_path: Path) -> None:
    inner = FakeLLMClient([LLMResponse(text="once")])
    CassetteClient(inner, name="single", mode="record", root=tmp_path).complete(MESSAGES)
    # The fake's script is now exhausted; replay must not consult it.
    player = CassetteClient(inner, name="single", mode="replay", root=tmp_path)
    assert player.complete(MESSAGES).text == "once"


def test_an_unknown_mode_is_rejected(tmp_path: Path) -> None:
    client = CassetteClient(FakeLLMClient(), name="x", mode="sideways", root=tmp_path)
    with pytest.raises(BioflowError) as excinfo:
        client.complete(MESSAGES)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED


def test_recording_without_a_live_client_is_refused(tmp_path: Path) -> None:
    client = CassetteClient(None, name="x", mode="record", root=tmp_path)
    with pytest.raises(BioflowError) as excinfo:
        client.complete(MESSAGES)
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT


def test_cassette_client_satisfies_the_protocol(tmp_path: Path) -> None:
    from bioflow.adapters.llm.base import LLMClient

    assert isinstance(CassetteClient(FakeLLMClient(), name="x", root=tmp_path), LLMClient)


@respx.mock
def test_re_recording_the_same_call_overwrites_one_file(tmp_path: Path) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))
    recorder = CassetteClient(_live_client(), name="stable", mode="record", root=tmp_path)
    recorder.complete(MESSAGES)
    recorder.complete(MESSAGES)
    # Same logical request, same hash, same file: cassettes do not accumulate dupes.
    assert len(list((tmp_path / "stable").glob("*.json"))) == 1


@respx.mock
def test_usage_from_a_replayed_call_still_accounts(tmp_path: Path) -> None:
    from bioflow.core.usage import UsageAccumulator

    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))
    CassetteClient(_live_client(), name="cost", mode="record", root=tmp_path).complete(MESSAGES)

    player = CassetteClient(
        None, name="cost", mode="replay", root=tmp_path, request_builder=_live_client()
    )
    accumulator = UsageAccumulator()
    accumulator.record(player.complete(MESSAGES))

    totals = accumulator.as_dict()
    assert totals["input_tokens"] == 120
    assert totals["n_calls"] == 1
    # Cost is preserved from the recording rather than recomputed at replay time.
    assert isinstance(totals["cost_usd"], float)
    assert totals["cost_usd"] > 0


def test_tool_call_model_round_trips_through_json() -> None:
    call = ToolCall(name="echo", arguments={"a": [1, None]}, call_id="toolu_1")
    assert ToolCall.model_validate_json(call.model_dump_json()) == call


# --------------------------------------------------------------------------------------
# Cross-provider behaviour
# --------------------------------------------------------------------------------------


def _openai_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        Settings(
            llm_provider="openai-compatible",
            llm_model="llama3.1:8b",
            llm_base_url="http://localhost:11434/v1",
            offline=False,
        )
    )


def _openai_body() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "llama3.1:8b",
        "choices": [{"message": {"content": "rna-seq"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 8},
    }


@respx.mock
def test_an_openai_compatible_call_records_and_replays(tmp_path: Path) -> None:
    client = _openai_client()
    respx.post(client.completions_url).mock(return_value=httpx.Response(200, json=_openai_body()))

    CassetteClient(client, name="local-triage", mode="record", root=tmp_path).complete(MESSAGES)

    player = CassetteClient(
        None, name="local-triage", mode="replay", root=tmp_path, request_builder=_openai_client()
    )
    replayed = player.complete(MESSAGES)
    assert replayed.text == "rna-seq"
    assert replayed.model == "llama3.1:8b"


def test_the_two_providers_hash_the_same_call_differently() -> None:
    # Request bodies differ (model, message shape, temperature), so a cassette recorded
    # against one provider can never be served to the other by accident.
    anthropic_request = _live_client().build_request(MESSAGES)
    openai_request = _openai_client().build_request(MESSAGES)
    assert request_hash(anthropic_request) != request_hash(openai_request)


@respx.mock
def test_a_provider_switch_misses_rather_than_serving_the_wrong_recording(tmp_path: Path) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))
    CassetteClient(_live_client(), name="shared", mode="record", root=tmp_path).complete(MESSAGES)

    player = CassetteClient(
        None, name="shared", mode="replay", root=tmp_path, request_builder=_openai_client()
    )
    with pytest.raises(BioflowError) as excinfo:
        player.complete(MESSAGES)
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND


def test_a_local_replay_needs_neither_credentials_nor_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The air-gapped case end to end: offline on, no key, no server running.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    builder = OpenAICompatibleClient(
        Settings(
            llm_provider="openai-compatible",
            llm_model="llama3.1:8b",
            llm_base_url="http://localhost:11434/v1",
            offline=True,
        )
    )
    directory = tmp_path / "airgap"
    directory.mkdir()
    (directory / f"{request_hash(builder.build_request(MESSAGES))}.json").write_text(
        json.dumps({"response": LLMResponse(text="offline answer").model_dump(mode="json")}),
        encoding="utf-8",
    )

    player = CassetteClient(
        None, name="airgap", mode="replay", root=tmp_path, request_builder=builder
    )
    assert player.complete(MESSAGES).text == "offline answer"
