"""Unit tests for the LLM adapter: request assembly, retries, pricing, prompts."""

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel, ValidationError
from tenacity import wait_none

from bioflow.adapters.llm.anthropic_client import (
    ANTHROPIC_VERSION,
    API_URL,
    OAUTH_BETA,
    AnthropicClient,
    parse_response,
    resolve_auth_headers,
)
from bioflow.adapters.llm.base import LLMResponse, StopReason, ToolCall, Usage
from bioflow.adapters.llm.fake_client import FakeLLMClient
from bioflow.adapters.llm.pricing import PRICES, cost_usd, is_priced, supports_sampling
from bioflow.config import Settings
from bioflow.core.models import BioflowError, ErrorCode
from bioflow.core.usage import UsageAccumulator
from bioflow.prompts.loader import available_prompts, load_prompt

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"llm_model": "claude-opus-5", "offline": False}
    base.update(overrides)
    return Settings(**base)


def _api_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    body.update(overrides)
    return body


MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------------------
# Request assembly
# --------------------------------------------------------------------------------------


def test_request_carries_the_required_fields() -> None:
    body = AnthropicClient(_settings()).build_request(MESSAGES)
    assert body["model"] == "claude-opus-5"
    assert body["messages"] == MESSAGES
    assert isinstance(body["max_tokens"], int)


def test_temperature_is_omitted_for_models_that_reject_it() -> None:
    # Sampling parameters were removed from the current frontier models; sending
    # temperature to claude-opus-5 is a 400, and it is BioFlow's default model.
    body = AnthropicClient(_settings(llm_model="claude-opus-5")).build_request(MESSAGES)
    assert "temperature" not in body


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-7"])
def test_temperature_omitted_across_the_frontier_models(model: str) -> None:
    assert "temperature" not in AnthropicClient(_settings(llm_model=model)).build_request(MESSAGES)
    assert not supports_sampling(model)


def test_temperature_is_sent_to_models_that_accept_it() -> None:
    client = AnthropicClient(_settings(llm_model="claude-sonnet-4-6", llm_temperature=0.0))
    body = client.build_request(MESSAGES)
    assert body["temperature"] == 0.0
    assert supports_sampling("claude-sonnet-4-6")


def test_tools_and_tool_choice_are_passed_through() -> None:
    tools = [{"name": "echo", "description": "d", "input_schema": {"type": "object"}}]
    body = AnthropicClient(_settings()).build_request(
        MESSAGES, tools=tools, tool_choice={"type": "auto"}
    )
    assert body["tools"] == tools
    assert body["tool_choice"] == {"type": "auto"}


def test_registry_schema_is_accepted_verbatim_as_tools() -> None:
    # Step 1.3 already emits the shape the Messages API wants; no translation needed.
    import bioflow.tools  # noqa: F401
    from bioflow.core.registry import registry

    body = AnthropicClient(_settings()).build_request(MESSAGES, tools=registry.to_llm_schema())
    tools = body["tools"]
    assert isinstance(tools, list)
    assert set(tools[0]) == {"name", "description", "input_schema"}


def test_system_prompt_is_included_only_when_given() -> None:
    client = AnthropicClient(_settings())
    assert "system" not in client.build_request(MESSAGES)
    assert client.build_request(MESSAGES, system="be terse")["system"] == "be terse"


def test_response_model_becomes_output_config() -> None:
    class Answer(BaseModel):
        verdict: str

    body = AnthropicClient(_settings()).build_request(MESSAGES, response_model=Answer)
    # output_config.format is current; the old output_format parameter is deprecated.
    config = body["output_config"]
    assert isinstance(config, dict)
    fmt = config["format"]
    assert isinstance(fmt, dict)
    assert fmt["type"] == "json_schema"


# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------


def test_api_key_uses_the_x_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_auth_headers() == {"x-api-key": "sk-test"}


def test_auth_token_uses_bearer_plus_the_oauth_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    # OAuth tokens are a different header shape, and /v1/messages needs the beta flag.
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-token")
    headers = resolve_auth_headers()
    assert headers["Authorization"] == "Bearer oauth-token"
    assert headers["anthropic-beta"] == OAUTH_BETA
    assert "x-api-key" not in headers


def test_both_credentials_set_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # The API rejects a request carrying both, so this fails early with an explanation.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-token")
    with pytest.raises(BioflowError) as excinfo:
        resolve_auth_headers()
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
    assert "both" in excinfo.value.error.message


def test_no_credentials_reports_how_to_get_some() -> None:
    with pytest.raises(BioflowError) as excinfo:
        resolve_auth_headers()
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT
    assert "ant auth login" in (excinfo.value.error.remediation or "")


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------


def test_text_response_is_parsed_and_priced() -> None:
    response = parse_response(_api_body(), latency_ms=42)
    assert response.text == "hello"
    assert response.stop_reason is StopReason.END_TURN
    assert response.model == "claude-opus-5"
    assert response.latency_ms == 42
    assert response.usage.input_tokens == 100
    assert response.usage.cost_usd == pytest.approx(100 * 5 / 1e6 + 20 * 25 / 1e6)
    assert response.usage.cost_is_estimated is False


def test_multiple_text_blocks_are_joined() -> None:
    body = _api_body(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert parse_response(body).text == "ab"


def test_tool_calls_are_parsed() -> None:
    body = _api_body(
        stop_reason="tool_use",
        content=[
            {"type": "text", "text": "let me look"},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "echo_artifact",
                "input": {"artifact_id": "art_0123456789ab"},
            },
        ],
    )
    response = parse_response(body)
    assert response.stop_reason is StopReason.TOOL_USE
    (call,) = response.tool_calls
    assert call.name == "echo_artifact"
    assert call.call_id == "toolu_abc"
    assert call.arguments == {"artifact_id": "art_0123456789ab"}


def test_refusal_is_surfaced_despite_a_200() -> None:
    # A safety decline arrives as HTTP 200; code checking only status would miss it.
    response = parse_response(_api_body(stop_reason="refusal", content=[]))
    assert response.stop_reason is StopReason.REFUSAL
    assert response.refused is True
    assert response.text is None


def test_an_unknown_stop_reason_does_not_lose_the_response() -> None:
    response = parse_response(_api_body(stop_reason="something_new"))
    assert response.stop_reason is StopReason.END_TURN
    assert response.text == "hello"


def test_an_unpriced_model_is_flagged_rather_than_raising() -> None:
    response = parse_response(_api_body(model="claude-experimental-9"))
    assert response.usage.cost_usd == 0.0
    assert response.usage.cost_is_estimated is True


def test_missing_usage_block_is_tolerated() -> None:
    body = _api_body()
    del body["usage"]
    assert parse_response(body).usage.input_tokens == 0


# --------------------------------------------------------------------------------------
# HTTP behaviour and retries
# --------------------------------------------------------------------------------------


@respx.mock
def test_a_successful_call_sends_the_expected_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json=_api_body()))

    response = AnthropicClient(_settings()).complete(MESSAGES)

    assert response.text == "hello"
    request = route.calls[0].request
    assert request.headers["anthropic-version"] == ANTHROPIC_VERSION
    assert request.headers["x-api-key"] == "sk-test"


@respx.mock
def test_a_429_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    route = respx.post(API_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"type": "rate_limit_error"}}),
            httpx.Response(200, json=_api_body()),
        ]
    )
    client = AnthropicClient(_settings(), max_attempts=3, retry_wait=wait_none())
    assert client.complete(MESSAGES).text == "hello"
    assert route.call_count == 2


@respx.mock
def test_a_529_overloaded_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    route = respx.post(API_URL).mock(
        side_effect=[
            httpx.Response(529, json={"error": {"type": "overloaded_error"}}),
            httpx.Response(200, json=_api_body()),
        ]
    )
    client = AnthropicClient(_settings(), max_attempts=3, retry_wait=wait_none())
    assert client.complete(MESSAGES).text == "hello"
    assert route.call_count == 2


@respx.mock
def test_retries_give_up_and_report_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom"}})
    )
    with pytest.raises(BioflowError) as excinfo:
        AnthropicClient(_settings(), max_attempts=2, retry_wait=wait_none()).complete(MESSAGES)
    assert route.call_count == 2
    assert "gave up after retries" in excinfo.value.error.message
    assert excinfo.value.error.retryable is True


@respx.mock
def test_a_400_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad model"}})
    )
    with pytest.raises(BioflowError) as excinfo:
        AnthropicClient(_settings(), max_attempts=3, retry_wait=wait_none()).complete(MESSAGES)
    # A malformed request stays malformed; retrying it only wastes time.
    assert route.call_count == 1
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
    assert "bad model" in excinfo.value.error.message


@respx.mock
def test_a_401_names_the_credential_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")
    respx.post(API_URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})
    )
    with pytest.raises(BioflowError) as excinfo:
        AnthropicClient(_settings(), max_attempts=2, retry_wait=wait_none()).complete(MESSAGES)
    assert "credentials" in (excinfo.value.error.remediation or "")


@respx.mock
def test_retry_after_header_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    slept: list[float] = []
    monkeypatch.setattr(
        "bioflow.adapters.llm.anthropic_client.time.sleep", lambda s: slept.append(s)
    )
    respx.post(API_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}, json={}),
            httpx.Response(200, json=_api_body()),
        ]
    )
    AnthropicClient(_settings(), max_attempts=3, retry_wait=wait_none()).complete(MESSAGES)
    assert 2.0 in slept


@respx.mock
def test_a_transport_error_becomes_a_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    respx.post(API_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(BioflowError) as excinfo:
        AnthropicClient(_settings(), max_attempts=2, retry_wait=wait_none()).complete(MESSAGES)
    # No raw httpx exception ever escapes the adapter.
    assert excinfo.value.error_code is ErrorCode.EXTERNAL_TOOL_FAILED
    assert excinfo.value.error.retryable is True


def test_offline_refuses_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(BioflowError) as excinfo:
        AnthropicClient(_settings(offline=True)).complete(MESSAGES)
    assert "offline" in excinfo.value.error.message


# --------------------------------------------------------------------------------------
# Pricing and usage
# --------------------------------------------------------------------------------------


def test_cost_arithmetic_is_per_million_tokens() -> None:
    assert cost_usd("claude-opus-5", 1_000_000, 0) == pytest.approx(5.0)
    assert cost_usd("claude-opus-5", 0, 1_000_000) == pytest.approx(25.0)
    assert cost_usd("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_zero_tokens_cost_nothing() -> None:
    assert cost_usd("claude-opus-5", 0, 0) == 0.0


def test_unknown_model_costs_nothing_and_is_not_priced() -> None:
    assert cost_usd("gpt-nonexistent", 1_000_000, 1_000_000) == 0.0
    assert is_priced("gpt-nonexistent") is False
    assert is_priced("claude-opus-5") is True


def test_the_default_model_is_priced() -> None:
    # A default model missing from the table would silently zero every run's cost.
    assert Settings().llm_model in PRICES


def test_usage_accumulates_across_calls() -> None:
    accumulator = UsageAccumulator()
    for _ in range(3):
        accumulator.record(parse_response(_api_body()))

    totals = accumulator.as_dict()
    assert totals["n_calls"] == 3
    assert totals["input_tokens"] == 300
    assert totals["output_tokens"] == 60
    assert totals["model"] == "claude-opus-5"
    assert isinstance(totals["cost_usd"], float)
    assert totals["cost_usd"] > 0


def test_usage_dict_has_the_manifest_shape() -> None:
    accumulator = UsageAccumulator()
    accumulator.record(parse_response(_api_body()))
    assert set(accumulator.as_dict()) == {
        "model",
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "n_calls",
    }


def test_usage_flags_an_estimated_total() -> None:
    accumulator = UsageAccumulator()
    accumulator.record(parse_response(_api_body(model="unknown-model")))
    assert accumulator.as_dict()["cost_is_estimated"] is True


def test_usage_records_the_prompt_version() -> None:
    accumulator = UsageAccumulator()
    accumulator.record(LLMResponse(model="claude-opus-5", prompt_version="v1"))
    assert accumulator.as_dict()["prompt_version"] == "v1"


def test_empty_usage_is_all_zeroes() -> None:
    totals = UsageAccumulator().as_dict()
    assert totals["n_calls"] == 0
    assert totals["cost_usd"] == 0.0
    assert totals["model"] is None


def test_run_context_carries_usage_into_the_manifest(tmp_path: Path) -> None:
    from bioflow.core.runctx import RunContext

    with RunContext.create(tmp_path / "workspace") as ctx:
        ctx.usage.record(parse_response(_api_body()))
        ctx.usage.record(parse_response(_api_body()))
        ctx.flush_usage()

        usage = ctx.manifest().llm_usage
        assert usage["n_calls"] == 2
        assert usage["model"] == "claude-opus-5"
        assert usage["input_tokens"] == 200


# --------------------------------------------------------------------------------------
# FakeLLMClient
# --------------------------------------------------------------------------------------


def test_fake_client_serves_its_script_in_order() -> None:
    first = LLMResponse(text="one")
    second = LLMResponse(text="two")
    client = FakeLLMClient([first, second])
    assert client.complete(MESSAGES) == first
    assert client.complete(MESSAGES) == second


def test_fake_client_records_what_it_was_asked() -> None:
    client = FakeLLMClient([LLMResponse(text="ok")])
    client.complete(MESSAGES, system="be terse")
    assert client.calls[0]["messages"] == MESSAGES
    assert client.calls[0]["system"] == "be terse"


def test_fake_client_errors_when_the_script_runs_out() -> None:
    # Repeating the last response would hide a test whose assumptions have drifted.
    client = FakeLLMClient([LLMResponse(text="only one")])
    client.complete(MESSAGES)
    with pytest.raises(BioflowError) as excinfo:
        client.complete(MESSAGES)
    assert "exhausted" in excinfo.value.error.message


def test_fake_client_satisfies_the_protocol() -> None:
    from bioflow.adapters.llm.base import LLMClient

    assert isinstance(FakeLLMClient(), LLMClient)
    assert isinstance(AnthropicClient(_settings()), LLMClient)


def test_push_appends_to_the_script() -> None:
    client = FakeLLMClient()
    client.push(LLMResponse(text="late"))
    assert client.complete(MESSAGES).text == "late"


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


def test_the_shipped_prompt_renders() -> None:
    prompt = load_prompt("assay_triage", reported_assay="RNA-Seq", organism="human")
    assert prompt.version == "v1"
    assert "RNA-Seq" in prompt.text
    assert "human" in prompt.text
    assert str(prompt) == prompt.text


def test_prompt_defaults_fill_in_absent_context() -> None:
    assert "not stated" in load_prompt("assay_triage").text


def test_prompt_loop_renders_a_list() -> None:
    text = load_prompt("assay_triage", metadata_fields=["library_layout", "instrument"]).text
    assert "library_layout" in text
    assert "instrument" in text


def test_available_prompts_lists_the_version() -> None:
    assert "assay_triage" in available_prompts("v1")
    assert available_prompts("v99") == []


def test_a_missing_prompt_names_the_alternatives() -> None:
    with pytest.raises(BioflowError) as excinfo:
        load_prompt("no_such_prompt")
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND
    assert "assay_triage" in (excinfo.value.error.remediation or "")


def test_an_undefined_variable_fails_loudly(tmp_path: Path) -> None:
    # StrictUndefined: a typo must not render as an empty string inside a prompt.
    version_dir = tmp_path / "v1"
    version_dir.mkdir(parents=True)
    (version_dir / "strict.md").write_text("Hello {{ required_name }}", encoding="utf-8")
    with pytest.raises(BioflowError) as excinfo:
        load_prompt("strict", root=tmp_path)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED


@pytest.mark.parametrize("name", ["../../../etc/passwd", "../escape", "/etc/passwd"])
def test_prompt_names_cannot_escape_the_prompt_tree(name: str) -> None:
    with pytest.raises(BioflowError) as excinfo:
        load_prompt(name)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED


def test_prompt_text_is_not_inlined_anywhere_in_code() -> None:
    # CLAUDE.md: prompts live in files and are versioned; never inline prompt text.
    source = Path(__file__).resolve().parents[2] / "src" / "bioflow"
    offenders = [
        path.name for path in source.rglob("*.py") if "You are " in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_tool_call_and_usage_models_are_frozen() -> None:
    call = ToolCall(name="echo", call_id="toolu_1")
    with pytest.raises(ValidationError, match="frozen"):
        call.name = "other"  # type: ignore[misc]
    assert Usage().input_tokens == 0
