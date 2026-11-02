"""Unit tests for the OpenAI-compatible client and the egress guard."""

from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel
from tenacity import wait_none

from bioflow.adapters.llm.base import LLMClient, StopReason
from bioflow.adapters.llm.openai_client import OpenAICompatibleClient
from bioflow.config import Settings
from bioflow.core.models import BioflowError, ErrorCode
from bioflow.core.net import endpoint_host, is_local_endpoint

pytestmark = pytest.mark.unit

LOCAL_URL = "http://localhost:11434/v1"
REMOTE_URL = "https://api.together.xyz/v1"
MESSAGES = [{"role": "user", "content": "classify this dataset"}]


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BIOFLOW_LLM_API_KEY", raising=False)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_provider": "openai-compatible",
        "llm_model": "llama3.1:8b",
        "llm_base_url": LOCAL_URL,
        "offline": False,
    }
    base.update(overrides)
    return Settings(**base)


def _client(**overrides: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(_settings(**overrides), retry_wait=wait_none())


def _api_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "model": "llama3.1:8b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "rna-seq"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 8},
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------------------
# The egress guard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
        "http://10.0.0.5:8000/v1",
        "http://192.168.1.10:8000/v1",
        "http://172.16.0.1:8000/v1",
        "http://169.254.10.1:8000/v1",
        "http://gpu-node.local:8000/v1",
        "http://box.internal:8000/v1",
    ],
)
def test_local_endpoints_are_recognized(url: str) -> None:
    assert is_local_endpoint(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://api.together.xyz/v1",
        "https://api.anthropic.com/v1",
        "http://8.8.8.8/v1",
        "http://example.com/v1",
        # A hostname we cannot classify must be assumed to leave the building.
        "http://my-gpu-box/v1",
        "not a url at all",
        "",
    ],
)
def test_non_local_endpoints_fail_closed(url: str) -> None:
    assert is_local_endpoint(url) is False


def test_endpoint_host_extraction() -> None:
    assert endpoint_host("http://localhost:11434/v1") == "localhost"
    assert endpoint_host("HTTP://LocalHost:1/v1") == "localhost"
    assert endpoint_host("") is None


def test_offline_permits_a_local_endpoint() -> None:
    # offline means "no egress", not "no LLM": this is what makes air-gapped operation
    # with a self-hosted model possible.
    with respx.mock:
        respx.post(f"{LOCAL_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_api_body())
        )
        assert _client(offline=True).complete(MESSAGES).text == "rna-seq"


def test_offline_refuses_a_remote_endpoint() -> None:
    with pytest.raises(BioflowError) as excinfo:
        _client(offline=True, llm_base_url=REMOTE_URL).complete(MESSAGES)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
    assert "offline mode permits only local endpoints" in excinfo.value.error.message


def test_offline_refusal_makes_no_request() -> None:
    # No respx mock at all: reaching the transport would raise something else entirely.
    with pytest.raises(BioflowError):
        _client(offline=True, llm_base_url=REMOTE_URL).complete(MESSAGES)


def test_anthropic_always_refuses_under_offline() -> None:
    from bioflow.adapters.llm.anthropic_client import AnthropicClient

    settings = Settings(llm_provider="anthropic", llm_model="claude-opus-5", offline=True)
    with pytest.raises(BioflowError) as excinfo:
        AnthropicClient(settings).complete(MESSAGES)
    assert "no egress" in excinfo.value.error.message


# --------------------------------------------------------------------------------------
# Construction and request assembly
# --------------------------------------------------------------------------------------


def test_a_base_url_is_required() -> None:
    with pytest.raises(BioflowError) as excinfo:
        OpenAICompatibleClient(Settings(llm_provider="openai-compatible"))
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT


def test_urls_are_built_from_the_base() -> None:
    client = _client(llm_base_url="http://localhost:8000/v1/")
    assert client.completions_url == "http://localhost:8000/v1/chat/completions"
    assert client.models_url == "http://localhost:8000/v1/models"


def test_the_system_prompt_becomes_a_message() -> None:
    # Unlike Anthropic, where system is a top-level parameter.
    body = _client().build_request(MESSAGES, system="be terse")
    messages = body["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == MESSAGES[0]
    assert "system" not in body


def test_no_system_message_when_none_is_given() -> None:
    messages = _client().build_request(MESSAGES)["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1


def test_temperature_is_always_sent() -> None:
    # The removal that forced conditional sampling on Claude is Anthropic-specific.
    body = _client(llm_temperature=0.3).build_request(MESSAGES)
    assert body["temperature"] == 0.3


def test_tools_are_translated_into_the_function_envelope() -> None:
    neutral = [{"name": "echo", "description": "d", "input_schema": {"type": "object"}}]
    body = _client().build_request(MESSAGES, tools=neutral, tool_choice={"type": "any"})
    tools = body["tools"]
    assert isinstance(tools, list)
    assert tools[0]["type"] == "function"
    assert body["tool_choice"] == "required"


def test_response_model_becomes_response_format() -> None:
    class Answer(BaseModel):
        verdict: str

    body = _client().build_request(MESSAGES, response_model=Answer)
    fmt = body["response_format"]
    assert isinstance(fmt, dict)
    assert fmt["type"] == "json_schema"
    schema = fmt["json_schema"]
    assert isinstance(schema, dict)
    assert schema["name"] == "Answer"


def test_no_auth_header_without_a_key() -> None:
    # A local Ollama needs no credential; sending an empty Bearer would be worse.
    assert _client().auth_headers() == {}


def test_auth_header_from_the_bioflow_setting() -> None:
    assert _client(llm_api_key="sk-local").auth_headers() == {"Authorization": "Bearer sk-local"}


def test_auth_header_falls_back_to_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert _client().auth_headers() == {"Authorization": "Bearer sk-env"}


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------


def test_text_response_is_parsed() -> None:
    response = _client().parse_response(_api_body(), latency_ms=12)
    assert response.text == "rna-seq"
    assert response.stop_reason is StopReason.END_TURN
    assert response.model == "llama3.1:8b"
    assert response.latency_ms == 12


def test_usage_is_read_from_prompt_and_completion_tokens() -> None:
    # Different field names from Anthropic's input_tokens / output_tokens.
    usage = _client().parse_response(_api_body()).usage
    assert usage.input_tokens == 120
    assert usage.output_tokens == 8


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("tool_calls", StopReason.TOOL_USE),
        ("function_call", StopReason.TOOL_USE),
        ("content_filter", StopReason.REFUSAL),
        ("something_new", StopReason.END_TURN),
    ],
)
def test_finish_reason_maps_onto_stop_reason(finish_reason: str, expected: StopReason) -> None:
    body = _api_body(choices=[{"message": {"content": "x"}, "finish_reason": finish_reason}])
    assert _client().parse_response(body).stop_reason is expected


def test_content_filter_is_surfaced_as_a_refusal() -> None:
    body = _api_body(choices=[{"message": {"content": None}, "finish_reason": "content_filter"}])
    assert _client().parse_response(body).refused is True


def test_tool_call_arguments_are_decoded_from_a_json_string() -> None:
    # The headline format difference: arguments arrive as a string, not an object.
    body = _api_body(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "echo_artifact",
                                "arguments": '{"artifact_id": "art_0123456789ab"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    response = _client().parse_response(body)
    (call,) = response.tool_calls
    assert call.name == "echo_artifact"
    assert call.call_id == "call_1"
    assert call.arguments == {"artifact_id": "art_0123456789ab"}
    assert isinstance(call.arguments, dict)


def test_tool_call_arguments_already_an_object_are_accepted() -> None:
    # Some servers ignore the spec and send an object; round-tripping it would be silly.
    body = _api_body(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "echo", "arguments": {"a": 1}}}
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    (call,) = _client().parse_response(body).tool_calls
    assert call.arguments == {"a": 1}


def test_malformed_tool_arguments_are_a_structured_error() -> None:
    # A small open-weight model emitting bad JSON is routine, so this has to be
    # something an agent loop can read back and retry, not a crash.
    body = _api_body(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "echo", "arguments": "{not json"}}
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    with pytest.raises(BioflowError) as excinfo:
        _client().parse_response(body)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
    assert "not valid JSON" in excinfo.value.error.message
    assert excinfo.value.error.details["tool"] == "echo"


def test_tool_arguments_decoding_to_a_non_object_are_rejected() -> None:
    body = _api_body(
        choices=[
            {
                "message": {
                    "tool_calls": [{"id": "c1", "function": {"name": "e", "arguments": "[1,2]"}}]
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    with pytest.raises(BioflowError) as excinfo:
        _client().parse_response(body)
    assert "not an object" in excinfo.value.error.message


def test_an_empty_choices_list_is_tolerated() -> None:
    response = _client().parse_response(_api_body(choices=[]))
    assert response.text is None
    assert response.stop_reason is StopReason.END_TURN


# --------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------


def test_a_local_model_is_free_not_unknown() -> None:
    # The distinction ADR-0001 set out to fix: self-hosted inference genuinely costs
    # nothing, which is not the same as a price we failed to look up.
    usage = _client().parse_response(_api_body()).usage
    assert usage.cost_usd == 0.0
    assert usage.cost_is_estimated is False


def test_configured_prices_are_applied() -> None:
    client = _client(
        llm_input_price_per_mtok=0.5,
        llm_output_price_per_mtok=1.5,
    )
    usage = client.parse_response(
        _api_body(usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    ).usage
    assert usage.cost_usd == pytest.approx(2.0)
    assert usage.cost_is_estimated is False


def test_a_remote_endpoint_without_prices_is_flagged_as_estimated() -> None:
    client = _client(llm_base_url=REMOTE_URL)
    usage = client.parse_response(_api_body(model="mixtral-8x7b")).usage
    assert usage.cost_usd == 0.0
    # Remote and unpriced: we genuinely do not know what this cost.
    assert usage.cost_is_estimated is True


def test_only_the_output_price_configured_is_honoured() -> None:
    client = _client(llm_output_price_per_mtok=2.0)
    usage = client.parse_response(
        _api_body(usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    ).usage
    assert usage.cost_usd == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# HTTP behaviour
# --------------------------------------------------------------------------------------


@respx.mock
def test_a_successful_call_posts_to_chat_completions() -> None:
    route = respx.post(f"{LOCAL_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_api_body())
    )
    assert _client().complete(MESSAGES).text == "rna-seq"
    assert route.call_count == 1
    assert "Authorization" not in route.calls[0].request.headers


@respx.mock
def test_the_key_is_sent_when_configured() -> None:
    route = respx.post(f"{LOCAL_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_api_body())
    )
    _client(llm_api_key="sk-local").complete(MESSAGES)
    assert route.calls[0].request.headers["Authorization"] == "Bearer sk-local"


@respx.mock
def test_a_429_is_retried() -> None:
    route = respx.post(f"{LOCAL_URL}/chat/completions").mock(
        side_effect=[httpx.Response(429, json={}), httpx.Response(200, json=_api_body())]
    )
    assert _client().complete(MESSAGES).text == "rna-seq"
    assert route.call_count == 2


@respx.mock
def test_a_400_is_not_retried() -> None:
    route = respx.post(f"{LOCAL_URL}/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "no such model"}})
    )
    with pytest.raises(BioflowError) as excinfo:
        _client().complete(MESSAGES)
    assert route.call_count == 1
    assert "no such model" in excinfo.value.error.message


@respx.mock
def test_a_dead_server_becomes_a_structured_error() -> None:
    respx.post(f"{LOCAL_URL}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(BioflowError) as excinfo:
        _client().complete(MESSAGES)
    assert excinfo.value.error_code is ErrorCode.EXTERNAL_TOOL_FAILED
    assert LOCAL_URL in excinfo.value.error.message
    assert excinfo.value.error.retryable is True


def test_the_client_satisfies_the_protocol() -> None:
    assert isinstance(_client(), LLMClient)


# --------------------------------------------------------------------------------------
# The explicit local-host allowlist
# --------------------------------------------------------------------------------------


def test_a_bare_hostname_is_remote_until_allowlisted() -> None:
    # The default fails closed, which is what makes the allowlist an explicit assertion
    # rather than a formality.
    assert is_local_endpoint("http://gpu-01:8000/v1") is False
    assert is_local_endpoint("http://gpu-01:8000/v1", ("gpu-01",)) is True


@pytest.mark.parametrize(
    ("host", "patterns", "expected"),
    [
        ("http://gpu-01:8000/v1", ("gpu-01",), True),
        ("http://gpu-01:8000/v1", ("gpu-02",), False),
        ("http://GPU-01:8000/v1", ("gpu-01",), True),
        ("http://gpu-01:8000/v1", (" gpu-01 ",), True),
        # Suffix forms, so a cluster need not list every node. (.internal and .local
        # are already built-in local suffixes, so a real public-ish zone is used here.)
        ("http://gpu-07.hpc.uni.edu:8000/v1", (".hpc.uni.edu",), True),
        ("http://gpu-07.hpc.uni.edu:8000/v1", ("*.hpc.uni.edu",), True),
        ("http://hpc.uni.edu:8000/v1", (".hpc.uni.edu",), True),
        ("http://gpu-07.other.uni.edu:8000/v1", (".hpc.uni.edu",), False),
        # A suffix must not match a host that merely ends in the same characters.
        ("http://evilhpc.uni.edu:8000/v1", (".hpc.uni.edu",), False),
        ("http://api.together.xyz/v1", ("gpu-01",), False),
        ("http://gpu-01:8000/v1", ("",), False),
    ],
)
def test_allowlist_matching(host: str, patterns: tuple[str, ...], expected: bool) -> None:
    assert is_local_endpoint(host, patterns) is expected


def test_allowlisting_does_not_widen_anything_else() -> None:
    # Naming one host must not make unrelated public endpoints local.
    allowed = ("gpu-01", ".cluster.internal")
    assert is_local_endpoint("https://api.together.xyz/v1", allowed) is False
    assert is_local_endpoint("http://8.8.8.8/v1", allowed) is False


def test_offline_permits_an_allowlisted_host() -> None:
    with respx.mock:
        respx.post("http://gpu-01:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_api_body())
        )
        client = _client(
            offline=True,
            llm_base_url="http://gpu-01:8000/v1",
            llm_local_hosts=("gpu-01",),
        )
        assert client.complete(MESSAGES).text == "rna-seq"


def test_offline_still_refuses_a_host_not_on_the_allowlist() -> None:
    client = _client(
        offline=True,
        llm_base_url="http://gpu-02:8000/v1",
        llm_local_hosts=("gpu-01",),
    )
    with pytest.raises(BioflowError) as excinfo:
        client.complete(MESSAGES)
    assert "offline mode permits only local endpoints" in excinfo.value.error.message


def test_an_allowlisted_host_counts_as_self_hosted_for_cost() -> None:
    # An allowlisted host is the operator's own infrastructure, so its inference is
    # free rather than of unknown price.
    client = _client(llm_base_url="http://gpu-01:8000/v1", llm_local_hosts=("gpu-01",))
    usage = client.parse_response(_api_body(model="llama3.1:70b")).usage
    assert usage.cost_usd == 0.0
    assert usage.cost_is_estimated is False


def test_a_non_allowlisted_remote_still_reports_unknown_cost() -> None:
    client = _client(llm_base_url="http://gpu-02:8000/v1", llm_local_hosts=("gpu-01",))
    assert client.parse_response(_api_body(model="llama3.1:70b")).usage.cost_is_estimated is True
