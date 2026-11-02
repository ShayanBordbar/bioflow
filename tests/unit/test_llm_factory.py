"""Unit tests for LLM client construction.

The point of the factory is that `llm_provider` and `cassette_mode` are real: before it
existed both settings were declared and read by nothing.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from bioflow.adapters.llm import PLANNED_PROVIDERS, SUPPORTED_PROVIDERS, get_llm_client
from bioflow.adapters.llm.anthropic_client import AnthropicClient
from bioflow.adapters.llm.base import LLMClient, LLMResponse
from bioflow.adapters.llm.cassette import CassetteClient, request_hash
from bioflow.adapters.llm.factory import PROVIDER_ALIASES
from bioflow.adapters.llm.openai_client import OpenAICompatibleClient
from bioflow.config import Settings
from bioflow.core.models import BioflowError, ErrorCode

pytestmark = pytest.mark.unit

MESSAGES = [{"role": "user", "content": "hi"}]


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"llm_model": "claude-opus-5", "offline": False}
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------------------
# Provider dispatch
# --------------------------------------------------------------------------------------


def test_the_default_provider_builds_an_anthropic_client() -> None:
    client = get_llm_client(_settings())
    assert isinstance(client, AnthropicClient)
    assert isinstance(client, LLMClient)


def test_provider_matching_ignores_case_and_padding() -> None:
    assert isinstance(get_llm_client(_settings(llm_provider="  Anthropic ")), AnthropicClient)


def test_the_openai_compatible_provider_builds_its_client() -> None:
    client = get_llm_client(
        _settings(llm_provider="openai-compatible", llm_base_url="http://localhost:11434/v1")
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert isinstance(client, LLMClient)


@pytest.mark.parametrize("alias", ["ollama", "vllm", "openai", "local", "llama-cpp", "tgi"])
def test_familiar_aliases_reach_the_openai_compatible_client(alias: str) -> None:
    # They differ only by base_url, so sending them all to one client is the point --
    # and someone typing "ollama" should not be told it is an unknown provider.
    client = get_llm_client(_settings(llm_provider=alias, llm_base_url="http://localhost:11434/v1"))
    assert isinstance(client, OpenAICompatibleClient)


def test_openai_compatible_without_a_base_url_is_refused() -> None:
    with pytest.raises(BioflowError) as excinfo:
        get_llm_client(_settings(llm_provider="openai-compatible"))
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT
    assert "BIOFLOW_LLM_BASE_URL" in excinfo.value.error.message
    # The remediation gives values someone can paste.
    assert "11434" in (excinfo.value.error.remediation or "")


def test_a_base_url_with_anthropic_is_refused() -> None:
    # Ignoring it would let someone believe they had redirected BioFlow at their own
    # endpoint while it kept calling api.anthropic.com.
    with pytest.raises(BioflowError) as excinfo:
        get_llm_client(_settings(llm_provider="anthropic", llm_base_url="http://localhost:8000/v1"))
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
    assert "fixed endpoint" in excinfo.value.error.message


def test_an_unknown_provider_lists_what_is_supported() -> None:
    with pytest.raises(BioflowError) as excinfo:
        get_llm_client(_settings(llm_provider="not-a-provider"))

    error = excinfo.value.error
    assert "unknown LLM provider" in error.message
    assert "anthropic" in (error.remediation or "")
    assert "openai-compatible" in (error.remediation or "")
    assert error.details["configured"] == "not-a-provider"
    assert error.details["supported"] == ["anthropic", "openai-compatible"]


def test_an_unsupported_provider_fails_rather_than_being_ignored() -> None:
    # The regression this guards: llm_provider used to be declared and read by nothing,
    # so setting it to anything at all silently produced an Anthropic client.
    with pytest.raises(BioflowError):
        get_llm_client(_settings(llm_provider="gpt4all"))


def test_the_provider_sets_agree_with_each_other() -> None:
    assert {"anthropic", "openai-compatible"} == SUPPORTED_PROVIDERS
    assert not (SUPPORTED_PROVIDERS & set(PLANNED_PROVIDERS))
    # Every alias points at something real.
    assert set(PROVIDER_ALIASES.values()) <= SUPPORTED_PROVIDERS


def test_client_kwargs_reach_the_provider_client() -> None:
    client = get_llm_client(_settings(), max_attempts=7)
    assert isinstance(client, AnthropicClient)
    assert client.max_attempts == 7


def test_settings_default_to_the_loaded_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOFLOW_LLM_MODEL", "claude-sonnet-4-6")
    from bioflow.config import reset_settings_cache

    reset_settings_cache()
    try:
        client = get_llm_client()
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-6"
    finally:
        reset_settings_cache()


# --------------------------------------------------------------------------------------
# Cassette wiring
# --------------------------------------------------------------------------------------


def test_cassette_mode_off_returns_a_bare_client() -> None:
    client = get_llm_client(_settings(cassette_mode="off"), cassette="triage")
    assert isinstance(client, AnthropicClient)


def test_no_cassette_name_returns_a_bare_client() -> None:
    assert isinstance(get_llm_client(_settings(cassette_mode="replay")), AnthropicClient)


@pytest.mark.parametrize("mode", ["record", "replay"])
def test_a_named_cassette_wraps_the_client(mode: str, tmp_path: Path) -> None:
    client = get_llm_client(
        _settings(cassette_mode=mode), cassette="triage", cassette_root=tmp_path
    )
    assert isinstance(client, CassetteClient)
    assert client.mode == mode
    assert client.name == "triage"
    assert client.directory == tmp_path / "triage"


def test_replay_needs_no_credentials_and_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No API key, no respx transport: if replay touched either, this would fail. That is
    # the property that lets the contract tier run in CI.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    builder = AnthropicClient(_settings())
    directory = tmp_path / "triage"
    directory.mkdir()
    (directory / f"{request_hash(builder.build_request(MESSAGES))}.json").write_text(
        json.dumps({"response": LLMResponse(text="from disk").model_dump(mode="json")}),
        encoding="utf-8",
    )

    client = get_llm_client(
        _settings(cassette_mode="replay"), cassette="triage", cassette_root=tmp_path
    )
    assert client.complete(MESSAGES).text == "from disk"


def test_the_wrapper_hashes_with_the_real_request_builder(tmp_path: Path) -> None:
    # The cassette key has to come from the provider's actual request body, not from the
    # call arguments, or recordings would not survive a change in request assembly.
    client = get_llm_client(
        _settings(cassette_mode="replay"), cassette="triage", cassette_root=tmp_path
    )
    assert isinstance(client, CassetteClient)
    assert isinstance(client._builder, AnthropicClient)


def test_a_cassette_wrapped_client_still_satisfies_the_protocol(tmp_path: Path) -> None:
    client = get_llm_client(
        _settings(cassette_mode="record"), cassette="triage", cassette_root=tmp_path
    )
    assert isinstance(client, LLMClient)


def test_provider_validation_happens_before_cassette_wrapping(tmp_path: Path) -> None:
    with pytest.raises(BioflowError) as excinfo:
        get_llm_client(
            _settings(llm_provider="gpt4all", cassette_mode="replay"),
            cassette="triage",
            cassette_root=tmp_path,
        )
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
