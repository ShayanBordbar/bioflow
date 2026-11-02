"""Anthropic Messages API client over raw httpx.

No vendor SDK by request. The wire format below follows the Messages API reference:
``POST /v1/messages`` with ``anthropic-version: 2023-06-01``.

Two things here are less obvious than they look:

* ``temperature`` is sent **conditionally**. Sampling parameters were removed from the
  current frontier models, including BioFlow's default, where sending one is a 400.
* An API key and an auth token are not interchangeable header shapes, and sending both
  is rejected by the API -- so this refuses that combination itself, with a message that
  says what to do, rather than surfacing an opaque 401.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel
from tenacity import wait_exponential_jitter
from tenacity.wait import WaitBaseT

from bioflow.adapters.llm._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    post_json,
)
from bioflow.adapters.llm.base import LLMResponse, StopReason, ToolCall, Usage
from bioflow.adapters.llm.pricing import cost_usd, is_priced, supports_sampling
from bioflow.adapters.llm.tool_schema import to_anthropic_tools
from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioflow.config import Settings

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# OAuth tokens need this on /v1/messages; an API key does not.
OAUTH_BETA = "oauth-2025-04-20"

_RAISED_BY = "bioflow.adapters.llm.anthropic_client"


def _fail(
    code: ErrorCode,
    message: str,
    *,
    remediation: str | None = None,
    details: dict[str, JSONValue] | None = None,
    retryable: bool = False,
) -> BioflowError:
    return BioflowError(
        StructuredError(
            error_code=code,
            message=message,
            remediation=remediation,
            details=details or {},
            retryable=retryable,
            raised_by=_RAISED_BY,
        )
    )


def resolve_auth_headers(
    api_key: str | None = None, auth_token: str | None = None
) -> dict[str, str]:
    """Build the credential headers, or explain why we cannot.

    ``ANTHROPIC_API_KEY`` goes in ``x-api-key``; ``ANTHROPIC_AUTH_TOKEN`` goes in
    ``Authorization: Bearer`` and additionally needs the OAuth beta header. Setting both
    makes the API reject the request, so that is caught here instead.
    """
    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
    token = auth_token if auth_token is not None else os.environ.get("ANTHROPIC_AUTH_TOKEN")

    if key and token:
        raise _fail(
            ErrorCode.VALIDATION_FAILED,
            "both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are set",
            remediation="Unset one of them; the API rejects a request carrying both.",
        )
    if key:
        return {"x-api-key": key}
    if token:
        return {"Authorization": f"Bearer {token}", "anthropic-beta": OAUTH_BETA}
    raise _fail(
        ErrorCode.MISSING_INPUT,
        "no Anthropic credentials found",
        remediation="Set ANTHROPIC_API_KEY, or run `ant auth login` and export "
        "ANTHROPIC_AUTH_TOKEN. `bioflow doctor` reports what is available.",
    )


class AnthropicClient:
    """An :class:`~bioflow.adapters.llm.base.LLMClient` over the Messages API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_wait: WaitBaseT | None = None,
    ) -> None:
        from bioflow.config import get_settings

        self.settings = settings if settings is not None else get_settings()
        self.model = self.settings.llm_model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.timeout = timeout
        # Injectable so tests can exercise the retry path without actually sleeping
        # through the backoff.
        self.retry_wait: WaitBaseT = (
            retry_wait
            if retry_wait is not None
            else wait_exponential_jitter(initial=1.0, max=30.0, jitter=1.0)
        )
        self._client = client
        self._owns_client = client is None

    # -- request assembly -----------------------------------------------------------

    def build_request(
        self,
        messages: Sequence[dict[str, JSONValue]],
        *,
        tools: Sequence[dict[str, JSONValue]] | None = None,
        tool_choice: dict[str, JSONValue] | None = None,
        response_model: type[BaseModel] | None = None,
        system: str | None = None,
    ) -> dict[str, JSONValue]:
        """Assemble the JSON body for a completion request."""
        body: dict[str, JSONValue] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": list(messages),
        }
        if system is not None:
            body["system"] = system
        if tools:
            translated: list[JSONValue] = list(to_anthropic_tools(tools))
            body["tools"] = translated
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if supports_sampling(self.model):
            # Omitted on models that removed sampling parameters, where sending it is a
            # 400 rather than a no-op.
            body["temperature"] = self.settings.llm_temperature
        if response_model is not None:
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": response_model.model_json_schema(),
                }
            }
        return body

    # -- the call -------------------------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict[str, JSONValue]],
        *,
        tools: Sequence[dict[str, JSONValue]] | None = None,
        tool_choice: dict[str, JSONValue] | None = None,
        response_model: type[BaseModel] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a completion request, retrying transient failures."""
        if self.settings.offline:
            # api.anthropic.com is always egress, so offline always refuses here. The
            # openai-compatible client permits a loopback or private-network endpoint,
            # which is how air-gapped operation works.
            raise _fail(
                ErrorCode.VALIDATION_FAILED,
                "refusing to call the Anthropic API: offline mode permits no egress",
                remediation=(
                    "Replay from a cassette, unset BIOFLOW_OFFLINE, or point "
                    "BIOFLOW_LLM_PROVIDER at a locally served open-weight model."
                ),
            )

        body = self.build_request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            response_model=response_model,
            system=system,
        )
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            **resolve_auth_headers(),
        }

        started = time.monotonic()
        payload = self._post_with_retries(body, headers)
        latency_ms = int((time.monotonic() - started) * 1000)
        return parse_response(payload, latency_ms=latency_ms)

    def _post_with_retries(
        self, body: dict[str, JSONValue], headers: dict[str, str]
    ) -> dict[str, Any]:
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            return post_json(
                client=client,
                url=API_URL,
                body=body,
                headers=headers,
                max_attempts=self.max_attempts,
                retry_wait=self.retry_wait,
                raised_by=_RAISED_BY,
                endpoint_label="the Anthropic API",
            )
        finally:
            if self._owns_client:
                client.close()


def parse_response(payload: dict[str, Any], *, latency_ms: int = 0) -> LLMResponse:
    """Turn a Messages API response body into an :class:`LLMResponse`."""
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in payload.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=str(block.get("name", "")),
                    arguments=dict(block.get("input", {}) or {}),
                    call_id=str(block.get("id", "")),
                )
            )

    raw_usage = payload.get("usage", {}) or {}
    input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
    output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    model = str(payload.get("model", ""))

    raw_stop = payload.get("stop_reason") or StopReason.END_TURN.value
    try:
        stop_reason = StopReason(str(raw_stop))
    except ValueError:
        # An unrecognized stop reason is a new API value, not a reason to lose the
        # response; treat it as a normal end of turn.
        stop_reason = StopReason.END_TURN

    return LLMResponse(
        text="".join(texts) or None,
        tool_calls=tuple(tool_calls),
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(model, input_tokens, output_tokens),
            cost_is_estimated=not is_priced(model),
        ),
        model=model,
        latency_ms=latency_ms,
    )
