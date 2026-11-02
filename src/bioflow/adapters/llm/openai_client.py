"""OpenAI-compatible chat completions client.

One client covers self-hosted vLLM, Ollama, llama.cpp, and TGI as well as hosted
OpenAI-compatible gateways, because they all expose the same ``/v1/chat/completions``
surface. That is the point: open-weight model support without a client per vendor.

Structured deliberately in parallel with :mod:`bioflow.adapters.llm.anthropic_client` --
same ``build_request`` / ``complete`` split, same retry policy, same error discipline --
so :class:`~bioflow.adapters.llm.cassette.CassetteClient` wraps it unchanged and the two
read side by side.

Four things genuinely differ from the Anthropic path, and each is commented where it
happens: the system prompt is a message, tool-call arguments arrive as a JSON *string*,
the stop reason is called ``finish_reason`` with different values, and ``temperature`` is
both accepted and expected.
"""

from __future__ import annotations

import json
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
from bioflow.adapters.llm.pricing import configured_rates, cost_usd, is_priced
from bioflow.adapters.llm.tool_schema import to_openai_tool_choice, to_openai_tools
from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError
from bioflow.core.net import is_local_endpoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioflow.config import Settings

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"

# OpenAI-compatible finish reasons. Anything unrecognized is treated as a normal end of
# turn rather than discarded -- a new server value must not lose a good response.
FINISH_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "content_filter": StopReason.REFUSAL,
}

_RAISED_BY = "bioflow.adapters.llm.openai_client"


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


class OpenAICompatibleClient:
    """An :class:`~bioflow.adapters.llm.base.LLMClient` over ``/v1/chat/completions``."""

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
        if not self.settings.llm_base_url:
            raise _fail(
                ErrorCode.MISSING_INPUT,
                "an OpenAI-compatible provider needs BIOFLOW_LLM_BASE_URL",
                remediation=(
                    "Set it to your server's OpenAI-compatible root, for example "
                    "http://localhost:11434/v1 (Ollama) or http://localhost:8000/v1 (vLLM)."
                ),
            )
        self.base_url = self.settings.llm_base_url.rstrip("/")
        self.model = self.settings.llm_model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.retry_wait: WaitBaseT = (
            retry_wait
            if retry_wait is not None
            else wait_exponential_jitter(initial=1.0, max=30.0, jitter=1.0)
        )
        self._client = client
        self._owns_client = client is None

    def _is_local(self) -> bool:
        """Whether this endpoint is inside the perimeter, allowlist included."""
        return is_local_endpoint(self.base_url, self.settings.llm_local_hosts)

    @property
    def completions_url(self) -> str:
        """The chat completions endpoint for this server."""
        return f"{self.base_url}{CHAT_COMPLETIONS_PATH}"

    @property
    def models_url(self) -> str:
        """The model-listing endpoint, used by ``bioflow doctor``."""
        return f"{self.base_url}{MODELS_PATH}"

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
        """Assemble the JSON body for a chat completions request."""
        # Unlike Anthropic, the system prompt is a message rather than a parameter.
        full_messages: list[JSONValue] = []
        if system is not None:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(dict(message) for message in messages)

        body: dict[str, JSONValue] = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            # Always sent here. The removal that forced conditional sampling on Claude is
            # Anthropic-specific; OpenAI-compatible servers expect this parameter, so
            # llm_temperature finally means what it says on this path.
            "temperature": self.settings.llm_temperature,
        }
        if tools:
            translated: list[JSONValue] = list(to_openai_tools(tools))
            body["tools"] = translated
            choice = to_openai_tool_choice(tool_choice)
            if choice is not None:
                body["tool_choice"] = choice
        if response_model is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }
        return body

    def auth_headers(self) -> dict[str, str]:
        """Return the credential header, or none at all.

        A local Ollama or an unauthenticated vLLM needs no credential, so an unset key
        means the header is simply omitted rather than an error.
        """
        key = self.settings.llm_api_key
        if not key:
            import os

            key = os.environ.get("OPENAI_API_KEY")
        return {"Authorization": f"Bearer {key}"} if key else {}

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
        """Send a chat completion request, retrying transient failures."""
        if self.settings.offline and not self._is_local():
            # offline means no egress, not no LLM: a model served on this machine or
            # this private network is still allowed.
            raise _fail(
                ErrorCode.VALIDATION_FAILED,
                f"refusing to call {self.base_url}: offline mode permits only local endpoints",
                remediation=(
                    "Point BIOFLOW_LLM_BASE_URL at a loopback or private-network address, "
                    "or unset BIOFLOW_OFFLINE."
                ),
                details={"base_url": self.base_url},
            )

        body = self.build_request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            response_model=response_model,
            system=system,
        )
        headers = {"content-type": "application/json", **self.auth_headers()}

        started = time.monotonic()
        payload = self._post_with_retries(body, headers)
        latency_ms = int((time.monotonic() - started) * 1000)
        return self.parse_response(payload, latency_ms=latency_ms)

    def _post_with_retries(
        self, body: dict[str, JSONValue], headers: dict[str, str]
    ) -> dict[str, Any]:
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            return post_json(
                client=client,
                url=self.completions_url,
                body=body,
                headers=headers,
                max_attempts=self.max_attempts,
                retry_wait=self.retry_wait,
                raised_by=_RAISED_BY,
                endpoint_label=f"the model server at {self.base_url}",
            )
        finally:
            if self._owns_client:
                client.close()

    # -- response parsing -----------------------------------------------------------

    def parse_response(self, payload: dict[str, Any], *, latency_ms: int = 0) -> LLMResponse:
        """Turn a chat completions response body into an :class:`LLMResponse`."""
        choices = payload.get("choices") or []
        message: dict[str, Any] = {}
        finish_reason = "stop"
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            finish_reason = str(choices[0].get("finish_reason") or "stop")

        text = message.get("content")
        tool_calls = tuple(_parse_tool_calls(message.get("tool_calls") or []))

        raw_usage = payload.get("usage") or {}
        # Different field names from Anthropic's input_tokens / output_tokens.
        input_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
        model = str(payload.get("model", "") or self.model)

        rates = configured_rates(
            self.settings.llm_input_price_per_mtok, self.settings.llm_output_price_per_mtok
        )
        # With no configured rates and no table entry, this is a self-hosted model: the
        # call really was free, which is a different fact from "price unknown".
        estimated = rates is None and not is_priced(model) and not self._is_local()

        return LLMResponse(
            text=str(text) if text else None,
            tool_calls=tool_calls,
            stop_reason=FINISH_REASONS.get(finish_reason, StopReason.END_TURN),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd(model, input_tokens, output_tokens, rates=rates),
                cost_is_estimated=estimated,
            ),
            model=model,
            latency_ms=latency_ms,
        )


def _parse_tool_calls(raw_calls: list[Any]) -> list[ToolCall]:
    """Decode tool calls, whose arguments arrive as a JSON string rather than an object.

    A small open-weight model emitting malformed JSON here is a routine event, not an
    exceptional one, so it becomes a structured error naming the tool and the offending
    text -- something an agent loop can feed back and retry.
    """
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        name = str(function.get("name", ""))
        raw_arguments = function.get("arguments", "{}")

        if isinstance(raw_arguments, dict):
            # Some servers return an object despite the spec; accept it rather than
            # round-tripping it through a string.
            arguments: dict[str, JSONValue] = dict(raw_arguments)
        else:
            try:
                decoded = json.loads(raw_arguments or "{}")
            except (TypeError, ValueError) as exc:
                raise _fail(
                    ErrorCode.VALIDATION_FAILED,
                    f"tool call {name!r} returned arguments that are not valid JSON: {exc}",
                    remediation="Retry the call, or use a model that emits well-formed tool calls.",
                    details={"tool": name, "arguments": str(raw_arguments)[:500]},
                ) from exc
            if not isinstance(decoded, dict):
                kind = type(decoded).__name__
                raise _fail(
                    ErrorCode.VALIDATION_FAILED,
                    f"tool call {name!r} arguments decoded to {kind}, not an object",
                    remediation="Retry the call, or use a model that emits well-formed tool calls.",
                    details={"tool": name, "arguments": str(raw_arguments)[:500]},
                )
            arguments = decoded

        calls.append(ToolCall(name=name, arguments=arguments, call_id=str(raw.get("id", ""))))
    return calls
