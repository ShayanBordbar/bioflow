"""A scripted LLM client for unit tests.

Hand it a queue of responses and it serves them in order. Running out is an error
rather than a repeat of the last response: a test that makes more calls than it scripted
is a test whose assumptions have drifted, and it should say so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from bioflow.adapters.llm.base import LLMResponse
from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_RAISED_BY = "bioflow.adapters.llm.fake_client"


class FakeLLMClient:
    """Serves a fixed script of responses, recording what it was asked."""

    def __init__(self, responses: Iterable[LLMResponse] | None = None) -> None:
        self.queue: list[LLMResponse] = list(responses or [])
        self.calls: list[dict[str, JSONValue]] = []

    def push(self, response: LLMResponse) -> None:
        """Add a response to the end of the script."""
        self.queue.append(response)

    def complete(
        self,
        messages: Sequence[dict[str, JSONValue]],
        *,
        tools: Sequence[dict[str, JSONValue]] | None = None,
        tool_choice: dict[str, JSONValue] | None = None,
        response_model: type[BaseModel] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Return the next scripted response."""
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "tool_choice": tool_choice,
                "system": system,
            }
        )
        if not self.queue:
            raise BioflowError(
                StructuredError(
                    error_code=ErrorCode.MISSING_INPUT,
                    message=(
                        f"FakeLLMClient script exhausted: call #{len(self.calls)} had no "
                        f"queued response"
                    ),
                    remediation="Queue another response, or assert on fewer calls.",
                    details={"calls_made": len(self.calls)},
                    raised_by=_RAISED_BY,
                )
            )
        return self.queue.pop(0)
