"""The LLM boundary.

Every model call in BioFlow goes through :class:`LLMClient`. That is not a style
preference: it is what makes calls recordable, replayable, and costable, and it is why
no other module is allowed to import a provider SDK.

The response types carry only JSON-native data, so an :class:`LLMResponse` serializes
straight into a cassette or a trace record without a custom encoder.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from bioflow.core.models import JSONValue

if TYPE_CHECKING:
    from collections.abc import Sequence


class StopReason(StrEnum):
    """Why the model stopped generating.

    ``REFUSAL`` is the one that bites: it arrives as a perfectly ordinary HTTP 200, so
    code that only checks the status code treats a safety decline as an empty answer.
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class ToolCall(BaseModel):
    """One tool invocation the model asked for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    call_id: str


class Usage(BaseModel):
    """Token counts and cost for a single call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    # True when the model was absent from the pricing table, so a zero cost can be told
    # apart from a genuinely free call.
    cost_is_estimated: bool = False


class LLMResponse(BaseModel):
    """One completion, with everything needed to record and price it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency_ms: int = Field(default=0, ge=0)
    prompt_version: str | None = None

    @property
    def refused(self) -> bool:
        """Whether the model declined the request."""
        return self.stop_reason is StopReason.REFUSAL


@runtime_checkable
class LLMClient(Protocol):
    """What every LLM implementation, wrapper, and fake must provide."""

    def complete(
        self,
        messages: Sequence[dict[str, JSONValue]],
        *,
        tools: Sequence[dict[str, JSONValue]] | None = None,
        tool_choice: dict[str, JSONValue] | None = None,
        response_model: type[BaseModel] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a completion request and return the parsed response."""
        ...
