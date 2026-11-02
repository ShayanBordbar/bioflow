"""LLM spend accounting for a run.

Totals end up in :attr:`~bioflow.core.models.RunManifest.llm_usage`, so a finished run
carries what it cost alongside what it concluded.

This lives in ``core`` rather than beside the LLM adapter because ``RunContext`` owns
one, and imports only ever point downward: the adapter may depend on core, never the
reverse. The only thing it needs from the adapter is a response shape, which is imported
for typing alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioflow.core.models import JSONValue

if TYPE_CHECKING:
    from bioflow.adapters.llm.base import LLMResponse


class UsageAccumulator:
    """Running totals across every LLM call in one run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.n_calls = 0
        self.model: str | None = None
        self.prompt_version: str | None = None
        # True once any call was priced with an unknown model, so the total is a floor
        # rather than an exact figure.
        self.cost_is_estimated = False

    def record(self, response: LLMResponse) -> None:
        """Fold one response into the totals."""
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cost_usd += response.usage.cost_usd
        self.n_calls += 1
        if response.model:
            self.model = response.model
        if response.prompt_version:
            self.prompt_version = response.prompt_version
        if response.usage.cost_is_estimated:
            self.cost_is_estimated = True

    def as_dict(self) -> dict[str, JSONValue]:
        """Render totals in the shape ``RunManifest.llm_usage`` documents."""
        usage: dict[str, JSONValue] = {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            # Sub-cent sums accumulate float noise; nine places is well beyond the
            # precision of any real price and keeps the number readable.
            "cost_usd": round(self.cost_usd, 9),
            "n_calls": self.n_calls,
        }
        if self.cost_is_estimated:
            usage["cost_is_estimated"] = True
        return usage

    def __repr__(self) -> str:
        return (
            f"UsageAccumulator(n_calls={self.n_calls}, "
            f"input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, "
            f"cost_usd={self.cost_usd:.6f})"
        )
