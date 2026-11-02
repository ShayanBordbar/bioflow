"""Per-model token prices and capability flags.

Prices are USD per million tokens, recorded as plain constants rather than fetched. A
manifest read a year from now must reproduce the cost that was recorded at the time, and
that is only true if the table is a constant.

Rates verified 2026-08-28 against the published Anthropic pricing.
"""

from __future__ import annotations

from decimal import Decimal

TOKENS_PER_UNIT = Decimal(1_000_000)

# model -> (input $/MTok, output $/MTok)
#
# Claude Sonnet 5 carried an introductory rate of $2/$10 that expired 2026-08-31; the
# standard rate below is what applies from then on.
PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "claude-fable-5": (Decimal("10.00"), Decimal("50.00")),
    "claude-mythos-5": (Decimal("10.00"), Decimal("50.00")),
    "claude-opus-5": (Decimal("5.00"), Decimal("25.00")),
    "claude-opus-4-8": (Decimal("5.00"), Decimal("25.00")),
    "claude-opus-4-7": (Decimal("5.00"), Decimal("25.00")),
    "claude-opus-4-6": (Decimal("5.00"), Decimal("25.00")),
    "claude-sonnet-5": (Decimal("3.00"), Decimal("15.00")),
    "claude-sonnet-4-6": (Decimal("3.00"), Decimal("15.00")),
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
}

# Models that still accept `temperature` / `top_p` / `top_k`.
#
# Sampling parameters were removed from the current frontier models: sending
# `temperature` to Opus 5, Opus 4.8/4.7, Sonnet 5, or Fable 5 returns HTTP 400. Since
# `claude-opus-5` is BioFlow's default model, the parameter has to be conditional or
# every default-configured request fails. Reproducibility comes from cassette replay
# instead -- `temperature=0` never guaranteed identical output in any case.
SUPPORTS_SAMPLING: frozenset[str] = frozenset(
    {
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    }
)


def supports_sampling(model: str) -> bool:
    """Report whether ``model`` accepts a ``temperature`` parameter."""
    return model in SUPPORTS_SAMPLING


def is_priced(model: str) -> bool:
    """Report whether a price is on file for ``model``."""
    return model in PRICES


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    rates: tuple[Decimal, Decimal] | None = None,
) -> float:
    """Return the USD cost of a call.

    ``rates`` overrides the table, for an endpoint whose price BioFlow cannot know --
    a paid OpenAI-compatible gateway configured through settings.

    An unpriced model costs 0.0 rather than raising: a gap in this table is a
    bookkeeping problem, and it must not be able to abort a scientific run. Callers
    distinguish the two cases with :func:`is_priced`.
    """
    resolved = rates if rates is not None else PRICES.get(model)
    if resolved is None:
        return 0.0
    input_rate, output_rate = resolved
    total = (
        Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
    ) / TOKENS_PER_UNIT
    return float(total)


def configured_rates(
    input_per_mtok: float | None, output_per_mtok: float | None
) -> tuple[Decimal, Decimal] | None:
    """Build a rate pair from settings, or ``None`` when neither price is configured.

    A single configured side is honoured with the other at zero: a gateway that charges
    only for output is unusual but not impossible, and silently dropping the one price
    the operator did supply would be worse.
    """
    if input_per_mtok is None and output_per_mtok is None:
        return None
    return (Decimal(str(input_per_mtok or 0.0)), Decimal(str(output_per_mtok or 0.0)))
