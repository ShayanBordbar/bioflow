"""Token counting and truncation.

Counting is a heuristic -- roughly four characters per token -- and deliberately not a
real tokenizer. Pulling in ``tiktoken`` would add a large dependency, tie the estimate to
one vendor's vocabulary, and still be wrong for every other provider. What matters here
is that the estimate is *consistent* and *conservative enough to bound a request*, not
that it is exact. :func:`set_token_counter` exists so a caller who needs exactness can
supply it.

Every truncation in this module leaves a marker. Silently dropping rows would let the
model reason confidently about a table it has only seen the top of, and a confident wrong
answer is the most expensive thing it can produce. Telling it the table was cut costs
eleven tokens.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CHARS_PER_TOKEN = 4
"""The heuristic's divisor. Close enough for English prose and tabular data alike."""


def _heuristic(text: str) -> int:
    """Estimate a token count from character length."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


_counter: Callable[[str], int] = _heuristic


def count_tokens(text: str) -> int:
    """Return the estimated token count of ``text``."""
    return _counter(text)


def set_token_counter(fn: Callable[[str], int] | None) -> None:
    """Swap the counting function. Passing ``None`` restores the heuristic."""
    global _counter
    _counter = fn if fn is not None else _heuristic


def truncation_marker(kept: int, total: int, unit: str = "rows") -> str:
    """Render the marker that says content was cut, and by how much."""
    return f"[truncated: {kept} of {total} {unit}]"


def truncate_rows(rows: Sequence[str], limit: int, unit: str = "rows") -> list[str]:
    """Keep at most ``limit`` rows, appending a marker when any were dropped."""
    total = len(rows)
    if limit >= total:
        return list(rows)
    kept = max(limit, 0)
    return [*rows[:kept], truncation_marker(kept, total, unit)]


def truncate_lines(text: str, limit: int, unit: str = "lines") -> str:
    """Keep at most ``limit`` lines of ``text``, appending a marker when cut."""
    lines = text.splitlines()
    return "\n".join(truncate_rows(lines, limit, unit))


def fit(text: str, budget_tokens: int, unit: str = "characters") -> str:
    """Cut ``text`` to fit a token budget, on a line boundary where possible.

    The result is **always** within budget, marker included. Cutting at a line boundary
    keeps the output parseable; the marker keeps it honest.

    A budget too small to hold even a marker yields the empty string. That is the one
    case where nothing is said, and it is deliberate: the section headings in
    ``agent_context.md`` remain, so an empty section is still visible to the reader,
    whereas a request that quietly ran over budget is not visible to anyone until a
    provider rejects it.
    """
    if budget_tokens <= 0:
        return ""
    if count_tokens(text) <= budget_tokens:
        return text

    lines = text.splitlines()
    # Reserve the largest marker this call could possibly emit. Reserving the smallest
    # one instead lets the result land a token or two over budget, because "[truncated:
    # 88 of 400 ...]" is longer than the "[truncated: 0 of 400 ...]" that was budgeted.
    reserve = (
        max(
            count_tokens(truncation_marker(len(lines), len(lines), "lines")),
            count_tokens(truncation_marker(len(text), len(text), unit)),
        )
        + 1  # the newline joining body to marker
    )
    allowance = budget_tokens - reserve

    if allowance <= 0:
        marker = truncation_marker(0, len(text), unit)
        return marker if count_tokens(marker) <= budget_tokens else ""

    kept: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line) + 1  # +1 for the newline
        if used + cost > allowance:
            break
        kept.append(line)
        used += cost

    if kept:
        body = "\n".join(kept)
        return f"{body}\n{truncation_marker(len(kept), len(lines), 'lines')}"

    # Not even one whole line fits: fall back to a character cut.
    chars = allowance * CHARS_PER_TOKEN
    return f"{text[:chars]}\n{truncation_marker(chars, len(text), unit)}"
