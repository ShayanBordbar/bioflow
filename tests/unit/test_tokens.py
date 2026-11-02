"""Unit tests for token estimation and truncation."""

from collections.abc import Iterator

import pytest

from bioflow.orchestration.tokens import (
    CHARS_PER_TOKEN,
    count_tokens,
    fit,
    set_token_counter,
    truncate_lines,
    truncate_rows,
    truncation_marker,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_counter() -> Iterator[None]:
    yield
    set_token_counter(None)


# --------------------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------------------


def test_the_heuristic_is_four_characters_per_token() -> None:
    assert count_tokens("a" * 4 * CHARS_PER_TOKEN) == 4


def test_counting_rounds_up_so_a_fragment_is_never_free() -> None:
    assert count_tokens("a") == 1
    assert count_tokens("") == 0


def test_the_counter_can_be_swapped() -> None:
    # The whole point of routing through one function: a caller who needs a real
    # tokenizer supplies it without this package depending on one.
    set_token_counter(lambda text: len(text.split()))
    assert count_tokens("one two three") == 3
    set_token_counter(None)
    assert count_tokens("one two three") == 4


# --------------------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------------------


def test_the_marker_names_what_was_kept_and_what_there_was() -> None:
    assert truncation_marker(3, 50) == "[truncated: 3 of 50 rows]"
    assert truncation_marker(1, 2, "matches") == "[truncated: 1 of 2 matches]"


def test_rows_within_the_limit_are_untouched() -> None:
    assert truncate_rows(["a", "b"], 5) == ["a", "b"]
    assert truncate_rows(["a", "b"], 2) == ["a", "b"]


def test_dropped_rows_always_leave_a_marker() -> None:
    # Silently cutting a table lets the model describe the top of it as the whole of it.
    result = truncate_rows(["a", "b", "c"], 2)
    assert result == ["a", "b", "[truncated: 2 of 3 rows]"]


def test_truncating_to_zero_still_says_so() -> None:
    assert truncate_rows(["a", "b"], 0) == ["[truncated: 0 of 2 rows]"]


def test_lines_truncate_like_rows() -> None:
    assert truncate_lines("a\nb\nc", 2) == "a\nb\n[truncated: 2 of 3 lines]"


# --------------------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------------------


def test_text_within_budget_is_returned_unchanged() -> None:
    text = "short enough"
    assert fit(text, 100) == text


def test_fitting_cuts_on_a_line_boundary_and_marks_it() -> None:
    text = "\n".join(f"line {index}" for index in range(50))
    result = fit(text, 20)
    assert "[truncated:" in result
    assert count_tokens(result) <= 20
    # Whole lines only, so the result stays parseable.
    body = result.split("\n[truncated:")[0]
    assert all(line.startswith("line ") for line in body.splitlines())


def test_fitting_falls_back_to_characters_when_no_line_fits() -> None:
    result = fit("x" * 400, 30)
    assert "[truncated:" in result
    assert count_tokens(result) <= 30


def test_a_budget_that_holds_only_a_marker_yields_the_marker() -> None:
    assert fit("x" * 400, 10) == "[truncated: 0 of 400 characters]"


def test_a_budget_too_small_for_a_marker_yields_nothing() -> None:
    # The section heading in the template survives, so an empty section is still visible.
    # A request that quietly ran over budget would not be visible to anyone until a
    # provider rejected it.
    assert fit("anything at all", 0) == ""
    assert fit("x" * 400, 3) == ""


@pytest.mark.parametrize("budget", [5, 10, 25, 60, 200])
def test_fit_never_exceeds_its_budget(budget: int) -> None:
    text = "\n".join(f"a moderately long line number {index}" for index in range(200))
    assert count_tokens(fit(text, budget)) <= budget
