"""Unit tests for neutral -> provider tool-schema translation."""

from typing import Any

import pytest

from bioflow.adapters.llm.tool_schema import (
    to_anthropic_tools,
    to_openai_tool_choice,
    to_openai_tools,
)
from bioflow.core.models import JSONValue

pytestmark = pytest.mark.unit

NEUTRAL: list[dict[str, JSONValue]] = [
    {
        "name": "echo_artifact",
        "description": "Describe a stored artifact in one line.",
        "input_schema": {
            "type": "object",
            "properties": {"artifact_id": {"type": "string"}},
            "required": ["artifact_id"],
        },
    }
]


def test_anthropic_translation_is_a_pass_through() -> None:
    assert to_anthropic_tools(NEUTRAL) == NEUTRAL


def test_anthropic_translation_copies_rather_than_aliases() -> None:
    translated = to_anthropic_tools(NEUTRAL)
    translated[0]["name"] = "mutated"
    assert NEUTRAL[0]["name"] == "echo_artifact"


def test_openai_translation_wraps_in_a_function_envelope() -> None:
    (tool,) = to_openai_tools(NEUTRAL)
    assert tool["type"] == "function"
    function = tool["function"]
    assert isinstance(function, dict)
    assert function["name"] == "echo_artifact"
    assert function["description"] == "Describe a stored artifact in one line."
    # The JSON Schema moves from input_schema to function.parameters.
    assert function["parameters"] == NEUTRAL[0]["input_schema"]
    assert "input_schema" not in function


def test_openai_translation_defaults_a_missing_schema() -> None:
    (tool,) = to_openai_tools([{"name": "noop", "description": "d"}])
    function = tool["function"]
    assert isinstance(function, dict)
    assert function["parameters"] == {"type": "object", "properties": {}}


def test_nested_defs_and_refs_survive_translation() -> None:
    nested: list[dict[str, JSONValue]] = [
        {
            "name": "complex",
            "description": "d",
            "input_schema": {
                "type": "object",
                "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
            },
        }
    ]
    (tool,) = to_openai_tools(nested)
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert "$defs" in parameters
    assert parameters["properties"] == {"inner": {"$ref": "#/$defs/Inner"}}


def test_an_empty_tool_list_stays_empty() -> None:
    assert to_openai_tools([]) == []
    assert to_anthropic_tools([]) == []


def test_the_real_registry_schema_translates() -> None:
    import bioflow.tools  # noqa: F401
    from bioflow.core.registry import registry

    neutral = registry.to_llm_schema()
    translated = to_openai_tools(neutral)
    assert len(translated) == len(neutral) > 0

    by_name: dict[str, Any] = {}
    for tool in translated:
        function = tool["function"]
        assert isinstance(function, dict)
        parameters = function["parameters"]
        assert isinstance(parameters, dict)
        assert parameters["type"] == "object"
        by_name[str(function["name"])] = function

    # Named explicitly rather than counted: the point is that every registered tool
    # survives translation, not that the registry happens to hold a particular number.
    assert "echo_artifact" in by_name
    assert {"request_review", "handoff"} <= set(by_name)


@pytest.mark.parametrize(
    ("anthropic_choice", "expected"),
    [
        (None, None),
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        (
            {"type": "tool", "name": "echo_artifact"},
            {"type": "function", "function": {"name": "echo_artifact"}},
        ),
        ({"type": "tool"}, "required"),
        ({"type": "something_new"}, "auto"),
    ],
)
def test_tool_choice_translation(anthropic_choice: Any, expected: Any) -> None:
    assert to_openai_tool_choice(anthropic_choice) == expected
