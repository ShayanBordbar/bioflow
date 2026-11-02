"""Tool-schema translation, neutral form to provider form.

:meth:`bioflow.core.registry.Registry.to_llm_schema` emits a neutral
``{name, description, input_schema}`` and stops there. Translation happens here, in the
adapter layer, because ``bioflow.core`` must not learn the names of LLM providers --
that dependency would point the wrong way.

(ADR-0001 originally proposed putting this on the registry; the amendment records why
that was wrong.)

Both functions are pure, so they are tested without a client or a transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioflow.core.models import JSONValue

if TYPE_CHECKING:
    from collections.abc import Sequence


def to_anthropic_tools(tools: Sequence[dict[str, JSONValue]]) -> list[dict[str, JSONValue]]:
    """Render neutral tool definitions for the Anthropic Messages API.

    The neutral form is already what that API wants, so this is a pass-through. It exists
    so both providers are translated the same way at the call site rather than one being
    special-cased.
    """
    return [dict(tool) for tool in tools]


def to_openai_tools(tools: Sequence[dict[str, JSONValue]]) -> list[dict[str, JSONValue]]:
    """Render neutral tool definitions for an OpenAI-compatible chat completions API.

    The JSON Schema moves from ``input_schema`` to ``function.parameters`` and the whole
    definition is wrapped in a ``function`` envelope.
    """
    translated: list[dict[str, JSONValue]] = []
    for tool in tools:
        function: dict[str, JSONValue] = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        }
        translated.append({"type": "function", "function": function})
    return translated


def to_openai_tool_choice(tool_choice: dict[str, JSONValue] | None) -> JSONValue:
    """Translate an Anthropic-shaped ``tool_choice`` into the OpenAI-compatible form.

    Anthropic uses ``{"type": "auto"|"any"|"tool", "name": ...}``; OpenAI-compatible uses
    the bare strings ``"auto"``/``"none"``/``"required"`` or a ``function`` object.
    """
    if tool_choice is None:
        return None

    kind = str(tool_choice.get("type", "auto"))
    if kind == "tool":
        name = tool_choice.get("name")
        if name is not None:
            return {"type": "function", "function": {"name": name}}
        return "required"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    return "auto"
