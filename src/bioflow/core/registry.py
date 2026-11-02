"""The agent-visible tool layer.

This is where invariant #1 stops being a promise. A tool call arrives as a plain
``dict`` of JSON, is validated against a Pydantic model, and either runs or comes back
as a :class:`~bioflow.core.models.StructuredError`. There is no path between those two
outcomes, and in particular no path that reaches a shell.

:func:`invoke` never raises. The caller is ultimately a language model, which will
eventually invent a tool name, omit a required field, or trip a bug inside a tool body.
A structured refusal is something it can read and correct; a traceback is not.

Registration failures behave the opposite way: a malformed ``@tool`` raises at import
time, because a broken tool definition should break the build rather than wait to
surprise someone mid-run.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel, ValidationError

from bioflow.core.ids import new_event_id
from bioflow.core.models import (
    ARTIFACT_ID_PATTERN,
    Actor,
    BioflowError,
    CostClass,
    ErrorCode,
    EventStatus,
    JSONValue,
    ProvenanceEvent,
    SideEffect,
    StructuredError,
    ToolSpec,
)

if TYPE_CHECKING:
    from bioflow.core.runctx import RunContext

_ARTIFACT_ID_RE = re.compile(ARTIFACT_ID_PATTERN)
_RAISED_BY = "bioflow.core.registry"

ToolFunction = Callable[[Any, "RunContext"], BaseModel]


@dataclass(frozen=True)
class ToolResult:
    """The envelope every tool call comes back in.

    A plain dataclass rather than a Pydantic model: it carries a ``BaseModel`` instance,
    which Pydantic would try to validate and re-serialize. This is a return envelope,
    not domain vocabulary.
    """

    ok: bool
    output: BaseModel | None
    error: StructuredError | None
    duration_ms: int

    def to_json_dict(self) -> dict[str, JSONValue]:
        """Render the envelope as plain JSON, for goldens and for the LLM."""
        return {
            "ok": self.ok,
            "output": None if self.output is None else self.output.model_dump(mode="json"),
            "error": None if self.error is None else self.error.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class RegisteredTool:
    """A tool's declared contract paired with the function that implements it."""

    spec: ToolSpec
    fn: ToolFunction
    params_model: type[BaseModel]
    output_model: type[BaseModel]


class Registry:
    """The set of tools an agent may call."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, entry: RegisteredTool) -> None:
        """Add a tool, refusing to shadow an existing name."""
        name = entry.spec.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = entry

    def get(self, name: str) -> RegisteredTool | None:
        """Return a registered tool, or ``None`` when the name is unknown."""
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        """Return every spec, sorted by name so prompts and tests stay stable."""
        return [self._tools[name].spec for name in sorted(self._tools)]

    def to_llm_schema(self) -> list[dict[str, JSONValue]]:
        """Render the registry as provider-agnostic tool definitions.

        Adapters translate this into whatever shape their provider wants. Nested
        parameter models legitimately produce ``$defs``/``$ref``; flattening them is an
        adapter's problem, not the registry's.
        """
        return [
            {
                "name": spec.name,
                "description": spec.summary,
                "input_schema": dict(spec.input_schema),
            }
            for spec in self.list_specs()
        ]

    def clear(self) -> None:
        """Drop every registration. Exists for tests that need an isolated registry."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


registry = Registry()
"""The module-level registry every ``@tool`` registers into."""


def tool(
    name: str,
    version: str,
    summary: str,
    side_effect: SideEffect = SideEffect.NONE,
    cost_class: CostClass = CostClass.CHEAP,
    requires_confirmation: bool = False,
    tags: tuple[str, ...] = (),
    into: Registry | None = None,
) -> Callable[[ToolFunction], ToolFunction]:
    """Register a function as an agent-callable tool.

    The decorated function must have the signature ``(params: PIn, ctx: RunContext) ->
    POut`` where ``PIn`` and ``POut`` are Pydantic models; both JSON schemas are derived
    from them. Anything else raises here, at import time.
    """

    def decorate(fn: ToolFunction) -> ToolFunction:
        params_model, output_model = _models_from_signature(fn, name)
        spec = ToolSpec(
            name=name,
            version=version,
            summary=summary,
            input_schema=params_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            side_effect=side_effect,
            cost_class=cost_class,
            requires_confirmation=requires_confirmation,
            tags=tags,
        )
        # `into or registry` would be wrong: Registry defines __len__, so an empty one
        # is falsy and would silently fall back to the module-level registry.
        target = into if into is not None else registry
        target.register(
            RegisteredTool(spec=spec, fn=fn, params_model=params_model, output_model=output_model)
        )
        return fn

    return decorate


def _models_from_signature(fn: ToolFunction, name: str) -> tuple[type[BaseModel], type[BaseModel]]:
    """Recover the params and output models from a tool's annotations."""
    try:
        hints = get_type_hints(fn)
    except Exception as exc:  # pragma: no cover - unresolvable annotations are rare
        raise TypeError(f"tool {name!r} has unresolvable annotations: {exc}") from exc

    missing = [field for field in ("params", "ctx", "return") if field not in hints]
    if missing:
        raise TypeError(
            f"tool {name!r} must be annotated (params: PIn, ctx: RunContext) -> POut; "
            f"missing annotations: {', '.join(missing)}"
        )

    params_model = hints["params"]
    output_model = hints["return"]
    for role, model in (("params", params_model), ("return", output_model)):
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(
                f"tool {name!r} {role} annotation must be a pydantic BaseModel subclass, "
                f"got {model!r}"
            )
    return params_model, output_model


def validate_args(
    name: str, raw_args: dict[str, JSONValue], *, into: Registry | None = None
) -> StructuredError | None:
    """Check arguments against a tool's schema without running it.

    Returns ``None`` when they are valid. This exists for the agent loop's repair cycle,
    which needs to know that a *call* was malformed rather than that a *tool* failed.
    Reading the error code off :func:`invoke` cannot tell those apart -- a tool may
    legitimately raise ``VALIDATION_FAILED`` from inside its own body -- and retrying
    that second case can never succeed.
    """
    active = into if into is not None else registry
    entry = active.get(name)
    if entry is None:
        available = ", ".join(spec.name for spec in active.list_specs())
        return StructuredError(
            error_code=ErrorCode.NOT_FOUND,
            message=f"no tool named {name!r}",
            remediation=f"Available tools: {available}.",
            details={"requested": name},
            raised_by=_RAISED_BY,
        )
    try:
        entry.params_model.model_validate(raw_args)
    except ValidationError as exc:
        return _validation_error(name, exc)
    return None


def invoke(
    name: str,
    raw_args: dict[str, JSONValue],
    ctx: RunContext,
    *,
    confirmed: bool = False,
    into: Registry | None = None,
) -> ToolResult:
    """Validate and run a tool call, recording it, and never raising.

    Every outcome -- unknown name, bad arguments, missing confirmation, a failure inside
    the tool -- comes back as a :class:`ToolResult` and leaves exactly one
    :class:`~bioflow.core.models.ProvenanceEvent` behind.
    """
    started = time.monotonic()
    # An empty Registry is falsy (it defines __len__), so this must test against None.
    active = into if into is not None else registry
    entry = active.get(name)

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    if entry is None:
        available = ", ".join(spec.name for spec in active.list_specs())
        error = StructuredError(
            error_code=ErrorCode.NOT_FOUND,
            message=f"no tool named {name!r}",
            remediation=f"Available tools: {available}.",
            details={"requested": name},
            raised_by=_RAISED_BY,
        )
        return _finish(ctx, name, None, raw_args, None, error, elapsed_ms())

    if entry.spec.requires_confirmation and not confirmed:
        # Fails closed on purpose: the caller is a language model, and a tool that asked
        # for confirmation must not run just because nobody checked.
        error = StructuredError(
            error_code=ErrorCode.VALIDATION_FAILED,
            message=f"tool {name!r} requires explicit confirmation before it can run",
            remediation="Re-issue the call once a human has confirmed it.",
            details={"tool": name},
            raised_by=_RAISED_BY,
        )
        return _finish(ctx, name, entry, raw_args, None, error, elapsed_ms())

    try:
        params = entry.params_model.model_validate(raw_args)
    except ValidationError as exc:
        return _finish(ctx, name, entry, raw_args, None, _validation_error(name, exc), elapsed_ms())

    try:
        output = entry.fn(params, ctx)
    except BioflowError as exc:
        # A tool that diagnosed its own failure keeps its diagnosis.
        return _finish(ctx, name, entry, raw_args, None, exc.error, elapsed_ms())
    except Exception as exc:
        error = StructuredError(
            error_code=ErrorCode.INTERNAL,
            message=f"{type(exc).__name__}: {exc}",
            details={"tool": name},
            raised_by=name,
        )
        return _finish(ctx, name, entry, raw_args, None, error, elapsed_ms())

    return _finish(ctx, name, entry, raw_args, output, None, elapsed_ms())


def _validation_error(name: str, exc: ValidationError) -> StructuredError:
    """Flatten Pydantic's error list into something a model can act on.

    The message is one compact line per bad field, because that is what gets read back
    into the next prompt. The full structured error list stays in ``details`` for a human
    debugging afterwards.
    """
    lines = []
    details: list[JSONValue] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"{location}: {item['msg']}")
        details.append({"loc": location, "msg": str(item["msg"]), "type": str(item["type"])})
    return StructuredError(
        error_code=ErrorCode.VALIDATION_FAILED,
        message=f"invalid arguments for {name!r}: " + "; ".join(lines),
        remediation="Correct the listed fields and call the tool again.",
        details={"tool": name, "errors": details},
        raised_by=_RAISED_BY,
    )


def _finish(
    ctx: RunContext,
    name: str,
    entry: RegisteredTool | None,
    raw_args: dict[str, JSONValue],
    output: BaseModel | None,
    error: StructuredError | None,
    duration_ms: int,
) -> ToolResult:
    """Emit the provenance event for a call and build its result envelope."""
    output_json = None if output is None else output.model_dump(mode="json")
    ctx.emit(
        ProvenanceEvent(
            event_id=new_event_id(),
            run_id=ctx.run_id,
            timestamp=datetime.now(UTC),
            actor=Actor.TOOL,
            action=name,
            tool=name,
            tool_version=None if entry is None else entry.spec.version,
            inputs=_artifact_ids(raw_args),
            outputs=_artifact_ids(output_json),
            params_sha256=_params_digest(raw_args),
            duration_ms=duration_ms,
            status=EventStatus.OK if error is None else EventStatus.ERROR,
            error=error,
        )
    )
    return ToolResult(ok=error is None, output=output, error=error, duration_ms=duration_ms)


def _params_digest(raw_args: dict[str, JSONValue]) -> str | None:
    """Hash the arguments canonically, so identical calls hash identically."""
    try:
        canonical = json.dumps(raw_args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - raw_args is JSON by contract
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_ids(value: JSONValue) -> tuple[str, ...]:
    """Collect every artifact ID appearing anywhere in a JSON structure.

    ``artifact_id`` has a closed format, so a string matching it is an artifact
    reference by construction. Deriving the provenance graph this way means every tool
    links its inputs to its outputs without writing any bookkeeping of its own.
    """
    found: list[str] = []
    _walk_for_ids(value, found)
    # Preserve first-seen order while dropping repeats.
    return tuple(dict.fromkeys(found))


def _walk_for_ids(value: JSONValue, found: list[str]) -> None:
    if isinstance(value, str):
        if _ARTIFACT_ID_RE.match(value):
            found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _walk_for_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_for_ids(item, found)
