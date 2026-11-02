"""Golden-case harness for tool contracts.

A tool's contract is what an agent sees: the arguments it accepts, the output it
returns, and — just as importantly — the way it refuses. Golden cases pin all three, so
a change to any of them shows up as a readable diff rather than as different agent
behaviour three layers away.

A case is a set of sibling files in a golden directory:

``<case>.in.json``
    The raw arguments, exactly as an LLM would emit them.
``<case>.out.json``
    The expected :class:`~bioflow.core.registry.ToolResult` envelope --
    ``{"ok": ..., "output": ..., "error": ...}``. Storing the whole envelope is what
    lets a case pin a rejection, not just a success.
``<case>.setup.json`` (optional)
    Artifacts to store in the temporary run before invoking, so a tool that reads from
    the run store has something to read. Each entry is bound to a name, and ``${name}``
    in ``<case>.in.json`` is replaced by the artifact ID that was minted.
``<case>.ignore.json`` (optional)
    Dotted paths to prune from *both* sides before comparing: timestamps, minted IDs,
    durations, absolute paths. ``*`` matches any dict key or list index.
"""

from __future__ import annotations

import difflib
import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from bioflow.core.models import ArtifactType, JSONValue
from bioflow.core.registry import Registry, invoke
from bioflow.core.runctx import RunContext

if TYPE_CHECKING:
    from collections.abc import Iterable

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

IN_SUFFIX = ".in.json"
OUT_SUFFIX = ".out.json"
SETUP_SUFFIX = ".setup.json"
IGNORE_SUFFIX = ".ignore.json"


def case_names(golden_dir: Path) -> list[str]:
    """Return the sorted case names found in a golden directory."""
    return sorted(p.name[: -len(IN_SUFFIX)] for p in golden_dir.glob(f"*{IN_SUFFIX}"))


def assert_tool_contract(tool_name: str, golden_dir: Path, *, into: Registry | None = None) -> None:
    """Run every golden case for a tool and assert each matches its recording."""
    golden_dir = Path(golden_dir)
    names = case_names(golden_dir)
    if not names:
        raise AssertionError(
            f"no golden cases for tool {tool_name!r}: expected at least one "
            f"*{IN_SUFFIX} file in {golden_dir}"
        )
    for case in names:
        assert_tool_case(tool_name, golden_dir, case, into=into)


def assert_tool_case(
    tool_name: str, golden_dir: Path, case: str, *, into: Registry | None = None
) -> None:
    """Run one golden case against a throwaway run and compare to its recording."""
    golden_dir = Path(golden_dir)
    expected_path = golden_dir / f"{case}{OUT_SUFFIX}"
    if not expected_path.is_file():
        raise AssertionError(f"golden case {case!r} for {tool_name!r} has no {expected_path.name}")

    raw_args = _load(golden_dir / f"{case}{IN_SUFFIX}")
    setup = _load_optional(golden_dir / f"{case}{SETUP_SUFFIX}")
    ignore = _load_optional(golden_dir / f"{case}{IGNORE_SUFFIX}")
    expected = _load(expected_path)

    with (
        tempfile.TemporaryDirectory() as tmp,
        RunContext.create(Path(tmp) / "workspace", {"case": case}) as ctx,
    ):
        bindings = _apply_setup(ctx, setup)
        args = _substitute(raw_args, bindings)
        if not isinstance(args, dict):
            raise AssertionError(f"{case}{IN_SUFFIX} must contain a JSON object")
        result = invoke(tool_name, args, ctx, confirmed=True, into=into)
        actual: JSONValue = result.to_json_dict()

    paths = _ignore_paths(ignore)
    pruned_actual = _prune(actual, paths)
    pruned_expected = _prune(_substitute(expected, bindings), paths)

    if pruned_actual != pruned_expected:
        raise AssertionError(
            f"tool {tool_name!r} golden case {case!r} does not match "
            f"{expected_path.name}:\n" + _diff(pruned_expected, pruned_actual)
        )


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------


def _apply_setup(ctx: RunContext, setup: JSONValue) -> dict[str, str]:
    """Seed the temporary run, returning the ``name -> artifact_id`` bindings.

    Without this the harness could only exercise tools that ignore ``ctx``, which is
    almost no real tool: they take artifact IDs and look them up.
    """
    if setup is None:
        return {}
    if not isinstance(setup, dict):
        raise AssertionError(f"{SETUP_SUFFIX} must contain a JSON object")

    bindings: dict[str, str] = {}
    declared = setup.get("artifacts", [])
    if not isinstance(declared, list):
        raise AssertionError(f"{SETUP_SUFFIX} 'artifacts' must be a list")

    for index, item in enumerate(declared):
        if not isinstance(item, dict):
            raise AssertionError(f"{SETUP_SUFFIX} artifact {index} must be a JSON object")
        content = item.get("content", "")
        artifact_type = ArtifactType(str(item.get("artifact_type", ArtifactType.OTHER.value)))
        produced_by = item.get("produced_by")
        ref = ctx.record(
            ctx.store.put_bytes(
                str(content).encode("utf-8"),
                artifact_type,
                produced_by=None if produced_by is None else str(produced_by),
                run_id=ctx.run_id,
            )
        )
        bindings[str(item.get("as", f"artifact{index}"))] = ref.artifact_id
    return bindings


def _substitute(value: JSONValue, bindings: dict[str, str]) -> JSONValue:
    """Replace ``${name}`` placeholders with the IDs minted during setup."""
    if isinstance(value, str):
        if not bindings:
            return value
        return _PLACEHOLDER_RE.sub(lambda m: bindings.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {key: _substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    return value


# --------------------------------------------------------------------------------------
# Ignoring and diffing
# --------------------------------------------------------------------------------------


def _ignore_paths(ignore: JSONValue) -> list[list[str]]:
    """Parse the ignore file into split dotted paths."""
    if ignore is None:
        return []
    if not isinstance(ignore, list):
        raise AssertionError(f"{IGNORE_SUFFIX} must contain a JSON list of dotted paths")
    return [str(entry).split(".") for entry in ignore]


def _prune(value: JSONValue, paths: Iterable[list[str]]) -> JSONValue:
    """Remove every ignored path from a structure, leaving the rest untouched."""
    result = value
    for path in paths:
        result = _prune_one(result, path)
    return result


def _prune_one(value: JSONValue, path: list[str]) -> JSONValue:
    if not path:
        return None
    head, rest = path[0], path[1:]

    if isinstance(value, dict):
        pruned: dict[str, JSONValue] = {}
        for key, item in value.items():
            if head not in ("*", key):
                pruned[key] = item  # not on the ignored path
            elif rest:
                pruned[key] = _prune_one(item, rest)
            # A matched leaf is dropped entirely rather than blanked, so a field that is
            # ignored on one side but absent on the other still compares equal.
        return pruned

    if isinstance(value, list):
        # Only "*" addresses list elements; a numeric index would pin the position of
        # something whose order is rarely part of the contract.
        if head != "*":
            return value
        return [] if not rest else [_prune_one(item, rest) for item in value]

    return value


def _diff(expected: JSONValue, actual: JSONValue) -> str:
    """Render a unified diff of two JSON structures."""
    lines = difflib.unified_diff(
        _render(expected).splitlines(),
        _render(actual).splitlines(),
        fromfile="expected (golden)",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(lines)


def _render(value: JSONValue) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _load(path: Path) -> JSONValue:
    if not path.is_file():
        raise AssertionError(f"missing golden file {path}")
    loaded: JSONValue = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _load_optional(path: Path) -> JSONValue:
    return _load(path) if path.is_file() else None
