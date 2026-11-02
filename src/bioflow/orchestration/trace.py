"""The agent loop's own record: ``runs/<run_id>/trace.jsonl``.

Provenance answers "what happened to which artifact". The trace answers a different
question -- "why did the agent do that" -- and the two are kept apart because merging
them would bury the handful of events a scientist cares about under a per-token audit of
a model's indecision.

One JSON object per line, in three kinds:

``header``
    Written once, first. Carries everything :func:`~bioflow.orchestration.replay.replay`
    needs to rebuild the run: the request, the cassette, the model, and the caps.
``call``
    One per LLM call: which state, which prompt, what was asked for, what was wrong with
    it, and what it cost.
``footer``
    Written once, last. The final state, the observed tool-call sequence, and the metrics.

Messages are stored as a hash, not as text. The cassette already holds the real request
byte for byte, and a trace that inlines every prompt stops being something a human will
open. Tool *arguments* are stored in full: they are model-emitted JSON holding artifact
IDs only (invariant #2), and they are precisely what someone debugging a bad run needs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError

if TYPE_CHECKING:
    from collections.abc import Sequence

TRACE_FILENAME = "trace.jsonl"

KIND_HEADER = "header"
KIND_CALL = "call"
KIND_FOOTER = "footer"

_RAISED_BY = "bioflow.orchestration.trace"


def messages_digest(messages: Sequence[dict[str, JSONValue]]) -> str:
    """Hash a message list canonically, so an identical conversation hashes identically."""
    canonical = json.dumps(list(messages), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def arguments_digest(arguments: dict[str, JSONValue]) -> str:
    """Hash tool arguments canonically."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TraceHeader:
    """The parsed header of a trace, and everything replay needs from it."""

    run_id: str
    request: dict[str, JSONValue]
    cassette: str | None
    model: str
    prompt_version: str
    limits: dict[str, JSONValue]
    root: str | None


class TraceWriter:
    """Appends trace records for one run.

    Opens the file per record rather than holding a handle. An agent run is long and can
    end abruptly; a trace that survives a crash is worth more than one buffered
    efficiently into oblivion.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.seq = 0

    def _append(self, record: dict[str, JSONValue]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record["seq"] = self.seq
        record["ts"] = datetime.now(UTC).isoformat()
        self.seq += 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def header(
        self,
        *,
        run_id: str,
        request: dict[str, JSONValue],
        cassette: str | None,
        model: str,
        prompt_version: str,
        limits: dict[str, JSONValue],
        root: str | None = None,
    ) -> None:
        """Write the opening record."""
        self._append(
            {
                "kind": KIND_HEADER,
                "run_id": run_id,
                "request": request,
                "cassette": cassette,
                "model": model,
                "prompt_version": prompt_version,
                "limits": limits,
                "root": root,
            }
        )

    def call(
        self,
        *,
        state: str,
        prompt_version: str,
        messages_sha256: str,
        tool_call: dict[str, JSONValue] | None,
        validation_errors: list[JSONValue],
        repair_attempt: int,
        confirmed: bool | None,
        usage: dict[str, JSONValue],
        latency_ms: int,
        model: str,
        outcome: str,
    ) -> None:
        """Write one LLM-call record."""
        self._append(
            {
                "kind": KIND_CALL,
                "state": state,
                "prompt_version": prompt_version,
                "messages_sha256": messages_sha256,
                "tool_call": tool_call,
                "validation_errors": validation_errors,
                "repair_attempt": repair_attempt,
                "confirmed": confirmed,
                "usage": usage,
                "latency_ms": latency_ms,
                "model": model,
                "outcome": outcome,
            }
        )

    def footer(
        self,
        *,
        final_state: str,
        tool_calls: Sequence[str],
        metrics: dict[str, JSONValue],
        error: dict[str, JSONValue] | None = None,
    ) -> None:
        """Write the closing record."""
        self._append(
            {
                "kind": KIND_FOOTER,
                "final_state": final_state,
                "tool_calls": list(tool_calls),
                "metrics": metrics,
                "error": error,
            }
        )


def read_trace(path: Path) -> list[dict[str, JSONValue]]:
    """Parse a trace file into its records, skipping blank lines."""
    path = Path(path)
    if not path.is_file():
        raise BioflowError(
            StructuredError(
                error_code=ErrorCode.NOT_FOUND,
                message=f"no trace file at {path}",
                remediation="Point at runs/<run_id>/trace.jsonl from a completed run.",
                details={"path": str(path)},
                raised_by=_RAISED_BY,
            )
        )
    records: list[dict[str, JSONValue]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BioflowError(
                StructuredError(
                    error_code=ErrorCode.SCHEMA_MISMATCH,
                    message=f"trace {path.name} line {number} is not valid JSON: {exc}",
                    details={"path": str(path), "line": number},
                    raised_by=_RAISED_BY,
                )
            ) from exc
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def parse_header(records: Sequence[dict[str, JSONValue]]) -> TraceHeader:
    """Extract the header record, or explain that the trace is unusable."""
    for record in records:
        if record.get("kind") != KIND_HEADER:
            continue
        request = record.get("request")
        limits = record.get("limits")
        cassette = record.get("cassette")
        root = record.get("root")
        return TraceHeader(
            run_id=str(record.get("run_id", "")),
            request=dict(request) if isinstance(request, dict) else {},
            cassette=None if cassette is None else str(cassette),
            model=str(record.get("model", "")),
            prompt_version=str(record.get("prompt_version", "")),
            limits=dict(limits) if isinstance(limits, dict) else {},
            root=None if root is None else str(root),
        )
    raise BioflowError(
        StructuredError(
            error_code=ErrorCode.SCHEMA_MISMATCH,
            message="trace has no header record",
            remediation="A replayable trace is written by Agent.run(); this one is not.",
            details={"records": len(records)},
            raised_by=_RAISED_BY,
        )
    )


def recorded_tool_calls(records: Sequence[dict[str, JSONValue]]) -> list[str]:
    """Return the tool-call sequence a trace recorded.

    Read from the footer when there is one, since that is the agent's own summary.
    Falls back to reconstructing it from the successful call records, so a trace from a
    run that died before writing a footer is still comparable.
    """
    for record in records:
        if record.get("kind") == KIND_FOOTER:
            calls = record.get("tool_calls")
            if isinstance(calls, list):
                return [str(name) for name in calls]

    names: list[str] = []
    for record in records:
        if record.get("kind") != KIND_CALL or record.get("outcome") != "invoked":
            continue
        call = record.get("tool_call")
        if isinstance(call, dict) and "name" in call:
            names.append(str(call["name"]))
    return names


def recorded_confirmations(records: Sequence[dict[str, JSONValue]]) -> list[bool]:
    """Return the confirmation decisions a trace recorded, in order."""
    return [
        bool(record["confirmed"])
        for record in records
        if record.get("kind") == KIND_CALL and record.get("confirmed") is not None
    ]
