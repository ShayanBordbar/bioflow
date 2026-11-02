"""Record and replay of LLM calls.

Cassettes are what let the contract tier test anything that touches a model. Record a
call once against the real API; every run afterwards replays it from disk, in CI, with
no network and no key.

The cassette key is a hash of the *canonicalized* request, so a call that is logically
the same hashes the same: keys are sorted, separators tight, and volatile fields
(identifiers, timestamps, request metadata) stripped before hashing. Without that, a
cassette recorded on Tuesday misses on Wednesday for no visible reason.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from bioflow.adapters.llm.base import LLMClient, LLMResponse
from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CASSETTE_ROOT = Path("tests/fixtures/cassettes")

# Stripped before hashing: none of these change what was asked, but all of them change
# from run to run.
VOLATILE_KEYS: frozenset[str] = frozenset({"metadata", "timestamp", "created_at", "request_id"})
_VOLATILE_SUFFIX = re.compile(r".*_id$")

_RAISED_BY = "bioflow.adapters.llm.cassette"


def canonicalize(request: dict[str, JSONValue]) -> str:
    """Render a request as the canonical JSON its hash is taken over."""
    return json.dumps(_strip_volatile(request), sort_keys=True, separators=(",", ":"), default=str)


def request_hash(request: dict[str, JSONValue]) -> str:
    """Return the cassette key for a request."""
    return hashlib.sha256(canonicalize(request).encode("utf-8")).hexdigest()


def _strip_volatile(value: JSONValue) -> JSONValue:
    """Recursively drop fields that vary between identical logical requests."""
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS and not _VOLATILE_SUFFIX.match(key)
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


class CassetteClient:
    """Wraps an :class:`LLMClient` with record / replay / passthrough behaviour."""

    def __init__(
        self,
        inner: LLMClient | None,
        *,
        name: str,
        mode: str = "off",
        root: Path = DEFAULT_CASSETTE_ROOT,
        request_builder: object | None = None,
    ) -> None:
        self.inner = inner
        self.name = name
        self.mode = mode
        self.root = Path(root)
        # In replay there may be no live client at all, so the request body is built by
        # whatever object can build one -- normally the wrapped client itself.
        self._builder = request_builder if request_builder is not None else inner

    @property
    def directory(self) -> Path:
        """Where this cassette's recordings live."""
        return self.root / self.name

    def path_for(self, request: dict[str, JSONValue]) -> Path:
        """The file a given request records to or replays from."""
        return self.directory / f"{request_hash(request)}.json"

    def complete(
        self,
        messages: Sequence[dict[str, JSONValue]],
        *,
        tools: Sequence[dict[str, JSONValue]] | None = None,
        tool_choice: dict[str, JSONValue] | None = None,
        response_model: type[BaseModel] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Serve the call from disk, or make it and record it, per the mode."""
        if self.mode == "off":
            return self._call_inner(messages, tools, tool_choice, response_model, system)

        request = self._build_request(messages, tools, tool_choice, response_model, system)

        if self.mode == "replay":
            return self._replay(request)
        if self.mode == "record":
            response = self._call_inner(messages, tools, tool_choice, response_model, system)
            self._write(request, response)
            return response

        raise BioflowError(
            StructuredError(
                error_code=ErrorCode.VALIDATION_FAILED,
                message=f"unknown cassette mode {self.mode!r}",
                remediation="Use one of: off, record, replay.",
                raised_by=_RAISED_BY,
            )
        )

    # -- internals ------------------------------------------------------------------

    def _build_request(
        self,
        messages: Sequence[dict[str, JSONValue]],
        tools: Sequence[dict[str, JSONValue]] | None,
        tool_choice: dict[str, JSONValue] | None,
        response_model: type[BaseModel] | None,
        system: str | None,
    ) -> dict[str, JSONValue]:
        builder = getattr(self._builder, "build_request", None)
        if builder is not None:
            built: dict[str, JSONValue] = builder(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                response_model=response_model,
                system=system,
            )
            return built
        # No builder available: hash the call arguments themselves. Enough to key a
        # cassette, and it keeps fakes usable under record/replay.
        return {
            "messages": list(messages),
            "tools": list(tools or []),
            "tool_choice": tool_choice,
            "system": system,
            "response_model": None if response_model is None else response_model.__name__,
        }

    def _call_inner(
        self,
        messages: Sequence[dict[str, JSONValue]],
        tools: Sequence[dict[str, JSONValue]] | None,
        tool_choice: dict[str, JSONValue] | None,
        response_model: type[BaseModel] | None,
        system: str | None,
    ) -> LLMResponse:
        if self.inner is None:
            raise BioflowError(
                StructuredError(
                    error_code=ErrorCode.MISSING_INPUT,
                    message=f"cassette {self.name!r} has no live client to call",
                    remediation="Use mode='replay', or construct it with a real client.",
                    raised_by=_RAISED_BY,
                )
            )
        return self.inner.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            response_model=response_model,
            system=system,
        )

    def _replay(self, request: dict[str, JSONValue]) -> LLMResponse:
        path = self.path_for(request)
        if not path.is_file():
            raise self._miss(request, path)
        recorded = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse.model_validate(recorded["response"])

    def _write(self, request: dict[str, JSONValue], response: LLMResponse) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(request)
        payload = {
            "cassette": self.name,
            "request_sha256": request_hash(request),
            # The request is stored for a human reading the diff; only its hash is
            # ever used to look a recording up.
            "request": _strip_volatile(request),
            "response": response.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _miss(self, request: dict[str, JSONValue], path: Path) -> BioflowError:
        """Build the error for a cassette miss.

        This message is the feature: a miss is the failure a contributor will hit most
        often, and it has to say which cassette, which hash, and how to fix it.
        """
        available = sorted(p.stem for p in self.directory.glob("*.json"))
        return BioflowError(
            StructuredError(
                error_code=ErrorCode.NOT_FOUND,
                message=(
                    f"cassette miss for {self.name!r}: no recording at {path.name}. "
                    f"The request hashed to {request_hash(request)}, and the cassette "
                    f"directory holds {len(available)} recording(s)."
                ),
                remediation=(
                    f"Re-record with BIOFLOW_CASSETTE_MODE=record (a real "
                    f"ANTHROPIC_API_KEY is needed), or check whether the prompt, model, "
                    f"or tool schemas changed since {self.directory} was recorded."
                ),
                details={
                    "cassette": self.name,
                    "directory": str(self.directory),
                    "expected_sha256": request_hash(request),
                    "available": list(available[:20]),
                },
                raised_by=_RAISED_BY,
            )
        )
