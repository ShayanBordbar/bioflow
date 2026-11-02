"""Core domain models shared by every BioFlow layer.

These are value objects, not behaviour. They encode the repository invariants that
everything else relies on:

* an artifact records *what* it is, *who* produced it, and its content hash;
* evidence is typed, so a measurement can never be confused with a model prediction;
* provenance events form an append-only record of who did what to which artifacts;
* failures cross capability boundaries as :class:`StructuredError`, never as bare
  exceptions with free-text messages.

All models except :class:`RunManifest` are frozen. Frozen models store their
sequences as tuples so a provenance list cannot be mutated in place. This does not
make them hashable: several carry ``dict`` fields, which are unhashable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
"""Anything representable in JSON. Used for free-form metadata and schema bodies."""


# --------------------------------------------------------------------------------------
# Identifier and value formats
# --------------------------------------------------------------------------------------

ARTIFACT_ID_PATTERN = r"^art_[0-9a-f]{12}$"
EVIDENCE_ID_PATTERN = r"^E[0-9a-f]{6}$"
RUN_ID_PATTERN = r"^run_[0-9a-f]{12}$"
EVENT_ID_PATTERN = r"^ev_[0-9a-f]{12}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
SEMVER_PATTERN = (
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def _require_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalize aware ones to UTC.

    An unanchored provenance timestamp is worse than no timestamp: it silently means
    something different on every machine that reads it.
    """
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


ArtifactId = Annotated[str, StringConstraints(pattern=ARTIFACT_ID_PATTERN)]
EvidenceId = Annotated[str, StringConstraints(pattern=EVIDENCE_ID_PATTERN)]
RunId = Annotated[str, StringConstraints(pattern=RUN_ID_PATTERN)]
EventId = Annotated[str, StringConstraints(pattern=EVENT_ID_PATTERN)]
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]
SemVer = Annotated[str, StringConstraints(pattern=SEMVER_PATTERN)]
UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]


# --------------------------------------------------------------------------------------
# Closed enumerations
# --------------------------------------------------------------------------------------


class ArtifactType(StrEnum):
    """The closed set of artifact shapes BioFlow knows how to move around."""

    FASTA = "fasta"
    FASTQ = "fastq"
    TSV = "tsv"
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    PARQUET = "parquet"
    CHECKPOINT = "checkpoint"
    LOG = "log"
    DIRECTORY = "directory"
    OTHER = "other"


class ErrorCode(StrEnum):
    """Closed failure taxonomy. Every error crossing a boundary maps to one member."""

    VALIDATION_FAILED = "VALIDATION_FAILED"
    MISSING_INPUT = "MISSING_INPUT"
    AMBIGUOUS_INPUT = "AMBIGUOUS_INPUT"
    UNSUPPORTED_ASSAY = "UNSUPPORTED_ASSAY"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    EXTERNAL_TOOL_FAILED = "EXTERNAL_TOOL_FAILED"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INTERNAL = "INTERNAL"


class EvidenceType(StrEnum):
    """How much epistemic weight a statement carries.

    The ordering is deliberate: ``MEASURED`` is an observation, ``MODEL_PREDICTED`` is a
    model output, and the two must never be phrased interchangeably in a report.
    """

    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    MODEL_PREDICTED = "MODEL_PREDICTED"
    MODEL_INTERPRETED = "MODEL_INTERPRETED"
    HYPOTHESIS = "HYPOTHESIS"


class Actor(StrEnum):
    """Who performed a provenance-recorded action."""

    USER = "USER"
    AGENT = "AGENT"
    TOOL = "TOOL"
    ADAPTER = "ADAPTER"
    SYSTEM = "SYSTEM"


class EventStatus(StrEnum):
    """Outcome of a single provenance event."""

    OK = "OK"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class RunStatus(StrEnum):
    """Outcome of a whole run.

    Distinct from :class:`EventStatus` despite the overlapping members: a run can be
    in flight, and it can end in ``NEEDS_REVIEW``, which is a correct answer for
    ambiguous input rather than a failure.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    OK = "OK"
    ERROR = "ERROR"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class SideEffect(StrEnum):
    """The strongest effect a tool may have on the world outside the process."""

    NONE = "NONE"
    READ = "READ"
    WRITE = "WRITE"
    NETWORK = "NETWORK"
    EXECUTE = "EXECUTE"


class CostClass(StrEnum):
    """Rough order-of-magnitude cost of invoking a tool."""

    FREE = "FREE"
    CHEAP = "CHEAP"
    MODERATE = "MODERATE"
    EXPENSIVE = "EXPENSIVE"


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Base for immutable domain records.

    ``extra="forbid"`` matters more than it looks: a misspelled provenance field must
    fail loudly rather than be silently dropped on the floor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def _format_bytes(size: int) -> str:
    """Render a byte count in the largest unit that keeps it under 1024."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


class ArtifactRef(_FrozenModel):
    """A content-addressed handle to something a tool produced.

    ``produced_by`` follows the convention ``name@version`` (for example
    ``kallisto@0.50.1``). It is documented rather than pattern-enforced because
    upstream tools report version strings in whatever shape they like.
    """

    artifact_id: ArtifactId
    artifact_type: ArtifactType
    uri: str = Field(min_length=1)
    sha256: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    media_type: str | None = None
    produced_by: str | None = None
    run_id: RunId | None = None
    schema_version: str = "1"
    created_at: UtcDatetime
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    def short(self) -> str:
        """Return a one-line summary suitable for LLM context.

        Deliberately omits ``uri``: the LLM handles artifact IDs only and must never
        see a filesystem path. Optional parts are dropped when unset.
        """
        parts = [f"{self.artifact_id} [{self.artifact_type.value}]"]
        if self.size_bytes is not None:
            parts.append(_format_bytes(self.size_bytes))
        if self.sha256 is not None:
            parts.append(f"sha256:{self.sha256[:8]}…")
        if self.produced_by is not None:
            parts.append(f"via {self.produced_by}")
        return " ".join(parts)


class StructuredError(_FrozenModel):
    """A machine-readable failure.

    ``retryable`` is the caller's cue, not a promise: it says a retry is *permitted*,
    not that it will succeed.
    """

    error_code: ErrorCode
    message: str
    remediation: str | None = None
    details: dict[str, JSONValue] = Field(default_factory=dict)
    retryable: bool = False
    raised_by: str | None = None


class BioflowError(Exception):
    """Exception wrapper carrying a :class:`StructuredError` payload."""

    def __init__(self, error: StructuredError) -> None:
        super().__init__(error.message)
        self.error = error

    @property
    def error_code(self) -> ErrorCode:
        """Shortcut to the wrapped error's code, for ``except`` bodies."""
        return self.error.error_code


class Evidence(_FrozenModel):
    """A single typed claim, traceable back to the artifacts that support it.

    Report sentences cite these by ``evidence_id``; a sentence with no citation fails
    the claim linter.
    """

    evidence_id: EvidenceId
    evidence_type: EvidenceType
    statement: str
    value: float | str | dict[str, JSONValue] | None = None
    units: str | None = None
    artifacts: tuple[ArtifactId, ...] = ()
    produced_by: str
    provenance_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


class ProvenanceEvent(_FrozenModel):
    """One append-only record of an action taken during a run.

    ``params_sha256`` hashes the canonicalized parameters rather than storing them, so
    two invocations can be compared for identity without keeping every payload.
    """

    event_id: EventId
    run_id: RunId
    timestamp: UtcDatetime
    actor: Actor
    action: str
    tool: str | None = None
    tool_version: str | None = None
    inputs: tuple[ArtifactId, ...] = ()
    outputs: tuple[ArtifactId, ...] = ()
    params_sha256: Sha256 | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    status: EventStatus = EventStatus.OK
    error: StructuredError | None = None


class ToolSpec(_FrozenModel):
    """The declared contract of a callable tool.

    ``input_schema`` and ``output_schema`` are JSON Schema documents; they are what the
    LLM is shown and what its emitted call is validated against.
    """

    name: str = Field(min_length=1)
    version: SemVer
    summary: str
    input_schema: dict[str, JSONValue] = Field(default_factory=dict)
    output_schema: dict[str, JSONValue] = Field(default_factory=dict)
    side_effect: SideEffect = SideEffect.NONE
    cost_class: CostClass = CostClass.CHEAP
    requires_confirmation: bool = False
    tags: tuple[str, ...] = ()


class RunManifest(BaseModel):
    """Everything produced by one run, in one serializable record.

    Unlike the other models this is mutable, because a run accumulates artifacts and
    evidence as it proceeds. SQLite remains the source of truth; a manifest is an
    export of it.

    ``llm_usage`` is a plain mapping with the expected keys ``model``,
    ``prompt_version``, ``input_tokens``, ``output_tokens``, ``cost_usd`` and
    ``n_calls``. It stays untyped until a second consumer justifies a model.

    ``metrics`` carries how the run behaved rather than what it cost -- the agent loop's
    call counts, repair rates, and transition tallies. Kept separate from ``llm_usage``
    because spend and behaviour are read by different people for different reasons, and
    folding them together would make neither queryable.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: RunId
    created_at: UtcDatetime
    request: dict[str, JSONValue] = Field(default_factory=dict)
    bioflow_version: str
    python_version: str
    git_commit: str | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    tool_versions: dict[str, str] = Field(default_factory=dict)
    llm_usage: dict[str, JSONValue] = Field(default_factory=dict)
    metrics: dict[str, JSONValue] = Field(default_factory=dict)
