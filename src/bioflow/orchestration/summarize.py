"""Turning an artifact into something a model can reason about.

An identifier and a size tell the model an artifact *exists*. To triage an assay it needs
the column names, the shape, and a few rows. This module produces that, under two hard
constraints.

**Nothing is read whole.** Every implementation reads a capped prefix or seeks to a tail.
A 500 MB table is described by its first megabyte, and when that happens the output says
so -- statistics over a sample presented as statistics over the file would be a confident
wrong answer, which is the most expensive thing this system can produce.

**Nothing emits a path.** Invariant #2 is usually easy because the model only ever handles
identifiers, but summarizers are the one place where *content* reaches the prompt, and
content contains paths: a log line naming a working directory, a TSV cell holding a
filename. Every summarizer's output is therefore redacted centrally on the way out, rather
than each one being trusted to have thought about it.

Detail level is chosen by ``budget_tokens``. At a large budget a summarizer produces its
full description; at a small one the same function degrades to a single headline, because
lines are ordered most-informative-first and the tail is what gets cut.
"""

from __future__ import annotations

import json
import re
import statistics

# Callable is used in a module-level type alias, so it has to exist at run time.
from collections.abc import Callable
from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

from bioflow.core.models import ArtifactRef, ArtifactType, JSONValue
from bioflow.orchestration.tokens import fit, truncation_marker

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from bioflow.core.store import ArtifactStore

# How much of a file any summarizer may look at. Generous enough to characterize a table
# properly, small enough that a multi-gigabyte artifact costs the same as a small one.
SCAN_BYTES = 1024 * 1024
TAIL_BYTES = 64 * 1024
# JSON has no useful prefix -- a truncated document does not parse -- so oversized ones
# are reported rather than guessed at.
JSON_MAX_BYTES = 2 * 1024 * 1024

HEAD_ROWS = 3
LOG_TAIL_LINES = 40
LOG_MATCH_CAP = 20
FASTA_EXAMPLES = 2
FASTA_RECORD_CHARS = 60

LOG_PATTERN = re.compile(r"ERROR|WARN|Exception", re.IGNORECASE)

# Anything that looks like a filesystem location. Deliberately broad: a false positive
# redacts a string that resembles a path, which costs nothing, while a false negative
# puts a real path in front of the model.
_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/(?:[\w.+@-]+/)+[\w.+@-]*"),  # POSIX, two or more segments
    re.compile(r"[A-Za-z]:\\(?:[\w.+@ -]+\\?)*"),  # Windows drive paths
    re.compile(r"\b(?:cas|file)://\S*"),  # store and file URIs
)
REDACTION = "<path>"


@runtime_checkable
class Source(Protocol):
    """Bounded read access to an artifact's bytes."""

    @property
    def size_bytes(self) -> int:
        """Total size of the object, without reading it."""
        ...

    def open(self) -> BinaryIO:
        """Open the object for binary reading."""
        ...


class StoreSource:
    """A :class:`Source` backed by the content-addressed store."""

    def __init__(self, store: ArtifactStore, ref: ArtifactRef) -> None:
        self._store = store
        self._ref = ref

    @property
    def size_bytes(self) -> int:
        """The recorded size, falling back to a stat when the ref carries none."""
        if self._ref.size_bytes is not None:
            return self._ref.size_bytes
        return self._store.resolve(self._ref).stat().st_size

    def open(self) -> BinaryIO:
        """Open the stored object."""
        return self._store.open(self._ref)


Summarizer = Callable[[ArtifactRef, Source, int], str]
"""``(ref, source, budget_tokens) -> str``.

The ``source`` is what makes bounded reading possible and injectable; a summarizer that
cannot read bytes cannot summarize, and threading the opener through explicitly is what
lets a test prove nothing oversized was read.
"""

_SUMMARIZERS: dict[str, Summarizer] = {}


def summarizer(*artifact_types: str) -> Callable[[Summarizer], Summarizer]:
    """Register a summarizer for one or more artifact types.

    Mirrors the ``@tool`` decorator's shape. Registering the same type twice raises at
    import time, because two summarizers for one type means one of them is dead and
    nobody would notice which.
    """

    def decorate(fn: Summarizer) -> Summarizer:
        for name in artifact_types:
            if name in _SUMMARIZERS:
                raise ValueError(f"a summarizer for {name!r} is already registered")
            _SUMMARIZERS[name] = fn
        return fn

    return decorate


def registered_types() -> list[str]:
    """Return every artifact type that has a summarizer, sorted."""
    return sorted(_SUMMARIZERS)


def get_summarizer(artifact_type: ArtifactType | str) -> Summarizer | None:
    """Return the summarizer for a type, or ``None`` when there is none."""
    key = artifact_type.value if isinstance(artifact_type, ArtifactType) else artifact_type
    return _SUMMARIZERS.get(key)


def headline(ref: ArtifactRef) -> str:
    """The irreducible one-line description, used when nothing else fits."""
    produced = f" via {ref.produced_by}" if ref.produced_by else ""
    return f"{ref.artifact_type.value} artifact{produced}"


def redact_paths(text: str) -> str:
    """Replace anything path-shaped with a placeholder.

    Applied to every summarizer's output. Content is the one channel through which a
    filesystem path can reach the model, so it is closed here once rather than in five
    places that each have to remember.
    """
    for pattern in _PATH_PATTERNS:
        text = pattern.sub(REDACTION, text)
    return text


def summarize(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """Describe an artifact within a token budget, never emitting a path."""
    if budget_tokens <= 0:
        return redact_paths(headline(ref))
    fn = _SUMMARIZERS.get(ref.artifact_type.value)
    if fn is None:
        return redact_paths(headline(ref))
    try:
        rendered = fn(ref, source, budget_tokens)
    except Exception as exc:
        # A summary is a convenience, never a correctness requirement. A malformed file
        # must not fail the run that was about to diagnose it.
        rendered = f"{headline(ref)} (could not summarize: {type(exc).__name__})"
    return redact_paths(fit(rendered, budget_tokens))


# --------------------------------------------------------------------------------------
# Bounded reading
# --------------------------------------------------------------------------------------


def _read_head(source: Source, limit: int = SCAN_BYTES) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes from the start. Returns ``(data, was_truncated)``."""
    with source.open() as handle:
        # One byte over the limit is how truncation is detected without a second call.
        data = handle.read(limit + 1)
    if len(data) > limit:
        return data[:limit], True
    return data, False


def _read_tail(source: Source, limit: int = TAIL_BYTES) -> tuple[bytes, bool]:
    """Seek to the last ``limit`` bytes and read them."""
    size = source.size_bytes
    with source.open() as handle:
        if size > limit:
            handle.seek(size - limit)
            return handle.read(limit), True
        return handle.read(limit), False


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _lines(data: bytes, truncated: bool, *, drop_first: bool = False) -> list[str]:
    """Split a byte window into whole lines, discarding partial ones at the edges."""
    text = _decode(data)
    rows = text.splitlines()
    if truncated and rows:
        # The window cut mid-line at whichever end we read from.
        rows = rows[1:] if drop_first else rows[:-1]
    return rows


def _scan_note(source: Source, truncated: bool) -> str | None:
    """Say that statistics come from a sample, when they do."""
    if not truncated:
        return None
    return truncation_marker(SCAN_BYTES, source.size_bytes, "bytes scanned")


# --------------------------------------------------------------------------------------
# Tabular
# --------------------------------------------------------------------------------------


def _infer_dtype(values: Iterable[str]) -> str:
    """Report the narrowest type every non-empty value fits."""
    seen = False
    integral = True
    for value in values:
        text = value.strip()
        if not text:
            continue
        seen = True
        try:
            number = float(text)
        except ValueError:
            return "str"
        if integral and (text.lstrip("+-").isdigit() and float(number).is_integer()):
            continue
        integral = False
    if not seen:
        return "empty"
    return "int" if integral else "float"


def _numeric_stats(values: Sequence[str]) -> str | None:
    """Render min/median/max for a column, or ``None`` if it is not numeric."""
    numbers: list[float] = []
    for value in values:
        text = value.strip()
        if not text:
            continue
        try:
            numbers.append(float(text))
        except ValueError:
            return None
    if not numbers:
        return None
    return (
        f"min={_number(min(numbers))} "
        f"median={_number(statistics.median(numbers))} "
        f"max={_number(max(numbers))}"
    )


def _number(value: float) -> str:
    """Render a number without trailing float noise."""
    return str(int(value)) if float(value).is_integer() else f"{value:.4g}"


def _tabular(ref: ArtifactRef, source: Source, budget_tokens: int, delimiter: str) -> str:
    """Describe a delimited table from a bounded prefix."""
    data, truncated = _read_head(source)
    rows = _lines(data, truncated)
    if not rows:
        return f"{ref.artifact_type.value}, empty"

    header = rows[0].split(delimiter)
    body = [row.split(delimiter) for row in rows[1:] if row]
    columns = len(header)

    shape = f"{len(body)} rows x {columns} columns"
    if truncated:
        shape += " (in the scanned prefix)"
    lines = [f"{ref.artifact_type.value}: {shape}"]

    # Columns, with a dtype inferred from the scanned rows.
    described: list[str] = []
    for index, name in enumerate(header):
        column = [row[index] for row in body if index < len(row)]
        described.append(f"{name.strip()}:{_infer_dtype(column)}")
    lines.append("columns: " + ", ".join(described))

    for row in body[:HEAD_ROWS]:
        lines.append("  " + delimiter.join(row))
    if len(body) > HEAD_ROWS:
        lines.append("  " + truncation_marker(HEAD_ROWS, len(body), "rows shown"))

    for index, name in enumerate(header):
        column = [row[index] for row in body if index < len(row)]
        stats = _numeric_stats(column)
        if stats is not None:
            lines.append(f"{name.strip()}: {stats}")

    note = _scan_note(source, truncated)
    if note is not None:
        lines.append(f"statistics computed over the scanned prefix {note}")
    return "\n".join(lines)


@summarizer(ArtifactType.TSV.value)
def summarize_tsv(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """Shape, columns with dtypes, head rows, and numeric ranges for a TSV."""
    return _tabular(ref, source, budget_tokens, "\t")


@summarizer(ArtifactType.CSV.value)
def summarize_csv(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """Shape, columns with dtypes, head rows, and numeric ranges for a CSV."""
    return _tabular(ref, source, budget_tokens, ",")


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------

# Keys whose scalar values are structural rather than scientific, and therefore safe and
# useful to show. A value not on this list is described by its type only: a p-value or a
# sample identifier reaching the prompt as a literal invites the model to reason from a
# number it was never asked to interpret.
VALUE_WHITELIST: frozenset[str] = frozenset(
    {
        "assay",
        "genome",
        "id",
        "kind",
        "name",
        "organism",
        "platform",
        "schema_version",
        "species",
        "status",
        "strandedness",
        "tool",
        "type",
        "version",
    }
)


def _skeleton(value: JSONValue, depth: int, key: str | None = None) -> str:
    """Render a JSON value's shape, descending at most ``depth`` levels."""
    if isinstance(value, dict):
        if depth <= 0:
            return f"object({len(value)} keys)"
        parts = [f"{name}: {_skeleton(item, depth - 1, name)}" for name, item in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        if not value:
            return "array(0)"
        if depth <= 0:
            return f"array({len(value)})"
        return f"array({len(value)}) of {_skeleton(value[0], depth - 1)}"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if key is not None and key.lower() in VALUE_WHITELIST:
        return json.dumps(value)
    return "string"


@summarizer(ArtifactType.JSON.value)
def summarize_json(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """Key skeleton to depth two, with whitelisted scalar values shown."""
    size = source.size_bytes
    if size > JSON_MAX_BYTES:
        # A truncated JSON document does not parse, so there is no partial answer to give.
        return f"json, {size} bytes: too large to parse for a summary"
    data, _ = _read_head(source, JSON_MAX_BYTES)
    parsed: JSONValue = json.loads(_decode(data))
    return f"json, {size} bytes\n{_skeleton(parsed, 2)}"


# --------------------------------------------------------------------------------------
# FASTA
# --------------------------------------------------------------------------------------


@summarizer(ArtifactType.FASTA.value)
def summarize_fasta(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """Record count, length distribution, mean GC, and two truncated records."""
    data, truncated = _read_head(source)
    rows = _lines(data, truncated)

    lengths: list[int] = []
    examples: list[str] = []
    current: list[str] = []
    header: str | None = None
    gc = 0
    total = 0

    def close() -> None:
        nonlocal header, gc, total
        if header is None:
            return
        sequence = "".join(current)
        lengths.append(len(sequence))
        upper = sequence.upper()
        gc += upper.count("G") + upper.count("C")
        total += len(upper)
        if len(examples) < FASTA_EXAMPLES:
            shown = sequence[:FASTA_RECORD_CHARS]
            suffix = "..." if len(sequence) > FASTA_RECORD_CHARS else ""
            examples.append(f"  {header}\n  {shown}{suffix}")
        header = None
        current.clear()

    for row in rows:
        if row.startswith(">"):
            close()
            header = row
        elif header is not None:
            current.append(row.strip())
    close()

    if not lengths:
        return "fasta, no records in the scanned prefix"

    counted = f"{len(lengths)} records"
    if truncated:
        counted += " in the scanned prefix"
    lines = [f"fasta: {counted}"]
    lines.append(
        f"length min={min(lengths)} median={_number(statistics.median(lengths))} max={max(lengths)}"
    )
    if total:
        lines.append(f"GC mean={gc / total:.3f}")
    lines.extend(examples)

    note = _scan_note(source, truncated)
    if note is not None:
        lines.append(note)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------------------


@summarizer(ArtifactType.LOG.value)
def summarize_log(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """The tail, plus whatever in it looks like a failure.

    Only the tail is scanned. A log's ending is where its failure is, and reading a
    gigabyte to find every WARN would cost more than the answer is worth -- so the output
    says which window it looked at rather than implying it saw everything.
    """
    data, truncated = _read_tail(source)
    rows = _lines(data, truncated, drop_first=True)

    lines = [f"log, {source.size_bytes} bytes"]
    matches = [row for row in rows if LOG_PATTERN.search(row)]
    if matches:
        shown = matches[:LOG_MATCH_CAP]
        lines.append(f"matching ERROR|WARN|Exception ({len(matches)} in the scanned tail):")
        lines.extend(f"  {row}" for row in shown)
        if len(matches) > LOG_MATCH_CAP:
            lines.append("  " + truncation_marker(LOG_MATCH_CAP, len(matches), "matches"))

    tail = rows[-LOG_TAIL_LINES:]
    lines.append(f"last {len(tail)} lines:")
    lines.extend(f"  {row}" for row in tail)

    if truncated:
        lines.append(truncation_marker(TAIL_BYTES, source.size_bytes, "bytes scanned (tail)"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Opaque
# --------------------------------------------------------------------------------------


@summarizer(ArtifactType.HTML.value)
def summarize_html(ref: ArtifactRef, source: Source, budget_tokens: int) -> str:
    """Report size only.

    A rendered report is for a human. Feeding its markup to the model would spend a large
    part of the budget on tags in exchange for nothing it can act on.
    """
    return f"binary report, {source.size_bytes} bytes"
