"""SQLite-backed run state.

This is the source of truth for a run. ``runs/<run_id>/provenance.jsonl`` is an export
of the ``provenance`` table, written for humans and for tools that would rather stream a
file than open a database — never read back as authoritative.

Two departures from the obvious schema are deliberate:

* ``provenance`` is ordered by an autoincrementing ``seq``, not by ``ts``. Several
  events inside one fast step routinely share a millisecond, so a timestamp alone does
  not define an order and any test asserting one would be asserting a coincidence.
* ``runs`` carries ``python_version``. :class:`~bioflow.core.models.RunManifest` requires
  it, and the only other source at read time is the *reader's* interpreter, which would
  quietly put the wrong answer into a provenance record.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from bioflow.core.models import (
    ArtifactRef,
    BioflowError,
    ErrorCode,
    Evidence,
    JSONValue,
    ProvenanceEvent,
    RunManifest,
    RunStatus,
    StructuredError,
)

SCHEMA_VERSION = 1

# kv is keyed by (run_id, key). Rows that describe the database as a whole rather than
# one run use this reserved run_id, which no minted run_id can collide with.
GLOBAL_SCOPE = ""

_SCHEMA_VERSION_KEY = "schema_version"
_TOOL_VERSIONS_KEY = "tool_versions"
_LLM_USAGE_KEY = "llm_usage"
_METRICS_KEY = "metrics"
_RAISED_BY = "bioflow.core.runstore"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    bioflow_version TEXT NOT NULL,
    python_version  TEXT NOT NULL,
    git_commit      TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    json        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    json        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provenance (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    run_id   TEXT NOT NULL REFERENCES runs(run_id),
    ts       TEXT NOT NULL,
    json     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run  ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_run   ON evidence(run_id);
CREATE INDEX IF NOT EXISTS idx_provenance_run ON provenance(run_id, seq);
"""


class RunRecord:
    """One row of the ``runs`` table.

    A persistence DTO, not a domain model: it is what ``get_run`` and ``list_runs``
    return so callers can browse runs without paying to reassemble every artifact and
    evidence record into a full :class:`RunManifest`.
    """

    __slots__ = (
        "bioflow_version",
        "created_at",
        "git_commit",
        "python_version",
        "request",
        "run_id",
        "status",
    )

    def __init__(
        self,
        *,
        run_id: str,
        created_at: datetime,
        status: RunStatus,
        request: dict[str, JSONValue],
        bioflow_version: str,
        python_version: str,
        git_commit: str | None,
    ) -> None:
        self.run_id = run_id
        self.created_at = created_at
        self.status = status
        self.request = request
        self.bioflow_version = bioflow_version
        self.python_version = python_version
        self.git_commit = git_commit

    def __repr__(self) -> str:
        return f"RunRecord(run_id={self.run_id!r}, status={self.status.value!r})"


def _fail(
    code: ErrorCode,
    message: str,
    *,
    remediation: str | None = None,
    details: dict[str, JSONValue] | None = None,
) -> BioflowError:
    return BioflowError(
        StructuredError(
            error_code=code,
            message=message,
            remediation=remediation,
            details=details or {},
            raised_by=_RAISED_BY,
        )
    )


class RunStore:
    """Run, artifact, evidence, and provenance persistence for one workspace root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.db_path = self.root / "runs.db"
        self.root.mkdir(parents=True, exist_ok=True)

        # isolation_level=None turns off the driver's implicit transaction handling, so
        # PRAGMAs and DDL behave predictably and every write batch below is an explicit
        # BEGIN IMMEDIATE ... COMMIT.
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    # -- lifecycle ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Create the schema, or refuse a database written by a newer BioFlow."""
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kv'"
        ).fetchone()
        if existing is None:
            # executescript() commits any open transaction before it runs, so the DDL
            # cannot live inside our explicit BEGIN/COMMIT. It is idempotent anyway --
            # every statement is CREATE ... IF NOT EXISTS.
            self._conn.executescript(_SCHEMA)
            with self._transaction():
                self._conn.execute(
                    "INSERT INTO kv (run_id, key, value) VALUES (?, ?, ?)",
                    (GLOBAL_SCOPE, _SCHEMA_VERSION_KEY, json.dumps(SCHEMA_VERSION)),
                )
            return

        found = self._read_kv(GLOBAL_SCOPE, _SCHEMA_VERSION_KEY)
        version = int(found) if isinstance(found, int) else 0
        if version > SCHEMA_VERSION:
            raise _fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"database at {self.db_path} uses schema v{version}, "
                f"but this BioFlow understands v{SCHEMA_VERSION}",
                remediation="Upgrade BioFlow, or point at a different workspace root.",
                details={"found": version, "supported": SCHEMA_VERSION, "path": str(self.db_path)},
            )

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    class _Transaction:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def __enter__(self) -> sqlite3.Connection:
            self._conn.execute("BEGIN IMMEDIATE")
            return self._conn

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            if exc_type is None:
                self._conn.execute("COMMIT")
            else:
                self._conn.execute("ROLLBACK")

    def _transaction(self) -> RunStore._Transaction:
        return RunStore._Transaction(self._conn)

    # -- runs -----------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        request: dict[str, JSONValue],
        bioflow_version: str,
        python_version: str,
        created_at: datetime,
        git_commit: str | None = None,
        status: RunStatus = RunStatus.PENDING,
    ) -> None:
        """Insert a new run. The caller mints ``run_id`` via :mod:`bioflow.core.ids`."""
        with self._transaction():
            self._conn.execute(
                "INSERT INTO runs (run_id, created_at, status, request_json, "
                "bioflow_version, python_version, git_commit) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created_at.isoformat(),
                    status.value,
                    json.dumps(request),
                    bioflow_version,
                    python_version,
                    git_commit,
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return the run header, or ``None`` if there is no such run."""
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _row_to_record(row)

    def list_runs(self, limit: int | None = None) -> list[RunRecord]:
        """Return run headers, newest first."""
        sql = "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [_row_to_record(row) for row in self._conn.execute(sql, params)]

    def set_status(self, run_id: str, status: RunStatus) -> None:
        """Move a run to a new status."""
        with self._transaction():
            cursor = self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?", (status.value, run_id)
            )
        if cursor.rowcount == 0:
            raise _fail(
                ErrorCode.NOT_FOUND,
                f"no run {run_id}",
                details={"run_id": run_id},
            )

    # -- artifacts and evidence -----------------------------------------------------

    def add_artifact(self, run_id: str, ref: ArtifactRef) -> None:
        """Record an artifact against a run."""
        with self._transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts (artifact_id, run_id, json) VALUES (?, ?, ?)",
                (ref.artifact_id, run_id, ref.model_dump_json()),
            )

    def list_artifacts(self, run_id: str) -> list[ArtifactRef]:
        """Return every artifact recorded against a run, in insertion order."""
        rows = self._conn.execute(
            "SELECT json FROM artifacts WHERE run_id = ? ORDER BY rowid", (run_id,)
        )
        return [ArtifactRef.model_validate_json(str(row["json"])) for row in rows]

    def add_evidence(self, run_id: str, evidence: Evidence) -> None:
        """Record a piece of evidence against a run."""
        with self._transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO evidence (evidence_id, run_id, json) VALUES (?, ?, ?)",
                (evidence.evidence_id, run_id, evidence.model_dump_json()),
            )

    def list_evidence(self, run_id: str) -> list[Evidence]:
        """Return every piece of evidence recorded against a run, in insertion order."""
        rows = self._conn.execute(
            "SELECT json FROM evidence WHERE run_id = ? ORDER BY rowid", (run_id,)
        )
        return [Evidence.model_validate_json(str(row["json"])) for row in rows]

    # -- provenance -----------------------------------------------------------------

    def append_provenance(self, event: ProvenanceEvent) -> None:
        """Append an event to the run's provenance, then mirror it to JSONL.

        The database is committed *first*. A crash between the two steps leaves the
        mirror one line short, which a re-export fixes; the reverse order could leave
        the mirror claiming an event the source of truth never recorded.
        """
        payload = event.model_dump_json()
        with self._transaction():
            self._conn.execute(
                "INSERT INTO provenance (event_id, run_id, ts, json) VALUES (?, ?, ?, ?)",
                (event.event_id, event.run_id, event.timestamp.isoformat(), payload),
            )

        path = self.provenance_path(event.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND plus a single write keeps concurrent writers from interleaving, and
        # the fsync is what makes the line survive a power cut rather than a clean exit.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def list_provenance(self, run_id: str) -> list[ProvenanceEvent]:
        """Return a run's provenance in emission order (by ``seq``, never by ``ts``)."""
        rows = self._conn.execute(
            "SELECT json FROM provenance WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        return [ProvenanceEvent.model_validate_json(str(row["json"])) for row in rows]

    def provenance_path(self, run_id: str) -> Path:
        """Return the JSONL mirror path for a run."""
        return self.runs_dir / run_id / "provenance.jsonl"

    # -- key/value ------------------------------------------------------------------

    def set_kv(self, run_id: str, key: str, value: JSONValue) -> None:
        """Store a JSON-encodable value against a run (plan and trace pointers, etc.)."""
        with self._transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, json.dumps(value)),
            )

    def get_kv(self, run_id: str, key: str) -> JSONValue:
        """Return a stored value, or ``None`` when the key is absent."""
        return self._read_kv(run_id, key)

    def _read_kv(self, run_id: str, key: str) -> JSONValue:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE run_id = ? AND key = ?", (run_id, key)
        ).fetchone()
        if row is None:
            return None
        decoded: JSONValue = json.loads(str(row["value"]))
        return decoded

    # -- manifest -------------------------------------------------------------------

    def get_manifest(self, run_id: str) -> RunManifest:
        """Reassemble the full manifest for a run."""
        record = self.get_run(run_id)
        if record is None:
            raise _fail(
                ErrorCode.NOT_FOUND,
                f"no run {run_id}",
                details={"run_id": run_id},
            )
        tool_versions = self._read_kv(run_id, _TOOL_VERSIONS_KEY)
        llm_usage = self._read_kv(run_id, _LLM_USAGE_KEY)
        metrics = self._read_kv(run_id, _METRICS_KEY)
        return RunManifest(
            run_id=record.run_id,
            created_at=record.created_at,
            request=record.request,
            bioflow_version=record.bioflow_version,
            python_version=record.python_version,
            git_commit=record.git_commit,
            artifacts=self.list_artifacts(run_id),
            evidence=self.list_evidence(run_id),
            status=record.status,
            tool_versions=dict(tool_versions) if isinstance(tool_versions, dict) else {},
            llm_usage=dict(llm_usage) if isinstance(llm_usage, dict) else {},
            metrics=dict(metrics) if isinstance(metrics, dict) else {},
        )

    def set_tool_versions(self, run_id: str, tool_versions: dict[str, str]) -> None:
        """Record the tool versions this run used."""
        self.set_kv(run_id, _TOOL_VERSIONS_KEY, dict(tool_versions))

    def set_llm_usage(self, run_id: str, usage: dict[str, JSONValue]) -> None:
        """Record this run's LLM token and cost accounting."""
        self.set_kv(run_id, _LLM_USAGE_KEY, dict(usage))

    def set_metrics(self, run_id: str, metrics: dict[str, JSONValue]) -> None:
        """Record how this run behaved: call counts, repair rates, transition tallies."""
        self.set_kv(run_id, _METRICS_KEY, dict(metrics))


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    """Convert a ``runs`` row into a typed record.

    sqlite3.Row lookups are ``Any``, so every column is converted here rather than
    leaking an untyped value into the rest of the codebase.
    """
    request: JSONValue = json.loads(str(row["request_json"]))
    git_commit = row["git_commit"]
    return RunRecord(
        run_id=str(row["run_id"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        status=RunStatus(str(row["status"])),
        request=dict(request) if isinstance(request, dict) else {},
        bioflow_version=str(row["bioflow_version"]),
        python_version=str(row["python_version"]),
        git_commit=None if git_commit is None else str(git_commit),
    )
