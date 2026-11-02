"""The handle a run is carried around by.

A :class:`RunContext` bundles the three things every unit of work needs — where its
files go, where its bytes are stored, and where its provenance is recorded — so that
call sites take one parameter instead of four.

:meth:`RunContext.step` is the part that matters. Provenance recorded only on the happy
path is worse than none, because the gaps look like work that never happened rather than
work that failed. ``step`` therefore emits an event on both paths, and re-raises.
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

import bioflow
from bioflow.core.ids import new_event_id, new_run_id
from bioflow.core.models import (
    Actor,
    ArtifactRef,
    BioflowError,
    ErrorCode,
    EventStatus,
    Evidence,
    JSONValue,
    ProvenanceEvent,
    RunManifest,
    RunStatus,
    StructuredError,
)
from bioflow.core.runstore import RunStore
from bioflow.core.store import ArtifactStore
from bioflow.core.usage import UsageAccumulator

if TYPE_CHECKING:
    from collections.abc import Iterator

RUN_SUBDIRS = ("inputs", "work", "outputs", "logs", "evidence", "report")


class Step:
    """Mutable scratch handed to a ``with ctx.step(...)`` body.

    The body attaches the artifacts it consumed and produced; ``step`` folds them into
    the emitted event so the provenance record links inputs to outputs instead of just
    noting that something happened.
    """

    __slots__ = ("inputs", "outputs", "params_sha256")

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.params_sha256: str | None = None

    def add_input(self, ref: ArtifactRef) -> None:
        """Record that this step consumed ``ref``."""
        self.inputs.append(ref.artifact_id)

    def add_output(self, ref: ArtifactRef) -> None:
        """Record that this step produced ``ref``."""
        self.outputs.append(ref.artifact_id)


class RunContext:
    """Everything one run needs: its directory, its object store, its run store."""

    def __init__(
        self, run_id: str, run_dir: Path, store: ArtifactStore, runstore: RunStore
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.store = store
        self.runstore = runstore
        # Every LLM call in this run folds into here; flush_usage() writes the totals
        # into the manifest.
        self.usage = UsageAccumulator()

    # -- construction ---------------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        request: dict[str, JSONValue] | None = None,
        *,
        bioflow_version: str | None = None,
        git_commit: str | None = None,
        run_id: str | None = None,
    ) -> Self:
        """Start a new run: mint the ID, build the directory, insert the run row.

        ``bioflow_version`` defaults to the in-tree literal. The CLI passes
        ``bioflow.cli.get_version()`` instead, which prefers the installed distribution
        version; ``core`` cannot import that itself, because imports only point down.
        """
        root = Path(root)
        run_id = run_id or new_run_id()
        payload = request or {}

        runstore = RunStore(root)
        run_dir = runstore.runs_dir / run_id
        for name in RUN_SUBDIRS:
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        (run_dir / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        runstore.create_run(
            run_id=run_id,
            request=payload,
            bioflow_version=bioflow_version or bioflow.__version__,
            python_version=platform.python_version(),
            created_at=datetime.now(UTC),
            git_commit=git_commit,
        )
        return cls(run_id, run_dir, ArtifactStore(root), runstore)

    @classmethod
    def open(cls, root: Path, run_id: str) -> Self:
        """Reattach to an existing run."""
        root = Path(root)
        runstore = RunStore(root)
        if runstore.get_run(run_id) is None:
            runstore.close()
            raise BioflowError(
                StructuredError(
                    error_code=ErrorCode.NOT_FOUND,
                    message=f"no run {run_id} in {root}",
                    details={"run_id": run_id, "root": str(root)},
                    raised_by="bioflow.core.runctx",
                )
            )
        return cls(run_id, runstore.runs_dir / run_id, ArtifactStore(root), runstore)

    def close(self) -> None:
        """Release the run store connection."""
        self.runstore.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- recording ------------------------------------------------------------------

    def emit(self, event: ProvenanceEvent) -> None:
        """Append a provenance event."""
        self.runstore.append_provenance(event)

    def record(self, artifact: ArtifactRef) -> ArtifactRef:
        """Record an artifact against this run and return it, for chaining."""
        self.runstore.add_artifact(self.run_id, artifact)
        return artifact

    def record_evidence(self, evidence: Evidence) -> Evidence:
        """Record a piece of evidence against this run and return it, for chaining."""
        self.runstore.add_evidence(self.run_id, evidence)
        return evidence

    def set_status(self, status: RunStatus) -> None:
        """Move this run to a new status."""
        self.runstore.set_status(self.run_id, status)

    def flush_usage(self) -> None:
        """Write accumulated LLM spend into the run's manifest."""
        self.runstore.set_llm_usage(self.run_id, self.usage.as_dict())

    def manifest(self) -> RunManifest:
        """Export this run's current state as a manifest."""
        return self.runstore.get_manifest(self.run_id)

    # -- the timed step ---------------------------------------------------------------

    @contextmanager
    def step(
        self,
        name: str,
        tool: str | None = None,
        tool_version: str | None = None,
        *,
        actor: Actor = Actor.SYSTEM,
    ) -> Iterator[Step]:
        """Time a block and emit exactly one provenance event describing it.

        A failing block emits an ``ERROR`` event and then re-raises. A
        :class:`BioflowError` keeps its own structured payload; anything else is wrapped
        as ``INTERNAL``, since an unexpected exception type is by definition a bug rather
        than a diagnosed condition.
        """
        handle = Step()
        started = time.monotonic()

        def finish(status: EventStatus, error: StructuredError | None) -> None:
            self.emit(
                ProvenanceEvent(
                    event_id=new_event_id(),
                    run_id=self.run_id,
                    timestamp=datetime.now(UTC),
                    actor=actor,
                    action=name,
                    tool=tool,
                    tool_version=tool_version,
                    inputs=tuple(handle.inputs),
                    outputs=tuple(handle.outputs),
                    params_sha256=handle.params_sha256,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    status=status,
                    error=error,
                )
            )

        try:
            yield handle
        except BioflowError as exc:
            finish(EventStatus.ERROR, exc.error)
            raise
        except Exception as exc:
            finish(
                EventStatus.ERROR,
                StructuredError(
                    error_code=ErrorCode.INTERNAL,
                    message=f"{type(exc).__name__}: {exc}",
                    details={"step": name},
                    raised_by=tool or name,
                ),
            )
            raise
        else:
            finish(EventStatus.OK, None)
