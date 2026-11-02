"""Three ways to execute a sidecar job.

All three satisfy :class:`SidecarRunner`, so a capability takes one by constructor
injection and never learns which it got. That is what lets the contract tier test a
capability against recorded fixtures while the integration tier runs the real thing.

The part worth reading closely is the timeout. ``subprocess.run(timeout=...)`` signals
only the direct child, so a sidecar that forked a CUDA worker and then hung would leave
that worker holding a GPU indefinitely -- the orchestrator would report a clean timeout
and the machine would quietly stay broken. ``start_new_session=True`` puts the child in
its own process group, and the timeout path kills the group.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bioflow.adapters.sidecar import _workdir
from bioflow.adapters.sidecar.contract import (
    SidecarJob,
    SidecarResult,
    SidecarStatus,
    job_sha256,
)
from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioflow.core.store import ArtifactStore

DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_SEQUENCE_MODULE = "bioflow_sequence"

# How long the process group gets to die after SIGKILL before we stop waiting. A group
# that ignores SIGKILL is unkillable from here, and blocking forever on the reap would
# turn a timeout into a hang.
REAP_TIMEOUT_SECONDS = 5.0

_RAISED_BY = "bioflow.adapters.sidecar.runner"


@runtime_checkable
class SidecarRunner(Protocol):
    """Executes a job and returns its result. Never raises for a failed *op*."""

    def run(self, job: SidecarJob, workdir: Path) -> SidecarResult:
        """Run ``job`` in ``workdir`` and return what the runtime reported."""
        ...


def _fail(
    code: ErrorCode,
    message: str,
    *,
    remediation: str | None = None,
    details: dict[str, JSONValue] | None = None,
) -> SidecarResult:
    """Build an error result. The runner reports failures, it does not raise them."""
    return SidecarResult(
        status=SidecarStatus.ERROR,
        error=StructuredError(
            error_code=code,
            message=message,
            remediation=remediation,
            details=details or {},
            raised_by=_RAISED_BY,
        ),
    )


class _ProcessRunner:
    """Shared machinery for the runners that actually start a process."""

    def __init__(
        self,
        *,
        store: ArtifactStore | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store
        self.timeout_seconds = timeout_seconds

    def _execute(self, job: SidecarJob, workdir: Path, argv: Sequence[str]) -> SidecarResult:
        """Stage, spawn, wait, and read back -- the whole loop, minus building argv."""
        workdir = Path(workdir)
        _workdir.prepare(workdir)
        staged = _workdir.stage_inputs(job, workdir, self.store)
        _workdir.write_job(staged, workdir)
        stdout_path, stderr_path = _workdir.log_paths(workdir)

        started = time.monotonic()
        timed_out = False
        returncode: int | None = None

        # Streams go straight to files rather than pipes: a chatty sidecar filling a pipe
        # buffer while we wait on it is a deadlock, and one that logs for an hour should
        # not be buffered in the orchestrator's memory.
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            try:
                process = subprocess.Popen(
                    list(argv),
                    cwd=workdir,
                    stdout=out,
                    stderr=err,
                    shell=False,
                    # The whole point: gives the child its own process group so the
                    # timeout path can kill everything it spawned, not just the child.
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                return _fail(
                    ErrorCode.EXTERNAL_TOOL_FAILED,
                    f"could not start the sidecar: {type(exc).__name__}: {exc}",
                    remediation="Check that the interpreter and module exist.",
                    details={"argv": list(argv)},
                )

            try:
                returncode = process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(process)

        duration_ms = int((time.monotonic() - started) * 1000)
        observed = {
            "stdout_tail": _workdir.tail(stdout_path),
            "stderr_tail": _workdir.tail(stderr_path),
            "duration_ms": duration_ms,
        }

        if timed_out:
            return _fail(
                ErrorCode.TIMEOUT,
                f"sidecar op {job.op!r} exceeded {self.timeout_seconds}s and was killed",
                remediation="Raise the timeout, or reduce the work the op is asked to do.",
                details={"op": job.op, "timeout_seconds": self.timeout_seconds},
            ).model_copy(update=observed)

        if returncode != 0:
            return _fail(
                ErrorCode.EXTERNAL_TOOL_FAILED,
                f"sidecar op {job.op!r} exited {returncode}",
                remediation="See stderr_tail, or the full log in the run's logs directory.",
                details={"op": job.op, "returncode": returncode},
            ).model_copy(update=observed)

        return _read_result(workdir, job).model_copy(update=observed)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's entire process group, then reap it.

    Best-effort by necessity: the group can disappear between the ``getpgid`` and the
    ``killpg``, which raises ``ProcessLookupError`` and means the thing we wanted has
    already happened.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    # A group that ignores SIGKILL is unkillable from here; waiting forever on the reap
    # would turn a timeout into the hang it was meant to end.
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=REAP_TIMEOUT_SECONDS)


def _read_result(workdir: Path, job: SidecarJob) -> SidecarResult:
    """Parse ``out/result.json``, or explain why the run cannot be believed."""
    path = _workdir.result_path(workdir)
    if not path.is_file():
        return _fail(
            ErrorCode.EXTERNAL_TOOL_FAILED,
            f"sidecar op {job.op!r} exited 0 but wrote no {_workdir.RESULT_FILENAME}",
            remediation=f"The sidecar must write {_workdir.RESULT_FILENAME} into --outdir.",
            details={"op": job.op, "expected": _workdir.RESULT_FILENAME},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(
            ErrorCode.SCHEMA_MISMATCH,
            f"sidecar result is not valid JSON: {exc}",
            details={"op": job.op},
        )
    try:
        return SidecarResult.model_validate(payload)
    except Exception as exc:
        # A result that does not match the contract is worse than no result: acting on it
        # would mean acting on a message from something that is not speaking v1.
        return _fail(
            ErrorCode.SCHEMA_MISMATCH,
            f"sidecar result does not satisfy contract v1: {exc}",
            remediation="Check the sidecar against schemas/sidecar_result.v1.json.",
            details={"op": job.op},
        )


class LocalSidecarRunner(_ProcessRunner):
    """Runs the sidecar as a subprocess of this machine's interpreter.

    The development and single-machine mode. It gives no isolation beyond the process
    boundary -- that is what :class:`DockerSidecarRunner` is for -- but it is the same
    process boundary, so a capability behaves identically under both.
    """

    def __init__(
        self,
        module: str = DEFAULT_SEQUENCE_MODULE,
        *,
        interpreter: str | None = None,
        store: ArtifactStore | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(store=store, timeout_seconds=timeout_seconds)
        self.module = module
        # sys.executable rather than "python": the sidecar must run in a known
        # interpreter, not whichever one happens to be first on PATH.
        self.interpreter = interpreter or sys.executable

    def argv(self) -> list[str]:
        """The command line, exactly as the container's entrypoint composes it."""
        return [
            self.interpreter,
            "-m",
            self.module,
            "run",
            "--job",
            _workdir.JOB_FILENAME,
            "--outdir",
            _workdir.OUT_DIR,
        ]

    def run(self, job: SidecarJob, workdir: Path) -> SidecarResult:
        """Run the job locally."""
        return self._execute(job, workdir, self.argv())


def docker_argv(
    image: str,
    workdir: Path,
    *,
    uid: int,
    gid: int,
    docker: str = "docker",
) -> list[str]:
    """Build the ``docker run`` command line.

    A pure function so the flags can be tested without a daemon. Every one of them is
    load-bearing:

    ``--network=none``
        The sidecar computes; it does not fetch. Anything it needs was staged.
    ``--read-only`` plus one rw bind
        The image cannot be modified by what runs in it, and the only writable place is
        the workdir the caller already owns.
    ``--tmpfs /tmp``
        Because ``--read-only`` otherwise breaks every library that writes a scratch file.
    ``-u <uid>:<gid>``
        Outputs land owned by the caller instead of by root, which is what makes the
        workdir usable afterwards without a chown.
    """
    return [
        docker,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "-u",
        f"{uid}:{gid}",
        "-v",
        f"{Path(workdir).resolve()}:/work:rw",
        "-w",
        "/work",
        image,
        "run",
        "--job",
        _workdir.JOB_FILENAME,
        "--outdir",
        _workdir.OUT_DIR,
    ]


class DockerSidecarRunner(_ProcessRunner):
    """Runs the sidecar in a container pinned by digest.

    The image reference must carry a digest. A floating tag would make a recorded run
    unreproducible in the quietest possible way: the same command, months later, running
    different code and reporting no difference.
    """

    def __init__(
        self,
        image: str,
        *,
        store: ArtifactStore | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        docker: str = "docker",
        require_digest: bool = True,
    ) -> None:
        super().__init__(store=store, timeout_seconds=timeout_seconds)
        if require_digest and "@sha256:" not in image:
            raise BioflowError(
                StructuredError(
                    error_code=ErrorCode.VALIDATION_FAILED,
                    message=f"sidecar image {image!r} is not pinned by digest",
                    remediation=(
                        "Set BIOFLOW_SIDECAR_IMAGE to name@sha256:<digest>. A moving tag "
                        "makes a recorded run unreproducible without saying so."
                    ),
                    details={"image": image},
                    raised_by=_RAISED_BY,
                )
            )
        self.image = image
        self.docker = docker

    def run(self, job: SidecarJob, workdir: Path) -> SidecarResult:
        """Run the job in a container, recording the image digest in ``versions``."""
        argv = docker_argv(
            self.image,
            Path(workdir),
            uid=os.getuid(),
            gid=os.getgid(),
            docker=self.docker,
        )
        result = self._execute(job, workdir, argv)
        # Stamped by the runner because the sidecar cannot know it: from inside the
        # container there is no way to see which image reference started it.
        return result.model_copy(update={"versions": {**result.versions, "image": self.image}})


class MockSidecarRunner:
    """Serves recorded results, keyed by the canonical hash of the job.

    The mechanism every contract test uses. Recording a real result once and replaying it
    thereafter is what lets a capability that depends on torch be tested on a laptop with
    no torch, no GPU, and no network.
    """

    def __init__(self, fixtures: Path, *, strict: bool = True) -> None:
        self.fixtures = Path(fixtures)
        self.strict = strict

    def path_for(self, job: SidecarJob) -> Path:
        """The file a given job replays from."""
        return self.fixtures / f"{job_sha256(job)}.json"

    def write_fixture(self, job: SidecarJob, result: SidecarResult) -> Path:
        """Record a result for a job.

        Exists so nobody has to hand-compute a sha256 filename, which is the kind of
        chore that ends with fixtures nobody updates.
        """
        self.fixtures.mkdir(parents=True, exist_ok=True)
        path = self.path_for(job)
        payload = {
            # Stored for a human reading the diff; only the hash is used to look it up.
            "job": json.loads(job.model_dump_json()),
            "job_sha256": job_sha256(job),
            "result": json.loads(result.model_dump_json()),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def run(self, job: SidecarJob, workdir: Path) -> SidecarResult:
        """Serve the recorded result for this job."""
        path = self.path_for(job)
        if not path.is_file():
            return self._miss(job, path)
        recorded = json.loads(path.read_text(encoding="utf-8"))
        return SidecarResult.model_validate(recorded["result"])

    def _miss(self, job: SidecarJob, path: Path) -> SidecarResult:
        """Explain a miss well enough to fix it.

        A miss is the failure a contributor hits most often, so the message names the op,
        the computed hash, and what is actually on disk. ``strict`` decides whether it
        arrives as an exception or as an error result -- a test wants the former.
        """
        available = (
            sorted(p.stem for p in self.fixtures.glob("*.json")) if self.fixtures.is_dir() else []
        )
        error = StructuredError(
            error_code=ErrorCode.NOT_FOUND,
            message=(
                f"no recorded sidecar result for op {job.op!r}: expected {path.name}. "
                f"The job hashed to {job_sha256(job)}, and the fixture directory holds "
                f"{len(available)} recording(s)."
            ),
            remediation=(
                "Record it with MockSidecarRunner.write_fixture(), or check whether the "
                "op, params, seed, or input digests changed since it was recorded."
            ),
            details={
                "op": job.op,
                "directory": str(self.fixtures),
                "expected_sha256": job_sha256(job),
                "available": list(available[:20]),
            },
            raised_by=_RAISED_BY,
        )
        if self.strict:
            raise BioflowError(error)
        return SidecarResult(status=SidecarStatus.ERROR, error=error)
