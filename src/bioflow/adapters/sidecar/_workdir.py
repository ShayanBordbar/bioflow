"""The directory a sidecar sees, and how inputs get into it.

Both real runners build the same layout, so a sidecar cannot tell whether it is running
locally or in a container:

    workdir/
      job.json                  the request
      inputs/<artifact_id>.<ext>  staged input bytes
      out/                      result.json and any produced artifacts
      logs/stdout.log, stderr.log

Staging is what makes the contract work at all. Inputs arrive as ``ArtifactRef`` with
``cas://<sha256>`` URIs, and a sidecar has no store to resolve those against — under
Docker it has no access to one by design. So the runner puts the bytes where the sidecar
can reach them and rewrites the URIs to workdir-relative names. The container then needs
exactly one writable bind mount and never sees the store.

Hardlinking rather than copying: a staged 40 GB BAM should cost no disk and no time. The
EXDEV fallback is the same one :mod:`bioflow.core.store` needs, for the same reason.
"""

from __future__ import annotations

import errno
import os
import shutil
from typing import TYPE_CHECKING

from bioflow.adapters.sidecar.contract import SidecarJob
from bioflow.core.models import ArtifactRef, ArtifactType, BioflowError, ErrorCode, StructuredError
from bioflow.core.paths import safe_resolve

if TYPE_CHECKING:
    from pathlib import Path

    from bioflow.core.store import ArtifactStore

JOB_FILENAME = "job.json"
RESULT_FILENAME = "result.json"
INPUTS_DIR = "inputs"
OUT_DIR = "out"
LOGS_DIR = "logs"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

SUBDIRS = (INPUTS_DIR, OUT_DIR, LOGS_DIR)

# How much of each stream is carried back in the result. Enough to see a traceback,
# little enough that a sidecar looping on a warning cannot fill the run store.
TAIL_BYTES = 16 * 1024

# Conventional suffixes, so a staged file looks like what it is to a tool that sniffs
# extensions. Cosmetic for BioFlow, load-bearing for some bioinformatics tools.
_SUFFIXES: dict[ArtifactType, str] = {
    ArtifactType.FASTA: ".fasta",
    ArtifactType.FASTQ: ".fastq",
    ArtifactType.TSV: ".tsv",
    ArtifactType.CSV: ".csv",
    ArtifactType.JSON: ".json",
    ArtifactType.YAML: ".yaml",
    ArtifactType.HTML: ".html",
    ArtifactType.PARQUET: ".parquet",
    ArtifactType.CHECKPOINT: ".ckpt",
    ArtifactType.LOG: ".log",
    ArtifactType.DIRECTORY: ".tar",
}

_RAISED_BY = "bioflow.adapters.sidecar"


def prepare(workdir: Path) -> Path:
    """Create the workdir skeleton, returning it."""
    for name in SUBDIRS:
        (workdir / name).mkdir(parents=True, exist_ok=True)
    return workdir


def staged_name(ref: ArtifactRef) -> str:
    """The workdir-relative name an input is staged under.

    Named by ``artifact_id`` rather than by anything the artifact carries: it is the one
    field guaranteed unique within a run, and it is already a closed format, so two
    inputs cannot collide and no caller-supplied string reaches the filesystem.
    """
    return f"{INPUTS_DIR}/{ref.artifact_id}{_SUFFIXES.get(ref.artifact_type, '')}"


def stage_inputs(job: SidecarJob, workdir: Path, store: ArtifactStore | None) -> SidecarJob:
    """Place every input under ``workdir/inputs`` and rewrite the job to point there.

    Returns a new job; the original is frozen and stays the canonical description of the
    work. Note that :func:`~bioflow.adapters.sidecar.contract.job_sha256` deliberately
    ignores ``uri``, so the staged job hashes identically to the unstaged one.
    """
    if not job.inputs:
        return job
    if store is None:
        raise BioflowError(
            StructuredError(
                error_code=ErrorCode.MISSING_INPUT,
                message=(
                    f"job {job.op!r} has {len(job.inputs)} input(s) but the runner has no store"
                ),
                remediation="Construct the runner with the run's ArtifactStore.",
                details={"op": job.op, "n_inputs": len(job.inputs)},
                raised_by=_RAISED_BY,
            )
        )

    staged: list[ArtifactRef] = []
    for ref in job.inputs:
        relative = staged_name(ref)
        # Through safe_resolve like every other name BioFlow did not choose itself.
        destination = safe_resolve(workdir, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(store.resolve(ref), destination)
        staged.append(ref.model_copy(update={"uri": relative}))

    return job.model_copy(update={"inputs": tuple(staged)})


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hardlink an object into place, copying only when the filesystem refuses."""
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError as exc:
        # EXDEV: the store is on another filesystem. EPERM/EMLINK: links refused or
        # exhausted. All three mean the same thing here -- copy instead.
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            raise
        shutil.copyfile(source, destination)


def write_job(job: SidecarJob, workdir: Path) -> Path:
    """Serialize the job into the workdir."""
    path = workdir / JOB_FILENAME
    path.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def result_path(workdir: Path) -> Path:
    """Where the sidecar is expected to leave its result."""
    return workdir / OUT_DIR / RESULT_FILENAME


def log_paths(workdir: Path) -> tuple[Path, Path]:
    """The stdout and stderr log files for a run."""
    logs = workdir / LOGS_DIR
    return logs / STDOUT_LOG, logs / STDERR_LOG


def tail(path: Path, limit: int = TAIL_BYTES) -> str:
    """Return the last ``limit`` bytes of a log file, on a line boundary.

    Seeks rather than reads: a sidecar that logged a gigabyte before dying is exactly the
    case where the tail matters, and exactly the case where reading it all would hurt.
    """
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
            data = handle.read(limit)
            # The window opened mid-line; drop the fragment.
            _, _, remainder = data.partition(b"\n")
            data = remainder
        else:
            data = handle.read(limit)
    return data.decode("utf-8", errors="replace")
