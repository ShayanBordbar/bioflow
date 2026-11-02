"""Content-addressed artifact storage.

Objects live at ``<root>/objects/<aa>/<bb>/<sha256>``, sharded two levels on the digest
so no single directory accumulates millions of entries. The digest is the address: two
callers storing identical bytes converge on one file.

Storing the same content twice is idempotent on disk but still mints a **new**
``artifact_id``. The bytes are the same event; the acts of producing them are not.

Ingested objects keep their original mode bits. Hard links share an inode, so making an
object read-only would also make the caller's own input file read-only — a nasty
surprise for a FASTQ sitting in someone's home directory. Integrity is therefore
*detected* by :meth:`ArtifactStore.verify` rather than enforced by the filesystem.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from bioflow.core.ids import new_artifact_id
from bioflow.core.models import (
    ArtifactRef,
    ArtifactType,
    BioflowError,
    ErrorCode,
    JSONValue,
    StructuredError,
)
from bioflow.core.paths import safe_resolve

CHUNK_SIZE = 64 * 1024
URI_SCHEME = "cas://"

_RAISED_BY = "bioflow.core.store"

# Tar members are normalized to these so that the same directory content always
# produces the same tarball, and therefore the same digest.
_TAR_DIR_MODE = 0o755
_TAR_FILE_MODE = 0o644


def _fail(
    code: ErrorCode,
    message: str,
    *,
    remediation: str | None = None,
    details: dict[str, JSONValue] | None = None,
) -> BioflowError:
    """Build a :class:`BioflowError` so call sites stay one line long."""
    return BioflowError(
        StructuredError(
            error_code=code,
            message=message,
            remediation=remediation,
            details=details or {},
            raised_by=_RAISED_BY,
        )
    )


def _normalize_member(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip every source of nondeterminism from a tar member header."""
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = _TAR_DIR_MODE if info.isdir() else _TAR_FILE_MODE
    return info


class ArtifactStore:
    """A sha256-addressed object store rooted at ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects_dir = self.root / "objects"

    # -- addressing -----------------------------------------------------------------

    def object_path(self, sha256: str) -> Path:
        """Return the on-disk location for a digest, whether or not it exists."""
        return self.objects_dir / sha256[:2] / sha256[2:4] / sha256

    def resolve(self, ref: ArtifactRef) -> Path:
        """Return the path backing ``ref``, raising if it is absent or not ours."""
        if not ref.uri.startswith(URI_SCHEME):
            raise _fail(
                ErrorCode.VALIDATION_FAILED,
                f"artifact {ref.artifact_id} has URI {ref.uri!r}, which is not a {URI_SCHEME} URI",
                remediation="Only artifacts produced by ArtifactStore can be resolved.",
                details={"artifact_id": ref.artifact_id, "uri": ref.uri},
            )
        digest = ref.uri.removeprefix(URI_SCHEME)
        path = self.object_path(digest)
        if not path.exists():
            raise _fail(
                ErrorCode.NOT_FOUND,
                f"no object stored for artifact {ref.artifact_id} (sha256 {digest})",
                remediation="The store root may be wrong, or the object was deleted.",
                details={"artifact_id": ref.artifact_id, "sha256": digest},
            )
        return path

    def open(self, ref: ArtifactRef) -> BinaryIO:
        """Open the object backing ``ref`` for binary reading."""
        return self.resolve(ref).open("rb")

    def verify(self, ref: ArtifactRef) -> bool:
        """Re-hash the stored object and report whether it still matches ``ref``.

        This is the only thing standing between a silently corrupted object and a
        result built on it, since objects are not write-protected on disk.
        """
        if ref.sha256 is None:
            raise _fail(
                ErrorCode.VALIDATION_FAILED,
                f"artifact {ref.artifact_id} carries no sha256 to verify against",
                remediation="Only artifacts produced by ArtifactStore can be verified.",
                details={"artifact_id": ref.artifact_id},
            )
        return _hash_file(self.resolve(ref)) == ref.sha256

    # -- ingestion ------------------------------------------------------------------

    def put(
        self,
        path: Path,
        artifact_type: ArtifactType,
        metadata: dict[str, JSONValue] | None = None,
        produced_by: str | None = None,
        run_id: str | None = None,
    ) -> ArtifactRef:
        """Ingest a file, hardlinking it into the store when the filesystem allows."""
        source = Path(path)
        if not source.is_file():
            raise _fail(
                ErrorCode.MISSING_INPUT,
                f"cannot store {source}: not an existing file",
                remediation="Check the path, or use put_dir() for a directory.",
                details={"path": str(source)},
            )
        digest = _hash_file(source)
        self._ingest(source, digest)
        return self._make_ref(
            digest=digest,
            size_bytes=self.object_path(digest).stat().st_size,
            artifact_type=artifact_type,
            metadata=metadata,
            produced_by=produced_by,
            run_id=run_id,
        )

    def put_bytes(
        self,
        data: bytes,
        artifact_type: ArtifactType,
        metadata: dict[str, JSONValue] | None = None,
        produced_by: str | None = None,
        run_id: str | None = None,
    ) -> ArtifactRef:
        """Ingest an in-memory payload."""
        digest = hashlib.sha256(data).hexdigest()
        destination = self.object_path(digest)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _atomic_target(destination) as tmp:
                tmp.write_bytes(data)
        return self._make_ref(
            digest=digest,
            size_bytes=len(data),
            artifact_type=artifact_type,
            metadata=metadata,
            produced_by=produced_by,
            run_id=run_id,
        )

    def put_dir(
        self,
        path: Path,
        metadata: dict[str, JSONValue] | None = None,
        produced_by: str | None = None,
        run_id: str | None = None,
    ) -> ArtifactRef:
        """Tar a directory and ingest the tarball as a ``directory`` artifact.

        The tar is normalized (sorted members, zeroed mtime/uid/gid, clamped mode) so
        that identical directory *content* always yields an identical digest. Without
        that, re-tarring the same directory a second later would produce a different
        object and dedup would never fire. The real member names, sizes, and mtimes are
        preserved in ``metadata["members"]``.
        """
        source = Path(path)
        if not source.is_dir():
            raise _fail(
                ErrorCode.MISSING_INPUT,
                f"cannot store {source}: not an existing directory",
                remediation="Check the path, or use put() for a single file.",
                details={"path": str(source)},
            )

        members: list[JSONValue] = []
        with tempfile.TemporaryDirectory() as staging:
            tarball = Path(staging) / "dir.tar"
            with tarfile.open(tarball, "w", format=tarfile.PAX_FORMAT) as tar:
                entries = sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix())
                for entry in entries:
                    relative = entry.relative_to(source).as_posix()
                    stat = entry.stat()
                    members.append(
                        {
                            "name": relative,
                            "size": 0 if entry.is_dir() else stat.st_size,
                            "mtime": stat.st_mtime,
                            "is_dir": entry.is_dir(),
                        }
                    )
                    tar.add(entry, arcname=relative, recursive=False, filter=_normalize_member)
            digest = _hash_file(tarball)
            self._ingest(tarball, digest)

        combined: dict[str, JSONValue] = dict(metadata or {})
        combined["members"] = members
        return self._make_ref(
            digest=digest,
            size_bytes=self.object_path(digest).stat().st_size,
            artifact_type=ArtifactType.DIRECTORY,
            metadata=combined,
            produced_by=produced_by,
            run_id=run_id,
        )

    # -- run directories ------------------------------------------------------------

    def link_into_run(self, ref: ArtifactRef, run_dir: Path, relative_name: str) -> Path:
        """Symlink an object into a run directory under a human-readable name.

        Run directories are what a scientist actually browses, so they get readable
        names pointing at the opaque digest-named objects.
        """
        # relative_name will eventually come from config or a tool call, so containment
        # is checked by the one shared implementation rather than a local copy.
        link = safe_resolve(Path(run_dir), relative_name)
        target = self.resolve(ref)
        link.parent.mkdir(parents=True, exist_ok=True)
        # Symlink then rename, so replacing an existing link is atomic.
        staging = link.parent / f".{link.name}.{os.getpid()}.tmp"
        staging.unlink(missing_ok=True)
        staging.symlink_to(target)
        os.replace(staging, link)
        return link

    # -- internals ------------------------------------------------------------------

    def _ingest(self, source: Path, digest: str) -> None:
        """Place ``source`` into the store under ``digest``, if not already there."""
        destination = self.object_path(digest)
        if destination.exists():
            return  # Dedup: identical bytes are already stored.
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError as exc:
            # EXDEV is the expected case: the source is on another filesystem, where a
            # hard link is impossible and a copy is the only option.
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
                raise
            with _atomic_target(destination) as tmp:
                shutil.copyfile(source, tmp)

    def _make_ref(
        self,
        *,
        digest: str,
        size_bytes: int,
        artifact_type: ArtifactType,
        metadata: dict[str, JSONValue] | None,
        produced_by: str | None,
        run_id: str | None,
    ) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=new_artifact_id(),
            artifact_type=artifact_type,
            uri=f"{URI_SCHEME}{digest}",
            sha256=digest,
            size_bytes=size_bytes,
            produced_by=produced_by,
            run_id=run_id,
            created_at=datetime.now(UTC),
            metadata=dict(metadata or {}),
        )


def _hash_file(path: Path) -> str:
    """Stream a file through sha256 so multi-GB inputs never land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class _atomic_target:
    """Yield a temp path beside ``destination``, then rename it into place.

    Writing directly to the final digest-named path would let a crash leave a truncated
    object sitting under a name that claims to be its hash. The rename is atomic within
    a directory, so an object either exists in full or not at all.
    """

    def __init__(self, destination: Path) -> None:
        self._destination = destination
        self._tmp: Path | None = None

    def __enter__(self) -> Path:
        handle, name = tempfile.mkstemp(
            dir=self._destination.parent, prefix=f".{self._destination.name}.", suffix=".tmp"
        )
        os.close(handle)
        self._tmp = Path(name)
        return self._tmp

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._tmp is not None
        if exc_type is not None:
            self._tmp.unlink(missing_ok=True)
            return
        with self._tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self._tmp, self._destination)
