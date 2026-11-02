"""Path safety.

Every file BioFlow writes on behalf of a name it did not choose itself -- a link name
from config, a filename from a tool call, an entry from a workflow's output manifest --
goes through :func:`safe_resolve`. The names will eventually come from an LLM, and a
containment check that lives in one place is the only kind that can be relied on.
"""

from __future__ import annotations

import os
from pathlib import Path

from bioflow.core.models import BioflowError, ErrorCode, StructuredError

_RAISED_BY = "bioflow.core.paths"


def safe_resolve(base: Path, candidate: str) -> Path:
    """Return ``base / candidate``, refusing anything that escapes ``base``.

    Containment is checked **lexically**, with ``os.path.normpath`` rather than
    ``Path.resolve()``. The difference matters: ``resolve()`` follows a symlink sitting
    at the candidate's own name, so re-linking an artifact that is already linked would
    resolve out to the object it points at and look like an escape. The question here is
    whether the *name* escapes, and that is a question about the string.

    Absolute candidates are rejected too -- ``Path("/base") / "/etc/passwd"`` is
    ``/etc/passwd``, which is exactly the trap this guards.
    """
    if not candidate or candidate in (".", "./"):
        raise _rejected(base, candidate, "the name is empty")
    if os.path.isabs(candidate) or Path(candidate).is_absolute():
        raise _rejected(base, candidate, "the name is an absolute path")

    base_abs = Path(os.path.abspath(base))
    resolved = Path(os.path.normpath(base_abs / candidate))
    if resolved == base_abs or not resolved.is_relative_to(base_abs):
        raise _rejected(base, candidate, "the name resolves outside the base directory")
    return resolved


def _rejected(base: Path, candidate: str, reason: str) -> BioflowError:
    return BioflowError(
        StructuredError(
            error_code=ErrorCode.VALIDATION_FAILED,
            message=f"unsafe path {candidate!r}: {reason}",
            remediation="Use a relative name below the base directory, without '..'.",
            details={"base": str(base), "candidate": candidate},
            raised_by=_RAISED_BY,
        )
    )
