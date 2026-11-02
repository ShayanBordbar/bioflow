"""Unit tests for path containment."""

from pathlib import Path

import pytest

from bioflow.core.models import BioflowError, ErrorCode
from bioflow.core.paths import safe_resolve

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "candidate",
    [
        "out.txt",
        "outputs/reads.fastq.gz",
        "a/b/c/deep.txt",
        "sub/../sibling.txt",  # normalizes back inside the base
        "./nested/file.txt",
        "file with spaces.txt",
        "..hidden.txt",  # a leading dot-dot in a *name*, not a traversal
    ],
)
def test_accepts_names_that_stay_inside(tmp_path: Path, candidate: str) -> None:
    resolved = safe_resolve(tmp_path, candidate)
    assert resolved.is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    "candidate",
    [
        "../escape.txt",
        "../../etc/passwd",
        "../../../../../../etc/passwd",
        "a/../../b",
        "sub/../../escape.txt",
        "ok/../../../etc/shadow",
        "..",
        "../",
    ],
)
def test_rejects_traversal(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(BioflowError) as excinfo:
        safe_resolve(tmp_path, candidate)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
    assert "unsafe path" in excinfo.value.error.message


@pytest.mark.parametrize("candidate", ["/etc/passwd", "/tmp/anything", "/"])
def test_rejects_absolute_paths(tmp_path: Path, candidate: str) -> None:
    # Path("/base") / "/etc/passwd" is "/etc/passwd" -- this is the trap being guarded.
    with pytest.raises(BioflowError) as excinfo:
        safe_resolve(tmp_path, candidate)
    assert "absolute" in excinfo.value.error.message


@pytest.mark.parametrize("candidate", ["", ".", "./"])
def test_rejects_names_that_address_the_base_itself(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(BioflowError):
        safe_resolve(tmp_path, candidate)


def test_resolution_is_lexical_not_symlink_following(tmp_path: Path) -> None:
    # The Step 1.2 regression: an existing symlink at the candidate's own name points
    # outside the base, but the *name* is still contained, so this must be accepted.
    base = tmp_path / "run"
    base.mkdir()
    outside = tmp_path / "objects" / "blob"
    outside.parent.mkdir()
    outside.write_bytes(b"content")
    (base / "link.txt").symlink_to(outside)

    assert safe_resolve(base, "link.txt") == (base.resolve() / "link.txt")


def test_error_details_name_both_sides(tmp_path: Path) -> None:
    with pytest.raises(BioflowError) as excinfo:
        safe_resolve(tmp_path, "../nope")
    details = excinfo.value.error.details
    assert details["candidate"] == "../nope"
    assert details["base"] == str(tmp_path)
    assert excinfo.value.error.remediation is not None


def test_base_need_not_exist(tmp_path: Path) -> None:
    # Callers resolve a name before creating the directory it will live in.
    assert safe_resolve(tmp_path / "not-created-yet", "file.txt").name == "file.txt"
