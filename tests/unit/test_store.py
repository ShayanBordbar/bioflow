"""Unit tests for the content-addressed artifact store."""

import errno
import hashlib
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from bioflow.core.models import ArtifactType, BioflowError, ErrorCode
from bioflow.core.store import CHUNK_SIZE, ArtifactStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "workspace")


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _stored_objects(store: ArtifactStore) -> list[Path]:
    return sorted(p for p in store.objects_dir.rglob("*") if p.is_file())


# --------------------------------------------------------------------------------------
# Addressing and dedup
# --------------------------------------------------------------------------------------


def test_put_returns_a_content_addressed_ref(store: ArtifactStore, tmp_path: Path) -> None:
    source = _write(tmp_path / "reads.fastq", b"@r1\nACGT\n+\nIIII\n")
    ref = store.put(source, ArtifactType.FASTQ, produced_by="fastp@0.23.4")

    assert ref.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert ref.uri == f"cas://{ref.sha256}"
    assert ref.size_bytes == source.stat().st_size
    assert ref.artifact_type is ArtifactType.FASTQ
    assert ref.produced_by == "fastp@0.23.4"


def test_objects_are_sharded_two_levels_on_the_digest(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"payload"), ArtifactType.OTHER)
    assert ref.sha256 is not None
    expected = store.objects_dir / ref.sha256[:2] / ref.sha256[2:4] / ref.sha256
    assert expected.is_file()
    assert store.resolve(ref) == expected


def test_identical_content_dedups_but_mints_a_new_artifact_id(
    store: ArtifactStore, tmp_path: Path
) -> None:
    first = store.put(_write(tmp_path / "one.txt", b"same bytes"), ArtifactType.OTHER)
    second = store.put(_write(tmp_path / "two.txt", b"same bytes"), ArtifactType.OTHER)

    # Same bytes are one object; producing them twice is two events.
    assert first.sha256 == second.sha256
    assert first.uri == second.uri
    assert first.artifact_id != second.artifact_id
    assert len(_stored_objects(store)) == 1


def test_different_content_produces_different_objects(store: ArtifactStore, tmp_path: Path) -> None:
    store.put(_write(tmp_path / "a.txt", b"alpha"), ArtifactType.OTHER)
    store.put(_write(tmp_path / "b.txt", b"beta"), ArtifactType.OTHER)
    assert len(_stored_objects(store)) == 2


def test_put_bytes_matches_put_of_the_same_content(store: ArtifactStore, tmp_path: Path) -> None:
    from_disk = store.put(_write(tmp_path / "x.json", b'{"k": 1}'), ArtifactType.JSON)
    in_memory = store.put_bytes(b'{"k": 1}', ArtifactType.JSON)
    assert from_disk.sha256 == in_memory.sha256
    assert len(_stored_objects(store)) == 1


def test_hash_is_correct_across_many_chunks(store: ArtifactStore, tmp_path: Path) -> None:
    # Deliberately not a multiple of CHUNK_SIZE, so the final short read is exercised.
    data = bytes(range(256)) * ((CHUNK_SIZE * 3) // 256 + 7)
    ref = store.put(_write(tmp_path / "big.bin", data), ArtifactType.OTHER)
    assert ref.sha256 == hashlib.sha256(data).hexdigest()
    assert ref.size_bytes == len(data)


def test_put_records_run_id_and_metadata(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(
        _write(tmp_path / "a.tsv", b"a\tb\n"),
        ArtifactType.TSV,
        metadata={"columns": ["a", "b"]},
        run_id="run_0123456789ab",
    )
    assert ref.run_id == "run_0123456789ab"
    assert ref.metadata == {"columns": ["a", "b"]}


def test_put_falls_back_to_copying_across_filesystems(
    store: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hardlinking is impossible across a filesystem boundary, which is the normal case
    # when a user's data lives on a different mount from the workspace. Simulating
    # EXDEV is the only way to exercise that branch in a unit test.
    def refuse_link(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", refuse_link)
    source = _write(tmp_path / "elsewhere.txt", b"copied not linked")
    ref = store.put(source, ArtifactType.OTHER)

    assert store.verify(ref) is True
    assert store.resolve(ref).read_bytes() == b"copied not linked"
    # A copy is a distinct inode, unlike the hardlink fast path.
    assert store.resolve(ref).stat().st_ino != source.stat().st_ino


def test_put_propagates_an_unexpected_link_failure(
    store: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_link(src: object, dst: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "link", refuse_link)
    with pytest.raises(OSError, match="No space left"):
        store.put(_write(tmp_path / "a.txt", b"x"), ArtifactType.OTHER)


def test_a_failed_write_leaves_no_object_behind(
    store: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An object named by a digest must never exist unless it holds those exact bytes,
    # so a write that dies partway through has to clean up its temp file.
    def refuse_link(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    def fail_copy(src: object, dst: object) -> None:
        raise RuntimeError("disk went away")

    monkeypatch.setattr(os, "link", refuse_link)
    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    with pytest.raises(RuntimeError, match="disk went away"):
        store.put(_write(tmp_path / "a.txt", b"x"), ArtifactType.OTHER)

    assert _stored_objects(store) == []
    assert list(store.objects_dir.rglob(".*")) == []


def test_put_rejects_a_missing_source(store: ArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(BioflowError) as excinfo:
        store.put(tmp_path / "nope.txt", ArtifactType.OTHER)
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT


def test_put_rejects_a_directory(store: ArtifactStore, tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    with pytest.raises(BioflowError) as excinfo:
        store.put(tmp_path / "adir", ArtifactType.OTHER)
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT


# --------------------------------------------------------------------------------------
# Reading, verification, corruption
# --------------------------------------------------------------------------------------


def test_open_reads_the_stored_bytes(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"hello store"), ArtifactType.OTHER)
    with store.open(ref) as handle:
        assert handle.read() == b"hello store"


def test_verify_accepts_an_untouched_object(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"intact"), ArtifactType.OTHER)
    assert store.verify(ref) is True


def test_verify_detects_a_corrupted_object(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"intact"), ArtifactType.OTHER)
    # Objects keep their original mode bits, so corruption is possible by design and
    # verify() is the thing that has to catch it.
    store.resolve(ref).write_bytes(b"tampered")
    assert store.verify(ref) is False


def test_verify_rejects_a_ref_with_no_digest(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"x"), ArtifactType.OTHER)
    with pytest.raises(BioflowError) as excinfo:
        store.verify(ref.model_copy(update={"sha256": None}))
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED


def test_resolve_reports_a_missing_object(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"gone soon"), ArtifactType.OTHER)
    store.resolve(ref).unlink()
    with pytest.raises(BioflowError) as excinfo:
        store.resolve(ref)
    assert excinfo.value.error_code is ErrorCode.NOT_FOUND


def test_resolve_rejects_a_foreign_uri(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"x"), ArtifactType.OTHER)
    with pytest.raises(BioflowError) as excinfo:
        store.resolve(ref.model_copy(update={"uri": "file:///etc/passwd"}))
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED


# --------------------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------------------


def _make_tree(root: Path) -> Path:
    (root / "sub").mkdir(parents=True)
    (root / "top.txt").write_bytes(b"top level")
    (root / "sub" / "nested.txt").write_bytes(b"nested")
    return root


def test_put_dir_stores_a_tar_and_lists_members(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put_dir(_make_tree(tmp_path / "tree"), metadata={"origin": "test"})

    assert ref.artifact_type is ArtifactType.DIRECTORY
    assert ref.metadata["origin"] == "test"
    members = ref.metadata["members"]
    assert isinstance(members, list)
    names = {m["name"] for m in members if isinstance(m, dict)}
    assert names == {"sub", "sub/nested.txt", "top.txt"}

    with tarfile.open(store.resolve(ref)) as tar:
        assert sorted(tar.getnames()) == ["sub", "sub/nested.txt", "top.txt"]
        extracted = tar.extractfile("sub/nested.txt")
        assert extracted is not None
        assert extracted.read() == b"nested"


def test_put_dir_is_reproducible_and_dedups(store: ArtifactStore, tmp_path: Path) -> None:
    first = store.put_dir(_make_tree(tmp_path / "a"))
    second = store.put_dir(_make_tree(tmp_path / "b"))

    # Identical content in a different location, tarred at a different moment, must
    # still land on one object -- that is what normalizing the tar headers buys.
    assert first.sha256 == second.sha256
    assert first.artifact_id != second.artifact_id
    assert len(_stored_objects(store)) == 1


def test_put_dir_notices_changed_content(store: ArtifactStore, tmp_path: Path) -> None:
    first = store.put_dir(_make_tree(tmp_path / "a"))
    changed = _make_tree(tmp_path / "b")
    (changed / "top.txt").write_bytes(b"different")
    assert store.put_dir(changed).sha256 != first.sha256


def test_put_dir_rejects_a_file(store: ArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(BioflowError) as excinfo:
        store.put_dir(_write(tmp_path / "a.txt", b"x"))
    assert excinfo.value.error_code is ErrorCode.MISSING_INPUT


# --------------------------------------------------------------------------------------
# Linking into run directories
# --------------------------------------------------------------------------------------


def test_link_into_run_creates_a_readable_symlink(store: ArtifactStore, tmp_path: Path) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"linked"), ArtifactType.OTHER)
    run_dir = tmp_path / "runs" / "run_0123456789ab"
    link = store.link_into_run(ref, run_dir, "outputs/reads.txt")

    assert link.is_symlink()
    assert link.read_bytes() == b"linked"
    assert link.resolve() == store.resolve(ref)


def test_link_into_run_replaces_an_existing_link(store: ArtifactStore, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_0123456789ab"
    first = store.put(_write(tmp_path / "a.txt", b"first"), ArtifactType.OTHER)
    second = store.put(_write(tmp_path / "b.txt", b"second"), ArtifactType.OTHER)

    store.link_into_run(first, run_dir, "out.txt")
    link = store.link_into_run(second, run_dir, "out.txt")
    assert link.read_bytes() == b"second"


@pytest.mark.parametrize("name", ["../escape.txt", "sub/../../escape.txt"])
def test_link_into_run_refuses_to_escape_the_run_dir(
    store: ArtifactStore, tmp_path: Path, name: str
) -> None:
    ref = store.put(_write(tmp_path / "a.txt", b"x"), ArtifactType.OTHER)
    run_dir = tmp_path / "runs" / "run_0123456789ab"
    run_dir.mkdir(parents=True)
    with pytest.raises(BioflowError) as excinfo:
        store.link_into_run(ref, run_dir, name)
    assert excinfo.value.error_code is ErrorCode.VALIDATION_FAILED
