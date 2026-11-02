"""Unit tests for artifact summarization.

Two properties matter more than any individual format: nothing is read whole, and nothing
emits a path. Both are asserted structurally rather than by inspection, because both are
easy to break with a plausible-looking edit.
"""

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import pytest

from bioflow.core.ids import new_artifact_id
from bioflow.core.models import ArtifactRef, ArtifactType
from bioflow.orchestration.summarize import (
    REDACTION,
    SCAN_BYTES,
    TAIL_BYTES,
    Source,
    get_summarizer,
    headline,
    redact_paths,
    registered_types,
    summarize,
    summarizer,
)
from bioflow.orchestration.tokens import count_tokens

pytestmark = pytest.mark.unit

# Anything that looks like a filesystem location reaching the model. Matches what the
# redactor targets, and is applied to every summarizer's output below.
PATH_LIKE = re.compile(r"/(?:[\w.+@-]+/)+|[A-Za-z]:\\|cas://|file://")


class FileSource:
    """A :class:`Source` over a real file, counting what gets read."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.bytes_read = 0
        self.unbounded_reads = 0

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def open(self) -> BinaryIO:
        return _CountingHandle(self.path.open("rb"), self)  # type: ignore[return-value]


class _CountingHandle:
    """Wraps a file handle, recording every read and refusing to hide an unbounded one."""

    def __init__(self, handle: BinaryIO, owner: FileSource) -> None:
        self._handle = handle
        self._owner = owner

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            # The failure mode this whole module exists to prevent.
            self._owner.unbounded_reads += 1
            raise AssertionError("summarizer called read() with no size limit")
        data = self._handle.read(size)
        self._owner.bytes_read += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def __enter__(self) -> "_CountingHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()


def make_ref(artifact_type: ArtifactType, path: Path) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=new_artifact_id(),
        artifact_type=artifact_type,
        uri=f"cas://{'a' * 64}",
        size_bytes=path.stat().st_size,
        created_at=datetime.now(UTC),
    )


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_the_documented_types_are_registered() -> None:
    assert set(registered_types()) == {"tsv", "csv", "json", "fasta", "log", "html"}


def test_registering_a_type_twice_raises_at_import_time() -> None:
    # Two summarizers for one type means one of them is dead and nobody knows which.
    with pytest.raises(ValueError, match="already registered"):

        @summarizer("tsv")
        def _duplicate(ref: ArtifactRef, source: Source, budget: int) -> str:
            return ""  # pragma: no cover


def test_an_unregistered_type_falls_back_to_a_headline(tmp_path: Path) -> None:
    path = write(tmp_path / "thing.bin", "whatever")
    ref = make_ref(ArtifactType.PARQUET, path)
    assert get_summarizer(ArtifactType.PARQUET) is None
    assert summarize(ref, FileSource(path), 100) == headline(ref)


def test_a_malformed_file_does_not_fail_the_run(tmp_path: Path) -> None:
    # A summary is a convenience. A broken file must not take down the run that was
    # about to diagnose it.
    path = write(tmp_path / "broken.json", "{not json at all")
    result = summarize(make_ref(ArtifactType.JSON, path), FileSource(path), 200)
    assert "could not summarize" in result


# --------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "wrote /Users/someone/data/run.log",
        "reading /var/folders/xy/T/pytest-1/ws/objects/ab/cd/deadbeef",
        "cas://abc123",
        "file:///etc/passwd",
        r"C:\Users\someone\data",
    ],
)
def test_path_shaped_text_is_redacted(text: str) -> None:
    assert not PATH_LIKE.search(redact_paths(text))
    assert REDACTION in redact_paths(text)


def test_ordinary_prose_survives_redaction() -> None:
    # Over-eager redaction would blank out the column names it is meant to show.
    text = "columns: gene_id:str, log2FC:float, padj:float — 12 and/or 13 rows"
    assert redact_paths(text) == text


# --------------------------------------------------------------------------------------
# Tabular
# --------------------------------------------------------------------------------------


@pytest.fixture
def tsv(tmp_path: Path) -> Path:
    rows = ["gene_id\tcount\tlog2FC"]
    rows += [f"ENSG{index:08d}\t{index * 3}\t{index / 4:.2f}" for index in range(1, 21)]
    return write(tmp_path / "counts.tsv", "\n".join(rows) + "\n")


def test_a_tsv_reports_shape_columns_and_dtypes(tsv: Path) -> None:
    result = summarize(make_ref(ArtifactType.TSV, tsv), FileSource(tsv), 400)
    assert "20 rows x 3 columns" in result
    assert "gene_id:str" in result
    assert "count:int" in result
    assert "log2FC:float" in result


def test_a_tsv_shows_three_head_rows_and_says_how_many_it_hid(tsv: Path) -> None:
    result = summarize(make_ref(ArtifactType.TSV, tsv), FileSource(tsv), 400)
    assert "ENSG00000001" in result
    assert "[truncated: 3 of 20 rows shown]" in result


def test_a_tsv_reports_numeric_ranges(tsv: Path) -> None:
    result = summarize(make_ref(ArtifactType.TSV, tsv), FileSource(tsv), 400)
    assert "count: min=3 median=31.5 max=60" in result


def test_a_csv_uses_commas(tmp_path: Path) -> None:
    path = write(tmp_path / "t.csv", "a,b\n1,2\n3,4\n")
    result = summarize(make_ref(ArtifactType.CSV, path), FileSource(path), 300)
    assert "2 rows x 2 columns" in result
    assert "a:int" in result


def test_an_empty_table_says_so(tmp_path: Path) -> None:
    path = write(tmp_path / "empty.tsv", "")
    assert "empty" in summarize(make_ref(ArtifactType.TSV, path), FileSource(path), 100)


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------


def test_json_renders_a_key_skeleton_with_types(tmp_path: Path) -> None:
    payload = {"assay": "rna-seq", "samples": [{"id": "S1", "reads": 1000}], "padj": 0.03}
    path = write(tmp_path / "meta.json", json.dumps(payload))
    result = summarize(make_ref(ArtifactType.JSON, path), FileSource(path), 300)
    assert '"rna-seq"' in result  # whitelisted key, value shown
    assert "padj: number" in result  # not whitelisted, type only
    assert "array(1)" in result


def test_json_stops_descending_at_depth_two(tmp_path: Path) -> None:
    payload = {"a": {"b": {"c": {"d": 1}}}}
    path = write(tmp_path / "deep.json", json.dumps(payload))
    result = summarize(make_ref(ArtifactType.JSON, path), FileSource(path), 300)
    assert "object(1 keys)" in result


def test_an_oversized_json_is_reported_rather_than_guessed_at(tmp_path: Path) -> None:
    # A truncated JSON document does not parse, so there is no partial answer to give.
    path = tmp_path / "huge.json"
    with path.open("wb") as handle:
        handle.truncate(4 * 1024 * 1024)
    result = summarize(make_ref(ArtifactType.JSON, path), FileSource(path), 300)
    assert "too large to parse" in result


# --------------------------------------------------------------------------------------
# FASTA
# --------------------------------------------------------------------------------------


@pytest.fixture
def fasta(tmp_path: Path) -> Path:
    records = []
    for index in range(5):
        records.append(f">seq{index} description here")
        # Every record is longer than FASTA_RECORD_CHARS, so the examples are truncated.
        records.append("ACGT" * (20 + index * 10))
    return write(tmp_path / "seqs.fasta", "\n".join(records) + "\n")


def test_fasta_reports_records_lengths_and_gc(fasta: Path) -> None:
    result = summarize(make_ref(ArtifactType.FASTA, fasta), FileSource(fasta), 400)
    assert "5 records" in result
    assert "length min=80" in result
    assert "max=240" in result
    assert "GC mean=0.500" in result  # ACGT repeated is exactly half G+C


def test_fasta_shows_two_truncated_records(fasta: Path) -> None:
    result = summarize(make_ref(ArtifactType.FASTA, fasta), FileSource(fasta), 400)
    assert result.count(">seq") == 2
    assert "..." in result


# --------------------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------------------


@pytest.fixture
def log(tmp_path: Path) -> Path:
    lines = [f"INFO step {index} complete" for index in range(100)]
    lines.insert(50, "ERROR alignment failed for sample S3")
    lines.insert(60, "WARN low mapping rate")
    return write(tmp_path / "run.log", "\n".join(lines) + "\n")


def test_a_log_surfaces_failures_and_the_tail(log: Path) -> None:
    result = summarize(make_ref(ArtifactType.LOG, log), FileSource(log), 800)
    assert "ERROR alignment failed for sample S3" in result
    assert "WARN low mapping rate" in result
    assert "last 40 lines" in result


def test_a_log_line_containing_a_path_is_redacted(tmp_path: Path) -> None:
    # The real leak vector: summarizers are the one channel where *content* reaches the
    # prompt, and tool logs are full of working directories.
    path = write(tmp_path / "paths.log", "ERROR could not open /Users/someone/data/run.bam\n")
    result = summarize(make_ref(ArtifactType.LOG, path), FileSource(path), 300)
    assert "/Users/someone" not in result
    assert REDACTION in result


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------


def test_html_is_reported_by_size_only(tmp_path: Path) -> None:
    # Spending budget on markup buys nothing the model can act on.
    path = write(tmp_path / "report.html", "<html><body>" + "x" * 500 + "</body></html>")
    result = summarize(make_ref(ArtifactType.HTML, path), FileSource(path), 300)
    assert result.startswith("binary report,")
    assert "<body>" not in result


# --------------------------------------------------------------------------------------
# The two properties that matter
# --------------------------------------------------------------------------------------


@pytest.fixture
def every_kind(
    tmp_path: Path, tsv: Path, fasta: Path, log: Path
) -> list[tuple[ArtifactType, Path]]:
    """One real file per registered type, each seeded with path-shaped content."""
    return [
        (ArtifactType.TSV, tsv),
        (ArtifactType.CSV, write(tmp_path / "x.csv", "name,file\na,/opt/data/x.bam\n")),
        (
            ArtifactType.JSON,
            write(tmp_path / "x.json", json.dumps({"name": "/opt/data/x.bam", "n": 1})),
        ),
        (ArtifactType.FASTA, fasta),
        (ArtifactType.LOG, log),
        (ArtifactType.HTML, write(tmp_path / "x.html", "<html>/opt/data/x.bam</html>")),
    ]


def test_no_summarizer_ever_emits_a_path(
    every_kind: list[tuple[ArtifactType, Path]], tmp_path: Path
) -> None:
    """Invariant #2, asserted over every registered summarizer's real output."""
    for artifact_type, path in every_kind:
        for budget in (16, 64, 400, 4000):
            result = summarize(make_ref(artifact_type, path), FileSource(path), budget)
            assert not PATH_LIKE.search(result), f"{artifact_type} at budget {budget}: {result}"
            assert str(tmp_path) not in result
            assert "cas://" not in result


def test_every_summarizer_respects_its_budget(
    every_kind: list[tuple[ArtifactType, Path]],
) -> None:
    for artifact_type, path in every_kind:
        for budget in (16, 64, 400, 4000):
            result = summarize(make_ref(artifact_type, path), FileSource(path), budget)
            assert count_tokens(result) <= budget, f"{artifact_type} at budget {budget}"


def test_every_summarizers_first_line_identifies_the_artifact(
    every_kind: list[tuple[ArtifactType, Path]],
) -> None:
    # The index takes the first line for its row, so the first line has to stand alone.
    # Ordering the output most-informative-first is also what makes truncation safe:
    # what `fit` cuts is always the least useful part.
    for artifact_type, path in every_kind:
        for budget in (20, 4000):
            first = summarize(make_ref(artifact_type, path), FileSource(path), budget).splitlines()
            assert first, f"{artifact_type} produced nothing at budget {budget}"
            assert artifact_type.value in first[0].lower() or "report" in first[0].lower(), (
                f"{artifact_type} first line does not identify it: {first[0]}"
            )


def test_no_summarizer_reads_a_file_whole(
    every_kind: list[tuple[ArtifactType, Path]],
) -> None:
    for artifact_type, path in every_kind:
        source = FileSource(path)
        summarize(make_ref(artifact_type, path), source, 4000)
        assert source.unbounded_reads == 0, f"{artifact_type} read without a size limit"


# --------------------------------------------------------------------------------------
# The large-file property
# --------------------------------------------------------------------------------------

HALF_GIGABYTE = 500 * 1024 * 1024


@pytest.fixture
def huge_log(tmp_path: Path) -> Iterator[Path]:
    """A sparse 500 MB log with real content at the end.

    Sparse, so it costs no disk and no time to create; the summarizer still has to seek
    past 500 MB to reach anything, which is the behaviour under test.
    """
    path = tmp_path / "huge.log"
    with path.open("wb") as handle:
        handle.truncate(HALF_GIGABYTE)
        handle.seek(HALF_GIGABYTE - 200)
        handle.write(b"INFO nearly done\nERROR out of memory in stage 4\nINFO shutting down\n")
    yield path
    path.unlink()


def test_a_500mb_log_is_summarized_from_its_tail(huge_log: Path) -> None:
    source = FileSource(huge_log)
    result = summarize(make_ref(ArtifactType.LOG, huge_log), source, 600)

    assert "ERROR out of memory in stage 4" in result
    assert source.size_bytes == HALF_GIGABYTE
    # The actual assertion: a fraction of a percent of the file was touched.
    assert source.bytes_read <= TAIL_BYTES
    assert source.unbounded_reads == 0


def test_a_500mb_table_is_summarized_from_a_bounded_prefix(tmp_path: Path) -> None:
    path = tmp_path / "huge.tsv"
    header = b"gene_id\tcount\n"
    row = b"ENSG00000000001\t42\n"
    with path.open("wb") as handle:
        handle.write(header)
        # Enough real rows to overflow the scan window several times over.
        handle.write(row * (SCAN_BYTES * 3 // len(row)))

    source = FileSource(path)
    result = summarize(make_ref(ArtifactType.TSV, path), source, 600)

    assert "gene_id" in result
    assert source.bytes_read <= SCAN_BYTES + 1
    assert source.unbounded_reads == 0
    # Statistics over a sample must say they are over a sample.
    assert "scanned" in result
    path.unlink()
