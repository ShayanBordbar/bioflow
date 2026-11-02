"""Contract tests for the sidecar boundary.

Three of these test features. Two test that the boundary is real: that a timeout kills
everything the sidecar spawned, and that the sidecar package cannot see the orchestrator.
Both are properties that hold today by construction and would be easy to lose to a
plausible-looking edit, which is exactly what a contract test is for.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bioflow.adapters.sidecar import _workdir
from bioflow.adapters.sidecar.contract import (
    SidecarJob,
    SidecarResult,
    SidecarStatus,
    job_sha256,
)
from bioflow.adapters.sidecar.runner import (
    DockerSidecarRunner,
    LocalSidecarRunner,
    MockSidecarRunner,
    SidecarRunner,
    docker_argv,
)
from bioflow.core.ids import new_artifact_id
from bioflow.core.models import ArtifactRef, ArtifactType, BioflowError, ErrorCode
from bioflow.core.store import ArtifactStore

pytestmark = pytest.mark.contract

SEQUENCE_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "bioflow_sequence"

# The orchestrator's own dependencies. The sidecar importing any of them would mean its
# environment can no longer be resolved independently, which is the whole point.
FORBIDDEN_IMPORTS = frozenset(
    {
        "bioflow",
        "pydantic",
        "pydantic_settings",
        "typer",
        "httpx",
        "structlog",
        "jinja2",
        "tenacity",
        "platformdirs",
        "rich",
    }
)


def make_ref(artifact_type: ArtifactType = ArtifactType.FASTA, sha: str = "a" * 64) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=new_artifact_id(),
        artifact_type=artifact_type,
        uri=f"cas://{sha}",
        sha256=sha,
        size_bytes=1234,
        created_at=datetime.now(UTC),
    )


def ok_result(**overrides: object) -> SidecarResult:
    payload: dict[str, object] = {
        "status": SidecarStatus.OK,
        "metrics": {"score": 0.5},
        "versions": {"torch": "2.4.0"},
    }
    payload.update(overrides)
    return SidecarResult(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Job canonicalization
# --------------------------------------------------------------------------------------


def test_the_hash_is_stable_across_dict_ordering() -> None:
    """The spec's requirement: two orderings of the same params are the same job."""
    left = SidecarJob(op="predict", params={"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2]})
    right = SidecarJob(op="predict", params={"c": [1, 2], "b": {"y": 2, "x": 1}, "a": 1})
    assert job_sha256(left) == job_sha256(right)


def test_the_hash_ignores_per_run_artifact_identity() -> None:
    # Without this every recorded fixture would miss on the next run, because artifact
    # ids and timestamps are minted fresh each time, and the contract tier would silently
    # be testing nothing.
    sha = "b" * 64
    first = SidecarJob(op="predict", inputs=(make_ref(sha=sha),))
    second = SidecarJob(op="predict", inputs=(make_ref(sha=sha),))
    assert first.inputs[0].artifact_id != second.inputs[0].artifact_id
    assert job_sha256(first) == job_sha256(second)


def test_the_hash_survives_staging() -> None:
    """Staging rewrites input URIs, and that must not change what the job *is*."""
    job = SidecarJob(op="predict", inputs=(make_ref(),))
    staged = job.model_copy(
        update={"inputs": (job.inputs[0].model_copy(update={"uri": "inputs/art_x.fasta"}),)}
    )
    assert job_sha256(job) == job_sha256(staged)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("op", "other"),
        ("params", {"a": 2}),
        ("seed", 99),
    ],
)
def test_the_hash_distinguishes_different_work(field: str, value: object) -> None:
    base = SidecarJob(op="predict", params={"a": 1}, seed=1)
    assert job_sha256(base) != job_sha256(base.model_copy(update={field: value}))


def test_the_hash_distinguishes_different_input_content() -> None:
    one = SidecarJob(op="predict", inputs=(make_ref(sha="c" * 64),))
    two = SidecarJob(op="predict", inputs=(make_ref(sha="d" * 64),))
    assert job_sha256(one) != job_sha256(two)


# --------------------------------------------------------------------------------------
# MockSidecarRunner
# --------------------------------------------------------------------------------------


def test_mock_round_trip(tmp_path: Path) -> None:
    """The spec's requirement, and the mechanism every later contract test will use."""
    runner = MockSidecarRunner(tmp_path / "fixtures")
    job = SidecarJob(op="predict", params={"model": "enformer"}, seed=7)
    recorded = ok_result(metrics={"auprc": 0.81})

    path = runner.write_fixture(job, recorded)
    assert path.name == f"{job_sha256(job)}.json"

    served = runner.run(job, tmp_path / "work")
    assert served.status is SidecarStatus.OK
    assert served.metrics == {"auprc": 0.81}
    assert served.versions == {"torch": "2.4.0"}


def test_a_mock_fixture_replays_for_a_differently_identified_job(tmp_path: Path) -> None:
    # The property that makes fixtures durable: record against one run's artifacts,
    # replay against another's.
    runner = MockSidecarRunner(tmp_path / "fixtures")
    sha = "e" * 64
    runner.write_fixture(SidecarJob(op="predict", inputs=(make_ref(sha=sha),)), ok_result())
    served = runner.run(SidecarJob(op="predict", inputs=(make_ref(sha=sha),)), tmp_path)
    assert served.status is SidecarStatus.OK


def test_a_mock_miss_says_what_to_do(tmp_path: Path) -> None:
    # A miss is the failure a contributor hits most often, so the message has to name the
    # op, the hash, and what is on disk.
    runner = MockSidecarRunner(tmp_path / "fixtures")
    job = SidecarJob(op="predict")
    with pytest.raises(BioflowError) as caught:
        runner.run(job, tmp_path)

    error = caught.value.error
    assert error.error_code is ErrorCode.NOT_FOUND
    assert job_sha256(job) in error.message
    assert "predict" in error.message
    assert error.details["expected_sha256"] == job_sha256(job)


def test_a_lenient_mock_returns_the_miss_as_a_result(tmp_path: Path) -> None:
    runner = MockSidecarRunner(tmp_path / "fixtures", strict=False)
    result = runner.run(SidecarJob(op="predict"), tmp_path)
    assert result.status is SidecarStatus.ERROR
    assert result.error is not None
    assert result.error.error_code is ErrorCode.NOT_FOUND


def test_the_fixture_stores_the_job_for_a_human_reading_the_diff(tmp_path: Path) -> None:
    runner = MockSidecarRunner(tmp_path / "fixtures")
    job = SidecarJob(op="predict", params={"model": "enformer"})
    payload = json.loads(runner.write_fixture(job, ok_result()).read_text())
    assert payload["job"]["op"] == "predict"
    assert payload["job_sha256"] == job_sha256(job)


@pytest.mark.parametrize(
    "runner",
    [
        LocalSidecarRunner(),
        DockerSidecarRunner("img@sha256:" + "0" * 64),
        MockSidecarRunner(Path("/nonexistent")),
    ],
)
def test_every_runner_satisfies_the_protocol(runner: object) -> None:
    assert isinstance(runner, SidecarRunner)


# --------------------------------------------------------------------------------------
# The real sidecar, end to end
# --------------------------------------------------------------------------------------


def test_the_sequence_sidecar_answers_ping(tmp_path: Path) -> None:
    """The whole loop: job.json, a real process, result.json. No Docker required."""
    result = LocalSidecarRunner(timeout_seconds=60).run(
        SidecarJob(op="ping", seed=42), tmp_path / "work"
    )

    assert result.status is SidecarStatus.OK, result.error
    assert result.error is None
    assert result.metrics["seed"] == 42
    assert result.versions["python"].startswith("3.12")
    assert "bioflow_sequence" in result.versions
    # Timed by the runner, not self-reported.
    assert result.duration_ms >= 0


def test_the_workdir_layout_is_what_the_contract_says(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    LocalSidecarRunner(timeout_seconds=60).run(SidecarJob(op="ping"), workdir)

    assert (workdir / _workdir.JOB_FILENAME).is_file()
    assert _workdir.result_path(workdir).is_file()
    for name in _workdir.SUBDIRS:
        assert (workdir / name).is_dir()
    stdout_log, stderr_log = _workdir.log_paths(workdir)
    assert stdout_log.is_file() and stderr_log.is_file()


def test_an_unknown_op_is_a_result_not_a_crash(tmp_path: Path) -> None:
    # A failure the sidecar diagnosed is a result. Exiting non-zero would make it
    # indistinguishable from a crash.
    result = LocalSidecarRunner(timeout_seconds=60).run(
        SidecarJob(op="no_such_op"), tmp_path / "work"
    )
    assert result.status is SidecarStatus.ERROR
    assert result.error is not None
    assert result.error.error_code is ErrorCode.NOT_FOUND
    assert "ping" in (result.error.remediation or "")


def test_a_version_mismatch_is_refused(tmp_path: Path) -> None:
    # A v2 job may mean something different by a field this build thinks it understands.
    workdir = _workdir.prepare(tmp_path / "work")
    (workdir / "job.json").write_text(json.dumps({"contract_version": "2", "op": "ping"}))
    runner = LocalSidecarRunner(timeout_seconds=60)

    import subprocess

    subprocess.run(runner.argv(), cwd=workdir, check=True, capture_output=True)
    payload = json.loads(_workdir.result_path(workdir).read_text())
    assert payload["status"] == "error"
    assert payload["error"]["error_code"] == "SCHEMA_MISMATCH"


# --------------------------------------------------------------------------------------
# Input staging
# --------------------------------------------------------------------------------------


def test_inputs_are_staged_into_the_workdir(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "ws")
    ref = store.put_bytes(b">seq1\nACGT\n", ArtifactType.FASTA)
    workdir = _workdir.prepare(tmp_path / "work")

    staged = _workdir.stage_inputs(SidecarJob(op="ping", inputs=(ref,)), workdir, store)

    relative = staged.inputs[0].uri
    assert relative == f"inputs/{ref.artifact_id}.fasta"
    # The sidecar can open it by that relative name from its own working directory.
    assert (workdir / relative).read_bytes() == b">seq1\nACGT\n"
    # And the URI it sees is not a store path.
    assert "cas://" not in relative
    assert str(tmp_path) not in relative


def test_staging_the_same_input_twice_is_idempotent(tmp_path: Path) -> None:
    # A retried job reuses its workdir; hardlinking onto an existing name would fail with
    # EEXIST, so the second attempt must replace rather than refuse.
    store = ArtifactStore(tmp_path / "ws")
    ref = store.put_bytes(b">seq1\nACGT\n", ArtifactType.FASTA)
    workdir = _workdir.prepare(tmp_path / "work")
    job = SidecarJob(op="ping", inputs=(ref,))

    first = _workdir.stage_inputs(job, workdir, store)
    second = _workdir.stage_inputs(job, workdir, store)

    assert first.inputs[0].uri == second.inputs[0].uri
    assert (workdir / second.inputs[0].uri).read_bytes() == b">seq1\nACGT\n"


def test_staging_without_a_store_is_refused(tmp_path: Path) -> None:
    job = SidecarJob(op="ping", inputs=(make_ref(),))
    with pytest.raises(BioflowError) as caught:
        _workdir.stage_inputs(job, tmp_path, None)
    assert caught.value.error.error_code is ErrorCode.MISSING_INPUT


def test_a_job_with_no_inputs_needs_no_store(tmp_path: Path) -> None:
    job = SidecarJob(op="ping")
    assert _workdir.stage_inputs(job, tmp_path, None) is job


# --------------------------------------------------------------------------------------
# The timeout, and what it kills
# --------------------------------------------------------------------------------------

# A sidecar that forks a worker and then hangs. The grandchild is the point: killing only
# the direct child would leave it running, which on a real machine means a GPU held
# forever by a process nobody can find.
HANGING_SIDECAR = textwrap.dedent(
    """
    import os, subprocess, sys, time
    child = subprocess.Popen(["sleep", "300"])
    (sys.argv[1:] and open(sys.argv[1], "w").write(str(child.pid)))
    time.sleep(300)
    """
)


def _alive(pid: int) -> bool:
    """Whether a pid still exists, without signalling it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but not ours
        return True
    return True


def test_a_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    """The spec's requirement, asserted on the grandchild rather than the child."""
    module_dir = tmp_path / "fake"
    module_dir.mkdir()
    (module_dir / "hanger.py").write_text(HANGING_SIDECAR)
    pid_file = tmp_path / "grandchild.pid"

    workdir = _workdir.prepare(tmp_path / "work")
    _workdir.write_job(SidecarJob(op="ping"), workdir)

    runner = LocalSidecarRunner(timeout_seconds=1.5)
    argv = [sys.executable, str(module_dir / "hanger.py"), str(pid_file)]

    started = time.monotonic()
    result = runner._execute(SidecarJob(op="ping"), workdir, argv)
    elapsed = time.monotonic() - started

    assert result.status is SidecarStatus.ERROR
    assert result.error is not None
    assert result.error.error_code is ErrorCode.TIMEOUT
    assert elapsed < 20, "the timeout did not fire promptly"

    grandchild = int(pid_file.read_text())
    # Signals are delivered asynchronously; give the group a moment to actually die.
    for _ in range(50):
        if not _alive(grandchild):
            break
        time.sleep(0.1)
    assert not _alive(grandchild), (
        f"grandchild {grandchild} survived the timeout: the process group was not killed"
    )


def test_the_child_runs_in_its_own_process_group(tmp_path: Path) -> None:
    # start_new_session=True is what creates the group there is to kill. Asserted
    # directly, because losing it would make the test above pass by luck on some systems.
    workdir = _workdir.prepare(tmp_path / "work")
    script = workdir / "pgid.py"
    script.write_text("import os; print(os.getpgid(0) == os.getpid())")

    runner = LocalSidecarRunner(timeout_seconds=30)
    runner._execute(SidecarJob(op="ping"), workdir, [sys.executable, str(script)])

    stdout_log, _ = _workdir.log_paths(workdir)
    assert stdout_log.read_text().strip() == "True"


def test_a_nonzero_exit_is_reported_with_its_stderr(tmp_path: Path) -> None:
    workdir = _workdir.prepare(tmp_path / "work")
    script = workdir / "boom.py"
    script.write_text("import sys; print('the reason', file=sys.stderr); sys.exit(3)")

    result = LocalSidecarRunner(timeout_seconds=30)._execute(
        SidecarJob(op="ping"), workdir, [sys.executable, str(script)]
    )
    assert result.error is not None
    assert result.error.error_code is ErrorCode.EXTERNAL_TOOL_FAILED
    assert result.error.details["returncode"] == 3
    assert "the reason" in result.stderr_tail


def test_a_missing_result_file_is_reported(tmp_path: Path) -> None:
    workdir = _workdir.prepare(tmp_path / "work")
    script = workdir / "quiet.py"
    script.write_text("pass")

    result = LocalSidecarRunner(timeout_seconds=30)._execute(
        SidecarJob(op="ping"), workdir, [sys.executable, str(script)]
    )
    assert result.error is not None
    assert result.error.error_code is ErrorCode.EXTERNAL_TOOL_FAILED
    assert "result.json" in result.error.message


def test_a_result_that_breaks_the_contract_is_refused(tmp_path: Path) -> None:
    # Acting on it would mean acting on a message from something not speaking v1.
    workdir = _workdir.prepare(tmp_path / "work")
    script = workdir / "wrong.py"
    script.write_text(
        "import json, pathlib; "
        "pathlib.Path('out/result.json').write_text(json.dumps({'status': 'maybe'}))"
    )

    result = LocalSidecarRunner(timeout_seconds=30)._execute(
        SidecarJob(op="ping"), workdir, [sys.executable, str(script)]
    )
    assert result.error is not None
    assert result.error.error_code is ErrorCode.SCHEMA_MISMATCH


def test_a_result_that_is_not_json_is_reported(tmp_path: Path) -> None:
    workdir = _workdir.prepare(tmp_path / "work")
    script = workdir / "garbage.py"
    script.write_text("import pathlib; pathlib.Path('out/result.json').write_text('{oops')")

    result = LocalSidecarRunner(timeout_seconds=30)._execute(
        SidecarJob(op="ping"), workdir, [sys.executable, str(script)]
    )
    assert result.error is not None
    assert result.error.error_code is ErrorCode.SCHEMA_MISMATCH
    assert "not valid JSON" in result.error.message


def test_an_unstartable_command_is_reported_not_raised(tmp_path: Path) -> None:
    workdir = _workdir.prepare(tmp_path / "work")
    result = LocalSidecarRunner(timeout_seconds=5)._execute(
        SidecarJob(op="ping"), workdir, ["/nonexistent/interpreter"]
    )
    assert result.error is not None
    assert result.error.error_code is ErrorCode.EXTERNAL_TOOL_FAILED


# --------------------------------------------------------------------------------------
# Docker argv
# --------------------------------------------------------------------------------------


def test_the_docker_command_carries_every_containment_flag(tmp_path: Path) -> None:
    argv = docker_argv("img@sha256:" + "0" * 64, tmp_path, uid=501, gid=20)
    joined = " ".join(argv)

    assert "--network=none" in argv  # it computes; it does not fetch
    assert "--read-only" in argv  # the image cannot be modified by what runs in it
    assert "--rm" in argv
    assert "-u 501:20" in joined  # outputs land owned by the caller, not root
    assert f"{tmp_path.resolve()}:/work:rw" in joined  # the one writable place
    assert "--tmpfs /tmp" in joined  # or --read-only breaks every scratch file
    assert argv[-4:] == ["--job", "job.json", "--outdir", "out"]


def test_the_docker_command_never_reaches_a_shell(tmp_path: Path) -> None:
    # Invariant #1. argv is a list, and every element is a separate argument.
    argv = docker_argv("img@sha256:" + "0" * 64, tmp_path, uid=1, gid=1)
    assert all(isinstance(part, str) for part in argv)
    assert not any(";" in part or "&&" in part or "|" in part for part in argv)


def test_the_store_is_never_mounted(tmp_path: Path) -> None:
    # The container sees its own inputs and nothing else. Mounting the store would give
    # every job visibility of every artifact in the workspace.
    argv = docker_argv("img@sha256:" + "0" * 64, tmp_path, uid=1, gid=1)
    mounts = [argv[i + 1] for i, part in enumerate(argv) if part == "-v"]
    assert len(mounts) == 1
    assert mounts[0].endswith(":/work:rw")


def test_an_unpinned_image_is_refused() -> None:
    # A moving tag makes a recorded run unreproducible in the quietest possible way.
    with pytest.raises(BioflowError) as caught:
        DockerSidecarRunner("bioflow/sequence-sidecar:latest")
    assert caught.value.error.error_code is ErrorCode.VALIDATION_FAILED
    assert "digest" in caught.value.error.message


def test_a_digest_pinned_image_is_accepted() -> None:
    runner = DockerSidecarRunner("bioflow/sequence-sidecar@sha256:" + "0" * 64)
    assert runner.image.endswith("0" * 64)


# --------------------------------------------------------------------------------------
# Isolation: the invariant this whole step exists to make structural
# --------------------------------------------------------------------------------------


def _imported_roots(path: Path) -> set[str]:
    """Every top-level module name imported by a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_sidecar_package_exists_where_it_claims_to() -> None:
    # Guards the tests below from passing vacuously if the package is ever moved.
    assert SEQUENCE_PACKAGE.is_dir()
    assert list(SEQUENCE_PACKAGE.glob("*.py"))


@pytest.mark.parametrize("source", sorted(SEQUENCE_PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_the_sidecar_imports_nothing_from_the_orchestrator(source: Path) -> None:
    """Invariant #9, enforced rather than documented.

    The container does not ship ``bioflow`` at all, so a violation would fail at run time
    in production. This catches it at test time instead, which is several hours earlier.
    """
    offenders = _imported_roots(source) & FORBIDDEN_IMPORTS
    assert not offenders, (
        f"{source.name} imports {sorted(offenders)}; the sidecar's environment must "
        f"resolve independently of the orchestrator's"
    )


def test_the_sidecar_is_importable_without_the_orchestrator(tmp_path: Path) -> None:
    """Proof rather than inference: import it in a subprocess where bioflow is banned."""
    guard = tmp_path / "guard.py"
    guard.write_text(
        textwrap.dedent(
            """
            import sys

            class Blocker:
                def find_module(self, name, path=None):
                    if name == "bioflow" or name.startswith("bioflow."):
                        raise ImportError(f"blocked: {name}")
                    return None

            sys.meta_path.insert(0, Blocker())
            import bioflow_sequence.__main__ as m
            import bioflow_sequence.ops as ops
            assert "ping" in ops.available()
            print("ok")
            """
        )
    )
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(guard)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_ping_does_not_import_the_runtimes_it_reports_on(tmp_path: Path) -> None:
    """Reporting a version must not cost the seconds and memory of loading torch."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import sys
            from bioflow_sequence.ops import ping
            ping({}, [], None)
            leaked = [n for n in ("torch", "grelu", "polygraph") if n in sys.modules]
            print(",".join(leaked) or "none")
            """
        )
    )
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "none"


# --------------------------------------------------------------------------------------
# Log tails
# --------------------------------------------------------------------------------------


def test_a_huge_log_is_tailed_not_read(tmp_path: Path) -> None:
    # The case where the tail matters most is the case where reading it all would hurt.
    path = tmp_path / "big.log"
    with path.open("wb") as handle:
        handle.truncate(50 * 1024 * 1024)
        handle.seek(50 * 1024 * 1024 - 40)
        handle.write(b"\nthe last thing it said before dying\n")

    tail = _workdir.tail(path)
    assert "the last thing it said before dying" in tail
    assert len(tail.encode()) <= _workdir.TAIL_BYTES


def test_tailing_a_missing_log_is_empty(tmp_path: Path) -> None:
    assert _workdir.tail(tmp_path / "nope.log") == ""


def test_signal_choice_is_sigkill() -> None:
    # SIGTERM can be caught and ignored, which is exactly what a wedged process does.
    from bioflow.adapters.sidecar import runner as runner_module

    source = Path(runner_module.__file__).read_text()
    assert "signal.SIGKILL" in source
    assert signal.SIGKILL is not None
