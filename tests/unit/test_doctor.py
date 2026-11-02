"""Unit tests for the environment probes.

The governing requirement is that doctor works on a machine with nothing installed, so
most of these tests remove things rather than provide them.
"""

import subprocess
from pathlib import Path

import pytest

from bioflow.config import Settings
from bioflow.core import doctor
from bioflow.core.doctor import (
    CheckResult,
    CheckStatus,
    check_bioflow_version,
    check_disk,
    check_docker,
    check_git_commit,
    check_gpu,
    check_java,
    check_llm_credentials,
    check_llm_endpoint,
    check_network,
    check_nextflow,
    check_python,
    check_sequence_sidecar,
    check_snakemake,
    has_missing,
    run_checks,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an empty PATH."""
    monkeypatch.setattr(doctor.shutil, "which", lambda _cmd: None)


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(doctor, "ANTHROPIC_PROFILE_DIR", tmp_path / "absent")


# --------------------------------------------------------------------------------------
# Checks that always work
# --------------------------------------------------------------------------------------


def test_python_and_version_always_report_ok() -> None:
    assert check_python().status is CheckStatus.OK
    assert check_bioflow_version().status is CheckStatus.OK
    assert check_bioflow_version().detail


def test_disk_reports_free_space(tmp_path: Path) -> None:
    result = check_disk("disk", tmp_path)
    assert result.status in (CheckStatus.OK, CheckStatus.DEGRADED)
    assert "GB free" in result.detail


def test_disk_walks_up_to_an_existing_parent(tmp_path: Path) -> None:
    # data_dir usually does not exist yet on a first run; that is not an error.
    result = check_disk("disk", tmp_path / "not" / "created" / "yet")
    assert result.status in (CheckStatus.OK, CheckStatus.DEGRADED)


def test_disk_degrades_when_space_is_low(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda _p: type("U", (), {"free": 1024**3})())
    assert check_disk("disk", tmp_path).status is CheckStatus.DEGRADED


def test_disk_reports_missing_when_it_cannot_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(_path: object) -> None:
        raise OSError("no such device")

    monkeypatch.setattr(doctor.shutil, "disk_usage", boom)
    assert check_disk("disk", tmp_path).status is CheckStatus.MISSING


# --------------------------------------------------------------------------------------
# Absent tooling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probe",
    [check_docker, check_nextflow, check_java, check_snakemake],
    ids=lambda p: p.__name__,
)
def test_absent_tools_report_missing_without_raising(
    probe: object, nothing_installed: None
) -> None:
    result = probe()  # type: ignore[operator]
    assert isinstance(result, CheckResult)
    assert result.status is CheckStatus.MISSING
    assert "not found on PATH" in result.detail


def test_git_commit_degrades_outside_a_checkout(nothing_installed: None) -> None:
    assert check_git_commit().status is CheckStatus.DEGRADED


def test_gpu_reports_something_on_every_platform(nothing_installed: None) -> None:
    # No nvidia-smi: either MISSING, or DEGRADED on Apple Silicon.
    assert check_gpu().status in (CheckStatus.MISSING, CheckStatus.DEGRADED)


def test_sequence_sidecar_missing_when_nothing_is_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _name: None)
    result = check_sequence_sidecar("local")
    assert result.status is CheckStatus.MISSING
    assert "sequence" in result.detail


def test_sequence_sidecar_found_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "SEQUENCE_MODULES", ("json",))
    result = check_sequence_sidecar("local")
    assert result.status is CheckStatus.OK
    assert "json" in result.detail


def test_sequence_sidecar_survives_a_broken_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_name: str) -> None:
        raise ValueError("__spec__ is not set")

    monkeypatch.setattr(doctor.importlib.util, "find_spec", boom)
    assert check_sequence_sidecar("local").status is CheckStatus.MISSING


def test_sequence_sidecar_in_docker_mode_needs_an_image(nothing_installed: None) -> None:
    result = check_sequence_sidecar("docker")
    assert result.status is CheckStatus.MISSING
    assert "BIOFLOW_SIDECAR_IMAGE" in result.detail
    assert doctor.SEQUENCE_IMAGE in result.detail  # names the expected shape


def test_an_unpinned_image_is_degraded_not_ok(nothing_installed: None) -> None:
    # A moving tag makes a recorded run unreproducible without saying so, so doctor says
    # so instead. DEGRADED rather than MISSING: the image may well be there and work.
    result = check_sequence_sidecar("docker", "bioflow/sequence-sidecar:latest")
    assert result.status is CheckStatus.DEGRADED
    assert "digest" in result.detail


def test_a_pinned_image_that_is_absent_is_missing(nothing_installed: None) -> None:
    result = check_sequence_sidecar("docker", f"bioflow/sequence-sidecar@sha256:{'0' * 64}")
    assert result.status is CheckStatus.MISSING
    assert "not present" in result.detail


def test_the_local_sidecar_entrypoint_is_present() -> None:
    # It ships in the wheel, so this is OK or DEGRADED on any correct install -- never
    # MISSING, which would break `doctor --strict` for everyone.
    result = check_sequence_sidecar("local")
    assert result.status is not CheckStatus.MISSING
    assert doctor.SEQUENCE_ENTRYPOINT in result.detail or "local:" in result.detail


def test_a_sidecar_without_heavy_runtimes_is_degraded_not_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A present entrypoint with no torch is a working sidecar with a limited op set.
    # Calling that MISSING would fail --strict on every machine that has not installed
    # a multi-gigabyte dependency it may never need.
    monkeypatch.setattr(doctor, "_importable", lambda name: name == doctor.SEQUENCE_ENTRYPOINT)
    result = check_sequence_sidecar("local")
    assert result.status is CheckStatus.DEGRADED
    assert "no heavy runtime" in result.detail


def test_a_missing_entrypoint_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_importable", lambda _name: False)
    result = check_sequence_sidecar("local")
    assert result.status is CheckStatus.MISSING
    assert "reinstall" in result.detail


# --------------------------------------------------------------------------------------
# Subprocess robustness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("gone"),
        PermissionError("denied"),
        subprocess.TimeoutExpired(cmd="docker", timeout=5),
        subprocess.SubprocessError("weird"),
    ],
    ids=["not-found", "denied", "timeout", "subprocess-error"],
)
def test_a_failing_subprocess_never_escapes(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    # A wedged docker daemon must not take the whole diagnosis down with it.
    monkeypatch.setattr(doctor.shutil, "which", lambda _cmd: "/usr/bin/docker")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(doctor.subprocess, "run", boom)
    assert doctor._run(["docker", "version"]) is None
    assert check_docker().status is CheckStatus.DEGRADED


def test_a_nonzero_exit_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _cmd: "/usr/bin/thing")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], returncode=1, stdout="", stderr="err"),
    )
    assert doctor._run(["thing", "--version"]) is None


def test_docker_cli_without_a_daemon_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _cmd: "/usr/bin/docker")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "version" in command and "--format" in command:
            return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="down")
        return subprocess.CompletedProcess(command, returncode=0, stdout="Docker version 27\n")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    result = check_docker()
    assert result.status is CheckStatus.DEGRADED
    assert "daemon unreachable" in result.detail


def test_probes_never_use_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    # Invariant #1: subprocess uses shell=False.
    seen: list[object] = []

    monkeypatch.setattr(doctor.shutil, "which", lambda _cmd: "/usr/bin/thing")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs.get("shell"))
        return subprocess.CompletedProcess(command, returncode=0, stdout="1.0\n")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    doctor._run(["thing", "--version"])
    assert seen == [False]


# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------


def test_credentials_found_via_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = check_llm_credentials()
    assert result.status is CheckStatus.OK
    assert "ANTHROPIC_API_KEY" in result.detail
    assert "sk-test" not in result.detail  # never echo the secret


def test_credentials_found_via_auth_token(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    assert check_llm_credentials().status is CheckStatus.OK


def test_credentials_found_via_stored_profile(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None, tmp_path: Path
) -> None:
    # An OAuth profile is a real credential; reporting MISSING here would fail --strict
    # for a perfectly well authenticated user.
    profile_dir = tmp_path / "anthropic"
    profile_dir.mkdir()
    (profile_dir / "profile.json").write_text("{}")
    monkeypatch.setattr(doctor, "ANTHROPIC_PROFILE_DIR", profile_dir)

    result = check_llm_credentials()
    assert result.status is CheckStatus.OK
    assert "profile" in result.detail


def test_openai_compatible_needs_no_credentials(no_credentials: None) -> None:
    # A self-hosted server usually has no auth at all. Reporting MISSING here would fail
    # --strict on a correctly configured machine -- the same false-negative class as
    # checking only ANTHROPIC_API_KEY when the user authenticated with `ant auth login`.
    result = check_llm_credentials("openai-compatible")
    assert result.status is CheckStatus.OK
    assert "not required" in result.detail


def test_openai_compatible_reports_a_configured_key(no_credentials: None) -> None:
    result = check_llm_credentials("openai-compatible", "sk-local")
    assert result.status is CheckStatus.OK
    assert "sk-local" not in result.detail


def test_openai_compatible_finds_openai_api_key(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert check_llm_credentials("openai-compatible").status is CheckStatus.OK


# --------------------------------------------------------------------------------------
# The model-server endpoint
# --------------------------------------------------------------------------------------


def test_endpoint_check_is_skipped_for_anthropic() -> None:
    result = check_llm_endpoint("anthropic", None)
    assert result.status is CheckStatus.OK
    assert "not applicable" in result.detail


def test_endpoint_check_requires_a_base_url() -> None:
    result = check_llm_endpoint("openai-compatible", None)
    assert result.status is CheckStatus.MISSING
    assert "BIOFLOW_LLM_BASE_URL" in result.detail


def test_endpoint_check_reports_the_served_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *_a, **_k: httpx.Response(200, json={"data": [{"id": "llama3.1:8b"}]}),
    )
    result = check_llm_endpoint("openai-compatible", "http://localhost:11434/v1")
    assert result.status is CheckStatus.OK
    assert "llama3.1:8b" in result.detail
    assert "(local)" in result.detail


def test_an_unreachable_server_is_degraded_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def refuse(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", refuse)
    result = check_llm_endpoint("openai-compatible", "http://localhost:11434/v1")
    # "Not started yet" is the overwhelmingly common cause and is not a broken install.
    assert result.status is CheckStatus.DEGRADED
    assert "is the server running?" in result.detail


def test_an_erroring_server_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: httpx.Response(503))
    result = check_llm_endpoint("openai-compatible", "http://localhost:11434/v1")
    assert result.status is CheckStatus.DEGRADED
    assert "503" in result.detail


def test_an_allowlisted_host_is_labelled_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: httpx.Response(200, json={"data": []}))
    result = check_llm_endpoint("openai-compatible", "http://gpu-01:8000/v1", ("gpu-01",))
    assert "(local)" in result.detail


def test_a_remote_endpoint_is_labelled_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: httpx.Response(200, json={"data": []}))
    result = check_llm_endpoint("openai-compatible", "https://api.together.xyz/v1")
    assert "(remote)" in result.detail


def test_credentials_missing_reports_how_to_fix_it(no_credentials: None) -> None:
    result = check_llm_credentials()
    assert result.status is CheckStatus.MISSING
    assert "ant auth login" in result.detail


def test_credentials_survive_an_unreadable_profile_dir(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None
) -> None:
    class Unreadable:
        def is_dir(self) -> bool:
            raise OSError("permission denied")

    monkeypatch.setattr(doctor, "ANTHROPIC_PROFILE_DIR", Unreadable())
    assert check_llm_credentials().status is CheckStatus.MISSING


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------


def test_network_is_skipped_when_offline() -> None:
    result = check_network(("example.com",), offline=True)
    assert result.status is CheckStatus.DEGRADED
    assert "offline" in result.detail


def test_network_with_no_domains_configured() -> None:
    assert check_network((), offline=False).status is CheckStatus.DEGRADED


def test_network_reports_unreachable_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def always_fail(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "head", always_fail)
    result = check_network(("a.example", "b.example"), offline=False)
    assert result.status is CheckStatus.MISSING
    assert "no connectivity" in result.detail


def test_network_reports_partial_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def sometimes(url: str, **_k: object) -> httpx.Response:
        if "bad" in url:
            raise httpx.ConnectError("nope")
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "head", sometimes)
    result = check_network(("good.example", "bad.example"), offline=False)
    assert result.status is CheckStatus.DEGRADED
    assert "bad.example" in result.detail


def test_network_all_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "head", lambda *_a, **_k: httpx.Response(200))
    assert check_network(("good.example",), offline=False).status is CheckStatus.OK


def test_a_server_error_counts_as_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "head", lambda *_a, **_k: httpx.Response(503))
    assert check_network(("down.example",), offline=False).status is CheckStatus.MISSING


# --------------------------------------------------------------------------------------
# The whole diagnosis
# --------------------------------------------------------------------------------------


def test_run_checks_on_a_bare_machine(
    tmp_path: Path, nothing_installed: None, no_credentials: None
) -> None:
    settings = Settings(data_dir=tmp_path / "workspace", offline=True)
    results = run_checks(settings)

    assert len(results) == 14
    assert {r.name for r in results} >= {
        "python",
        "docker",
        "gpu",
        "llm credentials",
        "llm endpoint",
        "network",
    }
    assert has_missing(results)
    # The acceptance criterion: nothing raised, every row has a status and a detail.
    assert all(isinstance(r.status, CheckStatus) and r.detail for r in results)


def test_run_checks_defaults_to_loaded_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOFLOW_OFFLINE", "true")
    results = run_checks()
    network = next(r for r in results if r.name == "network")
    assert "offline" in network.detail


def test_has_missing_is_false_when_nothing_is_missing() -> None:
    assert not has_missing([CheckResult("a", CheckStatus.OK, "fine")])
    assert not has_missing([CheckResult("a", CheckStatus.DEGRADED, "meh")])
    assert has_missing([CheckResult("a", CheckStatus.MISSING, "gone")])


def test_check_result_serializes_for_json_output() -> None:
    assert CheckResult("docker", CheckStatus.MISSING, "absent").to_json_dict() == {
        "name": "docker",
        "status": "MISSING",
        "detail": "absent",
    }
