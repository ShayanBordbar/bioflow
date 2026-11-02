"""Environment diagnosis.

``bioflow doctor`` has to work on a machine where nothing is installed. Everything below
is therefore best-effort: a probe reports what it found, and a probe that cannot run
reports ``MISSING`` rather than raising. A diagnostic that dies when the thing it
diagnoses is absent tells you nothing you did not already know.

The probes are pure functions returning :class:`CheckResult`, so the CLI only renders and
the tests never have to parse a table.

External commands are run with ``shell=False`` (invariant #1) and a timeout, because a
wedged ``docker`` daemon should not hang the diagnosis.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bioflow.core.models import JSONValue

PROBE_TIMEOUT_SECONDS = 5.0
NETWORK_TIMEOUT_SECONDS = 2.0
LOW_DISK_GB = 10.0

# Candidate module names for the isolated sequence runtime, and the image that carries
# it when the sidecar runs in docker.
SEQUENCE_MODULES = ("grelu", "polygraph")
SEQUENCE_IMAGE = "bioflow/sequence-sidecar"
# The sidecar entrypoint itself, distinct from the heavy runtimes it may carry: a
# present entrypoint with no torch is a working sidecar with a limited op set.
SEQUENCE_ENTRYPOINT = "bioflow_sequence"

ANTHROPIC_PROFILE_DIR = Path.home() / ".config" / "anthropic"


class CheckStatus(StrEnum):
    """How a single environment check came out."""

    OK = "OK"
    MISSING = "MISSING"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class CheckResult:
    """One row of the doctor table."""

    name: str
    status: CheckStatus
    detail: str

    def to_json_dict(self) -> dict[str, JSONValue]:
        """Render the row as plain JSON, for ``doctor --json``."""
        return {"name": self.name, "status": self.status.value, "detail": self.detail}


def _run(command: list[str]) -> str | None:
    """Run a command and return its first output line, or None if that is impossible.

    Swallows every failure mode a missing tool can produce -- absent binary, non-zero
    exit, hang, permission error -- because each of them means the same thing here.
    """
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0].strip() if output else ""


def _tool_check(name: str, command: list[str], *, hint: str) -> CheckResult:
    """Probe an external command that reports its own version."""
    version = _run(command)
    if version is None:
        return CheckResult(name, CheckStatus.MISSING, f"not found on PATH; {hint}")
    return CheckResult(name, CheckStatus.OK, version or "present (no version reported)")


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def check_python() -> CheckResult:
    """Report the running interpreter."""
    return CheckResult("python", CheckStatus.OK, f"{platform.python_version()} at {sys.executable}")


def check_bioflow_version() -> CheckResult:
    """Report the installed BioFlow version."""
    from bioflow.cli import get_version

    return CheckResult("bioflow", CheckStatus.OK, get_version())


def check_git_commit() -> CheckResult:
    """Report the current commit, when running from a checkout."""
    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    if commit is None:
        return CheckResult("git commit", CheckStatus.DEGRADED, "not a git checkout, or git absent")
    return CheckResult("git commit", CheckStatus.OK, commit)


def check_disk(name: str, path: Path) -> CheckResult:
    """Report free space for a directory, walking up to one that exists."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return CheckResult(name, CheckStatus.MISSING, f"cannot stat {path}: {exc}")
    free_gb = usage.free / 1024**3
    status = CheckStatus.OK if free_gb >= LOW_DISK_GB else CheckStatus.DEGRADED
    return CheckResult(name, status, f"{free_gb:.1f} GB free at {path}")


def check_docker() -> CheckResult:
    """Probe the docker CLI, distinguishing 'absent' from 'daemon not running'."""
    if shutil.which("docker") is None:
        return CheckResult(
            "docker", CheckStatus.MISSING, "not found on PATH; needed for containerized runs"
        )
    version = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    if version is None:
        client = _run(["docker", "--version"])
        detail = "CLI present but daemon unreachable"
        if client:
            detail = f"{detail} ({client})"
        return CheckResult("docker", CheckStatus.DEGRADED, detail)
    return CheckResult("docker", CheckStatus.OK, f"server {version}")


def check_nextflow() -> CheckResult:
    """Probe Nextflow."""
    return _tool_check("nextflow", ["nextflow", "-version"], hint="needed for nf-core pipelines")


def check_java() -> CheckResult:
    """Probe Java, which Nextflow needs."""
    return _tool_check("java", ["java", "-version"], hint="required by nextflow")


def check_snakemake() -> CheckResult:
    """Probe Snakemake."""
    return _tool_check("snakemake", ["snakemake", "--version"], hint="needed for MPRAsnakeflow")


def check_sequence_sidecar(sidecar_mode: str = "local", image: str | None = None) -> CheckResult:
    """Report whether the isolated sequence runtime is reachable.

    Uses ``find_spec``, which locates a module without importing it -- importing torch
    or gReLU here would violate invariant #9 and cost seconds of startup.

    The entrypoint and the runtimes are checked separately because they fail separately.
    A present entrypoint with no runtimes is a usable sidecar that can answer ``ping``
    and nothing else, which is DEGRADED rather than MISSING -- reporting it as MISSING
    would break ``--strict`` on every machine that has not installed torch.
    """
    if sidecar_mode == "docker":
        if image is None:
            return CheckResult(
                "sequence sidecar",
                CheckStatus.MISSING,
                f"sidecar_mode=docker but BIOFLOW_SIDECAR_IMAGE is unset "
                f"(expected {SEQUENCE_IMAGE}@sha256:...)",
            )
        reference = image
        if "@sha256:" not in reference:
            # A moving tag makes a recorded run unreproducible without saying so.
            return CheckResult(
                "sequence sidecar",
                CheckStatus.DEGRADED,
                f"image {reference} is not pinned by digest",
            )
        if _run(["docker", "image", "inspect", reference, "--format", "{{.Id}}"]) is None:
            return CheckResult(
                "sequence sidecar", CheckStatus.MISSING, f"docker image {reference} not present"
            )
        return CheckResult("sequence sidecar", CheckStatus.OK, f"docker image {reference}")

    if not _importable(SEQUENCE_ENTRYPOINT):
        return CheckResult(
            "sequence sidecar",
            CheckStatus.MISSING,
            f"{SEQUENCE_ENTRYPOINT} not importable; reinstall bioflow",
        )

    found = [module for module in SEQUENCE_MODULES if _importable(module)]
    if not found:
        return CheckResult(
            "sequence sidecar",
            CheckStatus.DEGRADED,
            f"{SEQUENCE_ENTRYPOINT} present; no heavy runtime "
            f"({', '.join(SEQUENCE_MODULES)}) installed",
        )
    return CheckResult("sequence sidecar", CheckStatus.OK, f"local: {', '.join(found)}")


def _importable(module: str) -> bool:
    """Whether a module can be located, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_gpu() -> CheckResult:
    """Report GPU availability without importing a deep-learning framework.

    Probing by executable keeps heavy runtimes out of the CLI's import graph
    (invariant #9) and keeps ``doctor`` fast.
    """
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if gpu:
        return CheckResult("gpu", CheckStatus.OK, f"nvidia: {gpu}")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return CheckResult(
            "gpu", CheckStatus.DEGRADED, "Apple Silicon; MPS likely but unverified here"
        )
    return CheckResult("gpu", CheckStatus.MISSING, "no nvidia-smi; CPU only")


def check_llm_credentials(provider: str = "anthropic", api_key: str | None = None) -> CheckResult:
    """Report whether the configured provider has the credentials it needs.

    Provider-aware on purpose. A self-hosted OpenAI-compatible server usually needs no
    credential at all, so reporting MISSING there would fail ``--strict`` on a correctly
    configured machine -- the same false-negative class as checking only
    ``ANTHROPIC_API_KEY`` when the user authenticated with ``ant auth login``.
    """
    if provider == "openai-compatible":
        if api_key or os.environ.get("BIOFLOW_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            return CheckResult("llm credentials", CheckStatus.OK, "API key configured")
        return CheckResult(
            "llm credentials",
            CheckStatus.OK,
            "none configured; not required for a local or unauthenticated endpoint",
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult("llm credentials", CheckStatus.OK, "ANTHROPIC_API_KEY is set")
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return CheckResult("llm credentials", CheckStatus.OK, "ANTHROPIC_AUTH_TOKEN is set")
    try:
        if ANTHROPIC_PROFILE_DIR.is_dir() and any(ANTHROPIC_PROFILE_DIR.iterdir()):
            return CheckResult(
                "llm credentials", CheckStatus.OK, f"profile stored in {ANTHROPIC_PROFILE_DIR}"
            )
    except OSError:
        pass
    return CheckResult(
        "llm credentials",
        CheckStatus.MISSING,
        "no API key, auth token, or stored profile; run `ant auth login`",
    )


def check_llm_endpoint(
    provider: str, base_url: str | None, local_hosts: tuple[str, ...] = ()
) -> CheckResult:
    """Probe a self-hosted model server, reporting what it serves.

    Only meaningful for ``openai-compatible``: the Anthropic endpoint is fixed and its
    reachability is already covered by the network check.

    An unreachable server is DEGRADED rather than MISSING, because "not started yet" is
    the overwhelmingly common cause and is not a broken installation.
    """
    if provider != "openai-compatible":
        return CheckResult("llm endpoint", CheckStatus.OK, f"not applicable for {provider}")
    if not base_url:
        return CheckResult(
            "llm endpoint",
            CheckStatus.MISSING,
            "BIOFLOW_LLM_BASE_URL is unset; required for openai-compatible",
        )

    import httpx

    from bioflow.core.net import is_local_endpoint

    scope = "local" if is_local_endpoint(base_url, local_hosts) else "remote"
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=NETWORK_TIMEOUT_SECONDS)
    except Exception as exc:
        return CheckResult(
            "llm endpoint",
            CheckStatus.DEGRADED,
            f"{base_url} ({scope}) unreachable: {type(exc).__name__}; is the server running?",
        )

    if response.status_code >= 400:
        return CheckResult(
            "llm endpoint",
            CheckStatus.DEGRADED,
            f"{base_url} ({scope}) returned {response.status_code}",
        )
    try:
        served = [str(item.get("id", "")) for item in response.json().get("data", [])]
    except (ValueError, AttributeError):
        served = []
    detail = f"{base_url} ({scope}) reachable"
    if served:
        detail += f"; serving {', '.join(served[:5])}"
    return CheckResult("llm endpoint", CheckStatus.OK, detail)


def check_network(domains: tuple[str, ...], *, offline: bool) -> CheckResult:
    """Probe reachability of the allowed download domains."""
    if offline:
        return CheckResult("network", CheckStatus.DEGRADED, "skipped: offline mode is enabled")
    if not domains:
        return CheckResult("network", CheckStatus.DEGRADED, "no allowed domains configured")

    import httpx

    reachable: list[str] = []
    unreachable: list[str] = []
    for domain in domains:
        try:
            response = httpx.head(
                f"https://{domain}", timeout=NETWORK_TIMEOUT_SECONDS, follow_redirects=True
            )
            (reachable if response.status_code < 500 else unreachable).append(domain)
        except Exception:
            unreachable.append(domain)

    detail = f"{len(reachable)}/{len(domains)} allowed domains reachable"
    if not reachable:
        return CheckResult("network", CheckStatus.MISSING, f"{detail}; no connectivity")
    if unreachable:
        failed = ", ".join(unreachable)
        return CheckResult("network", CheckStatus.DEGRADED, f"{detail}; failed: {failed}")
    return CheckResult("network", CheckStatus.OK, detail)


# --------------------------------------------------------------------------------------
# The whole diagnosis
# --------------------------------------------------------------------------------------


def run_checks(settings: object = None) -> list[CheckResult]:
    """Run every check and return the rows, in display order.

    Accepts settings loosely so the CLI can pass a :class:`~bioflow.config.Settings` and
    tests can pass a stub without importing the whole configuration machinery.
    """
    from bioflow.adapters.llm.factory import normalize_provider
    from bioflow.config import Settings

    resolved = settings if isinstance(settings, Settings) else Settings()
    provider = normalize_provider(resolved.llm_provider)

    return [
        check_python(),
        check_bioflow_version(),
        check_git_commit(),
        check_disk("disk (data_dir)", resolved.data_dir),
        check_disk("disk (run_dir)", resolved.runs_root),
        check_docker(),
        check_nextflow(),
        check_java(),
        check_snakemake(),
        check_sequence_sidecar(resolved.sidecar_mode, resolved.sidecar_image),
        check_gpu(),
        check_llm_credentials(provider, resolved.llm_api_key),
        check_llm_endpoint(provider, resolved.llm_base_url, resolved.llm_local_hosts),
        check_network(resolved.allowed_download_domains, offline=resolved.offline),
    ]


def has_missing(results: list[CheckResult]) -> bool:
    """Report whether any check came back MISSING, which is what ``--strict`` acts on."""
    return any(result.status is CheckStatus.MISSING for result in results)
