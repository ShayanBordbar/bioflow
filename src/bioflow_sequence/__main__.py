"""The sidecar entrypoint: ``python -m bioflow_sequence run --job job.json --outdir out``.

Exactly the command the container's ENTRYPOINT composes and exactly the one
``LocalSidecarRunner`` builds, so a job behaves identically under both.

This process always exits 0 when it produced a well-formed ``result.json``, including for
a failed op -- a failure the sidecar *diagnosed* is a result, not a crash, and the caller
distinguishes them by ``status``. A non-zero exit means something went wrong that the
sidecar could not describe, which the runner reports as ``EXTERNAL_TOOL_FAILED``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from bioflow_sequence import CONTRACT_VERSION, __version__
from bioflow_sequence.ops import available, get_op

RESULT_FILENAME = "result.json"

EXIT_OK = 0
# Reserved for failures that leave no result.json: bad arguments, an unreadable job, an
# unwritable outdir. The runner reports these from the exit code and the stderr tail.
EXIT_NO_RESULT = 2


def build_parser() -> argparse.ArgumentParser:
    """The CLI, kept to exactly what the contract specifies."""
    parser = argparse.ArgumentParser(prog="bioflow_sequence", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="Execute one job.")
    run.add_argument("--job", required=True, help="Path to job.json, relative to the workdir.")
    run.add_argument("--outdir", required=True, help="Directory to write result.json into.")
    return parser


def error_result(code: str, message: str, remediation: str | None = None) -> dict[str, Any]:
    """Build a v1 error result.

    The shape matches ``StructuredError`` because the orchestrator validates it against
    the same model everything else crosses that boundary as.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "error",
        "outputs": [],
        "metrics": {},
        "versions": {"bioflow_sequence": __version__},
        "error": {
            "error_code": code,
            "message": message,
            "remediation": remediation,
            "details": {},
            "retryable": False,
            "raised_by": "bioflow_sequence",
        },
    }


def execute(job: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one parsed job to its op and shape the result."""
    version = job.get("contract_version")
    if version != CONTRACT_VERSION:
        # Refusing is the only safe answer: a v2 job may mean something different by a
        # field this build thinks it understands.
        return error_result(
            "SCHEMA_MISMATCH",
            f"job declares contract_version {version!r}, but this sidecar "
            f"speaks v{CONTRACT_VERSION}",
            "Upgrade the sidecar image, or pin the orchestrator to the matching contract.",
        )

    name = job.get("op")
    op = get_op(str(name))
    if op is None:
        return error_result(
            "NOT_FOUND",
            f"unknown op {name!r}",
            f"This sidecar implements: {', '.join(available())}.",
        )

    try:
        produced = op(job.get("params") or {}, list(job.get("inputs") or []), job.get("seed"))
    except Exception as exc:
        # The op failed in a way it did not anticipate. Reported as a result rather than
        # a traceback on stderr, so the orchestrator gets something it can act on.
        return error_result(
            "INTERNAL",
            f"op {name!r} raised {type(exc).__name__}: {exc}",
        ) | {"metrics": {"traceback_lines": len(traceback.format_exc().splitlines())}}

    versions = {"bioflow_sequence": __version__, **produced.get("versions", {})}
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "ok",
        "outputs": produced.get("outputs", []),
        "metrics": produced.get("metrics", {}),
        "versions": versions,
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    """Read the job, run it, write the result. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    job_path = Path(args.job)
    outdir = Path(args.outdir)
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # No result.json is possible here: without a job there is nothing to describe.
        print(f"could not read job {args.job}: {exc}", file=sys.stderr)
        return EXIT_NO_RESULT

    result = (
        execute(job)
        if isinstance(job, dict)
        else error_result("SCHEMA_MISMATCH", "job.json must contain a JSON object")
    )

    try:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / RESULT_FILENAME).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"could not write {RESULT_FILENAME}: {exc}", file=sys.stderr)
        return EXIT_NO_RESULT

    # Zero even for a failed op: the failure is in the result, and the runner reads it
    # there. A non-zero exit means "no usable result", which is a different thing.
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
