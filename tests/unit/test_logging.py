"""Unit tests for structured logging."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from bioflow.core.logging import (
    bind_run,
    bind_step,
    bind_tool,
    clear_context,
    configure_logging,
    get_logger,
    logging_context,
    run_log_path,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_logging() -> Iterator[None]:
    """Leave global logging state as we found it."""
    clear_context()
    yield
    clear_context()
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_log_path_lands_under_logs(tmp_path: Path) -> None:
    assert run_log_path(tmp_path) == tmp_path / "logs" / "bioflow.jsonl"


def test_events_are_written_as_json_lines(tmp_path: Path) -> None:
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False)
    get_logger().info("quantifying", samples=6)

    (record,) = _lines(log_file)
    assert record["event"] == "quantifying"
    assert record["samples"] == 6
    assert record["level"] == "info"
    assert "timestamp" in record


def test_bound_context_rides_along(tmp_path: Path) -> None:
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False)
    bind_run("run_0123456789ab")
    bind_step("quantify")
    bind_tool("kallisto")
    get_logger().info("started")

    (record,) = _lines(log_file)
    assert record["run_id"] == "run_0123456789ab"
    assert record["step"] == "quantify"
    assert record["tool"] == "kallisto"


def test_third_party_stdlib_logs_reach_the_run_log(tmp_path: Path) -> None:
    # This is the whole reason for going through the stdlib bridge: httpx, Snakemake,
    # and the sidecars log through `logging`, and a run log without them is misleading.
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False)
    bind_run("run_0123456789ab")
    logging.getLogger("httpx").warning("connection reset")

    (record,) = _lines(log_file)
    assert record["event"] == "connection reset"
    assert record["level"] == "warning"
    assert record["logger"] == "httpx"
    assert record["run_id"] == "run_0123456789ab"


def test_logging_context_restores_the_outer_binding(tmp_path: Path) -> None:
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False)
    bind_run("run_0123456789ab")

    with logging_context(step="inner"):
        get_logger().info("during")
    get_logger().info("after")

    during, after = _lines(log_file)
    assert during["step"] == "inner"
    # Leaving a nested context must not take the outer run_id with it.
    assert after["run_id"] == "run_0123456789ab"
    assert "step" not in after


def test_clear_context_drops_everything(tmp_path: Path) -> None:
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False)
    bind_run("run_0123456789ab")
    clear_context()
    get_logger().info("orphan")

    (record,) = _lines(log_file)
    assert "run_id" not in record


def test_exceptions_are_serialized(tmp_path: Path) -> None:
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger().exception("failed")

    (record,) = _lines(log_file)
    assert record["level"] == "error"
    assert "ValueError" in str(record.get("exception", ""))


def test_reconfiguring_does_not_append_to_the_previous_run(tmp_path: Path) -> None:
    first = run_log_path(tmp_path / "run-a")
    second = run_log_path(tmp_path / "run-b")
    configure_logging(run_log=first, colors=False)
    get_logger().info("first run")
    configure_logging(run_log=second, colors=False)
    get_logger().info("second run")

    assert len(_lines(first)) == 1
    assert len(_lines(second)) == 1


def test_console_only_configuration_writes_no_file(tmp_path: Path) -> None:
    configure_logging(colors=False)
    get_logger().info("stderr only")
    assert not (tmp_path / "logs").exists()


def test_log_level_is_respected(tmp_path: Path) -> None:
    log_file = run_log_path(tmp_path)
    configure_logging(run_log=log_file, colors=False, level=logging.WARNING)
    get_logger().info("suppressed")
    get_logger().warning("kept")

    (record,) = _lines(log_file)
    assert record["event"] == "kept"
