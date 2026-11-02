"""Structured logging.

Two sinks, one event stream: human-readable colour on stderr for whoever is watching,
and JSON lines in ``runs/<run_id>/logs/bioflow.jsonl`` for whoever is reading afterwards.

The plumbing goes through stdlib ``logging`` via :class:`structlog.stdlib.ProcessorFormatter`
rather than structlog alone. That is more setup, but it means records emitted by httpx,
Snakemake, and the sidecar runtimes land in the run log too -- and those are exactly the
records someone opens a run log to find.

``run_id``, ``step``, and ``tool`` ride along on contextvars, so a log call deep inside a
capability does not have to thread them through its own signature.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

RUN_LOG_NAME = "bioflow.jsonl"

_shared_processors: tuple[Any, ...] = (
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    run_log: Path | None = None,
    colors: bool = True,
) -> None:
    """Install the stderr and (optionally) run-log handlers.

    Calling this again replaces the handlers, so a second run does not append to the
    first run's log file.
    """
    structlog.configure(
        processors=[
            *_shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=list(_shared_processors),
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=colors),
            ],
        )
    )
    root.addHandler(console)

    if run_log is not None:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_log, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=list(_shared_processors),
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        root.addHandler(file_handler)

    root.setLevel(level)


def get_logger(name: str = "bioflow") -> structlog.stdlib.BoundLogger:
    """Return a bound logger."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def run_log_path(run_dir: Path) -> Path:
    """Return the JSONL log path inside a run directory."""
    return run_dir / "logs" / RUN_LOG_NAME


# --------------------------------------------------------------------------------------
# Context binding
# --------------------------------------------------------------------------------------


def bind_run(run_id: str) -> None:
    """Attach a run ID to every subsequent log record on this task."""
    structlog.contextvars.bind_contextvars(run_id=run_id)


def bind_step(step: str) -> None:
    """Attach a step name to every subsequent log record on this task."""
    structlog.contextvars.bind_contextvars(step=step)


def bind_tool(tool: str) -> None:
    """Attach a tool name to every subsequent log record on this task."""
    structlog.contextvars.bind_contextvars(tool=tool)


def clear_context() -> None:
    """Drop everything bound to the current context."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def logging_context(**bindings: str) -> Iterator[None]:
    """Bind values for the duration of a block, then restore what was there before.

    Restoring rather than clearing matters for nesting: a ``step`` context inside a
    ``run`` context must not take the ``run_id`` with it on the way out.
    """
    tokens = structlog.contextvars.bind_contextvars(**bindings)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
