"""The operations this sidecar can perform.

One op for now. The registry shape exists from the start because the alternative -- an
``if op == ...`` chain -- is how a dispatch table becomes untestable.

Nothing here imports a heavy runtime. ``ping`` reports whether torch is *available* and
at what version, and does it without importing torch: ``find_spec`` locates a module
without executing it and ``importlib.metadata.version`` reads the installed distribution's
metadata. Importing torch to ask its version costs seconds and a large amount of memory,
which is a strange price for a health check.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from importlib import metadata
from importlib.util import find_spec
from typing import Any

# Runtimes worth reporting on, mapped from import name to distribution name. They differ
# often enough (sklearn/scikit-learn, PIL/pillow) that assuming they match is a bug
# waiting for the first package that does not.
REPORTED_RUNTIMES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("grelu", "grelu"),
    ("polygraph", "polygraph"),
)

Op = Callable[[dict[str, Any], list[dict[str, Any]], "int | None"], dict[str, Any]]


def runtime_versions() -> dict[str, str]:
    """Report the versions of the runtimes present, without importing any of them."""
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    for module, distribution in REPORTED_RUNTIMES:
        try:
            if find_spec(module) is None:
                continue
        except (ImportError, ValueError):
            # A namespace package or a broken installation. Either way it is not usable,
            # and a health check must not fail on the thing it is checking.
            continue
        try:
            versions[module] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            # Importable but not pip-installed -- a source checkout on PYTHONPATH.
            versions[module] = "present (version unknown)"
    return versions


def ping(params: dict[str, Any], inputs: list[dict[str, Any]], seed: int | None) -> dict[str, Any]:
    """Report that the sidecar is alive and what it has available.

    The op that proves the boundary works before anything depends on it working.
    """
    return {
        "outputs": [],
        "metrics": {"n_inputs": len(inputs), "seed": seed},
        "versions": runtime_versions(),
    }


OPS: dict[str, Op] = {"ping": ping}


def get_op(name: str) -> Op | None:
    """Return an op by name, or ``None`` when it is not implemented here."""
    return OPS.get(name)


def available() -> list[str]:
    """Every op this sidecar implements, sorted."""
    return sorted(OPS)


# Only reached when the module is run directly, which nothing does; kept out of the
# entrypoint so `python -c "import bioflow_sequence.ops"` stays free of side effects.
if __name__ == "__main__":  # pragma: no cover
    print(sys.version)
