"""The isolated sequence runtime.

This package is deliberately **not** part of ``bioflow``. It is a separate top-level
distribution with its own requirements, and it must never import anything from the
orchestrator -- not ``bioflow.core``, and not the orchestrator's dependencies either.
That is what lets its environment carry torch and gReLU without the CLI's dependency
resolver ever having to satisfy them, and it is what keeps ``bioflow doctor`` runnable on
a laptop with no CUDA.

It speaks the JSON contract in ``schemas/sidecar_{job,result}.v1.json``, using the
standard library only. A sidecar that needed the orchestrator's serialization library to
speak the contract would not be independent of it.

``tests/contract/test_sidecar.py`` enforces all of this by walking this package's AST.
"""

__version__ = "0.1.0"
CONTRACT_VERSION = "1"

__all__ = ["CONTRACT_VERSION", "__version__"]
