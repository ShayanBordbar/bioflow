"""Built-in agent-callable tools.

Importing this package registers every tool in it. Each submodule is imported
automatically, so dropping a new module in here cannot silently escape registration --
and therefore cannot escape ``tests/contract/test_all_tools_have_goldens.py``, which
requires a recorded golden case for every registered tool.
"""

from __future__ import annotations

import importlib
import pkgutil


def load_builtin_tools() -> None:
    """Import every submodule of this package, registering the tools it defines."""
    for module in pkgutil.iter_modules(__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module.name}")


load_builtin_tools()
