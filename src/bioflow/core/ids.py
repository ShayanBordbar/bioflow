"""Identifier minting.

The domain models in :mod:`bioflow.core.models` require every identifier to be passed
in explicitly — none of them carries a ``default_factory``. That is deliberate: it keeps
the minting policy in one place instead of scattering it across every model
construction site. This module is that place, and it is the only one allowed to invent
an identifier.

Identifiers are random, not content-derived. An artifact's *content* address is its
``sha256``; the ``artifact_id`` is a separate handle, so that storing the same bytes
twice produces two distinguishable records of two distinct events.
"""

from __future__ import annotations

import secrets

# Both halves of the format live in models.py as regex patterns; these lengths must
# agree with them. token_hex(n) returns 2n hex characters.
_LONG_ID_BYTES = 6  # -> 12 hex characters
_EVIDENCE_ID_BYTES = 3  # -> 6 hex characters


def new_artifact_id() -> str:
    """Mint an identifier matching ``ARTIFACT_ID_PATTERN``."""
    return f"art_{secrets.token_hex(_LONG_ID_BYTES)}"


def new_run_id() -> str:
    """Mint an identifier matching ``RUN_ID_PATTERN``."""
    return f"run_{secrets.token_hex(_LONG_ID_BYTES)}"


def new_event_id() -> str:
    """Mint an identifier matching ``EVENT_ID_PATTERN``."""
    return f"ev_{secrets.token_hex(_LONG_ID_BYTES)}"


def new_evidence_id() -> str:
    """Mint an identifier matching ``EVIDENCE_ID_PATTERN``."""
    return f"E{secrets.token_hex(_EVIDENCE_ID_BYTES)}"
