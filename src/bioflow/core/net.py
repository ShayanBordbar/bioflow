"""Endpoint classification for the egress guard.

``offline`` means "no egress", not "no LLM". A model served on the same machine or the
same private network is not data leaving the building, and an air-gapped cluster holding
controlled-access genomic data is exactly the place that needs both a local model and a
hard guarantee that nothing is sent outward.

The guarantee is only as good as this classification, so it fails closed: a hostname that
is not an IP literal and not an explicit local name is treated as remote. Resolving names
to decide would be worse, since DNS can change under us between the check and the call.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Sequence

# Hostnames that mean "this machine" without being IP literals.
LOCAL_HOSTNAMES: frozenset[str] = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})

# mDNS: names in these zones are resolvable only on the local link.
LOCAL_SUFFIXES: tuple[str, ...] = (".local", ".localdomain", ".internal")


def endpoint_host(url: str) -> str | None:
    """Return the lowercased hostname of a URL, or ``None`` if it has none."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = parsed.hostname
    return host.lower() if host else None


def host_matches(host: str, patterns: Sequence[str]) -> bool:
    """Report whether a hostname matches any allowlist entry.

    Two forms are accepted, because a cluster addresses its nodes by bare name and
    listing ``gpu-01`` through ``gpu-64`` individually would be absurd:

    * an exact hostname -- ``my-gpu-box``
    * a suffix, written ``.cluster.internal`` or ``*.cluster.internal``, which matches
      any name ending in it (and the bare domain itself)
    """
    for raw in patterns:
        pattern = raw.strip().lower().removeprefix("*")
        if not pattern:
            continue
        if pattern.startswith("."):
            if host.endswith(pattern) or host == pattern.lstrip("."):
                return True
        elif host == pattern:
            return True
    return False


def is_local_endpoint(url: str, extra_local_hosts: Sequence[str] = ()) -> bool:
    """Report whether a URL addresses this machine, a private network, or an allowed host.

    True for loopback, RFC1918 / ULA private ranges, link-local addresses, the explicit
    local hostnames above, and anything in ``extra_local_hosts``. False for everything
    else -- including any other hostname that is not an IP literal, because a name we
    cannot classify must be assumed to leave the building.

    ``extra_local_hosts`` is how an operator widens that deliberately: naming a host
    there is an assertion that it sits inside their perimeter. The default stays closed
    precisely so that assertion has to be made explicitly.
    """
    host = endpoint_host(url)
    if host is None:
        return False
    if host in LOCAL_HOSTNAMES or host.endswith(LOCAL_SUFFIXES):
        return True
    if extra_local_hosts and host_matches(host, extra_local_hosts):
        return True

    try:
        # Strips the brackets IPv6 URLs carry; urlparse.hostname already does, but a
        # bare zone index would still fail parsing, which is the closed-fail path.
        address = ipaddress.ip_address(host)
    except ValueError:
        return False

    return bool(address.is_loopback or address.is_private or address.is_link_local)
