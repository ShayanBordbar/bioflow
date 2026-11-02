"""Construction of the configured LLM client.

This is the one place that turns settings into a client. Before it existed,
``llm_provider`` and ``cassette_mode`` were declared in :mod:`bioflow.config` and read by
nothing -- settings that looked like switches and were not wired to anything. A setting
that does nothing is worse than an absent one, because it invites someone to change it
and conclude the change took effect.

Two providers are supported: ``anthropic`` and ``openai-compatible``. The latter is one
client for every server speaking ``/v1/chat/completions`` -- vLLM, Ollama, llama.cpp,
TGI, and hosted gateways -- so what distinguishes them is only ``llm_base_url``. See
``docs/decisions/0001-single-llm-provider.md``.

An unsupported provider fails here with a message naming what is available, rather than
being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioflow.adapters.llm.anthropic_client import AnthropicClient
from bioflow.adapters.llm.cassette import DEFAULT_CASSETTE_ROOT, CassetteClient
from bioflow.adapters.llm.openai_client import OpenAICompatibleClient
from bioflow.core.models import BioflowError, ErrorCode, StructuredError

if TYPE_CHECKING:
    from bioflow.adapters.llm.base import LLMClient
    from bioflow.config import Settings

SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai-compatible"})

# Providers that are designed for but not built. Kept here so the error message can name
# them, rather than leaving someone to guess whether their provider is unsupported or
# merely misspelled. Empty today; the branch below stays for the next one.
PLANNED_PROVIDERS: dict[str, str] = {}

# Aliases people will reasonably type for the OpenAI-compatible client. They all reach
# the same client -- what differs is only BIOFLOW_LLM_BASE_URL.
PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "vllm": "openai-compatible",
    "ollama": "openai-compatible",
    "llamacpp": "openai-compatible",
    "llama-cpp": "openai-compatible",
    "tgi": "openai-compatible",
    "local": "openai-compatible",
}

_RAISED_BY = "bioflow.adapters.llm.factory"


def get_llm_client(
    settings: Settings | None = None,
    *,
    cassette: str | None = None,
    cassette_root: Path = DEFAULT_CASSETTE_ROOT,
    **client_kwargs: Any,
) -> LLMClient:
    """Build the LLM client the settings describe.

    When ``cassette`` names a cassette and ``settings.cassette_mode`` is not ``off``, the
    provider client is wrapped for record or replay. In ``replay`` the wrapped client is
    never called, so replaying needs no credentials and no network.
    """
    from bioflow.config import get_settings

    resolved = settings if settings is not None else get_settings()
    client: LLMClient = _build_provider_client(resolved, **client_kwargs)

    if cassette is None or resolved.cassette_mode == "off":
        return client

    return CassetteClient(
        client,
        name=cassette,
        mode=resolved.cassette_mode,
        root=cassette_root,
        request_builder=client,
    )


def normalize_provider(provider: str) -> str:
    """Return the canonical provider name for a configured value."""
    cleaned = provider.strip().lower()
    return PROVIDER_ALIASES.get(cleaned, cleaned)


def _build_provider_client(settings: Settings, **client_kwargs: Any) -> LLMClient:
    """Dispatch on ``llm_provider``, or explain why we cannot."""
    provider = normalize_provider(settings.llm_provider)

    if provider == "anthropic":
        if settings.llm_base_url:
            # Silently ignoring it would let someone believe they had redirected
            # BioFlow at their own endpoint while it kept calling api.anthropic.com.
            raise BioflowError(
                StructuredError(
                    error_code=ErrorCode.VALIDATION_FAILED,
                    message=(
                        "BIOFLOW_LLM_BASE_URL is set, but the anthropic provider has a "
                        "fixed endpoint and would ignore it"
                    ),
                    remediation=(
                        "Unset BIOFLOW_LLM_BASE_URL, or set "
                        "BIOFLOW_LLM_PROVIDER=openai-compatible to use that endpoint."
                    ),
                    details={"llm_base_url": settings.llm_base_url},
                    raised_by=_RAISED_BY,
                )
            )
        return AnthropicClient(settings, **client_kwargs)

    if provider == "openai-compatible":
        # OpenAICompatibleClient raises MISSING_INPUT itself when llm_base_url is unset,
        # naming the setting and giving example values.
        return OpenAICompatibleClient(settings, **client_kwargs)

    planned = PLANNED_PROVIDERS.get(provider)
    supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
    if planned is not None:
        message = f"LLM provider {provider!r} is not implemented yet ({planned})"
        remediation = f"Set BIOFLOW_LLM_PROVIDER to one of: {supported}."
    else:
        message = f"unknown LLM provider {provider!r}"
        remediation = f"Set BIOFLOW_LLM_PROVIDER to one of: {supported}."
        if PLANNED_PROVIDERS:
            remediation += f" Planned: {', '.join(sorted(PLANNED_PROVIDERS))}."

    raise BioflowError(
        StructuredError(
            error_code=ErrorCode.VALIDATION_FAILED,
            message=message,
            remediation=remediation,
            details={
                "configured": settings.llm_provider,
                "supported": sorted(SUPPORTED_PROVIDERS),
                "planned": sorted(PLANNED_PROVIDERS),
            },
            raised_by=_RAISED_BY,
        )
    )
