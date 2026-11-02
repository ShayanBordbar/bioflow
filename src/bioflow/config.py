"""Runtime configuration.

Every setting is overridable from the environment with a ``BIOFLOW_`` prefix, or from a
``.env`` file. Defaults are chosen so that a fresh checkout works with no configuration
at all: directories land in the platform's standard data location, and anything that
costs money or reaches the network is conservative.

The safety-shaped settings -- ``allowed_download_domains``, ``max_download_gb``,
``max_runtime_minutes``, ``require_confirmation_above`` -- are a deny-by-default budget,
not advice. They exist because the layer above this one takes instructions from a
language model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from platformdirs import user_data_dir
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from bioflow.core.models import CostClass

SidecarMode = Literal["local", "docker"]
CassetteMode = Literal["off", "record", "replay"]

DEFAULT_ALLOWED_DOWNLOAD_DOMAINS: tuple[str, ...] = (
    "ncbi.nlm.nih.gov",
    "eutils.ncbi.nlm.nih.gov",
    "ftp.sra.ebi.ac.uk",
    "www.ebi.ac.uk",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
)


def _default_data_dir() -> Path:
    return Path(user_data_dir("bioflow"))


class Settings(BaseSettings):
    """BioFlow's runtime settings, read from ``BIOFLOW_*`` env vars and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="BIOFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- locations ------------------------------------------------------------------
    data_dir: Path = Field(default_factory=_default_data_dir)
    run_dir: Path | None = None
    store_dir: Path | None = None

    # -- LLM ------------------------------------------------------------------------
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_repair_attempts: int = Field(default=2, ge=0)
    # Required for openai-compatible (vLLM, Ollama, llama.cpp, TGI, hosted gateways);
    # rejected for anthropic, whose endpoint is fixed.
    llm_base_url: str | None = None
    # Optional: local servers usually need no credential at all, so an unset key means
    # no Authorization header rather than a failure.
    llm_api_key: str | None = None
    # Set these for a paid OpenAI-compatible gateway. Left unset, a non-Claude model is
    # treated as self-hosted and therefore free -- which is different from "unknown".
    llm_input_price_per_mtok: float | None = Field(default=None, ge=0.0)
    llm_output_price_per_mtok: float | None = Field(default=None, ge=0.0)
    # Hosts to treat as inside the perimeter under `offline`, beyond the loopback and
    # private-address ranges recognized automatically. A cluster addressing its nodes by
    # bare name ("gpu-01") cannot be classified from the name alone, so the guard fails
    # closed and the operator widens it here on purpose. Exact names or a
    # ".cluster.internal" suffix.
    llm_local_hosts: Annotated[tuple[str, ...], NoDecode] = ()

    # -- agent loop -------------------------------------------------------------------
    # Hard caps on the orchestration loop. There is no wall-clock setting of its own:
    # max_runtime_minutes below is the run's time budget, and the agent loop is part of
    # the run rather than something with a separate allowance.
    agent_max_steps: int = Field(default=25, gt=0)
    agent_max_tool_calls_per_state: int = Field(default=8, gt=0)
    # How much context one LLM call may carry. Estimated, not exact -- see
    # orchestration/tokens.py. Exceeding it truncates the elastic sections with an
    # explicit marker; a floor that exceeds it on its own is a RESOURCE_LIMIT failure.
    agent_token_budget: int = Field(default=12000, gt=0)
    # Turns kept verbatim. Everything older folds into one line each, so history stays
    # present without growing without bound.
    agent_context_window_turns: int = Field(default=4, ge=0)

    # -- budgets --------------------------------------------------------------------
    max_download_gb: float = Field(default=20.0, gt=0)
    max_runtime_minutes: int = Field(default=120, gt=0)
    require_confirmation_above: CostClass = CostClass.MODERATE
    # NoDecode stops pydantic-settings from JSON-decoding the env value before the
    # validator below runs; without it a comma-separated list raises during source
    # parsing, which no field validator can intercept.
    allowed_download_domains: Annotated[tuple[str, ...], NoDecode] = (
        DEFAULT_ALLOWED_DOWNLOAD_DOMAINS
    )

    # -- execution ------------------------------------------------------------------
    sidecar_mode: SidecarMode = "local"
    # The container the sequence sidecar runs in under sidecar_mode=docker. Must be
    # pinned by digest: a moving tag makes a recorded run unreproducible in the quietest
    # possible way -- the same command, months later, running different code and
    # reporting no difference. Unset means docker mode is unconfigured, which the runner
    # refuses rather than guessing a tag.
    sidecar_image: str | None = None
    # Wall clock for one sidecar job. On expiry the whole process group is killed, not
    # just the direct child.
    sidecar_timeout_seconds: float = Field(default=3600.0, gt=0)
    offline: bool = False
    cassette_mode: CassetteMode = "off"

    @field_validator("allowed_download_domains", "llm_local_hosts", mode="before")
    @classmethod
    def _split_domains(cls, value: object) -> object:
        """Accept a comma-separated string as well as a JSON list.

        ``BIOFLOW_ALLOWED_DOWNLOAD_DOMAINS=a.com,b.com`` is what someone will actually
        type, so both forms are handled. The fields are annotated ``NoDecode`` so the raw
        string reaches this validator instead of failing during source parsing.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                parsed: object = json.loads(text)
                return parsed
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _derive_directories(self) -> Settings:
        """Point run_dir and store_dir under data_dir unless they were set explicitly.

        Overriding ``data_dir`` alone therefore moves the whole workspace coherently,
        which is what someone pointing BioFlow at a scratch volume expects.
        """
        if self.run_dir is None:
            object.__setattr__(self, "run_dir", self.data_dir / "runs")
        if self.store_dir is None:
            object.__setattr__(self, "store_dir", self.data_dir / "store")
        return self

    @property
    def runs_root(self) -> Path:
        """The run directory, narrowed to a non-optional path for call sites."""
        assert self.run_dir is not None  # guaranteed by _derive_directories
        return self.run_dir

    @property
    def store_root(self) -> Path:
        """The object store directory, narrowed to a non-optional path."""
        assert self.store_dir is not None  # guaranteed by _derive_directories
        return self.store_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once."""
    return Settings()


def reset_settings_cache() -> None:
    """Forget the cached settings. For tests that manipulate the environment."""
    get_settings.cache_clear()
