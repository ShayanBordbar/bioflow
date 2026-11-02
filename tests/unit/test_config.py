"""Unit tests for settings loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bioflow.config import (
    DEFAULT_ALLOWED_DOWNLOAD_DOMAINS,
    Settings,
    get_settings,
    reset_settings_cache,
)
from bioflow.core.models import CostClass

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip BIOFLOW_ vars and the .env file so tests see real defaults."""
    monkeypatch.delenv("BIOFLOW_DATA_DIR", raising=False)
    monkeypatch.delenv("BIOFLOW_RUN_DIR", raising=False)
    monkeypatch.delenv("BIOFLOW_STORE_DIR", raising=False)
    monkeypatch.delenv("BIOFLOW_OFFLINE", raising=False)
    monkeypatch.delenv("BIOFLOW_LLM_MODEL", raising=False)
    monkeypatch.delenv("BIOFLOW_ALLOWED_DOWNLOAD_DOMAINS", raising=False)
    monkeypatch.delenv("BIOFLOW_LLM_LOCAL_HOSTS", raising=False)
    # chdir away from the repo so the real .env (if any) is not picked up.
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()


def test_defaults_are_usable_without_configuration() -> None:
    settings = Settings()
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-opus-5"
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_repair_attempts == 2
    assert settings.sidecar_mode == "local"
    assert settings.cassette_mode == "off"
    assert settings.offline is False
    assert settings.require_confirmation_above is CostClass.MODERATE
    assert settings.allowed_download_domains == DEFAULT_ALLOWED_DOWNLOAD_DOMAINS


def test_default_domains_cover_the_expected_hosts() -> None:
    assert set(DEFAULT_ALLOWED_DOWNLOAD_DOMAINS) == {
        "ncbi.nlm.nih.gov",
        "eutils.ncbi.nlm.nih.gov",
        "ftp.sra.ebi.ac.uk",
        "www.ebi.ac.uk",
        "huggingface.co",
        "cdn-lfs.huggingface.co",
    }


def test_data_dir_cascades_into_run_and_store(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "workspace")
    assert settings.runs_root == tmp_path / "workspace" / "runs"
    assert settings.store_root == tmp_path / "workspace" / "store"


def test_explicit_run_and_store_dirs_are_kept(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data", run_dir=tmp_path / "elsewhere", store_dir=tmp_path / "objects"
    )
    assert settings.runs_root == tmp_path / "elsewhere"
    assert settings.store_root == tmp_path / "objects"


def test_env_vars_override_with_the_bioflow_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIOFLOW_DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setenv("BIOFLOW_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("BIOFLOW_OFFLINE", "true")
    monkeypatch.setenv("BIOFLOW_MAX_RUNTIME_MINUTES", "5")

    settings = Settings()
    assert settings.data_dir == tmp_path / "from-env"
    assert settings.llm_model == "claude-sonnet-5"
    assert settings.offline is True
    assert settings.max_runtime_minutes == 5


def test_allowed_domains_accepts_a_comma_separated_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pydantic-settings would otherwise demand JSON here, and nobody types JSON into an
    # environment variable.
    monkeypatch.setenv("BIOFLOW_ALLOWED_DOWNLOAD_DOMAINS", "a.example, b.example ,c.example")
    assert Settings().allowed_download_domains == ("a.example", "b.example", "c.example")


def test_allowed_domains_still_accepts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOFLOW_ALLOWED_DOWNLOAD_DOMAINS", '["a.example", "b.example"]')
    assert Settings().allowed_download_domains == ("a.example", "b.example")


def test_llm_local_hosts_default_to_empty() -> None:
    # Fails closed by default: widening the egress guard has to be deliberate.
    assert Settings().llm_local_hosts == ()


def test_llm_local_hosts_accept_a_comma_separated_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIOFLOW_LLM_LOCAL_HOSTS", "gpu-01, .cluster.internal")
    assert Settings().llm_local_hosts == ("gpu-01", ".cluster.internal")


def test_llm_local_hosts_accept_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOFLOW_LLM_LOCAL_HOSTS", '["gpu-01"]')
    assert Settings().llm_local_hosts == ("gpu-01",)


def test_require_confirmation_above_accepts_an_enum_name() -> None:
    assert Settings(require_confirmation_above="EXPENSIVE").require_confirmation_above is (
        CostClass.EXPENSIVE
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sidecar_mode", "kubernetes"),
        ("cassette_mode", "sometimes"),
        ("require_confirmation_above", "ENORMOUS"),
        ("llm_temperature", 3.0),
        ("llm_max_repair_attempts", -1),
        ("max_download_gb", 0),
        ("max_runtime_minutes", 0),
    ],
)
def test_invalid_values_are_rejected_at_startup(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
    reset_settings_cache()
    assert get_settings() is not None
