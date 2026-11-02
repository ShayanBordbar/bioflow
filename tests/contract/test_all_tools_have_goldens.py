"""Enforcement: every registered tool must have recorded golden cases.

This file is not really testing behaviour. It is the mechanism that stops the
agent-visible surface from growing untested: register a tool and forget its goldens, and
the build fails here. Importing :mod:`bioflow.tools` is what populates the registry, and
that package imports its own submodules, so a new tool module cannot slip past.
"""

from pathlib import Path

import pytest

import bioflow.tools  # noqa: F401  (imported for the registration side effect)
from bioflow.core.contract_tests import IN_SUFFIX, assert_tool_contract, case_names
from bioflow.core.models import ToolSpec
from bioflow.core.registry import registry

pytestmark = pytest.mark.contract

GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "goldens"

SPECS = registry.list_specs()


def _ids(spec: ToolSpec) -> str:
    return spec.name


def test_at_least_one_tool_is_registered() -> None:
    # Without this, every parametrized test below would pass vacuously on an empty
    # registry -- the failure mode where the enforcement silently stops enforcing.
    assert SPECS, "no tools registered; bioflow.tools should have registered at least one"


@pytest.mark.parametrize("spec", SPECS, ids=_ids)
def test_tool_has_a_golden_directory(spec: ToolSpec) -> None:
    golden_dir = GOLDEN_ROOT / spec.name
    assert golden_dir.is_dir(), (
        f"tool {spec.name!r} has no golden directory. "
        f"Create {golden_dir.relative_to(GOLDEN_ROOT.parents[1])}/ with at least one "
        f"<case>{IN_SUFFIX} / <case>.out.json pair."
    )
    assert case_names(golden_dir), (
        f"tool {spec.name!r} has a golden directory with no cases in it; "
        f"add at least one <case>{IN_SUFFIX}."
    )


@pytest.mark.parametrize("spec", SPECS, ids=_ids)
def test_tool_matches_its_goldens(spec: ToolSpec) -> None:
    assert_tool_contract(spec.name, GOLDEN_ROOT / spec.name)


@pytest.mark.parametrize("spec", SPECS, ids=_ids)
def test_tool_schema_is_llm_ready(spec: ToolSpec) -> None:
    # Whatever else a tool declares, these are the parts a provider adapter has to be
    # able to hand to a model.
    assert spec.summary.strip(), f"tool {spec.name!r} has an empty summary"
    assert spec.input_schema.get("type") == "object"
    assert spec.output_schema.get("type") == "object"


def test_llm_schema_covers_every_registered_tool() -> None:
    schema = registry.to_llm_schema()
    assert [entry["name"] for entry in schema] == [spec.name for spec in SPECS]
    for entry in schema:
        assert set(entry) == {"name", "description", "input_schema"}
