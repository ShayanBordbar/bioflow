"""Versioned prompt loading.

Prompts live in ``src/bioflow/prompts/v<N>/*.md`` and are never inlined in code. That is
what makes ``prompt_version`` a real field: a recorded run names the exact prompt text it
used, and changing a prompt is a reviewable diff rather than an invisible edit inside a
function.

Rendering uses ``StrictUndefined`` -- a mistyped variable raises instead of silently
rendering as an empty string. A prompt with a hole in it is worse than one that fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateError

from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError
from bioflow.core.paths import safe_resolve

PROMPTS_ROOT = Path(__file__).resolve().parent
DEFAULT_VERSION = "v1"
PROMPT_SUFFIX = ".md"

_RAISED_BY = "bioflow.prompts.loader"

_environment = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt and the version it came from."""

    name: str
    version: str
    text: str

    def __str__(self) -> str:
        return self.text


def _fail(
    code: ErrorCode,
    message: str,
    *,
    remediation: str | None = None,
    details: dict[str, JSONValue] | None = None,
) -> BioflowError:
    return BioflowError(
        StructuredError(
            error_code=code,
            message=message,
            remediation=remediation,
            details=details or {},
            raised_by=_RAISED_BY,
        )
    )


def prompt_path(name: str, version: str = DEFAULT_VERSION, *, root: Path | None = None) -> Path:
    """Return the file backing a prompt, rejecting names that escape the prompt tree."""
    base = (root or PROMPTS_ROOT) / version
    return safe_resolve(base, f"{name}{PROMPT_SUFFIX}")


def available_prompts(version: str = DEFAULT_VERSION, *, root: Path | None = None) -> list[str]:
    """List the prompt names available in a version."""
    directory = (root or PROMPTS_ROOT) / version
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob(f"*{PROMPT_SUFFIX}"))


def load_prompt(
    name: str,
    version: str = DEFAULT_VERSION,
    *,
    root: Path | None = None,
    **context: object,
) -> Prompt:
    """Load and render a versioned prompt template."""
    path = prompt_path(name, version, root=root)
    if not path.is_file():
        raise _fail(
            ErrorCode.NOT_FOUND,
            f"no prompt {name!r} in version {version!r}",
            remediation=(
                f"Available in {version}: "
                f"{', '.join(available_prompts(version, root=root)) or '(none)'}."
            ),
            details={"name": name, "version": version, "path": str(path)},
        )

    try:
        rendered = _environment.from_string(path.read_text(encoding="utf-8")).render(**context)
    except TemplateError as exc:
        raise _fail(
            ErrorCode.VALIDATION_FAILED,
            f"could not render prompt {name!r} ({version}): {exc}",
            remediation="Pass every variable the template references.",
            details={"name": name, "version": version},
        ) from exc

    return Prompt(name=name, version=version, text=rendered)
