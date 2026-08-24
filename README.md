# BioFlow

Provenance-first orchestration for computational biology: it converts biological
objectives into validated, reproducible workflow and model operations while
keeping the scientific computation itself deterministic, inspectable, and
policy-controlled.

The agent is the control plane. Scientific workflows and models are the data
plane — BioFlow orchestrates established tools rather than reimplementing them.

> **Status:** pre-alpha. This is the repository scaffold; no capabilities are
> implemented yet.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --all-extras --dev
uv run bioflow --version
```

Checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

`pytest` deselects the `integration` and `e2e` markers by default; select them
explicitly with `-m integration` when the external tooling is available.

Optional install extras — `sequence` (Polygraph / gReLU / PyTorch) and
`workflows` (Snakemake tooling) — are declared but intentionally empty until the
integration spikes pin their dependencies.

## License

MIT. See [LICENSE](LICENSE).

<!-- TODO: temporary marker to verify the README check -->
