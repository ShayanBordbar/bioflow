# BioFlow — Project Memory

## What this is
A provenance-first orchestration layer where an LLM performs JUDGMENT (assay triage, metadata structuring, config synthesis, failure diagnosis, evidence-bounded interpretation) and all scientific computation is deterministic, schema-validated, versioned, and reproducible WITHOUT the LLM.

## Non-negotiable invariants
1. The LLM never emits shell text. It emits validated JSON tool calls. subprocess uses shell=False.
2. The LLM never sees or produces filesystem paths. It handles artifact IDs only.
3. Every capability is callable deterministically from the CLI with --no-llm.
4. Every artifact is content-addressed (sha256) and has a recorded producer, tool version, and parameters.
5. Every report sentence carries an evidence reference [E:<id>]. Unreferenced claims fail the claim linter.
6. Evidence is typed: MEASURED | DERIVED | MODEL_PREDICTED | MODEL_INTERPRETED | HYPOTHESIS.
7. External tools are orchestrated, never reimplemented, and always version-pinned.
8. Validation reuses UPSTREAM schemas (nf-core assets/schema_input.json, nextflow_schema.json, MPRAsnakeflow workflow/schemas/*) — we add only semantic checks on top.
9. Heavy runtimes (torch/Polygraph/gReLU, Snakemake, Nextflow) live behind a sidecar process contract. They are NEVER imported into bioflow.core or bioflow.capabilities.
10. Ambiguous input produces NEEDS_REVIEW. Refusing to run is a correct answer.

## Architecture rules
- Layering: cli -> orchestration -> capabilities -> adapters -> external runtimes. Imports only ever point downward. capabilities must not import adapters directly; they receive adapter instances via constructor injection (ports & adapters).
- Rule of three: do not generalize until there are two real consumers.
- Prefer a plain ordered Plan with declared dependencies over a general DAG engine.
- Run state lives in SQLite. JSON/JSONL files are exports, not the source of truth.

## Sidecar contract (v1)
Heavy runtimes expose exactly one entrypoint:
<runtime> run --job job.json --outdir OUT
job.json = {"op": str, "contract_version": str, "params": {...}, "inputs": [ArtifactRef]}
result.json = {"status": "ok"|"error", "outputs": [ArtifactRef], "metrics": {...}, "versions": {...}, "error": StructuredError|null}
Adapters mock this with golden job/result JSON fixtures.

## Testing tiers (enforced by markers)
- unit: pure logic, <60s total, runs on every push
- contract: adapters against recorded fixtures + LLM cassettes, <3min, every push
- integration: real Docker/Nextflow/Snakemake/torch, nightly only
- e2e: full demos, manual
Never add a test that requires network or Docker without the integration/e2e marker.

## LLM usage rules
- All LLM calls go through bioflow.adapters.llm. Never call a provider SDK elsewhere.
- Every call is recorded to trace.jsonl and replayable from a cassette.
- Prompts live in src/bioflow/prompts/v<N>/*.md and are versioned; never inline prompt text.
- Temperature 0 by default; record model id, prompt version, token counts, cost.

## Scientific guardrails
- Predictions are labelled with model id, revision, checkpoint sha256, and domain status (IN_DOMAIN | OUT_OF_DOMAIN | POSSIBLE_LEAKAGE | UNKNOWN).
- Never phrase a model output as a measurement. "gReLU predicts X" not "X is active".
- Designed sequences are always stamped: IN-SILICO CANDIDATE — NOT EXPERIMENTALLY VALIDATED.
- Benchmarks require: baselines, a similarity-aware split, bootstrap CIs, and the assay replicate-correlation ceiling as a reference line.

## Repo conventions
- Public functions typed; mypy strict in core/capabilities.
- Errors are StructuredError instances with a closed enum error_code, never bare exceptions crossing a capability boundary.
- New capability => new directory under capabilities/ with models.py, service.py, tools.py, tests/.
- ADRs in docs/decisions/, numbered, written when a decision is actually made (with evidence).

## Out of scope (do not implement)
Multi-agent conversation, self-modifying tools, Kubernetes, Celery, vector DBs, web UI, MCP server, SLURM/AWS backends, arbitrary workflow support, accession-only MPRA inference, therapeutic sequence design.

## Commands
uv sync --all-extras --dev
uv run pytest -m unit
uv run pytest -m "unit or contract"
uv run ruff check . && uv run mypy src
uv run bioflow doctor
