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
job.json = {"contract_version": "1", "op": str, "params": {...}, "inputs": [ArtifactRef], "seed": int|null}
result.json = {"contract_version": "1", "status": "ok"|"error", "outputs": [ArtifactRef], "metrics": {...}, "versions": {...}, "stdout_tail": str, "stderr_tail": str, "error": StructuredError|null, "duration_ms": int}
Canonical JSON Schemas are checked in at schemas/sidecar_{job,result}.v1.json and are what
a non-Python sidecar reads; a unit test fails if they drift from the models.
- stdout_tail, stderr_tail, and duration_ms are filled by the RUNNER, never by the
  sidecar. A process is not asked to tail its own logs or time itself.
- Sidecars live in their own top-level package (src/bioflow_sequence), never under
  bioflow.*, and must not import bioflow or any orchestrator dependency. Enforced by an
  AST walk in tests/contract/test_sidecar.py. See ADR-0003.
- Sidecars speak JSON with the standard library. Do not add pydantic to a sidecar.
- Inputs are STAGED: the runner hardlinks each into <workdir>/inputs/ and rewrites the
  URI to a workdir-relative name. Sidecars never resolve cas:// and the store is never
  mounted into a container. job_sha256 ignores uri so staging does not change the hash.
- Timeouts kill the whole process group (start_new_session=True + killpg). Never use
  subprocess.run(timeout=...) for a sidecar: it signals only the direct child.
- Docker images are pinned by digest. A moving tag makes a recorded run unreproducible
  without saying so, and DockerSidecarRunner refuses one.
Adapters mock this with MockSidecarRunner, keyed by sha256 of the canonical job.

## Testing tiers (enforced by markers)
- unit: pure logic, <60s total, runs on every push
- contract: adapters against recorded fixtures + LLM cassettes, <3min, every push
- integration: real Docker/Nextflow/Snakemake/torch, nightly only
- e2e: full demos, manual
Never add a test that requires network or Docker without the integration/e2e marker.

## LLM usage rules
- All LLM calls go through bioflow.adapters.llm. Never call a provider SDK elsewhere.
- Build clients with get_llm_client(); never construct a provider client directly. Two
  providers: anthropic, and openai-compatible (one client for vLLM/Ollama/llama.cpp/TGI
  and hosted gateways, distinguished only by BIOFLOW_LLM_BASE_URL).
- Provider request/response translation lives in the adapters. bioflow.core never learns
  provider names; the registry emits one neutral tool schema. See ADR-0001.
- offline means no egress, not no LLM: a loopback or private-network endpoint is still
  permitted, so an air-gapped run can use a self-hosted model. Endpoint classification is
  lexical and fails closed; BIOFLOW_LLM_LOCAL_HOSTS widens it deliberately for clusters
  that address nodes by bare hostname.
- Every call is recorded to trace.jsonl and replayable from a cassette.
- Prompts live in src/bioflow/prompts/v<N>/*.md and are versioned; never inline prompt text.
- Temperature 0 by default; record model id, prompt version, token counts, cost.

## Agent loop rules
- The agent is a state machine (orchestration/fsm.py). Legal moves are a declared
  frozenset of (from, to) pairs; anything else raises. Never widen it to make a case
  work — add the edge deliberately or use an escape.
- Code advances the happy path; the LLM never names a state. Its only control-flow
  influence is request_review and handoff, whose targets are fixed. See ADR-0002.
- Per state, the LLM is shown only the tools in STATE_TOOLS. A call to anything else is
  rejected before it reaches invoke(). AWAIT_CONFIRM and EXECUTE offer nothing: the
  confirmation decision is a policy's, and plan execution is deterministic code.
- validate_args() runs before invoke(), so a repairable bad argument is never confused
  with a tool that failed on its own terms.
- Caps (agent_max_steps, agent_max_tool_calls_per_state, max_runtime_minutes) all end
  the run in FAILED with RESOURCE_LIMIT. Every cap reports the same error code.
- Confirmation arrives as an injected ConfirmationPolicy; orchestration never imports
  the CLI. The default is DenyAll — an unattended run declines.
- Every run writes runs/<run_id>/trace.jsonl and is replayable via replay() against its
  cassette. Provenance says what happened to which artifact; the trace says why.

## Context assembly rules
- ContextBuilder (orchestration/context.py) decides everything the LLM sees. The agent
  never assembles a prompt itself, and the message list is rebuilt per call rather than
  appended to — that is what makes the budget mean anything.
- Bounded window: the last agent_context_window_turns turns stay verbatim, older ones
  fold into one line each. Context must not grow with run length.
- The floor (system prompt, state objective, tool schemas, request, correction) is never
  truncated and is measured by rendering the block empty, not estimated. A floor over
  budget is RESOURCE_LIMIT, never an oversized request sent hopefully.
- Truncation always leaves a [truncated: N of M ...] marker. Never drop silently: a model
  that thinks it saw a whole table will describe a sample as the whole.
- Summarizers (orchestration/summarize.py, @summarizer("tsv")) never read a file whole —
  capped prefix or seek-to-tail only, and statistics over a sample say so. Never call
  read() without a size.
- Summarizer output is the one channel where file *content* reaches the prompt, so it is
  redacted centrally by redact_paths(). Do not bypass summarize() to render content.
- Token counting is a heuristic (chars/4) behind count_tokens(), swappable via
  set_token_counter(). Do not add a tokenizer dependency to get exactness nobody needs.

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
