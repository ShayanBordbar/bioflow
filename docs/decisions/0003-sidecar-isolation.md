# ADR-0003: The sequence sidecar is a separate top-level package, not a subpackage

## Status
Accepted — 2026-08-29.

## Context

Invariant #9 says torch, Polygraph, and gReLU are never imported into `bioflow.core` or
`bioflow.capabilities`. Until this step that held for one reason: none of them were
installed. Nothing enforced it, and the enforcement question is not academic. The first
capability that wants a sequence model will reach for an import, because an import is the
path of least resistance and the failure it causes is not local.

What breaks is not obvious at the call site. `bioflow doctor` — whose entire job is to run
on an unconfigured machine and report what is missing — would import torch on startup and
take seconds, or fail outright on a laptop with no CUDA. And `uv sync` would have to
satisfy torch's pins and pydantic's simultaneously, forever, for every contributor
including those who never touch a model.

Step 1.8's specification said the sidecar lives at `src/bioflow/sidecars/sequence/` and
also gave its Dockerfile `ENTRYPOINT ["python","-m","bioflow_sequence"]`. Those are
different module names, and the discrepancy turned out to be the interesting part of the
step: the two lines encode two different theories of what isolation means.

## Decision

The sequence sidecar is **`src/bioflow_sequence/`** — a separate top-level package that
shares no parent with `bioflow`.

Isolation is therefore structural rather than aspirational. The container image copies
`src/bioflow_sequence` and *not* `src/bioflow`, so `import bioflow.core` inside the
sidecar raises `ImportError` at run time in production. There is no configuration under
which it quietly works.

Three supporting decisions follow:

- **The sidecar speaks JSON with the standard library.** It does not import pydantic.
  A sidecar that needed the orchestrator's serialization library to speak the contract
  would not be independent of it, and `environments/sequence/requirements.txt` exists
  precisely so its dependency set can be resolved without reference to the root
  `pyproject.toml`.
- **The package still ships in the wheel.** Isolation is a property of the import graph,
  not of distribution. Shipping it is what makes `sidecar_mode=local` work from a plain
  install, which is the mode `bioflow doctor` already reports on.
- **The contract's JSON Schemas are checked in** at `schemas/sidecar_{job,result}.v1.json`.
  A second implementation in another language reads those, not `contract.py`. A schema
  generated only during a Python build is not a contract anyone else can build against.

## Consequences

**Easy.** A capability depending on a sequence model can be tested on a laptop with no
torch, no GPU, and no network, against recorded fixtures keyed by `job_sha256`. The
orchestrator's dependency set stays small enough that `bioflow doctor` starts instantly.
Adding a second sidecar (workflows, say) is a matter of copying the pattern.

**Hard.** Two packages in one repository, so tooling scope, the wheel's `packages` list,
and the Dockerfile's `COPY` all have to name both. Anything genuinely shared between the
orchestrator and a sidecar has to be duplicated or passed over the wire — which is the
cost of the boundary, and is meant to be felt.

**Ruled out.** Passing Python objects to a heavy runtime, and any future in which a
capability calls torch in-process because it is "just this once".

**A note on what the enforcement actually catches.** `tests/contract/test_sidecar.py`
walks the package's AST and fails on any import of `bioflow` or of the orchestrator's
dependencies. That is a *lint*, and lints can be bypassed. It exists to catch the mistake
hours earlier than the image build would, not because it is the real boundary. The real
boundary is that the runtime never has those modules on its path.

## Evidence

- The isolation test verified to fail: adding `from bioflow.core.models import ArtifactRef`
  to `src/bioflow_sequence/ops.py` produces
  `ops.py imports ['bioflow']; the sidecar's environment must resolve independently`.
- `test_the_sidecar_is_importable_without_the_orchestrator` imports the package in a
  subprocess with a `sys.meta_path` blocker that raises on any `bioflow` import, so the
  claim is proved rather than inferred.
- `test_ping_does_not_import_the_runtimes_it_reports_on` asserts `torch`, `grelu`, and
  `polygraph` are absent from `sys.modules` after `ping()` — version reporting uses
  `find_spec` plus `importlib.metadata`, neither of which executes the module.
- The schema drift guard verified to fail: adding a field to `SidecarResult` produces
  `sidecar_result.v1.json is out of date with the model` and names the regeneration
  command.
- The process-group kill verified to matter: with `start_new_session=False` the child
  joins pytest's own process group, and `killpg` terminates the test runner. That is the
  same mechanism that, in production, would leave a forked CUDA worker alive after a
  timeout — or kill something it should not.
