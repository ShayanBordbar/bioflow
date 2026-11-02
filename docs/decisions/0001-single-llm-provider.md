# ADR-0001: One LLM provider now, behind a factory that can take a second

## Status
Accepted — 2026-08-28. Amended 2026-08-28 (see Amendment below): the deferred
`OpenAICompatibleClient` has since been built, and one part of the original plan was wrong.

## Context

Step 1.5 landed the LLM adapter with a single implementation, `AnthropicClient`, speaking
the Messages API over raw httpx. `bioflow.config.Settings` declared `llm_provider`
(default `anthropic`) and `cassette_mode` (default `off`), and **neither was read by any
code**: the provider was hardcoded at every construction site and cassettes were wired up
by hand in tests. Both settings looked like switches and were not connected to anything,
which is worse than not offering them — it invites someone to change one and conclude the
change took effect.

Separately, the question came up of whether BioFlow supports open-weight models. It does
not today. That is worth deciding deliberately rather than by omission, because the
project has real reasons to want them: reproducibility over long horizons, running without
egress on a cluster holding controlled-access genomic data, and cost on high-volume
metadata triage.

Note that "open source model" is ambiguous in this project. The *sequence* models
(gReLU, Polygraph) are open source, are already planned, and reach BioFlow through the
sidecar contract — not through this adapter. Invariant #9 keeps those runtimes out of
`core` and `capabilities` entirely. This ADR is only about chat LLMs doing judgment work.

## Decision

Build LLM clients through a single factory, `bioflow.adapters.llm.get_llm_client`, which
dispatches on `settings.llm_provider` and applies `settings.cassette_mode`. Support
exactly one provider, `anthropic`. Any other value fails with a `StructuredError` that
names what is supported and what is planned.

Defer a second provider. When it arrives it will be **one** `OpenAICompatibleClient`
speaking `/v1/chat/completions`, which covers self-hosted vLLM, Ollama, llama.cpp, and
TGI as well as hosted OpenAI-compatible endpoints — rather than one client per vendor.

## Consequences

**Makes easy.** A second provider has exactly one place to land, and the settings now mean
what they say. Cassette record/replay is applied uniformly instead of per-call-site.
Replay needs no credentials and no network, because the wrapped client is never invoked.

**Makes hard, and what still has to be solved.** The `LLMClient` Protocol is neutral in its
return types but not yet in its inputs: `complete(messages=...)` takes provider-shaped
dicts, and `Registry.to_llm_schema()` emits Anthropic's `{name, description, input_schema}`
verbatim. For plain text turns the two formats coincide; they diverge on tool turns. A
second provider therefore requires four pieces of work, none of them the HTTP:

1. **Tool schema translation.** OpenAI-compatible wants
   `{type: "function", function: {name, description, parameters}}`. This belongs on the
   registry as `to_provider_schema(provider)`, not inside each client.
2. **Tool-call decoding.** `tool_calls[].function.arguments` arrives as a **JSON string**,
   not an object, and must be parsed rather than string-matched.
3. **Stop reasons.** `finish_reason` (`stop`, `length`, `tool_calls`) maps onto `StopReason`,
   and there is no `refusal` equivalent — a declining open-weight model just answers.
4. **Pricing.** Self-hosted inference is free, which is not the same as
   `cost_is_estimated=True`; that flag currently means "model absent from the table". A
   `self_hosted` price entry of zero is needed so free and unknown stay distinguishable.

**Rules out.** Per-vendor clients (one for Ollama, one for vLLM, …). They would all be the
same HTTP surface with different base URLs, and the rule of three does not justify them.

## Evidence

- `grep -rn "llm_provider\|cassette_mode" src/` before this change returned only the
  declarations in `config.py` and their assertions in `tests/unit/test_config.py` — no
  reader anywhere in `src/`.
- `AnthropicClient` hardcodes `https://api.anthropic.com/v1/messages`, the
  `anthropic-version: 2023-06-01` header, and the `x-api-key` / `Authorization: Bearer`
  header shapes, so it cannot be pointed at another endpoint by configuration alone.
- `pricing.PRICES` contains Claude models only; `cost_usd` returns `0.0` with
  `cost_is_estimated=True` for anything else, which is the behaviour a self-hosted model
  would incorrectly inherit today.
- The OpenAI-compatible surface is what vLLM, Ollama, llama.cpp, and TGI all expose, so
  one client covers the realistic self-hosting options rather than a family of them.

## Amendment — 2026-08-28

The deferred client is now implemented. Three things changed from the plan above, and
they are recorded here rather than left to contradict the original text.

**1. Tool-schema translation lives in the adapters, not on the registry.** The Decision
section proposed `Registry.to_provider_schema(provider)`. That was wrong: it would make
`bioflow.core` import the names of LLM providers, a dependency pointing against the
layering rule — and `Registry.to_llm_schema`'s own docstring already promised the
opposite ("Adapters translate this into whatever shape their provider wants"). The
registry still emits one neutral schema; `adapters/llm/tool_schema.py` translates.

**2. `offline` now means "no egress", not "no LLM".** Originally any LLM call under
`offline=True` was refused. A model served on loopback or a private network is not data
leaving the building, and refusing it made the air-gapped case — the main reason to want
open-weight support — impossible to satisfy. `core/net.py:is_local_endpoint` classifies
the endpoint and fails closed on any hostname it cannot resolve to a private address.
The Anthropic client still always refuses.

Because that classification is lexical, a cluster addressing nodes by bare name
(`gpu-01`) would be refused despite being inside the perimeter. `BIOFLOW_LLM_LOCAL_HOSTS`
is the deliberate widening: exact names or a `.hpc.example.edu` suffix. It defaults to
empty, so the guard stays closed unless an operator asserts otherwise — an assertion is
exactly what this should require, since nothing in the request itself can establish that
a name is internal.

**3. Cost distinguishes free from unknown via configured prices.** The plan called for a
`self_hosted` zero-price entry. What shipped instead is two optional settings
(`BIOFLOW_LLM_INPUT_PRICE_PER_MTOK` / `_OUTPUT_`). Unset with a local endpoint means
`cost_usd=0.0, cost_is_estimated=False` — genuinely free. A *remote* unpriced endpoint
stays `cost_is_estimated=True`, because there we really do not know.

Also added beyond the plan: `PROVIDER_ALIASES`, so `ollama`, `vllm`, `openai`, `local`
and friends all reach the one client rather than being reported as unknown providers;
and `doctor` gained an `llm endpoint` check plus provider-aware credentials, since the
old check reported MISSING for a correctly configured keyless local server.

Shared HTTP retry machinery moved to `adapters/llm/_http.py` during this work. Both
clients had the same ~25-line retry loop, and two copies would have drifted on retry
semantics — invisibly, until one of them hammered a rate-limited endpoint.
