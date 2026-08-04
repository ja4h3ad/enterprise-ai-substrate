# ADR 0002: untrusted evidence, explicit degradation, and privacy-safe telemetry

Status: Accepted

## Context

Contract retrieval can surface instruction-like text, malformed model output, invalid tool arguments, or a slow local retriever. Treating those conditions as ordinary prompt content or silently dropping a component would make the security and reliability claims impossible to inspect.

## Decision

All retrieved clause text is data in the `untrusted_evidence` trust zone. The model-provider boundary receives a typed grounding envelope with a fixed policy field and a separate tuple of evidence records. Instruction-like phrases are flagged deterministically after retrieval, retained as evidence when relevant, and recorded by clause identifier and rule name. They cannot modify the already validated retrieval plan, tool selection, limits, evidence contract, or answer policy.

Pydantic models forbid extra fields. Typed graph requests validate contract and clause identifier formats, graph depth, and candidate limits before NetworkX is called. Boundary validation errors return an `error` disposition with no answer or claim and a sanitized error category.

The run configuration owns deterministic fault switches. The vector-timeout switch executes vector retrieval in a worker and exercises a real future timeout at the configured local deadline. The workflow records `vector_retrieval_timeout`, continues lexical and graph retrieval, reassesses the same evidence contract, and answers only if that contract still passes. The optional cross-encoder follows the same policy: an unavailable model records `reranker_unavailable`, preserves fused order, and continues.

OpenTelemetry uses an application-owned Python SDK provider and replaceable exporter. The run uses `gen_ai.operation.name=invoke_agent`; model and retrieval nodes use applicable GenAI operation names. Project-specific attributes use `contractgraph.*`. Default spans contain run IDs, hashes, limits, node names, iteration numbers, status, and low-cardinality degradation/security facts. They exclude full questions, prompts, clause text, and answers. Explicit `capture_content` configuration adds synthetic question and answer content only for local inspection.

The implementation follows the [OpenTelemetry Python manual instrumentation guidance](https://opentelemetry.io/docs/languages/python/instrumentation/) and the privacy warnings in the [GenAI semantic attribute registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/).

## Consequences

- Security and failure behavior is visible in domain traces and OTEL spans.
- A component failure cannot silently disappear from an otherwise plausible answer.
- Tests can trigger the same bounded timeout without network calls or API keys.
- Domain traces may retain synthetic evidence locally; production telemetry remains content-free by default.
- Exporter choice, retention, access controls, sampling, and redaction remain production deployment concerns.
