# ADR 0003: Model economics and exact response caching

## Status

Accepted for the interview POC on 2026-08-04.

## Decision

Replay and live OpenAI adapters implement the same typed planning, reformulation, and synthesis protocol. Replay remains the default and all tests and published evaluations are keyless. Explicit live mode uses the Responses API and requires `OPENAI_API_KEY`.

Routine nodes route to `gpt-5.6-luna` with low reasoning effort. When explicitly enabled, only qualifying asymmetric synthesis routes to `gpt-5.6-sol` with high reasoning effort, and the deterministic reason is recorded. These roles follow the current OpenAI model guidance: Luna is the cost-sensitive high-volume model and Sol is the complex professional-work model.

Provider prompts place the structured-output schema, policy, prompt version, and operation in a stable prefix before the dynamic question or evidence. GPT-5.6 requests use an explicit prompt-cache breakpoint, cache key, and explicit cache mode. Provider-reported `cached_tokens` and `cache_write_tokens` remain distinct from the local cache status.

The local SQLite response cache performs exact key lookup only. Its SHA-256 identity includes model, exact node prompt, prompt version, complete response schema, policy, corpus digest, normalized structured input, and ordered selected-evidence hashes. There is no embedding, nearest-neighbor, similarity threshold, or semantic cache path. Cache lookup occurs only inside model nodes, so each run still performs retrieval, evidence assessment, citation verification, checkpointing, and trace creation.

The pricing catalog is versioned `openai-2026-08-04`. It includes only configured Luna and Sol text-token rates, including cached reads and the 1.25x cache-write rate. Unknown models or pricing versions return `unknown` rather than extrapolating.

## Sources

- OpenAI model catalog: https://developers.openai.com/api/docs/models
- GPT-5.6 Luna: https://developers.openai.com/api/docs/models/gpt-5.6-luna
- GPT-5.6 Sol: https://developers.openai.com/api/docs/models/gpt-5.6-sol
- Prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Structured outputs: https://developers.openai.com/api/docs/guides/structured-outputs

## Consequences

- Local cache hits reduce model-node cost without weakening evidence freshness.
- Replay token counts are deterministic transparent estimates and replay cost is zero; live usage comes from the provider response.
- The catalog must be consciously revised when models or prices change.
- The live adapter is intentionally narrow and does not introduce provider fallback, retries, streaming, or a general model gateway.
