# ADR 0001: pinned local retrieval models with offline fallback

Status: Accepted

## Decision

The production-shaped local path is configured for these immutable model revisions:

| Use | Model | Revision | Declared license |
|---|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | `826711e54e001c83835913827a843d8dd0a1def9` | Apache-2.0 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | `c5ee24c` | Apache-2.0 |

Models are loaded with `local_files_only=True` from `${XDG_CACHE_HOME:-~/.cache}/contractgraph/models`; weights are never committed. `sentence-transformers` is deliberately optional for the interview clone. When weights or the package are unavailable, semantic retrieval uses a deterministic competency-vocabulary embedder and reranking preserves the RRF order. Both degradations are returned as structured reasons.

RRF is deterministic rank mathematics, not an LLM operation. It combines lexical, embedding, and graph result positions because their raw score scales are not comparable. The cross-encoder receives only the configured top eight fused candidates and records every position change.

## Consequences

- Ingestion, replay, and integration tests require no OpenAI key or model download.
- A prepared demo machine can pre-populate the external cache and exercise the pinned neural models.
- Offline fallback is intentionally less capable, visible in traces, and separately evaluable.
- Model-card licensing is documented, but production adoption still requires organizational review of model training data and intended use.
