# ContractGraph domain context

ContractGraph is a trustworthy retrieval prototype over fictional enterprise IT agreements. The active model is intentionally competency-driven: an entity exists only when ingestion populates it, a bounded traversal or answer consumes it, and an integration test can validate it.

## Active vocabulary

| Layer | Active entities | Purpose |
|---|---|---|
| Contract | Contract, Amendment, Exhibit, Clause, Party, Obligation, Event, Policy, ProductOrService | Represents agreement structure and the operational meaning needed by the golden questions. |
| Provenance | SourceDocument, Page, Chunk | Locates every returned clause in its canonical synthetic source. |
| Execution | AgentRun, ToolCall, Evidence, Claim, EvaluationResult | Records how evidence was retrieved, used, and evaluated. These are application/trace records; they need not all be graph nodes. |

`MissingReference` is a technical sentinel, not a legal-domain class. It records that a clause refers to unavailable evidence without manufacturing the missing schedule.

## Competency map

| Entity or relationship | Competency question | Population path | Consumer | Validation |
|---|---|---|---|---|
| Amendment; AMENDS, MODIFIES, SUPERSEDES | What language is operative after amendments? | Reviewed manifest assertions | `resolve_operative_clause` | Operative clause and three-hop path match the reviewed amendment. |
| Exhibit; HAS_EXHIBIT, CONTAINS | What does the cybersecurity exhibit require? | Canonical document plus reviewed link | Exhibit traversal and evidence selection | Result is an exact exhibit clause with document/page/section provenance. |
| Party, Obligation; CREATES_OBLIGATION, OWED_BY, OWED_TO | What must a named supplier do? | Reviewed clause assertions | `obligations_for_party` | Every obligation has a source clause and obligor; obligee is returned when known. |
| Event; TRIGGERED_BY | What must happen after a security event or severity-one incident? | Reviewed clause assertion | Trigger-constrained obligation traversal | Returned obligation is connected to the requested event within three hops. |
| Policy; REFERENCES | Which policy governs the notification duty? | Reviewed reference assertion | Obligation answer enrichment | Referenced policy ID exists and the source clause is preserved. |
| ProductOrService; COVERS | Compare similar supplier duties across contracts. | Reviewed covered-service assertion | `compare_contracts` | Only requested contracts and bounded matching clauses are returned. |
| CONFLICTS_WITH | Do competing clauses have a deterministic precedence path? | Reviewed assertion | Conflict assessment and bounded analyst review | Superseded conflicts resolve automatically; an unresolved agreement/exhibit pair pauses for analyst abstention or controlling evidence. |
| LOCATED_ON, EXTRACTED_FROM | Where did this evidence come from? | Canonical parser records or reviewed assertion | `trace_provenance`, citations | Clause, page, chunk, and source document all exist. |
| REFERENCES MissingReference | What does missing Schedule Z require? | Reviewed missing-reference assertion | Insufficient-evidence response | The sentinel is returned; no schedule content or answer is invented. |

## Deferred vocabulary

`BusinessUnit`, `Right`, `Risk`, `Jurisdiction`, and `MonetaryCommitment` remain documentation-only candidates. They move into the active model only with a competency question, population rule, traversal/answer consumer, and validation case. In particular, `Right` should be added when the golden set asks who holds a termination right rather than merely asking for operative language or notice obligations.

Memorable rule: **no ontology class without a competency question; no relationship without a traversal or validation use case.**
