"""Application-level seam for the competency questions exercised by Issue 5."""

from __future__ import annotations

from dataclasses import dataclass

from contractgraph.graph import ContractGraph
from contractgraph.models import (
    ContractComparison,
    CorpusArtifact,
    GraphStep,
    ObligationResolution,
    OperativeClauseResolution,
    ProvenanceTrace,
    SearchResult,
)
from contractgraph.retrieval_stack import LocalRetrievalStack, RetrievalStackOutcome


@dataclass(frozen=True, slots=True)
class MissingEvidenceResult:
    clause_id: str
    missing_reference_ids: tuple[str, ...]
    answerable: bool = False
    reason: str = "referenced_evidence_not_in_corpus"


class ContractIntelligenceService:
    def __init__(self, artifact: CorpusArtifact) -> None:
        self._graph = ContractGraph(artifact)
        self._retrieval = LocalRetrievalStack(artifact.clauses)

    def retrieve(
        self, query: str, *, graph_results: tuple[SearchResult, ...] = ()
    ) -> RetrievalStackOutcome:
        return self._retrieval.search(query, graph_results=graph_results)

    def exhibit_clauses(self, contract_id: str) -> tuple[str, ...]:
        return self._graph.exhibit_clause_ids(contract_id)

    def triggered_obligations(
        self, party_id: str, event_id: str
    ) -> tuple[ObligationResolution, ...]:
        return self._graph.obligations_for_party(party_id, event_id=event_id)

    def compare(self, contract_ids: tuple[str, ...], title_contains: str) -> ContractComparison:
        return self._graph.compare_contracts(contract_ids, title_contains=title_contains)

    def compare_triggered_obligations(
        self, party_ids: tuple[str, ...], event_id: str
    ) -> tuple[ObligationResolution, ...]:
        if not 1 <= len(party_ids) <= 6:
            raise ValueError("party comparison must contain between one and six parties")
        return tuple(
            obligation
            for party_id in party_ids
            for obligation in self._graph.obligations_for_party(party_id, event_id=event_id)
        )

    def operative_clause(
        self, contract_id: str, base_clause_id: str
    ) -> OperativeClauseResolution:
        return self._graph.resolve_operative_clause(
            contract_id=contract_id, base_clause_id=base_clause_id
        )

    def conflicts(self, clause_id: str) -> tuple[GraphStep, ...]:
        return self._graph.conflicts_for_clause(clause_id)

    def missing_evidence(self, clause_id: str) -> MissingEvidenceResult:
        references = tuple(
            entity_id
            for entity_id in self._graph.referenced_entity_ids(clause_id)
            if entity_id.startswith("MISSING-")
        )
        return MissingEvidenceResult(clause_id, references)

    def provenance(self, clause_id: str) -> ProvenanceTrace:
        return self._graph.trace_provenance(clause_id)
