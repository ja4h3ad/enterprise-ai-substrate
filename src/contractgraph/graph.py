"""Typed, bounded graph operations backed by NetworkX."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from contractgraph.models import (
    ContractComparison,
    CorpusArtifact,
    GraphStep,
    ObligationResolution,
    OperativeClauseResolution,
    ProvenanceTrace,
)


class GraphLookupError(LookupError):
    """Raised when a typed graph operation cannot resolve its requested entities."""


class ContractGraph:
    def __init__(self, artifact: CorpusArtifact) -> None:
        self._artifact = artifact
        self._graph = nx.MultiDiGraph()
        self._load_nodes(artifact)
        for triple in artifact.triples:
            self._graph.add_edge(
                triple.subject,
                triple.object,
                key=triple.predicate,
                predicate=triple.predicate,
                source_clause_id=triple.source_clause_id,
                population_method=triple.population_method,
            )

    def _load_nodes(self, artifact: CorpusArtifact) -> None:
        for document in artifact.documents:
            self._graph.add_node(
                document.document_id,
                entity_type="SourceDocument",
                effective_date=document.effective_date,
            )
            self._graph.add_node(document.contract_id, entity_type="Contract")
        for page in artifact.pages:
            self._graph.add_node(page.page_id, entity_type="Page")
        for clause in artifact.clauses:
            self._graph.add_node(clause.clause_id, entity_type="Clause")
        for chunk in artifact.chunks:
            self._graph.add_node(chunk.chunk_id, entity_type="Chunk")
        for entity in artifact.entities:
            self._graph.add_node(
                entity.entity_id,
                entity_type=entity.entity_type,
                name=entity.name,
            )

    def resolve_operative_clause(
        self,
        *,
        contract_id: str,
        base_clause_id: str,
        max_depth: int = 3,
        max_candidates: int = 20,
    ) -> OperativeClauseResolution:
        if contract_id not in self._graph:
            raise GraphLookupError(f"Unknown contract: {contract_id}")
        if base_clause_id not in self._graph:
            raise GraphLookupError(f"Unknown clause: {base_clause_id}")
        if not 1 <= max_depth <= 3:
            raise ValueError("max_depth must be between 1 and 3")
        if not 1 <= max_candidates <= 20:
            raise ValueError("max_candidates must be between 1 and 20")
        if max_depth < 3:
            raise GraphLookupError("Amendment resolution requires graph depth 3")

        resolutions: list[tuple[str, tuple[GraphStep, ...]]] = []
        for amendment_id, amends_data in self._incoming(contract_id, "AMENDS"):
            if len(resolutions) >= max_candidates:
                break
            for amended_clause_id, contains_data in self._outgoing(amendment_id, "CONTAINS"):
                supersedes_data = self._edge(amended_clause_id, base_clause_id, "SUPERSEDES")
                if supersedes_data is None:
                    continue
                resolutions.append(
                    (
                        amended_clause_id,
                        (
                            GraphStep(
                                from_id=contract_id,
                                predicate="AMENDS",
                                to_id=amendment_id,
                                traversal="reverse",
                                source_clause_id=amends_data["source_clause_id"],
                            ),
                            GraphStep(
                                from_id=amendment_id,
                                predicate="CONTAINS",
                                to_id=amended_clause_id,
                                traversal="forward",
                                source_clause_id=contains_data["source_clause_id"],
                            ),
                            GraphStep(
                                from_id=amended_clause_id,
                                predicate="SUPERSEDES",
                                to_id=base_clause_id,
                                traversal="forward",
                                source_clause_id=supersedes_data["source_clause_id"],
                            ),
                        ),
                    )
                )
                if len(resolutions) >= max_candidates:
                    break

        if not resolutions:
            return OperativeClauseResolution(
                contract_id=contract_id,
                base_clause_id=base_clause_id,
                operative_clause_id=base_clause_id,
                path=(),
            )
        operative_clause_id, path = sorted(resolutions, key=lambda item: item[0])[-1]
        return OperativeClauseResolution(
            contract_id=contract_id,
            base_clause_id=base_clause_id,
            operative_clause_id=operative_clause_id,
            path=path,
        )

    def obligations_for_party(
        self, party_id: str, *, event_id: str | None = None, limit: int = 20
    ) -> tuple[ObligationResolution, ...]:
        """Return reviewed clause→obligation→party paths, optionally constrained by trigger."""
        if party_id not in self._graph:
            raise GraphLookupError(f"Unknown party: {party_id}")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        resolutions: list[ObligationResolution] = []
        for obligation_id, owed_by in self._incoming(party_id, "OWED_BY"):
            trigger = next(self._outgoing(obligation_id, "TRIGGERED_BY"), None)
            if event_id is not None and (trigger is None or trigger[0] != event_id):
                continue
            source = next(self._incoming(obligation_id, "CREATES_OBLIGATION"), None)
            if source is None:
                continue
            owed_to = next(self._outgoing(obligation_id, "OWED_TO"), None)
            policies = tuple(
                target for target, _ in self._outgoing(source[0], "REFERENCES")
            )
            steps = [
                self._step(source[0], "CREATES_OBLIGATION", obligation_id, source[1]),
                self._step(obligation_id, "OWED_BY", party_id, owed_by),
            ]
            if trigger:
                steps.append(self._step(obligation_id, "TRIGGERED_BY", trigger[0], trigger[1]))
            resolutions.append(
                ObligationResolution(
                    obligation_id=obligation_id,
                    clause_id=source[0],
                    owed_by=party_id,
                    owed_to=owed_to[0] if owed_to else None,
                    triggered_by=trigger[0] if trigger else None,
                    policy_ids=policies,
                    path=tuple(steps),
                )
            )
        return tuple(sorted(resolutions, key=lambda item: item.obligation_id)[:limit])

    def compare_contracts(
        self, contract_ids: tuple[str, ...], *, title_contains: str, limit: int = 20
    ) -> ContractComparison:
        if not 1 <= len(contract_ids) <= 6 or not 1 <= limit <= 20:
            raise ValueError("comparison limits exceeded")
        documents = {
            document.document_id: document
            for document in self._artifact.documents
            if document.document_type == "Contract"
        }
        clauses = []
        for clause in self._artifact.clauses:
            document = documents.get(clause.document_id)
            if (
                document
                and document.contract_id in contract_ids
                and title_contains.casefold() in clause.title.casefold()
            ):
                clauses.append(clause.clause_id)
        services = {
            target
            for contract_id in contract_ids
            for target, _ in self._outgoing(contract_id, "COVERS")
        }
        return ContractComparison(
            contract_ids=contract_ids,
            clause_ids=tuple(sorted(clauses)[:limit]),
            shared_service_type="ProductOrService" if services else None,
        )

    def trace_provenance(self, clause_id: str) -> ProvenanceTrace:
        clause = next(
            (item for item in self._artifact.clauses if item.clause_id == clause_id),
            None,
        )
        if clause is None:
            raise GraphLookupError(f"Unknown clause: {clause_id}")
        chunk = next(item for item in self._artifact.chunks if item.clause_id == clause_id)
        document = next(
            item
            for item in self._artifact.documents
            if item.document_id == clause.document_id
        )
        return ProvenanceTrace(
            clause_id=clause_id,
            document_id=document.document_id,
            page_id=clause.page_id,
            chunk_id=chunk.chunk_id,
            path=(
                GraphStep(clause_id, "LOCATED_ON", clause.page_id, "forward", clause_id),
                GraphStep(clause_id, "EXTRACTED_FROM", chunk.chunk_id, "forward", clause_id),
            ),
        )

    def exhibit_clause_ids(self, contract_id: str, *, limit: int = 20) -> tuple[str, ...]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        clause_ids = [
            clause_id
            for exhibit_id, _ in self._outgoing(contract_id, "HAS_EXHIBIT")
            for clause_id, _ in self._outgoing(exhibit_id, "CONTAINS")
        ]
        return tuple(sorted(clause_ids)[:limit])

    def referenced_entity_ids(self, clause_id: str, *, limit: int = 20) -> tuple[str, ...]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        return tuple(target for target, _ in self._outgoing(clause_id, "REFERENCES"))[:limit]

    def conflicts_for_clause(self, clause_id: str, *, limit: int = 20) -> tuple[GraphStep, ...]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        return tuple(
            self._step(clause_id, "CONFLICTS_WITH", target, data)
            for target, data in self._outgoing(clause_id, "CONFLICTS_WITH")
        )[:limit]

    @staticmethod
    def _step(source: str, predicate: str, target: str, data: dict[str, str]) -> GraphStep:
        return GraphStep(source, predicate, target, "forward", data["source_clause_id"])

    def _incoming(self, node_id: str, predicate: str) -> Iterable[tuple[str, dict[str, str]]]:
        for source, _, _, data in self._graph.in_edges(node_id, keys=True, data=True):
            if data["predicate"] == predicate:
                yield source, data

    def _outgoing(self, node_id: str, predicate: str) -> Iterable[tuple[str, dict[str, str]]]:
        for _, target, _, data in self._graph.out_edges(node_id, keys=True, data=True):
            if data["predicate"] == predicate:
                yield target, data

    def _edge(self, source: str, target: str, predicate: str) -> dict[str, str] | None:
        data = self._graph.get_edge_data(source, target, key=predicate)
        if data and data["predicate"] == predicate:
            return data
        return None
