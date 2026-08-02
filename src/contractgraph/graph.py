"""Typed, bounded graph operations backed by NetworkX."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from contractgraph.models import CorpusArtifact, GraphStep, OperativeClauseResolution


class GraphLookupError(LookupError):
    """Raised when a typed graph operation cannot resolve its requested entities."""


class ContractGraph:
    def __init__(self, artifact: CorpusArtifact) -> None:
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
