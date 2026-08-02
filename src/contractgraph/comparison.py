"""The first end-to-end retrieval contrast for the interview hero question."""

from __future__ import annotations

from dataclasses import dataclass

from contractgraph.graph import ContractGraph
from contractgraph.models import (
    Clause,
    CorpusArtifact,
    OperativeClauseResolution,
    SearchResult,
)
from contractgraph.retrieval import BM25Retriever, ExactVectorRetriever

HERO_QUESTION = (
    "How much notice must Atlas Network Services provide before terminating for convenience?"
)
HERO_RETRIEVAL_QUERY = "termination for convenience notice"


@dataclass(frozen=True, slots=True)
class HeroComparison:
    question: str
    retrieval_query: str
    lexical_results: tuple[SearchResult, ...]
    vector_results: tuple[SearchResult, ...]
    graph_resolution: OperativeClauseResolution
    operative_clause: Clause


class HeroComparisonService:
    def __init__(self, artifact: CorpusArtifact) -> None:
        self._artifact = artifact
        self._clauses = {clause.clause_id: clause for clause in artifact.clauses}
        self._documents = {
            document.document_id: document for document in artifact.documents
        }
        self._lexical = BM25Retriever(artifact.clauses)
        self._vector = ExactVectorRetriever(artifact.clauses)
        self._graph = ContractGraph(artifact)

    def compare(
        self,
        question: str = HERO_QUESTION,
        *,
        retrieval_query: str = HERO_RETRIEVAL_QUERY,
    ) -> HeroComparison:
        lexical_results = self._lexical.search(retrieval_query, limit=5)
        vector_results = self._vector.search(retrieval_query, limit=5)
        if not vector_results:
            raise LookupError("Vector retrieval returned no candidates")
        base_result = vector_results[0]
        source_document = self._documents[base_result.document_id]
        resolution = self._graph.resolve_operative_clause(
            contract_id=source_document.contract_id,
            base_clause_id=base_result.clause_id,
            max_depth=3,
            max_candidates=20,
        )
        return HeroComparison(
            question=question,
            retrieval_query=retrieval_query,
            lexical_results=lexical_results,
            vector_results=vector_results,
            graph_resolution=resolution,
            operative_clause=self._clauses[resolution.operative_clause_id],
        )


def render_comparison(comparison: HeroComparison) -> str:
    vector = comparison.vector_results[0]
    operative = comparison.operative_clause
    lines = [
        "ContractGraph hero comparison",
        f"Question: {comparison.question}",
        f"Deterministic retrieval query: {comparison.retrieval_query}",
        "",
        "Lexical retrieval (top 3)",
    ]
    lines.extend(_result_line(result) for result in comparison.lexical_results[:3])
    lines.extend(
        (
            "",
            "Vector-only baseline",
            _result_line(vector),
            f"Selected language: {vector.text}",
            (
                "Outcome: INCORRECT — selected clause is superseded by an amendment."
                if comparison.graph_resolution.path
                else "Outcome: NO AMENDMENT PATH FOUND"
            ),
            "",
            "Graph-grounded resolution",
            f"Operative clause: {operative.clause_id}",
            f"Citation: {operative.document_id}, p.{operative.page_number}, "
            f"§{operative.section}, {operative.clause_id}",
            f"Selected language: {operative.text}",
            (
                "Outcome: CORRECT — graph traversal selected the operative "
                "amended language."
                if comparison.graph_resolution.path
                else "Outcome: UNCHANGED — the retrieved clause remains operative."
            ),
            "Graph path:",
        )
    )
    for step in comparison.graph_resolution.path:
        arrow = "<--" if step.traversal == "reverse" else "--"
        lines.append(
            f"- {step.from_id} {arrow}{step.predicate}-- {step.to_id} "
            f"[source={step.source_clause_id}]"
        )
    return "\n".join(lines) + "\n"


def _result_line(result: SearchResult) -> str:
    return (
        f"{result.rank}. {result.clause_id} score={result.score:.6f} "
        f"[{result.citation}]"
    )
