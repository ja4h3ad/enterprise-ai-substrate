"""Public local hybrid retrieval seam with inspectable fusion and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from contractgraph.application_models import FusedCandidate
from contractgraph.local_models import load_cross_encoder, load_embedder
from contractgraph.models import Clause, SearchResult
from contractgraph.reranking import LocalCrossEncoderReranker, RerankOutcome
from contractgraph.retrieval import BM25Retriever, SemanticVectorRetriever


@dataclass(frozen=True, slots=True)
class RetrievalStackOutcome:
    lexical: tuple[SearchResult, ...]
    vector: tuple[SearchResult, ...]
    graph: tuple[SearchResult, ...]
    fused: tuple[FusedCandidate, ...]
    reranked: RerankOutcome
    degraded_components: tuple[str, ...]


class LocalRetrievalStack:
    def __init__(self, clauses: Sequence[Clause], *, candidate_limit: int = 8) -> None:
        if not 1 <= candidate_limit <= 20:
            raise ValueError("candidate_limit must be between 1 and 20")
        self._clauses = {clause.clause_id: clause for clause in clauses}
        self._lexical = BM25Retriever(clauses)
        embedder, self._embedding_degradation = load_embedder()
        self._vector = SemanticVectorRetriever(clauses, embedder)
        scorer, self._reranker_degradation = load_cross_encoder()
        self._reranker = LocalCrossEncoderReranker(scorer, top_n=candidate_limit)
        self._limit = candidate_limit

    def search(
        self, query: str, *, graph_results: Sequence[SearchResult] = ()
    ) -> RetrievalStackOutcome:
        lexical = self._lexical.search(query, limit=self._limit)
        vector = self._vector.search(query, limit=self._limit)
        graph = tuple(graph_results[: self._limit])
        fused = reciprocal_rank_fusion(lexical, vector, graph, limit=self._limit)
        fused_results = tuple(self._as_search_result(candidate) for candidate in fused)
        reranked = self._reranker.rerank(query, fused_results)
        degraded = tuple(
            reason
            for reason in (self._embedding_degradation, self._reranker_degradation)
            if reason is not None
        )
        return RetrievalStackOutcome(lexical, vector, graph, fused, reranked, degraded)

    def _as_search_result(self, candidate: FusedCandidate) -> SearchResult:
        clause = self._clauses[candidate.clause_id]
        return SearchResult(
            clause.clause_id,
            clause.document_id,
            clause.page_number,
            clause.section,
            clause.title,
            clause.text,
            candidate.score,
            candidate.rank,
            "rrf",
        )


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[SearchResult], limit: int = 20, k: int = 60
) -> tuple[FusedCandidate, ...]:
    """Fuse incomparable scores by rank; graph is an explicit third contribution."""
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for results in ranked_lists:
        for result in results:
            scores[result.clause_id] = scores.get(result.clause_id, 0.0) + 1.0 / (k + result.rank)
            sources.setdefault(result.clause_id, set()).add(result.retriever)
    ordered = sorted(scores, key=lambda clause_id: (-scores[clause_id], clause_id))[:limit]
    return tuple(
        FusedCandidate(
            clause_id=clause_id,
            score=scores[clause_id],
            rank=rank,
            sources=tuple(sorted(sources[clause_id])),
        )
        for rank, clause_id in enumerate(ordered, 1)
    )
