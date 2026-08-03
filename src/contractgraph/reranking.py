"""Bounded, inspectable cross-encoder reranking with a visible fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from contractgraph.models import SearchResult


class PassageScorer(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class PositionChange:
    clause_id: str
    before: int
    after: int
    reranker_score: float | None


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    results: tuple[SearchResult, ...]
    changes: tuple[PositionChange, ...]
    model: str
    degraded: bool
    degradation_reason: str | None


class LocalCrossEncoderReranker:
    def __init__(self, scorer: PassageScorer | None, *, top_n: int = 8) -> None:
        if not 1 <= top_n <= 20:
            raise ValueError("top_n must be between 1 and 20")
        self._scorer = scorer
        self._top_n = top_n

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> RerankOutcome:
        bounded = tuple(candidates[: self._top_n])
        if self._scorer is None:
            return RerankOutcome(
                results=tuple(candidates),
                changes=tuple(
                    PositionChange(item.clause_id, item.rank, item.rank, None)
                    for item in bounded
                ),
                model="unavailable",
                degraded=True,
                degradation_reason="cross_encoder_not_cached",
            )
        try:
            scores = self._scorer.score(query, [item.text for item in bounded])
        except Exception as error:
            return RerankOutcome(
                results=tuple(candidates),
                changes=tuple(
                    PositionChange(item.clause_id, item.rank, item.rank, None)
                    for item in bounded
                ),
                model=self._scorer.name,
                degraded=True,
                degradation_reason=f"cross_encoder_error:{type(error).__name__}",
            )
        ordered = sorted(
            zip(bounded, scores, strict=True),
            key=lambda item: (-item[1], item[0].clause_id),
        )
        reranked = tuple(
            SearchResult(
                item.clause_id,
                item.document_id,
                item.page_number,
                item.section,
                item.title,
                item.text,
                score,
                rank,
                f"reranker:{self._scorer.name}",
            )
            for rank, (item, score) in enumerate(ordered, 1)
        )
        positions = {item.clause_id: item.rank for item in reranked}
        score_by_id = {item.clause_id: score for item, score in ordered}
        return RerankOutcome(
            results=(*reranked, *tuple(candidates[self._top_n :])),
            changes=tuple(
                PositionChange(
                    item.clause_id,
                    item.rank,
                    positions[item.clause_id],
                    score_by_id[item.clause_id],
                )
                for item in bounded
            ),
            model=self._scorer.name,
            degraded=False,
            degradation_reason=None,
        )
