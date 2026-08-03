"""Deterministic lexical and exact-vector retrieval for contract clauses."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence

from contractgraph.models import Clause, SearchResult

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.casefold()))


class ClauseRetriever(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 20) -> tuple[SearchResult, ...]: ...


class BM25Retriever:
    """A compact BM25 implementation over clause titles and text."""

    name = "lexical-bm25"

    def __init__(self, clauses: Sequence[Clause], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._clauses = tuple(clauses)
        self._k1 = k1
        self._b = b
        self._documents = tuple(
            tokenize(f"{clause.title} {clause.text}") for clause in self._clauses
        )
        self._average_length = (
            sum(len(document) for document in self._documents) / len(self._documents)
            if self._documents
            else 0.0
        )
        self._document_frequency = Counter(
            token for document in self._documents for token in set(document)
        )

    def search(self, query: str, *, limit: int = 20) -> tuple[SearchResult, ...]:
        _validate_limit(limit)
        query_tokens = tokenize(query)
        scored = [
            (clause, self._score(document, query_tokens))
            for clause, document in zip(self._clauses, self._documents, strict=True)
        ]
        return _ranked_results(scored, retriever=self.name, limit=limit)

    def _score(self, document: tuple[str, ...], query: tuple[str, ...]) -> float:
        if not document or not query or not self._average_length:
            return 0.0
        frequencies = Counter(document)
        document_count = len(self._documents)
        score = 0.0
        # Stable token order prevents process-randomized set iteration from changing
        # floating-point accumulation and tie ordering across evaluation runs.
        for token in sorted(set(query)):
            frequency = frequencies[token]
            if not frequency:
                continue
            containing_documents = self._document_frequency[token]
            inverse_document_frequency = math.log(
                1 + (document_count - containing_documents + 0.5) / (containing_documents + 0.5)
            )
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * len(document) / self._average_length
            )
            score += inverse_document_frequency * frequency * (self._k1 + 1) / denominator
        return score


class ExactVectorRetriever:
    """Exact cosine search using a deterministic local token-frequency embedding.

    The embedding is intentionally replaceable. Ticket #5 swaps this tracer-bullet
    baseline for a pinned Sentence Transformers model without changing callers.
    """

    name = "vector-exact-token-frequency-v1"

    def __init__(self, clauses: Sequence[Clause]) -> None:
        self._clauses = tuple(clauses)
        self._vocabulary = tuple(
            sorted(
                {
                    token
                    for clause in self._clauses
                    for token in tokenize(f"{clause.title} {clause.text}")
                }
            )
        )
        self._vectors = tuple(
            self._embed(f"{clause.title} {clause.text}") for clause in self._clauses
        )

    def search(self, query: str, *, limit: int = 20) -> tuple[SearchResult, ...]:
        _validate_limit(limit)
        query_vector = self._embed(query)
        scored = [
            (clause, _cosine(query_vector, vector))
            for clause, vector in zip(self._clauses, self._vectors, strict=True)
        ]
        return _ranked_results(scored, retriever=self.name, limit=limit)

    def _embed(self, text: str) -> tuple[float, ...]:
        frequencies = Counter(tokenize(text))
        total = sum(frequencies.values())
        if not total:
            return tuple(0.0 for _ in self._vocabulary)
        return tuple(frequencies[token] / total for token in self._vocabulary)


class SemanticVectorRetriever:
    """Exact local cosine search over embeddings supplied by a pinned-model adapter."""

    def __init__(self, clauses: Sequence[Clause], embedder: object) -> None:
        self._clauses = tuple(clauses)
        self._embedder = embedder
        self.name = f"vector:{getattr(embedder, 'name', 'unknown')}"

    def search(self, query: str, *, limit: int = 20) -> tuple[SearchResult, ...]:
        _validate_limit(limit)
        texts = [query, *(f"{clause.title} {clause.text}" for clause in self._clauses)]
        vectors = self._embedder.encode(texts)
        query_vector, document_vectors = vectors[0], vectors[1:]
        scored = [
            (clause, _cosine(query_vector, vector))
            for clause, vector in zip(self._clauses, document_vectors, strict=True)
        ]
        return _ranked_results(scored, retriever=self.name, limit=limit)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _validate_limit(limit: int) -> None:
    if not 1 <= limit <= 20:
        raise ValueError("retrieval limit must be between 1 and 20")


def _ranked_results(
    scored: Sequence[tuple[Clause, float]], *, retriever: str, limit: int
) -> tuple[SearchResult, ...]:
    ordered = sorted(scored, key=lambda item: (-item[1], item[0].clause_id))[:limit]
    return tuple(
        SearchResult(
            clause_id=clause.clause_id,
            document_id=clause.document_id,
            page_number=clause.page_number,
            section=clause.section,
            title=clause.title,
            text=clause.text,
            score=score,
            rank=rank,
            retriever=retriever,
        )
        for rank, (clause, score) in enumerate(ordered, start=1)
    )
