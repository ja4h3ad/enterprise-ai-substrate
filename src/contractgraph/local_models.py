"""Pinned local model adapters with explicit offline degradation."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Protocol, Sequence

from contractgraph.retrieval import tokenize

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "826711e54e001c83835913827a843d8dd0a1def9"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "c5ee24c"
MODEL_LICENSE = "Apache-2.0"


def model_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "contractgraph" / "models"


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class DeterministicSemanticEmbedder:
    """Small offline fallback that normalizes only corpus competency vocabulary."""

    name = "deterministic-competency-semantic-v1"
    _concepts = (
        ("notify", "alert", "inform", "advise", "report", "escalation"),
        ("security", "cyber", "attacker", "unauthorized", "compromise", "vulnerability"),
        ("incident", "event", "access"),
        ("terminate", "termination", "end"),
        ("support", "service", "assist"),
        ("deadline", "hours", "days", "within"),
        ("availability", "uptime", "unavailable"),
    )

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vocabulary = tuple(sorted({token for text in texts for token in self._normalize(text)}))
        vectors = []
        for text in texts:
            counts = Counter(self._normalize(text))
            vectors.append(tuple(float(counts[token]) for token in vocabulary))
        return tuple(vectors)

    def _normalize(self, text: str) -> tuple[str, ...]:
        aliases = {alias: group[0] for group in self._concepts for alias in group}
        return tuple(aliases.get(token, token) for token in tokenize(text))


class SentenceTransformerEmbedder:
    name = f"{EMBEDDING_MODEL_ID}@{EMBEDDING_MODEL_REVISION}"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            revision=EMBEDDING_MODEL_REVISION,
            cache_folder=str(model_cache_dir()),
            local_files_only=True,
        )

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return tuple(tuple(float(value) for value in vector) for vector in vectors)


class CrossEncoderScorer:
    name = f"{RERANKER_MODEL_ID}@{RERANKER_MODEL_REVISION}"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            RERANKER_MODEL_ID,
            revision=RERANKER_MODEL_REVISION,
            cache_folder=str(model_cache_dir()),
            local_files_only=True,
        )

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        scores = self._model.predict([(query, passage) for passage in passages])
        return tuple(float(score) for score in scores)


def load_embedder() -> tuple[Embedder, str | None]:
    try:
        return SentenceTransformerEmbedder(), None
    except (ImportError, OSError):
        return DeterministicSemanticEmbedder(), "embedding_model_not_cached"


def load_cross_encoder() -> tuple[CrossEncoderScorer | None, str | None]:
    try:
        return CrossEncoderScorer(), None
    except (ImportError, OSError):
        return None, "cross_encoder_not_cached"
