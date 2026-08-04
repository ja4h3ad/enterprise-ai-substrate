"""Deterministic model routing, exact response caching, and cost accounting."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from contractgraph.application_models import ModelEconomicsConfig


@dataclass(frozen=True)
class ModelRoute:
    route: Literal["economical", "higher_reasoning"]
    model: str
    reason: str
    reasoning_effort: str


class ModelRouter:
    def __init__(self, config: ModelEconomicsConfig) -> None:
        self._config = config

    def route(self, operation: str, normalized_input: str) -> ModelRoute:
        asymmetric_markers = (
            "compare",
            "conflict",
            "which deadline controls",
            "across contracts",
        )
        qualifies = operation == "synthesize" and any(
            marker in normalized_input for marker in asymmetric_markers
        )
        if self._config.enable_higher_reasoning and qualifies:
            return ModelRoute(
                "higher_reasoning",
                self._config.higher_reasoning_model,
                "configured_asymmetric_synthesis",
                "high",
            )
        return ModelRoute(
            "economical",
            self._config.economical_model,
            "default_economical_route",
            "low",
        )


class ExactResponseCache:
    """Exact-key cache only; no similarity lookup or semantic index exists."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_response_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT response_json FROM model_response_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, cache_key: str, response: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO model_response_cache VALUES (?, ?, CURRENT_TIMESTAMP)",
            (cache_key, _canonical_json(response)),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


def exact_cache_key(
    *,
    model: str,
    prompt: str,
    prompt_version: str,
    response_schema: dict[str, Any],
    policy: str,
    corpus_digest: str,
    normalized_input: str,
    selected_evidence_hashes: tuple[str, ...],
) -> str:
    material = {
        "model": model,
        "prompt": prompt,
        "prompt_version": prompt_version,
        "response_schema": response_schema,
        "policy": policy,
        "corpus_digest": corpus_digest,
        "normalized_input": normalized_input,
        "selected_evidence_hashes": selected_evidence_hashes,
    }
    return hashlib.sha256(_canonical_json(material).encode()).hexdigest()


def normalized_json(value: object) -> str:
    return _canonical_json(value).casefold()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Price:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


class PricingCatalog:
    """Versioned snapshot; unknown models deliberately produce no estimate."""

    VERSION = "openai-2026-08-04"
    PRICES = {
        "gpt-5.6-luna": Price(0.20, 0.02, 1.20),
        "gpt-5.6-sol": Price(5.00, 0.50, 30.00),
    }

    @classmethod
    def estimate(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        cache_write_tokens: int = 0,
    ) -> float | None:
        price = cls.PRICES.get(model)
        if price is None:
            return None
        uncached = max(0, input_tokens - cached_input_tokens - cache_write_tokens)
        return round(
            (
                uncached * price.input_per_million
                + cached_input_tokens * price.cached_input_per_million
                + cache_write_tokens * price.input_per_million * 1.25
                + output_tokens * price.output_per_million
            )
            / 1_000_000,
            8,
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
