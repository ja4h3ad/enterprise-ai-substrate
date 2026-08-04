"""Model-provider seam with a deterministic, keyless replay implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from contractgraph.application_models import (
    GroundingEnvelope,
    ProviderCall,
    RetrievalPlan,
    SynthesisResult,
)


class ModelProvider(Protocol):
    def analyze_and_plan(self, question: str) -> tuple[RetrievalPlan, ProviderCall]: ...

    def reformulate(
        self, question: str, insufficiency_reasons: Sequence[str]
    ) -> tuple[str, ProviderCall]: ...

    def synthesize(
        self, envelope: GroundingEnvelope
    ) -> tuple[SynthesisResult, ProviderCall]: ...


class ReplayModelProvider:
    """Return validated structured model outputs committed for the synthetic hero run."""

    provider_name = "replay"
    model_name = "recorded-structured-output"

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path
        self._fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def analyze_and_plan(self, question: str) -> tuple[RetrievalPlan, ProviderCall]:
        expected_question = self._fixture["question"]
        if question != expected_question:
            raise ValueError("Replay fixture does not match the requested question")
        return RetrievalPlan.model_validate(self._fixture["plan"]), self._call(
            "analyze_and_plan"
        )

    def reformulate(
        self, question: str, insufficiency_reasons: Sequence[str]
    ) -> tuple[str, ProviderCall]:
        if question != self._fixture["question"]:
            raise ValueError("Replay fixture does not match the requested question")
        if not insufficiency_reasons:
            raise ValueError("Recovery requires an explicit insufficiency reason")
        return str(self._fixture["reformulation"]["retrieval_query"]), self._call(
            "reformulate"
        )

    def synthesize(
        self, envelope: GroundingEnvelope
    ) -> tuple[SynthesisResult, ProviderCall]:
        if envelope.question != self._fixture["question"]:
            raise ValueError("Replay fixture does not match the requested question")
        # Provider outputs are untrusted even in replay mode. The deterministic
        # citation-verification node, not this adapter, owns support validation.
        # The fixed policy is structurally separate from document content, and every
        # evidence item is explicitly tagged as untrusted evidence.
        if any(item.trust_zone != "untrusted_evidence" for item in envelope.evidence):
            raise ValueError("Grounding envelope contains an invalid evidence trust zone")
        result = SynthesisResult.model_validate(self._fixture["synthesis"])
        return result, self._call("synthesize")

    def _call(self, operation: str) -> ProviderCall:
        return ProviderCall(
            operation=operation,
            provider=self.provider_name,
            model=self.model_name,
            replay_key=f"{self._fixture_path.name}:{operation}",
        )
