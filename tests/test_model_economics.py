from __future__ import annotations

import json
from pathlib import Path

import pytest

from contractgraph.application import ContractGraphApplication
from contractgraph.application_models import (
    ContractGraphRunConfig,
    ModelEconomicsConfig,
    RetrievalPlan,
    RunLimits,
)
from contractgraph.comparison import HERO_QUESTION
from contractgraph.ingestion import build_corpus, persist_corpus
from contractgraph.model_economics import (
    ModelRouter,
    PricingCatalog,
    exact_cache_key,
)
from contractgraph.providers import OpenAIModelProvider, ReplayModelProvider

PROJECT_ROOT = Path(__file__).parents[1]
CORPUS_ROOT = PROJECT_ROOT / "corpus"
HERO_FIXTURE = PROJECT_ROOT / "replay" / "hero.json"


def _artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    persist_corpus(build_corpus(CORPUS_ROOT), root)
    return root


def test_exact_cache_key_is_stable_and_every_required_version_dimension_invalidates() -> None:
    base = {
        "model": "gpt-5.6-luna",
        "prompt": "plan from supplied identifiers",
        "prompt_version": "v1",
        "response_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        "policy": "ground in evidence",
        "corpus_digest": "corpus-a",
        "normalized_input": '{"question":"notice"}',
        "selected_evidence_hashes": ("evidence-a",),
    }
    first = exact_cache_key(**base)
    reordered_schema = {
        **base,
        "response_schema": {"properties": {"x": {"type": "string"}}, "type": "object"},
    }
    assert exact_cache_key(**reordered_schema) == first

    for field, changed in {
        "model": "gpt-5.6-sol",
        "prompt": "changed prompt",
        "prompt_version": "v2",
        "response_schema": {"type": "array"},
        "policy": "changed policy",
        "corpus_digest": "corpus-b",
        "normalized_input": '{"question":"different"}',
        "selected_evidence_hashes": ("evidence-b",),
    }.items():
        assert exact_cache_key(**{**base, field: changed}) != first


def test_local_hits_reuse_only_model_nodes_and_still_create_fresh_retrieval_trace(
    tmp_path: Path,
) -> None:
    with ContractGraphApplication(
        artifact_root=_artifact_root(tmp_path),
        state_db=tmp_path / "state.db",
        replay_fixture=HERO_FIXTURE,
    ) as application:
        first = application.run(HERO_QUESTION)
        second = application.run(HERO_QUESTION)

    assert first.run_id != second.run_id
    assert [call.local_cache_status for call in first.provider_calls] == ["miss", "miss"]
    assert [call.local_cache_status for call in second.provider_calls] == ["hit", "hit"]
    assert all(call.estimated_cost_usd == 0 for call in second.provider_calls)
    assert second.status == "answered"
    assert {event.node for event in second.trace_events} >= {
        "lexical_retrieve",
        "vector_retrieve",
        "graph_retrieve",
        "assess_evidence",
        "verify_citations",
    }


def test_routing_and_versioned_pricing_are_deterministic() -> None:
    default = ModelRouter(ModelEconomicsConfig()).route(
        "synthesize", "compare terms across contracts"
    )
    escalated = ModelRouter(
        ModelEconomicsConfig(enable_higher_reasoning=True)
    ).route("synthesize", "compare terms across contracts")
    routine = ModelRouter(
        ModelEconomicsConfig(enable_higher_reasoning=True)
    ).route("analyze_and_plan", "direct clause lookup")

    assert (default.route, default.model, default.reason) == (
        "economical",
        "gpt-5.6-luna",
        "default_economical_route",
    )
    assert (escalated.route, escalated.model, escalated.reason) == (
        "higher_reasoning",
        "gpt-5.6-sol",
        "configured_asymmetric_synthesis",
    )
    assert routine.route == "economical"
    assert PricingCatalog.estimate("gpt-5.6-luna", 1000, 100, 500) == pytest.approx(
        0.00023
    )
    assert PricingCatalog.estimate("unconfigured-model", 1000, 100, 0) is None


def test_replay_and_live_providers_share_typed_contract_and_live_usage_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_plan = json.loads(HERO_FIXTURE.read_text(encoding="utf-8"))["plan"]
    captured: dict[str, bytes] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": json.dumps(fixture_plan)}
                            ]
                        }
                    ],
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 40,
                        "input_tokens_details": {
                            "cached_tokens": 80,
                            "cache_write_tokens": 0,
                        },
                    },
                }
            ).encode()

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["payload"] = request.data  # type: ignore[attr-defined]
        assert timeout == 60
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    economics = ModelEconomicsConfig(local_response_cache=False)
    replay = ReplayModelProvider(HERO_FIXTURE, economics=economics)
    live = OpenAIModelProvider(
        api_key="synthetic-test-key",
        economics=economics,
        corpus_digest="digest",
        cache=None,
    )

    replay_plan, replay_call = replay.analyze_and_plan(HERO_QUESTION)
    live_plan, live_call = live.analyze_and_plan(HERO_QUESTION)
    unpriced_live = OpenAIModelProvider(
        api_key="synthetic-test-key",
        economics=ModelEconomicsConfig(
            economical_model="unconfigured-model",
            pricing_version="unconfigured-pricing-version",
            local_response_cache=False,
        ),
        corpus_digest="digest",
        cache=None,
    )
    _, unpriced_call = unpriced_live.analyze_and_plan(HERO_QUESTION)

    assert isinstance(replay_plan, RetrievalPlan)
    assert live_plan == replay_plan
    assert replay_call.provider == "replay"
    assert live_call.provider == "openai"
    assert live_call.provider_cached_input_tokens == 80
    assert live_call.input_tokens == 120
    assert live_call.output_tokens == 40
    assert live_call.estimated_cost_usd is not None
    assert unpriced_call.estimated_cost_usd is None
    raw_payload = captured["payload"]
    assert raw_payload.index(b'"instructions"') < raw_payload.index(b'"input"')
    assert b'"prompt_cache_key"' in raw_payload
    assert b'"prompt_cache_breakpoint": {"mode": "explicit"}' in raw_payload
    assert b'"prompt_cache_options": {"mode": "explicit"}' in raw_payload


def test_normal_path_uses_two_calls_and_successful_recovery_uses_three(
    tmp_path: Path,
) -> None:
    fixture = json.loads(HERO_FIXTURE.read_text(encoding="utf-8"))
    fixture["plan"]["retrieval_query"] = "unrelated penguin warranty phrase"
    fixture["reformulation"]["retrieval_query"] = (
        "sixty 60 days termination convenience"
    )
    recovery_fixture = tmp_path / "recovery.json"
    recovery_fixture.write_text(json.dumps(fixture), encoding="utf-8")
    artifacts = _artifact_root(tmp_path)

    with ContractGraphApplication(
        artifact_root=artifacts,
        state_db=tmp_path / "normal.db",
        replay_fixture=HERO_FIXTURE,
    ) as application:
        normal = application.run(HERO_QUESTION)
    with ContractGraphApplication(
        artifact_root=artifacts,
        state_db=tmp_path / "recovery.db",
        replay_fixture=recovery_fixture,
    ) as application:
        recovered = application.run(
            HERO_QUESTION,
            ContractGraphRunConfig(
                limits=RunLimits(max_candidates_per_retriever=1)
            ),
        )

    assert normal.model_calls == len(normal.provider_calls) == 2
    assert recovered.status == "answered"
    assert recovered.iterations == 2
    assert recovered.model_calls == len(recovered.provider_calls) == 3
    assert [call.operation for call in recovered.provider_calls] == [
        "analyze_and_plan",
        "reformulate",
        "synthesize",
    ]
    assert recovered.model_calls == recovered.limits.max_model_calls
