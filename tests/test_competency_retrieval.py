from __future__ import annotations

from pathlib import Path

from contractgraph.competencies import ContractIntelligenceService
from contractgraph.ingestion import build_corpus
from contractgraph.local_models import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
)
from contractgraph.models import SearchResult

CORPUS_ROOT = Path(__file__).parents[1] / "corpus"


def _service() -> ContractIntelligenceService:
    return ContractIntelligenceService(build_corpus(CORPUS_ROOT))


def test_semantic_equivalence_and_offline_model_fallback_are_inspectable() -> None:
    outcome = _service().retrieve(
        "Which vendor has to alert us after an attacker gets into our information?"
    )

    assert any("CYBER" in item.clause_id or "4.1" in item.clause_id for item in outcome.vector[:5])
    assert outcome.fused
    assert all(candidate.sources for candidate in outcome.fused)
    assert len(outcome.reranked.changes) == 8
    assert outcome.reranked.degraded is True
    assert "cross_encoder_not_cached" in outcome.degraded_components
    assert EMBEDDING_MODEL_ID and EMBEDDING_MODEL_REVISION
    assert RERANKER_MODEL_ID and RERANKER_MODEL_REVISION


def test_graph_candidates_contribute_explicitly_to_rrf() -> None:
    graph_result = SearchResult(
        "CLAUSE-ATLAS-EXHIBIT-C-C.6",
        "DOC-ATLAS-EXHIBIT-C",
        2,
        "C.6",
        "Notification Obligation",
        "Supplier must notify Customer within twenty-four hours.",
        1.0,
        1,
        "graph:HAS_EXHIBIT/CONTAINS",
    )
    outcome = _service().retrieve("security event notification", graph_results=(graph_result,))
    candidate = next(item for item in outcome.fused if item.clause_id == graph_result.clause_id)
    assert "graph:HAS_EXHIBIT/CONTAINS" in candidate.sources
    assert candidate.score > 1 / 61


def test_exhibit_traversal_returns_exact_cyber_notification_clause() -> None:
    clauses = _service().exhibit_clauses("CONTRACT-ATLAS-001")
    assert clauses == ("CLAUSE-ATLAS-EXHIBIT-C-C.6",)
    provenance = _service().provenance(clauses[0])
    assert provenance.document_id == "DOC-ATLAS-EXHIBIT-C"
    assert provenance.page_id.endswith("PAGE:002")


def test_multi_hop_obligation_connects_supplier_trigger_and_policy() -> None:
    obligations = _service().triggered_obligations("PARTY-ATLAS", "EVENT-SECURITY")
    assert len(obligations) == 1
    obligation = obligations[0]
    assert obligation.clause_id == "CLAUSE-ATLAS-EXHIBIT-C-C.6"
    assert obligation.owed_to == "PARTY-NORTHSTAR"
    assert obligation.policy_ids == ("POLICY-ORION-SECURITY",)
    assert [step.predicate for step in obligation.path] == [
        "CREATES_OBLIGATION",
        "OWED_BY",
        "TRIGGERED_BY",
    ]


def test_cross_contract_comparison_selects_similar_incident_terms() -> None:
    obligations = _service().compare_triggered_obligations(
        ("PARTY-BOREALIS", "PARTY-CEDAR"), "EVENT-SECURITY"
    )
    assert tuple(item.clause_id for item in obligations) == (
        "CLAUSE-BOREALIS-AMENDMENT-001-2",
        "CLAUSE-CEDAR-ESA-2025-4.1",
    )
    assert {item.owed_by for item in obligations} == {"PARTY-BOREALIS", "PARTY-CEDAR"}


def test_supersession_resolves_borealis_incident_deadline() -> None:
    resolution = _service().operative_clause(
        "CONTRACT-BOREALIS-001", "CLAUSE-BOREALIS-CHA-2025-4.2"
    )
    assert resolution.operative_clause_id == "CLAUSE-BOREALIS-AMENDMENT-001-2"
    assert [step.predicate for step in resolution.path] == ["AMENDS", "CONTAINS", "SUPERSEDES"]


def test_conflict_detection_is_paired_with_explicit_supersession() -> None:
    conflicts = _service().conflicts("CLAUSE-BOREALIS-AMENDMENT-001-2")
    assert len(conflicts) == 1
    assert conflicts[0].to_id == "CLAUSE-BOREALIS-CHA-2025-4.2"
    resolution = _service().operative_clause("CONTRACT-BOREALIS-001", conflicts[0].to_id)
    assert resolution.operative_clause_id == conflicts[0].from_id


def test_missing_schedule_returns_insufficient_evidence_without_invention() -> None:
    result = _service().missing_evidence("CLAUSE-FJORD-SLA-2025-3.1")
    assert result.answerable is False
    assert result.missing_reference_ids == ("MISSING-SCHEDULE-Z",)
    assert result.reason == "referenced_evidence_not_in_corpus"
