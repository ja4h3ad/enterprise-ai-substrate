from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contractgraph.application import ContractGraphApplication
from contractgraph.application_models import AnalystDecision
from contractgraph.ingestion import build_corpus, persist_corpus

PROJECT_ROOT = Path(__file__).parents[1]
CORPUS_ROOT = PROJECT_ROOT / "corpus"
CONFLICT_FIXTURE = PROJECT_ROOT / "replay" / "conflict.json"
CONFLICT_QUESTION = (
    "Delta's agreement says twenty-four hours while Exhibit D says forty-eight hours "
    "for confirmed Customer Data compromise. Which deadline controls?"
)
BASE_EVIDENCE = "E-CLAUSE-DELTA-DPA-2025-4.1"
EXHIBIT_EVIDENCE = "E-CLAUSE-DELTA-EXHIBIT-D-D.13"


def _application(tmp_path: Path) -> ContractGraphApplication:
    artifact_root = tmp_path / "artifacts"
    persist_corpus(build_corpus(CORPUS_ROOT), artifact_root)
    return ContractGraphApplication(
        artifact_root=artifact_root,
        state_db=tmp_path / "state.db",
        replay_fixture=CONFLICT_FIXTURE,
    )


def test_conflict_creates_a_durable_public_review_packet(tmp_path: Path) -> None:
    with _application(tmp_path) as application:
        result = application.run(CONFLICT_QUESTION)
        packet = application.get_review(result.run_id)
        persisted = application.read_persisted_run(result.run_id)
        history = application.checkpoint_history(result.run_id)

    assert result.status == "review_required"
    assert result.answer is None
    assert result.model_calls == 1
    assert packet.status == "pending"
    assert packet.run_id == result.run_id
    assert packet.conflict_reasons == (
        "competing_clause_language",
        "no_deterministic_precedence_path",
    )
    assert {item.evidence_id for item in packet.evidence} == {
        BASE_EVIDENCE,
        EXHIBIT_EVIDENCE,
    }
    assert all(item.document_id and item.page_number for item in packet.evidence)
    assert packet.limits == result.limits
    assert packet.checkpoint_id
    assert persisted["run"]["status"] == "review_required"
    assert persisted["run"]["completed_at"] is None
    assert any(snapshot.interrupts for snapshot in history)
    checkpoint = next(snapshot for snapshot in history if snapshot.interrupts)
    assert checkpoint.values["run_id"] == result.run_id
    assert checkpoint.values["limits"] == result.limits
    assert checkpoint.values["conflict_reasons"] == list(packet.conflict_reasons)
    assert {item.evidence_id for item in checkpoint.values["selected_evidence"]} == {
        BASE_EVIDENCE,
        EXHIBIT_EVIDENCE,
    }


def test_analyst_can_abstain_and_audit_the_resumed_trajectory(tmp_path: Path) -> None:
    with _application(tmp_path) as application:
        paused = application.run(CONFLICT_QUESTION)
        resolution = application.resume_review(
            paused.run_id,
            AnalystDecision(
                disposition="abstain",
                rationale="The corpus contains no precedence rule for these clauses.",
            ),
        )
        persisted = application.read_persisted_run(paused.run_id)

    assert resolution.packet.status == "resolved"
    assert resolution.packet.resolved_at is not None
    assert resolution.result.status == "insufficient_evidence"
    assert resolution.result.answer is None
    assert resolution.result.claims == ()
    assert resolution.result.citations == ()
    assert "analyst_abstained" in resolution.result.uncertainty_reasons
    event = next(
        event
        for event in resolution.result.trace_events
        if event.event_type == "human_decision_recorded"
    )
    assert event.details["disposition"] == "abstain"
    assert event.details["before_status"] == "review_required"
    assert event.details["after_status"] == "insufficient_evidence"
    assert event.details["rationale"]
    assert event.details["decided_at"]
    assert event.details["before_state"]["status"] == "review_required"
    assert event.details["after_state"]["status"] == "insufficient_evidence"
    assert event.details["timestamps"]["review_requested_at"]
    assert event.details["timestamps"]["decided_at"]
    assert persisted["run"]["status"] == "insufficient_evidence"
    assert len(persisted["events"]) == len(resolution.result.trace_events)


def test_analyst_selection_is_grounded_and_replay_is_rejected(tmp_path: Path) -> None:
    with _application(tmp_path) as application:
        paused = application.run(CONFLICT_QUESTION)
        resolution = application.resume_review(
            paused.run_id,
            AnalystDecision(
                disposition="select_controlling_evidence",
                controlling_evidence_id=BASE_EVIDENCE,
                rationale="The analyst determines the signed Agreement controls Exhibit D.",
            ),
        )

        assert resolution.result.status == "answered"
        assert "twenty-four hours" in (resolution.result.answer or "")
        assert resolution.result.claims[0].evidence_ids == (BASE_EVIDENCE,)
        assert resolution.result.citations[0].clause_id == (
            "CLAUSE-DELTA-DPA-2025-4.1"
        )
        assert resolution.result.citations[0].document_id == "DOC-DELTA-DPA-2025"
        assert resolution.result.citations[0].page_number == 2
        assert resolution.result.citations[0].section == "4.1"

        with pytest.raises(RuntimeError, match="stale_or_already_resolved_checkpoint"):
            application.resume_review(
                paused.run_id,
                AnalystDecision(
                    disposition="abstain",
                    rationale="A second decision must not resume an old checkpoint.",
                ),
            )


def test_invalid_review_decisions_fail_before_checkpoint_consumption(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        AnalystDecision(disposition="abstain", rationale="short")

    with _application(tmp_path) as application:
        paused = application.run(CONFLICT_QUESTION)
        with pytest.raises(ValueError, match="not in the review packet"):
            application.resume_review(
                paused.run_id,
                AnalystDecision(
                    disposition="select_controlling_evidence",
                    controlling_evidence_id="E-NOT-IN-PACKET",
                    rationale="This fabricated identifier must be rejected safely.",
                ),
            )
        assert application.get_review(paused.run_id).status == "pending"
        persisted = application.read_persisted_run(paused.run_id)

    assert persisted["run"]["status"] == "review_required"
    assert json.loads(persisted["run"]["result_json"])["status"] == (
        "review_required"
    )
