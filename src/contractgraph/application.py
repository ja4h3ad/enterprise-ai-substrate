"""Primary ContractGraph application seam."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from contractgraph.application_models import (
    AnswerResult,
    AnalystDecision,
    ContractGraphRunConfig,
    ReviewEvidence,
    ReviewPacket,
    ReviewResolution,
    TraceEvent,
)
from contractgraph.artifacts import load_corpus
from contractgraph.persistence import SQLiteCheckpointStore, SQLiteTraceRepository
from contractgraph.providers import ReplayModelProvider
from contractgraph.telemetry import RecordedSpan, TelemetryRecorder, content_hash
from contractgraph.workflow import RetrievalState, build_workflow

DEFAULT_STATE_DB = Path(".contractgraph/state.db")
DEFAULT_REPLAY_FIXTURE = Path("replay/hero.json")


class ContractGraphApplication:
    """Run bounded retrieval workflows without exposing orchestration internals."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        state_db: Path = DEFAULT_STATE_DB,
        replay_fixture: Path = DEFAULT_REPLAY_FIXTURE,
        telemetry: TelemetryRecorder | None = None,
    ) -> None:
        artifact = load_corpus(artifact_root)
        self._trace_repository = SQLiteTraceRepository(state_db)
        self._checkpoint_store = SQLiteCheckpointStore(state_db)
        self._provider = ReplayModelProvider(replay_fixture)
        self._telemetry = telemetry or TelemetryRecorder()
        self._owns_telemetry = telemetry is None
        self._workflow = build_workflow(
            artifact,
            self._provider,
            checkpointer=self._checkpoint_store.saver,
            telemetry=self._telemetry,
        )

    def run(
        self,
        question: str,
        run_config: ContractGraphRunConfig | None = None,
    ) -> AnswerResult:
        config = run_config or ContractGraphRunConfig()
        run_id = str(uuid4())
        self._trace_repository.create_run(run_id, question, config.limits)
        initial_state: RetrievalState = {
            "run_id": run_id,
            "question": question,
            "limits": config.limits,
            "faults": config.faults,
            "capture_content": config.telemetry.capture_content,
            "plan": None,
            "retrieval_query": "",
            "lexical_results": (),
            "vector_results": (),
            "graph_results": (),
            "graph_paths": (),
            "fused_candidates": (),
            "reranked_results": (),
            "selected_evidence": (),
            "evidence_requirements": {},
            "evidence_sufficient": False,
            "insufficiency_reasons": (),
            "iteration": 0,
            "model_calls": 0,
            "provider_calls": (),
            "claims": (),
            "citations": (),
            "citations_valid": False,
            "answer": None,
            "status": "running",
            "review_required": False,
            "conflict_reasons": (),
            "review_decision": None,
            "review_requested_at": None,
            "degraded_components": (),
            "errors": (),
            "trace_events": (),
        }
        graph_config = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": config.limits.recursion_limit,
        }
        span_attributes: dict[str, object] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": "replay",
            "gen_ai.request.model": "recorded-structured-output",
            "contractgraph.run.id": run_id,
            "contractgraph.question.sha256": content_hash(question),
            "contractgraph.capture_content": config.telemetry.capture_content,
            "contractgraph.limit.retrieval_iterations": (
                config.limits.max_retrieval_iterations
            ),
            "contractgraph.limit.graph_depth": config.limits.max_graph_depth,
            "contractgraph.limit.candidates": (
                config.limits.max_candidates_per_retriever
            ),
            "contractgraph.limit.timeout_ms": (
                config.limits.local_retriever_timeout_ms
            ),
        }
        if config.telemetry.capture_content:
            span_attributes["contractgraph.content.question"] = question
        with self._telemetry.span("contractgraph.run", span_attributes) as span:
            try:
                state = self._workflow.invoke(
                    initial_state,
                    config=graph_config,
                    durability="sync",
                )
                result = _answer_result(state)
            except Exception as error:
                result = _error_result(run_id, config, error)
            span.set_attribute("contractgraph.run.status", result.status)
            span.set_attribute(
                "contractgraph.degraded_components", list(result.degraded_components)
            )
            if config.telemetry.capture_content and result.answer:
                span.set_attribute("contractgraph.content.answer", result.answer)
            if result.status == "error":
                self._telemetry.mark_error(
                    span, result.uncertainty_reasons[0]
                )
        self._trace_repository.append_events(run_id, result.trace_events)
        if result.status == "review_required":
            snapshot = self._workflow.get_state(graph_config)
            checkpoint_id = str(snapshot.config["configurable"]["checkpoint_id"])
            packet = ReviewPacket(
                run_id=run_id,
                status="pending",
                conflict_reasons=tuple(state["conflict_reasons"]),
                evidence=tuple(
                    ReviewEvidence(
                        evidence_id=item.evidence_id,
                        clause_id=item.clause_id,
                        document_id=item.document_id,
                        page_number=item.page_number,
                        section=item.section,
                        text=item.text,
                    )
                    for item in state["selected_evidence"]
                ),
                limits=config.limits,
                checkpoint_id=checkpoint_id,
                created_at=datetime.fromisoformat(str(state["review_requested_at"])),
            )
            self._trace_repository.pause_for_review(result, packet)
        else:
            self._trace_repository.complete_run(result)
        return result

    def get_review(self, run_id: str) -> ReviewPacket:
        """Return the stable public review packet, never internal graph state."""
        return self._trace_repository.read_review(run_id)

    def resume_review(
        self, run_id: str, decision: AnalystDecision
    ) -> ReviewResolution:
        packet = self._trace_repository.read_review(run_id)
        valid_evidence_ids = {item.evidence_id for item in packet.evidence}
        if (
            decision.controlling_evidence_id is not None
            and decision.controlling_evidence_id not in valid_evidence_ids
        ):
            raise ValueError("controlling evidence is not in the review packet")
        self._trace_repository.begin_review_resolution(
            run_id, packet.checkpoint_id, decision
        )
        decided_at = datetime.now(UTC)
        resume_payload = decision.model_dump(mode="json")
        resume_payload["decided_at"] = decided_at.isoformat()
        graph_config = {
            "configurable": {
                "thread_id": run_id,
                "checkpoint_id": packet.checkpoint_id,
            },
            "recursion_limit": packet.limits.recursion_limit,
        }
        state = self._workflow.invoke(
            Command(resume=resume_payload), config=graph_config, durability="sync"
        )
        result = _answer_result(state)
        self._trace_repository.append_events(run_id, result.trace_events)
        self._trace_repository.complete_run(result)
        resolved_packet = self._trace_repository.complete_review_resolution(
            result, decided_at.isoformat()
        )
        return ReviewResolution(packet=resolved_packet, result=result)

    def read_persisted_run(self, run_id: str) -> dict[str, object]:
        return self._trace_repository.read_run(run_id)

    def checkpoint_history(self, run_id: str) -> tuple[object, ...]:
        config = {"configurable": {"thread_id": run_id}}
        return tuple(self._workflow.get_state_history(config))

    def telemetry_spans(self, run_id: str) -> tuple[RecordedSpan, ...]:
        return self._telemetry.spans_for_run(run_id)

    def close(self) -> None:
        self._checkpoint_store.close()
        self._trace_repository.close()
        if self._owns_telemetry:
            self._telemetry.shutdown()

    def __enter__(self) -> ContractGraphApplication:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _answer_result(state: dict[str, Any]) -> AnswerResult:
    uncertainty = tuple(state["insufficiency_reasons"])
    if state["errors"]:
        uncertainty = (*uncertainty, *state["errors"])
    return AnswerResult(
        run_id=state["run_id"],
        status=state["status"],
        answer=state["answer"],
        claims=tuple(state["claims"]),
        citations=tuple(state["citations"]),
        uncertainty_reasons=uncertainty,
        degraded_components=tuple(dict.fromkeys(state["degraded_components"])),
        graph_paths=tuple(state["graph_paths"]),
        evidence=tuple(state["selected_evidence"]),
        fused_candidates=tuple(state["fused_candidates"]),
        iterations=state["iteration"],
        model_calls=state["model_calls"],
        limits=state["limits"],
        trace_events=tuple(state["trace_events"]),
    )


def _error_result(
    run_id: str, config: ContractGraphRunConfig, error: Exception
) -> AnswerResult:
    error_type = type(error).__name__
    return AnswerResult(
        run_id=run_id,
        status="error",
        answer=None,
        claims=(),
        citations=(),
        uncertainty_reasons=(f"validated_boundary_error:{error_type}",),
        degraded_components=(),
        graph_paths=(),
        evidence=(),
        fused_candidates=(),
        iterations=0,
        model_calls=0,
        limits=config.limits,
        trace_events=(
            TraceEvent(
                node="application_boundary",
                event_type="validation_failed",
                iteration=0,
                details={"error_type": error_type, "failed_safely": True},
            ),
        ),
    )


def render_answer(result: AnswerResult) -> str:
    lines = [
        "ContractGraph agent run",
        f"Run ID: {result.run_id}",
        f"Status: {result.status}",
        f"Answer: {result.answer or '[no answer]'}",
        f"Iterations: {result.iterations}/{result.limits.max_retrieval_iterations}",
        f"Model calls: {result.model_calls}/{result.limits.max_model_calls}",
        "Degraded components: "
        + (", ".join(result.degraded_components) or "none"),
        "",
        "Claims and citations",
    ]
    citations_by_claim: dict[str, list[str]] = {}
    for citation in result.citations:
        citations_by_claim.setdefault(citation.claim_id, []).append(
            f"{citation.document_id}, p.{citation.page_number}, "
            f"§{citation.section}, {citation.clause_id}"
        )
    for claim in result.claims:
        lines.append(f"- {claim.claim_id}: {claim.text}")
        for citation in citations_by_claim.get(claim.claim_id, []):
            lines.append(f"  - {citation}")
    security_findings = [
        evidence for evidence in result.evidence if evidence.security_flags
    ]
    lines.extend(("", "Security findings"))
    if not security_findings:
        lines.append("- none")
    for evidence in security_findings:
        lines.append(
            f"- {evidence.clause_id}: {', '.join(evidence.security_flags)} "
            "[retained as untrusted evidence]"
        )
    lines.extend(("", "Graph path"))
    for step in result.graph_paths:
        arrow = "<--" if step.traversal == "reverse" else "--"
        lines.append(
            f"- {step.from_id} {arrow}{step.predicate}-- {step.to_id} "
            f"[source={step.source_clause_id}]"
        )
    lines.extend(("", "Execution trace"))
    for event in result.trace_events:
        lines.append(f"- iteration={event.iteration} {event.node}: {event.event_type}")
    return "\n".join(lines) + "\n"
