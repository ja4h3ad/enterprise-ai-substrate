"""Explicit, bounded LangGraph workflow for ContractGraph retrieval."""

from __future__ import annotations

import operator
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from contractgraph.application_models import (
    AmendmentResolutionRequest,
    Citation,
    Claim,
    Evidence,
    FusedCandidate,
    FaultInjection,
    GroundingEnvelope,
    ProviderCall,
    RetrievalPlan,
    RunLimits,
    TraceEvent,
)
from contractgraph.graph import ContractGraph
from contractgraph.models import Clause, CorpusArtifact, GraphStep, SearchResult
from contractgraph.providers import ModelProvider
from contractgraph.reranking import LocalCrossEncoderReranker
from contractgraph.retrieval import BM25Retriever, ExactVectorRetriever
from contractgraph.telemetry import TelemetryRecorder, content_hash


class RetrievalState(TypedDict):
    run_id: str
    question: str
    limits: RunLimits
    faults: FaultInjection
    capture_content: bool
    plan: RetrievalPlan | None
    retrieval_query: str
    lexical_results: tuple[SearchResult, ...]
    vector_results: tuple[SearchResult, ...]
    graph_results: tuple[SearchResult, ...]
    graph_paths: tuple[GraphStep, ...]
    fused_candidates: tuple[FusedCandidate, ...]
    reranked_results: tuple[SearchResult, ...]
    selected_evidence: tuple[Evidence, ...]
    evidence_requirements: dict[str, bool]
    evidence_sufficient: bool
    insufficiency_reasons: tuple[str, ...]
    iteration: int
    model_calls: int
    provider_calls: tuple[ProviderCall, ...]
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]
    citations_valid: bool
    answer: str | None
    status: str
    degraded_components: Annotated[tuple[str, ...], operator.add]
    errors: Annotated[tuple[str, ...], operator.add]
    trace_events: Annotated[tuple[TraceEvent, ...], operator.add]


def build_workflow(
    artifact: CorpusArtifact,
    provider: ModelProvider,
    *,
    checkpointer: BaseCheckpointSaver,
    telemetry: TelemetryRecorder | None = None,
):
    clauses = {clause.clause_id: clause for clause in artifact.clauses}
    lexical = BM25Retriever(artifact.clauses)
    vector = ExactVectorRetriever(artifact.clauses)
    graph_retriever = ContractGraph(artifact)
    reranker = LocalCrossEncoderReranker(None, top_n=10)

    def analyze_and_plan(state: RetrievalState) -> dict[str, object]:
        plan, provider_call = provider.analyze_and_plan(state["question"])
        return {
            "plan": plan,
            "retrieval_query": plan.retrieval_query,
            "iteration": 1,
            "model_calls": state["model_calls"] + 1,
            "provider_calls": (*state["provider_calls"], provider_call),
            "trace_events": (
                _event(
                    "analyze_and_plan",
                    "model_call",
                    1,
                    intent=plan.intent,
                    entities=list(plan.entities),
                    retrieval_query=plan.retrieval_query,
                    required_evidence=list(plan.required_evidence),
                ),
            ),
        }

    def lexical_retrieve(state: RetrievalState) -> dict[str, object]:
        results = lexical.search(
            state["retrieval_query"],
            limit=state["limits"].max_candidates_per_retriever,
        )
        return {
            "lexical_results": results,
            "trace_events": (
                _retrieval_event("lexical_retrieve", state, results),
            ),
        }

    def vector_retrieve(state: RetrievalState) -> dict[str, object]:
        timeout_ms = state["limits"].local_retriever_timeout_ms
        if state["faults"].vector_timeout:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _delayed_vector_search,
                vector,
                state["retrieval_query"],
                state["limits"].max_candidates_per_retriever,
                timeout_ms,
            )
            try:
                results = future.result(timeout=timeout_ms / 1000)
            except FutureTimeoutError:
                future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                return {
                    "vector_results": (),
                    "degraded_components": ("vector_retrieval_timeout",),
                    "trace_events": (
                        _event(
                            "vector_retrieve",
                            "retrieval_timeout",
                            state["iteration"],
                            timeout_ms=timeout_ms,
                            fault_injected=True,
                            continued=True,
                        ),
                    ),
                }
            else:
                executor.shutdown(wait=True)
        else:
            results = vector.search(
                state["retrieval_query"],
                limit=state["limits"].max_candidates_per_retriever,
            )
        return {
            "vector_results": results,
            "trace_events": (
                _retrieval_event("vector_retrieve", state, results),
            ),
        }

    def graph_retrieve(state: RetrievalState) -> dict[str, object]:
        plan = _require_plan(state)
        request = AmendmentResolutionRequest(
            contract_id=plan.contract_id,
            base_clause_id=plan.base_clause_id,
            max_depth=state["limits"].max_graph_depth,
            max_candidates=state["limits"].max_candidates_per_retriever,
        )
        resolution = graph_retriever.resolve_operative_clause(
            contract_id=request.contract_id,
            base_clause_id=request.base_clause_id,
            max_depth=request.max_depth,
            max_candidates=request.max_candidates,
        )
        operative = clauses[resolution.operative_clause_id]
        result = _graph_result(operative)
        return {
            "graph_results": (result,),
            "graph_paths": resolution.path,
            "trace_events": (
                _event(
                    "graph_retrieve",
                    "retrieval_completed",
                    state["iteration"],
                    candidates=[operative.clause_id],
                    path=[
                        {
                            "from": step.from_id,
                            "predicate": step.predicate,
                            "to": step.to_id,
                            "traversal": step.traversal,
                            "source_clause_id": step.source_clause_id,
                        }
                        for step in resolution.path
                    ],
                ),
            ),
        }

    def fuse_candidates(state: RetrievalState) -> dict[str, object]:
        fused = _reciprocal_rank_fusion(
            state["lexical_results"],
            state["vector_results"],
            state["graph_results"],
            limit=state["limits"].max_fused_candidates,
        )
        return {
            "fused_candidates": fused,
            "trace_events": (
                _event(
                    "fuse_candidates",
                    "rrf_completed",
                    state["iteration"],
                    candidates=[
                        {
                            "clause_id": candidate.clause_id,
                            "rank": candidate.rank,
                            "score": candidate.score,
                            "sources": list(candidate.sources),
                        }
                        for candidate in fused
                    ],
                ),
            ),
        }

    def rerank_candidates(state: RetrievalState) -> dict[str, object]:
        candidates = tuple(
            _candidate_result(candidate, clauses[candidate.clause_id])
            for candidate in state["fused_candidates"]
        )
        outcome = reranker.rerank(state["retrieval_query"], candidates)
        return {
            "reranked_results": outcome.results,
            "degraded_components": ("reranker_unavailable",)
            if outcome.degraded
            else (),
            "trace_events": (
                _event(
                    "rerank_candidates",
                    "reranker_degraded" if outcome.degraded else "reranking_completed",
                    state["iteration"],
                    model=outcome.model,
                    degradation_reason=outcome.degradation_reason,
                    position_changes=[
                        {
                            "clause_id": change.clause_id,
                            "before": change.before,
                            "after": change.after,
                        }
                        for change in outcome.changes
                    ],
                ),
            ),
        }

    def assess_evidence(state: RetrievalState) -> dict[str, object]:
        plan = _require_plan(state)
        predicates = {step.predicate for step in state["graph_paths"]}
        retrieved_ids = {
            result.clause_id
            for result in (*state["lexical_results"], *state["vector_results"])
        }
        operative_ids = {result.clause_id for result in state["graph_results"]}
        requirements = {
            "base_clause": plan.base_clause_id in retrieved_ids,
            "applicable_amendment": "AMENDS" in predicates,
            "supersession_path": "SUPERSEDES" in predicates,
            "operative_clause": bool(operative_ids - {plan.base_clause_id}),
            "source_provenance": all(
                step.source_clause_id in clauses for step in state["graph_paths"]
            )
            and bool(state["graph_paths"]),
        }
        reasons = tuple(
            f"missing:{requirement}"
            for requirement in plan.required_evidence
            if not requirements.get(requirement, False)
        )
        evidence = _select_evidence(state, clauses)
        sufficient = not reasons and len(evidence) <= state["limits"].max_evidence_items
        security_findings = [
            {
                "clause_id": item.clause_id,
                "flags": list(item.security_flags),
                "disposition": "retained_as_untrusted_evidence",
            }
            for item in evidence
            if item.security_flags
        ]
        events = [
            _event(
                "assess_evidence",
                "evidence_assessed",
                state["iteration"],
                requirements=requirements,
                sufficient=sufficient,
                reasons=list(reasons),
                selected=[item.evidence_id for item in evidence],
            )
        ]
        if security_findings:
            events.append(
                _event(
                    "assess_evidence",
                    "suspicious_evidence_flagged",
                    state["iteration"],
                    findings=security_findings,
                    plan_unchanged=True,
                    limits_unchanged=True,
                    answer_policy_unchanged=True,
                )
            )
        return {
            "selected_evidence": evidence,
            "evidence_requirements": requirements,
            "evidence_sufficient": sufficient,
            "insufficiency_reasons": reasons,
            "trace_events": tuple(events),
        }

    def reformulate_query(state: RetrievalState) -> dict[str, object]:
        if state["model_calls"] >= state["limits"].max_model_calls:
            return {
                "errors": ("model_call_limit_reached",),
                "trace_events": (
                    _event(
                        "reformulate_query",
                        "limit_reached",
                        state["iteration"],
                        model_calls=state["model_calls"],
                    ),
                ),
            }
        retrieval_query, provider_call = provider.reformulate(
            state["question"], state["insufficiency_reasons"]
        )
        return {
            "retrieval_query": retrieval_query,
            "iteration": state["iteration"] + 1,
            "model_calls": state["model_calls"] + 1,
            "provider_calls": (*state["provider_calls"], provider_call),
            "trace_events": (
                _event(
                    "reformulate_query",
                    "model_call",
                    state["iteration"] + 1,
                    retrieval_query=retrieval_query,
                    reasons=list(state["insufficiency_reasons"]),
                ),
            ),
        }

    def synthesize_answer(state: RetrievalState) -> dict[str, object]:
        if state["model_calls"] >= state["limits"].max_model_calls:
            return {
                "status": "error",
                "errors": ("model_call_limit_reached",),
                "trace_events": (
                    _event(
                        "synthesize_answer",
                        "limit_reached",
                        state["iteration"],
                        model_calls=state["model_calls"],
                    ),
                ),
            }
        synthesis, provider_call = provider.synthesize(
            GroundingEnvelope(
                question=state["question"], evidence=state["selected_evidence"]
            )
        )
        return {
            "answer": synthesis.answer,
            "claims": synthesis.claims,
            "model_calls": state["model_calls"] + 1,
            "provider_calls": (*state["provider_calls"], provider_call),
            "trace_events": (
                _event(
                    "synthesize_answer",
                    "model_call",
                    state["iteration"],
                    claims=[claim.claim_id for claim in synthesis.claims],
                    evidence=[item.evidence_id for item in state["selected_evidence"]],
                ),
            ),
        }

    def verify_citations(state: RetrievalState) -> dict[str, object]:
        evidence = {item.evidence_id: item for item in state["selected_evidence"]}
        errors: list[str] = []
        citations: list[Citation] = []
        for claim in state["claims"]:
            if not claim.evidence_ids:
                errors.append(f"unsupported_claim:{claim.claim_id}")
            for evidence_id in claim.evidence_ids:
                item = evidence.get(evidence_id)
                if item is None:
                    errors.append(f"unknown_evidence:{claim.claim_id}:{evidence_id}")
                    continue
                citations.append(
                    Citation(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        document_id=item.document_id,
                        page_number=item.page_number,
                        section=item.section,
                        clause_id=item.clause_id,
                    )
                )
        valid = not errors and bool(state["claims"]) and bool(citations)
        return {
            "citations": tuple(citations),
            "citations_valid": valid,
            "status": "answered" if valid else "error",
            "errors": tuple(errors),
            "trace_events": (
                _event(
                    "verify_citations",
                    "citations_verified",
                    state["iteration"],
                    valid=valid,
                    citations=len(citations),
                    errors=errors,
                ),
            ),
        }

    def finalize_insufficient(state: RetrievalState) -> dict[str, object]:
        return {
            "status": "insufficient_evidence",
            "answer": None,
            "trace_events": (
                _event(
                    "finalize_insufficient",
                    "abstained",
                    state["iteration"],
                    reasons=list(state["insufficiency_reasons"]),
                ),
            ),
        }

    def finalize_trace(state: RetrievalState) -> dict[str, object]:
        return {
            "trace_events": (
                _event(
                    "finalize_trace",
                    "run_completed",
                    state["iteration"],
                    status=state["status"],
                    model_calls=state["model_calls"],
                ),
            ),
        }

    builder = StateGraph(RetrievalState)
    nodes: tuple[
        tuple[str, Callable[[RetrievalState], dict[str, object]]], ...
    ] = (
        ("analyze_and_plan", analyze_and_plan),
        ("lexical_retrieve", lexical_retrieve),
        ("vector_retrieve", vector_retrieve),
        ("graph_retrieve", graph_retrieve),
        ("fuse_candidates", fuse_candidates),
        ("rerank_candidates", rerank_candidates),
        ("assess_evidence", assess_evidence),
        ("reformulate_query", reformulate_query),
        ("synthesize_answer", synthesize_answer),
        ("verify_citations", verify_citations),
        ("finalize_insufficient", finalize_insufficient),
        ("finalize_trace", finalize_trace),
    )
    for node_name, node in nodes:
        builder.add_node(node_name, _instrument_node(telemetry, node_name, node))

    builder.add_edge(START, "analyze_and_plan")
    for retriever_node in ("lexical_retrieve", "vector_retrieve", "graph_retrieve"):
        builder.add_edge("analyze_and_plan", retriever_node)
    builder.add_edge(
        ["lexical_retrieve", "vector_retrieve", "graph_retrieve"], "fuse_candidates"
    )
    builder.add_edge("fuse_candidates", "rerank_candidates")
    builder.add_edge("rerank_candidates", "assess_evidence")
    builder.add_conditional_edges("assess_evidence", _route_after_assessment)
    for retriever_node in ("lexical_retrieve", "vector_retrieve", "graph_retrieve"):
        builder.add_edge("reformulate_query", retriever_node)
    builder.add_edge("synthesize_answer", "verify_citations")
    builder.add_edge("verify_citations", "finalize_trace")
    builder.add_edge("finalize_insufficient", "finalize_trace")
    builder.add_edge("finalize_trace", END)
    return builder.compile(checkpointer=checkpointer)


def _route_after_assessment(
    state: RetrievalState,
) -> Literal["synthesize_answer", "reformulate_query", "finalize_insufficient"]:
    if state["evidence_sufficient"]:
        return "synthesize_answer"
    if state["iteration"] < state["limits"].max_retrieval_iterations:
        return "reformulate_query"
    return "finalize_insufficient"


def _require_plan(state: RetrievalState) -> RetrievalPlan:
    plan = state["plan"]
    if plan is None:
        raise RuntimeError("Retrieval plan is not available")
    return plan


def _graph_result(clause: Clause) -> SearchResult:
    return SearchResult(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        page_number=clause.page_number,
        section=clause.section,
        title=clause.title,
        text=clause.text,
        score=1.0,
        rank=1,
        retriever="graph-bounded-amendment",
    )


def _candidate_result(candidate: FusedCandidate, clause: Clause) -> SearchResult:
    return SearchResult(
        clause_id=clause.clause_id,
        document_id=clause.document_id,
        page_number=clause.page_number,
        section=clause.section,
        title=clause.title,
        text=clause.text,
        score=candidate.score,
        rank=candidate.rank,
        retriever="rrf",
    )


def _delayed_vector_search(
    vector: ExactVectorRetriever,
    query: str,
    limit: int,
    timeout_ms: int,
) -> tuple[SearchResult, ...]:
    time.sleep((timeout_ms + 5) / 1000)
    return vector.search(query, limit=limit)


def _reciprocal_rank_fusion(
    lexical: tuple[SearchResult, ...],
    vector: tuple[SearchResult, ...],
    graph: tuple[SearchResult, ...],
    *,
    limit: int,
    rank_constant: int = 60,
) -> tuple[FusedCandidate, ...]:
    scores: defaultdict[str, float] = defaultdict(float)
    sources: defaultdict[str, set[str]] = defaultdict(set)
    for result_list in (lexical, vector, graph):
        for result in result_list:
            scores[result.clause_id] += 1 / (rank_constant + result.rank)
            sources[result.clause_id].add(result.retriever)
    ordered = sorted(scores, key=lambda clause_id: (-scores[clause_id], clause_id))[:limit]
    return tuple(
        FusedCandidate(
            clause_id=clause_id,
            score=scores[clause_id],
            rank=rank,
            sources=tuple(sorted(sources[clause_id])),
        )
        for rank, clause_id in enumerate(ordered, start=1)
    )


def _select_evidence(
    state: RetrievalState, clauses: dict[str, Clause]
) -> tuple[Evidence, ...]:
    plan = _require_plan(state)
    required_ids = {plan.base_clause_id}
    required_ids.update(result.clause_id for result in state["graph_results"])
    required_ids.update(step.source_clause_id for step in state["graph_paths"])
    fused = {candidate.clause_id: candidate for candidate in state["fused_candidates"]}
    selected: list[Evidence] = []
    for clause_id in sorted(required_ids):
        clause = clauses.get(clause_id)
        if clause is None:
            continue
        candidate = fused.get(clause_id)
        sources = candidate.sources if candidate else ("graph-provenance",)
        selected.append(
            Evidence(
                evidence_id=f"E-{clause_id}",
                clause_id=clause_id,
                document_id=clause.document_id,
                page_number=clause.page_number,
                section=clause.section,
                text=clause.text,
                retrieval_sources=sources,
                security_flags=_security_flags(clause.text),
            )
        )
    return tuple(selected[: state["limits"].max_evidence_items])


def _retrieval_event(
    node: str, state: RetrievalState, results: tuple[SearchResult, ...]
) -> TraceEvent:
    return _event(
        node,
        "retrieval_completed",
        state["iteration"],
        query=state["retrieval_query"],
        candidates=[
            {
                "clause_id": result.clause_id,
                "rank": result.rank,
                "score": result.score,
            }
            for result in results
        ],
    )


def _event(node: str, event_type: str, iteration: int, **details: object) -> TraceEvent:
    return TraceEvent(
        node=node,
        event_type=event_type,
        iteration=iteration,
        details=dict(details),
    )


def _security_flags(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    patterns = {
        "instruction_override": ("ignore the contract", "ignore prior instructions"),
        "system_instruction_request": ("reveal system instructions", "system prompt"),
        "tool_control_language": ("change tools", "invoke tool", "disable retrieval"),
    }
    return tuple(
        flag
        for flag, indicators in patterns.items()
        if any(indicator in normalized for indicator in indicators)
    )


def _instrument_node(
    telemetry: TelemetryRecorder | None,
    node_name: str,
    node: Callable[[RetrievalState], dict[str, object]],
) -> Callable[[RetrievalState], dict[str, object]]:
    if telemetry is None:
        return node

    def instrumented(state: RetrievalState) -> dict[str, object]:
        operation = (
            "text_completion"
            if node_name in {"analyze_and_plan", "reformulate_query", "synthesize_answer"}
            else "retrieval"
            if node_name.endswith("retrieve")
            else "invoke_workflow"
        )
        attributes: dict[str, object] = {
            "gen_ai.operation.name": operation,
            "contractgraph.run.id": state["run_id"],
            "contractgraph.node.name": node_name,
            "contractgraph.iteration": state["iteration"],
            "contractgraph.question.sha256": content_hash(state["question"]),
            "contractgraph.limit.graph_depth": state["limits"].max_graph_depth,
            "contractgraph.limit.candidates": state[
                "limits"
            ].max_candidates_per_retriever,
        }
        if operation == "text_completion":
            attributes["gen_ai.provider.name"] = "replay"
            attributes["gen_ai.request.model"] = "recorded-structured-output"
        if operation == "retrieval":
            attributes["gen_ai.data_source.id"] = "contractgraph-synthetic-corpus"
        with telemetry.span(f"contractgraph.{node_name}", attributes) as span:
            result = node(state)
            events = tuple(result.get("trace_events", ()))
            span.set_attribute(
                "contractgraph.degraded",
                bool(result.get("degraded_components")),
            )
            span.set_attribute(
                "contractgraph.security.suspicious_evidence",
                any(
                    event.event_type == "suspicious_evidence_flagged"
                    for event in events
                ),
            )
            span.set_attribute(
                "contractgraph.timeout",
                any(event.event_type == "retrieval_timeout" for event in events),
            )
            return result

    return instrumented
    AmendmentResolutionRequest,
