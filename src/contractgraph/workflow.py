"""Explicit, bounded LangGraph workflow for ContractGraph retrieval."""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from contractgraph.application_models import (
    Citation,
    Claim,
    Evidence,
    FusedCandidate,
    ProviderCall,
    RetrievalPlan,
    RunLimits,
    TraceEvent,
)
from contractgraph.graph import ContractGraph
from contractgraph.models import Clause, CorpusArtifact, GraphStep, SearchResult
from contractgraph.providers import ModelProvider
from contractgraph.retrieval import BM25Retriever, ExactVectorRetriever


class RetrievalState(TypedDict):
    run_id: str
    question: str
    limits: RunLimits
    plan: RetrievalPlan | None
    retrieval_query: str
    lexical_results: tuple[SearchResult, ...]
    vector_results: tuple[SearchResult, ...]
    graph_results: tuple[SearchResult, ...]
    graph_paths: tuple[GraphStep, ...]
    fused_candidates: tuple[FusedCandidate, ...]
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
    degraded_components: tuple[str, ...]
    errors: Annotated[tuple[str, ...], operator.add]
    trace_events: Annotated[tuple[TraceEvent, ...], operator.add]


def build_workflow(
    artifact: CorpusArtifact,
    provider: ModelProvider,
    *,
    checkpointer: BaseCheckpointSaver,
):
    clauses = {clause.clause_id: clause for clause in artifact.clauses}
    lexical = BM25Retriever(artifact.clauses)
    vector = ExactVectorRetriever(artifact.clauses)
    graph_retriever = ContractGraph(artifact)

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
        resolution = graph_retriever.resolve_operative_clause(
            contract_id=plan.contract_id,
            base_clause_id=plan.base_clause_id,
            max_depth=state["limits"].max_graph_depth,
            max_candidates=state["limits"].max_candidates_per_retriever,
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
        return {
            "selected_evidence": evidence,
            "evidence_requirements": requirements,
            "evidence_sufficient": sufficient,
            "insufficiency_reasons": reasons,
            "trace_events": (
                _event(
                    "assess_evidence",
                    "evidence_assessed",
                    state["iteration"],
                    requirements=requirements,
                    sufficient=sufficient,
                    reasons=list(reasons),
                    selected=[item.evidence_id for item in evidence],
                ),
            ),
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
            state["question"], state["selected_evidence"]
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
    builder.add_node("analyze_and_plan", analyze_and_plan)
    builder.add_node("lexical_retrieve", lexical_retrieve)
    builder.add_node("vector_retrieve", vector_retrieve)
    builder.add_node("graph_retrieve", graph_retrieve)
    builder.add_node("fuse_candidates", fuse_candidates)
    builder.add_node("assess_evidence", assess_evidence)
    builder.add_node("reformulate_query", reformulate_query)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("verify_citations", verify_citations)
    builder.add_node("finalize_insufficient", finalize_insufficient)
    builder.add_node("finalize_trace", finalize_trace)

    builder.add_edge(START, "analyze_and_plan")
    for retriever_node in ("lexical_retrieve", "vector_retrieve", "graph_retrieve"):
        builder.add_edge("analyze_and_plan", retriever_node)
    builder.add_edge(
        ["lexical_retrieve", "vector_retrieve", "graph_retrieve"], "fuse_candidates"
    )
    builder.add_edge("fuse_candidates", "assess_evidence")
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
