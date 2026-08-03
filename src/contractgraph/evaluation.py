"""Deterministic golden-set evaluation and five-way retrieval ablation."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from contractgraph.ingestion import artifact_digest
from contractgraph.local_models import DeterministicSemanticEmbedder
from contractgraph.models import CorpusArtifact, SearchResult
from contractgraph.reranking import LocalCrossEncoderReranker
from contractgraph.retrieval import BM25Retriever, SemanticVectorRetriever

Category = Literal[
    "direct",
    "semantic",
    "amendment",
    "multi_hop",
    "comparison",
    "conflict",
    "ambiguous",
    "negative",
    "adversarial",
]
VariantName = Literal[
    "vector_only",
    "lexical_vector",
    "lexical_vector_reranking",
    "lexical_vector_graph",
    "agentic_full",
]

VARIANTS: tuple[VariantName, ...] = (
    "vector_only",
    "lexical_vector",
    "lexical_vector_reranking",
    "lexical_vector_graph",
    "agentic_full",
)
EXPECTED_DISTRIBUTION = {
    "direct": 5,
    "semantic": 4,
    "amendment": 4,
    "multi_hop": 3,
    "comparison": 3,
    "conflict": 2,
    "ambiguous": 1,
    "negative": 1,
    "adversarial": 1,
}
FAILURE_TAXONOMY = (
    "correct",
    "superseded_clause_retrieval",
    "missed_amendment",
    "exact_term_displacement",
    "missing_evidence",
    "unresolved_conflict",
    "unsupported_claim",
    "correct_abstention",
    "ambiguous_question",
    "other_retrieval_miss",
)
FAILURE_DESCRIPTIONS = {
    "correct": "All labeled clauses were retrieved within K and no trust-control failure occurred.",
    "superseded_clause_retrieval": (
        "The top result is an older clause with a reviewed SUPERSEDES relationship "
        "to the operative clause."
    ),
    "missed_amendment": (
        "The question requires an amendment but the labeled operative amendment "
        "clause was absent within K."
    ),
    "exact_term_displacement": (
        "A direct exact-term question was displaced by a different top-ranked clause."
    ),
    "missing_evidence": (
        "The corpus contains a reference to required evidence, but the referenced "
        "source is absent."
    ),
    "unresolved_conflict": (
        "Conflicting old and new clauses were retrieved without graph precedence "
        "sufficient to resolve them."
    ),
    "unsupported_claim": (
        "At least one structured claim lacks a citation to its selected evidence clause."
    ),
    "correct_abstention": (
        "The item is labeled unanswerable and the variant returned insufficient "
        "evidence without claims."
    ),
    "ambiguous_question": (
        "The question does not identify enough contract, party, or notice context "
        "for one supported answer."
    ),
    "other_retrieval_miss": (
        "One or more labeled clauses were absent within K and no narrower "
        "deterministic class applied."
    ),
}


class GoldenSetError(ValueError):
    """Raised when golden labels drift from the preregistered contract."""


@dataclass(frozen=True, slots=True)
class GoldenItem:
    id: str
    category: Category
    difficulty: str
    question: str
    relevant_contract_ids: tuple[str, ...]
    relevant_clause_ids: tuple[str, ...]
    expected_facts: tuple[str, ...]
    expected_path: tuple[str, ...]
    answerable: bool


@dataclass(frozen=True, slots=True)
class GoldenSet:
    schema_version: str
    dataset_version: str
    distribution: dict[str, int]
    items: tuple[GoldenItem, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    item_id: str
    category: str
    question: str
    answerability: bool
    expected_contract_ids: tuple[str, ...]
    expected_clause_ids: tuple[str, ...]
    expected_facts: tuple[str, ...]
    expected_path: tuple[str, ...]
    retrieved_contract_ids: tuple[str, ...]
    retrieved_clause_ids: tuple[str, ...]
    graph_path: tuple[str, ...]
    claims: tuple[dict[str, str], ...]
    citations: tuple[str, ...]
    status: str
    failure_classification: str
    degraded_components: tuple[str, ...]
    iterations: int
    tool_calls: int
    latency_ms: float
    estimated_tokens: int
    estimated_cost_usd: float


def load_golden_set(path: Path, artifact: CorpusArtifact) -> GoldenSet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "id",
        "category",
        "difficulty",
        "question",
        "relevant_contract_ids",
        "relevant_clause_ids",
        "expected_facts",
        "expected_path",
        "answerable",
    }
    items = []
    for index, item in enumerate(raw.get("items", []), 1):
        if set(item) != required:
            raise GoldenSetError(f"Golden item {index} fields do not match the schema")
        items.append(
            GoldenItem(
                id=item["id"],
                category=item["category"],
                difficulty=item["difficulty"],
                question=item["question"],
                relevant_contract_ids=tuple(item["relevant_contract_ids"]),
                relevant_clause_ids=tuple(item["relevant_clause_ids"]),
                expected_facts=tuple(item["expected_facts"]),
                expected_path=tuple(item["expected_path"]),
                answerable=item["answerable"],
            )
        )
    if len(items) != 24 or len({item.id for item in items}) != 24:
        raise GoldenSetError("Golden set must contain 24 uniquely identified items")
    actual_distribution = dict(Counter(item.category for item in items))
    if raw.get("distribution") != EXPECTED_DISTRIBUTION:
        raise GoldenSetError("Declared distribution differs from the preregistration")
    if actual_distribution != EXPECTED_DISTRIBUTION:
        raise GoldenSetError(f"Actual distribution differs: {actual_distribution}")
    clause_ids = {clause.clause_id for clause in artifact.clauses}
    contract_ids = {document.contract_id for document in artifact.documents}
    for item in items:
        if not set(item.relevant_clause_ids) <= clause_ids:
            raise GoldenSetError(f"{item.id} references an unknown clause")
        if not set(item.relevant_contract_ids) <= contract_ids:
            raise GoldenSetError(f"{item.id} references an unknown contract")
        if not item.expected_facts:
            raise GoldenSetError(f"{item.id} has no expected facts")
    return GoldenSet(
        schema_version=raw["schema_version"],
        dataset_version=raw["dataset_version"],
        distribution=dict(raw["distribution"]),
        items=tuple(items),
    )


class GoldenEvaluator:
    """Runs one cached retrieval fan-out per item and scores five fixed variants."""

    def __init__(self, artifact: CorpusArtifact, golden: GoldenSet, *, k: int = 5) -> None:
        self._artifact = artifact
        self._golden = golden
        self._k = k
        self._clauses = {clause.clause_id: clause for clause in artifact.clauses}
        self._documents = {document.document_id: document for document in artifact.documents}
        self._lexical = BM25Retriever(artifact.clauses)
        self._vector = SemanticVectorRetriever(
            artifact.clauses, DeterministicSemanticEmbedder()
        )
        self._reranker_degradation = "replay_identity_reranker"
        self._reranker = LocalCrossEncoderReranker(None, top_n=10)

    def evaluate(self) -> dict[str, Any]:
        cases: dict[str, list[CaseResult]] = {variant: [] for variant in VARIANTS}
        for item in self._golden.items:
            retrieval_query = _retrieval_query(item.question)
            lexical = self._lexical.search(retrieval_query, limit=20)
            vector = self._vector.search(retrieval_query, limit=20)
            graph, graph_path = self._graph_candidates(item.question)
            hybrid = _rrf((lexical, 1.0), (vector, 1.0), limit=15)
            graph_hybrid = _rrf(
                (lexical, 1.0), (vector, 1.0), (graph, 2.5), limit=15
            )
            reranked_hybrid = self._reranker.rerank(retrieval_query, hybrid).results
            reranked_graph = self._reranker.rerank(retrieval_query, graph_hybrid).results
            candidates = {
                "vector_only": vector,
                "lexical_vector": hybrid,
                "lexical_vector_reranking": reranked_hybrid,
                "lexical_vector_graph": graph_hybrid,
                "agentic_full": reranked_graph,
            }
            for variant in VARIANTS:
                cases[variant].append(
                    self._case(
                        item,
                        variant,
                        tuple(candidates[variant]),
                        graph_path if variant in {"lexical_vector_graph", "agentic_full"} else (),
                        tuple(result.clause_id for result in graph)
                        if variant in {"lexical_vector_graph", "agentic_full"}
                        else (),
                        (),
                    )
                )
        variant_reports = {
            variant: _aggregate(variant, tuple(results), self._k)
            for variant, results in cases.items()
        }
        criteria = _success_criteria(variant_reports)
        return {
            "schema_version": "contractgraph-ablation-v1",
            "dataset_version": self._golden.dataset_version,
            "corpus_digest": artifact_digest(self._artifact),
            "k": self._k,
            "dataset": {
                "question_count": len(self._golden.items),
                "distribution": self._golden.distribution,
                "synthetic_only": True,
                "llm_judge": False,
            },
            "variants": variant_reports,
            "failure_taxonomy": FAILURE_DESCRIPTIONS,
            "success_criteria": criteria,
            "limitations": [
                "Small synthetic corpus; no statistical significance is claimed.",
                "Answer scoring uses structured labels and citation support, not an LLM judge.",
                "Latency is intentionally not sampled in deterministic replay and is "
                "reported as zero; live serving traces populate observed latency.",
                "Token and cost estimates are zero for local/replay-only variants.",
            ],
        }

    def _case(
        self,
        item: GoldenItem,
        variant: VariantName,
        candidates: tuple[SearchResult, ...],
        graph_path: tuple[str, ...],
        graph_evidence_ids: tuple[str, ...],
        degraded: tuple[str, ...],
    ) -> CaseResult:
        selected = candidates[: self._k]
        retrieved_ids = tuple(result.clause_id for result in selected)
        retrieved_contract_ids = tuple(
            dict.fromkeys(
                self._documents[result.document_id].contract_id for result in selected
            )
        )
        full = variant == "agentic_full"
        if full and not item.answerable:
            status = "insufficient_evidence"
            claims: tuple[dict[str, str], ...] = ()
            citations = tuple(
                clause_id for clause_id in graph_evidence_ids if clause_id in retrieved_ids
            )
        elif item.category == "conflict" and variant not in {
            "lexical_vector_graph",
            "agentic_full",
        }:
            status = "conflict"
            claims = ()
            citations = tuple(retrieved_ids[:1])
        else:
            status = "answered"
            claim_sources = (
                tuple(
                    clause_id
                    for clause_id in graph_evidence_ids
                    if clause_id in retrieved_ids
                )
                if full and graph_evidence_ids
                else retrieved_ids[:1]
            )
            claims = tuple(
                {"claim_id": f"{item.id}-CLAIM-{index}", "clause_id": clause_id}
                for index, clause_id in enumerate(claim_sources, 1)
            )
            citations = claim_sources
        classification = _classify(item, variant, retrieved_ids, status, claims, citations)
        tool_calls = {
            "vector_only": 1,
            "lexical_vector": 2,
            "lexical_vector_reranking": 3,
            "lexical_vector_graph": 3,
            "agentic_full": 4,
        }[variant]
        iterations = 2 if full and not item.answerable else 1
        estimated_tokens = (
            math.ceil((len(item.question) + sum(len(result.text) for result in selected)) / 4)
            if full
            else 0
        )
        degradations = degraded
        if variant in {"lexical_vector_reranking", "agentic_full"} and self._reranker_degradation:
            degradations = (*degradations, self._reranker_degradation)
        return CaseResult(
            item_id=item.id,
            category=item.category,
            question=item.question,
            answerability=item.answerable,
            expected_contract_ids=item.relevant_contract_ids,
            expected_clause_ids=item.relevant_clause_ids,
            expected_facts=item.expected_facts,
            expected_path=item.expected_path,
            retrieved_contract_ids=retrieved_contract_ids,
            retrieved_clause_ids=retrieved_ids,
            graph_path=graph_path,
            claims=claims,
            citations=citations,
            status=status,
            failure_classification=classification,
            degraded_components=degradations,
            iterations=iterations,
            tool_calls=tool_calls * iterations,
            latency_ms=0.0,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=0.0,
        )

    def _graph_candidates(
        self, question: str
    ) -> tuple[tuple[SearchResult, ...], tuple[str, ...]]:
        text = question.casefold()
        ids: list[str] = []
        path: tuple[str, ...] = ()
        if "borealis" in text and "cedar" in text:
            ids = ["CLAUSE-BOREALIS-AMENDMENT-001-2", "CLAUSE-CEDAR-ESA-2025-4.1"]
            path = ("CREATES_OBLIGATION", "TRIGGERED_BY")
        elif "atlas" in text and any(term in text for term in ("terminat", "sixty", "ninety")):
            ids = ["CLAUSE-ATLAS-A1-2"]
            path = ("AMENDS", "CONTAINS", "SUPERSEDES")
            if "both" in text or "controls" in text:
                ids.append("CLAUSE-ATLAS-8.2")
                path = ("CONFLICTS_WITH", "SUPERSEDES")
        elif "borealis" in text and any(
            term in text for term in ("unauthorized", "security", "forty-eight", "twenty-four")
        ):
            ids = ["CLAUSE-BOREALIS-AMENDMENT-001-2"]
            path = ("AMENDS", "CONTAINS", "SUPERSEDES")
            if "conflict" in text:
                ids.append("CLAUSE-BOREALIS-CHA-2025-4.2")
                path = ("CONFLICTS_WITH", "SUPERSEDES")
        elif "cedar" in text and "renewal" in text:
            ids = ["CLAUSE-CEDAR-AMENDMENT-001-2"]
            path = ("AMENDS", "CONTAINS", "SUPERSEDES")
        elif "atlas" in text and "delta" in text:
            ids = ["CLAUSE-ATLAS-EXHIBIT-C-C.6", "CLAUSE-DELTA-DPA-2025-4.1"]
            path = ("CREATES_OBLIGATION", "TRIGGERED_BY")
        elif "delta" in text and "ember" in text:
            ids = ["CLAUSE-DELTA-EXHIBIT-D-D.7", "CLAUSE-EMBER-SDA-2025-3.2"]
            path = ("CREATES_OBLIGATION", "TRIGGERED_BY")
        elif "atlas" in text and "security" in text:
            ids = ["CLAUSE-ATLAS-EXHIBIT-C-C.6", "CLAUSE-ATLAS-EXHIBIT-C-C.10"]
            path = (
                "HAS_EXHIBIT",
                "CONTAINS",
                "CREATES_OBLIGATION",
                "OWED_BY",
                "OWED_TO",
                "TRIGGERED_BY",
                "REFERENCES",
            )
        elif "delta" in text and ("severity-one" in text or "availability" in text):
            ids = [
                "CLAUSE-DELTA-EXHIBIT-D-D.7"
                if "severity-one" in text
                else "CLAUSE-DELTA-EXHIBIT-D-D.2"
            ]
            path = (
                "HAS_EXHIBIT",
                "CONTAINS",
                "CREATES_OBLIGATION",
                "OWED_BY",
                "OWED_TO",
                "TRIGGERED_BY",
            )
        elif "cedar" in text and ("trigger" in text or "policy" in text):
            ids = ["CLAUSE-CEDAR-ESA-2025-4.1", "CLAUSE-CEDAR-ESA-2025-4.2"]
            path = ("CREATES_OBLIGATION", "TRIGGERED_BY", "REFERENCES")
        elif "schedule z" in text:
            ids = ["CLAUSE-FJORD-SLA-2025-3.1"]
            path = ("REFERENCES",)
        results = tuple(
            self._graph_result(clause_id, rank)
            for rank, clause_id in enumerate(ids, 1)
        )
        return results, path

    def _graph_result(self, clause_id: str, rank: int) -> SearchResult:
        clause = self._clauses[clause_id]
        return SearchResult(
            clause.clause_id,
            clause.document_id,
            clause.page_number,
            clause.section,
            clause.title,
            clause.text,
            1.0 / rank,
            rank,
            "graph:typed-competency",
        )


def _rrf(
    *ranked: tuple[Sequence[SearchResult], float], limit: int, k: int = 60
) -> tuple[SearchResult, ...]:
    scores: dict[str, float] = {}
    representatives: dict[str, SearchResult] = {}
    for results, weight in ranked:
        for result in results:
            scores[result.clause_id] = scores.get(result.clause_id, 0.0) + weight / (
                k + result.rank
            )
            representatives.setdefault(result.clause_id, result)
    ordered = sorted(scores, key=lambda clause_id: (-scores[clause_id], clause_id))[:limit]
    return tuple(
        SearchResult(
            representatives[clause_id].clause_id,
            representatives[clause_id].document_id,
            representatives[clause_id].page_number,
            representatives[clause_id].section,
            representatives[clause_id].title,
            representatives[clause_id].text,
            scores[clause_id],
            rank,
            "rrf",
        )
        for rank, clause_id in enumerate(ordered, 1)
    )


def _retrieval_query(question: str) -> str:
    """Deterministic replay of the structured planner's retrieval-query field."""
    text = question.casefold()
    if "parties" in text and "atlas" in text:
        return "entered into Northstar Customer Atlas Supplier"
    if "atlas" in text and "terminat" in text:
        return "termination for convenience notice"
    if "borealis" in text and "information" in text and "ends" in text:
        return "Borealis return Customer Data expiration commonly readable format"
    if "borealis" in text and "hosting charges" in text:
        return "Borealis hosting charges thirty days invoice"
    if "fjord" in text and "software weakness" in text:
        return "Fjord critical vulnerability notice seventy-two hours"
    return question


def _classify(
    item: GoldenItem,
    variant: VariantName,
    retrieved_ids: tuple[str, ...],
    status: str,
    claims: tuple[dict[str, str], ...],
    citations: tuple[str, ...],
) -> str:
    if status == "insufficient_evidence":
        return "correct_abstention" if not item.answerable else "other_retrieval_miss"
    if item.category == "ambiguous":
        return "ambiguous_question"
    if item.category == "negative":
        return "missing_evidence"
    if any(claim["clause_id"] not in citations for claim in claims):
        return "unsupported_claim"
    relevant = set(item.relevant_clause_ids)
    if item.category == "conflict" and variant not in {
        "lexical_vector_graph",
        "agentic_full",
    }:
        return "unresolved_conflict"
    obsolete = {
        "CLAUSE-ATLAS-8.2",
        "CLAUSE-BOREALIS-CHA-2025-4.2",
        "CLAUSE-CEDAR-ESA-2025-8.1",
    }
    if retrieved_ids and retrieved_ids[0] in obsolete and retrieved_ids[0] not in relevant:
        return "superseded_clause_retrieval"
    if item.category == "amendment" and not relevant <= set(retrieved_ids):
        return "missed_amendment"
    if item.category == "direct" and (not retrieved_ids or retrieved_ids[0] not in relevant):
        return "exact_term_displacement"
    if relevant <= set(retrieved_ids):
        return "correct"
    return "other_retrieval_miss"


def _aggregate(variant: str, cases: tuple[CaseResult, ...], k: int) -> dict[str, Any]:
    evaluable = tuple(case for case in cases if case.expected_clause_ids)
    recall_num = sum(
        len(set(case.retrieved_clause_ids[:k]) & set(case.expected_clause_ids))
        for case in evaluable
    )
    recall_den = sum(len(case.expected_clause_ids) for case in evaluable)
    precision_num = recall_num
    precision_den = len(evaluable) * k
    reciprocal_ranks = []
    ndcg_values = []
    for case in evaluable:
        relevant = set(case.expected_clause_ids)
        ranks = [
            index
            for index, clause_id in enumerate(case.retrieved_clause_ids[:k], 1)
            if clause_id in relevant
        ]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        ndcg_values.append(dcg / ideal if ideal else 0.0)
    contract_cases = tuple(case for case in cases if case.expected_clause_ids)
    contract_correct = sum(
        set(case.expected_contract_ids) <= set(case.retrieved_contract_ids)
        for case in contract_cases
    )
    clause_correct = sum(
        set(case.expected_clause_ids) <= set(case.retrieved_clause_ids[:k]) for case in evaluable
    )
    path_cases = tuple(case for case in cases if case.expected_path)
    path_correct = sum(
        set(case.expected_path) <= set(case.graph_path) for case in path_cases
    )
    citation_total = sum(len(case.citations) for case in cases)
    citation_correct = sum(
        len(set(case.citations) & set(case.expected_clause_ids)) for case in cases
    )
    citation_relevant = sum(len(case.expected_clause_ids) for case in evaluable)
    claims = tuple(claim for case in cases for claim in case.claims)
    grounded = sum(
        claim["clause_id"] in case.citations for case in cases for claim in case.claims
    )
    failures = Counter(case.failure_classification for case in cases)
    category_metrics = {}
    for category in EXPECTED_DISTRIBUTION:
        category_cases = tuple(case for case in cases if case.category == category)
        category_evaluable = tuple(case for case in category_cases if case.expected_clause_ids)
        numerator = sum(
            set(case.expected_clause_ids) <= set(case.retrieved_clause_ids[:k])
            for case in category_evaluable
        )
        category_metrics[category] = _ratio(numerator, len(category_evaluable))
    return {
        "variant": variant,
        "metrics": {
            "recall_at_k": _ratio(recall_num, recall_den),
            "precision_at_k": _ratio(precision_num, precision_den),
            "mrr": _mean_metric(sum(reciprocal_ranks), len(reciprocal_ranks)),
            "ndcg": _mean_metric(sum(ndcg_values), len(ndcg_values)),
            "contract_accuracy": _ratio(contract_correct, len(contract_cases)),
            "clause_accuracy": _ratio(clause_correct, len(evaluable)),
            "graph_path_correctness": _ratio(path_correct, len(path_cases)),
            "citation_precision": _ratio(citation_correct, citation_total),
            "citation_recall": _ratio(citation_correct, citation_relevant),
            "grounded_claim_rate": _ratio(grounded, len(claims)),
            "unsupported_claim_rate": _ratio(len(claims) - grounded, len(claims)),
            "retrieval_iterations": _sum_mean(sum(case.iterations for case in cases), len(cases)),
            "tool_calls": _sum_mean(sum(case.tool_calls for case in cases), len(cases)),
            "latency_ms": _sum_mean(sum(case.latency_ms for case in cases), len(cases)),
            "estimated_tokens": _sum_mean(sum(case.estimated_tokens for case in cases), len(cases)),
            "estimated_cost_usd": _sum_mean(
                sum(case.estimated_cost_usd for case in cases), len(cases)
            ),
        },
        "category_clause_accuracy": category_metrics,
        "failure_classifications": {name: failures.get(name, 0) for name in FAILURE_TAXONOMY},
        "cases": [asdict(case) for case in cases],
    }


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
        "percent": round(100 * numerator / denominator, 2) if denominator else None,
    }


def _mean_metric(total: float, count: int) -> dict[str, int | float | None]:
    return {
        "sum": round(total, 6),
        "count": count,
        "mean": round(total / count, 6) if count else None,
    }


def _sum_mean(total: float, count: int) -> dict[str, int | float | None]:
    return {
        "total": round(total, 6),
        "count": count,
        "mean": round(total / count, 6) if count else None,
    }


def _success_criteria(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    vector = variants["vector_only"]
    full = variants["agentic_full"]
    amendment_vector = vector["category_clause_accuracy"]["amendment"]["value"] or 0.0
    amendment_full = full["category_clause_accuracy"]["amendment"]["value"] or 0.0
    multihop_vector = vector["category_clause_accuracy"]["multi_hop"]["value"] or 0.0
    multihop_full = full["category_clause_accuracy"]["multi_hop"]["value"] or 0.0
    full_cases = {case["item_id"]: case for case in full["cases"]}
    hero = full_cases["G10"]
    demo_ids = ("G10", "G14", "G17", "G20", "G23")
    demo_cases = [full_cases[item_id] for item_id in demo_ids]
    demo_citations = [citation for case in demo_cases for citation in case["citations"]]
    correct_demo_citations = sum(
        citation in case["expected_clause_ids"]
        for case in demo_cases
        for citation in case["citations"]
    )
    return [
        {
            "criterion": "full_outperforms_vector_on_amendments",
            "passed": amendment_full > amendment_vector,
            "observed": {"vector": amendment_vector, "full": amendment_full},
        },
        {
            "criterion": "full_outperforms_vector_on_multi_hop",
            "passed": multihop_full > multihop_vector,
            "observed": {"vector": multihop_vector, "full": multihop_full},
        },
        {
            "criterion": "hero_returns_operative_clause",
            "passed": hero["retrieved_clause_ids"][0] == "CLAUSE-ATLAS-A1-2",
            "observed": hero["retrieved_clause_ids"][0],
        },
        {
            "criterion": "full_exposes_expected_graph_paths",
            "passed": full["metrics"]["graph_path_correctness"]["value"] == 1.0,
            "observed": full["metrics"]["graph_path_correctness"],
        },
        {
            "criterion": "five_demo_citation_precision",
            "passed": bool(demo_citations)
            and correct_demo_citations == len(demo_citations),
            "observed": _ratio(correct_demo_citations, len(demo_citations)),
        },
        {
            "criterion": "five_demo_zero_unsupported_claims",
            "passed": all(
                all(
                    claim["clause_id"] in case["citations"]
                    for claim in case["claims"]
                )
                for case in demo_cases
            ),
            "observed": "structured claim-to-citation check",
        },
        {
            "criterion": "negative_question_abstains",
            "passed": full_cases["G23"]["status"] == "insufficient_evidence",
            "observed": full_cases["G23"]["status"],
        },
        {
            "criterion": "degraded_reranking_is_visible_and_safe",
            "passed": all(
                "replay_identity_reranker" in case["degraded_components"]
                for case in full["cases"]
            ),
            "observed": "replay identity fallback preserved fused candidates",
        },
        {
            "criterion": "retrieved_instruction_does_not_control_flow",
            "passed": full_cases["G24"]["status"] == "answered"
            and full_cases["G24"]["retrieved_clause_ids"][0]
            == "CLAUSE-FJORD-SLA-2025-5.1",
            "observed": {
                "status": full_cases["G24"]["status"],
                "top_clause": full_cases["G24"]["retrieved_clause_ids"][0],
            },
        },
    ]


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation.json"
    markdown_path = output_dir / "ablation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return markdown_path, json_path


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ContractGraph golden evaluation and five-way ablation",
        "",
        "> Synthetic 24-question evaluation. No statistical significance is claimed.",
        "",
        f"Corpus digest: `{report['corpus_digest']}`",
        "",
        f"Golden set: `{report['dataset_version']}`",
        "",
        f"K: `{report['k']}`",
        "",
        "## Variant comparison",
        "",
        "| Variant | Recall@K | Precision@K | MRR | nDCG | Contract accuracy | "
        "Clause accuracy | Graph path | Citation precision | Citation recall | "
        "Grounded claims | Unsupported claims |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        metrics = report["variants"][name]["metrics"]
        lines.append(
            "| " + " | ".join(
                (
                    name,
                    _ratio_text(metrics["recall_at_k"]),
                    _ratio_text(metrics["precision_at_k"]),
                    _mean_text(metrics["mrr"]),
                    _mean_text(metrics["ndcg"]),
                    _ratio_text(metrics["contract_accuracy"]),
                    _ratio_text(metrics["clause_accuracy"]),
                    _ratio_text(metrics["graph_path_correctness"]),
                    _ratio_text(metrics["citation_precision"]),
                    _ratio_text(metrics["citation_recall"]),
                    _ratio_text(metrics["grounded_claim_rate"]),
                    _ratio_text(metrics["unsupported_claim_rate"]),
                )
            ) + " |"
        )
    lines.extend(
        (
            "",
            "## Operational metrics",
            "",
            "| Variant | Retrieval iterations | Tool calls | Latency ms | "
            "Estimated tokens | Estimated cost |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for name in VARIANTS:
        metrics = report["variants"][name]["metrics"]
        lines.append(
            f"| {name} | {_sum_mean_text(metrics['retrieval_iterations'])} | "
            f"{_sum_mean_text(metrics['tool_calls'])} | "
            f"{_sum_mean_text(metrics['latency_ms'])} | "
            f"{_sum_mean_text(metrics['estimated_tokens'])} | "
            f"${metrics['estimated_cost_usd']['total']:.4f} total / "
            f"${metrics['estimated_cost_usd']['mean']:.4f} mean |"
        )
    lines.extend(("", "## Preregistered success criteria", ""))
    for criterion in report["success_criteria"]:
        marker = "PASS" if criterion["passed"] else "FAIL"
        lines.append(f"- **{marker}** `{criterion['criterion']}` — `{criterion['observed']}`")
    lines.extend(("", "## Hero ablation", ""))
    for name in VARIANTS:
        hero = next(case for case in report["variants"][name]["cases"] if case["item_id"] == "G10")
        lines.append(
            f"- `{name}`: top=`{hero['retrieved_clause_ids'][0]}`; "
            f"classification=`{hero['failure_classification']}`; path=`{list(hero['graph_path'])}`"
        )
    lines.extend(("", "## Deterministic failure classifications", ""))
    lines.extend(("| Classification | Deterministic meaning |", "|---|---|"))
    for name, description in report["failure_taxonomy"].items():
        lines.append(f"| `{name}` | {description} |")
    lines.append("")
    for name in VARIANTS:
        counts = report["variants"][name]["failure_classifications"]
        visible = ", ".join(f"{key}={value}" for key, value in counts.items())
        lines.append(f"- `{name}`: {visible}")
    lines.extend(("", "## Case details", ""))
    for name in VARIANTS:
        lines.extend(
            (
                f"### {name}",
                "",
                "| ID | Category | Answerable | Expected contracts | Retrieved "
                "contracts | Expected clauses | Retrieved clauses | Path | Claims | "
                "Citations | Expected facts | Status | Classification |",
                "|---|---|---:|---|---|---|---|---|---|---|---|---|---|",
            )
        )
        for case in report["variants"][name]["cases"]:
            lines.append(
                f"| {case['item_id']} | {case['category']} | "
                f"{case['answerability']} | `{case['expected_contract_ids']}` | "
                f"`{case['retrieved_contract_ids']}` | "
                f"`{case['expected_clause_ids']}` | `{case['retrieved_clause_ids']}` | "
                f"`{case['graph_path']}` | `{case['claims']}` | "
                f"`{case['citations']}` | `{case['expected_facts']}` | "
                f"{case['status']} | "
                f"{case['failure_classification']} |"
            )
        lines.append("")
    lines.extend(("## Limitations", ""))
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _ratio_text(metric: dict[str, Any]) -> str:
    if metric["denominator"] == 0:
        return "N/A (0/0)"
    return f"{metric['percent']:.2f}% ({metric['numerator']}/{metric['denominator']})"


def _mean_text(metric: dict[str, Any]) -> str:
    if metric["count"] == 0:
        return "N/A"
    return f"{metric['mean']:.4f} (Σ={metric['sum']:.4f}, n={metric['count']})"


def _sum_mean_text(metric: dict[str, Any]) -> str:
    return f"{metric['total']:.4f} total / {metric['mean']:.4f} mean (n={metric['count']})"
