from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from contractgraph.application import ContractGraphApplication
from contractgraph.comparison import HERO_QUESTION
from contractgraph.ingestion import build_corpus
from contractgraph.retrieval import BM25Retriever

PROJECT_ROOT = Path(__file__).parents[1]
CORPUS_ROOT = PROJECT_ROOT / "corpus"
REPLAY_FIXTURE = PROJECT_ROOT / "replay" / "hero.json"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "contractgraph.cli", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_ingestion_is_deterministic_and_inspectable(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    first = _run_cli(
        "ingest",
        "--corpus",
        str(CORPUS_ROOT),
        "--artifacts",
        str(artifact_root),
    )
    first_corpus = (artifact_root / "corpus.json").read_bytes()
    first_triples = (artifact_root / "triples.jsonl").read_bytes()
    first_digest = (artifact_root / "sha256.txt").read_text(encoding="utf-8").strip()

    second = _run_cli(
        "ingest",
        "--corpus",
        str(CORPUS_ROOT),
        "--artifacts",
        str(artifact_root),
    )

    assert first.stdout == second.stdout
    assert first_corpus == (artifact_root / "corpus.json").read_bytes()
    assert first_triples == (artifact_root / "triples.jsonl").read_bytes()
    assert first_digest == hashlib.sha256(first_corpus).hexdigest()
    assert first_digest == (artifact_root / "sha256.txt").read_text(encoding="utf-8").strip()

    artifact = json.loads(first_corpus)
    assert {document["document_type"] for document in artifact["documents"]} == {
        "Contract",
        "Amendment",
        "Exhibit",
    }
    assert len(artifact["documents"]) == 11
    assert len(artifact["clauses"]) == 100
    assert sum(item["document_type"] == "Contract" for item in artifact["documents"]) == 6
    assert sum(item["document_type"] == "Amendment" for item in artifact["documents"]) == 3
    assert sum(item["document_type"] == "Exhibit" for item in artifact["documents"]) == 2
    assert {triple["population_method"] for triple in artifact["triples"]} == {
        "document_structure",
        "reviewed_assertion",
    }
    assert all(triple["source_clause_id"] for triple in artifact["triples"])

    inspection = _run_cli("inspect", "--artifacts", str(artifact_root)).stdout
    assert "DOC-ATLAS-AMENDMENT-001 [Amendment] -> amends CONTRACT-ATLAS-001" in inspection
    assert "CLAUSE-ATLAS-A1-2 --SUPERSEDES--> CLAUSE-ATLAS-8.2" in inspection
    assert "source=CLAUSE-ATLAS-A1-2; method=reviewed_assertion" in inspection


def test_public_comparison_exposes_obsolete_and_operative_clauses(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _run_cli(
        "ingest",
        "--corpus",
        str(CORPUS_ROOT),
        "--artifacts",
        str(artifact_root),
    )

    comparison = _run_cli("compare", "--artifacts", str(artifact_root)).stdout

    assert "Deterministic retrieval query: termination for convenience notice" in comparison
    assert "Lexical retrieval (top 3)\n1. CLAUSE-ATLAS-8.2" in comparison
    assert "Vector-only baseline\n1. CLAUSE-ATLAS-8.2" in comparison
    assert "sixty (60) days" in comparison
    assert "Outcome: INCORRECT" in comparison
    assert "Graph-grounded resolution" in comparison
    assert "Operative clause: CLAUSE-ATLAS-A1-2" in comparison
    assert "ninety (90) days" in comparison
    assert "Outcome: CORRECT" in comparison
    assert "CONTRACT-ATLAS-001 <--AMENDS-- DOC-ATLAS-AMENDMENT-001" in comparison
    assert "DOC-ATLAS-AMENDMENT-001 --CONTAINS-- CLAUSE-ATLAS-A1-2" in comparison
    assert "CLAUSE-ATLAS-A1-2 --SUPERSEDES-- CLAUSE-ATLAS-8.2" in comparison


def test_lexical_retrieval_preserves_exact_contract_language() -> None:
    retriever = BM25Retriever(build_corpus(CORPUS_ROOT).clauses)

    assert retriever.search("Atlas Network Services", limit=1)[0].clause_id == (
        "CLAUSE-ATLAS-A1-1"
    )
    result = retriever.search("entered into Northstar Customer Atlas Supplier", limit=1)
    assert result[0].clause_id == "CLAUSE-ATLAS-1.1"
    assert retriever.search("Termination for Convenience", limit=1)[0].clause_id == (
        "CLAUSE-ATLAS-8.2"
    )
    assert retriever.search("sixty 60 days", limit=1)[0].clause_id == "CLAUSE-ATLAS-8.2"


def test_application_runs_bounded_replay_workflow_and_persists_trace(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    state_db = tmp_path / "state.db"
    _run_cli(
        "ingest",
        "--corpus",
        str(CORPUS_ROOT),
        "--artifacts",
        str(artifact_root),
    )

    with ContractGraphApplication(
        artifact_root=artifact_root,
        state_db=state_db,
        replay_fixture=REPLAY_FIXTURE,
    ) as application:
        result = application.run(HERO_QUESTION)
        persisted = application.read_persisted_run(result.run_id)
        checkpoints = application.checkpoint_history(result.run_id)

    assert result.status == "answered"
    assert result.answer == (
        "Atlas Network Services must provide at least ninety (90) days' prior written "
        "notice before terminating for convenience."
    )
    assert result.iterations == 1
    assert result.model_calls == 2
    assert result.limits.max_retrieval_iterations == 2
    assert result.limits.max_graph_depth == 3
    assert result.fused_candidates[0].clause_id == "CLAUSE-ATLAS-A1-2"
    assert result.claims[0].evidence_ids == ("E-CLAUSE-ATLAS-A1-2",)
    assert result.citations[0].clause_id == "CLAUSE-ATLAS-A1-2"
    assert result.citations[0].page_number == 1
    assert result.citations[0].section == "2"
    assert [step.predicate for step in result.graph_paths] == [
        "AMENDS",
        "CONTAINS",
        "SUPERSEDES",
    ]
    trace_nodes = {event.node for event in result.trace_events}
    assert {
        "analyze_and_plan",
        "lexical_retrieve",
        "vector_retrieve",
        "graph_retrieve",
        "fuse_candidates",
        "assess_evidence",
        "synthesize_answer",
        "verify_citations",
        "finalize_trace",
    } <= trace_nodes
    assert persisted["run"]["status"] == "answered"
    assert len(persisted["events"]) == len(result.trace_events)
    assert len(checkpoints) >= 6


def test_application_recovery_is_bounded_before_abstention(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    state_db = tmp_path / "state.db"
    fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    fixture["plan"]["base_clause_id"] = "CLAUSE-ATLAS-8.1"
    replay_fixture = tmp_path / "insufficient.json"
    replay_fixture.write_text(json.dumps(fixture), encoding="utf-8")
    _run_cli(
        "ingest",
        "--corpus",
        str(CORPUS_ROOT),
        "--artifacts",
        str(artifact_root),
    )

    with ContractGraphApplication(
        artifact_root=artifact_root,
        state_db=state_db,
        replay_fixture=replay_fixture,
    ) as application:
        result = application.run(HERO_QUESTION)

    assert result.status == "insufficient_evidence"
    assert result.answer is None
    assert result.iterations == 2
    assert result.model_calls == 2
    assert sum(event.node == "reformulate_query" for event in result.trace_events) == 1
    assert "missing:applicable_amendment" in result.uncertainty_reasons
    assert "missing:supersession_path" in result.uncertainty_reasons


def test_application_rejects_claims_with_unknown_evidence(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    state_db = tmp_path / "state.db"
    fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
    fixture["synthesis"]["claims"][0]["evidence_ids"] = ["E-NOT-RETRIEVED"]
    replay_fixture = tmp_path / "unsupported.json"
    replay_fixture.write_text(json.dumps(fixture), encoding="utf-8")
    _run_cli(
        "ingest",
        "--corpus",
        str(CORPUS_ROOT),
        "--artifacts",
        str(artifact_root),
    )

    with ContractGraphApplication(
        artifact_root=artifact_root,
        state_db=state_db,
        replay_fixture=replay_fixture,
    ) as application:
        result = application.run(HERO_QUESTION)

    assert result.status == "error"
    assert result.citations == ()
    assert result.uncertainty_reasons == (
        "unknown_evidence:CLAIM-001:E-NOT-RETRIEVED",
    )
