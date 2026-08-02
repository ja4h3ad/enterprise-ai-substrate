from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
CORPUS_ROOT = PROJECT_ROOT / "corpus"


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
    }
    assert {triple["population_method"] for triple in artifact["triples"]} == {
        "document_structure",
        "reviewed_assertion",
    }
    assert all(triple["source_clause_id"] for triple in artifact["triples"])

    inspection = _run_cli("inspect", "--artifacts", str(artifact_root)).stdout
    assert "DOC-ATLAS-AMENDMENT-001 [Amendment] -> amends CONTRACT-ATLAS-001" in inspection
    assert "CLAUSE-ATLAS-A1-2 --SUPERSEDES--> CLAUSE-ATLAS-8.2" in inspection
    assert "source=CLAUSE-ATLAS-A1-2; method=reviewed_assertion" in inspection
