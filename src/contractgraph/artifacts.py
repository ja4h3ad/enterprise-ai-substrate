"""Load persisted ContractGraph artifacts through typed domain records."""

from __future__ import annotations

import json
from pathlib import Path

from contractgraph.models import Clause, Chunk, CorpusArtifact, Document, Page, Triple


def load_corpus(artifact_root: Path) -> CorpusArtifact:
    artifact_path = artifact_root / "corpus.json"
    if not artifact_path.exists():
        raise FileNotFoundError("No corpus artifact found; run `contractgraph ingest` first")
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    return CorpusArtifact(
        schema_version=raw["schema_version"],
        corpus_version=raw["corpus_version"],
        documents=tuple(Document(**item) for item in raw["documents"]),
        pages=tuple(Page(**item) for item in raw["pages"]),
        clauses=tuple(Clause(**item) for item in raw["clauses"]),
        chunks=tuple(Chunk(**item) for item in raw["chunks"]),
        triples=tuple(Triple(**item) for item in raw["triples"]),
    )
