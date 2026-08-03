"""Deterministic ingestion and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

from contractgraph.models import CorpusArtifact, Entity, Triple
from contractgraph.parser import parse_document

SCHEMA_VERSION = "contractgraph-provenance-v1"
ACTIVE_ENTITY_TYPES = {
    "Party",
    "Obligation",
    "Event",
    "Policy",
    "ProductOrService",
    "MissingReference",
}
ACTIVE_PREDICATES = {
    "REPRESENTS",
    "CONTAINS",
    "HAS_EXHIBIT",
    "AMENDS",
    "MODIFIES",
    "SUPERSEDES",
    "CREATES_OBLIGATION",
    "OWED_BY",
    "OWED_TO",
    "TRIGGERED_BY",
    "REFERENCES",
    "CONFLICTS_WITH",
    "COVERS",
    "LOCATED_ON",
    "EXTRACTED_FROM",
}


class IngestionError(ValueError):
    """Raised when source records cannot produce a valid deterministic corpus."""


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise IngestionError("Corpus manifest must be a JSON object")
    return manifest


def _triple(raw: dict[str, Any]) -> Triple:
    required = {field.name for field in fields(Triple)}
    if set(raw) != required:
        raise IngestionError(
            f"Triple fields must be exactly {sorted(required)}; received {sorted(raw)}"
        )
    if raw["population_method"] not in {"document_structure", "reviewed_assertion"}:
        raise IngestionError(f"Unsupported population method: {raw['population_method']}")
    if raw["predicate"] not in ACTIVE_PREDICATES:
        raise IngestionError(f"Predicate is outside the active ontology: {raw['predicate']}")
    return Triple(**raw)


def build_corpus(corpus_root: Path) -> CorpusArtifact:
    manifest = _load_manifest(corpus_root / "manifest.json")
    corpus_version = str(manifest.get("corpus_version", ""))
    document_paths = manifest.get("documents")
    assertion_path = manifest.get("assertions")
    entity_path = manifest.get("entities")
    if (
        not corpus_version
        or not isinstance(document_paths, list)
        or not assertion_path
        or not entity_path
    ):
        raise IngestionError(
            "Manifest requires corpus_version, documents, entities, and assertions"
        )

    parsed = [
        parse_document(corpus_root / relative_path, corpus_root=corpus_root)
        for relative_path in sorted(document_paths)
    ]
    documents = tuple(sorted((item.document for item in parsed), key=lambda item: item.document_id))
    pages = tuple(
        sorted(
            (page for item in parsed for page in item.pages),
            key=lambda item: item.page_id,
        )
    )
    # Canonical document paths and source order make the human-facing hierarchy
    # deterministic without sorting section identifiers lexicographically (where
    # section 12 would otherwise precede section 2).
    clauses = tuple(clause for item in parsed for clause in item.clauses)
    chunks = tuple(chunk for item in parsed for chunk in item.chunks)
    raw_entities = json.loads((corpus_root / entity_path).read_text(encoding="utf-8"))
    if not isinstance(raw_entities, list):
        raise IngestionError("Entity file must contain a JSON list")
    entities = tuple(
        sorted((Entity(**raw) for raw in raw_entities), key=lambda item: item.entity_id)
    )
    unsupported_entity_types = {
        entity.entity_type for entity in entities if entity.entity_type not in ACTIVE_ENTITY_TYPES
    }
    if unsupported_entity_types:
        raise IngestionError(
            f"Entity types are outside the active ontology: {sorted(unsupported_entity_types)}"
        )

    clause_ids = {clause.clause_id for clause in clauses}
    entity_ids = (
        {document.document_id for document in documents}
        | {document.contract_id for document in documents}
        | {page.page_id for page in pages}
        | clause_ids
        | {chunk.chunk_id for chunk in chunks}
        | {entity.entity_id for entity in entities}
    )
    assertions = json.loads((corpus_root / assertion_path).read_text(encoding="utf-8"))
    if not isinstance(assertions, list):
        raise IngestionError("Assertions file must contain a JSON list")
    triples = tuple(sorted((_triple(raw) for raw in assertions), key=_triple_sort_key))
    for triple in triples:
        if triple.source_clause_id not in clause_ids:
            raise IngestionError(f"Unknown triple source clause: {triple.source_clause_id}")
        if triple.subject not in entity_ids or triple.object not in entity_ids:
            raise IngestionError(
                "Triple references unknown entity: "
                f"{triple.subject} {triple.predicate} {triple.object}"
            )

    _reject_duplicates(documents, "document_id")
    _reject_duplicates(pages, "page_id")
    _reject_duplicates(clauses, "clause_id")
    _reject_duplicates(chunks, "chunk_id")
    _reject_duplicates(entities, "entity_id")
    if len(triples) != len(set(triples)):
        raise IngestionError("Duplicate triples are not allowed")

    return CorpusArtifact(
        schema_version=SCHEMA_VERSION,
        corpus_version=corpus_version,
        documents=documents,
        pages=pages,
        clauses=clauses,
        chunks=chunks,
        entities=entities,
        triples=triples,
    )


def _triple_sort_key(triple: Triple) -> tuple[str, str, str, str, str]:
    return (
        triple.subject,
        triple.predicate,
        triple.object,
        triple.source_clause_id,
        triple.population_method,
    )


def _reject_duplicates(records: tuple[Any, ...], identifier: str) -> None:
    values = [getattr(record, identifier) for record in records]
    if len(values) != len(set(values)):
        raise IngestionError(f"Duplicate {identifier} values are not allowed")


def canonical_json(artifact: CorpusArtifact) -> str:
    return json.dumps(artifact.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def artifact_digest(artifact: CorpusArtifact) -> str:
    return hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()


def persist_corpus(artifact: CorpusArtifact, artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    corpus_text = canonical_json(artifact)
    triples_text = "".join(
        json.dumps(
            {
                "object": triple.object,
                "population_method": triple.population_method,
                "predicate": triple.predicate,
                "source_clause_id": triple.source_clause_id,
                "subject": triple.subject,
            },
            sort_keys=True,
        )
        + "\n"
        for triple in artifact.triples
    )
    _atomic_write(artifact_root / "corpus.json", corpus_text)
    _atomic_write(artifact_root / "triples.jsonl", triples_text)
    _atomic_write(artifact_root / "sha256.txt", artifact_digest(artifact) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
