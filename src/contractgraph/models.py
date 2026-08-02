"""Small, explicit domain records for the first ContractGraph slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DocumentType = Literal["Contract", "Amendment"]
PopulationMethod = Literal["document_structure", "reviewed_assertion"]


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    document_type: DocumentType
    title: str
    effective_date: str
    contract_id: str
    amends_contract_id: str | None
    source_path: str


@dataclass(frozen=True, slots=True)
class Page:
    page_id: str
    document_id: str
    page_number: int


@dataclass(frozen=True, slots=True)
class Clause:
    clause_id: str
    document_id: str
    page_id: str
    page_number: int
    section: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    clause_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source_clause_id: str
    population_method: PopulationMethod


@dataclass(frozen=True, slots=True)
class CorpusArtifact:
    schema_version: str
    corpus_version: str
    documents: tuple[Document, ...]
    pages: tuple[Page, ...]
    clauses: tuple[Clause, ...]
    chunks: tuple[Chunk, ...]
    triples: tuple[Triple, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
