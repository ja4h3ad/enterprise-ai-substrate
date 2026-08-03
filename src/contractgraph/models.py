"""Small, explicit domain records for the first ContractGraph slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DocumentType = Literal["Contract", "Amendment", "Exhibit"]
PopulationMethod = Literal["document_structure", "reviewed_assertion"]

EntityType = Literal[
    "Party",
    "Obligation",
    "Event",
    "Policy",
    "ProductOrService",
    "MissingReference",
]


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
class Entity:
    entity_id: str
    entity_type: EntityType
    name: str


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
    entities: tuple[Entity, ...]
    triples: tuple[Triple, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    clause_id: str
    document_id: str
    page_number: int
    section: str
    title: str
    text: str
    score: float
    rank: int
    retriever: str

    @property
    def citation(self) -> str:
        return (
            f"{self.document_id}, p.{self.page_number}, "
            f"§{self.section}, {self.clause_id}"
        )


@dataclass(frozen=True, slots=True)
class GraphStep:
    from_id: str
    predicate: str
    to_id: str
    traversal: Literal["forward", "reverse"]
    source_clause_id: str


@dataclass(frozen=True, slots=True)
class OperativeClauseResolution:
    contract_id: str
    base_clause_id: str
    operative_clause_id: str
    path: tuple[GraphStep, ...]


@dataclass(frozen=True, slots=True)
class ObligationResolution:
    obligation_id: str
    clause_id: str
    owed_by: str
    owed_to: str | None
    triggered_by: str | None
    policy_ids: tuple[str, ...]
    path: tuple[GraphStep, ...]


@dataclass(frozen=True, slots=True)
class ContractComparison:
    contract_ids: tuple[str, ...]
    clause_ids: tuple[str, ...]
    shared_service_type: str | None


@dataclass(frozen=True, slots=True)
class ProvenanceTrace:
    clause_id: str
    document_id: str
    page_id: str
    chunk_id: str
    path: tuple[GraphStep, ...]
