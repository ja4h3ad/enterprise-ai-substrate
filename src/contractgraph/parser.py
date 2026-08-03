"""Parser for ContractGraph's deliberately constrained canonical Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from contractgraph.models import Clause, Chunk, Document, Page

_FRONT_MATTER = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
_PAGE = re.compile(r"<!--\s*page:\s*(?P<number>\d+)\s*-->")
_CLAUSE = re.compile(
    r"<!--\s*clause:\s*\n(?P<meta>.*?)\n-->\s*\n"
    r"(?P<text>.*?)\n<!--\s*/clause\s*-->",
    re.DOTALL,
)


class DocumentFormatError(ValueError):
    """Raised when a canonical source document violates its explicit format."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    document: Document
    pages: tuple[Page, ...]
    clauses: tuple[Clause, ...]
    chunks: tuple[Chunk, ...]


def _key_values(raw: str, *, context: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise DocumentFormatError(f"Invalid {context} metadata line: {line!r}")
        normalized_key = key.strip()
        if normalized_key in values:
            raise DocumentFormatError(f"Duplicate {context} key: {normalized_key}")
        values[normalized_key] = value.strip()
    return values


def _required(values: dict[str, str], keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise DocumentFormatError(f"Missing {context} keys: {', '.join(missing)}")


def parse_document(path: Path, *, corpus_root: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    front_matter = _FRONT_MATTER.search(raw)
    if front_matter is None:
        raise DocumentFormatError(f"{path} has no YAML-style front matter")

    metadata = _key_values(front_matter.group("body"), context="document")
    _required(
        metadata,
        ("document_id", "document_type", "title", "effective_date", "contract_id"),
        context="document",
    )
    document_type = metadata["document_type"]
    if document_type not in {"Contract", "Amendment", "Exhibit"}:
        raise DocumentFormatError(f"Unsupported document_type: {document_type}")

    document = Document(
        document_id=metadata["document_id"],
        document_type=document_type,  # type: ignore[arg-type]
        title=metadata["title"],
        effective_date=metadata["effective_date"],
        contract_id=metadata["contract_id"],
        amends_contract_id=metadata.get("amends_contract_id"),
        source_path=path.relative_to(corpus_root).as_posix(),
    )
    if document.document_type == "Amendment" and not document.amends_contract_id:
        raise DocumentFormatError("Amendment must declare amends_contract_id")

    page_markers = list(_PAGE.finditer(raw))
    if not page_markers:
        raise DocumentFormatError(f"{path} has no explicit page markers")
    page_numbers = [int(match.group("number")) for match in page_markers]
    if page_numbers != sorted(set(page_numbers)):
        raise DocumentFormatError(f"{path} page numbers must be unique and increasing")

    pages = tuple(
        Page(
            page_id=f"{document.document_id}:PAGE:{page_number:03d}",
            document_id=document.document_id,
            page_number=page_number,
        )
        for page_number in page_numbers
    )
    clauses: list[Clause] = []
    chunks: list[Chunk] = []
    seen_clause_ids: set[str] = set()
    for clause_match in _CLAUSE.finditer(raw):
        clause_metadata = _key_values(clause_match.group("meta"), context="clause")
        _required(clause_metadata, ("clause_id", "section", "title"), context="clause")
        clause_id = clause_metadata["clause_id"]
        if clause_id in seen_clause_ids:
            raise DocumentFormatError(f"Duplicate clause_id: {clause_id}")
        seen_clause_ids.add(clause_id)

        preceding_pages = [
            marker for marker in page_markers if marker.start() < clause_match.start()
        ]
        if not preceding_pages:
            raise DocumentFormatError(f"Clause {clause_id} precedes the first page marker")
        page_number = int(preceding_pages[-1].group("number"))
        text = " ".join(clause_match.group("text").split())
        if not text:
            raise DocumentFormatError(f"Clause {clause_id} has no text")
        page_id = f"{document.document_id}:PAGE:{page_number:03d}"
        clauses.append(
            Clause(
                clause_id=clause_id,
                document_id=document.document_id,
                page_id=page_id,
                page_number=page_number,
                section=clause_metadata["section"],
                title=clause_metadata["title"],
                text=text,
            )
        )
        chunks.append(Chunk(chunk_id=f"{clause_id}:CHUNK:001", clause_id=clause_id, text=text))

    if not clauses:
        raise DocumentFormatError(f"{path} has no clauses")
    return ParsedDocument(document, pages, tuple(clauses), tuple(chunks))
