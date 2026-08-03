"""Human-readable inspection of persisted provenance artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def render_artifact(artifact_root: Path) -> str:
    artifact_path = artifact_root / "corpus.json"
    if not artifact_path.exists():
        raise FileNotFoundError("No corpus artifact found; run `contractgraph ingest` first")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    lines = [
        "ContractGraph provenance artifact",
        f"Schema: {artifact['schema_version']}",
        f"Corpus: {artifact['corpus_version']}",
        f"Documents: {len(artifact['documents'])}",
        f"Pages: {len(artifact['pages'])}",
        f"Clauses: {len(artifact['clauses'])}",
        f"Chunks: {len(artifact['chunks'])}",
        f"Semantic entities: {len(artifact.get('entities', []))}",
        f"Triples: {len(artifact['triples'])}",
        "",
        "Document hierarchy",
    ]
    clauses_by_document: dict[str, list[dict[str, object]]] = {}
    for clause in artifact["clauses"]:
        clauses_by_document.setdefault(clause["document_id"], []).append(clause)
    for document in artifact["documents"]:
        relationship = (
            f" -> amends {document['amends_contract_id']}"
            if document["amends_contract_id"]
            else ""
        )
        lines.append(
            f"- {document['document_id']} [{document['document_type']}]{relationship}: "
            f"{document['title']}"
        )
        for clause in clauses_by_document.get(document["document_id"], []):
            lines.append(
                f"  - p.{clause['page_number']} §{clause['section']} "
                f"{clause['clause_id']}: {clause['title']}"
            )

    lines.extend(("", "Provenance-backed triples"))
    for triple in artifact["triples"]:
        lines.append(
            f"- {triple['subject']} --{triple['predicate']}--> {triple['object']} "
            f"[source={triple['source_clause_id']}; method={triple['population_method']}]"
        )
    return "\n".join(lines) + "\n"
