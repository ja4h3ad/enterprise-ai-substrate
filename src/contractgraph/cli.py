"""Public command-line interface for deterministic provenance workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from contractgraph.ingestion import artifact_digest, build_corpus, persist_corpus
from contractgraph.inspection import render_artifact

DEFAULT_CORPUS_ROOT = Path("corpus")
DEFAULT_ARTIFACT_ROOT = Path(".contractgraph/artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contractgraph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="rebuild deterministic provenance artifacts")
    ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    ingest.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)

    inspect = subparsers.add_parser("inspect", help="display hierarchy and provenance")
    inspect.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "ingest":
        artifact = build_corpus(arguments.corpus)
        persist_corpus(artifact, arguments.artifacts)
        print(
            "Ingested "
            f"{len(artifact.documents)} documents, {len(artifact.clauses)} clauses, "
            f"and {len(artifact.triples)} triples."
        )
        print(f"Corpus digest: {artifact_digest(artifact)}")
        print(f"Artifacts: {arguments.artifacts.resolve()}")
        return 0
    if arguments.command == "inspect":
        print(render_artifact(arguments.artifacts), end="")
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
