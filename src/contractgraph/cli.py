"""Public command-line interface for deterministic provenance workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from contractgraph.application import (
    DEFAULT_REPLAY_FIXTURE,
    DEFAULT_STATE_DB,
    ContractGraphApplication,
    render_answer,
)
from contractgraph.artifacts import load_corpus
from contractgraph.comparison import (
    HERO_QUESTION,
    HERO_RETRIEVAL_QUERY,
    HeroComparisonService,
    render_comparison,
)
from contractgraph.evaluation import GoldenEvaluator, load_golden_set, write_reports
from contractgraph.ingestion import build_corpus
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

    compare = subparsers.add_parser(
        "compare", help="show vector-only and graph-grounded hero results"
    )
    compare.add_argument("--question", default=HERO_QUESTION)
    compare.add_argument("--retrieval-query", default=HERO_RETRIEVAL_QUERY)
    compare.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)

    run = subparsers.add_parser("run", help="run the bounded replay-backed agent workflow")
    run.add_argument("--question", default=HERO_QUESTION)
    run.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    run.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    run.add_argument("--replay-fixture", type=Path, default=DEFAULT_REPLAY_FIXTURE)

    evaluate = subparsers.add_parser(
        "evaluate", help="run the 24-question five-way golden ablation"
    )
    evaluate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    evaluate.add_argument(
        "--golden", type=Path, default=Path("evaluation/golden.json")
    )
    evaluate.add_argument("--reports", type=Path, default=Path("reports"))
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
    if arguments.command == "compare":
        artifact = load_corpus(arguments.artifacts)
        comparison = HeroComparisonService(artifact).compare(
            arguments.question, retrieval_query=arguments.retrieval_query
        )
        print(render_comparison(comparison), end="")
        return 0
    if arguments.command == "run":
        with ContractGraphApplication(
            artifact_root=arguments.artifacts,
            state_db=arguments.state_db,
            replay_fixture=arguments.replay_fixture,
        ) as application:
            result = application.run(arguments.question)
        print(render_answer(result), end="")
        return 0
    if arguments.command == "evaluate":
        artifact = build_corpus(arguments.corpus)
        golden = load_golden_set(arguments.golden, artifact)
        report = GoldenEvaluator(artifact, golden).evaluate()
        markdown_path, json_path = write_reports(report, arguments.reports)
        passed = sum(item["passed"] for item in report["success_criteria"])
        total = len(report["success_criteria"])
        print(f"Evaluated {len(golden.items)} questions across 5 variants.\n")
        print(f"Preregistered criteria: {passed}/{total} passed.\n")
        print(f"Markdown report: {markdown_path}\n")
        print(f"JSON report: {json_path}\n")
        return 0 if passed == total else 1
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
