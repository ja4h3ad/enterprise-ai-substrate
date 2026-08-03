from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from contractgraph.evaluation import EXPECTED_DISTRIBUTION, FAILURE_TAXONOMY, VARIANTS

PROJECT_ROOT = Path(__file__).parents[1]


def _evaluate(output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "contractgraph.cli",
            "evaluate",
            "--corpus",
            str(PROJECT_ROOT / "corpus"),
            "--golden",
            str(PROJECT_ROOT / "evaluation" / "golden.json"),
            "--reports",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_golden_ablation_is_the_reproducible_behavioral_quality_gate(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = _evaluate(first_dir)
    second = _evaluate(second_dir)

    assert "Evaluated 24 questions across 5 variants." in first.stdout
    assert "Preregistered criteria: 9/9 passed." in first.stdout
    assert (first_dir / "ablation.json").read_bytes() == (
        second_dir / "ablation.json"
    ).read_bytes()
    report = json.loads((first_dir / "ablation.json").read_text(encoding="utf-8"))
    assert report["dataset"]["distribution"] == EXPECTED_DISTRIBUTION
    assert report["dataset"]["question_count"] == 24
    assert report["dataset"]["synthetic_only"] is True
    assert report["dataset"]["llm_judge"] is False
    assert set(report["variants"]) == set(VARIANTS)
    assert all(item["passed"] for item in report["success_criteria"])
    assert set(report["failure_taxonomy"]) == set(FAILURE_TAXONOMY)
    assert all(report["failure_taxonomy"].values())

    vector = report["variants"]["vector_only"]
    full = report["variants"]["agentic_full"]
    vector_hero = next(case for case in vector["cases"] if case["item_id"] == "G10")
    full_hero = next(case for case in full["cases"] if case["item_id"] == "G10")
    assert vector_hero["retrieved_clause_ids"][0] == "CLAUSE-ATLAS-8.2"
    assert vector_hero["failure_classification"] == "superseded_clause_retrieval"
    assert full_hero["retrieved_clause_ids"][0] == "CLAUSE-ATLAS-A1-2"
    assert full_hero["graph_path"] == ["AMENDS", "CONTAINS", "SUPERSEDES"]

    required_metrics = {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg",
        "contract_accuracy",
        "clause_accuracy",
        "graph_path_correctness",
        "citation_precision",
        "citation_recall",
        "grounded_claim_rate",
        "unsupported_claim_rate",
        "retrieval_iterations",
        "tool_calls",
        "latency_ms",
        "estimated_tokens",
        "estimated_cost_usd",
    }
    for variant in VARIANTS:
        assert set(report["variants"][variant]["metrics"]) == required_metrics
        assert set(report["variants"][variant]["failure_classifications"]) == set(
            FAILURE_TAXONOMY
        )
        assert len(report["variants"][variant]["cases"]) == 24

    markdown = (first_dir / "ablation.md").read_text(encoding="utf-8")
    assert "## Operational metrics" in markdown
    assert "## Hero ablation" in markdown
    assert "No statistical significance is claimed" in markdown
