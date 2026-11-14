"""Evaluator-only feature perturbation controls, not instrument injection tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.truth import TruthRecord, TruthType
from examples.catalog.build_demo import build
from sieve.catalog.models import CatalogConfig
from sieve.catalog.workflow import run_catalog


def evaluate(directory: Path) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs"
    build(inputs, target_rows=256)
    # Evaluator-only labels never enter a mapped input or production inference.
    with (inputs / "target.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    truth = [
        TruthRecord(
            id="control-truth:" + row["source_id"],
            source_type=TruthType.SIMULATED_TRUTH,
            source_ref="examples/catalog/build_demo.py: deterministic flux_a perturbation",
            task_scope="synthetic_feature_perturbation",
            member_ids=(row["source_id"],),
            object_ids=("perturbed" if i % 19 == 0 else "baseline",),
            frozen=True,
            complete_for_scope=True,
            audit_ref="evaluation/catalog_controls.py",
            limitations=("Feature-space control only; no physical source or instrument response",),
        )
        for i, row in enumerate(rows)
    ]
    (directory / "evaluator_truth.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in truth], indent=2)
    )
    summary = run_catalog(
        inputs / "reference.manifest.json",
        inputs / "target.manifest.json",
        directory / "run",
        CatalogConfig(review_budget=20, seed=42),
    )
    output = [
        json.loads(line) for line in (directory / "run/records.jsonl").read_text().splitlines()
    ]
    labels = {t.member_ids[0]: int(t.object_ids[0] == "perturbed") for t in truth}
    scored = [r for r in output if r["score"] is not None]
    y = np.asarray([labels[r["source_id"]] for r in scored])
    scores = np.asarray([r["score"]["value"] for r in scored])
    shuffled = np.random.default_rng(42).permutation(scores)
    eligible_perturbed = int(y.sum())
    total_perturbed = sum(labels.values())
    surfaced_perturbed = sum(labels[r["source_id"]] for r in scored if r["decision"] == "SURFACED")
    report = {
        "status": "MEASURED_SYNTHETIC_FEATURE_CONTROL_ONLY",
        "scope": "one frozen trigonometric feature fixture; not real astronomy",
        "total_target_rows": len(output),
        "total_perturbed_rows": total_perturbed,
        "eligible_perturbed_rows": eligible_perturbed,
        "excluded_perturbed_rows": total_perturbed - eligible_perturbed,
        "surfaced_perturbed_rows": surfaced_perturbed,
        "deferred_perturbed_rows": eligible_perturbed - surfaced_perturbed,
        "auc_on_scored_feature_control": float(roc_auc_score(y, scores)),
        "shuffled_score_negative_control_auc": float(roc_auc_score(y, shuffled)),
        "surfacing_fraction_all_perturbed": surfaced_perturbed / total_perturbed,
        "surfacing_fraction_eligible_perturbed": surfaced_perturbed / eligible_perturbed,
        "review_budget": summary["review_budget"],
        "limitations": [
            "No pixel injection, realistic PSF/noise, astrometric fit or instrument qualification",
            "AUC and surfacing fractions refer solely to synthetic feature perturbations",
            "Shuffled-score control is one fixed permutation, not a significance test",
            "No anomaly probability, case-level FDR or real sensitivity estimate",
        ],
    }
    (directory / "control-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args().output), indent=2))
