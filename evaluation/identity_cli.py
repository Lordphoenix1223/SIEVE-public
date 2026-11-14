"""Offline evaluator entry point; never imported by the inference package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evaluation.identity import (
    ComparisonProtocol,
    FrozenThreshold,
    TruthSet,
    _counts,
    _rates,
    accepted,
    compare_paired,
    freeze_threshold,
)
from sieve.identity.models import PredictionTable
from sieve.identity.workflow import reject_constant, unique_keys, write_json
from sieve.provenance.models import content_id
from sieve.provenance.runtime import source_identity


def load[T: BaseModel](path: Path, model: type[T]) -> tuple[T, bytes]:
    if path.stat().st_size > 100_000_000:
        raise ValueError("evaluator input exceeds 100 MB reference cap")
    raw = path.read_bytes()
    return model.model_validate(
        json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    ), raw


def per_case(
    baseline: PredictionTable,
    treatment: PredictionTable,
    truth: TruthSet,
    protocol: ComparisonProtocol,
) -> list[dict[str, Any]]:
    """All paired cases including abstentions; evidence stays at pinned upstream locators."""
    b_rows = {p.case_id: p for p in baseline.predictions}
    t_rows = {p.case_id: p for p in treatment.predictions}
    results = []
    for case in sorted(truth.cases, key=lambda c: c.id):
        b = _rates(_counts(baseline, truth, protocol.baseline, {case.id}))
        t = _rates(_counts(treatment, truth, protocol.treatment, {case.id}))
        failures = []
        for name, metrics, row, threshold in (
            ("baseline", b, b_rows[case.id], protocol.baseline),
            ("treatment", t, t_rows[case.id], protocol.treatment),
        ):
            if metrics["counts"]["false_merge_cases"]:
                failures.append(name + ":false_merge")
            if metrics["counts"]["split_objects"]:
                failures.append(name + ":false_split")
            if metrics["counts"]["contaminated_slots"]:
                failures.append(name + ":artifact_contamination")
            if not accepted(row, threshold):
                failures.append(name + ":abstention")
        results.append(
            dict(
                case_id=case.id,
                member_ids=case.member_ids,
                dependence_groups=case.dependence_groups,
                baseline_prediction=b_rows[case.id].model_dump(mode="json"),
                treatment_prediction=t_rows[case.id].model_dump(mode="json"),
                baseline=b,
                treatment=t,
                failure_reasons=failures,
            )
        )
    return results


def compare_files(
    baseline_path: Path,
    treatment_path: Path,
    truth_path: Path,
    protocol_path: Path,
    protocol_sha256: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output exists; comparisons cannot be overwritten")
    started = datetime.now(UTC)
    protocol, protocol_raw = load(protocol_path, ComparisonProtocol)
    if hashlib.sha256(protocol_raw).hexdigest() != protocol_sha256:
        raise ValueError("protocol file bytes differ from the preregistered SHA256")
    baseline, baseline_raw = load(baseline_path, PredictionTable)
    treatment, treatment_raw = load(treatment_path, PredictionTable)
    # Predictions are supplied artifacts: this evaluator has no prediction/training callback.
    truth, truth_raw = load(truth_path, TruthSet)
    report = compare_paired(baseline, treatment, truth, protocol)
    cases = per_case(baseline, treatment, truth, protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".comparison-", dir=output.parent))
    try:
        inputs = {
            "baseline.json": baseline_raw,
            "treatment.json": treatment_raw,
            "evaluator-truth.json": truth_raw,
            "protocol.json": protocol_raw,
        }
        for name, raw in inputs.items():
            (stage / name).write_bytes(raw)
        write_json(stage / "paired-metrics.json", report)
        for name, rows in (
            ("per-case.jsonl", cases),
            ("failure-examples.jsonl", [c for c in cases if c["failure_reasons"]]),
        ):
            with (stage / name).open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        commit, core_tree, dirty, environment = source_identity()
        evaluator_sources = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        }
        write_json(
            stage / "receipt.json",
            dict(
                schema_version="sieve.identity.evaluation.v1",
                code_commit=commit,
                core_tree_sha256=core_tree,
                dirty=dirty,
                environment_id=environment,
                evaluator_source_sha256=content_id("evaluator", evaluator_sources).split(":")[1],
                evaluator_sources=evaluator_sources,
                input_file_sha256={
                    name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()
                },
                output_metrics_sha256=hashlib.sha256(
                    (stage / "paired-metrics.json").read_bytes()
                ).hexdigest(),
                random_seeds={"bootstrap": protocol.bootstrap_seed},
                started_at=started.isoformat(),
                ended_at=datetime.now(UTC).isoformat(),
                limitations=[
                    "Local receipts do not enforce a blind external holdout "
                    "or authenticate producer claims"
                ],
            ),
        )
        write_json(
            stage / "artifact-hashes.json",
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.iterdir())},
        )
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze on VALIDATION truth only")
    freeze.add_argument("--predictions", type=Path, required=True)
    freeze.add_argument("--truth", type=Path, required=True)
    freeze.add_argument("--target-recall", type=float, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Evaluate supplied sealed predictions")
    for name in ("baseline", "treatment", "truth", "protocol", "output"):
        compare.add_argument("--" + name, type=Path, required=True)
    compare.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            table, _ = load(args.predictions, PredictionTable)
            truth, _ = load(args.truth, TruthSet)
            threshold: FrozenThreshold = freeze_threshold(table, truth, args.target_recall)
            # Exclusive creation prevents silent replacement of a published operating point.
            with args.output.open("x") as stream:
                stream.write(threshold.model_dump_json(indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "frozen_threshold": str(args.output),
                        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                    }
                )
            )
        else:
            report = compare_files(
                args.baseline,
                args.treatment,
                args.truth,
                args.protocol,
                args.protocol_sha256,
                args.output,
            )
            print(json.dumps(report, sort_keys=True, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"identity evaluation refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
