"""One offline entry point; no plugin loading, remote calls or hidden truth reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sieve import __version__
from sieve.catalog.models import CatalogConfig
from sieve.contracts.bundle import InferenceBundle
from sieve.pipeline import report_payload, triage
from sieve.provenance.models import (
    Agent,
    Entity,
    ProvenanceGraph,
    ProvenanceRecord,
    RunReceipt,
    content_id,
)
from sieve.provenance.runtime import source_identity as source_identity
from sieve.reporting import FunnelReport
from sieve.triage.budget import CandidateBudget
from sieve.triage.dedup import DedupPolicy


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_bundle(path: Path) -> tuple[InferenceBundle, str]:
    raw = path.read_bytes()
    data = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
    return InferenceBundle.model_validate(data), hashlib.sha256(raw).hexdigest()


def run(path: Path, budget: CandidateBudget, dedup: DedupPolicy) -> FunnelReport:
    started = datetime.now(UTC)
    bundle, input_hash = read_bundle(path)
    cases, decisions, counts = triage(bundle, budget, dedup)
    payload = report_payload(cases, decisions, counts)
    output_hash = content_id("report", payload).split(":", 1)[1]
    commit, tree, dirty, environment = source_identity()
    configuration = {
        "budget": budget.model_dump(mode="json"),
        "dedup": dedup.model_dump(mode="json"),
        "code_tree_sha256": tree,
        "dirty": dirty,
    }
    ended = datetime.now(UTC)
    run_id = content_id(
        "run",
        {
            "input": input_hash,
            "output": output_hash,
            "configuration": configuration,
            "commit": commit,
            "started_at": started.isoformat(),
        },
    )
    input_entity = Entity(
        id="input:" + input_hash,
        kind="input_manifest",
        uri=path.resolve().as_uri(),
        provider="local",
        dataset_release="sieve.v0",
        sha256=input_hash,
    )
    output_entity = Entity(id="report:" + output_hash, kind="triage_payload", sha256=output_hash)
    agent = Agent(id=run_id + ":agent", name="sieve-offline", role="review_triage")
    record = ProvenanceRecord(
        id=run_id + ":activity",
        activity="offline_candidate_funnel",
        method_version=__version__,
        agent_ids=(agent.id,),
        used_entity_ids=(input_entity.id,),
        generated_entity_ids=(output_entity.id,),
        derivations={output_entity.id: (input_entity.id,)},
        code_commit=commit,
        software_version=__version__,
        configuration=configuration,
        request_parameters={"input_uri": input_entity.uri},
        random_seeds={},
        environment_id=environment,
        started_at=started,
        ended_at=ended,
    )
    graph = ProvenanceGraph(
        entities=bundle.provenance.entities + (input_entity, output_entity),
        agents=bundle.provenance.agents + (agent,),
        records=bundle.provenance.records + (record,),
    )
    receipt = RunReceipt(
        id=run_id,
        provenance_ref=record.id,
        input_sha256=input_hash,
        output_sha256=output_hash,
        code_commit=commit,
        code_tree_sha256=tree,
        dirty=dirty,
        software_version=__version__,
        environment_id=environment,
        configuration=configuration,
        random_seeds={},
        started_at=started,
        ended_at=ended,
    )
    return FunnelReport(
        counts=counts,
        cases=cases,
        audit_decisions=decisions,
        provenance=graph,
        receipt=receipt,
        statistical_filter={
            "status": "FDR_NOT_APPLICABLE",
            "reason": "Offline manifest has no declared valid null test family",
        },
        limitations=(
            "Synthetic reference triage; no production scientific measurement performed",
            "Budget bounds surfaced cases; audit and input storage remain proportional to input",
            "Scores and evidence are conditional; no automatic discovery or truth claim",
            "Receipt output hash covers counts/cases/audit_decisions, excluding receipt timestamps",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="sieve")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("triage", help="triage a validated offline manifest")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--budget", type=int, default=5)
    command.add_argument("--per-region", type=int)
    command.add_argument("--per-time-window", type=int)
    command.add_argument("--per-regime", type=int)
    command.add_argument("--hypothesis-budget", type=int, default=20)
    command.add_argument("--per-case-hypotheses", type=int, default=10)
    command.add_argument("--approximate-dedup", action="store_true")
    command.add_argument(
        "--full-report",
        action="store_true",
        help="emit full case material, per-item audit and lineage graph",
    )
    catalog = sub.add_parser("catalog", help="import local archive catalogs and rank for review")
    catalog.add_argument(
        "--reference", type=Path, required=True, help="reference-only fit manifest"
    )
    catalog.add_argument("--input", type=Path, required=True, help="target catalog manifest")
    catalog.add_argument("--output", type=Path, required=True, help="new immutable run directory")
    catalog.add_argument("--budget", type=int, default=20)
    catalog.add_argument("--per-region", type=int, default=5)
    catalog.add_argument("--training-rows", type=int, default=4096)
    catalog.add_argument("--batch-rows", type=int, default=1024)
    catalog.add_argument("--max-rows", type=int, default=1_000_000)
    catalog.add_argument("--max-input-bytes", type=int, default=2_000_000_000)
    catalog.add_argument("--max-work-bytes", type=int, default=4_000_000_000)
    catalog.add_argument("--seed", type=int, default=42)
    catalog.add_argument(
        "--json-array", action="store_true", help="also stream a complete records.json"
    )
    verify = sub.add_parser(
        "catalog-verify", help="independently verify local output bytes/accounting"
    )
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--receipt-sha256", help="optional independently retained receipt hash")
    curves = sub.add_parser(
        "curves", help="AUXILIARY frozen Kepler ranking; no physical identity inference"
    )
    curves.add_argument("--input", type=Path, required=True)
    curves.add_argument("--model", type=Path, required=True)
    curves.add_argument("--profile-sha256", required=True)
    curves.add_argument("--output", type=Path, required=True)
    curves.add_argument("--budget", type=int, default=20)
    curves.add_argument("--json-array", action="store_true")
    curves.add_argument("--max-rows", type=int, default=100_000)
    curves.add_argument("--max-input-bytes", type=int, default=2_000_000_000)
    identity = sub.add_parser(
        "identity", help="retain and test competing physical-object hypotheses"
    )
    identity.add_argument("--input", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    identity.add_argument("--budget", type=int, default=20)
    identity.add_argument("--hypothesis-budget", type=int, default=200)
    identity.add_argument("--per-case-hypotheses", type=int, default=50)
    loo = sub.add_parser("identity-loo", help="seal N-1 baseline prediction before held-out reveal")
    loo.add_argument("--input", type=Path, required=True)
    loo.add_argument("--hypothesis", required=True)
    loo.add_argument("--heldout", required=True)
    loo.add_argument("--output", type=Path, required=True)
    comparison = sub.add_parser(
        "compare-qr3", help="experimental QR3 candidates with fail-closed science gates"
    )
    comparison.add_argument("--first", type=Path, required=True)
    comparison.add_argument("--second", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "compare-qr3":
            from sieve.registration.pipeline import run_qr3_pair

            result = run_qr3_pair(args.first, args.second, args.output)
            print(
                json.dumps(
                    {
                        "mode": result["mode"],
                        "source_association": result["source_association"],
                        "motion_claim": result["motion_claim"],
                        "flux_variability_claim": result["flux_variability_claim"],
                    }
                )
            )
            return 0
        if args.command == "identity":
            from sieve.identity.graph import IdentityBudget
            from sieve.identity.workflow import run_identity

            print(
                json.dumps(
                    run_identity(
                        args.input,
                        args.output,
                        IdentityBudget(
                            review_cases=args.budget,
                            review_hypotheses=args.hypothesis_budget,
                            per_case_hypotheses=args.per_case_hypotheses,
                        ),
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        if args.command == "identity-loo":
            from sieve.identity.workflow import run_loo

            print(
                json.dumps(
                    run_loo(args.input, args.hypothesis, args.heldout, args.output),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        if args.command == "curves":
            from sieve.specialist.workflow import run_curves

            summary = run_curves(
                args.input,
                args.model,
                args.output,
                args.profile_sha256,
                args.budget,
                args.json_array,
                args.max_rows,
                args.max_input_bytes,
            )
            print(json.dumps(summary, indent=2, allow_nan=False))
            return 0
        if args.command == "catalog-verify":
            from sieve.catalog.verify import verify_run

            print(json.dumps(verify_run(args.run, args.receipt_sha256), indent=2, allow_nan=False))
            return 0
        if args.command == "catalog":
            from sieve.catalog.workflow import run_catalog

            summary = run_catalog(
                args.reference,
                args.input,
                args.output,
                CatalogConfig(
                    review_budget=args.budget,
                    per_region=args.per_region,
                    training_rows=args.training_rows,
                    batch_rows=args.batch_rows,
                    max_rows_per_input=args.max_rows,
                    max_input_bytes=args.max_input_bytes,
                    max_work_bytes=args.max_work_bytes,
                    seed=args.seed,
                    export_json_array=args.json_array,
                ),
            )
            print(json.dumps(summary, indent=2, allow_nan=False))
            return 0
        budget = CandidateBudget(
            total=args.budget,
            per_region=args.per_region,
            per_time_window=args.per_time_window,
            per_regime=args.per_regime,
            total_hypotheses=args.hypothesis_budget,
            per_case_hypotheses=args.per_case_hypotheses,
        )
        report = run(args.input, budget, DedupPolicy(approximate=args.approximate_dedup))
    except (OSError, ValueError, ValidationError, subprocess.SubprocessError) as exc:
        print(
            json.dumps({"status": "INPUT_OR_EXECUTION_ERROR", "reason": str(exc)}, allow_nan=False),
            file=sys.stderr,
        )
        return 2
    if args.full_report:
        print(report.model_dump_json(indent=2))
    else:
        print(json.dumps(compact_report(report), indent=2, allow_nan=False))
    return 0


def compact_report(report: FunnelReport) -> dict[str, object]:
    """Bounded review projection; full original material is in the hashed input."""
    cases = []
    for item in report.cases:
        case = item.case
        summaries: dict[tuple[str, str], dict[str, Any]] = {}
        for result in case.evidence_results:
            key = (result.request.hypothesis_id, result.status)
            summary = summaries.setdefault(
                key,
                {
                    "hypothesis_id": key[0],
                    "status": key[1],
                    "request_count": 0,
                    "factor_count": 0,
                    "interpretations": set(),
                    "calibrated_factor_count": 0,
                },
            )
            summary["request_count"] += 1
            summary["factor_count"] += len(result.factors)
            summary["interpretations"].update(f.interpretation for f in result.factors)
            summary["calibrated_factor_count"] += sum(f.calibrated for f in result.factors)
        for summary in summaries.values():
            summary["interpretations"] = sorted(summary["interpretations"])
        cases.append(
            {
                "case": {
                    "id": case.id,
                    "scope": case.scope.model_dump(mode="json", exclude={"schema_version"}),
                    "proposal_ids": [p.id for p in case.proposals[:10]],
                    "proposal_count": len(case.proposals),
                    "proposal_ids_omitted_from_view": max(0, len(case.proposals) - 10),
                    "anchor_method": case.proposals[0].method.model_dump(mode="json"),
                    "hypotheses": [
                        {
                            "id": h.id,
                            "object_count": h.object_count,
                            "member_count": len(h.member_ids),
                            "explanation": h.explanation,
                            "provenance_ref": h.provenance_ref,
                        }
                        for h in case.hypotheses
                    ],
                    "evidence_results": [summaries[key] for key in sorted(summaries)],
                    "anchor_score": case.proposals[0].score.model_dump(mode="json")
                    if case.proposals[0].score
                    else None,
                },
                "assessments": [
                    a.model_dump(mode="json", exclude={"schema_version"}) for a in item.assessments
                ],
                "decision": item.decision.model_dump(mode="json", exclude={"schema_version"}),
            }
        )
    run_record = next(r for r in report.provenance.records if r.id == report.receipt.provenance_ref)
    return {
        "schema_version": "sieve.v0",
        "report_view": "compact",
        "counts": report.counts,
        "cases": cases,
        "audit_action_counts": dict(
            sorted(Counter(d.action for d in report.audit_decisions).items())
        ),
        "statistical_filter": report.statistical_filter,
        "receipt": report.receipt.model_dump(mode="json"),
        "provenance": {
            "run_activity": run_record.model_dump(mode="json"),
            "run_entities": [
                e.model_dump(mode="json")
                for e in report.provenance.entities
                if e.id in run_record.used_entity_ids + run_record.generated_entity_ids
            ],
            "lineage_counts": {
                "entities": len(report.provenance.entities),
                "activities": len(report.provenance.records),
            },
            "source": "Complete source lineage and deferred material remain in the hashed input; "
            "use --full-report for the full report projection",
        },
        "limitations": report.limitations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
