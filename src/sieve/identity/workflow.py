"""Immutable identity graph exports with complete inventory and run receipts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sieve.identity.graph import IdentityBudget, identity_funnel
from sieve.identity.models import IdentityInput
from sieve.provenance.models import RunReceipt, content_id
from sieve.provenance.runtime import source_identity


def reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON: {value}")


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_identity(path: Path) -> tuple[IdentityInput, str]:
    if path.stat().st_size > 100_000_000:
        raise ValueError("identity manifest exceeds 100 MB cap")
    raw = path.read_bytes()
    obj = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    return IdentityInput.model_validate(obj), hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run_identity(path: Path, output: Path, budget: IdentityBudget) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output exists; immutable runs cannot be overwritten")
    started = datetime.now(UTC)
    bundle, input_hash = read_identity(path)
    funnel = identity_funnel(bundle, budget)
    commit, tree, dirty, environment = source_identity()
    configuration = budget.model_dump(mode="json")
    payload = funnel.model_dump(mode="json")
    payload_hash = content_id("payload", payload).split(":")[1]
    receipt = RunReceipt(
        id=content_id("identity-run", (input_hash, payload_hash, configuration)),
        provenance_ref="identity-core-runtime",
        input_sha256=input_hash,
        output_sha256=payload_hash,
        code_commit=commit,
        code_tree_sha256=tree,
        dirty=dirty,
        software_version="identity.v1",
        environment_id=environment,
        configuration=configuration,
        random_seeds={},
        started_at=started,
        ended_at=datetime.now(UTC),
    )
    # Runtime receipt covers input -> graph; scientific producer provenance remains in input.json.
    from sieve.provenance.models import Agent, Entity, ProvenanceGraph, ProvenanceRecord

    input_id, output_id = "input:" + input_hash, "graph:" + payload_hash
    runtime_graph = ProvenanceGraph(
        entities=(
            Entity(
                id=input_id, kind="identity_input", uri=path.resolve().as_uri(), sha256=input_hash
            ),
            Entity(id=output_id, kind="identity_funnel", sha256=payload_hash),
        ),
        agents=(Agent(id="identity-core", name="identity-core", role="conditional_graph_review"),),
        records=(
            ProvenanceRecord(
                id=receipt.provenance_ref,
                activity="identity_graph_review",
                method_version="identity.v1",
                agent_ids=("identity-core",),
                used_entity_ids=(input_id,),
                generated_entity_ids=(output_id,),
                derivations={output_id: (input_id,)},
                code_commit=commit,
                software_version="identity.v1",
                configuration=configuration,
                request_parameters={},
                random_seeds={},
                environment_id=environment,
                started_at=started,
                ended_at=receipt.ended_at,
            ),
        ),
    )
    observations = {o.id: o for o in bundle.observations}
    case_actions = {d.subject_id: d.action for d in funnel.decisions}
    case_by_member = {m: c.id for c in funnel.cases for m in c.member_ids}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".identity-", dir=output.parent))
    try:
        (staging / "input.json").write_bytes(path.read_bytes())
        # Verify copied bytes still equal the admitted immutable input.
        if hashlib.sha256((staging / "input.json").read_bytes()).hexdigest() != input_hash:
            raise ValueError("input changed during execution")
        write_json(staging / "graph.json", payload)
        surfaced = {d.subject_id for d in funnel.decisions if d.action == "SURFACED"}
        write_json(
            staging / "cases.json",
            [c.model_dump(mode="json") for c in funnel.cases if c.id in surfaced],
        )
        with (staging / "records.jsonl").open("w") as stream:
            for measurement in sorted(bundle.measurements, key=lambda m: m.id):
                observation = observations[measurement.observation_id]
                valid = measurement.quality.usable and observation.quality.usable
                case_id = case_by_member.get(measurement.id)
                row = dict(
                    id=measurement.id,
                    observation_id=observation.id,
                    raw_source_id=measurement.raw_source_id,
                    source=observation.source.model_dump(mode="json"),
                    case_id=case_id,
                    input_status="ACCEPTED" if valid else "EXCLUDED",
                    review_status=case_actions[case_id] if case_id else "UNASSIGNED",
                    physical_identity="NOT_ESTABLISHED",
                )
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_json(staging / "provenance.json", runtime_graph.model_dump(mode="json"))
        write_json(staging / "run-receipt.json", receipt.model_dump(mode="json"))
        summary = dict(
            schema_version="sieve.identity.run.v1",
            counts=funnel.counts,
            receipt=receipt.model_dump(mode="json"),
            scientific_status="CONDITIONAL_IDENTITY_ONLY",
            limitations=[
                "No qualified SPHEREx factor or orbit adapter executed",
                "No unique object identity, discovery or case-level FDR established",
            ],
        )
        write_json(staging / "summary.json", summary)
        artifacts = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staging.iterdir())
        }
        write_json(staging / "artifacts.json", dict(artifacts=artifacts))
        staging.rename(output)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run_loo(path: Path, hypothesis_id: str, heldout_id: str, output: Path) -> dict[str, Any]:
    from sieve.identity.models import Explanation
    from sieve.identity.prediction import (
        LinearTangentBaseline,
        PredictionContext,
        leave_one_observation_out,
    )

    bundle, _ = read_identity(path)
    hypotheses = {h.id: h for h in bundle.hypotheses}
    if hypothesis_id not in hypotheses:
        raise ValueError("unknown hypothesis")
    hypothesis = hypotheses[hypothesis_id]
    if hypothesis.explanation not in {Explanation.ONE_OBJECT, Explanation.KNOWN_MATCH}:
        raise ValueError("baseline LOO requires a declared one-object hypothesis")
    measurements = {m.id: m for m in bundle.measurements}
    observations = {o.id: o for o in bundle.observations}
    assigned_members = hypothesis.assignments[0].member_ids
    if heldout_id not in assigned_members:
        raise ValueError("held-out measurement must be assigned to the requested object slot")
    heldout = measurements[heldout_id]
    observed = observations[heldout.observation_id]
    training = [
        measurements[key]
        for key in assigned_members
        if observations[measurements[key].observation_id].dependence_group
        != observed.dependence_group
    ]
    fit_observations = [observations[key] for key in sorted({m.observation_id for m in training})]
    context = PredictionContext(
        observation_id=observed.id,
        heldout_measurement_id=heldout.id,
        source=observed.source,
        dependence_group=observed.dependence_group,
        time=observed.time,
        observer_ref=observed.observer_ref,
        observation_quality=observed.quality,
    )
    return leave_one_observation_out(
        LinearTangentBaseline(), fit_observations, training, context, lambda: heldout, output
    )
