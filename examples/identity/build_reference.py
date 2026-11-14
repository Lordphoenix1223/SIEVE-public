"""Four tiny identity-contract cases; evaluator truth is written outside inference.

Injected test statistics exercise interfaces only. No instrument/evidence model,
no independent Harness and no estimated physical-factor benefit are implemented.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
STAMP = "2026-10-06T00:00:00Z"
METHOD = dict(name="reference-contract", version="v1")


def build(kind: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations = []
    measurements = []
    n = 2 if kind == "sparse" else 4
    for i in range(n):
        position = dict(longitude_deg=10 + i * 0.0001, latitude_deg=1.0, frame="icrs")
        observations.append(
            dict(
                id=f"o{i}",
                source=dict(
                    provider="synthetic-reference",
                    dataset="identity-contract",
                    release="v1",
                    exposure_id=f"ex{i}",
                    source_id=f"row{i}",
                ),
                time=dict(jd1=2461000.0, jd2=i * 0.02, scale="tai"),
                dependence_group=f"ex{i}",
                raw_asset_ids=("raw-observations",),
                provenance_ref="activity-observe",
            )
        )
        measurements.append(
            dict(
                id=f"m{i}",
                observation_id=f"o{i}",
                raw_source_id=f"measurement-row{i}",
                position=position,
                covariance=dict(
                    values=((0.04, 0.0), (0.0, 0.04)),
                    unit="arcsec2",
                    basis="tangent_longitude_cos_latitude_latitude",
                ),
                quantities=(),
                method=METHOD,
                quality=dict(usable=kind != "artifact"),
                raw_output_asset_ids=("raw-measurements",),
                provenance_ref="activity-measure",
            )
        )
    members = tuple(m["id"] for m in measurements)
    proposal = dict(
        id="proposal",
        member_ids=members,
        method=METHOD,
        generation_input_ids=members,
        raw_output_asset_ids=("raw-proposal",),
        rationale="Fixed synthetic candidate; no identity claim",
        provenance_ref="activity-propose",
    )
    hypotheses = []

    def h(
        key: str,
        explanation: str,
        partitions: tuple[tuple[str, ...], ...],
        known: str | None = None,
    ) -> None:
        hypotheses.append(
            dict(
                id=key,
                proposal_ids=("proposal",),
                member_ids=members,
                explanation=explanation,
                assignments=tuple(
                    dict(slot_id=key + f":slot{i}", member_ids=ids, known_record_ref=known)
                    for i, ids in enumerate(partitions)
                ),
                unassigned_member_ids=members if not partitions else (),
                generation_input_ids=members,
                rationale="Conditional synthetic explanation; retained alongside alternatives",
                provenance_ref="activity-hypothesize",
            )
        )

    h("h-one", "ONE_OBJECT", (members,))
    h("h-unresolved", "UNRESOLVED", ())
    if kind == "merge":
        h("h-two", "MULTIPLE_OBJECTS", (members[:2], members[2:]))
    if kind == "clean":
        h("h-known-a", "KNOWN_MATCH", (members,), "catalog-record-a")
        h("h-known-b", "KNOWN_MATCH", (members,), "catalog-record-b")
        h("h-split", "MULTIPLE_OBJECTS", (members[:2], members[2:]))
    if kind == "artifact":
        h("h-artifact", "ARTIFACT", ())
    diagnostic = 5.0 if kind == "merge" else 0.0
    raw_values = {
        "raw-observations": observations,
        "raw-measurements": measurements,
        "raw-proposal": proposal,
        "raw-test-output": dict(diagnostic=diagnostic, synthetic=True),
        "physical-test-input": dict(injected_diagnostic=diagnostic, scope="contract-only"),
        "test-protocol": dict(
            rule="diagnostic > 3 contradicts; otherwise insufficient (synthetic claim only)",
            scientific_qualification="NONE",
        ),
        "dependence-control": dict(
            shared_input_control="fixture declaration only", scientific_qualification="NONE"
        ),
        "catalog-record-a": dict(record="a", association="candidate only"),
        "catalog-record-b": dict(record="b", association="competing candidate only"),
    }
    raw_dir = ROOT / "examples/identity/raw" / kind
    raw_dir.mkdir(parents=True, exist_ok=True)
    entities = []
    for key, value in raw_values.items():
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        (raw_dir / (key + ".json")).write_bytes(raw)
        entities.append(
            dict(
                id=key,
                kind="synthetic_raw_or_protocol",
                uri=f"repo://SIEVE/examples/identity/raw/{kind}/{key}.json",
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    request = dict(
        id="test-request",
        hypothesis_id="h-one",
        method=METHOD,
        purpose="TEST",
        input_ids=("physical-test-input",),
        tested_claim="synthetic one-object diagnostic",
        dependence="DISJOINT_INPUTS",
        provenance_ref="activity-request",
    )
    factor = dict(
        id="test-factor",
        request_id="test-request",
        hypothesis_id="h-one",
        method=METHOD,
        purpose="TEST",
        input_ids=("physical-test-input",),
        tested_claim=request["tested_claim"],
        verdict="CONTRADICTS" if kind == "merge" else "INSUFFICIENT",
        quantities=(dict(name="contract_statistic", quantity=dict(value=diagnostic, unit="1")),),
        test_protocol_ref="test-protocol",
        rationale="Injected interface test, not scientific evidence",
        limitations=("No physical factor measured; contract fixture only",),
        provenance_ref="activity-evidence",
    )
    result = dict(
        request=request,
        status="SUCCESS",
        factors=(factor,),
        reason="contract-only fixture",
        raw_output_asset_ids=("raw-test-output",),
    )
    layers = [
        ("observe", observations, lambda o: o["raw_asset_ids"]),
        ("measure", measurements, lambda m: (m["observation_id"], *m["raw_output_asset_ids"])),
        ("propose", [proposal], lambda p: (*p["member_ids"], *p["raw_output_asset_ids"])),
        (
            "hypothesize",
            hypotheses,
            lambda h: (
                "proposal",
                *h["member_ids"],
                *(a["known_record_ref"] for a in h["assignments"] if a["known_record_ref"]),
            ),
        ),
        ("request", [request], lambda r: (r["hypothesis_id"], *r["input_ids"])),
        (
            "evidence",
            [factor],
            lambda f: (
                f["request_id"],
                f["hypothesis_id"],
                *f["input_ids"],
                f["test_protocol_ref"],
                "raw-test-output",
            ),
        ),
    ]
    records = []
    for name, objects, parents in layers:
        derivations = {o["id"]: tuple(parents(o)) for o in objects}
        for o in objects:
            entities.append(
                dict(
                    id=o["id"],
                    kind=name,
                    sha256=hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest(),
                )
            )
        records.append(
            dict(
                id="activity-" + name,
                activity="synthetic-" + name,
                method_version="contract.v1",
                agent_ids=("fixture-author",),
                used_entity_ids=sorted({p for ps in derivations.values() for p in ps}),
                generated_entity_ids=tuple(derivations),
                derivations=derivations,
                code_commit="REFERENCE_CONTRACT_V1",
                software_version="identity.v1",
                configuration=dict(synthetic=True),
                request_parameters={},
                random_seeds=dict(fixture=0),
                environment_id="stdlib-json-reference",
                started_at=STAMP,
                ended_at=STAMP,
            )
        )
    bundle = dict(
        schema_version="sieve.identity.v1",
        observations=observations,
        measurements=measurements,
        proposals=(proposal,),
        hypotheses=hypotheses,
        evidence_results=(result,),
        provenance=dict(
            entities=entities,
            agents=(
                dict(id="fixture-author", name="fixture-author", role="contract-fixture-author"),
            ),
            records=records,
        ),
    )
    truth = []
    for i in range(n):
        object_id = (
            ()
            if kind == "artifact"
            else (("synthetic-B",) if kind == "merge" and i >= 2 else ("synthetic-A",))
        )
        truth.append(
            dict(
                id=f"truth-{i}",
                source_type="SIMULATED_TRUTH",
                source_ref="reference-contract.v1",
                task_scope="membership",
                member_ids=(f"m{i}",),
                object_ids=object_id,
                frozen=True,
                complete_for_scope=True,
                limitations=("Contract-only synthetic memberships",),
            )
        )
    return bundle, truth


if __name__ == "__main__":
    from sieve.identity.models import IdentityInput

    for kind in ("clean", "merge", "sparse", "artifact"):
        bundle, truth = build(kind)
        IdentityInput.model_validate(bundle)
        (ROOT / f"examples/identity/{kind}.json").write_text(json.dumps(bundle, indent=2) + "\n")
        (ROOT / f"evaluation/fixtures/identity-{kind}-truth.json").write_text(
            json.dumps(truth, indent=2) + "\n"
        )
