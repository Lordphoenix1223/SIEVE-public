"""Rebuild tiny offline manifests; evaluator labels go only in evaluation/fixtures.

This is a deterministic fixture constructor, not a source detector or motion fitter.
The injected evidence is explicitly synthetic and uncalibrated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHOD = {"name": "reference", "version": "fixture.v0"}
STAMP = "2026-10-05T00:00:00Z"


def score(value: float) -> dict:
    return {
        "value": value,
        "semantics": "RANKING_ONLY",
        "direction": "higher",
        "calibrated": False,
        "domain": "synthetic-fixture-only",
        "event_or_null": "review-interest-ranking",
    }


def fixture(*, stress: bool = False) -> dict:
    source = {
        "id": "fixture-seed",
        "kind": "synthetic_source",
        "uri": "synthetic://sieve/v0",
        "provider": "SIEVE-reference",
        "provider_identifier": "seed-0",
        "dataset_release": "fixture.v0",
        "sha256": hashlib.sha256(b"SIEVE_REFERENCE_V0_SEED_0").hexdigest(),
        "citations": ["Synthetic fixture; no telescope data"],
    }
    observations, detections, proposals, hypotheses, results = [], [], [], [], []
    layers: dict[str, list] = {
        "observe": observations,
        "detect": detections,
        "propose": proposals,
        "hypothesize": hypotheses,
        "request": [],
        "evidence": [],
    }
    parents: dict[str, tuple[str, ...]] = {}

    def point(name: str, longitude: float, day: int, *, flags=(), usable=True) -> str:
        oid, did = "o-" + name, "d-" + name
        position = {"longitude_deg": longitude, "latitude_deg": 0, "frame": "icrs"}
        observations.append(
            {
                "id": oid,
                "exposure_id": "ex-" + name,
                "dataset": "synthetic",
                "provider": "reference",
                "release": "v0",
                "time": {"jd1": 2461000, "jd2": day / 10, "scale": "tai"},
                "position": position,
                "dependence_group": "ex-" + name,
                "wavelengths": [{"value": 1.5, "unit": "um"}],
                "asset_ids": [source["id"]],
                "provenance_ref": "activity-observe",
            }
        )
        detections.append(
            {
                "id": did,
                "observation_id": oid,
                "position": position,
                "measurements": {"flux": {"value": 10, "unit": "uJy", "uncertainty": 1}},
                "method": METHOD,
                "quality": {"flags": flags, "usable": usable},
                "asset_ids": [source["id"]],
                "provenance_ref": "activity-detect",
            }
        )
        parents[oid], parents[did] = (source["id"],), (oid,)
        return did

    def proposal(
        name: str,
        members: list[str],
        *,
        value=1.0,
        explanation="one-object",
        family=None,
        task="moving",
    ) -> str:
        pid = "p-" + name
        proposals.append(
            {
                "id": pid,
                "member_ids": members,
                "method": METHOD,
                "task_type": task,
                "explanation_key": explanation,
                "equivalence_family": family,
                "score": score(value),
                "provenance_ref": "activity-propose",
            }
        )
        parents[pid] = tuple(members)
        return pid

    def hypothesis(
        name: str,
        pid: str,
        members: list[str],
        count: int | None,
        explanation: str,
        interpretation: str | None = None,
        status="SUCCESS",
    ) -> None:
        hid, rid, fid = "h-" + name, "request-" + name, "factor-" + name
        hypotheses.append(
            {
                "id": hid,
                "proposal_ids": [pid],
                "member_ids": members,
                "object_count": count,
                "explanation": explanation,
                "provenance_ref": "activity-hypothesize",
            }
        )
        parents[hid] = (pid, *members)
        request = {
            "id": rid,
            "hypothesis_id": hid,
            "method": METHOD,
            "input_ids": members,
            "provenance_ref": "activity-request",
        }
        layers["request"].append(request)
        parents[rid] = (hid, *members)
        factors = []
        if interpretation:
            factor = {
                "id": fid,
                "hypothesis_id": hid,
                "method": METHOD,
                "input_ids": members,
                "measurements": {
                    "diagnostic": {
                        "value": None,
                        "unit": "1",
                        "missing_reason": "not_measured_in_fixture",
                    }
                },
                "polarity": "for",
                "interpretation": interpretation,
                "rationale": "Injected synthetic narrative, not a measured scientific result",
                "calibrated": False,
                "provenance_ref": "activity-evidence",
                "limitations": ["Synthetic evidence; no measurement or orbit fitting"],
            }
            factors.append(factor)
            layers["evidence"].append(factor)
            parents[fid] = (rid, hid, *members)
        results.append(
            {
                "request": request,
                "status": status,
                "factors": factors,
                "reason": "synthetic_reference_result",
            }
        )

    if stress:
        # Many high ranked cases in one region plus distinct low ranked regions.
        for i in range(240):
            members = [point(f"stress-{i:03}", 10 + i * 1e-5, 0)]
            pid = proposal(f"stress-{i:03}", members, value=1000 - i)
            hypothesis(f"stress-{i:03}", pid, members, None, "Unresolved synthetic detection")
        for i in range(1, 7):
            members = [point(f"diverse-{i}", 40 * i, i)]
            pid = proposal(f"diverse-{i}", members, value=-i, task="anomaly" if i % 2 else "moving")
            hypothesis(f"diverse-{i}", pid, members, None, "Unresolved synthetic detection")
    else:
        a = [point(f"a{i}", 10 + i * 0.001, i) for i in range(4)]
        b = [point(f"b{i}", 11 - i * 0.002, i) for i in range(3)]
        noise = [point("noise0", 80, 0), point("noise1", 82, 1)]
        anomaly = [point("anomaly", 140, 1)]
        insufficient = [point("insufficient", 200, 1)]
        static = [point("static", 30, 1, flags=("REFERENCE_STATIC",))]
        unusable = [point("unusable", 40, 1, usable=False)]
        clean = proposal("clean", a, value=5, family="a-sampling")
        exact = proposal("exact", list(reversed(a)), value=4, family="a-sampling")
        overlap = proposal("overlap", a[:3], value=3, family="a-sampling")
        # 3/4 overlap is below the default 0.8; proximity to same first point groups it.
        hypothesis(
            "clean", clean, a, 1, "Single coherent synthetic moving source", "MOTION_CONSISTENT"
        )
        hypothesis("exact", exact, a, 1, "Repeated proposal of the same measurement set")
        hypothesis("overlap", overlap, a[:3], 1, "Alternative sampling of source A")
        mixed = [a[0], a[1], b[1], b[2]]
        franken = proposal("frankenstein", mixed, value=8)
        hypothesis(
            "frankenstein-one",
            franken,
            mixed,
            1,
            "Possible single-source false merge",
            "AMBIGUOUS_IDENTITY",
        )
        hypothesis(
            "frankenstein-two",
            franken,
            mixed,
            2,
            "Possible mixture of two objects",
            "AMBIGUOUS_IDENTITY",
        )
        alternative = proposal("alternative", a[:3], explanation="two-objects", value=2)
        hypothesis("alternative", alternative, a[:3], 2, "Competing multi-object explanation")
        artifact = proposal("artifact", noise, value=900)
        hypothesis(
            "artifact", artifact, noise, 0, "Possible artifact explanation", "ARTIFACT_SUGGESTIVE"
        )
        high = proposal("high-anomaly", anomaly, value=1_000_000, task="anomaly")
        hypothesis(
            "high-anomaly",
            high,
            anomaly,
            None,
            "High rank is not physical truth",
            "CATALOG_NO_MATCH",
        )
        low = proposal("insufficient", insufficient)
        hypothesis("insufficient", low, insufficient, None, "Insufficient measurements")
        proposal("static", static)
        proposal("unusable", unusable)
        # Exact duplicate input observation is counted and normalized, never new support.
        observations.append(dict(observations[0]))

    entities = [source]
    records = []
    for layer, objects in layers.items():
        if not objects:
            continue
        # Observation duplicate produces only one entity.
        ids = sorted({obj["id"] for obj in objects})
        for key in ids:
            obj = next(obj for obj in objects if obj["id"] == key)
            entities.append(
                {
                    "id": key,
                    "kind": layer,
                    "sha256": hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(),
                }
            )
        used = sorted({p for key in ids for p in parents[key]})
        records.append(
            {
                "id": "activity-" + layer,
                "activity": "synthetic_" + layer,
                "method_version": "fixture.v0",
                "agent_ids": ["fixture-author"],
                "used_entity_ids": used,
                "generated_entity_ids": ids,
                "derivations": {key: parents[key] for key in ids},
                "code_commit": "REFERENCE_GENERATOR_V0",
                "software_version": "fixture.v0",
                "configuration": {"synthetic": True, "stress": stress},
                "request_parameters": {},
                "random_seeds": {"fixture": 0},
                "environment_id": "stdlib-json-fixture.v0",
                "started_at": STAMP,
                "ended_at": STAMP,
            }
        )
    return {
        "observations": observations,
        "detections": detections,
        "proposals": proposals,
        "hypotheses": hypotheses,
        "evidence_results": results,
        "provenance": {
            "entities": entities,
            "agents": [
                {
                    "id": "fixture-author",
                    "name": "offline-fixture-builder",
                    "role": "synthetic_fixture_author",
                }
            ],
            "records": records,
        },
    }


def main() -> None:
    for name, stress in (("frankenstein_case.json", False), ("stress_case.json", True)):
        data = fixture(stress=stress)
        (ROOT / "tests/fixtures" / name).write_text(
            json.dumps(data, indent=None if stress else 2) + "\n"
        )
    truth_dir = ROOT / "evaluation/fixtures"
    truth_dir.mkdir(exist_ok=True)
    truth = {
        "source_type": "SIMULATED_TRUTH",
        "source_ref": "fixture-generator.v0",
        "task_scope": "membership",
        "frozen": True,
        "complete_for_scope": True,
        "limitations": ["Synthetic object assignments only"],
    }
    records = []
    for i in range(4):
        records.append(
            {**truth, "id": f"truth-a{i}", "member_ids": [f"d-a{i}"], "object_ids": ["object-A"]}
        )
    for i in range(3):
        records.append(
            {**truth, "id": f"truth-b{i}", "member_ids": [f"d-b{i}"], "object_ids": ["object-B"]}
        )
    (truth_dir / "frankenstein_truth.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
