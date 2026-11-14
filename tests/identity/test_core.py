from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sieve.identity.attachment import attach_evidence, merge_provenance
from sieve.identity.graph import IdentityBudget, identity_funnel
from sieve.identity.models import IdentityInput, ScientificDisposition
from sieve.identity.prediction import (
    LinearTangentBaseline,
    PredictionContext,
    leave_one_observation_out,
)
from sieve.identity.workflow import read_identity, run_identity, run_loo

ROOT = Path(__file__).parents[2]


def bundle(kind="merge"):
    return read_identity(ROOT / f"examples/identity/{kind}.json")[0]


def test_zero_one_many_known_and_unresolved_compete():
    data = bundle("clean")
    graph = identity_funnel(data, IdentityBudget())
    assert len(graph.cases) == 1
    assert len(graph.cases[0].hypothesis_ids) == 5
    assert graph.cases[0].disposition == ScientificDisposition.UNRESOLVED
    assert {h.object_count for h in data.hypotheses} == {1, 2, None}
    assert len([h for h in data.hypotheses if h.explanation == "KNOWN_MATCH"]) == 2
    artifact = bundle("artifact")
    assert any(h.object_count == 0 for h in artifact.hypotheses)


def test_false_merge_contradiction_does_not_force_other_partition():
    data = bundle()
    graph = identity_funnel(data, IdentityBudget())
    by_id = {a.hypothesis_id: a for a in graph.assessments}
    assert by_id["h-one"].disposition == ScientificDisposition.FALSIFIED_UNDER_DECLARED_TEST
    assert by_id["h-one"].contradiction_factor_ids == ("test-factor",)
    assert graph.cases[0].disposition == ScientificDisposition.UNRESOLVED
    assert len(graph.cases[0].hypothesis_ids) == 3


def test_budget_defers_whole_family_without_discarding_hypotheses():
    data = bundle()
    graph = identity_funnel(data, IdentityBudget(per_case_hypotheses=1))
    assert graph.counts["cases_surfaced"] == 0
    assert graph.counts["cases_deferred"] == 1
    assert graph.counts["hypotheses_preserved"] == 3
    assert len(graph.assessments) == 3


def test_unknown_dependence_is_not_admitted_as_falsification():
    data = bundle().model_dump(mode="json")
    data["evidence_results"][0]["request"]["dependence"] = "UNKNOWN"
    graph = identity_funnel(IdentityInput.model_validate(data), IdentityBudget())
    assessment = next(a for a in graph.assessments if a.hypothesis_id == "h-one")
    assert assessment.disposition == ScientificDisposition.INSUFFICIENT_EVIDENCE
    assert not assessment.tested_factor_ids


def test_generation_evidence_cannot_test_its_own_hypothesis():
    data = bundle().model_dump(mode="json")
    result = data["evidence_results"][0]
    result["request"]["purpose"] = "GENERATE"
    result["factors"][0]["purpose"] = "GENERATE"
    graph = identity_funnel(IdentityInput.model_validate(data), IdentityBudget())
    assert not next(a for a in graph.assessments if a.hypothesis_id == "h-one").tested_factor_ids


def test_indirect_reused_generation_source_cannot_claim_disjoint_test():
    data = bundle().model_dump(mode="json")
    data["evidence_results"][0]["request"]["input_ids"] = ["m0"]
    data["evidence_results"][0]["factors"][0]["input_ids"] = ["m0"]
    for record in data["provenance"]["records"]:
        if record["id"] in {"activity-request", "activity-evidence"}:
            record["used_entity_ids"].append("m0")
            for parents in record["derivations"].values():
                parents.append("m0")
    with pytest.raises(ValueError, match="reuses generation source"):
        IdentityInput.model_validate(data)


@pytest.mark.parametrize(
    "change", ["truth", "duplicate", "missing_raw", "partition", "missing_control"]
)
def test_corrupt_contract_refused(change):
    data = bundle().model_dump(mode="json")
    if change == "truth":
        data["truth"] = {"m0": "hidden-object"}
    elif change == "duplicate":
        data["measurements"].append(copy.deepcopy(data["measurements"][0]))
    elif change == "missing_raw":
        data["provenance"]["entities"][0]["sha256"] = None
        data["provenance"]["entities"][0]["hash_missing_reason"] = "unpinned"
    elif change == "partition":
        data["hypotheses"][0]["assignments"][0]["member_ids"].pop()
    else:
        data["evidence_results"][0]["request"]["dependence"] = "CONTROLLED_SHARED_INPUTS"
    with pytest.raises(ValueError):
        IdentityInput.model_validate(data)


def test_failed_measurement_is_not_scientific_negative():
    data = bundle().model_dump(mode="json")
    data["evidence_results"][0].update(status="TIMEOUT", factors=[])
    graph = identity_funnel(IdentityInput.model_validate(data), IdentityBudget())
    assert (
        next(a for a in graph.assessments if a.hypothesis_id == "h-one").disposition
        == ScientificDisposition.UNRESOLVED
    )


def test_attach_evidence_preserves_inputs_and_checks_conflicts():
    data = bundle()
    empty = IdentityInput.model_validate({**data.model_dump(mode="json"), "evidence_results": []})
    attached = attach_evidence(empty, data.evidence_results, data.provenance)
    assert attached.observations == data.observations
    assert attached.measurements == data.measurements
    assert attached.hypotheses == data.hypotheses
    assert attached.evidence_results == data.evidence_results
    assert {e.id: e for e in attached.provenance.entities} == {
        e.id: e for e in data.provenance.entities
    }
    assert not empty.evidence_results
    conflicting = data.provenance.model_dump(mode="json")
    conflicting["entities"][0]["sha256"] = "1" * 64
    from sieve.provenance.models import ProvenanceGraph

    with pytest.raises(ValueError, match="conflicting immutable"):
        merge_provenance(data.provenance, ProvenanceGraph.model_validate(conflicting))


def test_exports_complete_accounting_and_repeat_payload(tmp_path):
    source = ROOT / "examples/identity/artifact.json"
    first = run_identity(source, tmp_path / "one", IdentityBudget())
    second = run_identity(source, tmp_path / "two", IdentityBudget())
    assert first["receipt"]["output_sha256"] == second["receipt"]["output_sha256"]
    rows = list(map(json.loads, (tmp_path / "one/records.jsonl").read_text().splitlines()))
    assert len(rows) == 4
    assert all(r["input_status"] == "EXCLUDED" for r in rows)
    assert all(r["physical_identity"] == "NOT_ESTABLISHED" for r in rows)
    assert all(r["source"]["exposure_id"] for r in rows)
    receipt = json.loads((tmp_path / "one/artifacts.json").read_text())
    assert all(
        hashlib.sha256((tmp_path / "one" / name).read_bytes()).hexdigest() == sha
        for name, sha in receipt["artifacts"].items()
    )
    with pytest.raises(ValueError, match="overwritten"):
        run_identity(source, tmp_path / "one", IdentityBudget())


def test_prediction_is_sealed_before_reveal(tmp_path):
    data = bundle("clean")
    heldout = data.measurements[-1]
    observed = data.observations[-1]
    context = PredictionContext(
        observation_id=observed.id,
        heldout_measurement_id=heldout.id,
        source=observed.source,
        dependence_group=observed.dependence_group,
        time=observed.time,
        observer_ref=observed.observer_ref,
    )
    output = tmp_path / "loo"

    def reveal():
        assert (output / "prediction.json").exists()
        assert not (output / "heldout.json").exists()
        return heldout

    result = leave_one_observation_out(
        LinearTangentBaseline(),
        data.observations[:-1],
        data.measurements[:-1],
        context,
        reveal,
        output,
    )
    assert result["status"] == "SUCCESS", result
    assert result["normalized_residual"] < 0.001
    assert result["training_ids"] == ["m0", "m1", "m2"]
    assert not result["calibrated"]
    assert "position" not in context.model_dump()


def test_sparse_loo_failure_retains_receipt_and_no_negative_evidence(tmp_path):
    result = run_loo(ROOT / "examples/identity/sparse.json", "h-one", "m1", tmp_path / "sparse")
    assert result["status"] == "UNSUPPORTED"
    assert result["normalized_residual"] is None
    assert (tmp_path / "sparse/run-receipt.json").exists()
    assert not (tmp_path / "sparse/heldout.json").exists()


def test_heldout_dependence_group_cannot_be_in_fit(tmp_path):
    data = bundle("clean")
    observed = data.observations[0]
    context = PredictionContext(
        observation_id="other-observation",
        heldout_measurement_id="other",
        source=observed.source,
        dependence_group=observed.dependence_group,
        time=observed.time,
    )
    with pytest.raises(ValueError, match="leaked"):
        leave_one_observation_out(
            LinearTangentBaseline(),
            data.observations,
            data.measurements,
            context,
            lambda: data.measurements[0],
            tmp_path / "bad",
        )


def test_large_jump_increases_heldout_residual(tmp_path):
    data = bundle("clean")
    heldout = data.measurements[-1].model_dump(mode="json")
    heldout["position"]["longitude_deg"] += 0.001
    from sieve.identity.models import Measurement

    observed = data.observations[-1]
    context = PredictionContext(
        observation_id=observed.id,
        heldout_measurement_id=heldout["id"],
        source=observed.source,
        dependence_group=observed.dependence_group,
        time=observed.time,
    )
    result = leave_one_observation_out(
        LinearTangentBaseline(),
        data.observations[:-1],
        data.measurements[:-1],
        context,
        lambda: Measurement.model_validate(heldout),
        tmp_path / "jump",
    )
    assert result["status"] == "SUCCESS"
    assert result["normalized_residual"] > 5


def test_prediction_covariance_includes_fit_uncertainty(tmp_path):
    result = run_loo(ROOT / "examples/identity/clean.json", "h-one", "m3", tmp_path / "loo")
    covariance = np.asarray(result["prediction"]["covariance"]["values"])
    assert np.linalg.eigvalsh(covariance).min() > 0.04


def test_stage_provenance_fit_and_predict_cannot_use_revealed_entity(tmp_path):
    run_loo(ROOT / "examples/identity/clean.json", "h-one", "m3", tmp_path / "stages")
    graph = json.loads((tmp_path / "stages/provenance.json").read_text())
    revealed = {e["id"] for e in graph["entities"] if e["kind"] == "heldout-evidence"}
    assert revealed
    for record in graph["records"]:
        if record["id"] in {"fit-stage", "predict-stage"}:
            assert not revealed & set(record["used_entity_ids"])
    receipt = json.loads((tmp_path / "stages/run-receipt.json").read_text())
    assert (
        receipt["configuration"]["sealed_prediction_sha256"]
        == hashlib.sha256((tmp_path / "stages/prediction.json").read_bytes()).hexdigest()
    )


def test_nonzero_epoch_uncertainty_is_not_silently_ignored(tmp_path):
    data = bundle("clean")
    observed = data.observations[-1]
    time = observed.time.model_copy(update={"uncertainty_seconds": 1.0})
    context = PredictionContext(
        observation_id=observed.id,
        heldout_measurement_id="m3",
        source=observed.source,
        dependence_group=observed.dependence_group,
        time=time,
    )
    result = leave_one_observation_out(
        LinearTangentBaseline(),
        data.observations[:-1],
        data.measurements[:-1],
        context,
        lambda: data.measurements[-1],
        tmp_path / "epoch",
    )
    assert result["status"] == "UNSUPPORTED"
    assert result["normalized_residual"] is None
    assert not (tmp_path / "epoch/heldout.json").exists()


def test_missing_quantities_cannot_falsify_a_hypothesis():
    data = bundle().model_dump(mode="json")
    factor = data["evidence_results"][0]["factors"][0]
    for quantity in factor["quantities"]:
        quantity["quantity"].update(value=None, uncertainty=None, missing_reason="No measurement")
    factor["score"] = None
    with pytest.raises(ValueError, match="explicit quantitative result"):
        IdentityInput.model_validate(data)


def test_loo_fits_only_the_conditional_object_slot(tmp_path):
    data = bundle("clean").model_dump(mode="json")
    one = next(h for h in data["hypotheses"] if h["id"] == "h-one")
    one["assignments"][0]["member_ids"] = ["m0", "m1", "m3"]
    one["unassigned_member_ids"] = ["m2"]
    data["measurements"][2]["position"]["longitude_deg"] += 0.005
    path = tmp_path / "input.json"
    path.write_text(IdentityInput.model_validate(data).model_dump_json())
    result = run_loo(path, "h-one", "m3", tmp_path / "run")
    assert result["status"] == "SUCCESS"
    assert result["training_ids"] == ["m0", "m1"]
    assert result["normalized_residual"] < 0.01
    with pytest.raises(ValueError, match="assigned to the requested object slot"):
        run_loo(path, "h-one", "m2", tmp_path / "unassigned")
    assert not (tmp_path / "unassigned").exists()


def test_empty_training_after_group_removal_has_unsupported_receipt(tmp_path):
    data = bundle("clean")
    heldout, observation = data.measurements[3], data.observations[3]
    context = PredictionContext(
        observation_id=observation.id,
        heldout_measurement_id=heldout.id,
        source=observation.source,
        dependence_group=observation.dependence_group,
        time=observation.time,
    )
    revealed = []
    result = leave_one_observation_out(
        LinearTangentBaseline(), (), (), context, lambda: revealed.append(True), tmp_path / "run"
    )
    assert result["status"] == "UNSUPPORTED"
    assert not revealed
    assert (tmp_path / "run/run-receipt.json").exists()
    assert (tmp_path / "run/artifacts.json").exists()


def test_renaming_shared_raw_bytes_cannot_manufacture_disjointness():
    data = bundle().model_dump(mode="json")
    source = next(e for e in data["provenance"]["entities"] if e["id"] == "raw-observations")
    alias = {**source, "id": "renamed-source"}
    data["provenance"]["entities"].append(alias)
    result = data["evidence_results"][0]
    result["request"]["input_ids"] = [alias["id"]]
    result["factors"][0]["input_ids"] = [alias["id"]]
    for record in data["provenance"]["records"]:
        if record["id"] in {"activity-request", "activity-evidence"}:
            record["used_entity_ids"] = [
                alias["id"] if key == "physical-test-input" else key
                for key in record["used_entity_ids"]
            ]
            record["derivations"] = {
                key: [
                    alias["id"] if parent == "physical-test-input" else parent for parent in parents
                ]
                for key, parents in record["derivations"].items()
            }
    with pytest.raises(ValueError, match="reuses generation source lineage"):
        IdentityInput.model_validate(data)


def test_observation_batch_validates_the_data_boundary():
    from sieve.interfaces import ObservationBatch

    data = bundle()
    batch = ObservationBatch(
        observations=data.observations, measurements=data.measurements, provenance=data.provenance
    )
    assert batch.observations == data.observations
    assert batch.measurements == data.measurements


@pytest.mark.parametrize("change", ["parent", "duplicate", "source_key", "raw_pin", "producer"])
def test_observation_batch_rejects_broken_intake_before_proposing(change):
    from sieve.interfaces import ObservationBatch

    data = bundle().model_dump(mode="json")
    if change == "parent":
        data["measurements"][0]["observation_id"] = "missing-observation"
    elif change == "duplicate":
        data["measurements"].append(copy.deepcopy(data["measurements"][0]))
    elif change == "source_key":
        data["observations"][1]["source"] = copy.deepcopy(data["observations"][0]["source"])
    elif change == "raw_pin":
        entity = next(e for e in data["provenance"]["entities"] if e["id"] == "raw-observations")
        entity.update(sha256=None, hash_missing_reason="Missing checksum")
    else:
        data["measurements"][0]["provenance_ref"] = "missing-producer"
    with pytest.raises(ValueError):
        ObservationBatch.model_validate(
            {
                "observations": data["observations"],
                "measurements": data["measurements"],
                "provenance": data["provenance"],
            }
        )


def test_collaborator_walkthrough_runs_and_keeps_ambiguity():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "examples/identity/plug_in_walkthrough.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["scientific_claim_status"] == "CONTRACT_ONLY"
    assert report["provider_measurements"] == 4
    assert report["hypotheses_before"] == report["hypotheses_after"]
    assert set(report["hypotheses_after"]) == {"h-one", "h-two", "h-unresolved"}
    assert report["source_ids"] == ["row0", "row1", "row2", "row3"]
    assert report["case_disposition"] == "UNRESOLVED"
    assert (
        sum(a["disposition"] == "FALSIFIED_UNDER_DECLARED_TEST" for a in report["assessments"]) == 1
    )


def test_walkthrough_unsupported_request_has_no_scientific_factor():
    from examples.identity.plug_in_walkthrough import FixtureEvidenceProvider

    data = bundle()
    request = data.evidence_results[0].request.model_copy(update={"hypothesis_id": "absent"})
    result = FixtureEvidenceProvider(data).evaluate(
        data.hypotheses[0], request, data.observations, data.measurements
    )
    assert result.status == "UNSUPPORTED"
    assert not result.factors
