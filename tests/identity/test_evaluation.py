from __future__ import annotations

import hashlib

import pytest

from evaluation.identity import (
    ComparisonProtocol,
    EvaluationCase,
    TruthSet,
    calibration_metrics,
    compare_paired,
    freeze_threshold,
    identity_metrics,
    physical_groups,
)
from evaluation.truth import TruthRecord
from sieve.contracts.models import MethodSpec, Score
from sieve.identity.models import IdentityPrediction, ObjectAssignment, PredictionTable
from sieve.provenance.models import Entity


def dataset(role="VALIDATION", n=2, prefix="val", split=False, correct=False, score=0.8):
    cases = []
    records = []
    predictions = []
    for g in range(n):
        a = (f"{prefix}:{g}:a0", f"{prefix}:{g}:a1")
        b = (f"{prefix}:{g}:b0", f"{prefix}:{g}:b1")
        for kind, members in (("clean", a), ("merge", a + b)):
            case_id = f"{prefix}:{g}:{kind}"
            cases.append(
                EvaluationCase(
                    id=case_id, member_ids=members, dependence_groups=(f"{prefix}:ex:{g}",)
                )
            )
            parts = (
                (a, b)
                if correct and kind == "merge"
                else ((a[:1], a[1:]) if split and kind == "clean" else (members,))
            )
            predictions.append(
                IdentityPrediction(
                    case_id=case_id,
                    hypothesis_id="conditional:" + case_id,
                    member_ids=members,
                    assignments=tuple(
                        ObjectAssignment(slot_id=f"slot:{i}", member_ids=m)
                        for i, m in enumerate(parts)
                    ),
                    score=Score(
                        value=score,
                        semantics="RANKING_ONLY",
                        direction="higher",
                        domain="contract-fixture",
                        event_or_null="identity consistency",
                    ),
                    state="PROPOSED",
                    inspected=True,
                    raw_output_ref="raw:" + case_id,
                )
            )
        for i, m in enumerate(a + b):
            records.append(
                TruthRecord(
                    id="truth:" + m,
                    source_type="SIMULATED_TRUTH",
                    source_ref="fixture",
                    task_scope="membership",
                    member_ids=(m,),
                    object_ids=(f"{prefix}:physical:{g}:{'A' if i < 2 else 'B'}",),
                    frozen=True,
                    complete_for_scope=True,
                    limitations=("Contract test only",),
                )
            )
    sha = hashlib.sha256(prefix.encode()).hexdigest()
    truth = TruthSet(
        role=role,
        task_scope="membership",
        input_sha256=sha,
        cases=tuple(cases),
        records=tuple(records),
        limitations=("Synthetic evaluator inputs, not scientific evidence",),
    )
    method = "treatment" if correct else "baseline"
    table = PredictionTable(
        method=MethodSpec(name=method, version="v1"),
        input_sha256=sha,
        astrometry_config_sha256="a" * 64,
        predictions=tuple(predictions),
        factor_protocol_refs=("single-factor",) if correct else (),
        control_artifact_refs=("factor-control",) if correct else (),
        evidence_protocol_artifacts=(
            Entity(id="single-factor", kind="test-protocol", sha256="b" * 64),
            Entity(id="factor-control", kind="control", sha256="c" * 64),
        ),
        runtime_seconds=1.0,
        peak_memory_bytes=1000,
        limitations=("Manually authored evaluator test predictions",),
    )
    return truth, table


def protocol():
    truth, base = dataset()
    _, treatment = dataset(correct=True)
    return ComparisonProtocol(
        baseline=freeze_threshold(base, truth, 1.0),
        treatment=freeze_threshold(treatment, truth, 1.0),
        physical_factor_ref="single-factor",
        physical_factor_sha256="b" * 64,
        control_artifact_ref="factor-control",
        control_artifact_sha256="c" * 64,
        bootstrap_replicates=100,
        limitations=("Only checks evaluator arithmetic; no treatment implementation",),
    )


def test_false_merge_recall_and_purity_denominators():
    truth, base = dataset()
    threshold = freeze_threshold(base, truth, 1.0)
    metrics = identity_metrics(base, truth, threshold)
    assert metrics["true_link_recall"] == 1
    assert metrics["false_merge_fraction"] == 1
    assert metrics["merged_slot_fraction"] == 0.5
    assert metrics["link_purity"] == pytest.approx(3 / 7)
    assert metrics["counts"]["multi_object_cases"] == 2
    assert metrics["counts"]["inspected_cases"] == 4


def test_true_pairs_are_not_credited_across_competing_candidate_cases():
    truth, base = dataset(n=1)
    metrics = identity_metrics(base, truth, freeze_threshold(base, truth, 1.0))
    assert metrics["counts"]["true_links"] == 3  # one clean pair + two within mixed candidate


def test_split_metric_is_conditional_and_recall_includes_missing_links():
    truth, table = dataset(split=True)
    threshold = freeze_threshold(table, truth, 0.5)
    result = identity_metrics(table, truth, threshold)
    assert result["true_link_recall"] == pytest.approx(2 / 3)
    assert result["false_split_fraction"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="unattainable"):
        freeze_threshold(table, truth, 1.0)


def test_no_threshold_selection_on_heldout_truth():
    truth, table = dataset(role="HELD_OUT")
    with pytest.raises(ValueError, match="held-out truth"):
        freeze_threshold(table, truth, 1.0)


def test_paired_bootstrap_groups_physical_sources_and_shared_exposures():
    truth, base = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    assert len(physical_groups(truth)) == 8
    assert all(len(g) == 2 for g in physical_groups(truth))
    result = compare_paired(base, treatment, truth, protocol())
    assert result["matched_recall"]
    assert result["false_merge_reduction"] == 1
    assert result["paired_source_group_bootstrap95"] == [1.0, 1.0]
    assert result == compare_paired(base, treatment, truth, protocol())


def test_too_few_sources_refuses_confident_go():
    truth, base = dataset(role="HELD_OUT", n=2, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=2, prefix="test", correct=True)
    result = compare_paired(base, treatment, truth, protocol())
    assert result["matched_recall"]
    assert result["verdict"] == "UNQUALIFIED"
    assert result["paired_source_group_bootstrap95"] is None


def test_mismatched_recall_cannot_claim_physical_factor_benefit():
    truth, base = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    data = treatment.model_dump(mode="json")
    data["predictions"][0]["test_eligible"] = False
    data["predictions"][0]["eligibility_reason"] = "rejected by contract test gate"
    result = compare_paired(base, PredictionTable.model_validate(data), truth, protocol())
    assert result["verdict"] == "UNMATCHED_RECALL"
    assert not result["matched_recall"]


def test_negative_treatment_is_not_decorated():
    truth, base = dataset(role="HELD_OUT", n=8, prefix="test")
    data = base.model_dump(mode="json")
    data.update(
        method=dict(name="treatment", version="v1"),
        factor_protocol_refs=["single-factor"],
        control_artifact_refs=["factor-control"],
    )
    result = compare_paired(base, PredictionTable.model_validate(data), truth, protocol())
    assert result["verdict"] == "NO_GO_OR_INCONCLUSIVE"
    assert result["false_merge_reduction"] == 0


def test_extra_factor_and_missing_controls_refused():
    truth, base = dataset(role="HELD_OUT", prefix="test")
    _, treatment = dataset(role="HELD_OUT", prefix="test", correct=True)
    data = treatment.model_dump(mode="json")
    data["factor_protocol_refs"].append("second-modality")
    with pytest.raises(ValueError, match="exactly the single"):
        compare_paired(base, PredictionTable.model_validate(data), truth, protocol())
    data["factor_protocol_refs"].pop()
    data["control_artifact_refs"] = []
    with pytest.raises(ValueError, match="controls"):
        compare_paired(base, PredictionTable.model_validate(data), truth, protocol())


def test_cross_split_source_leakage_refused():
    truth, base = dataset(role="HELD_OUT", prefix="val")
    _, treatment = dataset(role="HELD_OUT", prefix="val", correct=True)
    changed = TruthSet.model_validate({**truth.model_dump(mode="json"), "input_sha256": "1" * 64})
    base = PredictionTable.model_validate(
        {**base.model_dump(mode="json"), "input_sha256": "1" * 64}
    )
    treatment = PredictionTable.model_validate(
        {**treatment.model_dump(mode="json"), "input_sha256": "1" * 64}
    )
    with pytest.raises(ValueError, match="physical source leaked"):
        compare_paired(base, treatment, changed, protocol())


def test_missing_case_and_member_cannot_raise_apparent_recall():
    truth, table = dataset()
    data = table.model_dump(mode="json")
    data["predictions"].pop()
    with pytest.raises(ValueError, match="all cases"):
        freeze_threshold(PredictionTable.model_validate(data), truth, 1.0)
    data = table.model_dump(mode="json")
    data["predictions"][0]["member_ids"].pop()
    data["predictions"][0]["assignments"][0]["member_ids"].pop()
    with pytest.raises(ValueError, match="remove"):
        freeze_threshold(PredictionTable.model_validate(data), truth, 1.0)


def test_uncalibrated_rank_cannot_receive_calibration_metrics():
    truth, table = dataset()
    assert calibration_metrics(table, truth)["status"] == "NOT_APPLICABLE"


def test_baseline_cannot_contain_a_physical_factor():
    truth, baseline = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    baseline = baseline.model_copy(update={"factor_protocol_refs": ("prior-physical",)})
    with pytest.raises(ValueError, match="astrometry-only"):
        compare_paired(baseline, treatment, truth, protocol())


def test_factor_control_hashes_must_match_frozen_protocol():
    truth, baseline = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    altered = protocol().model_copy(update={"physical_factor_sha256": "d" * 64})
    with pytest.raises(ValueError, match="bytes differ"):
        compare_paired(baseline, treatment, truth, altered)


def test_evaluator_cli_exports_all_cases_and_failures(tmp_path):
    import json

    from evaluation.identity_cli import main

    truth, baseline = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    for name, obj in (
        ("truth", truth),
        ("baseline", baseline),
        ("treatment", treatment),
        ("protocol", protocol()),
    ):
        (tmp_path / (name + ".json")).write_text(obj.model_dump_json())
    protocol_hash = hashlib.sha256((tmp_path / "protocol.json").read_bytes()).hexdigest()
    args = ["compare"]
    for name in ("truth", "baseline", "treatment", "protocol"):
        args.extend(["--" + name, str(tmp_path / (name + ".json"))])
    args.extend(["--protocol-sha256", protocol_hash, "--output", str(tmp_path / "run")])
    assert main(args) == 0
    rows = [json.loads(line) for line in (tmp_path / "run/per-case.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    failures = [
        json.loads(line)
        for line in (tmp_path / "run/failure-examples.jsonl").read_text().splitlines()
    ]
    assert len(failures) == 8
    assert all(row["failure_reasons"] == ["baseline:false_merge"] for row in failures)
    assert (
        json.loads((tmp_path / "run/paired-metrics.json").read_text())["verdict"]
        == "CONTRACT_CHECK_PASSED"
    )
    hashes = json.loads((tmp_path / "run/artifact-hashes.json").read_text())
    assert all(
        hashlib.sha256((tmp_path / "run" / p).read_bytes()).hexdigest() == h
        for p, h in hashes.items()
    )
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    args[-1] = str(tmp_path / "different")
    args[args.index("--protocol-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert not (tmp_path / "different").exists()


def test_evaluator_cli_freeze_does_not_accept_heldout_or_overwrite(tmp_path):
    from evaluation.identity_cli import main

    truth, table = dataset()
    (tmp_path / "truth.json").write_text(truth.model_dump_json())
    (tmp_path / "predictions.json").write_text(table.model_dump_json())
    args = [
        "freeze",
        "--truth",
        str(tmp_path / "truth.json"),
        "--predictions",
        str(tmp_path / "predictions.json"),
        "--target-recall",
        "1",
        "--output",
        str(tmp_path / "frozen.json"),
    ]
    assert main(args) == 0
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    truth = truth.model_copy(update={"role": "HELD_OUT"})
    (tmp_path / "truth.json").write_text(truth.model_dump_json())
    args[-1] = str(tmp_path / "heldout.json")
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert not (tmp_path / "heldout.json").exists()


def test_cross_split_member_reuse_refused_even_with_changed_truth_labels():
    truth, baseline = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    frozen = protocol()
    frozen = frozen.model_copy(
        update={
            "baseline": frozen.baseline.model_copy(update={"validation_member_ids": ("test:0:a0",)})
        }
    )
    with pytest.raises(ValueError, match="measurement member leaked"):
        compare_paired(baseline, treatment, truth, frozen)


def abstain(table, state="UNRESOLVED"):
    rows = tuple(
        IdentityPrediction(
            case_id=row.case_id,
            member_ids=row.member_ids,
            state=state,
            inspected=row.inspected,
            raw_output_ref=row.raw_output_ref,
        )
        for row in table.predictions
    )
    return PredictionTable.model_validate({**table.model_dump(mode="json"), "predictions": rows})


@pytest.mark.parametrize("state", ["UNRESOLVED", "UNSUPPORTED"])
def test_all_abstentions_keep_denominators_and_cannot_pass_matched_recall(state):
    truth, baseline = dataset(role="HELD_OUT", n=8, prefix="test")
    _, treatment = dataset(role="HELD_OUT", n=8, prefix="test", correct=True)
    treatment = abstain(treatment, state)
    result = compare_paired(baseline, treatment, truth, protocol())
    metrics = result["treatment"]
    assert metrics["true_link_recall"] == 0
    assert metrics["counts"]["cases"] == 16
    assert metrics["counts"]["abstained_cases"] == 16
    assert metrics["counts"]["true_links"] == 24
    assert metrics["counts"]["multi_object_cases"] == 8
    assert metrics["counts"]["inspected_cases"] == 16
    assert metrics["link_purity"] is None
    assert result["verdict"] == "UNMATCHED_RECALL"
    assert calibration_metrics(treatment, truth)["status"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("field", ["name", "version"])
def test_all_abstentions_cannot_change_frozen_method_identity(field):
    truth, table = dataset()
    frozen = freeze_threshold(table, truth, 1)
    table = abstain(table)
    table = table.model_copy(update={"method": table.method.model_copy(update={field: "changed"})})
    with pytest.raises(ValueError, match="method/version mismatch"):
        identity_metrics(table, truth, frozen)


def test_threshold_cannot_be_selected_from_all_abstentions():
    truth, table = dataset()
    with pytest.raises(ValueError, match="no defined scored candidates"):
        freeze_threshold(abstain(table), truth, 1)
