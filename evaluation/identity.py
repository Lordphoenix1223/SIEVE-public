"""Evaluator-only physical identity metrics, validation freeze and paired grouped CI.

Nothing under sieve imports this module. Frozen thresholds cannot be selected on
HELD_OUT truth. Rates are conditional on the declared shared candidate universe.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from evaluation.truth import TruthRecord, benchmark_eligible
from sieve.contracts.models import Finite, Identifier, ScoreSemantics, Text
from sieve.identity.models import IdentityContract, IdentityPrediction, PredictionTable, unique
from sieve.provenance.models import content_id


class EvaluationCase(IdentityContract):
    id: Identifier
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    dependence_groups: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_members(self) -> Self:
        unique(self.member_ids, "case members")
        unique(self.dependence_groups, "case dependence groups")
        return self


class TruthSet(IdentityContract):
    role: Literal["VALIDATION", "HELD_OUT"]
    task_scope: Literal["membership", "catalog_identity"]
    input_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)
    records: tuple[TruthRecord, ...] = Field(min_length=1)
    limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_truth(self) -> Self:
        unique(tuple(c.id for c in self.cases), "truth case IDs")
        unique(tuple(r.id for r in self.records), "truth record IDs")
        members = tuple(m for r in self.records for m in r.member_ids)
        unique(members, "truth membership records")
        if set(members) != {m for c in self.cases for m in c.member_ids}:
            raise ValueError("truth must cover exactly the declared candidate members")
        if any(
            not benchmark_eligible(r, self.task_scope) or len(r.object_ids) > 1
            for r in self.records
        ):
            raise ValueError("identity evaluator requires admissible complete single-source truth")
        return self


def truth_mapping(truth: TruthSet) -> dict[str, str | None]:
    return {
        m: (r.object_ids[0] if r.object_ids else None) for r in truth.records for m in r.member_ids
    }


class ScoreDefinition(IdentityContract):
    method_name: Identifier
    method_version: Text
    semantics: ScoreSemantics
    direction: Literal["higher", "lower"]
    domain: Text
    event_or_null: Text
    calibration_ref: Identifier | None


class FrozenThreshold(IdentityContract):
    definition: ScoreDefinition
    threshold: Finite
    target_true_link_recall: Annotated[Finite, Field(gt=0, le=1)]
    validation_true_link_recall: Annotated[Finite, Field(ge=0, le=1)]
    astrometry_config_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    validation_input_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    validation_truth_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    validation_prediction_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    validation_object_ids: tuple[Identifier, ...]
    validation_member_ids: tuple[Identifier, ...]
    validation_dependence_groups: tuple[Identifier, ...]
    selection_rule: Literal["least_permissive_validation_threshold_meeting_recall"] = (
        "least_permissive_validation_threshold_meeting_recall"
    )


class ComparisonProtocol(IdentityContract):
    baseline: FrozenThreshold
    treatment: FrozenThreshold
    recall_tolerance: Annotated[Finite, Field(ge=0, le=0.05)] = 0.0
    minimum_absolute_merge_reduction: Annotated[Finite, Field(gt=0, le=1)] = 0.05
    bootstrap_replicates: int = Field(default=2000, ge=100, le=10000)
    bootstrap_seed: int = 42
    minimum_independent_groups: int = Field(default=8, ge=2, le=1000)
    physical_factor_ref: Identifier
    physical_factor_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    added_physical_factor_count: Literal[1] = 1
    control_artifact_ref: Identifier
    control_artifact_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    qualification: Literal["CONTRACT_ONLY", "EXTERNAL_RESEARCH"] = "CONTRACT_ONLY"
    limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def paired_validation(self) -> Self:
        if self.baseline.astrometry_config_sha256 != self.treatment.astrometry_config_sha256:
            raise ValueError("treatment must retain the frozen baseline astrometry configuration")
        if self.baseline.validation_input_sha256 != self.treatment.validation_input_sha256:
            raise ValueError("thresholds must use identical validation candidates")
        if self.baseline.validation_truth_sha256 != self.treatment.validation_truth_sha256:
            raise ValueError("thresholds must use identical validation truth")
        if self.baseline.target_true_link_recall != self.treatment.target_true_link_recall:
            raise ValueError("both methods must target the same validation recall")
        if self.baseline.definition.method_name == self.treatment.definition.method_name:
            raise ValueError("paired comparison needs distinct declared methods")
        return self


def score_definition(table: PredictionTable) -> ScoreDefinition:
    scores = [p.score for p in table.predictions if p.score is not None]
    if not scores:
        raise ValueError("method has no defined scored candidates")
    definitions = [
        ScoreDefinition(
            method_name=table.method.name,
            method_version=table.method.version,
            semantics=s.semantics,
            direction=s.direction,
            domain=s.domain,
            event_or_null=s.event_or_null,
            calibration_ref=s.calibration_ref,
        )
        for s in scores
    ]
    if any(d != definitions[0] for d in definitions):
        raise ValueError("mixed score cohorts cannot share a threshold")
    return definitions[0]


def _check_table(table: PredictionTable, truth: TruthSet) -> None:
    if table.input_sha256 != truth.input_sha256:
        raise ValueError("prediction/input identity mismatch")
    rows = {p.case_id: p for p in table.predictions}
    if rows.keys() != {c.id for c in truth.cases}:
        raise ValueError("all cases require an explicit prediction or abstention")
    if any(set(rows[c.id].member_ids) != set(c.member_ids) for c in truth.cases):
        raise ValueError("method cannot silently remove candidate measurements")


def accepted(prediction: IdentityPrediction, threshold: FrozenThreshold) -> bool:
    score = prediction.score
    if (
        prediction.state != "PROPOSED"
        or not prediction.test_eligible
        or score is None
        or score.value is None
    ):
        return False
    return (
        score.value >= threshold.threshold
        if threshold.definition.direction == "higher"
        else score.value <= threshold.threshold
    )


def _counts(
    table: PredictionTable,
    truth: TruthSet,
    threshold: FrozenThreshold,
    case_ids: set[str] | None = None,
) -> dict[str, float]:
    mapping = truth_mapping(truth)
    rows = {p.case_id: p for p in table.predictions}
    counts = dict(
        true_links=0.0,
        recovered_true_links=0.0,
        claimed_links=0.0,
        merged_slots=0.0,
        multi_object_cases=0.0,
        false_merge_cases=0.0,
        claimed_slots=0.0,
        contaminated_slots=0.0,
        split_objects=0.0,
        covered_objects=0.0,
        expected_objects=0.0,
        cases=0.0,
        accepted_cases=0.0,
        abstained_cases=0.0,
        inspected_cases=0.0,
        heldout_error_sum=0.0,
        heldout_error_count=0.0,
    )
    for case in truth.cases:
        if case_ids is not None and case.id not in case_ids:
            continue
        prediction = rows[case.id]
        counts["cases"] += 1
        counts["inspected_cases"] += prediction.inspected
        object_counts = Counter(mapping[m] for m in case.member_ids if mapping[m] is not None)
        real_pair_count = sum(n * (n - 1) // 2 for n in object_counts.values())
        counts["true_links"] += real_pair_count
        truth_objects = {mapping[m] for m in case.member_ids if mapping[m] is not None}
        counts["expected_objects"] += len(truth_objects)
        counts["multi_object_cases"] += len(truth_objects) > 1
        if prediction.heldout_normalized_residual is not None:
            counts["heldout_error_sum"] += prediction.heldout_normalized_residual
            counts["heldout_error_count"] += 1
        if not accepted(prediction, threshold):
            counts["abstained_cases"] += 1
            continue
        counts["accepted_cases"] += 1
        counts["claimed_links"] += sum(
            len(slot.member_ids) * (len(slot.member_ids) - 1) // 2
            for slot in prediction.assignments
        )
        for slot in prediction.assignments:
            shared = Counter(mapping[m] for m in slot.member_ids if mapping[m] is not None)
            counts["recovered_true_links"] += sum(n * (n - 1) // 2 for n in shared.values())
        slots_by_object: dict[str, set[str]] = defaultdict(set)
        members_by_object: dict[str, set[str]] = defaultdict(set)
        merged_case = False
        for slot in prediction.assignments:
            counts["claimed_slots"] += 1
            objects = {mapping[m] for m in slot.member_ids if mapping[m] is not None}
            counts["merged_slots"] += len(objects) > 1
            merged_case = merged_case or len(objects) > 1
            counts["contaminated_slots"] += any(mapping[m] is None for m in slot.member_ids)
            for m in slot.member_ids:
                object_id = mapping[m]
                if object_id is not None:
                    slots_by_object[object_id].add(slot.slot_id)
                    members_by_object[object_id].add(m)
        counts["false_merge_cases"] += merged_case
        for object_id, members in members_by_object.items():
            if len(members) >= 2:
                counts["covered_objects"] += 1
                counts["split_objects"] += len(slots_by_object[object_id]) > 1

    return counts


def _rates(c: dict[str, float]) -> dict[str, Any]:
    def ratio(numerator: str, denominator: str) -> float | None:
        return c[numerator] / c[denominator] if c[denominator] else None

    return dict(
        counts=c,
        true_link_recall=ratio("recovered_true_links", "true_links"),
        false_merge_fraction=ratio("false_merge_cases", "multi_object_cases"),
        merged_slot_fraction=ratio("merged_slots", "claimed_slots"),
        false_split_fraction=ratio("split_objects", "covered_objects"),
        link_purity=ratio("recovered_true_links", "claimed_links"),
        artifact_contamination_fraction=ratio("contaminated_slots", "claimed_slots"),
        accepted_case_fraction=ratio("accepted_cases", "cases"),
        mean_heldout_normalized_residual=ratio("heldout_error_sum", "heldout_error_count"),
    )


def identity_metrics(
    table: PredictionTable, truth: TruthSet, threshold: FrozenThreshold
) -> dict[str, Any]:
    """Evaluate a frozen operating point, retaining even an all-abstention method."""
    _check_table(table, truth)
    if table.astrometry_config_sha256 != threshold.astrometry_config_sha256:
        raise ValueError("method changed frozen astrometry configuration")
    if (table.method.name, table.method.version) != (
        threshold.definition.method_name,
        threshold.definition.method_version,
    ):
        raise ValueError("frozen threshold method/version mismatch")
    # An abstaining method has no scores to compare. Its missing links still count;
    # validation threshold fitting continues to require scored candidates.
    if any(p.score is not None for p in table.predictions) and (
        score_definition(table) != threshold.definition
    ):
        raise ValueError("frozen threshold score definition mismatch")
    result = _rates(_counts(table, truth, threshold))
    result.update(runtime_seconds=table.runtime_seconds, peak_memory_bytes=table.peak_memory_bytes)
    return result


def freeze_threshold(
    table: PredictionTable, truth: TruthSet, target_recall: float
) -> FrozenThreshold:
    if truth.role != "VALIDATION":
        raise ValueError("threshold selection cannot use held-out truth")
    if not 0 < target_recall <= 1:
        raise ValueError("target recall out of range")
    _check_table(table, truth)
    definition = score_definition(table)
    values = sorted(
        {p.score.value for p in table.predictions if p.score and p.score.value is not None},
        reverse=definition.direction == "higher",
    )
    for value in values:
        threshold = FrozenThreshold(
            definition=definition,
            threshold=value,
            target_true_link_recall=target_recall,
            validation_true_link_recall=0.0,
            astrometry_config_sha256=table.astrometry_config_sha256,
            validation_input_sha256=truth.input_sha256,
            validation_truth_sha256=content_id("truth", truth.model_dump(mode="json")).split(":")[
                1
            ],
            validation_prediction_sha256=content_id(
                "predictions", table.model_dump(mode="json")
            ).split(":")[1],
            validation_object_ids=tuple(sorted({o for r in truth.records for o in r.object_ids})),
            validation_member_ids=tuple(sorted({m for c in truth.cases for m in c.member_ids})),
            validation_dependence_groups=tuple(
                sorted({g for c in truth.cases for g in c.dependence_groups})
            ),
        )
        recall = _rates(_counts(table, truth, threshold))["true_link_recall"]
        if recall is not None and recall >= target_recall:
            return FrozenThreshold.model_validate(
                {**threshold.model_dump(mode="json"), "validation_true_link_recall": recall}
            )
    raise ValueError(
        "target true-link recall unattainable on validation; no threshold manufactured"
    )


def physical_groups(truth: TruthSet) -> tuple[set[str], ...]:
    """Connected case/source/dependence incidence groups, including hard merges."""
    mapping = truth_mapping(truth)
    parents = {c.id: c.id for c in truth.cases}

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen: dict[tuple[str, str], str] = {}
    for case in truth.cases:
        object_ids = [obj for m in case.member_ids if isinstance(obj := mapping[m], str)]
        keys = (
            [("member", m) for m in case.member_ids]
            + [("object", obj) for obj in object_ids]
            + [("dependence", g) for g in case.dependence_groups]
        )
        for key in keys:
            if key in seen:
                a, b = sorted((root(case.id), root(seen[key])))
                parents[b] = a
            seen[key] = case.id
    groups: dict[str, set[str]] = defaultdict(set)
    for case in truth.cases:
        groups[root(case.id)].add(case.id)
    return tuple(groups[key] for key in sorted(groups))


def compare_paired(
    baseline: PredictionTable,
    treatment: PredictionTable,
    truth: TruthSet,
    protocol: ComparisonProtocol,
) -> dict[str, Any]:
    if truth.role != "HELD_OUT":
        raise ValueError("go/no-go comparison requires held-out truth")
    if baseline.factor_protocol_refs:
        raise ValueError("astrometry-only baseline cannot include physical factor protocols")
    added = set(treatment.factor_protocol_refs) - set(baseline.factor_protocol_refs)
    if added != {protocol.physical_factor_ref} or not set(baseline.factor_protocol_refs) <= set(
        treatment.factor_protocol_refs
    ):
        raise ValueError("treatment must add exactly the single frozen physical factor protocol")
    if protocol.control_artifact_ref not in treatment.control_artifact_refs:
        raise ValueError("treatment lacks its declared physical-factor controls")
    artifact_pins = {e.id: e.sha256 for e in treatment.evidence_protocol_artifacts}
    if (
        artifact_pins.get(protocol.physical_factor_ref) != protocol.physical_factor_sha256
        or artifact_pins.get(protocol.control_artifact_ref) != protocol.control_artifact_sha256
    ):
        raise ValueError("physical factor/control bytes differ from frozen protocol")
    validation_objects = set(protocol.baseline.validation_object_ids) | set(
        protocol.treatment.validation_object_ids
    )
    validation_groups = set(protocol.baseline.validation_dependence_groups) | set(
        protocol.treatment.validation_dependence_groups
    )
    if truth.input_sha256 == protocol.baseline.validation_input_sha256:
        raise ValueError("validation input cannot be held-out input")
    validation_members = set(protocol.baseline.validation_member_ids) | set(
        protocol.treatment.validation_member_ids
    )
    if validation_objects & {o for r in truth.records for o in r.object_ids}:
        raise ValueError("physical source leaked across validation and holdout")
    if validation_groups & {g for c in truth.cases for g in c.dependence_groups}:
        raise ValueError("observation dependence group leaked across split")
    if validation_members & {m for c in truth.cases for m in c.member_ids}:
        raise ValueError("measurement member leaked across validation and holdout")
    b = identity_metrics(baseline, truth, protocol.baseline)
    t = identity_metrics(treatment, truth, protocol.treatment)
    rb, rt = b["true_link_recall"], t["true_link_recall"]
    matched = (
        rb is not None
        and rt is not None
        and abs(rb - rt) <= protocol.recall_tolerance
        and min(rb, rt) >= protocol.baseline.target_true_link_recall - protocol.recall_tolerance
    )
    groups = physical_groups(truth)
    interval = None
    bootstrap = []
    if len(groups) >= protocol.minimum_independent_groups:
        bc = [_counts(baseline, truth, protocol.baseline, g) for g in groups]
        tc = [_counts(treatment, truth, protocol.treatment, g) for g in groups]
        rng = np.random.default_rng(protocol.bootstrap_seed)
        for _ in range(protocol.bootstrap_replicates):
            draw = rng.integers(0, len(groups), len(groups))
            ba = _rates({k: sum(bc[i][k] for i in draw) for k in bc[0]})
            ta = _rates({k: sum(tc[i][k] for i in draw) for k in tc[0]})
            if ba["false_merge_fraction"] is not None and ta["false_merge_fraction"] is not None:
                bootstrap.append(ba["false_merge_fraction"] - ta["false_merge_fraction"])
        if len(bootstrap) >= 0.9 * protocol.bootstrap_replicates:
            interval = [float(v) for v in np.quantile(bootstrap, [0.025, 0.975])]
    reduction = (
        b["false_merge_fraction"] - t["false_merge_fraction"]
        if b["false_merge_fraction"] is not None and t["false_merge_fraction"] is not None
        else None
    )
    verdict = "UNQUALIFIED"
    if not matched:
        verdict = "UNMATCHED_RECALL"
    elif interval is not None and reduction is not None:
        verdict = (
            "GO_CONDITIONAL"
            if reduction >= protocol.minimum_absolute_merge_reduction
            and interval[0] >= protocol.minimum_absolute_merge_reduction
            else "NO_GO_OR_INCONCLUSIVE"
        )
    if verdict == "GO_CONDITIONAL" and protocol.qualification == "CONTRACT_ONLY":
        verdict = "CONTRACT_CHECK_PASSED"
    return dict(
        schema_version="sieve.identity.evaluation.v1",
        baseline=b,
        treatment=t,
        matched_recall=matched,
        recall_gap=abs(rb - rt) if rb is not None and rt is not None else None,
        false_merge_reduction=reduction,
        paired_source_group_bootstrap95=interval,
        independent_groups=len(groups),
        bootstrap_defined_replicates=len(bootstrap),
        verdict=verdict,
        protocol_sha256=content_id("protocol", protocol.model_dump(mode="json")).split(":")[1],
        limitations=list(truth.limitations) + list(protocol.limitations),
        scientific_claim_status=(
            "NO_THESIS_EVIDENCE_CONTRACT_ONLY"
            if protocol.qualification == "CONTRACT_ONLY"
            else "CONDITIONAL_ON_EXTERNAL_FACTOR_AND_DATA_QUALIFICATION"
        ),
        review_burden=dict(
            baseline=b["counts"]["inspected_cases"], treatment=t["counts"]["inspected_cases"]
        ),
    )


def calibration_metrics(table: PredictionTable, truth: TruthSet) -> dict[str, Any]:
    _check_table(table, truth)
    mapping = truth_mapping(truth)
    artifacts = {e.id: e for e in table.calibration_artifacts}
    outcomes = []
    probabilities = []
    for prediction in table.predictions:
        score = prediction.score
        if (
            score is None
            or score.semantics != ScoreSemantics.CALIBRATED_PROBABILITY
            or not score.calibrated
            or score.value is None
            or score.event_or_null != "all_assigned_members_share_one_physical_object"
            or score.calibration_ref not in artifacts
            or not artifacts[score.calibration_ref].sha256
            or len(prediction.assignments) != 1
            or len(prediction.assignments[0].member_ids) < 2
        ):
            return dict(
                status="NOT_APPLICABLE", reason="No complete audited probabilistic event output"
            )
        members = prediction.assignments[0].member_ids
        objects = {mapping[m] for m in members}
        outcomes.append(float(len(objects) == 1 and None not in objects))
        probabilities.append(score.value)
    if not outcomes:
        return dict(status="NOT_APPLICABLE", reason="No probabilistic observations")
    return dict(
        status="CONDITIONAL_ON_DECLARED_CALIBRATION",
        n=len(outcomes),
        brier=float(np.mean((np.asarray(probabilities) - np.asarray(outcomes)) ** 2)),
        calibration_refs=sorted(artifacts),
        limitations=["Artifact presence is not independent calibration verification"],
    )
