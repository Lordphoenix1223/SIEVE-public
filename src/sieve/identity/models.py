"""Identity v1 composes the frozen v0 unit/time/score/provenance primitives.

A hypothesis is a conditional partition, never a forced physical identity.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sieve.contracts.models import (
    ExecutionStatus,
    Finite,
    Identifier,
    MethodSpec,
    ObservationTime,
    PositionCovariance,
    Quality,
    QuantityValue,
    Score,
    SkyPosition,
    Text,
    check_finite,
)
from sieve.provenance.models import Entity, ProvenanceGraph


class IdentityContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )
    schema_version: Literal["sieve.identity.v1"] = "sieve.identity.v1"

    @model_validator(mode="after")
    def finite_json(self) -> Self:
        check_finite(self.model_dump(mode="python"))
        return self


def unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")


class SourceKey(IdentityContract):
    provider: Text
    dataset: Text
    release: Text
    exposure_id: Identifier
    source_id: Identifier


class Observation(IdentityContract):
    id: Identifier
    source: SourceKey
    time: ObservationTime
    pointing: SkyPosition | None = None
    observer_ref: Identifier | None = None
    dependence_group: Identifier
    bandpass_ref: Identifier | None = None
    quality: Quality = Field(default_factory=Quality)
    raw_asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    provenance_ref: Identifier

    @model_validator(mode="after")
    def unique_assets(self) -> Self:
        unique(self.raw_asset_ids, "observation raw assets")
        return self


class ScalarMeasurement(IdentityContract):
    name: Identifier
    quantity: QuantityValue


class Measurement(IdentityContract):
    id: Identifier
    observation_id: Identifier
    raw_source_id: Identifier
    position: SkyPosition | None = None
    covariance: PositionCovariance | None = None
    quantities: tuple[ScalarMeasurement, ...] = ()
    method: MethodSpec
    quality: Quality = Field(default_factory=Quality)
    raw_output_asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    provenance_ref: Identifier

    @model_validator(mode="after")
    def coherent_measurement(self) -> Self:
        unique(tuple(q.name for q in self.quantities), "quantity names")
        unique(self.raw_output_asset_ids, "measurement raw outputs")
        if self.position is None and self.covariance is not None:
            raise ValueError("position covariance requires position")
        return self


# Same explicit measured-record contract, not an automatic detection or truth claim.
Detection = Measurement


class CandidateProposal(IdentityContract):
    id: Identifier
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    method: MethodSpec
    generation_input_ids: tuple[Identifier, ...] = Field(min_length=1)
    raw_output_asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    score: Score | None = None
    rationale: Text
    provenance_ref: Identifier

    @model_validator(mode="after")
    def unique_support(self) -> Self:
        for name in ("member_ids", "generation_input_ids", "raw_output_asset_ids"):
            unique(getattr(self, name), name)
        return self


class ObjectAssignment(IdentityContract):
    slot_id: Identifier  # local conditional object slot; not evaluator object ID
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    known_record_ref: Identifier | None = None

    @model_validator(mode="after")
    def unique_members(self) -> Self:
        unique(self.member_ids, "assignment members")
        return self


class Explanation(StrEnum):
    ARTIFACT = "ARTIFACT"
    ONE_OBJECT = "ONE_OBJECT"
    MULTIPLE_OBJECTS = "MULTIPLE_OBJECTS"
    KNOWN_MATCH = "KNOWN_MATCH"
    UNRESOLVED = "UNRESOLVED"


class ObjectHypothesis(IdentityContract):
    id: Identifier
    proposal_ids: tuple[Identifier, ...] = Field(min_length=1)
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    explanation: Explanation
    assignments: tuple[ObjectAssignment, ...] = ()
    unassigned_member_ids: tuple[Identifier, ...] = ()
    generation_input_ids: tuple[Identifier, ...] = Field(min_length=1)
    rationale: Text
    provenance_ref: Identifier

    @property
    def object_count(self) -> int | None:
        return None if self.explanation == Explanation.UNRESOLVED else len(self.assignments)

    @model_validator(mode="after")
    def complete_conditional_partition(self) -> Self:
        for name in ("proposal_ids", "member_ids", "unassigned_member_ids", "generation_input_ids"):
            unique(getattr(self, name), name)
        unique(tuple(a.slot_id for a in self.assignments), "object slots")
        claimed = [m for a in self.assignments for m in a.member_ids]
        all_members = claimed + list(self.unassigned_member_ids)
        if len(set(all_members)) != len(all_members) or set(all_members) != set(self.member_ids):
            raise ValueError("each member must occur exactly once within a conditional partition")
        count = len(self.assignments)
        if self.explanation in {Explanation.ARTIFACT, Explanation.UNRESOLVED} and count:
            raise ValueError("artifact/unresolved explanation cannot assign physical object slots")
        if self.explanation in {Explanation.ONE_OBJECT, Explanation.KNOWN_MATCH} and count != 1:
            raise ValueError("one-object explanation requires one slot")
        if self.explanation == Explanation.MULTIPLE_OBJECTS and count < 2:
            raise ValueError("multiple-object explanation requires at least two slots")
        if self.explanation == Explanation.KNOWN_MATCH and not self.assignments[0].known_record_ref:
            raise ValueError("known match requires a retained external record reference")
        return self


class EvidenceVerdict(StrEnum):
    CONSISTENT = "CONSISTENT"
    CONTRADICTS = "CONTRADICTS"
    INSUFFICIENT = "INSUFFICIENT"


class EvidenceRequest(IdentityContract):
    id: Identifier
    hypothesis_id: Identifier
    method: MethodSpec
    purpose: Literal["GENERATE", "TEST"]
    input_ids: tuple[Identifier, ...] = Field(min_length=1)
    tested_claim: Text
    dependence: Literal["UNKNOWN", "DISJOINT_INPUTS", "CONTROLLED_SHARED_INPUTS"]
    control_ref: Identifier | None = None
    provenance_ref: Identifier

    @model_validator(mode="after")
    def controlled_use(self) -> Self:
        unique(self.input_ids, "evidence inputs")
        if self.dependence == "CONTROLLED_SHARED_INPUTS" and not self.control_ref:
            raise ValueError("controlled shared inputs require an explicit control artifact")
        return self


class EvidenceFactor(IdentityContract):
    id: Identifier
    request_id: Identifier
    hypothesis_id: Identifier
    method: MethodSpec
    purpose: Literal["GENERATE", "TEST"]
    input_ids: tuple[Identifier, ...] = Field(min_length=1)
    tested_claim: Text
    verdict: EvidenceVerdict
    quantities: tuple[ScalarMeasurement, ...] = ()
    score: Score | None = None
    test_protocol_ref: Identifier
    rationale: Text
    limitations: tuple[Text, ...] = Field(min_length=1)
    provenance_ref: Identifier

    @model_validator(mode="after")
    def unique_quantities(self) -> Self:
        unique(self.input_ids, "factor inputs")
        unique(tuple(q.name for q in self.quantities), "factor quantities")
        if self.verdict != EvidenceVerdict.INSUFFICIENT:
            if not any(q.quantity.value is not None for q in self.quantities) and (
                self.score is None or self.score.value is None
            ):
                raise ValueError(
                    "decisive conditional verdict requires an explicit quantitative result"
                )
        return self


class EvidenceResult(IdentityContract):
    request: EvidenceRequest
    status: ExecutionStatus
    factors: tuple[EvidenceFactor, ...] = ()
    reason: Text
    raw_output_asset_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def coherent_result(self) -> Self:
        unique(tuple(f.id for f in self.factors), "result factors")
        unique(self.raw_output_asset_ids, "result raw outputs")
        if self.status != ExecutionStatus.SUCCESS and self.factors:
            raise ValueError("execution failure cannot provide scientific evidence")
        if self.status == ExecutionStatus.SUCCESS and not self.raw_output_asset_ids:
            raise ValueError("successful adapter result must preserve raw output")
        for factor in self.factors:
            if factor.method != self.request.method:
                raise ValueError("factor method differs from requested method")
            if (factor.hypothesis_id, factor.request_id, factor.purpose, factor.tested_claim) != (
                self.request.hypothesis_id,
                self.request.id,
                self.request.purpose,
                self.request.tested_claim,
            ):
                raise ValueError(
                    "factor must address the exact requested hypothesis, purpose and claim"
                )
            if not set(factor.input_ids) <= set(self.request.input_ids):
                raise ValueError("factor inputs must be declared by request")
        return self


class ScientificDisposition(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    FALSIFIED_UNDER_DECLARED_TEST = "FALSIFIED_UNDER_DECLARED_TEST"
    CONSISTENT_UNDER_DECLARED_TESTS = "CONSISTENT_UNDER_DECLARED_TESTS"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    UNSUPPORTED_REGIME = "UNSUPPORTED_REGIME"
    KNOWN_MATCH_CANDIDATE = "KNOWN_MATCH_CANDIDATE"
    ARTIFACT_CANDIDATE = "ARTIFACT_CANDIDATE"


class HypothesisAssessment(IdentityContract):
    hypothesis_id: Identifier
    disposition: ScientificDisposition
    tested_factor_ids: tuple[Identifier, ...]
    contradiction_factor_ids: tuple[Identifier, ...]
    reasons: tuple[Text, ...] = Field(min_length=1)


class TriageDecision(IdentityContract):
    subject_id: Identifier
    action: Literal["SURFACED", "DEFERRED", "EXCLUDED"]
    reasons: tuple[Text, ...] = Field(min_length=1)


class IdentityInput(IdentityContract):
    observations: tuple[Observation, ...]
    measurements: tuple[Measurement, ...]
    proposals: tuple[CandidateProposal, ...]
    hypotheses: tuple[ObjectHypothesis, ...]
    evidence_results: tuple[EvidenceResult, ...] = ()
    provenance: ProvenanceGraph

    @model_validator(mode="after")
    def closed_identity_graph(self) -> Self:
        obs = {o.id: o for o in self.observations}
        measurements = {m.id: m for m in self.measurements}
        proposals = {p.id: p for p in self.proposals}
        hypotheses = {h.id: h for h in self.hypotheses}
        requests = [r.request for r in self.evidence_results]
        factors = [f for r in self.evidence_results for f in r.factors]
        objects: tuple[
            Observation
            | Measurement
            | CandidateProposal
            | ObjectHypothesis
            | EvidenceRequest
            | EvidenceFactor,
            ...,
        ] = (
            *self.observations,
            *self.measurements,
            *self.proposals,
            *self.hypotheses,
            *requests,
            *factors,
        )
        unique(tuple(o.id for o in objects), "graph object IDs")
        entities = {e.id: e for e in self.provenance.entities}
        records = {r.id: r for r in self.provenance.records}
        for obj in objects:
            record = records.get(obj.provenance_ref)
            if (
                obj.id not in entities
                or record is None
                or obj.id not in record.generated_entity_ids
            ):
                raise ValueError(f"missing producing provenance for {obj.id}")

        def parents(identifier: str, provenance_ref: str, required: tuple[str, ...]) -> None:
            if not set(required) <= set(records[provenance_ref].derivations[identifier]):
                raise ValueError(f"structural inputs missing from derivation: {identifier}")

        def assets(ids: tuple[str, ...]) -> None:
            for key in ids:
                if key not in entities or not entities[key].uri or not entities[key].sha256:
                    raise ValueError("raw asset requires a pinned entity and URI")

        origin_keys = [o.source.model_dump_json() for o in self.observations]
        unique(
            tuple(origin_keys), "immutable source keys (normalize exact repeats before ingestion)"
        )
        exposure_groups: dict[tuple[str, str, str, str], str] = {}
        for observation in self.observations:
            source = observation.source
            key = (source.provider, source.dataset, source.release, source.exposure_id)
            previous = exposure_groups.setdefault(key, observation.dependence_group)
            if previous != observation.dependence_group:
                raise ValueError("one exposure must retain one dependence group")
            assets(observation.raw_asset_ids)
            parents(observation.id, observation.provenance_ref, observation.raw_asset_ids)
            for ref in (observation.observer_ref, observation.bandpass_ref):
                if ref and ref not in entities:
                    raise ValueError("unknown observer/bandpass entity")
                if ref:
                    parents(observation.id, observation.provenance_ref, (ref,))
        for measurement in self.measurements:
            if measurement.observation_id not in obs:
                raise ValueError("unknown parent observation")
            assets(measurement.raw_output_asset_ids)
            parents(
                measurement.id,
                measurement.provenance_ref,
                (measurement.observation_id, *measurement.raw_output_asset_ids),
            )
        for proposal in self.proposals:
            if not set(proposal.member_ids) <= measurements.keys():
                raise ValueError("proposal references unknown measurements")
            assets(proposal.raw_output_asset_ids)
            parents(
                proposal.id,
                proposal.provenance_ref,
                (
                    *proposal.member_ids,
                    *proposal.generation_input_ids,
                    *proposal.raw_output_asset_ids,
                ),
            )
            if proposal.score and proposal.score.calibration_ref:
                parents(proposal.id, proposal.provenance_ref, (proposal.score.calibration_ref,))
        for hypothesis in self.hypotheses:
            if not set(hypothesis.proposal_ids) <= proposals.keys():
                raise ValueError("unknown hypothesis proposal")
            support = {m for p in hypothesis.proposal_ids for m in proposals[p].member_ids}
            if not set(hypothesis.member_ids) <= support:
                raise ValueError("hypothesis members outside supporting proposal")
            parents(
                hypothesis.id,
                hypothesis.provenance_ref,
                (
                    *hypothesis.proposal_ids,
                    *hypothesis.member_ids,
                    *hypothesis.generation_input_ids,
                ),
            )
            for assignment in hypothesis.assignments:
                if assignment.known_record_ref and assignment.known_record_ref not in entities:
                    raise ValueError("unknown retained known-object record")
                if assignment.known_record_ref:
                    parents(
                        hypothesis.id, hypothesis.provenance_ref, (assignment.known_record_ref,)
                    )
        # Ancestor sets expose shared source bytes, including indirect adapter reuse.
        derivations = {
            key: value for r in self.provenance.records for key, value in r.derivations.items()
        }

        def leaves(ids: tuple[str, ...]) -> set[tuple[str, str]]:
            pending = list(ids)
            found: set[tuple[str, str]] = set()
            seen: set[str] = set()
            while pending:
                key = pending.pop()
                if key in seen:
                    continue
                seen.add(key)
                if key in derivations:
                    pending.extend(derivations[key])
                else:
                    found.add(("id", key))
                    # Renaming the same declared bytes must not manufacture disjointness.
                    entity = entities[key]
                    if entity.sha256:
                        found.add(("sha256", entity.sha256))
                    if entity.uri:
                        found.add(("uri", entity.uri))
            return found

        for result in self.evidence_results:
            request = result.request
            if request.hypothesis_id not in hypotheses:
                raise ValueError("unknown requested hypothesis")
            parents(request.id, request.provenance_ref, (request.hypothesis_id, *request.input_ids))
            assets(result.raw_output_asset_ids)
            if request.control_ref:
                if request.control_ref not in entities or not entities[request.control_ref].sha256:
                    raise ValueError("unknown/unpinned dependence control")
                parents(request.id, request.provenance_ref, (request.control_ref,))
            if request.purpose == "TEST" and request.dependence == "DISJOINT_INPUTS":
                generated = (request.hypothesis_id,)
                if leaves(request.input_ids) & leaves(generated):
                    raise ValueError("independent test assertion reuses generation source lineage")
            for factor in result.factors:
                parents(
                    factor.id,
                    factor.provenance_ref,
                    (
                        request.id,
                        factor.hypothesis_id,
                        *factor.input_ids,
                        factor.test_protocol_ref,
                        *result.raw_output_asset_ids,
                    ),
                )
                if (
                    factor.test_protocol_ref not in entities
                    or not entities[factor.test_protocol_ref].sha256
                ):
                    raise ValueError("unknown/unpinned test protocol")
                if factor.score and factor.score.calibration_ref:
                    if factor.score.calibration_ref not in entities:
                        raise ValueError("unknown score calibration")
                    parents(factor.id, factor.provenance_ref, (factor.score.calibration_ref,))
        return self


class IdentityPrediction(IdentityContract):
    """Operational evaluation projection; competing graph is retained separately."""

    case_id: Identifier
    hypothesis_id: Identifier | None = None
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    assignments: tuple[ObjectAssignment, ...] = ()
    score: Score | None = None
    state: Literal["PROPOSED", "UNRESOLVED", "UNSUPPORTED"]
    test_eligible: bool = True
    eligibility_reason: Text = "No additional evidence gate"
    inspected: bool = False
    heldout_normalized_residual: Annotated[Finite, Field(ge=0)] | None = None
    raw_output_ref: Identifier

    @model_validator(mode="after")
    def honest_prediction(self) -> Self:
        unique(self.member_ids, "prediction members")
        unique(tuple(a.slot_id for a in self.assignments), "prediction slots")
        members = [m for a in self.assignments for m in a.member_ids]
        unique(tuple(members), "within-prediction assigned members")
        if not set(members) <= set(self.member_ids):
            raise ValueError("prediction assigns outside its candidate")
        if self.state != "PROPOSED" and (self.assignments or self.score is not None):
            raise ValueError("abstention cannot carry accepted object claims or scores")
        if self.state == "PROPOSED" and (self.hypothesis_id is None or self.score is None):
            raise ValueError("proposed identity requires explicit hypothesis and score semantics")
        return self


class PredictionTable(IdentityContract):
    method: MethodSpec
    input_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    astrometry_config_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    predictions: tuple[IdentityPrediction, ...]
    calibration_artifacts: tuple[Entity, ...] = ()
    factor_protocol_refs: tuple[Identifier, ...] = ()
    control_artifact_refs: tuple[Identifier, ...] = ()
    evidence_protocol_artifacts: tuple[Entity, ...] = ()
    runtime_seconds: Annotated[Finite, Field(ge=0)]
    peak_memory_bytes: Annotated[int, Field(ge=0)] | None
    limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        unique(tuple(p.case_id for p in self.predictions), "method case IDs")
        unique(self.factor_protocol_refs, "factor protocol references")
        unique(self.control_artifact_refs, "control artifact references")
        unique(tuple(e.id for e in self.evidence_protocol_artifacts), "evidence artifact IDs")
        return self
