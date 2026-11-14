"""Closed offline input manifest. Providers must meet the same reference checks."""

from typing import Self

from pydantic import model_validator

from sieve.contracts.models import (
    CandidateProposal,
    Contract,
    Detection,
    EvidenceFactor,
    EvidenceRequest,
    EvidenceResult,
    ObjectHypothesis,
    Observation,
)
from sieve.provenance.models import ProvenanceGraph


class InferenceBundle(Contract):
    observations: tuple[Observation, ...]
    detections: tuple[Detection, ...]
    proposals: tuple[CandidateProposal, ...]
    hypotheses: tuple[ObjectHypothesis, ...]
    evidence_results: tuple[EvidenceResult, ...] = ()
    provenance: ProvenanceGraph

    @model_validator(mode="after")
    def reference_integrity(self) -> Self:
        observations: dict[str, Observation] = {}
        exposure_groups: dict[tuple[str, str, str, str], str] = {}
        for observation in self.observations:
            if observation.id in observations and observations[observation.id] != observation:
                raise ValueError("conflicting duplicate observation ID")
            observations[observation.id] = observation
            if observation.exposure_id:
                key = (
                    observation.provider,
                    observation.dataset,
                    observation.release,
                    observation.exposure_id,
                )
                if key in exposure_groups and exposure_groups[key] != observation.dependence_group:
                    raise ValueError("samples from one exposure must share a dependence group")
                exposure_groups[key] = observation.dependence_group
        detections = {d.id: d for d in self.detections}
        proposals = {p.id: p for p in self.proposals}
        hypotheses = {h.id: h for h in self.hypotheses}
        requests = [r.request for r in self.evidence_results]
        factors = [f for r in self.evidence_results for f in r.factors]
        objects: list[
            Observation
            | Detection
            | CandidateProposal
            | ObjectHypothesis
            | EvidenceRequest
            | EvidenceFactor
        ] = []
        objects.extend(observations.values())
        objects.extend(self.detections)
        objects.extend(self.proposals)
        objects.extend(self.hypotheses)
        objects.extend(requests)
        objects.extend(factors)
        ids = [obj.id for obj in objects]
        if len(ids) != len(set(ids)):
            raise ValueError("production object IDs must be globally unique")
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
        assets_objects: list[Observation | Detection] = list(observations.values())
        assets_objects.extend(self.detections)
        for asset_object in assets_objects:
            if any(a not in entities or not entities[a].uri for a in asset_object.asset_ids):
                raise ValueError("source asset requires a known entity and URI")
        for observation in observations.values():
            parents = records[observation.provenance_ref].derivations[observation.id]
            if not set(observation.asset_ids) <= set(parents):
                raise ValueError("observation assets missing from derivation")
        for detection in self.detections:
            if detection.observation_id not in observations:
                raise ValueError("unknown parent observation")
            parents = records[detection.provenance_ref].derivations[detection.id]
            if detection.observation_id not in parents:
                raise ValueError("parent observation missing from detection derivation")
        for proposal in self.proposals:
            if not set(proposal.member_ids) <= detections.keys():
                raise ValueError("proposal references unknown detection")
            parents = records[proposal.provenance_ref].derivations[proposal.id]
            if not set(proposal.member_ids) <= set(parents):
                raise ValueError("proposal members missing from derivation")
        for hypothesis in self.hypotheses:
            if not set(hypothesis.proposal_ids) <= proposals.keys():
                raise ValueError("hypothesis references unknown proposal")
            members = {d for p in hypothesis.proposal_ids for d in proposals[p].member_ids}
            if not set(hypothesis.member_ids) <= members:
                raise ValueError("hypothesis members must belong to its supporting proposals")
            parents = records[hypothesis.provenance_ref].derivations[hypothesis.id]
            if not set(hypothesis.proposal_ids + hypothesis.member_ids) <= set(parents):
                raise ValueError("hypothesis support missing from derivation")
        for result in self.evidence_results:
            if result.request.hypothesis_id not in hypotheses:
                raise ValueError("request references unknown hypothesis")
            request_parents = records[result.request.provenance_ref].derivations[result.request.id]
            if result.request.hypothesis_id not in request_parents:
                raise ValueError("requested hypothesis missing from request derivation")
            evidence_objects: list[EvidenceRequest | EvidenceFactor] = [result.request]
            evidence_objects.extend(result.factors)
            for evidence_object in evidence_objects:
                if not set(evidence_object.input_ids) <= entities.keys():
                    raise ValueError("unknown evidence input entity")
                if not set(evidence_object.input_ids) <= set(
                    records[evidence_object.provenance_ref].derivations[evidence_object.id]
                ):
                    raise ValueError("evidence inputs missing from producing activity")
            for factor in result.factors:
                parents = records[factor.provenance_ref].derivations[factor.id]
                if not {result.request.id, factor.hypothesis_id} <= set(parents):
                    raise ValueError("evidence request/hypothesis missing from factor derivation")
        scores = [p.score for p in self.proposals] + [d.score for d in self.detections]
        calibrations = [s.calibration_ref for s in scores if s and s.calibration_ref]
        calibrations += [f.calibration_ref for f in factors if f.calibration_ref]
        if any(ref not in entities for ref in calibrations):
            raise ValueError("unknown calibration entity")
        scored_objects: list[CandidateProposal | Detection] = list(self.proposals)
        scored_objects.extend(self.detections)
        for scored in scored_objects:
            if scored.score and scored.score.calibration_ref:
                parents = records[scored.provenance_ref].derivations[scored.id]
                if scored.score.calibration_ref not in parents:
                    raise ValueError("score calibration missing from derivation")
        for factor in factors:
            if factor.calibration_ref:
                parents = records[factor.provenance_ref].derivations[factor.id]
                if factor.calibration_ref not in parents:
                    raise ValueError("evidence calibration missing from derivation")
        return self
