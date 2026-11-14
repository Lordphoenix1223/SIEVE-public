from pydantic import Field, JsonValue

from sieve.contracts.models import (
    CandidateProposal,
    Contract,
    EvidenceResult,
    Identifier,
    ObjectHypothesis,
    ScientificDisposition,
    Text,
    TriageDecision,
)
from sieve.provenance.models import ProvenanceGraph, RunReceipt


class Scope(Contract):
    region: Identifier
    time_window: Identifier
    regime: Identifier


class Assessment(Contract):
    hypothesis_id: Identifier
    disposition: ScientificDisposition
    reasons: tuple[Text, ...] = Field(min_length=1)


class Case(Contract):
    id: Identifier
    scope: Scope
    proposals: tuple[CandidateProposal, ...] = Field(min_length=1)
    hypotheses: tuple[ObjectHypothesis, ...]
    evidence_results: tuple[EvidenceResult, ...]


class CaseReport(Contract):
    case: Case
    assessments: tuple[Assessment, ...]
    decision: TriageDecision


class FunnelReport(Contract):
    counts: dict[str, int]
    cases: tuple[CaseReport, ...]
    # Audit volume may grow with input; this is not the surfaced review queue.
    audit_decisions: tuple[TriageDecision, ...]
    statistical_filter: dict[str, JsonValue]
    receipt: RunReceipt
    provenance: ProvenanceGraph
    limitations: tuple[Text, ...]
