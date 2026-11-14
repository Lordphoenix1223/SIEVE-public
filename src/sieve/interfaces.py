"""Contributor boundaries. Adapters return contracts, never evaluator truth."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, Self

from pydantic import JsonValue, model_validator

from sieve.contracts.models import (
    CandidateProposal,
    Detection,
    EvidenceRequest,
    EvidenceResult,
    Observation,
    Score,
)

# Explicit aliases keep identity v1 distinct from the v0 compatibility protocols.
from sieve.identity.models import (
    CandidateProposal as IdentityProposal,
)
from sieve.identity.models import (
    EvidenceRequest as IdentityEvidenceRequest,
)
from sieve.identity.models import (
    EvidenceResult as IdentityEvidenceResult,
)
from sieve.identity.models import (
    IdentityContract,
    IdentityInput,
    Measurement,
)
from sieve.identity.models import (
    ObjectHypothesis as IdentityHypothesis,
)
from sieve.identity.models import (
    Observation as IdentityObservation,
)
from sieve.provenance.models import ProvenanceGraph
from sieve.reporting import Assessment, Case


class ObservationProvider(Protocol):
    def observations(self, query: Mapping[str, JsonValue]) -> Iterable[Observation]: ...
    def provenance(self) -> ProvenanceGraph: ...


class Detector(Protocol):
    def detect(self, observations: Sequence[Observation]) -> Iterable[Detection]: ...
    def provenance(self) -> ProvenanceGraph: ...


class CandidateGenerator(Protocol):
    def propose(self, detections: Sequence[Detection]) -> Iterable[CandidateProposal]: ...
    def provenance(self) -> ProvenanceGraph: ...


class EvidenceProvider(Protocol):
    def evaluate(self, request: EvidenceRequest) -> EvidenceResult: ...
    def provenance(self) -> ProvenanceGraph: ...


class AnomalyScorer(Protocol):
    def score(self, proposal: CandidateProposal) -> Score: ...
    def provenance(self) -> ProvenanceGraph: ...


class CandidateRanker(Protocol):
    # Only compare values within the ranker's declared cohort. Core checks finiteness.
    def cohort(self, case: Case) -> str: ...
    def priority(self, case: Case) -> float: ...


class TriagePolicy(Protocol):
    def assess(self, case: Case) -> tuple[Assessment, ...]: ...


class ObservationBatch(IdentityContract):
    """Validate producer records before any proposal/evidence adapter consumes them."""

    observations: tuple[IdentityObservation, ...]
    measurements: tuple[Measurement, ...]
    provenance: ProvenanceGraph

    @model_validator(mode="after")
    def closed_observation_batch(self) -> Self:
        # Reuse the graph's intake rules rather than maintaining a second validator.
        IdentityInput(
            observations=self.observations,
            measurements=self.measurements,
            proposals=(),
            hypotheses=(),
            provenance=self.provenance,
        )
        return self


class IdentityObservationProvider(Protocol):
    def load(self, query: Mapping[str, JsonValue]) -> ObservationBatch: ...


class IdentityProposalProvider(Protocol):
    def propose(self, batch: ObservationBatch) -> Iterable[IdentityProposal]: ...
    def provenance(self) -> ProvenanceGraph: ...


class PhysicalEvidenceProvider(Protocol):
    def evaluate(
        self,
        hypothesis: IdentityHypothesis,
        request: IdentityEvidenceRequest,
        observations: Sequence[IdentityObservation],
        measurements: Sequence[Measurement],
    ) -> IdentityEvidenceResult: ...
    def provenance(self) -> ProvenanceGraph: ...
