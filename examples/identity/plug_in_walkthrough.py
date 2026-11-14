"""Run the two collaborator boundaries using fixture playback, not instrument science.

Run from a bootstrapped checkout:
    uv run --no-sync python examples/identity/plug_in_walkthrough.py

Replace these replay classes with qualified provider/evidence implementations. The
candidate hypotheses are supplied upstream; the evidence statistic is already injected.
No evaluator memberships are loaded and no SPHEREx factor is measured here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from sieve.identity.attachment import attach_evidence
from sieve.identity.graph import IdentityBudget, identity_funnel
from sieve.identity.models import (
    EvidenceRequest,
    EvidenceResult,
    IdentityInput,
    Measurement,
    ObjectHypothesis,
    Observation,
)
from sieve.identity.workflow import read_identity
from sieve.interfaces import (
    IdentityObservationProvider,
    ObservationBatch,
    PhysicalEvidenceProvider,
)
from sieve.provenance.models import ProvenanceGraph

FIXTURE = Path(__file__).resolve().parent / "merge.json"


class FixtureObservationProvider:
    """Ted's interface: load measured records plus their closed producer lineage."""

    def __init__(self, fixture: IdentityInput):
        self.fixture = fixture

    def load(self, query: Mapping[str, JsonValue]) -> ObservationBatch:
        if dict(query) != {"fixture": "merge"}:
            raise ValueError("This playback supports only the published merge fixture")
        # Construction now validates original keys, parents, raw pins and producers.
        return ObservationBatch(
            observations=self.fixture.observations,
            measurements=self.fixture.measurements,
            provenance=self.fixture.provenance,
        )


class FixtureEvidenceProvider:
    """Jessica's interface: replay one exact declared test and preserve raw output."""

    def __init__(self, fixture: IdentityInput):
        self.fixture = fixture

    def evaluate(
        self,
        hypothesis: ObjectHypothesis,
        request: EvidenceRequest,
        observations: Sequence[Observation],
        measurements: Sequence[Measurement],
    ) -> EvidenceResult:
        result = self.fixture.evidence_results[0]
        expected = next((h for h in self.fixture.hypotheses if h.id == request.hypothesis_id), None)
        if (
            hypothesis != expected
            or request != result.request
            or tuple(observations) != self.fixture.observations
            or tuple(measurements) != self.fixture.measurements
        ):
            return EvidenceResult(
                request=request,
                status="UNSUPPORTED",
                reason="Fixture playback cannot evaluate other inputs; "
                "no scientific factor returned",
            )
        return result

    def provenance(self) -> ProvenanceGraph:
        return self.fixture.provenance


def demonstrate() -> dict[str, Any]:
    fixture, _ = read_identity(FIXTURE)
    data_provider: IdentityObservationProvider = FixtureObservationProvider(fixture)
    evidence_provider: PhysicalEvidenceProvider = FixtureEvidenceProvider(fixture)
    batch = data_provider.load({"fixture": "merge"})

    # Proposals/competing hypotheses were produced upstream. Start before evidence attachment.
    base = IdentityInput(
        observations=batch.observations,
        measurements=batch.measurements,
        proposals=fixture.proposals,
        hypotheses=fixture.hypotheses,
        provenance=batch.provenance,
    )
    request = fixture.evidence_results[0].request
    hypothesis = next(h for h in base.hypotheses if h.id == request.hypothesis_id)
    result = evidence_provider.evaluate(hypothesis, request, batch.observations, batch.measurements)
    attached = attach_evidence(base, (result,), evidence_provider.provenance())
    graph = identity_funnel(attached, IdentityBudget(review_cases=1))
    return dict(
        exercise="FIXTURE_PLAYBACK_ONLY",
        scientific_claim_status="CONTRACT_ONLY",
        provider_measurements=len(batch.measurements),
        hypotheses_before=[h.id for h in base.hypotheses],
        hypotheses_after=[h.id for h in attached.hypotheses],
        source_ids=[o.source.source_id for o in attached.observations],
        evidence_origin="INJECTED_REFERENCE_STATISTIC",
        assessments=[a.model_dump(mode="json") for a in graph.assessments],
        case_disposition=graph.cases[0].disposition,
    )


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2, allow_nan=False))
