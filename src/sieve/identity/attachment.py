"""Attach externally measured evidence without losing upstream bytes or hypotheses."""

from collections.abc import Sequence

from sieve.identity.models import EvidenceResult, IdentityInput
from sieve.provenance.models import Agent, Entity, ProvenanceGraph, ProvenanceRecord


def merge_provenance(left: ProvenanceGraph, right: ProvenanceGraph) -> ProvenanceGraph:
    def merge[T: Entity | Agent | ProvenanceRecord](
        a: tuple[T, ...], b: tuple[T, ...]
    ) -> tuple[T, ...]:
        by_id: dict[str, T] = {}
        for item in (*a, *b):
            key = item.id
            if key in by_id and by_id[key] != item:
                raise ValueError(f"conflicting immutable provenance ID: {key}")
            by_id[key] = item
        return tuple(by_id[k] for k in sorted(by_id))

    entities: tuple[Entity, ...] = merge(left.entities, right.entities)
    agents: tuple[Agent, ...] = merge(left.agents, right.agents)
    records: tuple[ProvenanceRecord, ...] = merge(left.records, right.records)
    return ProvenanceGraph(entities=entities, agents=agents, records=records)


def attach_evidence(
    bundle: IdentityInput, results: Sequence[EvidenceResult], producer_provenance: ProvenanceGraph
) -> IdentityInput:
    graph = merge_provenance(bundle.provenance, producer_provenance)
    data = bundle.model_dump(mode="json")
    data["evidence_results"] += [result.model_dump(mode="json") for result in results]
    data["provenance"] = graph.model_dump(mode="json")
    return IdentityInput.model_validate(data)
