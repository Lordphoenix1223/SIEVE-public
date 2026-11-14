from __future__ import annotations

import hashlib
import json
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from sieve.contracts.models import Contract, Identifier, Text

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def content_id(prefix: str, value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return prefix + ":" + hashlib.sha256(encoded).hexdigest()


class Entity(Contract):
    id: Identifier
    kind: Text
    uri: Text | None = None
    provider: Text | None = None
    provider_identifier: Text | None = None
    dataset_release: Text | None = None
    sha256: Digest | None = None
    hash_missing_reason: Text | None = None
    citations: tuple[Text, ...] = ()
    acknowledgements: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def hash_status(self) -> Self:
        if (self.sha256 is None) != (self.hash_missing_reason is not None):
            raise ValueError("unhashed entity requires explicit hash_missing_reason")
        return self


class Agent(Contract):
    id: Identifier
    name: Text
    role: Text
    uri: Text | None = None


class ProvenanceRecord(Contract):
    id: Identifier
    activity: Text
    method_version: Text
    agent_ids: tuple[Identifier, ...] = Field(min_length=1)
    used_entity_ids: tuple[Identifier, ...] = Field(min_length=1)
    generated_entity_ids: tuple[Identifier, ...] = Field(min_length=1)
    # derived entity -> input entities; every used/generated relation remains explicit.
    derivations: dict[Identifier, tuple[Identifier, ...]]
    code_commit: Text
    software_version: Text
    configuration: dict[str, JsonValue]
    request_parameters: dict[str, JsonValue]
    random_seeds: dict[str, int]
    environment_id: Text
    started_at: AwareDatetime
    ended_at: AwareDatetime
    external_service: Text | None = None

    @model_validator(mode="after")
    def activity_consistency(self) -> Self:
        for ids in (self.agent_ids, self.used_entity_ids, self.generated_entity_ids):
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate activity reference IDs")
        if self.ended_at < self.started_at:
            raise ValueError("activity end precedes start")
        if set(self.used_entity_ids) & set(self.generated_entity_ids):
            raise ValueError("an immutable entity cannot be both input and output")
        if set(self.derivations) != set(self.generated_entity_ids):
            raise ValueError("every output must have an explicit derivation")
        for parents in self.derivations.values():
            if not parents or not set(parents) <= set(self.used_entity_ids):
                raise ValueError("derivation parents must be activity inputs")
        return self


class ProvenanceGraph(Contract):
    entities: tuple[Entity, ...]
    agents: tuple[Agent, ...]
    records: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def closed_graph(self) -> Self:
        entities = {e.id for e in self.entities}
        agents = {a.id for a in self.agents}
        records = {r.id for r in self.records}
        all_ids = [e.id for e in self.entities] + [a.id for a in self.agents]
        all_ids += [r.id for r in self.records]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("lineage IDs must be globally unique")
        generated: set[str] = set()
        edges: dict[str, tuple[str, ...]] = {}
        for record in self.records:
            if not set(record.agent_ids) <= agents:
                raise ValueError("unknown provenance agent")
            if not set(record.used_entity_ids + record.generated_entity_ids) <= entities:
                raise ValueError("unknown lineage entity")
            if generated & set(record.generated_entity_ids):
                raise ValueError("entity has multiple producing activities")
            generated.update(record.generated_entity_ids)
            edges.update(record.derivations)
        # Iterative topological validation avoids recursion limits on long derivations.
        remaining = {key: set(parents) & generated for key, parents in edges.items()}
        resolved = {key for key, parents in remaining.items() if not parents}
        while resolved:
            for key in resolved:
                del remaining[key]
            for parents in remaining.values():
                parents.difference_update(resolved)
            resolved = {key for key, parents in remaining.items() if not parents}
        if remaining:
            raise ValueError("cyclic provenance derivation")
        if records & entities:
            raise ValueError("entity and activity namespaces collide")
        return self


class RunReceipt(Contract):
    id: Identifier
    provenance_ref: Identifier
    input_sha256: Digest
    output_sha256: Digest
    code_commit: Text
    code_tree_sha256: Digest
    dirty: bool
    software_version: Text
    environment_id: Text
    configuration: dict[str, JsonValue]
    random_seeds: dict[str, int]
    started_at: AwareDatetime
    ended_at: AwareDatetime
