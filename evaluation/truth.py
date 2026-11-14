from enum import StrEnum
from typing import Self

from pydantic import model_validator

from sieve.contracts.models import Contract, Identifier, Text


class TruthType(StrEnum):
    SIMULATED_TRUTH = "SIMULATED_TRUTH"
    BENCHMARK_TRUTH = "BENCHMARK_TRUTH"
    FROZEN_HISTORICAL_TRUTH = "FROZEN_HISTORICAL_TRUTH"
    AUTHORITATIVE_EXTERNAL_RECORD = "AUTHORITATIVE_EXTERNAL_RECORD"
    HUMAN_ANNOTATION = "HUMAN_ANNOTATION"
    UNKNOWN = "UNKNOWN"


class TruthRecord(Contract):
    id: Identifier
    source_type: TruthType
    source_ref: Text
    task_scope: Identifier
    member_ids: tuple[Identifier, ...]
    object_ids: tuple[Identifier, ...]  # empty is a declared artifact in a complete simulation
    frozen: bool
    complete_for_scope: bool
    audit_ref: Text | None = None
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def never_promote_annotations(self) -> Self:
        if self.source_type in {TruthType.HUMAN_ANNOTATION, TruthType.UNKNOWN}:
            if self.complete_for_scope:
                raise ValueError("annotation/unknown cannot declare complete benchmark truth")
        return self


def benchmark_eligible(record: TruthRecord, task_scope: str) -> bool:
    if not record.frozen or not record.complete_for_scope or record.task_scope != task_scope:
        return False
    if record.source_type == TruthType.SIMULATED_TRUTH:
        return True
    if record.source_type in {TruthType.BENCHMARK_TRUTH, TruthType.FROZEN_HISTORICAL_TRUTH}:
        return record.audit_ref is not None
    # External records may score narrowly audited identity tasks, never open-world absence.
    return (
        record.source_type == TruthType.AUTHORITATIVE_EXTERNAL_RECORD
        and task_scope == "catalog_identity"
        and record.audit_ref is not None
    )
