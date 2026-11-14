from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field

from sieve.contracts.models import Contract
from sieve.reporting import Case

if TYPE_CHECKING:
    from sieve.interfaces import CandidateRanker


class CandidateBudget(Contract):
    total: int = Field(default=5, ge=0)
    total_hypotheses: int = Field(default=20, ge=0)
    per_case_hypotheses: int = Field(default=10, ge=0)
    per_region: int | None = Field(default=None, ge=0)
    per_time_window: int | None = Field(default=None, ge=0)
    per_regime: int | None = Field(default=None, ge=0)
    cell_degrees: float = Field(default=10, gt=0, le=180)
    window_days: float = Field(default=1, gt=0)


def select_diverse(
    cases: Sequence[Case], budget: CandidateBudget, ranker: CandidateRanker
) -> tuple[tuple[Case, ...], tuple[Case, ...]]:
    """Deterministic round-robin over region/time/regime/ranking cohorts.

    Different model scales are never compared. A lexicographic bucket order is a
    disclosed policy, not an optimized allocation of scientific attention.
    """
    buckets: dict[tuple[str, str, str, str], list[tuple[float, Case]]] = defaultdict(list)
    for case in cases:
        priority = ranker.priority(case)
        if not math.isfinite(priority):
            raise ValueError("ranker returned nonfinite priority")
        scope = case.scope
        buckets[(scope.region, scope.time_window, scope.regime, ranker.cohort(case))].append(
            (priority, case)
        )
    queues = [
        deque(c for _, c in sorted(buckets[key], key=lambda x: (-x[0], x[1].id)))
        for key in sorted(buckets)
    ]
    selected: list[Case] = []
    hypothesis_count = 0
    regions: Counter[str] = Counter()
    times: Counter[str] = Counter()
    regimes: Counter[str] = Counter()
    while queues and len(selected) < budget.total:
        next_queues = []
        for queue in queues:
            chosen = None
            while queue:
                case = queue.popleft()
                s = case.scope
                if (
                    len(case.hypotheses) <= budget.per_case_hypotheses
                    and hypothesis_count + len(case.hypotheses) <= budget.total_hypotheses
                    and (budget.per_region is None or regions[s.region] < budget.per_region)
                    and (
                        budget.per_time_window is None
                        or times[s.time_window] < budget.per_time_window
                    )
                    and (budget.per_regime is None or regimes[s.regime] < budget.per_regime)
                ):
                    chosen = case
                    break
            if chosen is not None and len(selected) < budget.total:
                selected.append(chosen)
                hypothesis_count += len(chosen.hypotheses)
                regions[chosen.scope.region] += 1
                times[chosen.scope.time_window] += 1
                regimes[chosen.scope.regime] += 1
            if queue:
                next_queues.append(queue)
        queues = next_queues
    kept = {c.id for c in selected}
    return tuple(selected), tuple(
        sorted((c for c in cases if c.id not in kept), key=lambda c: c.id)
    )
