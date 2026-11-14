from __future__ import annotations

from collections.abc import Sequence

import astropy.units as u
from pydantic import Field

from sieve.contracts.models import CandidateProposal, Contract, Detection, Observation


class DedupPolicy(Contract):
    approximate: bool = False
    jaccard_min: float = Field(default=0.8, gt=0, le=1)
    separation_arcsec: float = Field(default=1, ge=0)
    time_seconds: float = Field(default=1, ge=0)


def unique_observations(observations: Sequence[Observation]) -> tuple[Observation, ...]:
    by_id: dict[str, Observation] = {}
    for observation in observations:
        if observation.id in by_id and by_id[observation.id] != observation:
            raise ValueError("conflicting duplicate observation ID")
        by_id[observation.id] = observation
    return tuple(by_id[key] for key in sorted(by_id))


def collapse_proposals(
    proposals: Sequence[CandidateProposal],
    detections: dict[str, Detection],
    observations: dict[str, Observation],
    policy: DedupPolicy,
) -> tuple[tuple[CandidateProposal, ...], ...]:
    """Anchor grouping, not connected components; retain each original proposal."""
    groups: list[list[CandidateProposal]] = []
    exact: dict[tuple[str, str, tuple[str, ...]], list[CandidateProposal]] = {}
    families: dict[tuple[str, str, str, str, str], list[list[CandidateProposal]]] = {}
    for proposal in sorted(proposals, key=lambda p: p.id):
        exact_key = (proposal.task_type, proposal.explanation_key, proposal.member_ids)
        if exact_key in exact:
            exact[exact_key].append(proposal)
            continue
        family_key = (
            proposal.task_type,
            proposal.explanation_key,
            proposal.method.name,
            proposal.method.version,
            proposal.equivalence_family or "",
        )
        target = None
        if policy.approximate and proposal.equivalence_family:
            for group in families.get(family_key, []):
                anchor = group[0]
                a, b = set(anchor.member_ids), set(proposal.member_ids)
                overlap = len(a & b) / len(a | b)
                da = detections[anchor.member_ids[0]]
                db = detections[proposal.member_ids[0]]
                dt = abs(
                    float(
                        (
                            observations[da.observation_id].time.as_time().tai
                            - observations[db.observation_id].time.as_time().tai
                        ).sec
                    )
                )
                separation = float(
                    da.position.as_coord().separation(db.position.as_coord()).to_value(u.arcsec)
                )
                if overlap >= policy.jaccard_min or (
                    separation <= policy.separation_arcsec and dt <= policy.time_seconds
                ):
                    target = group
                    break
        if target is None:
            target = []
            groups.append(target)
            families.setdefault(family_key, []).append(target)
        target.append(proposal)
        exact[exact_key] = target
    return tuple(tuple(group) for group in groups)
