"""Offline foundation funnel. Science adapters supply proposals and evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence

from sieve.contracts.bundle import InferenceBundle
from sieve.contracts.models import ScientificDisposition, TriageAction, TriageDecision
from sieve.interfaces import CandidateRanker, TriagePolicy
from sieve.provenance.models import content_id
from sieve.reporting import Case, CaseReport, Scope
from sieve.triage.budget import CandidateBudget, select_diverse
from sieve.triage.dedup import DedupPolicy, collapse_proposals, unique_observations
from sieve.triage.policy import ConservativeTriage, ReferenceRanker


def build_cases(
    bundle: InferenceBundle, budget: CandidateBudget, dedup: DedupPolicy
) -> tuple[tuple[Case, ...], tuple[TriageDecision, ...], dict[str, int]]:
    # Revalidate at the trust boundary, including models constructed unsafely by adapters.
    bundle = InferenceBundle.model_validate(bundle.model_dump(mode="json"))
    observations = {o.id: o for o in unique_observations(bundle.observations)}
    if any(o.time.scale == "ut1" for o in observations.values()):
        raise ValueError("offline UT1 conversion requires a pinned Earth orientation adapter")
    detections = {d.id: d for d in bundle.detections}
    eligible = {
        d.id
        for d in bundle.detections
        if d.quality.usable
        and observations[d.observation_id].quality.usable
        and not set(d.quality.flags) & {"REFERENCE_STATIC", "REFERENCE_JUNK"}
    }
    decisions = []
    for detection in sorted(bundle.detections, key=lambda d: d.id):
        if detection.id not in eligible:
            decisions.append(
                TriageDecision(
                    subject_id=detection.id,
                    action=TriageAction.SUPPRESSED,
                    reasons=("quality_or_explicit_reference_screen",),
                )
            )
    proposals = []
    for proposal in sorted(bundle.proposals, key=lambda p: p.id):
        if not set(proposal.member_ids) <= eligible:
            decisions.append(
                TriageDecision(
                    subject_id=proposal.id,
                    action=TriageAction.SUPPRESSED,
                    reasons=("proposal_contains_screened_detection",),
                )
            )
        else:
            proposals.append(proposal)
    groups = collapse_proposals(proposals, detections, observations, dedup)
    cases = []
    for group in groups:
        anchor = detections[group[0].member_ids[0]]
        coord = anchor.position.as_coord().icrs
        obs = observations[anchor.observation_id]
        tai = obs.time.as_time().tai
        days = (float(tai.jd1) - 2451545) + float(tai.jd2)
        scope = Scope(
            region=f"r{math.floor(float(coord.ra.deg) / budget.cell_degrees)}"
            f"d{math.floor((float(coord.dec.deg) + 90) / budget.cell_degrees)}",
            time_window=f"w{math.floor(days / budget.window_days)}",
            regime=group[0].task_type,
        )
        proposal_ids = {p.id for p in group}
        hypotheses = tuple(
            sorted(
                (h for h in bundle.hypotheses if proposal_ids & set(h.proposal_ids)),
                key=lambda h: h.id,
            )
        )
        hypothesis_ids = {h.id for h in hypotheses}
        results = tuple(
            sorted(
                (r for r in bundle.evidence_results if r.request.hypothesis_id in hypothesis_ids),
                key=lambda r: r.request.id,
            )
        )
        case = Case(
            id=content_id("case", sorted(proposal_ids)),
            scope=scope,
            proposals=group,
            hypotheses=hypotheses,
            evidence_results=results,
        )
        cases.append(case)
        for alias in group[1:]:
            decisions.append(
                TriageDecision(
                    subject_id=alias.id,
                    action=TriageAction.MERGED_AS_DUPLICATE,
                    related_id=case.id,
                    reasons=("review_envelope_only_original_evidence_retained",),
                )
            )
    counts = {
        "observations_entered": len(bundle.observations),
        "observations_unique": len(observations),
        "observations_deduplicated": len(bundle.observations) - len(observations),
        "detections_raw": len(bundle.detections),
        "detections_eligible": len(eligible),
        "detections_suppressed": len(bundle.detections) - len(eligible),
        "proposals_raw": len(bundle.proposals),
        "proposals_suppressed": len(bundle.proposals) - len(proposals),
        "proposals_merged": len(proposals) - len(groups),
        "cases_normalized": len(cases),
        "hypotheses_surviving": len({h.id for c in cases for h in c.hypotheses}),
    }
    return tuple(cases), tuple(decisions), counts


def triage(
    bundle: InferenceBundle,
    budget: CandidateBudget,
    dedup: DedupPolicy,
    *,
    ranker: CandidateRanker | None = None,
    policy: TriagePolicy | None = None,
) -> tuple[tuple[CaseReport, ...], tuple[TriageDecision, ...], dict[str, int]]:
    cases, audit, counts = build_cases(bundle, budget, dedup)
    selected, deferred = select_diverse(cases, budget, ranker or ReferenceRanker())
    reports = []
    policy = policy or ConservativeTriage()
    for case in selected:
        assessments = policy.assess(case)
        if {a.hypothesis_id for a in assessments} != {h.id for h in case.hypotheses}:
            raise ValueError("triage policy must assess every hypothesis exactly once")
        if len(assessments) != len(case.hypotheses):
            raise ValueError("duplicate hypothesis assessment")
        action = TriageAction.SURFACED
        if not assessments or any(
            a.disposition
            in {
                ScientificDisposition.INSUFFICIENT_EVIDENCE,
                ScientificDisposition.REVIEW_REQUIRED,
                ScientificDisposition.AMBIGUOUS_IDENTITY,
                ScientificDisposition.UNSUPPORTED_REGIME,
            }
            for a in assessments
        ):
            action = TriageAction.REVIEW_REQUIRED
        decision = TriageDecision(
            subject_id=case.id,
            action=action,
            reasons=("selected_within_diversity_and_review_budget",),
        )
        reports.append(CaseReport(case=case, assessments=assessments, decision=decision))
    deferred_decisions = tuple(
        TriageDecision(
            subject_id=c.id,
            action=TriageAction.DEFERRED,
            reasons=("review_budget_exceeded_preserved_in_input",),
        )
        for c in deferred
    )
    counts.update(
        cases_surfaced=len(reports),
        cases_deferred=len(deferred),
        hypotheses_surfaced=sum(len(r.case.hypotheses) for r in reports),
    )
    return tuple(reports), audit + deferred_decisions, counts


def report_payload(
    cases: Sequence[CaseReport], decisions: Sequence[TriageDecision], counts: dict[str, int]
) -> dict[str, object]:
    return {
        "cases": [c.model_dump(mode="json") for c in cases],
        "audit_decisions": [d.model_dump(mode="json") for d in decisions],
        "counts": counts,
    }
