"""Deterministic hypothesis graph: review grouping never allocates physical identity."""

from __future__ import annotations

from collections import defaultdict

from pydantic import Field

from sieve.contracts.models import ExecutionStatus, Identifier, Text
from sieve.identity.models import (
    EvidenceVerdict,
    Explanation,
    HypothesisAssessment,
    IdentityContract,
    IdentityInput,
    ScientificDisposition,
    TriageDecision,
)
from sieve.provenance.models import content_id


class IdentityBudget(IdentityContract):
    review_cases: int = Field(default=20, ge=1, le=1000)
    review_hypotheses: int = Field(default=200, ge=1, le=10000)
    per_case_hypotheses: int = Field(default=50, ge=1, le=1000)
    max_graph_nodes: int = Field(default=50000, ge=1, le=100000)


class IdentityCase(IdentityContract):
    id: Identifier
    proposal_ids: tuple[Identifier, ...]
    hypothesis_ids: tuple[Identifier, ...]
    member_ids: tuple[Identifier, ...]
    disposition: ScientificDisposition
    reasons: tuple[Text, ...]


class IdentityFunnel(IdentityContract):
    cases: tuple[IdentityCase, ...]
    assessments: tuple[HypothesisAssessment, ...]
    decisions: tuple[TriageDecision, ...]
    counts: dict[str, int]


def assess(bundle: IdentityInput) -> tuple[HypothesisAssessment, ...]:
    measurements = {m.id: m for m in bundle.measurements}
    observations = {o.id: o for o in bundle.observations}
    results = defaultdict(list)
    for result in bundle.evidence_results:
        results[result.request.hypothesis_id].append(result)
    assessments = []
    for hypothesis in sorted(bundle.hypotheses, key=lambda h: h.id):
        tests = [r for r in results[hypothesis.id] if r.request.purpose == "TEST"]
        admitted = [
            f
            for r in tests
            if r.status == ExecutionStatus.SUCCESS and r.request.dependence != "UNKNOWN"
            for f in r.factors
        ]
        against = [f.id for f in admitted if f.verdict == EvidenceVerdict.CONTRADICTS]
        claims = defaultdict(set)
        for factor in admitted:
            if factor.verdict != EvidenceVerdict.INSUFFICIENT:
                claims[factor.tested_claim].add(factor.verdict)
        invalid = any(
            not measurements[m].quality.usable
            or not observations[measurements[m].observation_id].quality.usable
            for m in hypothesis.member_ids
        )
        disposition = ScientificDisposition.INSUFFICIENT_EVIDENCE
        reason = (
            "No admitted decisive test; rank, generation evidence and no-match are not identity"
        )
        if invalid:
            disposition = ScientificDisposition.UNSUPPORTED_REGIME
            reason = (
                "Hypothesis includes unusable measurements; invalid input is not scientific truth"
            )
        elif any(r.status != ExecutionStatus.SUCCESS for r in tests):
            disposition = ScientificDisposition.UNRESOLVED
            reason = "Requested test incomplete; execution failure does not falsify identity"
        elif any(len(v) > 1 for v in claims.values()):
            disposition = ScientificDisposition.CONFLICTING_EVIDENCE
            reason = "Tests disagree on the same declared claim; preserve contradictions"
        elif against:
            disposition = ScientificDisposition.FALSIFIED_UNDER_DECLARED_TEST
            reason = "Conditional contradiction under retained test/control artifacts"
        elif any(f.verdict == EvidenceVerdict.CONSISTENT for f in admitted):
            disposition = (
                ScientificDisposition.ARTIFACT_CANDIDATE
                if hypothesis.explanation == Explanation.ARTIFACT
                else ScientificDisposition.KNOWN_MATCH_CANDIDATE
                if hypothesis.explanation == Explanation.KNOWN_MATCH
                else ScientificDisposition.CONSISTENT_UNDER_DECLARED_TESTS
            )
            reason = "Consistent with declared tests; no unique identity or discovery established"
        elif hypothesis.explanation == Explanation.UNRESOLVED:
            disposition = ScientificDisposition.UNRESOLVED
            reason = "Upstream explanation explicitly leaves object identity unresolved"
        assessments.append(
            HypothesisAssessment(
                hypothesis_id=hypothesis.id,
                disposition=disposition,
                tested_factor_ids=tuple(sorted(f.id for f in admitted)),
                contradiction_factor_ids=tuple(sorted(against)),
                reasons=(reason,),
            )
        )
    return tuple(assessments)


def identity_funnel(bundle: IdentityInput, budget: IdentityBudget) -> IdentityFunnel:
    bundle = IdentityInput.model_validate(bundle.model_dump(mode="json"))
    nodes = (
        len(bundle.observations)
        + len(bundle.measurements)
        + len(bundle.proposals)
        + len(bundle.hypotheses)
        + len(bundle.evidence_results)
        + sum(len(r.factors) for r in bundle.evidence_results)
    )
    if nodes > budget.max_graph_nodes:
        raise ValueError("identity graph exceeds node budget")
    assessments = assess(bundle)
    by_hypothesis = {a.hypothesis_id: a for a in assessments}
    parents = {p.id: p.id for p in bundle.proposals}

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a: str, b: str) -> None:
        x, y = sorted((root(a), root(b)))
        parents[y] = x

    by_member: dict[str, str] = {}
    for proposal in sorted(bundle.proposals, key=lambda p: p.id):
        for member in proposal.member_ids:
            if member in by_member:
                join(proposal.id, by_member[member])
            by_member[member] = proposal.id
    for hypothesis in bundle.hypotheses:
        for proposal_id in hypothesis.proposal_ids[1:]:
            join(hypothesis.proposal_ids[0], proposal_id)
    groups: dict[str, list[str]] = defaultdict(list)
    for proposal in bundle.proposals:
        groups[root(proposal.id)].append(proposal.id)
    proposals = {p.id: p for p in bundle.proposals}
    cases = []
    for ids in sorted(groups.values(), key=lambda values: tuple(sorted(values))):
        hids = tuple(sorted(h.id for h in bundle.hypotheses if set(h.proposal_ids) & set(ids)))
        surviving = [
            h
            for h in hids
            if by_hypothesis[h].disposition != ScientificDisposition.FALSIFIED_UNDER_DECLARED_TEST
        ]
        disposition = ScientificDisposition.UNRESOLVED
        reason = "No exclusive identity selected; all competing hypotheses retained"
        if not surviving and hids:
            reason = (
                "All supplied hypotheses contradicted; an absent alternative is not a discovery"
            )
        elif len(surviving) > 1:
            reason = "Multiple overlapping explanations survive; retain ambiguity"
        elif len(surviving) == 1:
            disposition = by_hypothesis[surviving[0]].disposition
            reason = (
                "One supplied explanation survives conditionally; candidate set may be incomplete"
            )
        cases.append(
            IdentityCase(
                id=content_id("identity-case", sorted(ids)),
                proposal_ids=tuple(sorted(ids)),
                hypothesis_ids=hids,
                member_ids=tuple(sorted({m for key in ids for m in proposals[key].member_ids})),
                disposition=disposition,
                reasons=(reason,),
            )
        )
    decisions = []
    surfaced = reviewed_hypotheses = 0
    # No mixed-method score aggregation: review order is a frozen stable case-ID order.
    for case in sorted(cases, key=lambda c: c.id):
        n = len(case.hypothesis_ids)
        fits = (
            surfaced < budget.review_cases
            and n <= budget.per_case_hypotheses
            and reviewed_hypotheses + n <= budget.review_hypotheses
        )
        if fits:
            surfaced += 1
            reviewed_hypotheses += n
        decisions.append(
            TriageDecision(
                subject_id=case.id,
                action="SURFACED" if fits else "DEFERRED",
                reasons=(
                    "bounded review of full competing family"
                    if fits
                    else "budget exceeded; complete family retained in graph",
                ),
            )
        )
    observations = {o.id: o for o in bundle.observations}
    invalid = [
        m.id
        for m in bundle.measurements
        if not m.quality.usable or not observations[m.observation_id].quality.usable
    ]
    decisions.extend(
        TriageDecision(
            subject_id=key,
            action="EXCLUDED",
            reasons=("unusable measurement retained; no physical artifact label inferred",),
        )
        for key in sorted(invalid)
    )
    return IdentityFunnel(
        cases=tuple(sorted(cases, key=lambda c: c.id)),
        assessments=assessments,
        decisions=tuple(decisions),
        counts={
            "observations": len(bundle.observations),
            "measurements": len(bundle.measurements),
            "measurements_usable": len(bundle.measurements) - len(invalid),
            "measurements_excluded": len(invalid),
            "proposals": len(bundle.proposals),
            "hypotheses": len(bundle.hypotheses),
            "cases": len(cases),
            "cases_surfaced": surfaced,
            "cases_deferred": len(cases) - surfaced,
            "hypotheses_reviewed": reviewed_hypotheses,
            "hypotheses_preserved": len(bundle.hypotheses),
            "evidence_results": len(bundle.evidence_results),
        },
    )
