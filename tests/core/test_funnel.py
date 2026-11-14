import copy
from pathlib import Path

import pytest

from sieve.contracts.bundle import InferenceBundle
from sieve.contracts.models import (
    CandidateProposal,
    EvidenceResult,
    ScientificDisposition,
    TriageAction,
)
from sieve.pipeline import build_cases, triage
from sieve.triage.budget import CandidateBudget
from sieve.triage.dedup import DedupPolicy, collapse_proposals

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def bundle():
    return InferenceBundle.model_validate_json((FIXTURES / "frankenstein_case.json").read_text())


def test_observation_and_membership_dedup(bundle):
    cases, _, counts = build_cases(bundle, CandidateBudget(), DedupPolicy())
    assert counts["observations_deduplicated"] == 1
    assert counts["proposals_merged"] == 1
    clean = next(c for c in cases if c.proposals[0].id == "p-clean")
    assert {p.id for p in clean.proposals} == {"p-clean", "p-exact"}
    assert {h.id for h in clean.hypotheses} == {"h-clean", "h-exact"}


def test_approximate_dedup_keeps_original_memberships(bundle):
    cases, _, counts = build_cases(bundle, CandidateBudget(), DedupPolicy(approximate=True))
    assert counts["proposals_merged"] == 2
    clean = next(c for c in cases if c.proposals[0].id == "p-clean")
    assert len(clean.proposals) == 3
    assert len(clean.proposals[-1].member_ids) == 3
    assert len(clean.proposals[0].member_ids) == 4
    assert any(c.proposals[0].id == "p-alternative" for c in cases)


def test_frankenstein_preserves_competing_explanations(bundle):
    cases, _, _ = build_cases(bundle, CandidateBudget(), DedupPolicy(approximate=True))
    case = next(c for c in cases if c.proposals[0].id == "p-frankenstein")
    assert {h.object_count for h in case.hypotheses} == {1, 2}
    assert case.hypotheses[0].member_ids == case.hypotheses[1].member_ids


def test_determinism_under_input_permutation(bundle):
    first = triage(bundle, CandidateBudget(total=4), DedupPolicy(approximate=True))
    data = bundle.model_dump(mode="json")
    for key in ("observations", "detections", "proposals", "hypotheses", "evidence_results"):
        data[key].reverse()
    second = triage(
        InferenceBundle.model_validate(data),
        CandidateBudget(total=4),
        DedupPolicy(approximate=True),
    )
    assert first == second


@pytest.mark.parametrize("budget", [0, 1, 3, 5, 1000])
def test_budget_enforced(bundle, budget):
    reports, decisions, counts = triage(bundle, CandidateBudget(total=budget), DedupPolicy())
    assert len(reports) <= budget
    assert counts["cases_surfaced"] + counts["cases_deferred"] == counts["cases_normalized"]
    assert sum(d.action == TriageAction.DEFERRED for d in decisions) == counts["cases_deferred"]


def test_diversity_under_pressure():
    bundle = InferenceBundle.model_validate_json((FIXTURES / "stress_case.json").read_text())
    reports, _, counts = triage(bundle, CandidateBudget(total=5), DedupPolicy())
    assert counts["proposals_raw"] == 246
    assert counts["cases_surfaced"] == 5
    assert len({r.case.scope.region for r in reports}) == 5
    assert any(r.case.proposals[0].score.value < 0 for r in reports)
    assert counts["cases_deferred"] == 241


@pytest.mark.parametrize("dimension", ["per_region", "per_time_window", "per_regime"])
def test_dimension_caps(bundle, dimension):
    budget = CandidateBudget(total=100, **{dimension: 1})
    reports, _, _ = triage(bundle, budget, DedupPolicy())
    attr = {"per_region": "region", "per_time_window": "time_window", "per_regime": "regime"}[
        dimension
    ]
    scopes = [getattr(r.case.scope, attr) for r in reports]
    assert len(scopes) == len(set(scopes))


def test_screening_is_audited(bundle):
    _, decisions, counts = triage(bundle, CandidateBudget(), DedupPolicy())
    assert counts["detections_suppressed"] == 2
    assert counts["proposals_suppressed"] == 2
    assert {"d-static", "d-unusable", "p-static", "p-unusable"} <= {
        d.subject_id for d in decisions if d.action == TriageAction.SUPPRESSED
    }


def dispositions(bundle):
    reports, _, _ = triage(bundle, CandidateBudget(total=100), DedupPolicy())
    return {a.hypothesis_id: a.disposition for r in reports for a in r.assessments}


def test_catalog_no_match_and_high_score_are_not_new_objects(bundle):
    d = dispositions(bundle)
    assert d["h-high-anomaly"] == ScientificDisposition.INSUFFICIENT_EVIDENCE
    assert d["h-insufficient"] == ScientificDisposition.INSUFFICIENT_EVIDENCE
    assert d["h-artifact"] == ScientificDisposition.LIKELY_ARTIFACT
    assert d["h-clean"] == ScientificDisposition.CONSISTENT_MOVING_SOURCE


def test_agreement_is_not_truth(bundle):
    data = bundle.model_dump(mode="json")
    result = next(
        r for r in data["evidence_results"] if r["request"]["hypothesis_id"] == "h-high-anomaly"
    )
    result["factors"][0]["interpretation"] = "ALGORITHM_AGREEMENT"
    assert dispositions(InferenceBundle.model_validate(data))["h-high-anomaly"] == (
        ScientificDisposition.INSUFFICIENT_EVIDENCE
    )


@pytest.mark.parametrize("status", ["ERROR", "TIMEOUT", "MISSING"])
def test_execution_failure_is_not_scientific_rejection(bundle, status):
    data = bundle.model_dump(mode="json")
    result = next(r for r in data["evidence_results"] if r["request"]["hypothesis_id"] == "h-clean")
    result.update(status=status, factors=[], reason="execution_failed")
    assert dispositions(InferenceBundle.model_validate(data))["h-clean"] == (
        ScientificDisposition.INSUFFICIENT_EVIDENCE
    )


def test_failed_execution_cannot_return_factors(bundle):
    raw = bundle.evidence_results[0].model_dump(mode="json")
    raw["status"] = "TIMEOUT"
    with pytest.raises(ValueError, match="failed execution"):
        EvidenceResult.model_validate(raw)


def test_unsupported_regime(bundle):
    data = bundle.model_dump(mode="json")
    result = next(r for r in data["evidence_results"] if r["request"]["hypothesis_id"] == "h-clean")
    result.update(status="UNSUPPORTED", factors=[], reason="regime_not_supported")
    assert dispositions(InferenceBundle.model_validate(data))["h-clean"] == (
        ScientificDisposition.UNSUPPORTED_REGIME
    )


def test_near_identical_without_shared_members_requires_family(bundle):
    p = bundle.proposals[0].model_dump(mode="json")
    p.update(id="p-near", member_ids=["d-a1"], equivalence_family="a-sampling")
    near = CandidateProposal.model_validate(p)
    d = {d.id: d for d in bundle.detections}
    o = {o.id: o for o in bundle.observations}
    policy = DedupPolicy(approximate=True, separation_arcsec=10, time_seconds=10000)
    groups = collapse_proposals([bundle.proposals[0], near], d, o, policy)
    assert len(groups) == 1
    p["equivalence_family"] = None
    groups = collapse_proposals(
        [bundle.proposals[0], CandidateProposal.model_validate(p)], d, o, policy
    )
    assert len(groups) == 2


def test_anchor_grouping_does_not_chain(bundle):
    d = {d.id: d for d in bundle.detections}
    o = {o.id: o for o in bundle.observations}
    raw = bundle.proposals[0].model_dump(mode="json")
    proposals = []
    for i in range(3):
        p = copy.deepcopy(raw)
        p.update(id=f"p-{i}", member_ids=[f"d-a{i}"])
        proposals.append(CandidateProposal.model_validate(p))
    groups = collapse_proposals(
        proposals, d, o, DedupPolicy(approximate=True, separation_arcsec=4, time_seconds=10000)
    )
    assert [len(g) for g in groups] == [2, 1]


def test_unsafe_adapter_construct_is_revalidated(bundle):
    bad = InferenceBundle.model_construct(**{**bundle.__dict__, "detections": ()})
    with pytest.raises(ValueError):
        triage(bad, CandidateBudget(), DedupPolicy())


def test_ut1_is_explicitly_unsupported_offline(bundle):
    data = bundle.model_dump(mode="json")
    # A non-anchor member must also trigger refusal, before any attempted IERS access.
    data["observations"][2]["time"]["scale"] = "ut1"
    with pytest.raises(ValueError, match="pinned Earth orientation"):
        triage(InferenceBundle.model_validate(data), CandidateBudget(), DedupPolicy())


def test_nonfinite_ranker_cannot_bypass_budget(bundle):
    class BadRanker:
        def cohort(self, case):
            return "bad"

        def priority(self, case):
            return float("nan")

    with pytest.raises(ValueError, match="nonfinite priority"):
        triage(bundle, CandidateBudget(), DedupPolicy(), ranker=BadRanker())


def test_hypothesis_budget_does_not_truncate_competing_explanations(bundle):
    reports, _, counts = triage(
        bundle,
        CandidateBudget(total=100, per_case_hypotheses=1, total_hypotheses=2),
        DedupPolicy(approximate=True),
    )
    assert sum(len(r.case.hypotheses) for r in reports) <= 2
    assert all(len(r.case.hypotheses) <= 1 for r in reports)
    assert not any(p.id == "p-frankenstein" for r in reports for p in r.case.proposals)
    assert counts["cases_deferred"] > 0
    # Its intact competing hypotheses remain in the normalization layer/input.
    cases, _, _ = build_cases(bundle, CandidateBudget(), DedupPolicy())
    mixed = next(c for c in cases if c.proposals[0].id == "p-frankenstein")
    assert len(mixed.hypotheses) == 2
