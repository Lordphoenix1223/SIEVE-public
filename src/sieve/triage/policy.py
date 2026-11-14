"""Conservative reference interpretation; real measurement rules belong to science."""

from sieve.contracts.models import ExecutionStatus, Interpretation, ScientificDisposition
from sieve.provenance.models import content_id
from sieve.reporting import Assessment, Case


class ReferenceRanker:
    def cohort(self, case: Case) -> str:
        score = case.proposals[0].score
        method = case.proposals[0].method
        if score is None:
            return "unscored"
        return content_id(
            "cohort",
            (
                method.name,
                method.version,
                score.semantics,
                score.domain,
                score.event_or_null,
                score.direction,
                score.calibration_ref,
            ),
        )

    def priority(self, case: Case) -> float:
        score = case.proposals[0].score
        if score is None or score.value is None:
            return 0
        return score.value if score.direction == "higher" else -score.value


class ConservativeTriage:
    def assess(self, case: Case) -> tuple[Assessment, ...]:
        assessments = []
        for hypothesis in case.hypotheses:
            results = [r for r in case.evidence_results if r.request.hypothesis_id == hypothesis.id]
            positive = {f.interpretation for r in results for f in r.factors if f.polarity == "for"}
            against = any(f.polarity == "against" for r in results for f in r.factors)
            failures = {r.status for r in results if r.status != ExecutionStatus.SUCCESS}
            disposition = ScientificDisposition.INSUFFICIENT_EVIDENCE
            reason = (
                "No decisive conditional evidence; scores and NO_MATCH do not establish identity"
            )
            if failures & {ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT, ExecutionStatus.MISSING}:
                reason = "Evidence execution incomplete; failure is not scientific rejection"
            elif ExecutionStatus.UNSUPPORTED in failures:
                disposition = ScientificDisposition.UNSUPPORTED_REGIME
                reason = "Requested method does not support this regime"
            elif Interpretation.AMBIGUOUS_IDENTITY in positive:
                disposition = ScientificDisposition.AMBIGUOUS_IDENTITY
                reason = "Evidence explicitly records competing identity explanations"
            elif against or (
                Interpretation.ARTIFACT_SUGGESTIVE in positive
                and Interpretation.MOTION_CONSISTENT in positive
            ):
                disposition = ScientificDisposition.REVIEW_REQUIRED
                reason = "Conflicting evidence requires review"
            elif Interpretation.ARTIFACT_SUGGESTIVE in positive:
                disposition = ScientificDisposition.LIKELY_ARTIFACT
                reason = "Conditional artifact evidence; not a truth label"
            elif Interpretation.CATALOG_MATCH in positive:
                disposition = ScientificDisposition.KNOWN_MATCH
                reason = "Conditional association to a catalog record; not authoritative truth"
            elif Interpretation.MOTION_CONSISTENT in positive:
                disposition = ScientificDisposition.CONSISTENT_MOVING_SOURCE
                reason = "Consistent under the evidence method's stated assumptions and limitations"
            assessments.append(
                Assessment(hypothesis_id=hypothesis.id, disposition=disposition, reasons=(reason,))
            )
        return tuple(assessments)
