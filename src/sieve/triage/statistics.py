"""Step-up selection on declared null tests, never a score-to-probability conversion."""

from collections.abc import Sequence
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from sieve.contracts.models import Contract, Identifier, Score, ScoreSemantics, Text


class NullTest(Contract):
    id: Identifier
    score: Score


class FDRFamily(Contract):
    id: Identifier
    expected_test_ids: tuple[Identifier, ...] = Field(min_length=1)
    null_model_ref: Identifier
    selection_accounting_ref: Identifier
    validity_justification: Text
    dependence: Literal["independent", "PRDS", "arbitrary", "unknown"]

    @model_validator(mode="after")
    def unique_family(self) -> Self:
        if len(set(self.expected_test_ids)) != len(self.expected_test_ids):
            raise ValueError("test family cannot repeat IDs")
        return self


class FDRResult(Contract):
    status: Literal["APPLIED_DECLARED_ASSUMPTIONS", "FDR_NOT_APPLICABLE"]
    method: Literal["BH", "BY"]
    alpha: float = Field(gt=0, lt=1)
    selected_ids: tuple[Identifier, ...] = ()
    reason: Text
    family_id: Identifier | None = None


class StatisticalFilter(Protocol):
    def apply(
        self,
        tests: Sequence[NullTest],
        family: FDRFamily | None,
        *,
        alpha: float,
        method: Literal["BH", "BY"],
    ) -> FDRResult: ...


class StepUpFDR:
    def apply(
        self,
        tests: Sequence[NullTest],
        family: FDRFamily | None,
        *,
        alpha: float = 0.05,
        method: Literal["BH", "BY"] = "BY",
    ) -> FDRResult:
        tests = [NullTest.model_validate(test) for test in tests]
        family = FDRFamily.model_validate(family) if family is not None else None
        if method not in {"BH", "BY"}:
            raise ValueError("unsupported FDR method")
        reason = None
        if family is None:
            reason = "No declared null model, complete family or selection accounting"
        elif len(tests) != len({t.id for t in tests}):
            reason = "Duplicate test IDs"
        elif {t.id for t in tests} != set(family.expected_test_ids):
            reason = "Tests do not equal the complete declared search family"
        elif any(
            t.score.semantics != ScoreSemantics.P_VALUE
            or not t.score.calibrated
            or t.score.value is None
            for t in tests
        ):
            reason = "All tests require defined calibrated null p-values, not anomaly scores"
        elif method == "BH" and family.dependence not in {"independent", "PRDS"}:
            reason = "BH requires justified independence or PRDS"
        if reason:
            return FDRResult(
                status="FDR_NOT_APPLICABLE",
                method=method,
                alpha=alpha,
                family_id=family.id if family else None,
                reason=reason,
            )
        values = sorted((t.score.value, t.id) for t in tests if t.score.value is not None)
        n = len(values)
        correction = sum(1 / i for i in range(1, n + 1)) if method == "BY" else 1
        k = max(
            (i for i, (p, _) in enumerate(values, 1) if p <= i * alpha / (n * correction)),
            default=0,
        )
        return FDRResult(
            status="APPLIED_DECLARED_ASSUMPTIONS",
            method=method,
            alpha=alpha,
            family_id=family.id if family else None,
            selected_ids=tuple(sorted(key for _, key in values[:k])),
            reason="Selection conditional on declared valid p-values and test family; "
            "not a case-level error guarantee or physical truth",
        )


class EmpiricalThreshold(Contract):
    """Contract for a future validation artifact; no calibration is manufactured."""

    method: Text
    score_semantics: ScoreSemantics
    direction: Literal["higher", "lower"]
    threshold: float
    validation_artifact_ref: Identifier
    validation_domain: Text
    injection_recovery_ref: Identifier | None = None
    limitations: tuple[Text, ...] = Field(min_length=1)
