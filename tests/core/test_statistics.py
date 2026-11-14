import pytest

from sieve.contracts.models import Score
from sieve.triage.statistics import FDRFamily, NullTest, StepUpFDR


def pvalue(value):
    return Score(
        value=value,
        semantics="P_VALUE",
        direction="lower",
        calibrated=True,
        calibration_ref="synthetic-calibration",
        domain="synthetic-null",
        event_or_null="uniform-null-test",
    )


def family(dependence="independent"):
    return FDRFamily(
        id="family",
        expected_test_ids=("t1", "t2", "t3"),
        null_model_ref="null",
        selection_accounting_ref="predeclared-family",
        validity_justification="Synthetic example only",
        dependence=dependence,
    )


def null_tests():
    return [
        NullTest(id=key, score=pvalue(value))
        for key, value in (("t1", 0.01), ("t2", 0.03), ("t3", 0.8))
    ]


def test_bh_step_up_known_result():
    result = StepUpFDR().apply(null_tests(), family(), method="BH", alpha=0.05)
    assert result.selected_ids == ("t1", "t2")
    assert result.status == "APPLIED_DECLARED_ASSUMPTIONS"


def test_by_known_result_arbitrary_dependence():
    result = StepUpFDR().apply(null_tests(), family("arbitrary"), method="BY", alpha=0.05)
    assert result.selected_ids == ()
    assert result.status == "APPLIED_DECLARED_ASSUMPTIONS"


def test_no_null_family_is_not_applicable():
    assert StepUpFDR().apply(null_tests(), None).status == "FDR_NOT_APPLICABLE"


@pytest.mark.parametrize(
    "semantics", ["RANKING_ONLY", "UNKNOWN_UNCALIBRATED", "PSEUDO_PROBABILITY", "LIKELIHOOD_LIKE"]
)
def test_fdr_rejects_arbitrary_scores(semantics):
    t = null_tests()
    t[0] = NullTest(
        id="t1",
        score=Score(
            value=0.00001,
            semantics=semantics,
            direction="higher",
            domain="toy",
            event_or_null="interest",
        ),
    )
    assert StepUpFDR().apply(t, family()).status == "FDR_NOT_APPLICABLE"


def test_probability_is_not_a_pvalue():
    t = null_tests()
    t[0] = NullTest(
        id="t1",
        score=Score(
            value=0.99,
            semantics="CALIBRATED_PROBABILITY",
            direction="higher",
            calibrated=True,
            calibration_ref="c",
            domain="toy",
            event_or_null="object_is_neo",
        ),
    )
    assert StepUpFDR().apply(t, family()).status == "FDR_NOT_APPLICABLE"


def test_incomplete_selected_family_is_not_applicable():
    assert StepUpFDR().apply(null_tests()[:1], family()).status == "FDR_NOT_APPLICABLE"


def test_bh_unknown_dependence_is_not_applicable():
    assert (
        StepUpFDR().apply(null_tests(), family("unknown"), method="BH").status
        == "FDR_NOT_APPLICABLE"
    )


def test_duplicate_tests_are_not_applicable():
    assert StepUpFDR().apply([null_tests()[0]] * 3, family()).status == "FDR_NOT_APPLICABLE"


def test_missing_pvalue_is_not_applicable():
    t = null_tests()
    t[0] = NullTest(
        id="t1",
        score=Score(
            value=None,
            semantics="P_VALUE",
            direction="lower",
            calibrated=True,
            calibration_ref="c",
            domain="toy",
            event_or_null="null",
            missing_reason="not_tested",
        ),
    )
    assert StepUpFDR().apply(t, family()).status == "FDR_NOT_APPLICABLE"


@pytest.mark.parametrize("alpha", [0, 1, float("nan"), float("inf")])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        StepUpFDR().apply(null_tests(), family(), alpha=alpha)


def test_unsafe_score_construction_cannot_bypass_fdr_validation():
    forged = Score.model_construct(
        value=-1,
        semantics="P_VALUE",
        direction="lower",
        calibrated=True,
        calibration_ref="c",
        domain="toy",
        event_or_null="null",
    )
    with pytest.raises(ValueError):
        NullTest(id="t1", score=forged)
