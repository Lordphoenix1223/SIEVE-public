import copy
import json
from pathlib import Path

import pytest
from astropy.time import Time
from pydantic import ValidationError

from sieve.contracts.bundle import InferenceBundle
from sieve.contracts.models import (
    CandidateProposal,
    ObservationTime,
    PositionCovariance,
    QuantityValue,
    ScientificDisposition,
    Score,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/frankenstein_case.json"


@pytest.fixture
def data():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def bundle(data):
    return InferenceBundle.model_validate(data)


def test_serialization_round_trip(bundle):
    assert InferenceBundle.model_validate_json(bundle.model_dump_json()) == bundle
    assert "NaN" not in bundle.model_dump_json()


def test_extra_fields_rejected(data):
    data["hidden_truth"] = {"object": "secret"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        InferenceBundle.model_validate(data)


@pytest.mark.parametrize("unit", ["bananas", "not-a-unit"])
def test_bad_units(unit):
    with pytest.raises(ValidationError):
        QuantityValue(value=1, unit=unit)


def test_split_time_precision_and_scale():
    raw = Time("2026-10-05T00:00:00.123456789", scale="tai")
    stamp = ObservationTime(jd1=raw.jd1, jd2=raw.jd2, scale="tai")
    restored = ObservationTime.model_validate_json(stamp.model_dump_json())
    assert abs((restored.as_time() - raw).sec) < 1e-10
    with pytest.raises(ValidationError):
        ObservationTime(jd1=1, jd2=0, scale="local")


@pytest.mark.parametrize("values", [((1, 2), (2, 1)), ((1, 1), (0, 1))])
def test_invalid_covariance(values):
    with pytest.raises(ValidationError):
        PositionCovariance(values=values, basis="tangent_longitude_cos_latitude_latitude")


def test_covariance_unit():
    with pytest.raises(ValidationError):
        PositionCovariance(
            values=((1, 0), (0, 1)), unit="arcsec", basis="tangent_longitude_cos_latitude_latitude"
        )


def test_same_exposure_not_independent(data):
    sample = copy.deepcopy(data["observations"][1])
    sample["exposure_id"] = data["observations"][0]["exposure_id"]
    data["observations"][1] = sample
    with pytest.raises(ValidationError, match="dependence group"):
        InferenceBundle.model_validate(data)


def test_conflicting_observation_id(data):
    data["observations"][-1]["release"] = "changed"
    with pytest.raises(ValidationError, match="conflicting duplicate"):
        InferenceBundle.model_validate(data)


def test_missing_provenance(data):
    data["detections"][0]["provenance_ref"] = "missing"
    with pytest.raises(ValidationError, match="producing provenance"):
        InferenceBundle.model_validate(data)


def test_missing_asset(data):
    data["detections"][0]["asset_ids"] = ["missing"]
    with pytest.raises(ValidationError, match="source asset"):
        InferenceBundle.model_validate(data)


def test_duplicate_member_ids(data):
    p = data["proposals"][0]
    p["member_ids"].append(p["member_ids"][0])
    with pytest.raises(ValidationError, match="duplicate member"):
        CandidateProposal.model_validate(p)


@pytest.mark.parametrize("semantics", ["CALIBRATED_PROBABILITY", "P_VALUE"])
def test_uncalibrated_cannot_masquerade_as_probability(data, semantics):
    raw = data["proposals"][0]["score"]
    raw["semantics"] = semantics
    with pytest.raises(ValidationError):
        Score.model_validate(raw)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_probability_range_and_finite(value):
    with pytest.raises(ValidationError):
        Score(
            value=value,
            semantics="CALIBRATED_PROBABILITY",
            direction="higher",
            calibrated=True,
            calibration_ref="c",
            domain="fixture",
            event_or_null="event",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_undefined_numbers_rejected(value):
    with pytest.raises(ValidationError):
        QuantityValue(value=value, unit="uJy")


def test_unknown_values_are_explicit_null():
    quantity = QuantityValue(value=None, unit="uJy", missing_reason="not_measured")
    assert json.loads(quantity.model_dump_json())["value"] is None
    with pytest.raises(ValidationError):
        QuantityValue(value=None, unit="uJy")


def test_nested_nonfinite_configuration(data):
    data["provenance"]["records"][0]["configuration"] = {"nested": [float("nan")]}
    with pytest.raises(ValidationError, match="finite"):
        InferenceBundle.model_validate(data)


def test_no_discovery_disposition():
    assert all("DISCOVERED" not in d and "NEW_" not in d for d in ScientificDisposition)


def test_global_id_collision(data):
    data["proposals"][0]["id"] = data["detections"][0]["id"]
    with pytest.raises(ValidationError, match="globally unique"):
        InferenceBundle.model_validate(data)


def test_unknown_proposal_detection(data):
    data["proposals"][0]["member_ids"].append("missing")
    with pytest.raises(ValidationError, match="unknown detection"):
        InferenceBundle.model_validate(data)


def test_release_flags_need_definition(data):
    data["observations"][0]["quality"] = {"bitmask": 1}
    with pytest.raises(ValidationError, match="definition reference"):
        InferenceBundle.model_validate(data)


def test_wavelength_is_not_flux(data):
    data["observations"][0]["wavelengths"][0]["unit"] = "Jy"
    with pytest.raises(ValidationError, match="length units"):
        InferenceBundle.model_validate(data)
