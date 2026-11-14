"""Validated measurements and conditional claims, with no production truth model."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Text = Annotated[str, Field(min_length=1)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


def check_finite(value: object) -> None:
    """Reject non-JSON numbers even inside free configuration/parameter trees."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN/Infinity are forbidden; use null with an explicit reason")
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            check_finite(item)


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, revalidate_instances="always"
    )
    schema_version: Literal["sieve.v0"] = "sieve.v0"

    @model_validator(mode="after")
    def finite_json(self) -> Self:
        check_finite(self.model_dump(mode="python"))
        return self


class QuantityValue(Contract):
    value: Finite | None
    unit: Text
    uncertainty: Annotated[Finite, Field(ge=0)] | None = None
    missing_reason: Text | None = None

    @model_validator(mode="after")
    def valid_quantity(self) -> Self:
        u.Unit(self.unit, parse_strict="raise")
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("null requires missing_reason; present values cannot have one")
        if self.value is None and self.uncertainty is not None:
            raise ValueError("undefined value cannot have a numeric uncertainty")
        return self


class ObservationTime(Contract):
    """Split Julian date avoids rounding away subsecond precision in one float."""

    jd1: Finite
    jd2: Finite
    scale: Literal["utc", "tai", "tt", "tdb", "tcb", "tcg", "ut1"]
    uncertainty_seconds: Annotated[Finite, Field(ge=0)] | None = None

    def as_time(self) -> Time:
        return Time(self.jd1, self.jd2, format="jd", scale=self.scale)


class SkyPosition(Contract):
    longitude_deg: Annotated[Finite, Field(ge=0, lt=360)]
    latitude_deg: Annotated[Finite, Field(ge=-90, le=90)]
    frame: Literal["icrs", "galactic"]

    def as_coord(self) -> SkyCoord:
        return SkyCoord(self.longitude_deg * u.deg, self.latitude_deg * u.deg, frame=self.frame)


class PositionCovariance(Contract):
    values: tuple[tuple[Finite, Finite], tuple[Finite, Finite]]
    unit: Text = "arcsec2"
    basis: Literal["tangent_longitude_cos_latitude_latitude"]

    @model_validator(mode="after")
    def valid_covariance(self) -> Self:
        if not u.Unit(self.unit).is_equivalent(u.arcsec**2):
            raise ValueError("position covariance must have squared angular units")
        a = np.asarray(self.values)
        if not np.array_equal(a, a.T) or np.linalg.eigvalsh(a).min() < 0:
            raise ValueError("covariance must be symmetric positive semidefinite")
        return self


class Quality(Contract):
    flags: tuple[Text, ...] = ()
    bitmask: Annotated[int, Field(ge=0)] | None = None
    flag_definition_ref: Text | None = None
    usable: bool = True

    @model_validator(mode="after")
    def flags_documented(self) -> Self:
        if self.bitmask is not None and self.flag_definition_ref is None:
            raise ValueError("a bitmask requires its release-specific definition reference")
        return self


class MethodSpec(Contract):
    name: Identifier
    version: Text


class ScoreSemantics(StrEnum):
    RANKING_ONLY = "RANKING_ONLY"
    CALIBRATED_PROBABILITY = "CALIBRATED_PROBABILITY"
    LIKELIHOOD_LIKE = "LIKELIHOOD_LIKE"
    P_VALUE = "P_VALUE"
    PSEUDO_PROBABILITY = "PSEUDO_PROBABILITY"
    UNKNOWN_UNCALIBRATED = "UNKNOWN_UNCALIBRATED"


class Score(Contract):
    value: Finite | None
    semantics: ScoreSemantics
    direction: Literal["higher", "lower"]
    calibrated: bool = False
    calibration_ref: Identifier | None = None
    domain: Text
    event_or_null: Text
    missing_reason: Text | None = None

    @model_validator(mode="after")
    def honest_score(self) -> Self:
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("undefined score requires missing_reason")
        probabilistic = self.semantics in {
            ScoreSemantics.CALIBRATED_PROBABILITY,
            ScoreSemantics.P_VALUE,
        }
        if probabilistic:
            if not self.calibrated or not self.calibration_ref:
                raise ValueError("probability/p-value requires identified calibration")
            if self.value is not None and not 0 <= self.value <= 1:
                raise ValueError("probability/p-value must be in [0, 1]")
        elif self.calibrated:
            raise ValueError("use probabilistic semantics for statistical calibration")
        if self.semantics == ScoreSemantics.P_VALUE and self.direction != "lower":
            raise ValueError("null p-values must use lower direction")
        return self


class Observation(Contract):
    id: Identifier
    exposure_id: Identifier | None = None
    dataset: Text
    provider: Text
    release: Text
    time: ObservationTime
    position: SkyPosition
    covariance: PositionCovariance | None = None
    observer_ref: Text | None = None
    dependence_group: Identifier
    wavelengths: tuple[QuantityValue, ...] = ()
    bandpass_ref: Text | None = None
    quality: Quality = Field(default_factory=Quality)
    asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    provenance_ref: Identifier

    @model_validator(mode="after")
    def wavelength_units(self) -> Self:
        for wavelength in self.wavelengths:
            if not u.Unit(wavelength.unit).is_equivalent(u.m):
                raise ValueError("wavelengths require length units")
            if wavelength.value is not None and wavelength.value <= 0:
                raise ValueError("wavelength must be positive")
        return self


class Detection(Contract):
    id: Identifier
    observation_id: Identifier
    position: SkyPosition
    covariance: PositionCovariance | None = None
    measurements: dict[Identifier, QuantityValue]
    method: MethodSpec
    quality: Quality = Field(default_factory=Quality)
    asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    score: Score | None = None
    provenance_ref: Identifier


class CandidateProposal(Contract):
    id: Identifier
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    method: MethodSpec
    task_type: Identifier
    explanation_key: Identifier
    # A method-authored assertion that approximate proposals are interchangeable
    # for review grouping, not an assertion of physical identity.
    equivalence_family: Identifier | None = None
    score: Score | None = None
    provenance_ref: Identifier

    @model_validator(mode="after")
    def normalized_members(self) -> Self:
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("duplicate member IDs inside a proposal")
        object.__setattr__(self, "member_ids", tuple(sorted(self.member_ids)))
        return self


class ObjectHypothesis(Contract):
    id: Identifier
    proposal_ids: tuple[Identifier, ...] = Field(min_length=1)
    member_ids: tuple[Identifier, ...] = Field(min_length=1)
    object_count: Annotated[int, Field(ge=0)] | None
    explanation: Text
    provenance_ref: Identifier

    @model_validator(mode="after")
    def unique_support(self) -> Self:
        if len(set(self.proposal_ids)) != len(self.proposal_ids):
            raise ValueError("duplicate supporting proposal IDs")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("duplicate hypothesis member IDs")
        return self


class Interpretation(StrEnum):
    MOTION_CONSISTENT = "MOTION_CONSISTENT"
    ARTIFACT_SUGGESTIVE = "ARTIFACT_SUGGESTIVE"
    CATALOG_MATCH = "CATALOG_MATCH"
    CATALOG_NO_MATCH = "CATALOG_NO_MATCH"
    ALGORITHM_AGREEMENT = "ALGORITHM_AGREEMENT"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    OTHER = "OTHER"


class EvidenceFactor(Contract):
    id: Identifier
    hypothesis_id: Identifier
    method: MethodSpec
    input_ids: tuple[Identifier, ...] = Field(min_length=1)
    measurements: dict[Identifier, QuantityValue]
    polarity: Literal["for", "against", "neutral"]
    interpretation: Interpretation
    rationale: Text
    calibrated: bool = False
    calibration_ref: Identifier | None = None
    provenance_ref: Identifier
    limitations: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def calibrated_evidence(self) -> Self:
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("duplicate evidence input IDs")
        if self.calibrated and not self.calibration_ref:
            raise ValueError("calibrated evidence requires a calibration artifact")
        return self


class ExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED = "UNSUPPORTED"
    MISSING = "MISSING"


class EvidenceRequest(Contract):
    id: Identifier
    hypothesis_id: Identifier
    method: MethodSpec
    input_ids: tuple[Identifier, ...] = Field(min_length=1)
    provenance_ref: Identifier

    @model_validator(mode="after")
    def unique_inputs(self) -> Self:
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("duplicate request input IDs")
        return self


class EvidenceResult(Contract):
    request: EvidenceRequest
    status: ExecutionStatus
    factors: tuple[EvidenceFactor, ...] = ()
    reason: Text

    @model_validator(mode="after")
    def result_consistency(self) -> Self:
        if self.status != ExecutionStatus.SUCCESS and self.factors:
            raise ValueError("failed execution cannot provide scientific evidence")
        if any(f.hypothesis_id != self.request.hypothesis_id for f in self.factors):
            raise ValueError("evidence must address the requested hypothesis")
        return self


class ScientificDisposition(StrEnum):
    KNOWN_MATCH = "KNOWN_MATCH"
    LIKELY_ARTIFACT = "LIKELY_ARTIFACT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONSISTENT_MOVING_SOURCE = "CONSISTENT_MOVING_SOURCE"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNSUPPORTED_REGIME = "UNSUPPORTED_REGIME"


class TriageAction(StrEnum):
    SURFACED = "surfaced"
    SUPPRESSED = "suppressed"
    MERGED_AS_DUPLICATE = "merged_as_duplicate"
    DEFERRED = "deferred"
    ESCALATED = "escalated"
    REVIEW_REQUIRED = "review_required"


class TriageDecision(Contract):
    subject_id: Identifier
    action: TriageAction
    reasons: tuple[Text, ...] = Field(min_length=1)
    related_id: Identifier | None = None
