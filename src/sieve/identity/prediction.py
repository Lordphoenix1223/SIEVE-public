"""Fit -> predict -> seal prediction -> reveal -> residual. No truth or orbit claim."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from sieve.contracts.models import (
    Finite,
    Identifier,
    MethodSpec,
    ObservationTime,
    PositionCovariance,
    Quality,
    SkyPosition,
    check_finite,
)
from sieve.identity.models import IdentityContract, Measurement, Observation, SourceKey
from sieve.identity.workflow import write_json
from sieve.provenance.models import (
    Agent,
    Entity,
    ProvenanceGraph,
    ProvenanceRecord,
    RunReceipt,
    content_id,
)
from sieve.provenance.runtime import source_identity


class PredictionContext(IdentityContract):
    observation_id: Identifier
    heldout_measurement_id: Identifier
    source: SourceKey
    dependence_group: Identifier
    time: ObservationTime
    observer_ref: Identifier | None = None
    observation_quality: Quality = Field(default_factory=Quality)


class LocationPrediction(IdentityContract):
    method: MethodSpec
    context: PredictionContext
    position: SkyPosition
    covariance: PositionCovariance
    training_ids: tuple[Identifier, ...]
    assumptions: tuple[str, ...] = Field(min_length=1)


class MotionPredictor[FitT: BaseModel](Protocol):
    def fit(
        self, observations: Sequence[Observation], measurements: Sequence[Measurement]
    ) -> FitT: ...
    def predict(self, state: FitT, context: PredictionContext) -> LocationPrediction: ...


class LinearTangentState(IdentityContract):
    origin: SkyPosition
    epoch: ObservationTime
    parameters: tuple[Finite, Finite, Finite, Finite]  # x0,y0,vx,vy
    parameter_units: tuple[
        Literal["arcsec"], Literal["arcsec"], Literal["arcsec/day"], Literal["arcsec/day"]
    ] = ("arcsec", "arcsec", "arcsec/day", "arcsec/day")
    parameter_covariance: tuple[tuple[Finite, ...], ...]
    training_ids: tuple[Identifier, ...]
    observer_ref: Identifier | None
    max_span_days: Finite = Field(gt=0, le=2)
    max_radius_arcsec: Finite = Field(gt=0, le=60)
    training_time_offsets_days: tuple[Finite, ...]

    @model_validator(mode="after")
    def valid_state(self) -> Self:
        c = np.asarray(self.parameter_covariance)
        if c.shape != (4, 4) or not np.array_equal(c, c.T) or np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError("baseline parameter covariance must be positive definite 4x4")
        if len(set(self.training_ids)) != len(self.training_ids) or len(self.training_ids) < 2:
            raise ValueError("invalid baseline fit membership")
        if len(self.training_ids) != len(self.training_time_offsets_days):
            raise ValueError("state membership/time offsets mismatch")
        return self


def _basis(
    position: SkyPosition,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    c = position.as_coord().icrs
    a, d = float(c.ra.rad), float(c.dec.rad)
    radial = np.array([math.cos(a) * math.cos(d), math.sin(a) * math.cos(d), math.sin(d)])
    east = np.array([-math.sin(a), math.cos(a), 0.0])
    north = np.array([-math.cos(a) * math.sin(d), -math.sin(a) * math.sin(d), math.cos(d)])
    return radial, east, north


def _project(
    position: SkyPosition, origin: SkyPosition
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    radial, east, north = _basis(origin)
    r, local_east, local_north = _basis(position)
    denominator = float(r @ radial)
    if denominator <= 0:
        raise ValueError("position outside supported tangent hemisphere")
    numbers = np.array([r @ east, r @ north])
    xy = numbers / denominator * (180 / math.pi) * 3600
    jacobian = np.array(
        [
            [
                (float(v @ axis) * denominator - numbers[i] * float(v @ radial)) / denominator**2
                for v in (local_east, local_north)
            ]
            for i, axis in enumerate((east, north))
        ]
    )
    return xy, jacobian


def _unproject(xy: NDArray[np.float64], origin: SkyPosition) -> SkyPosition:
    radial, east, north = _basis(origin)
    r = radial + east * xy[0] / 206264.80624709636 + north * xy[1] / 206264.80624709636
    r /= np.linalg.norm(r)
    return SkyPosition(
        longitude_deg=float(math.degrees(math.atan2(r[1], r[0])) % 360),
        latitude_deg=float(math.degrees(math.asin(r[2]))),
        frame="icrs",
    )


def _covariance(measurement: Measurement) -> NDArray[np.float64]:
    if (
        measurement.position is None
        or measurement.position.frame != "icrs"
        or measurement.covariance is None
    ):
        raise ValueError("baseline requires ICRS positions and explicit angular covariance")
    factor = float(u.Unit(measurement.covariance.unit).to(u.arcsec**2))
    c = np.asarray(measurement.covariance.values, dtype=float) * factor
    if np.linalg.eigvalsh(c).min() <= 0:
        raise ValueError("baseline requires positive definite positional covariance")
    if not measurement.quality.usable:
        raise ValueError("unusable measurement cannot be fitted or scored")
    return c


class LinearTangentBaseline:
    """Short-arc weighted straight-line reference only; no orbital dynamics."""

    def __init__(self, max_span_days: float = 2.0, max_radius_arcsec: float = 60.0):
        if not 0 < max_span_days <= 2 or not 0 < max_radius_arcsec <= 60:
            raise ValueError("baseline support limits cannot exceed 2 days / 60 arcsec")
        self.max_span_days = max_span_days
        self.max_radius_arcsec = max_radius_arcsec

    def fit(
        self, observations: Sequence[Observation], measurements: Sequence[Measurement]
    ) -> LinearTangentState:
        if len(measurements) < 2:
            raise ValueError("baseline fit needs at least two training measurements")
        obs = {o.id: o for o in observations}
        ordered = sorted(measurements, key=lambda m: m.id)
        if len({m.id for m in ordered}) != len(ordered):
            raise ValueError("duplicate training measurement")
        first = ordered[0]
        if first.position is None:
            raise ValueError("baseline needs measured positions")
        origin = first.position.as_coord().icrs
        center = SkyPosition(
            longitude_deg=float(origin.ra.deg), latitude_deg=float(origin.dec.deg), frame="icrs"
        )
        epoch = obs[first.observation_id].time
        if any(o.time.scale == "ut1" or not o.quality.usable for o in observations):
            raise ValueError("unsupported UT1 or unusable observation")
        if any(o.time.uncertainty_seconds not in (None, 0) for o in observations):
            raise ValueError("baseline does not model nonzero epoch uncertainty")
        observer_refs = {o.observer_ref for o in observations}
        if len(observer_refs) != 1:
            raise ValueError("baseline cannot qualify mixed observer contexts")
        times = [
            float((obs[m.observation_id].time.as_time().tai - epoch.as_time().tai).to_value(u.day))
            for m in ordered
        ]
        if max(times) - min(times) > self.max_span_days or len(set(times)) < 2:
            raise ValueError("time span unsupported or no distinct training epochs")
        normal = np.zeros((4, 4))
        rhs = np.zeros(4)
        for m, dt in zip(ordered, times, strict=True):
            c = _covariance(m)
            assert m.position is not None
            xy, jacobian = _project(m.position, center)
            if np.linalg.norm(xy) > self.max_radius_arcsec:
                raise ValueError("trajectory outside short-arc tangent support")
            weight = np.linalg.inv(jacobian @ c @ jacobian.T)
            design = np.array([[1, 0, dt, 0], [0, 1, 0, dt]], dtype=float)
            normal += design.T @ weight @ design
            rhs += design.T @ weight @ xy
        if np.linalg.cond(normal) > 1e12:
            raise ValueError("ill-conditioned baseline motion fit")
        covariance = np.linalg.inv(normal)
        covariance = (covariance + covariance.T) / 2
        parameters = covariance @ rhs
        return LinearTangentState(
            origin=center,
            epoch=epoch,
            parameters=tuple(float(v) for v in parameters),
            parameter_covariance=tuple(tuple(float(v) for v in row) for row in covariance),
            training_ids=tuple(m.id for m in ordered),
            observer_ref=next(iter(observer_refs)),
            max_span_days=self.max_span_days,
            max_radius_arcsec=self.max_radius_arcsec,
            training_time_offsets_days=tuple(times),
        )

    def predict(self, state: LinearTangentState, context: PredictionContext) -> LocationPrediction:
        if not context.observation_quality.usable:
            raise ValueError("held-out observation is unusable")
        if context.time.uncertainty_seconds not in (None, 0):
            raise ValueError("baseline does not model held-out epoch uncertainty")
        if context.time.scale == "ut1" or context.observer_ref != state.observer_ref:
            raise ValueError("prediction observer/time regime unsupported")
        dt = float((context.time.as_time().tai - state.epoch.as_time().tai).to_value(u.day))
        times = (*state.training_time_offsets_days, dt)
        if max(times) - min(times) > state.max_span_days:
            raise ValueError("held-out time outside baseline support")
        design = np.array([[1, 0, dt, 0], [0, 1, 0, dt]], dtype=float)
        xy = design @ np.asarray(state.parameters)
        if np.linalg.norm(xy) > state.max_radius_arcsec:
            raise ValueError("prediction outside baseline angular support")
        projected_covariance = design @ np.asarray(state.parameter_covariance) @ design.T
        position = _unproject(xy, state.origin)
        _, jacobian = _project(position, state.origin)
        inv = np.linalg.inv(jacobian)
        covariance = inv @ projected_covariance @ inv.T
        covariance = (covariance + covariance.T) / 2
        return LocationPrediction(
            method=MethodSpec(name="linear-tangent-baseline", version="v1"),
            context=context,
            position=position,
            covariance=PositionCovariance(
                values=(
                    (float(covariance[0, 0]), float(covariance[0, 1])),
                    (float(covariance[1, 0]), float(covariance[1, 1])),
                ),
                unit="arcsec2",
                basis="tangent_longitude_cos_latitude_latitude",
            ),
            training_ids=state.training_ids,
            assumptions=(
                "Short apparent angular arc only; no acceleration/parallax/orbit dynamics",
                "Supplied covariance, independent errors and linear propagation assumed",
                "Normalized residual is a diagnostic; no calibrated null probability",
            ),
        )


def normalized_residual(prediction: LocationPrediction, measurement: Measurement) -> float:
    if prediction.position.frame != "icrs":
        raise ValueError("default residual requires an ICRS forecast covariance basis")
    observed = _covariance(measurement)
    assert measurement.position is not None
    delta, jacobian = _project(measurement.position, prediction.position)
    forecast = np.asarray(prediction.covariance.values) * u.Unit(prediction.covariance.unit).to(
        u.arcsec**2
    )
    covariance = forecast + jacobian @ observed @ jacobian.T
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("undefined normalized residual covariance")
    return float(np.sqrt(delta @ np.linalg.solve(covariance, delta)))


def leave_one_observation_out[FitT: BaseModel](
    predictor: MotionPredictor[FitT],
    observations: Sequence[Observation],
    training: Sequence[Measurement],
    context: PredictionContext,
    reveal: Callable[[], Measurement],
    output: Path,
) -> dict[str, Any]:
    """Fresh immutable directory; prediction.json is sealed before reveal() is called."""
    if output.exists():
        raise ValueError("prediction output exists")
    if set(o.id for o in observations) != {m.observation_id for m in training}:
        raise ValueError("fit receives only observations supporting the N-1 measurements")
    if context.heldout_measurement_id in {m.id for m in training} or any(
        o.id == context.observation_id or o.dependence_group == context.dependence_group
        for o in observations
    ):
        raise ValueError("held-out observation/dependence group leaked into fit")
    if len({m.id for m in training}) != len(training) or len({o.id for o in observations}) != len(
        observations
    ):
        raise ValueError("duplicate fit input")
    # Reserve run directory before executing adapters. Incomplete runs lack artifacts.json.
    output.mkdir(parents=True)
    started = datetime.now(UTC)
    inputs = dict(
        observations=[o.model_dump(mode="json") for o in sorted(observations, key=lambda o: o.id)],
        measurements=[m.model_dump(mode="json") for m in sorted(training, key=lambda m: m.id)],
        context=context.model_dump(mode="json"),
    )
    write_json(output / "fit-input.json", inputs)
    fit_id = content_id("fit-input", inputs)
    status, reason = "SUCCESS", "fit/predict completed before held-out measurement was revealed"
    state_json: Any = None
    prediction_json: Any = None
    revealed_json: Any = None
    residual: float | None = None
    sealed_prediction: bytes | None = None
    reveal_attempted = False
    try:
        state = predictor.fit(observations, training)
        candidate_state = state.model_dump(mode="json")
        check_finite(candidate_state)
        state_json = candidate_state
        prediction = LocationPrediction.model_validate(
            predictor.predict(state, context).model_dump(mode="json")
        )
        if prediction.context != context or set(prediction.training_ids) != {
            m.id for m in training
        }:
            raise ValueError("adapter prediction does not match requested context/training IDs")
        prediction_json = prediction.model_dump(mode="json")
        write_json(output / "fit-state.json", state_json)
        write_json(output / "prediction.json", prediction_json)
        sealed_prediction = (output / "prediction.json").read_bytes()
        reveal_attempted = True
        revealed = Measurement.model_validate(reveal().model_dump(mode="json"))
        if (
            revealed.id != context.heldout_measurement_id
            or revealed.observation_id != context.observation_id
        ):
            raise ValueError("reveal returned the wrong held-out measurement")
        if (output / "prediction.json").read_bytes() != sealed_prediction:
            raise ValueError("prediction changed after reveal")
        revealed_json = revealed.model_dump(mode="json")
        write_json(output / "heldout.json", revealed_json)
        residual = normalized_residual(prediction, revealed)
    except ValueError as exc:
        status, reason = "UNSUPPORTED", str(exc)
    except Exception as exc:  # preserve failed adapter execution; never turn it into evidence
        status, reason = "ERROR", f"{type(exc).__name__}: {exc}"
    payload = dict(
        schema_version="sieve.identity.prediction.v1",
        status=status,
        reason=reason,
        normalized_residual=residual,
        residual_unit="1",
        score_semantics="LIKELIHOOD_LIKE",
        calibrated=False,
        training_ids=sorted(m.id for m in training),
        heldout_id=context.heldout_measurement_id,
        prediction=prediction_json,
        fit_state_sha256=content_id("state", state_json).split(":")[1]
        if state_json is not None
        else None,
        limitations=[
            "Linear tangent baseline is not an orbit/dynamics solution",
            "This protocol boundary is not a sandbox for malicious adapter code",
        ],
    )
    commit, tree, dirty, environment = source_identity()
    ended = datetime.now(UTC)
    generated_id = content_id("prediction-result", payload)
    used = [fit_id]
    entities = [Entity(id=fit_id, kind="fit-input", sha256=fit_id.split(":")[1])]
    if revealed_json is not None:
        heldout_id = content_id("revealed-measurement", revealed_json)
        used.append(heldout_id)
        entities.append(
            Entity(id=heldout_id, kind="heldout-evidence", sha256=heldout_id.split(":")[1])
        )
    entities.append(
        Entity(id=generated_id, kind="prediction-result", sha256=generated_id.split(":")[1])
    )
    # Per-stage hashes record admitted bytes and ordering; authenticity is externally anchored.
    stages = dict(
        fit_input_sha256=fit_id.split(":")[1],
        sealed_prediction_sha256=hashlib.sha256(sealed_prediction).hexdigest()
        if sealed_prediction is not None
        else None,
        revealed_input_sha256=content_id("heldout", revealed_json).split(":")[1]
        if revealed_json is not None
        else None,
        parameter_state=state_json,
        reveal_attempted=reveal_attempted,
        reveal_after_prediction=reveal_attempted and sealed_prediction is not None,
        context=context.model_dump(mode="json"),
    )
    stage_records = []
    common = dict(
        method_version="identity.v1",
        agent_ids=("prediction-runner",),
        code_commit=commit,
        software_version="identity.v1",
        request_parameters={},
        random_seeds={},
        environment_id=environment,
        started_at=started,
        ended_at=ended,
    )
    state_id = None
    if state_json is not None:
        state_id = content_id("fit-state", state_json)
        entities.append(Entity(id=state_id, kind="fit-state", sha256=state_id.split(":")[1]))
        stage_records.append(
            ProvenanceRecord(
                id="fit-stage",
                activity="N_minus_one_fit",
                used_entity_ids=(fit_id,),
                generated_entity_ids=(state_id,),
                derivations={state_id: (fit_id,)},
                configuration={"heldout_visible": False},
                **common,
            )
        )
    if prediction_json is not None and state_id is not None:
        prediction_id = content_id("sealed-prediction", prediction_json)
        entities.append(
            Entity(id=prediction_id, kind="sealed-prediction", sha256=prediction_id.split(":")[1])
        )
        stage_records.append(
            ProvenanceRecord(
                id="predict-stage",
                activity="predict_before_reveal",
                used_entity_ids=(fit_id, state_id),
                generated_entity_ids=(prediction_id,),
                derivations={prediction_id: (fit_id, state_id)},
                configuration={"heldout_visible": False},
                **common,
            )
        )
        used.append(prediction_id)
    record = ProvenanceRecord(
        id="prediction-protocol",
        activity="leave_one_observation_out",
        method_version="identity.v1",
        agent_ids=("prediction-runner",),
        used_entity_ids=tuple(used),
        generated_entity_ids=(generated_id,),
        derivations={generated_id: tuple(used)},
        code_commit=commit,
        software_version="identity.v1",
        configuration=stages,
        request_parameters={},
        random_seeds={},
        environment_id=environment,
        started_at=started,
        ended_at=ended,
    )
    graph = ProvenanceGraph(
        entities=tuple(entities),
        agents=(
            Agent(
                id="prediction-runner", name="prediction-runner", role="sealed_prediction_protocol"
            ),
        ),
        records=(*stage_records, record),
    )
    receipt = RunReceipt(
        id=content_id("loo-run", (fit_id, generated_id)),
        provenance_ref=record.id,
        input_sha256=content_id("all-inputs", (inputs, revealed_json)).split(":")[1],
        output_sha256=generated_id.split(":")[1],
        code_commit=commit,
        code_tree_sha256=tree,
        dirty=dirty,
        software_version="identity.v1",
        environment_id=environment,
        configuration=stages,
        random_seeds={},
        started_at=started,
        ended_at=ended,
    )
    write_json(output / "result.json", payload)
    write_json(output / "provenance.json", graph.model_dump(mode="json"))
    write_json(output / "run-receipt.json", receipt.model_dump(mode="json"))
    write_json(
        output / "artifacts.json",
        dict(
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.iterdir())
                if p.is_file()
            }
        ),
    )
    return payload
