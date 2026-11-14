"""Real QR3 records and competing hypotheses through the existing identity v1 intake.

Image profile diagnostics remain INSUFFICIENT evidence. This adapter does not
invent covariance, reference identities, known matches or posterior probabilities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from astropy.time import Time
from pydantic import JsonValue

from sieve.contracts.models import (
    ExecutionStatus,
    MethodSpec,
    ObservationTime,
    Quality,
    QuantityValue,
    SkyPosition,
)
from sieve.identity.models import (
    CandidateProposal,
    EvidenceFactor,
    EvidenceRequest,
    EvidenceResult,
    EvidenceVerdict,
    Explanation,
    IdentityInput,
    Measurement,
    ObjectAssignment,
    ObjectHypothesis,
    Observation,
    ScalarMeasurement,
    SourceKey,
)
from sieve.interfaces import ObservationBatch
from sieve.photometry.qualified import file_digest, load_qr3_frame
from sieve.provenance.models import Agent, Entity, ProvenanceGraph, ProvenanceRecord

METHOD = MethodSpec(name="qr3-finite-epsf-pilot", version="1")
TESTED_CLAIM = "held-out spatial profile distinguishes conditional object count"


def _entity(model: Any) -> Entity:
    return Entity(
        id=model.id,
        kind=type(model).__name__,
        sha256=hashlib.sha256(model.model_dump_json().encode()).hexdigest(),
    )


def _record(
    identifier: str, generated: tuple[str, ...], parents: tuple[str, ...], report: dict[str, Any]
) -> ProvenanceRecord:
    registered = datetime.now(UTC)
    return ProvenanceRecord(
        id=identifier,
        activity="register_qr3_pilot_lineage",
        method_version="1",
        agent_ids=("sieve-qr3-pilot",),
        used_entity_ids=parents,
        generated_entity_ids=generated,
        derivations=dict.fromkeys(generated, parents),
        code_commit=report["code_commit"],
        software_version="sieve.qr3.pilot.v1",
        configuration={
            "qualified_identity": False,
            "measurement_started_at": report["started_at"],
            "measurement_ended_at": report["ended_at"],
            "timestamp_semantics": "lineage registration, not original asset creation",
        },
        request_parameters={},
        random_seeds={},
        environment_id=report["environment_id"],
        started_at=registered,
        ended_at=registered,
    )


class QR3ObservationProvider:
    """Load only the pinned acquisition manifest and its retained measurement output."""

    def __init__(self, manifest: Path, measurements: Path) -> None:
        self.manifest, self.measurements = manifest.resolve(), measurements.resolve()

    def load(self, query: Mapping[str, JsonValue]) -> ObservationBatch:
        if query:
            raise ValueError("pinned pilot has no live query API; load with an empty query")
        manifest = json.loads(self.manifest.read_text())
        report = json.loads(self.measurements.read_text())
        if report["manifest_sha256"] != file_digest(self.manifest):
            raise ValueError("measurement output belongs to a different acquisition manifest")
        entities = [
            Entity(
                id="manifest",
                kind="acquisition_manifest",
                uri=self.manifest.as_uri(),
                sha256=file_digest(self.manifest),
            ),
            Entity(
                id="measurements-output",
                kind="conditional_photometry",
                uri=self.measurements.as_uri(),
                sha256=file_digest(self.measurements),
            ),
        ]
        records = []
        for asset in manifest["parents"]:
            entities.append(
                Entity(
                    id="parent:" + asset["id"],
                    kind="IRSA_raw_asset",
                    uri=asset["url"],
                    sha256=asset["sha256"],
                    provider="IRSA",
                    dataset_release="qr3",
                )
            )
        observations, measurements = [], []
        for exposure in manifest["exposures"]:
            frame = load_qr3_frame(self.manifest.parent / exposure["file"], exposure["sha256"])
            if frame.header["OBSID"] != exposure["obs_id"]:
                raise ValueError("manifest/exposure observation ID mismatch")
            cutout_id = "cutout:" + exposure["obs_id"]
            entities.append(
                Entity(
                    id=cutout_id,
                    kind="derived_registered_cutout",
                    uri=frame.path.resolve().as_uri(),
                    sha256=frame.sha256,
                )
            )
            raw = ("parent:" + exposure["parent_id"], "parent:solid_angle", "parent:spectral_wcs")
            records.append(_record("derive:" + exposure["obs_id"], (cutout_id,), raw, report))
            time = Time(frame.header["MJD-AVG"], format="mjd", scale="utc")
            oid = "observation:" + exposure["obs_id"]
            ref = "produce:" + exposure["obs_id"]
            observation = Observation(
                id=oid,
                source=SourceKey(
                    provider="IRSA",
                    dataset="SPHEREx-spectral-image",
                    release="qr3",
                    exposure_id=exposure["obs_id"],
                    source_id="D1",
                ),
                time=ObservationTime(jd1=float(time.jd1), jd2=float(time.jd2), scale="utc"),
                pointing=SkyPosition(
                    longitude_deg=float(frame.header["CRVAL1"]),
                    latitude_deg=float(frame.header["CRVAL2"]),
                    frame="icrs",
                ),
                dependence_group="exposure:" + exposure["obs_id"],
                bandpass_ref="parent:spectral_wcs",
                raw_asset_ids=(*raw, cutout_id),
                quality=Quality(flags=("QR3_D1_POINT_SOURCE_PILOT",)),
                provenance_ref=ref,
            )
            observations.append(observation)
            entities.append(_entity(observation))
            records.append(_record(ref, (oid,), (*raw, cutout_id), report))
            measured_exposure = next(
                (e for e in report["exposures"] if e["obs_id"] == exposure["obs_id"]), None
            )
            if measured_exposure is None:
                continue  # held-out exposure is a real observation, not a generation measurement
            if measured_exposure["source_sha256"] != frame.sha256:
                raise ValueError("photometry input SHA mismatch")
            for row in measured_exposure["measurements"]:
                mid = row["measurement_id"]
                measured = row["flux_uJy"] is not None
                quantities: tuple[ScalarMeasurement, ...] = (
                    ScalarMeasurement(
                        name="finite_epsf_flux",
                        quantity=QuantityValue(
                            value=row["flux_uJy"],
                            unit="uJy",
                            uncertainty=row["flux_error_uJy"],
                            missing_reason=None if measured else row["reason"],
                        ),
                    ),
                )
                if measured:
                    quantities += (
                        ScalarMeasurement(
                            name="central_wavelength",
                            quantity=QuantityValue(value=row["cwave_um"], unit="um"),
                        ),
                        ScalarMeasurement(
                            name="bandwidth",
                            quantity=QuantityValue(value=row["cband_um"], unit="um"),
                        ),
                    )
                position = frame.wcs.pixel_to_world(row["x_pixel"], row["y_pixel"])
                mref = "produce:" + mid
                measurement = Measurement(
                    id=mid,
                    observation_id=oid,
                    raw_source_id=row["raw_source_id"],
                    position=SkyPosition(
                        longitude_deg=float(position.ra.deg),
                        latitude_deg=float(position.dec.deg),
                        frame="icrs",
                    ),
                    covariance=None,
                    quantities=quantities,
                    method=METHOD,
                    quality=Quality(
                        usable=measured,
                        flags=("UNCERTAINTY_UNQUALIFIED", "ASTROMETRIC_COVARIANCE_MISSING"),
                        bitmask=row.get("flag_bits", 0),
                        flag_definition_ref="https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf",
                    ),
                    raw_output_asset_ids=("measurements-output",),
                    provenance_ref=mref,
                )
                measurements.append(measurement)
                entities.append(_entity(measurement))
                records.append(
                    _record(mref, (mid,), (oid, cutout_id, "measurements-output"), report)
                )
        measured_cutouts = tuple("cutout:" + e["obs_id"] for e in report["exposures"])
        records.append(
            _record(
                "produce:measurements-output",
                ("measurements-output",),
                (*measured_cutouts, "manifest"),
                report,
            )
        )
        provenance = ProvenanceGraph(
            entities=tuple(entities),
            agents=(
                Agent(id="sieve-qr3-pilot", name="SIEVE QR3 pilot", role="conditional_measurement"),
            ),
            records=tuple(records),
        )
        return ObservationBatch(
            observations=tuple(observations),
            measurements=tuple(measurements),
            provenance=provenance,
        )


class HeldoutProfileEvidenceProvider:
    """Expose retained diagnostics without promoting them to qualified evidence."""

    def __init__(self, profiles: Path, protocol: Path, graph: ProvenanceGraph) -> None:
        self.profiles = json.loads(profiles.read_text())
        self.protocol, self.graph = protocol, graph
        if self.profiles["protocol_sha256"] != file_digest(protocol):
            raise ValueError("profile output belongs to a different pilot protocol")

    def provenance(self) -> ProvenanceGraph:
        return self.graph

    def evaluate(
        self,
        hypothesis: ObjectHypothesis,
        request: EvidenceRequest,
        observations: Sequence[Observation],
        measurements: Sequence[Measurement],
    ) -> EvidenceResult:
        if (
            request.hypothesis_id != hypothesis.id
            or request.purpose != "TEST"
            or request.method != METHOD
            or request.tested_claim != TESTED_CLAIM
        ):
            raise ValueError("exact TEST hypothesis required")
        # Only this bounded UNKNOWN-dependence profile has been measured. A caller cannot
        # manufacture an independence declaration through the public provider interface.
        if request.dependence != "UNKNOWN":
            raise ValueError("shared calibration covariance has no qualified independence control")
        case_id = hypothesis.proposal_ids[0]
        profile = next((p for p in self.profiles["profiles"] if p["case_id"] == case_id), None)
        if profile is None:
            return EvidenceResult(
                request=request,
                status=ExecutionStatus.MISSING,
                reason="no retained profile for this hypothesis",
            )
        expected_inputs = {"cutout:" + self.profiles["heldout_obs_id"], "profiles-output"}
        if set(request.input_ids) != expected_inputs:
            raise ValueError("request does not declare the exact retained held-out inputs")
        if set(hypothesis.member_ids) != set(profile["member_ids"]):
            raise ValueError("hypothesis does not use the measured generation members")
        if not set(hypothesis.member_ids) <= {m.id for m in measurements}:
            raise ValueError("generation measurements are absent")
        if not any(o.source.exposure_id == self.profiles["heldout_obs_id"] for o in observations):
            raise ValueError("held-out observation is absent")
        quantities: tuple[ScalarMeasurement, ...] = ()
        if profile["delta_bic_two_minus_one"] is not None:
            quantities = (
                ScalarMeasurement(
                    name="delta_bic_two_minus_one",
                    quantity=QuantityValue(value=profile["delta_bic_two_minus_one"], unit="1"),
                ),
            )
        factor = EvidenceFactor(
            id="factor:" + hypothesis.id,
            request_id=request.id,
            hypothesis_id=hypothesis.id,
            method=request.method,
            purpose="TEST",
            input_ids=request.input_ids,
            tested_claim=request.tested_claim,
            verdict=EvidenceVerdict.INSUFFICIENT,
            quantities=quantities,
            test_protocol_ref="pilot-protocol",
            provenance_ref="test:" + hypothesis.id,
            rationale=profile["reason"],
            limitations=tuple(profile["limitations"]),
        )
        return EvidenceResult(
            request=request,
            status=ExecutionStatus.SUCCESS,
            factors=(factor,),
            reason="conditional diagnostics; scientific identity gate closed",
            raw_output_asset_ids=("profiles-output",),
        )


def build_competing_input(
    batch: ObservationBatch, profiles: Path, protocol: Path, report: dict[str, Any]
) -> IdentityInput:
    """Retain one/multiple/artifact/unresolved alternatives for each real proximity pair."""
    payload = json.loads(profiles.read_text())
    entities, records = list(batch.provenance.entities), list(batch.provenance.records)
    entities.extend(
        [
            Entity(
                id="profiles-output",
                kind="heldout_profile_diagnostics",
                uri=profiles.resolve().as_uri(),
                sha256=file_digest(profiles),
            ),
            Entity(
                id="pilot-protocol",
                kind="frozen_pilot_protocol",
                uri=protocol.resolve().as_uri(),
                sha256=file_digest(protocol),
            ),
        ]
    )
    proposals, hypotheses, results = [], [], []
    records.append(
        _record(
            "produce:profiles-output",
            ("profiles-output",),
            ("measurements-output", "cutout:" + payload["heldout_obs_id"], "pilot-protocol"),
            report,
        )
    )
    provider = HeldoutProfileEvidenceProvider(profiles, protocol, batch.provenance)
    for profile in payload["profiles"]:
        pid = profile["case_id"]
        members = tuple(profile["member_ids"])
        pref = "produce:" + pid
        proposal = CandidateProposal(
            id=pid,
            member_ids=members,
            method=METHOD,
            generation_input_ids=members,
            raw_output_asset_ids=("measurements-output",),
            rationale="Cross-exposure pair within 12 arcsec; proximity is not identity",
            provenance_ref=pref,
        )
        proposals.append(proposal)
        entities.append(_entity(proposal))
        records.append(_record(pref, (pid,), (*members, "measurements-output"), report))
        for name, explanation in (
            ("one", Explanation.ONE_OBJECT),
            ("multiple", Explanation.MULTIPLE_OBJECTS),
            ("artifact", Explanation.ARTIFACT),
            ("unresolved", Explanation.UNRESOLVED),
        ):
            hid = "hypothesis:" + pid + ":" + name
            href = "produce:" + hid
            assignments = (
                (ObjectAssignment(slot_id="conditional:one", member_ids=members),)
                if name == "one"
                else tuple(
                    ObjectAssignment(slot_id=f"conditional:{i}", member_ids=(member,))
                    for i, member in enumerate(members)
                )
                if name == "multiple"
                else ()
            )
            hypothesis = ObjectHypothesis(
                id=hid,
                proposal_ids=(pid,),
                member_ids=members,
                explanation=explanation,
                assignments=assignments,
                unassigned_member_ids=() if assignments else members,
                generation_input_ids=members,
                rationale="Conditional alternative; no external object identity assigned",
                provenance_ref=href,
            )
            hypotheses.append(hypothesis)
            entities.append(_entity(hypothesis))
            records.append(_record(href, (hid,), (pid, *members), report))
            if name == "unresolved":
                continue
            request = EvidenceRequest(
                id="request:" + hid,
                hypothesis_id=hid,
                method=METHOD,
                purpose="TEST",
                input_ids=("cutout:" + payload["heldout_obs_id"], "profiles-output"),
                tested_claim=TESTED_CLAIM,
                dependence="UNKNOWN",
                provenance_ref="request-produce:" + hid,
            )
            result = provider.evaluate(hypothesis, request, batch.observations, batch.measurements)
            results.append(result)
            entities.append(_entity(request))
            records.append(
                _record(request.provenance_ref, (request.id,), (hid, *request.input_ids), report)
            )
            for factor in result.factors:
                entities.append(_entity(factor))
                records.append(
                    _record(
                        factor.provenance_ref,
                        (factor.id,),
                        (request.id, hid, *request.input_ids, "pilot-protocol"),
                        report,
                    )
                )
    return IdentityInput(
        observations=batch.observations,
        measurements=batch.measurements,
        proposals=tuple(proposals),
        hypotheses=tuple(hypotheses),
        evidence_results=tuple(results),
        provenance=ProvenanceGraph(
            entities=tuple(entities), agents=batch.provenance.agents, records=tuple(records)
        ),
    )
