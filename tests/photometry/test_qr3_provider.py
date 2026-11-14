"""Real-asset contract integration; these tests are not identity-performance evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sieve.identity.graph import IdentityBudget, identity_funnel
from sieve.identity.models import EvidenceVerdict
from sieve.photometry.profiles import compare_profiles
from sieve.photometry.provider import (
    HeldoutProfileEvidenceProvider,
    QR3ObservationProvider,
    build_competing_input,
)
from sieve.photometry.qualified import file_digest, fit_flux, load_qr3_frame

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qualification"


@pytest.fixture
def integrated(tmp_path):
    manifest = json.loads((EXAMPLE / "source.json").read_text())
    frames = [load_qr3_frame(EXAMPLE / e["file"], e["sha256"]) for e in manifest["exposures"]]
    points = [
        (232.8219284670584, -10.644165580713109),
        (232.8219284670584 + 0.0001, -10.644165580713109),
    ]
    exposures = []
    members = []
    for i, (frame, point) in enumerate(zip(frames[:2], points, strict=True)):
        x, y = frame.wcs.world_to_pixel_values(*point)
        row = fit_flux(frame, frame.image, float(x), float(y))
        row.update(measurement_id=f"measurement:{i}", raw_source_id=f"position:{i}")
        members.append(row["measurement_id"])
        exposures.append(
            {"obs_id": frame.header["OBSID"], "source_sha256": frame.sha256, "measurements": [row]}
        )
    now = datetime.now(UTC).isoformat()
    report = {
        "manifest_sha256": file_digest(EXAMPLE / "source.json"),
        "exposures": exposures,
        "code_commit": "CONTRACT_TEST",
        "environment_id": "CONTRACT_TEST",
        "started_at": now,
        "ended_at": now,
    }
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps(report))
    profile = compare_profiles(frames[2], points)
    profile.update(case_id="proposal:test", member_ids=members)
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "profiles": [profile],
                "heldout_obs_id": frames[2].header["OBSID"],
                "protocol_sha256": file_digest(EXAMPLE / "protocol.json"),
            }
        )
    )
    provider = QR3ObservationProvider(EXAMPLE / "source.json", measurements)
    batch = provider.load({})
    bundle = build_competing_input(batch, profiles, EXAMPLE / "protocol.json", report)
    return provider, batch, bundle, profiles


def test_real_records_keep_times_covariance_and_all_alternatives(integrated):
    _, batch, bundle, _ = integrated
    assert len(batch.observations) == 3
    assert len({o.dependence_group for o in batch.observations}) == 3
    assert all(o.time.scale == "utc" for o in batch.observations)
    assert all(m.covariance is None for m in batch.measurements)
    assert len(bundle.hypotheses) == 4
    assert all(
        f.verdict == EvidenceVerdict.INSUFFICIENT
        for r in bundle.evidence_results
        for f in r.factors
    )
    funnel = identity_funnel(bundle, IdentityBudget())
    assert all("CONSISTENT" not in a.disposition for a in funnel.assessments)
    # The held-out exposure has no generation measurements.
    assert {m.observation_id for m in batch.measurements} == {o.id for o in batch.observations[:2]}


def test_provider_refuses_manufactured_independence_and_stale_protocol(integrated, tmp_path):
    _, batch, bundle, profiles = integrated
    provider = HeldoutProfileEvidenceProvider(
        profiles, EXAMPLE / "protocol.json", bundle.provenance
    )
    hypothesis = bundle.hypotheses[0]
    request = bundle.evidence_results[0].request.model_copy(
        update={"dependence": "DISJOINT_INPUTS"}
    )
    with pytest.raises(ValueError, match="independence control"):
        provider.evaluate(hypothesis, request, batch.observations, batch.measurements)
    stale = tmp_path / "protocol.json"
    stale.write_text("{}")
    with pytest.raises(ValueError, match="different pilot protocol"):
        HeldoutProfileEvidenceProvider(profiles, stale, bundle.provenance)


def test_provider_refuses_live_query_and_stale_measurement_manifest(integrated):
    provider, _, _, _ = integrated
    with pytest.raises(ValueError, match="no live query"):
        provider.load({"region": "elsewhere"})
    report = json.loads(provider.measurements.read_text())
    report["manifest_sha256"] = "0" * 64
    provider.measurements.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different acquisition manifest"):
        provider.load({})
