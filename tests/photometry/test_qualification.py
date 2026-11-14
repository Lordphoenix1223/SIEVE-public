from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from evaluation.photometry_recovery import match_positions
from sieve.photometry.profiles import compare_profiles
from sieve.photometry.qualified import (
    SOURCE_BIT,
    detect_positions,
    file_digest,
    fit_flux,
    fit_flux_group,
    load_qr3_frame,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qualification"


@pytest.fixture(scope="module")
def frame():
    exposure = json.loads((EXAMPLE / "source.json").read_text())["exposures"][0]
    return load_qr3_frame(EXAMPLE / exposure["file"], exposure["sha256"])


@pytest.mark.parametrize("phase", [0.0, 0.25, 0.5, 0.9])
def test_finite_psf_flux_and_local_background_normalization(frame, phase):
    frame = replace(frame, bad=np.zeros_like(frame.bad))
    x, y = 100 + phase, 140 + phase
    psf, slices, _ = frame.stamp(x, y)
    assert abs(psf.sum() - 1) < 1e-14
    data = np.full_like(frame.image, 120.0)
    data[slices] += 1375.0 * psf
    row = fit_flux(frame, data, x, y)
    assert row["flux_uJy"] == pytest.approx(1375, rel=1e-12)
    assert row["background_uJy_per_pixel"] == pytest.approx(120, rel=1e-12)
    assert row["flux_background_covariance"] < 0
    assert row["flux_error_uJy"] > 0


def test_joint_fixed_position_photometry_recovers_blend_with_sloped_sky(frame):
    frame = replace(frame, bad=np.zeros_like(frame.bad))
    positions = [(120.2, 120.4), (121.7, 120.5)]
    fluxes = [1200.0, 700.0]
    yy, xx = np.indices(frame.image.shape)
    data = 100 + 0.7 * (xx - 120) - 0.4 * (yy - 120)
    for (x, y), flux in zip(positions, fluxes, strict=True):
        psf, slices, _ = frame.stamp(x, y)
        data[slices] += flux * psf

    joint = fit_flux_group(frame, data, positions, background_order=1)
    independent = [fit_flux(frame, data, *position) for position in positions]

    assert joint["status"] == "MEASURED_CONDITIONAL"
    np.testing.assert_allclose([row["flux_uJy"] for row in joint["sources"]], fluxes, rtol=1e-9)
    assert joint["amplitude_covariance_uJy2"][0][1] < 0
    assert any(
        abs(row["flux_uJy"] - truth) > 0.05 for row, truth in zip(independent, fluxes, strict=True)
    )


def test_joint_photometry_retains_singular_blend_failure(frame):
    frame = replace(frame, bad=np.zeros_like(frame.bad))
    position = (120.2, 120.4)
    row = fit_flux_group(frame, np.zeros_like(frame.image), [position, position])
    assert row["status"] == "FAILED"
    assert row["reason"] == "RANK_DEFICIENT_OR_ILL_CONDITIONED"


def test_formal_errors_cover_independent_archive_noise_under_exact_model(frame):
    frame = replace(frame, bad=np.zeros_like(frame.bad))
    rng = np.random.default_rng(127)
    psf, slices, _ = frame.stamp(120.3, 120.7)
    standardized = []
    for _ in range(40):
        data = rng.normal(size=frame.image.shape) * np.sqrt(frame.variance) + 120
        data[slices] += 4000 * psf
        row = fit_flux(frame, data, 120.3, 120.7)
        standardized.append((row["flux_uJy"] - 4000) / row["flux_error_uJy"])
    assert 0.65 < np.std(standardized) < 1.35
    assert abs(np.mean(standardized)) < 0.4


def test_source_flag_does_not_mask_photometry(frame):
    flags = np.full_like(frame.flags, SOURCE_BIT)
    altered = replace(frame, flags=flags, bad=np.zeros_like(frame.bad))
    data = np.zeros_like(frame.image)
    psf, slices, _ = altered.stamp(120.2, 120.3)
    data[slices] = 1400 * psf
    row = fit_flux(altered, data, 120.2, 120.3)
    assert row["source_flag_at_centroid"]
    assert row["masked_pixels"] == 0
    assert row["flux_uJy"] == pytest.approx(1400)


def test_bad_core_and_partial_edge_profile_keep_explicit_coverage(frame):
    bad = frame.bad.copy()
    bad[120, 120] = True
    altered = replace(frame, bad=bad)
    row = fit_flux(altered, altered.image, 120.0, 120.0)
    assert row["status"] == "FAILED"
    assert row["flux_uJy"] is None
    assert row["reason"] == "INSUFFICIENT_PSF_COVERAGE"
    edge = 0.2, 100.4
    psf, slices, _ = frame.stamp(*edge)
    injected = frame.image.copy()
    injected[slices] = 1375.0 * psf + 120.0
    edge_fit = fit_flux(frame, injected, *edge)
    assert edge_fit["status"] == "MEASURED_CONDITIONAL"
    assert edge_fit["flux_uJy"] == pytest.approx(1375.0, rel=1e-10)
    assert edge_fit["psf_edge_information_coverage"] < 1.0
    assert 0.8 <= edge_fit["psf_information_coverage"] < 1.0
    assert np.isfinite(edge_fit["flux_error_uJy"]) and edge_fit["flux_error_uJy"] > 0

    corner = fit_flux(frame, frame.image, 1, 1)
    assert corner["status"] == "MEASURED_CONDITIONAL"
    assert corner["psf_information_coverage"] >= 0.8


def test_source_mask_is_only_used_in_background_for_detector(frame):
    # Same image/variance/bad mask; toggling SOURCE changes sky sampling, not the input peak veto.
    data = np.zeros_like(frame.image)
    psf, slices, _ = frame.stamp(120.2, 120.3)
    data[slices] = 100000 * psf
    flags = np.zeros_like(frame.flags)
    flags[slices] = SOURCE_BIT
    altered = replace(frame, flags=flags, bad=np.zeros_like(frame.bad))
    peaks = detect_positions(altered, data, "DAO")
    assert np.min(np.hypot(peaks[:, 0] - 120.2, peaks[:, 1] - 120.3)) < 0.3


def test_blend_recovery_is_one_to_one_and_not_remote(frame):
    truth = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    detections = np.array([[0.5, 0.0], [50.0, 0.0]])
    matched, unmatched = match_positions(truth, detections)
    assert sum(m["recovered"] for m in matched) == 1
    assert not matched[2]["recovered"]
    assert unmatched == [1]


def test_hash_and_detector_calibration_refuse(frame, tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        load_qr3_frame(frame.path, "0" * 64)
    output = tmp_path / "wrong-detector.fits"
    with fits.open(frame.path) as hdul:
        hdul["SOLIDANGLE"].header["DETECTOR"] = 4
        hdul.writeto(output)
    with pytest.raises(ValueError, match="detector mismatch"):
        load_qr3_frame(output, file_digest(output))


def test_profile_diagnostic_never_promotes_object_identity(frame):
    points = [frame.wcs.pixel_to_world_values(x, 120.0) for x in (119.5, 121.0)]
    result = compare_profiles(frame, [(float(ra), float(dec)) for ra, dec in points])
    assert result["status"] == "INCONCLUSIVE"
    assert np.isfinite(result["delta_bic_two_minus_one"])
    assert set(result["models"]) == {"one", "multiple", "constant_sky_null"}
    assert all(m["status"] == "DIAGNOSTIC_ONLY" for m in result["models"].values())


def test_all_science_planes_share_sky_wcs_and_detector_origin(frame):
    from astropy.wcs import WCS

    with fits.open(frame.path) as hdul:
        origin = (frame.header["XORIGIN"], frame.header["YORIGIN"])
        reference = frame.wcs.pixel_to_world_values([0.0, 255.0], [0.0, 255.0])
        assert "PSF_FWHM" in hdul["IMAGE"].header
        assert "VERSION" in hdul[0].header
        for name in ("FLAGS", "VARIANCE", "ZODI", "SOLIDANGLE", "CWAVE", "CBAND"):
            header = hdul[name].header
            assert (header["XORIGIN"], header["YORIGIN"]) == origin
            assert "WCSAXESW" not in header
            np.testing.assert_allclose(
                WCS(header).celestial.pixel_to_world_values([0.0, 255.0], [0.0, 255.0]),
                reference,
                rtol=0,
                atol=1e-10,
            )
