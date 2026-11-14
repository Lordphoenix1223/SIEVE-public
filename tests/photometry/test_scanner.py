from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sieve.photometry import scan_fits, write_scan

ROOT = Path(__file__).resolve().parents[2]
REAL_SAMPLE = ROOT / "examples/photometry/qr3-d1-cutout.fits"
REAL_SAMPLE_SHA256 = "9e905964b48a0f63f93c87a3ec00a7b472ba79d00aa8e08af8c5372298a8dd8a"


def write_mef(path: Path, *, shape_mismatch: bool = False) -> None:
    rng = np.random.default_rng(414)
    image = rng.normal(0.0, 0.2, (128, 128))
    yy, xx = np.indices(image.shape)
    image += 5.0 * np.exp(-((xx - 57.2) ** 2 + (yy - 62.7) ** 2) / 2.0)
    wcs = fits.Header(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 64.5,
            "CRPIX2": 64.5,
            "CRVAL1": 150.0,
            "CRVAL2": 2.0,
            "CDELT1": -0.0001,
            "CDELT2": 0.0001,
            "BUNIT": "adu",
            "OBSID": "test-exp-1",
            "MJD-OBS": 60000.0,
            "DATE-OBS": "2023-02-25",
            "DETECTOR": 1,
        }
    )
    flags = np.zeros(image.shape, dtype=np.int32)
    flags[10, 10] = 1 << 6
    variance = np.full(image.shape, 0.04, dtype=np.float32)
    if shape_mismatch:
        flags = flags[:-1]
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image.astype(np.float32), header=wcs, name="IMAGE"),
            fits.ImageHDU(flags, name="FLAGS"),
            fits.ImageHDU(variance, name="VARIANCE"),
        ]
    ).writeto(path)


def test_real_qr3_sample_produces_measured_sky_positions() -> None:
    result = scan_fits(REAL_SAMPLE, release="QR3", bandpass="D1", background_box_pixels=32)

    assert result.source_sha256 == REAL_SAMPLE_SHA256
    assert result.exposure_id == "2026W30_1B_0001_1"
    assert result.release == "QR3"
    assert result.bandpass == "D1"
    assert result.detector == 1
    assert result.observation_mjd == pytest.approx(61241.004784375)
    assert result.data_unit == "MJy / sr"
    assert result.detected_peak_count == 418
    assert result.known_source_flagged_peak_count == 418
    assert result.unflagged_peak_count == 0
    assert result.known_source_flagged_measurement_count == 409
    assert result.rejected_truncated_aperture_count == 9
    assert len(result.detections) == 409
    assert sum(not row["aperture_usable"] for row in result.detections) == 136
    first = result.detections[0]
    assert first["ra_deg"] == pytest.approx(232.7528113249, abs=1e-8)
    assert first["dec_deg"] == pytest.approx(-10.5248158114, abs=1e-8)
    assert first["aperture_sum"] == pytest.approx(0.6500992189, abs=1e-7)
    assert first["centroid_flag_bits"] == 1 << 21  # known-source mask bit is retained


def test_synthetic_source_gets_wcs_and_variance_measurements(tmp_path: Path) -> None:
    source = tmp_path / "one-source.fits"
    write_mef(source)

    result = scan_fits(source, background_box_pixels=32)

    assert result.detected_peak_count >= 1
    measured = min(result.detections, key=lambda row: abs(float(row["x_pixel"]) - 57.2))
    assert measured["x_pixel"] == pytest.approx(57.2, abs=0.5)
    assert measured["y_pixel"] == pytest.approx(62.7, abs=0.5)
    assert 149.99 < measured["ra_deg"] < 150.01
    assert 1.0 < measured["aperture_snr_from_variance"] < 1000.0
    assert result.exposure_id == "test-exp-1"
    assert len(result.source_tree_sha256) == 64
    assert measured["aperture_usable"]


def test_shape_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "bad-shapes.fits"
    write_mef(source, shape_mismatch=True)

    with pytest.raises(ValueError, match="shapes must match"):
        scan_fits(source, background_box_pixels=32)


def test_json_export_is_lossless_finite_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "one-source.fits"
    output = tmp_path / "scan.json"
    write_mef(source)
    result = scan_fits(source, background_box_pixels=32)
    write_scan(result, output)

    stored = json.loads(output.read_text())
    assert stored["schema_version"] == "sieve.photometry.scan.v0"
    assert stored["detections"][0]["ra_deg"] == result.detections[0]["ra_deg"]
    with pytest.raises(ValueError, match="cannot be overwritten"):
        write_scan(result, output)


def test_masked_nonfinite_image_and_negative_variance_do_not_poison_scan(tmp_path: Path) -> None:
    source = tmp_path / "masked.fits"
    write_mef(source)
    with fits.open(source, mode="update") as hdul:
        hdul["IMAGE"].data[10, 11] = np.nan
        hdul["VARIANCE"].data[10, 12] = -1.0
    result = scan_fits(source, background_box_pixels=32)
    assert len(result.detections) >= 1
    assert result.masked_fraction >= 3 / (128 * 128)


def test_blank_plane_retains_empty_detection_result(tmp_path: Path) -> None:
    source = tmp_path / "blank.fits"
    write_mef(source)
    with fits.open(source, mode="update") as hdul:
        hdul["IMAGE"].data[:] = 0
    result = scan_fits(source, background_box_pixels=32)
    assert result.detected_peak_count == 0
    assert result.detections == ()
