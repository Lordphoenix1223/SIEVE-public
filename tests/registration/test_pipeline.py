"""Exercise fail-closed comparison through the real CLI entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
EXPOSURE = ROOT / "examples/qualification/qr3-d1-exposure-1.fits"


def invoke(first: Path, second: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sieve.cli",
            "compare-qr3",
            "--first",
            str(first),
            "--second",
            str(second),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_duplicate_exposure_refuses_claims_and_pairing(tmp_path):
    out = tmp_path / "result.json"
    p = invoke(EXPOSURE, EXPOSURE, out)
    assert p.returncode == 0, p.stderr
    r = json.loads(out.read_text())
    assert r["source_association"] == "REFUSED"
    assert r["motion_claim"] == "REFUSED" and r["flux_variability_claim"] == "REFUSED"
    assert r["geometry"]["temporal_status"] == "DUPLICATE_EXPOSURE"
    assert not r["pairings"]["candidate_pairs"]


def test_invalid_wcs_refused_by_cli(tmp_path):
    invalid = tmp_path / "invalid.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        del copied["IMAGE"].header["CTYPE1"]
        copied.writeto(invalid)
    out = tmp_path / "out.json"
    p = invoke(invalid, EXPOSURE, out)
    assert p.returncode != 0 and not out.exists()
    assert "INVALID_WCS" in p.stderr


def test_no_overlap_refuses_pairing(tmp_path):
    other = tmp_path / "elsewhere.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        copied["IMAGE"].header["CRVAL1"] += 20
        copied["IMAGE"].header["OBSID"] = "different-exposure"
        copied.writeto(other)
    out = tmp_path / "out.json"
    p = invoke(EXPOSURE, other, out)
    assert p.returncode == 0, p.stderr
    r = json.loads(out.read_text())
    assert r["geometry"]["spatial_status"] == "NO_SAMPLED_COMMON_COVERAGE"
    assert r["source_association"] == "REFUSED" and r["motion_claim"] == "REFUSED"


def test_missing_time_scale_refuses_validity_claims(tmp_path):
    other = tmp_path / "missing-time.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        del copied["IMAGE"].header["TIMESYS"]
        copied["IMAGE"].header["OBSID"] = "different-exposure"
        copied.writeto(other)
    out = tmp_path / "out.json"
    p = invoke(EXPOSURE, other, out)
    assert p.returncode != 0 and not out.exists()


def test_incompatible_nominal_band_never_yields_variability_claim(tmp_path):
    other = tmp_path / "different-band.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        copied["IMAGE"].header["OBSID"] = "different-exposure"
        copied["IMAGE"].header["MJD-AVG"] += 0.01
        copied["CWAVE"].data[:] = 5.0
        copied.writeto(other)
    out = tmp_path / "out.json"
    p = invoke(EXPOSURE, other, out)
    assert p.returncode == 0, p.stderr
    result = json.loads(out.read_text())
    assert result["geometry"]["spectral_status"] == "POSSIBLE_BANDPASS_MISMATCH"
    assert result["flux_variability_claim"] == "REFUSED"
    assert result["motion_claim"] == "REFUSED"


def test_all_bad_pixels_refuse_analysis(tmp_path):
    other = tmp_path / "all-bad.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        copied["IMAGE"].header["OBSID"] = "different-exposure"
        copied["IMAGE"].header["MJD-AVG"] += 0.01
        copied["FLAGS"].data[:] = 1 << 6
        copied.writeto(other)
    out = tmp_path / "out.json"
    p = invoke(EXPOSURE, other, out)
    assert p.returncode != 0 and not out.exists()


def test_malformed_midpoint_is_refused(tmp_path):
    other = tmp_path / "bad-time.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        copied["IMAGE"].header["OBSID"] = "other-exposure"
        copied["IMAGE"].header["MJD-AVG"] = "not-a-time"
        copied.writeto(other)
    out = tmp_path / "out.json"
    p = invoke(EXPOSURE, other, out)
    assert p.returncode != 0 and not out.exists()


def test_consistent_but_wrong_wcs_never_yields_scientific_claim(tmp_path):
    other = tmp_path / "shifted.fits"
    with fits.open(EXPOSURE) as original:
        copied = fits.HDUList([x.copy() for x in original])
        for name in ("IMAGE", "FLAGS", "VARIANCE", "SOLIDANGLE", "CWAVE", "CBAND", "ZODI"):
            copied[name].header["CRVAL1"] += 0.005
        copied["IMAGE"].header["OBSID"] = "other-exposure"
        copied["IMAGE"].header["MJD-AVG"] += 0.01
        copied.writeto(other)
    out = tmp_path / "out.json"
    p = invoke(EXPOSURE, other, out)
    if p.returncode == 0:
        result = json.loads(out.read_text())
        assert result["motion_claim"] == "REFUSED"
        assert result["flux_variability_claim"] == "REFUSED"
        assert result["geometry"]["spatial_status"] == "REGISTRATION_UNQUALIFIED"
    else:
        assert not out.exists()
