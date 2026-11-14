"""Frontend markers retain measured coordinates and conservative status."""

import json
from pathlib import Path

import pytest
from astropy.io import fits
from astropy.wcs import WCS
from pydantic import ValidationError

from examples.photometry.export_frontend import Evidence, export
from sieve.photometry.scanner import PhotometryScan


def test_real_scan_markers_match_wcs() -> None:
    root = Path(__file__).resolve().parents[2]
    scan = PhotometryScan.model_validate_json(
        (root / "examples/photometry/expected-scan.json").read_text()
    )
    evidence = export(scan)
    assert evidence.scientific_status == "MEASUREMENTS_ONLY_NOT_IDENTITY"
    assert len(evidence.markers) == len(scan.detections)
    assert evidence.source_sha256 == scan.source_sha256
    with fits.open(root / "examples/photometry/qr3-d1-cutout.fits") as hdul:
        wcs = WCS(hdul["IMAGE"].header, fobj=hdul).celestial
        for marker, row in zip(evidence.markers, scan.detections, strict=True):
            ra, dec = wcs.pixel_to_world_values(row["x_pixel"], row["y_pixel"])
            assert marker.ra_deg == pytest.approx(ra, abs=1e-9)
            assert marker.dec_deg == pytest.approx(dec, abs=1e-9)
    payload = json.loads(evidence.model_dump_json())
    payload["markers"][0]["dec_deg"] = 91
    with pytest.raises(ValidationError):
        Evidence.model_validate(payload)
