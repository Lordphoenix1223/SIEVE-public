"""Scientific example retains real input lineage and explicitly bounded outcomes."""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from examples.source_investigation.build_evidence import Package

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/source_investigation"


def test_real_source_record_matches_fits_wcs_and_defers_identity() -> None:
    package = Package.model_validate_json((EXAMPLE / "evidence.json").read_text())
    case = package.cases[0]
    assert case.status == "TENTATIVE_ASSOCIATION"
    assert case.observation["position_uncertainty_arcsec"] is None
    assert case.checks[0]["verdict"] == "INCONCLUSIVE"
    assert len(case.catalog_candidates) == 1
    assert case.catalog_candidates[0]["separation_arcsec"] == pytest.approx(0.5913, abs=0.001)
    row = case.measurements[0]
    with fits.open(ROOT / "examples/qualification" / case.observation["input_file"]) as hdul:
        ra, dec = WCS(hdul["IMAGE"].header).celestial.pixel_to_world_values(
            row["x_pixel"], row["y_pixel"]
        )
    assert case.ra_deg == pytest.approx(ra, abs=1e-9)
    assert case.dec_deg == pytest.approx(dec, abs=1e-9)
    assert row["masked_pixels"] == 0
    assert 30 < row["x_pixel"] < 225 and 30 < row["y_pixel"] < 225


def test_artifact_holdout_has_disjoint_sites_and_is_not_a_production_verdict() -> None:
    payload = json.loads((EXAMPLE / "artifact-results.json").read_text())
    pos = np.asarray(payload["site_positions"])
    assert pos.shape == (64, 2)
    assert np.min(np.linalg.norm(pos[:32, None] - pos[None, 32:], axis=2)) > 12
    heldout = next(
        r
        for r in payload["results"]
        if r["split"] == "heldout" and r["method"] == "DAO" and r["arm"] == "pixel_blank"
    )
    assert heldout["detected"] == 32 and heldout["above_threshold"] == 32
    package = Package.model_validate_json((EXAMPLE / "evidence.json").read_text())
    assert package.cases[1].kind == "controlled_injection"
    assert package.cases[1].status == "ARTIFACT_UNRESOLVED"
