"""Reproduce three bounded QR3 learning examples from pinned pixels and Gaia DR3."""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from photutils.centroids import centroid_com
from pydantic import BaseModel, ConfigDict, Field

from sieve.photometry.qualified import detect_positions, fit_flux, load_qr3_frame

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "examples/qualification"


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    kind: Literal["real_observation", "controlled_injection"]
    status: Literal["TENTATIVE_ASSOCIATION", "ARTIFACT_UNRESOLVED", "INCONCLUSIVE"]
    ra_deg: float = Field(ge=0, lt=360)
    dec_deg: float = Field(ge=-90, le=90)
    observation: dict[str, Any]
    measurements: list[dict[str, Any]]
    catalog_candidates: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    reasons: list[str]


class Package(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["sieve.learning_evidence.v1"] = "sieve.learning_evidence.v1"
    catalog: dict[str, Any]
    cases: list[Case] = Field(min_length=3, max_length=3)


def main(
    output_dir: Path = HERE,
    catalog_path: Path | None = None,
    catalog_mode: str = "retained_snapshot",
) -> None:
    catalog_path = catalog_path or HERE / "gaia-dr3-cone.csv"
    manifest = json.loads((BASE / "source.json").read_text())
    exposures = manifest["exposures"][:2]
    frames = [load_qr3_frame(BASE / e["file"], e["sha256"]) for e in exposures]
    # Frozen location selects a previously measured interior peak; detection remains blind.
    target = SkyCoord(232.64278847144752 * u.deg, -10.546006460187737 * u.deg)
    measurements = []
    independent = []
    for frame, exposure in zip(frames, exposures, strict=True):
        points = detect_positions(frame, frame.image, "DAO")
        ra, dec = frame.wcs.pixel_to_world_values(points[:, 0], points[:, 1])
        distances = target.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec
        i = int(np.argmin(distances))
        if distances[i] > 2:
            raise ValueError("Frozen selected field no longer has a peak within 2 arcsec")
        x, y = map(float, points[i])
        fit = fit_flux(frame, frame.image, x, y)
        if fit["status"] != "MEASURED_CONDITIONAL":
            raise ValueError("Selected source measurement failed")
        fit.update(
            measurement_id=f"measurement:{exposure['obs_id']}:{i:05d}",
            obs_id=exposure["obs_id"],
            mjd_avg_utc=exposure["mjd_avg"],
        )
        measurements.append(fit)
        ix, iy = round(x), round(y)
        sl = (slice(iy - 5, iy + 6), slice(ix - 5, ix + 6))
        image, bad = frame.image[sl], frame.bad[sl]
        # Median of border pixels; independent intensity-moment centroid, not a PSF fit.
        border = np.ones(image.shape, dtype=bool)
        border[1:-1, 1:-1] = False
        sky = float(np.median(image[border & ~bad]))
        cx, cy = centroid_com(np.maximum(image - sky, 0), mask=bad)
        cra, cdec = frame.wcs.pixel_to_world_values(float(cx) + ix - 5, float(cy) + iy - 5)
        independent.append(
            {
                "obs_id": exposure["obs_id"],
                "ra_deg": float(cra),
                "dec_deg": float(cdec),
                "background_uJy_per_pixel": sky,
                "window_pixels": 11,
            }
        )
    c1 = SkyCoord(independent[0]["ra_deg"] * u.deg, independent[0]["dec_deg"] * u.deg)
    c2 = SkyCoord(independent[1]["ra_deg"] * u.deg, independent[1]["dec_deg"] * u.deg)
    measured = SkyCoord(measurements[0]["ra_deg"] * u.deg, measurements[0]["dec_deg"] * u.deg)
    with catalog_path.open(newline="") as inp:
        raw = list(csv.DictReader(inp))
    candidates = []
    for row in raw:
        cat = SkyCoord(
            ra=float(row["ra"]) * u.deg,
            dec=float(row["dec"]) * u.deg,
            pm_ra_cosdec=float(row["pmra"]) * u.mas / u.yr,
            pm_dec=float(row["pmdec"]) * u.mas / u.yr,
            obstime=Time(float(row["ref_epoch"]), format="jyear"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # ERFA assumes a distance when no parallax/RV is provided.
            propagated = cat.apply_space_motion(
                new_obstime=Time(exposures[0]["mjd_avg"], format="mjd", scale="utc")
            )
        candidates.append(
            {
                "catalog": "Gaia DR3",
                "source_id": row["source_id"],
                "catalog_ra_deg": float(row["ra"]),
                "catalog_dec_deg": float(row["dec"]),
                "ref_epoch_jyear": float(row["ref_epoch"]),
                "pmra_mas_per_year": float(row["pmra"]),
                "pmdec_mas_per_year": float(row["pmdec"]),
                "ra_error_mas": float(row["ra_error"]),
                "dec_error_mas": float(row["dec_error"]),
                "ra_dec_correlation": float(row["ra_dec_corr"]),
                "phot_g_mean_mag": float(row["phot_g_mean_mag"]),
                "ruwe": float(row["ruwe"]),
                "pmra_error_mas_per_year": float(row["pmra_error"]),
                "pmdec_error_mas_per_year": float(row["pmdec_error"]),
                "propagated_ra_deg": float(propagated.ra.deg),
                "propagated_dec_deg": float(propagated.dec.deg),
                "separation_arcsec": float(measured.separation(propagated).arcsec),
                "warning": "No distance or radial velocity; covariance propagation and SPHEREx positional errors unavailable.",  # noqa: E501
            }
        )
    check = {
        "verdict": "INCONCLUSIVE",
        "question": "Are two detections astrometrically consistent with a persistent source?",
        "method": "Photutils centroid_com on nonnegative background-subtracted 11x11 pixels with border-median background",  # noqa: E501
        "input_sha256": [x["sha256"] for x in exposures],
        "centroids": independent,
        "separation_arcsec": float(c1.separation(c2).arcsec),
        "uncertainty_arcsec": None,
        "reasons": [
            "No calibrated centroid or registration covariance; shared instrumental calibration.",
            "Distinct wavelengths and structured backgrounds can change centroids.",
            "Clipping negative pixels biases moment centroids; method is exploratory.",
        ],
    }
    parent = manifest["parents"][0]
    obs = {
        "release": "QR3",
        "obs_id": exposures[0]["obs_id"],
        "detector": 1,
        "mjd_avg_utc": exposures[0]["mjd_avg"],
        "input_file": exposures[0]["file"],
        "input_sha256": exposures[0]["sha256"],
        "parent_url": parent["url"],
        "parent_sha256": parent["sha256"],
        "frame": "ICRS",
        "position_uncertainty_arcsec": None,
        "input_exposures": [
            {
                "obs_id": e["obs_id"],
                "mjd_avg_utc": e["mjd_avg"],
                "cutout_file": e["file"],
                "cutout_sha256": e["sha256"],
                "parent_url": manifest["parents"][i]["url"],
                "parent_sha256": manifest["parents"][i]["sha256"],
            }
            for i, e in enumerate(exposures)
        ],
    }
    artifact = json.loads((HERE / "artifact-results.json").read_text())
    arm = next(
        r
        for r in artifact["results"]
        if r["split"] == "heldout" and r["arm"] == "pixel_blank" and r["method"] == "DAO"
    )
    site = arm["rows"][0]["site"]
    x, y = artifact["site_positions"][32 + site]
    ara, adec = frames[0].wcs.pixel_to_world_values(x, y)
    package = Package(
        catalog={
            "name": "Gaia DR3",
            "access_mode": catalog_mode,
            "association_path": "direct Gaia investigation; SearchX Identify not executed",
            "service": "https://gea.esac.esa.int/tap-server/tap/sync",
            "cone_center_deg": [232.64278847144752, -10.546006460187737],
            "radius_arcsec": 20,
            "adql": "SELECT TOP 100 source_id,ra,dec,ra_error,dec_error,ra_dec_corr,pmra,pmdec,pmra_error,pmdec_error,ref_epoch,phot_g_mean_mag,ruwe FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',232.64278847144752,-10.546006460187737,0.00555555556))",  # noqa: E501
            "epoch_filter": "catalog positions at epoch 2016; high proper motion sources outside cone may be missed",  # noqa: E501
            "raw_file": "gaia-dr3-cone.csv",
            "raw_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        },
        cases=[
            Case(
                id="A",
                kind="real_observation",
                status="TENTATIVE_ASSOCIATION",
                ra_deg=measurements[0]["ra_deg"],
                dec_deg=measurements[0]["dec_deg"],
                observation=obs,
                measurements=measurements,
                catalog_candidates=candidates,
                checks=[check],
                reasons=[
                    "One Gaia DR3 candidate in the 20 arcsec epoch-2016 cone.",
                    "No SPHEREx astrometric covariance or calibrated chance-association probability; identification is tentative.",  # noqa: E501
                ],
            ),
            Case(
                id="B",
                kind="controlled_injection",
                status="ARTIFACT_UNRESOLVED",
                ra_deg=float(ara),
                dec_deg=float(adec),
                observation=obs,
                measurements=[
                    {
                        "site_id": site,
                        "arm": "one-pixel injection in Gaussian noise using archive variance",
                        "concentration": arm["rows"][0]["concentration"],
                        "unit": "dimensionless",
                    }
                ],
                catalog_candidates=[],
                checks=[
                    {
                        "verdict": "INCONCLUSIVE",
                        "method": "development-frozen 3x3 concentration diagnostic",
                        "threshold": arm["development_threshold"],
                        "reason": "Injected label is known but the threshold is not calibrated on independent real artifacts.",  # noqa: E501
                    }
                ],
                reasons=["Synthetic injection, not a detected celestial object."],
            ),
            Case(
                id="C",
                kind="real_observation",
                status="INCONCLUSIVE",
                ra_deg=measurements[0]["ra_deg"],
                dec_deg=measurements[0]["dec_deg"],
                observation=obs,
                measurements=measurements,
                catalog_candidates=candidates,
                checks=[check],
                reasons=[
                    "Same real source as A, illustrating the separate question of temporal persistence.",  # noqa: E501
                    "The 129-second pair has unqualified astrometric uncertainty and differing wavelengths.",  # noqa: E501
                ],
            ),
        ],
    )
    (output_dir / "evidence.json").write_text(package.model_dump_json(indent=2) + "\n")
    (output_dir / "evidence.schema.json").write_text(
        json.dumps(Package.model_json_schema(), indent=2) + "\n"
    )
    print(
        "Gaia candidates",
        len(candidates),
        "separations",
        [c["separation_arcsec"] for c in candidates],
    )
    print("independent centroid separation arcsec", check["separation_arcsec"])
    print("wrote", output_dir / "evidence.json")


if __name__ == "__main__":
    main()
