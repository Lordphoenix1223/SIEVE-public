"""Run the pinned SearchX matcher on the public SIEVE QR3 investigation."""

import json
import subprocess
import warnings
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from erfa import ErfaWarning

RUNNER = Path(__file__).resolve().parent / "identify" / "match.mjs"
REVISION = "7c24aeead8587215f9f9c73c0db51e2e17043cc8"


def build(case, catalog):
    """Keep the original measurement and evidence alongside each match."""
    observations = {o["obs_id"]: o for o in case.observation["input_exposures"]}
    results = []
    for measurement in case.measurements:
        observation = observations[measurement["obs_id"]]
        time = Time(observation["mjd_avg_utc"], format="mjd", scale="utc")
        flags = []
        if measurement.get("flag_bits"):
            flags.append(f"FITS_FLAG_BITS:{measurement['flag_bits']}")
        raw = {
            "id": measurement["measurement_id"],
            "ra_deg": measurement["ra_deg"],
            "dec_deg": measurement["dec_deg"],
            "epoch_year": float(time.tt.jyear),
            "flags": flags,
        }
        rows = []
        for record in case.catalog_candidates:
            source = SkyCoord(
                ra=record["catalog_ra_deg"] * u.deg,
                dec=record["catalog_dec_deg"] * u.deg,
                pm_ra_cosdec=record["pmra_mas_per_year"] * u.mas / u.yr,
                pm_dec=record["pmdec_mas_per_year"] * u.mas / u.yr,
                obstime=Time(record["ref_epoch_jyear"], format="jyear", scale="tt"),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*distance overridden.*", category=ErfaWarning
                )
                pos = source.apply_space_motion(new_obstime=time.tt)
            rows.append(
                {
                    "id": record["source_id"],
                    "catalog": "Gaia DR3",
                    "source_url": catalog["service"],
                    "ra_deg": float(pos.ra.deg),
                    "dec_deg": float(pos.dec.deg),
                    "epoch_year": raw["epoch_year"],
                }
            )
        proc = subprocess.run(
            ["node", str(RUNNER)],
            input=json.dumps({"detection": raw, "catalog": rows}, allow_nan=False),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        association = json.loads(proc.stdout)
        results.append(
            {
                "schema_version": "sieve.searchx.identify.v1",
                "status": association["status"],
                "detection": {
                    **raw,
                    "observation_id": measurement["obs_id"],
                    "wavelength_um": measurement.get("cwave_um"),
                    "bandwidth_um": measurement.get("cband_um"),
                    "sigma_arcsec": None,
                    "measurement_status": measurement["status"],
                    "flux": {
                        "value": measurement.get("flux_uJy"),
                        "sigma": measurement.get("flux_error_uJy"),
                        "unit": "uJy",
                        "method": "fixed-position point-source profile",
                    },
                    "producer": "sieve.photometry",
                    "measurement": measurement,
                },
                "observation": observation,
                "association": association,
                "candidate": None,
                "checks": case.checks,
                "provenance": {
                    "matcher_revision": REVISION,
                    "catalog": catalog,
                    "original_catalog_candidates": case.catalog_candidates,
                    "epoch_convention": "TT Julian year",
                    "limitations": [
                        "No qualified positional or propagated uncertainty.",
                        "Gaia cone selected at catalog epoch; it can miss fast movers.",
                        "No distance or radial velocity used in propagation.",
                    ],
                },
            }
        )
    return results
