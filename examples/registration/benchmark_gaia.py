"""Bounded Gaia DR3 astrometric diagnostic; associations are not truth labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time

EXPECTED = "c6ca7cb28496f14e0146835018a7605130177b2dfc4be9a8fcb0572a72957517"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).with_name("gaia-qr3-reference.csv")
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Gaia reference bytes differ from retained query")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) >= 3000:
        raise ValueError("empty or potentially truncated Gaia query")
    measurements = json.loads(args.measurements.read_text())["exposures"][0]
    detected = [r for r in measurements["measurements"] if r["status"] == "MEASURED_CONDITIONAL"]
    reference = SkyCoord(
        ra=[float(r["ra"]) for r in rows] * u.deg,
        dec=[float(r["dec"]) for r in rows] * u.deg,
        pm_ra_cosdec=[float(r["pmra"]) for r in rows] * u.mas / u.yr,
        pm_dec=[float(r["pmdec"]) for r in rows] * u.mas / u.yr,
        obstime=Time([float(r["ref_epoch"]) for r in rows], format="jyear"),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        propagated = reference.apply_space_motion(
            new_obstime=Time(measurements["mjd_avg_utc"], format="mjd", scale="utc")
        )
    propagated = SkyCoord(ra=propagated.ra, dec=propagated.dec, frame="icrs")
    sky = SkyCoord(
        [r["ra_deg"] for r in detected] * u.deg, [r["dec_deg"] for r in detected] * u.deg
    )
    i, j, sep, _ = search_around_sky(sky, propagated, 3 * u.arcsec)
    ni = np.bincount(i, minlength=len(sky))
    nj = np.bincount(j, minlength=len(reference))
    pairs = [
        (int(a), int(b), float(c))
        for a, b, c in zip(i, j, sep.arcsec, strict=True)
        if ni[a] == 1 and nj[b] == 1
    ]
    pairs.sort(key=lambda p: int(rows[p[1]]["source_id"]))
    if len(pairs) < 6:
        raise ValueError("fewer than six unique candidate references")
    di = np.array([p[0] for p in pairs])
    gi = np.array([p[1] for p in pairs])
    east, north = sky[di].spherical_offsets_to(propagated[gi])
    offsets = np.column_stack([east.arcsec, north.arcsec])
    train = np.array(
        [hashlib.sha256(rows[p[1]]["source_id"].encode()).digest()[0] % 2 == 0 for p in pairs]
    )
    test = ~train
    if train.sum() < 3 or test.sum() < 3:
        raise ValueError("insufficient train or held-out references")
    shift = np.median(offsets[train], axis=0)
    before = np.linalg.norm(offsets[test], axis=1)
    after = np.linalg.norm(offsets[test] - shift, axis=1)
    out = {
        "schema_version": "sieve.registration.gaia-diagnostic.v1",
        "catalog": "Gaia DR3",
        "query": "SELECT TOP 3000 source_id,ra,dec,ra_error,dec_error,pmra,pmdec,parallax,ref_epoch,phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',232.663579354,-10.7213238583,0.25)) AND pmra IS NOT NULL AND pmdec IS NOT NULL AND phot_g_mean_mag < 18",  # noqa: E501
        "catalog_url": "https://gea.esac.esa.int/tap-server/tap/sync",
        "catalog_sha256": EXPECTED,
        "input_measurements_sha256": hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
        "propagation_warnings": [str(w.message) for w in captured],
        "catalog_rows": len(rows),
        "detected_measurements": len(detected),
        "candidate_pairs": len(i),
        "unique_candidate_pairs": len(pairs),
        "ambiguous_candidate_pairs": len(i) - len(pairs),
        "train_count": int(train.sum()),
        "heldout_count": int(test.sum()),
        "shift_east_north_arcsec": shift.tolist(),
        "heldout_median_before_arcsec": float(np.median(before)),
        "heldout_median_after_arcsec": float(np.median(after)),
        "heldout_p95_after_arcsec": float(np.quantile(after, 0.95)),
        "status": "REGISTRATION_UNQUALIFIED",
        "limitations": [
            "Unique 3-arcsecond proximity is not independently confirmed identity.",
            "Gaia proper motion is propagated without distance/radial velocity; spacecraft parallax is not corrected.",  # noqa: E501
            "SIEVE astrometric covariance and false-match ground truth are unavailable.",
            "Split is deterministic SHA-256 partition of Gaia source IDs, not source-population or calibration independence.",  # noqa: E501
        ],
    }
    with args.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("query", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
