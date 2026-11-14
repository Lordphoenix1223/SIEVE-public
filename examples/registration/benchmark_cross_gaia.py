"""Cross-instrument candidate checks against Gaia; no certified identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from sieve.registration import associate, load_observation

EXPECTED = "71d7063bb0f61cef9cdbc402a78433738d6dd969dedc03124fe43cddf8a43c6d"
FILES = [
    ("2mass.fits", "763ef344df3ac8fa80ff46f00ca1ec59946ca3f99502562d6fcfb73320b1cec3"),
    ("msx.fits", "3687fb3763911825f981e74b6a9b82c0e618f7e592b1e0cb17e2c63164e28cd6"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("gaia-cross-reference.csv")
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Gaia hash mismatch")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) >= 2000:
        raise ValueError("empty or truncated Gaia result")
    gaia = SkyCoord([float(r["ra"]) for r in rows] * u.deg, [float(r["dec"]) for r in rows] * u.deg)
    observations = []
    xy = []
    maps = []
    summaries = []
    for name, expected in FILES:
        path = args.data_dir / name
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("image hash mismatch")
        o = load_observation(path, "PRIMARY")
        observations.append(o)
        _, med, std = sigma_clipped_stats(o.image[np.isfinite(o.image)], sigma=3)
        table = DAOStarFinder(threshold=5 * std, fwhm=3)(np.nan_to_num(o.image - med))
        points = np.column_stack((table["x_centroid"], table["y_centroid"]))
        xy.append(points)
        sky = o.wcs.pixel_to_world(points[:, 0], points[:, 1]).icrs
        i, j, sep, _ = search_around_sky(sky, gaia, 12 * u.arcsec)
        ni = np.bincount(i, minlength=len(sky))
        nj = np.bincount(j, minlength=len(gaia))
        selected = [
            (int(a), int(b), float(c))
            for a, b, c in zip(i, j, sep.arcsec, strict=True)
            if ni[a] == 1 and nj[b] == 1
        ]
        maps.append({a: (b, c) for a, b, c in selected})
        summaries.append(
            {
                "image": name,
                "detections": len(points),
                "gaia_candidate_pairs": len(i),
                "unique_candidates": len(selected),
                "ambiguous_pairs": len(i) - len(selected),
                "median_separation_arcsec": float(np.median([v[2] for v in selected]))
                if selected
                else None,
            }
        )
    pairs = associate(observations[0], observations[1], xy[0], xy[1], 12)["unambiguous_pairs"]
    shared = []
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        if a in maps[0] and b in maps[1] and maps[0][a][0] == maps[1][b][0]:
            g = maps[0][a][0]
            shared.append(
                {
                    "gaia_source_id": rows[g]["source_id"],
                    "cross_separation_arcsec": pair["separation_arcsec"],
                    "2mass_gaia_arcsec": maps[0][a][1],
                    "msx_gaia_arcsec": maps[1][b][1],
                }
            )
    out = {
        "schema_version": "sieve.registration.cross-gaia.v1",
        "gaia_sha256": EXPECTED,
        "gaia_service": "https://gea.esac.esa.int/tap-server/tap/sync",
        "gaia_query": (
            "SELECT TOP 2000 source_id,ra,dec,ra_error,dec_error,pmra,pmdec,ref_epoch,"
            "phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
            "CIRCLE('ICRS',266.4,-28.9333,0.2)) AND pmra IS NOT NULL AND pmdec IS NOT NULL "
            "AND phot_g_mean_mag < 15"
        ),
        "image_sha256": {name: sha for name, sha in FILES},
        "gaia_rows": len(rows),
        "gaia_catalog_epoch_jyear": 2016.0,
        "images": summaries,
        "cross_instrument_unique_pairs": len(pairs),
        "shared_unique_gaia_candidates": shared,
        "status": "REGISTRATION_UNQUALIFIED",
        "limitations": [
            "Unknown image observation epochs prevent proper-motion propagation.",
            "Gaia is independent; proximity does not certify source identity.",
            "Pixel scales, PSFs and bands differ; covariance and observer geometry are unknown.",
            "A single shared candidate cannot qualify a registration model or false-match rate.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
