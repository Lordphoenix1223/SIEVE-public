"""Compare existing and new sky pairers on a small labeled geometry fixture."""

from __future__ import annotations

import argparse
import json
import runpy
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import search_around_sky

from sieve.registration import associate, load_observation

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The fixture generator is shared with the geometry regression tests.
    make = runpy.run_path(str(ROOT / "tests/registration/test_registration.py"))["make_image"]
    cases = []
    with tempfile.TemporaryDirectory(prefix="sieve-geometry-") as directory:
        base = Path(directory)
        configurations = [
            (30, 1, False, False),
            (90, 1, False, False),
            (180, 1, False, False),
            (30, 1.3, False, False),
            (30, 1, True, False),
            (30, 1, False, True),
        ]
        for index, (angle, scale, flip, sip) in enumerate(configurations):
            pa, pb = base / f"a{index}.fits", base / f"b{index}.fits"
            make(pa)
            make(pb, angle, scale, reflect=flip, sip=sip)
            a, b = load_observation(pa), load_observation(pb)
            xy = np.array([[22.0, 22.0], [29.0, 29.0], [36.0, 35.0], [20.0, 35.0]])
            sky = a.wcs.pixel_to_world(xy[:, 0], xy[:, 1])
            x, y = b.wcs.world_to_pixel(sky)
            other = np.column_stack((x, y))
            sky_b = b.wcs.pixel_to_world(other[:, 0], other[:, 1])
            i, j, sep, _ = search_around_sky(sky, sky_b, 0.01 * u.arcsec)
            old = {(int(x), int(y)) for x, y in zip(i, j, strict=True)}
            proposed = associate(a, b, xy, other, 0.01)
            new = {(r["a"], r["b"]) for r in proposed["unambiguous_pairs"]}
            # Astropy nearest sky match is a separate interface, not independent WCS truth.
            nearest, separations, _ = sky.match_to_catalog_sky(sky_b)
            reference = {
                (k, int(v)) for k, v in enumerate(nearest) if separations[k].arcsec <= 0.01
            }
            truth = {(k, k) for k in range(len(xy))}
            cases.append(
                {
                    "rotation_deg": angle,
                    "scale_ratio": scale,
                    "reflection": flip,
                    "sip": sip,
                    "truth_count": len(truth),
                    "original_correct": len(old & truth),
                    "original_incorrect": len(old - truth),
                    "upgraded_correct": len(new & truth),
                    "upgraded_incorrect": len(new - truth),
                    "reference_correct": len(reference & truth),
                    "reference_incorrect": len(reference - truth),
                    "maximum_angular_error_arcsec": float(max(sep.arcsec)),
                    "ambiguous_pairs": len(proposed["ambiguous_pairs"]),
                }
            )
    out = {
        "schema_version": "sieve.registration.controlled-benchmark.v1",
        "cases": cases,
        "limitations": [
            "Noise-free repeated sources; not independent skies or measured completeness.",
            "Truth uses the same WCS as the methods; not external calibration.",
            "The original and upgraded paths both use Astropy search_around_sky.",
        ],
    }
    with args.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
