"""Compare original sky matching and upgraded association on identical real peaks."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import search_around_sky

from sieve.photometry.qualified import detect_positions, load_qr3_frame
from sieve.registration import associate, load_observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "qualification"
    manifest = json.loads((root / "source.json").read_text())["exposures"][:2]
    obs = [load_observation(root / e["file"]) for e in manifest]
    frames = [load_qr3_frame(root / e["file"], e["sha256"]) for e in manifest]
    points = [detect_positions(f, f.image, "DAO") for f in frames]
    old_times = []
    new_times = []
    for _ in range(20):
        tic = time.perf_counter()
        sky = [
            o.wcs.pixel_to_world(x[:, 0], x[:, 1]).icrs for o, x in zip(obs, points, strict=True)
        ]
        i, j, _, _ = search_around_sky(sky[0], sky[1], 12 * u.arcsec)
        old_times.append(time.perf_counter() - tic)
        tic = time.perf_counter()
        new = associate(obs[0], obs[1], points[0], points[1], 12)
        new_times.append(time.perf_counter() - tic)
    out = {
        "schema_version": "sieve.registration.timing.v1",
        "repeats": 20,
        "original_median_seconds": float(np.median(old_times)),
        "upgraded_median_seconds": float(np.median(new_times)),
        "original_pairs": len(i),
        "upgraded_pairs": len(new["candidate_pairs"]),
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_note": "bytes on macOS, KiB on Linux; includes detection/imports",
        "limitations": [
            "Single process and field; not a throughput or scaling benchmark.",
            "Both methods use Astropy sky search; reference candidates lack truth labels.",
        ],
    }
    with args.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
