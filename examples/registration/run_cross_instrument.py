"""Compare public Astropy 2MASS K and MSX E examples; no astrophysical truth labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from sieve.provenance.runtime import source_identity
from sieve.registration import associate, compare, load_observation

INPUTS = [
    (
        "2mass.fits",
        "763ef344df3ac8fa80ff46f00ca1ec59946ca3f99502562d6fcfb73320b1cec3",
        "https://www.astropy.org/astropy-data/galactic_center/gc_2mass_k.fits",
    ),
    (
        "msx.fits",
        "3687fb3763911825f981e74b6a9b82c0e618f7e592b1e0cb17e2c63164e28cd6",
        "https://www.astropy.org/astropy-data/galactic_center/gc_msx_e.fits",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    observations = []
    positions = []
    for name, expected, _ in INPUTS:
        path = args.data_dir / name
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"input hash mismatch: {name}")
        o = load_observation(path, "PRIMARY")
        observations.append(o)
        _, median, std = sigma_clipped_stats(o.image[np.isfinite(o.image)], sigma=3)
        if not np.isfinite(std) or std <= 0:
            raise ValueError("invalid background noise")
        table = DAOStarFinder(threshold=5 * std, fwhm=3)(np.nan_to_num(o.image - median))
        positions.append(
            np.empty((0, 2))
            if table is None
            else np.column_stack((table["x_centroid"], table["y_centroid"]))
        )
    matches = associate(*observations, *positions, radius_arcsec=12)
    commit, tree, dirty, environment = source_identity()
    result = {
        "provenance": {
            "code_commit": commit,
            "source_tree_sha256": tree,
            "dirty_source": dirty,
            "environment_id": environment,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "schema_version": "sieve.registration.cross-example.v1",
        "input_urls": [x[2] for x in INPUTS],
        "peak_counts": [len(x) for x in positions],
        "candidate_association": matches,
        "diagnostics": compare(*observations, matches),
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Different instruments and passbands; source identity is not independently labeled.",
            "The 12-arcsecond gate and point-source finder are illustrative, not calibrated.",
            "No variance, quality planes, PSF matching or qualified registration solution.",
        ],
    }
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        "peaks",
        result["peak_counts"],
        "pairs",
        len(matches["candidate_pairs"]),
        "unique",
        len(matches["unambiguous_pairs"]),
        "ambiguous",
        len(matches["ambiguous_pairs"]),
    )
    print(
        "spatial",
        result["diagnostics"]["spatial_status"],
        "spectral",
        result["diagnostics"]["spectral_status"],
    )


if __name__ == "__main__":
    main()
