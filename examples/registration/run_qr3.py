"""Run conservative WCS diagnostics on the two pinned QR3 exposures."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from sieve.photometry.qualified import detect_positions, load_qr3_frame
from sieve.photometry.scanner import _BAD_FLAG_BITS
from sieve.provenance.runtime import source_identity
from sieve.registration.core import (
    associate,
    compare,
    load_observation,
    reference_residuals,
    reproject_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / "qualification"
    metadata = json.loads((base / "source.json").read_text())["exposures"][:2]
    started = time.perf_counter()
    frames = [load_qr3_frame(base / e["file"], e["sha256"]) for e in metadata]
    observations = [load_observation(base / e["file"]) for e in metadata]
    if any(o.sha256 != e["sha256"] for o, e in zip(observations, metadata, strict=True)):
        raise ValueError("input checksum mismatch")
    points = [detect_positions(f, f.image, "DAO") for f in frames]
    pairs = associate(*observations, *points, radius_arcsec=12)
    diag = compare(*observations, pairs)
    accepted = sorted(pairs["unambiguous_pairs"], key=lambda p: (p["a"], p["b"]))
    refs = reference_residuals(
        *observations,
        np.asarray([points[0][p["a"]] for p in accepted]),
        np.asarray([points[1][p["b"]] for p in accepted]),
    )
    distance = cdist(points[0], points[1])
    raw = np.argmin(distance, axis=1)
    # Comparison only: two-pixel nearest-neighbor correspondence is not truth.
    diagnostic_raw = {(i, int(j)) for i, j in enumerate(raw) if distance[i, j] <= 2}
    sky_pairs = {(p["a"], p["b"]) for p in accepted}
    visual = reproject_visualization(observations[1], observations[0], _BAD_FLAG_BITS)
    commit, tree, dirty, environment = source_identity()
    result = {
        "provenance": {
            "code_commit": commit,
            "source_tree_sha256": tree,
            "dirty_source": dirty,
            "environment_id": environment,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "schema_version": "sieve.registration.qr3-example.v1",
        "diagnostics": diag,
        "candidate_association": pairs,
        "reprojection": {
            "status": visual["status"],
            "valid_fraction": float(np.mean(visual["coverage"])),
            "variance": None,
            "limitations": visual["limitations"],
        },
        "reference_candidate_diagnostics": refs,
        "peak_counts": [len(p) for p in points],
        "raw_pixel_nearest_2px": len(diagnostic_raw),
        "raw_pixel_and_world_agree": len(diagnostic_raw & sky_pairs),
        "raw_pixel_only": len(diagnostic_raw - sky_pairs),
        "world_only": len(sky_pairs - diagnostic_raw),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_note": "ru_maxrss units are platform-dependent (bytes on macOS, KiB on Linux)",
        "limitations": [
            "Comparisons are unlabeled; disagreement is not a measured false-match rate.",
            "Candidate reference pairs are not independently verified stationary stars.",
            "Native measurements retained; no corrected astrometry or anomaly verdict is produced.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "peak_counts",
                    "raw_pixel_nearest_2px",
                    "raw_pixel_and_world_agree",
                    "raw_pixel_only",
                    "world_only",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    print(
        "spatial",
        diag["spatial_status"],
        "spectral",
        diag["spectral_status"],
        "heldout",
        refs["heldout_median_after_arcsec"],
    )


if __name__ == "__main__":
    main()
